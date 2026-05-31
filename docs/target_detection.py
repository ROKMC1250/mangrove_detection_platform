#!/usr/bin/env python3
"""Standalone classical target detection for multiband GeoTIFFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import numpy as np
import rasterio


Point = Tuple[int, int]


def parse_point(value: str) -> Point:
    """Parse a CLI point in col,row order."""
    try:
        col_text, row_text = value.split(",", maxsplit=1)
        return int(col_text.strip()), int(row_text.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid point '{value}'. Use col,row, for example 120,88."
        ) from exc


def parse_band_list(value: str) -> list[int]:
    """Parse 1-based GeoTIFF band indexes."""
    try:
        bands = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid band list '{value}'. Use values such as 1,2,3."
        ) from exc
    if not bands or any(band < 1 for band in bands):
        raise argparse.ArgumentTypeError("Band indexes must be positive and 1-based.")
    return bands


def validate_points(points: Sequence[Point], width: int, height: int, label: str) -> None:
    for col, row in points:
        if not (0 <= col < width and 0 <= row < height):
            raise ValueError(
                f"{label} point ({col},{row}) is outside raster size {width}x{height}."
            )


def read_cube(path: Path, bands: Sequence[int] | None) -> tuple[np.ndarray, dict, list[int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        selected_bands = list(bands) if bands else list(range(1, src.count + 1))
        if src.count < 1:
            raise ValueError(f"Input GeoTIFF has no bands: {path}")
        invalid = [band for band in selected_bands if band > src.count]
        if invalid:
            raise ValueError(
                f"Band indexes {invalid} exceed input band count {src.count}."
            )
        band_first = src.read(selected_bands).astype(np.float32)
        profile = src.profile.copy()

    cube = np.moveaxis(band_first, 0, -1)
    return cube, profile, selected_bands


def finite_pixel_mask(cube: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(cube), axis=-1)


def make_exclusion_mask(
    height: int,
    width: int,
    points: Iterable[Point],
    radius: int,
) -> np.ndarray:
    excluded = np.zeros((height, width), dtype=bool)
    for col, row in points:
        row_min = max(0, row - radius)
        row_max = min(height, row + radius + 1)
        col_min = max(0, col - radius)
        col_max = min(width, col + radius + 1)
        excluded[row_min:row_max, col_min:col_max] = True
    return excluded


def sample_background(
    cube: np.ndarray,
    valid_mask: np.ndarray,
    excluded_points: Sequence[Point],
    sample_count: int,
    exclude_radius: int,
    seed: int,
) -> np.ndarray:
    height, width, _ = cube.shape
    candidate_mask = valid_mask & ~make_exclusion_mask(
        height,
        width,
        excluded_points,
        exclude_radius,
    )
    rows_cols = np.argwhere(candidate_mask)
    if rows_cols.size == 0:
        raise ValueError("No finite background pixels remain after point exclusion.")

    rng = np.random.default_rng(seed)
    take = min(int(sample_count), len(rows_cols))
    chosen = rng.choice(len(rows_cols), size=take, replace=False)
    sampled = rows_cols[chosen]
    return cube[sampled[:, 0], sampled[:, 1], :]


def target_spectrum(cube: np.ndarray, positive_points: Sequence[Point]) -> np.ndarray:
    spectra = np.asarray([cube[row, col, :] for col, row in positive_points])
    if not np.all(np.isfinite(spectra)):
        raise ValueError("At least one positive point lands on a non-finite pixel.")
    return spectra.mean(axis=0).astype(np.float64)


def fit_background(background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if background.ndim != 2 or background.shape[0] < 2:
        raise ValueError("At least two background spectra are required.")

    mean = background.mean(axis=0).astype(np.float64)
    covariance = np.cov(background.astype(np.float64), rowvar=False)
    covariance = np.atleast_2d(covariance)
    trace = float(np.trace(covariance))
    regularizer = 1e-6 * trace / max(covariance.shape[0], 1)
    if not np.isfinite(regularizer) or regularizer <= 0:
        regularizer = 1e-6
    covariance = covariance + regularizer * np.eye(covariance.shape[0])
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
    return mean, inverse


def detect_sam(cube: np.ndarray, target: np.ndarray) -> np.ndarray:
    height, width, channels = cube.shape
    pixels = cube.reshape(-1, channels).astype(np.float64)
    target_norm = max(float(np.linalg.norm(target)), 1e-10)
    pixel_norms = np.maximum(np.linalg.norm(pixels, axis=1), 1e-10)
    cos_angles = np.clip((pixels @ target) / (pixel_norms * target_norm), -1.0, 1.0)
    angles = np.arccos(cos_angles)
    scores = np.nanmax(angles) - angles
    return scores.reshape(height, width)


def detect_ace(
    cube: np.ndarray,
    target: np.ndarray,
    background_mean: np.ndarray,
    background_inverse: np.ndarray,
) -> np.ndarray:
    height, width, channels = cube.shape
    pixels = cube.reshape(-1, channels).astype(np.float64)
    centered_pixels = pixels - background_mean
    centered_target = target - background_mean
    whitened_target = background_inverse @ centered_target
    numerator = (centered_pixels @ whitened_target) ** 2
    target_term = centered_target @ whitened_target
    pixel_term = np.sum((centered_pixels @ background_inverse) * centered_pixels, axis=1)
    denominator = np.maximum(target_term * pixel_term, 1e-10)
    return np.clip(numerator / denominator, 0.0, 1.0).reshape(height, width)


def detect_cem(
    cube: np.ndarray,
    target: np.ndarray,
    background_inverse: np.ndarray,
) -> np.ndarray:
    height, width, channels = cube.shape
    pixels = cube.reshape(-1, channels).astype(np.float64)
    rd = background_inverse @ target
    denominator = max(float(target @ rd), 1e-10)
    weights = rd / denominator
    return (pixels @ weights).reshape(height, width)


def detect_mf(
    cube: np.ndarray,
    target: np.ndarray,
    background_mean: np.ndarray,
    background_inverse: np.ndarray,
) -> np.ndarray:
    height, width, channels = cube.shape
    pixels = cube.reshape(-1, channels).astype(np.float64)
    centered_pixels = pixels - background_mean
    centered_target = target - background_mean
    covariance_target = background_inverse @ centered_target
    denominator = max(float(centered_target @ covariance_target), 1e-10)
    weights = covariance_target / denominator
    return (centered_pixels @ weights).reshape(height, width)


def otsu_threshold(scores: np.ndarray) -> float:
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        raise ValueError("Detection produced no finite scores.")

    minimum = float(valid.min())
    maximum = float(valid.max())
    if maximum - minimum < 1e-10:
        return (minimum + maximum) / 2.0

    normalized = ((valid - minimum) / (maximum - minimum) * 255).astype(np.uint8)
    histogram, _ = np.histogram(normalized, bins=256, range=(0, 256))
    histogram = histogram.astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return (minimum + maximum) / 2.0

    sum_all = np.dot(np.arange(256), histogram)
    sum_background = 0.0
    weight_background = 0.0
    best_index = 0
    best_variance = -1.0
    for index in range(256):
        weight_background += histogram[index]
        if weight_background <= 0:
            continue
        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break
        sum_background += index * histogram[index]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_all - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (
            mean_background - mean_foreground
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            best_index = index

    return minimum + (best_index / 255.0) * (maximum - minimum)


def run_detection(
    cube: np.ndarray,
    positive_points: Sequence[Point],
    negative_points: Sequence[Point],
    algorithm: str,
    background_samples: int,
    exclude_radius: int,
    seed: int,
    threshold: float | None,
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    height, width, _ = cube.shape
    validate_points(positive_points, width, height, "Positive")
    validate_points(negative_points, width, height, "Negative")

    valid_mask = finite_pixel_mask(cube)
    target = target_spectrum(cube, positive_points)
    background = sample_background(
        cube,
        valid_mask,
        [*positive_points, *negative_points],
        background_samples,
        exclude_radius,
        seed,
    )
    background_mean, background_inverse = fit_background(background)

    selected_algorithm = algorithm.upper()
    if selected_algorithm == "SAM":
        scores = detect_sam(cube, target)
    elif selected_algorithm == "ACE":
        scores = detect_ace(cube, target, background_mean, background_inverse)
    elif selected_algorithm == "CEM":
        scores = detect_cem(cube, target, background_inverse)
    elif selected_algorithm == "MF":
        scores = detect_mf(cube, target, background_mean, background_inverse)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    scores = scores.astype(np.float32)
    scores[~valid_mask] = np.nan
    used_threshold = float(threshold) if threshold is not None else otsu_threshold(scores)
    mask = np.isfinite(scores) & (scores >= used_threshold)
    valid_scores = scores[np.isfinite(scores)]
    stats = {
        "background_sample_count": int(background.shape[0]),
        "detected_pixels": int(mask.sum()),
        "finite_score_pixels": int(valid_scores.size),
        "score_min": float(valid_scores.min()),
        "score_max": float(valid_scores.max()),
        "score_mean": float(valid_scores.mean()),
    }
    return mask, scores, used_threshold, stats


def write_mask(path: Path, mask: np.ndarray, source_profile: dict) -> None:
    profile = source_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone classical target detection on a multiband GeoTIFF."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input multiband GeoTIFF.")
    parser.add_argument(
        "--positive-point",
        action="append",
        type=parse_point,
        required=True,
        help="Positive target point in col,row order. Repeat for multiple points.",
    )
    parser.add_argument(
        "--negative-point",
        action="append",
        default=[],
        type=parse_point,
        help="Optional non-target point in col,row order. Repeat as needed.",
    )
    parser.add_argument(
        "--algorithm",
        choices=("SAM", "ACE", "CEM", "MF"),
        default="SAM",
        help="Classical detector to run.",
    )
    parser.add_argument(
        "--bands",
        type=parse_band_list,
        help="Optional 1-based GeoTIFF bands, for example 1,2,3,4.",
    )
    parser.add_argument("--threshold", type=float, help="Optional manual score threshold.")
    parser.add_argument(
        "--background-samples",
        type=int,
        default=10000,
        help="Maximum random background spectra used for covariance fitting.",
    )
    parser.add_argument(
        "--exclude-radius",
        type=int,
        default=5,
        help="Pixel radius excluded around positive and negative points.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Background sampling seed.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.background_samples < 2:
        parser.error("--background-samples must be at least 2.")
    if args.exclude_radius < 0:
        parser.error("--exclude-radius must be non-negative.")

    cube, profile, bands = read_cube(args.input, args.bands)
    mask, _scores, threshold, stats = run_detection(
        cube=cube,
        positive_points=args.positive_point,
        negative_points=args.negative_point,
        algorithm=args.algorithm,
        background_samples=args.background_samples,
        exclude_radius=args.exclude_radius,
        seed=args.seed,
        threshold=args.threshold,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "target_detection_mask.tif"
    metadata_path = args.output_dir / "target_detection_metadata.json"
    write_mask(mask_path, mask, profile)
    write_json(
        metadata_path,
        {
            "input_path": str(args.input.resolve()),
            "mask_path": str(mask_path.resolve()),
            "algorithm": args.algorithm,
            "selected_bands_1_based": bands,
            "positive_points_col_row": [list(point) for point in args.positive_point],
            "negative_points_col_row": [list(point) for point in args.negative_point],
            "threshold": threshold,
            "threshold_source": "manual" if args.threshold is not None else "otsu",
            "background_samples_requested": args.background_samples,
            "exclude_radius_pixels": args.exclude_radius,
            **stats,
        },
    )
    print(f"Mask: {mask_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Detected pixels: {stats['detected_pixels']} at threshold {threshold:.6g}")


if __name__ == "__main__":
    main()
