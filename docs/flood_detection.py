#!/usr/bin/env python3
"""Standalone Sentinel-1 VV flood segmentation for GeoTIFF rasters."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import rasterio


DB_MIN = -30.0
DB_MAX = 10.0
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_FLOOD_ASSET_DIR = SCRIPT_DIR / "checkpoints" / "flood_unetpp_vh_v1"
DEFAULT_FLOOD_CHECKPOINT = DEFAULT_FLOOD_ASSET_DIR / "best.pt"
DEFAULT_FLOOD_CONFIG = DEFAULT_FLOOD_ASSET_DIR / "config.yaml"


def choose_device(torch_module):
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def load_model(checkpoint: Path, device):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Flood checkpoint not found: {checkpoint}")
    try:
        import torch
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "Flood dependencies are unavailable. Install torch and "
            "segmentation-models-pytorch."
        ) from exc

    model = smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )
    try:
        raw_state = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        raw_state = torch.load(checkpoint, map_location=device)
    state_dict = raw_state["model"] if isinstance(raw_state, dict) and "model" in raw_state else raw_state
    if not isinstance(state_dict, dict):
        raise ValueError("Flood checkpoint does not contain a model state dictionary.")
    if any(key.startswith("net.") for key in state_dict):
        state_dict = {
            key[len("net."):]: value
            for key, value in state_dict.items()
            if key.startswith("net.")
        }
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {
            key[len("module."):]: value
            for key, value in state_dict.items()
        }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return torch, model


def read_vv(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input GeoTIFF not found: {path}")
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"Input GeoTIFF has no bands: {path}")
        vv = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata = src.nodata

    data_mask = np.isfinite(vv)
    if nodata is not None:
        data_mask &= vv != nodata
    vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
    vv = np.clip(vv, DB_MIN, DB_MAX)
    vv = (vv - DB_MIN) / (DB_MAX - DB_MIN)
    vv[~data_mask] = 0.0
    return vv.astype(np.float32), data_mask, profile


def pad_amount(size: int, patch_size: int, stride: int) -> int:
    if size <= patch_size:
        return patch_size - size
    remainder = (size - patch_size) % stride
    return 0 if remainder == 0 else stride - remainder


def pad_image(image: np.ndarray, pad_h: int, pad_w: int) -> np.ndarray:
    mode = "reflect" if min(image.shape) > 1 else "edge"
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode=mode)


def gaussian_weight(patch_size: int) -> np.ndarray:
    sigma = patch_size / 4.0
    axis = np.linspace(-(patch_size - 1) / 2.0, (patch_size - 1) / 2.0, patch_size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    weight = np.exp(-0.5 * ((yy / sigma) ** 2 + (xx / sigma) ** 2))
    return np.clip(weight, 0.01, 1.0).astype(np.float32)


def patch_starts(height: int, width: int, patch_size: int, stride: int) -> list[tuple[int, int]]:
    return [
        (row, col)
        for row in range(0, height - patch_size + 1, stride)
        for col in range(0, width - patch_size + 1, stride)
    ]


def predict_probability(
    torch_module,
    model,
    device,
    vv_normalized: np.ndarray,
    data_mask: np.ndarray,
    patch_size: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    if overlap >= patch_size:
        raise ValueError("--overlap must be smaller than --patch-size.")
    stride = patch_size - overlap
    height, width = vv_normalized.shape
    padded = pad_image(
        vv_normalized,
        pad_amount(height, patch_size, stride),
        pad_amount(width, patch_size, stride),
    )
    padded_height, padded_width = padded.shape
    starts = patch_starts(padded_height, padded_width, patch_size, stride)
    if not starts:
        raise RuntimeError("No flood inference patches were generated.")

    weight = gaussian_weight(patch_size)
    weighted_sum = np.zeros_like(padded, dtype=np.float32)
    weight_sum = np.zeros_like(padded, dtype=np.float32)
    weight_tensor = (
        torch_module.from_numpy(weight)
        .to(device)
        .unsqueeze(0)
        .unsqueeze(0)
    )

    for offset in range(0, len(starts), batch_size):
        positions = starts[offset:offset + batch_size]
        patches = [
            np.repeat(
                padded[row:row + patch_size, col:col + patch_size][None, ...],
                3,
                axis=0,
            )
            for row, col in positions
        ]
        batch = torch_module.from_numpy(np.stack(patches).astype(np.float32)).to(device)
        with torch_module.no_grad():
            logits = model(batch)
            probabilities = torch_module.sigmoid(logits) * weight_tensor
        batch_probabilities = probabilities.detach().cpu().numpy().astype(np.float32)
        for index, (row, col) in enumerate(positions):
            weighted_sum[row:row + patch_size, col:col + patch_size] += batch_probabilities[index, 0]
            weight_sum[row:row + patch_size, col:col + patch_size] += weight
        print(
            f"Processed {min(offset + batch_size, len(starts))}/{len(starts)} flood tiles",
            end="\r",
            flush=True,
        )
    print()

    probability = weighted_sum[:height, :width] / np.clip(
        weight_sum[:height, :width],
        1e-6,
        None,
    )
    probability = probability.astype(np.float32)
    probability[~data_mask] = 0.0
    return probability


def write_mask(path: Path, mask: np.ndarray, source_profile: dict) -> None:
    profile = source_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone flood segmentation on band-1 Sentinel-1 VV data."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input SAR GeoTIFF.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_FLOOD_CHECKPOINT,
        help="UNet++ flood checkpoint path. Defaults to the bundled docs asset.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for the binary flood mask.",
    )
    parser.add_argument("--patch-size", type=int, default=512, help="Inference tile size.")
    parser.add_argument(
        "--overlap",
        type=int,
        default=128,
        help="Overlapping pixels between adjacent inference tiles.",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Inference tile batch size.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be a finite value between 0 and 1.")
    if args.patch_size < 1:
        parser.error("--patch-size must be positive.")
    if args.overlap < 0 or args.overlap >= args.patch_size:
        parser.error("--overlap must be non-negative and smaller than --patch-size.")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    vv_normalized, data_mask, profile = read_vv(args.input)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Flood detection requires torch.") from exc
    device = choose_device(torch)
    torch_module, model = load_model(args.checkpoint, device)
    probability = predict_probability(
        torch_module=torch_module,
        model=model,
        device=device,
        vv_normalized=vv_normalized,
        data_mask=data_mask,
        patch_size=args.patch_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    mask = (probability > args.threshold) & data_mask

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mask_path = args.output_dir / "flood_mask.tif"
    metadata_path = args.output_dir / "flood_metadata.json"
    write_mask(mask_path, mask, profile)
    write_json(
        metadata_path,
        {
            "input_path": str(args.input.resolve()),
            "checkpoint_path": str(args.checkpoint.resolve()),
            "config_path": str(DEFAULT_FLOOD_CONFIG.resolve())
            if DEFAULT_FLOOD_CONFIG.is_file()
            else None,
            "mask_path": str(mask_path.resolve()),
            "device": str(device),
            "input_policy": "GeoTIFF band 1 is normalized VV dB and repeated to 3 model channels.",
            "threshold": args.threshold,
            "patch_size": args.patch_size,
            "overlap": args.overlap,
            "batch_size": args.batch_size,
            "sar_db_min": DB_MIN,
            "sar_db_max": DB_MAX,
            "detected_pixels": int(mask.sum()),
            "valid_source_pixels": int(data_mask.sum()),
            "probability_min": float(probability.min()),
            "probability_max": float(probability.max()),
        },
    )
    print(f"Mask: {mask_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Detected pixels: {int(mask.sum())} at threshold {args.threshold:.6g}")


if __name__ == "__main__":
    main()
