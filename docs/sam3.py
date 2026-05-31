#!/usr/bin/env python3
"""Standalone SAM3 point or text segmentation for GeoTIFF images."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence, Tuple

import numpy as np
import rasterio
from PIL import Image


Point = Tuple[int, int]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAM3_ASSET_DIR = SCRIPT_DIR / "checkpoints" / "sam3"


def parse_point(value: str) -> Point:
    try:
        col_text, row_text = value.split(",", maxsplit=1)
        return int(col_text.strip()), int(row_text.strip())
    except (AttributeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid point '{value}'. Use col,row, for example 320,220."
        ) from exc


def parse_rgb_bands(value: str) -> list[int]:
    try:
        bands = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid RGB bands '{value}'. Use 3 values such as 3,2,1."
        ) from exc
    if len(bands) != 3 or any(band < 1 for band in bands):
        raise argparse.ArgumentTypeError("RGB bands must be 3 positive 1-based indexes.")
    return bands


def validate_points(points: Sequence[Point], width: int, height: int, label: str) -> None:
    for col, row in points:
        if not (0 <= col < width and 0 <= row < height):
            raise ValueError(
                f"{label} point ({col},{row}) is outside raster size {width}x{height}."
            )


def read_rgb(path: Path, rgb_bands: Sequence[int]) -> tuple[np.ndarray, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Input GeoTIFF not found: {path}")

    with rasterio.open(path) as src:
        invalid = [band for band in rgb_bands if band > src.count]
        if invalid:
            raise ValueError(
                f"RGB band indexes {invalid} exceed input band count {src.count}."
            )
        channels = [src.read(band).astype(np.float32) for band in rgb_bands]
        profile = src.profile.copy()

    rgb = np.stack(channels, axis=-1)
    valid = rgb[np.isfinite(rgb) & (rgb > 0)]
    if valid.size:
        lower, upper = np.percentile(valid, [2, 98])
        if upper > lower:
            rgb = (rgb - lower) / (upper - lower)
        else:
            rgb = rgb / max(float(np.nanmax(rgb)), 1e-8)
    else:
        rgb = np.zeros_like(rgb)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(rgb * 255, 0, 255).astype(np.uint8), profile


def resolve_sam3_assets(
    checkpoint_dir: Path | None,
    bpe_path: Path | None,
) -> tuple[Path, Path | None]:
    asset_dir = checkpoint_dir or DEFAULT_SAM3_ASSET_DIR
    checkpoint_path = asset_dir / "sam3.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            "SAM3 checkpoint not found. Expected sam3.pt at "
            f"{checkpoint_path}. Pass --checkpoint-dir to use another asset directory."
        )
    resolved_bpe = bpe_path or asset_dir / "bpe_simple_vocab_16e6.txt.gz"
    if bpe_path is not None and not resolved_bpe.is_file():
        raise FileNotFoundError(f"SAM3 BPE asset not found: {resolved_bpe}")
    if not resolved_bpe.is_file():
        resolved_bpe = None
    return checkpoint_path, resolved_bpe


def import_sam3_package():
    """Import the external SAM3 package without this `sam3.py` shadowing it."""
    script_path = Path(__file__).resolve()
    loaded = sys.modules.get("sam3")
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path and Path(loaded_path).resolve() == script_path:
        del sys.modules["sam3"]

    original_sys_path = sys.path[:]
    sys.path[:] = [
        entry
        for entry in original_sys_path
        if Path(entry or ".").resolve() != SCRIPT_DIR
    ]
    try:
        sam3_package = importlib.import_module("sam3")
        processor_module = importlib.import_module("sam3.model.sam3_image_processor")
    finally:
        sys.path[:] = original_sys_path
    return sam3_package.build_sam3_image_model, processor_module.Sam3Processor


def load_sam3(checkpoint_path: Path, bpe_path: Path | None):
    try:
        import torch
        build_sam3_image_model, Sam3Processor = import_sam3_package()
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 dependencies are unavailable. Install torch and the sam3 package."
        ) from exc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs: dict[str, Any] = {"enable_inst_interactivity": True}
    kwargs["checkpoint_path"] = str(checkpoint_path.resolve())
    kwargs["load_from_HF"] = False
    if bpe_path is not None:
        kwargs["bpe_path"] = str(bpe_path.resolve())
    model = build_sam3_image_model(**kwargs)
    try:
        model.to(device)
    except Exception:
        pass
    return torch, model, Sam3Processor(model), device


def amp_context(torch_module, device):
    if device.type == "cuda":
        return torch_module.autocast("cuda", dtype=torch_module.bfloat16)
    return contextlib.nullcontext()


def to_numpy(value, torch_module):
    if value is None:
        return None
    if isinstance(value, torch_module.Tensor):
        if value.dtype in (torch_module.bfloat16, torch_module.float16):
            value = value.float()
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)) and value:
        if isinstance(value[0], torch_module.Tensor):
            return np.stack([to_numpy(item, torch_module) for item in value], axis=0)
    return np.asarray(value)


def encode_image(torch_module, processor, device, rgb: np.ndarray):
    with amp_context(torch_module, device), torch_module.inference_mode():
        return processor.set_image(Image.fromarray(rgb, mode="RGB"))


def predict_point_mask(
    torch_module,
    model,
    device,
    state,
    positive_points: Sequence[Point],
    negative_points: Sequence[Point],
) -> tuple[np.ndarray, float]:
    all_points = [*positive_points, *negative_points]
    all_labels = [1] * len(positive_points) + [0] * len(negative_points)
    point_coords = np.asarray(all_points, dtype=np.float32)
    point_labels = np.asarray(all_labels, dtype=np.int32)

    with amp_context(torch_module, device), torch_module.inference_mode():
        masks, scores, _logits = model.predict_inst(
            state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=None,
            multimask_output=True,
        )

    mask_array = to_numpy(masks, torch_module)
    score_array = to_numpy(scores, torch_module).reshape(-1)
    if mask_array.ndim == 4:
        mask_array = mask_array.squeeze(1)
    best_index = int(np.argmax(score_array))
    return mask_array[best_index].astype(bool), float(score_array[best_index])


def predict_text_masks(
    torch_module,
    processor,
    device,
    state,
    prompt: str,
    threshold: float,
) -> list[dict]:
    with amp_context(torch_module, device), torch_module.inference_mode():
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks_raw = output.get("masks") if isinstance(output, dict) else getattr(output, "masks", None)
    scores_raw = output.get("scores") if isinstance(output, dict) else getattr(output, "scores", None)
    boxes_raw = output.get("boxes") if isinstance(output, dict) else getattr(output, "boxes", None)
    if masks_raw is None:
        raise RuntimeError("SAM3 text prediction returned no masks.")

    masks = to_numpy(masks_raw, torch_module)
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    if masks.ndim == 2:
        masks = masks[None, ...]
    scores = (
        to_numpy(scores_raw, torch_module).reshape(-1)
        if scores_raw is not None
        else np.ones(masks.shape[0], dtype=np.float32)
    )
    boxes = (
        to_numpy(boxes_raw, torch_module).reshape(-1, 4)
        if boxes_raw is not None
        else np.zeros((masks.shape[0], 4), dtype=np.float32)
    )

    instances: list[dict] = []
    for index, mask in enumerate(masks):
        score = float(scores[index]) if index < len(scores) else 0.0
        binary = mask.astype(bool)
        if score < threshold or not binary.any():
            continue
        box = [float(value) for value in boxes[index]] if index < len(boxes) else None
        instances.append({"mask": binary, "score": score, "bbox_pixel": box})
    return instances


def write_mask(path: Path, mask: np.ndarray, source_profile: dict) -> None:
    profile = source_profile.copy()
    profile.update(count=1, dtype="uint8", nodata=0)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone SAM3 point or text segmentation on a GeoTIFF."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input GeoTIFF.")
    parser.add_argument(
        "--rgb-bands",
        type=parse_rgb_bands,
        default=parse_rgb_bands("3,2,1"),
        help="1-based GeoTIFF bands for R,G,B. Default: 3,2,1.",
    )
    parser.add_argument(
        "--positive-point",
        action="append",
        default=[],
        type=parse_point,
        help="Point-mode positive point in col,row order. Repeat as needed.",
    )
    parser.add_argument(
        "--negative-point",
        action="append",
        default=[],
        type=parse_point,
        help="Point-mode negative point in col,row order. Repeat as needed.",
    )
    parser.add_argument("--text-prompt", help="Text-mode noun phrase prompt.")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum text-mode SAM3 instance score.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="SAM3 asset directory containing sam3.pt. Defaults to the bundled docs asset.",
    )
    parser.add_argument("--bpe-path", type=Path, help="Optional SAM3 BPE asset path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    has_text = bool(args.text_prompt and args.text_prompt.strip())
    has_points = bool(args.positive_point or args.negative_point)
    if has_text and has_points:
        parser.error("Choose text mode or point mode, not both.")
    if not has_text and not args.positive_point:
        parser.error("Point mode requires at least one --positive-point.")
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0 and 1.")

    rgb, profile = read_rgb(args.input, args.rgb_bands)
    height, width = rgb.shape[:2]
    validate_points(args.positive_point, width, height, "Positive")
    validate_points(args.negative_point, width, height, "Negative")
    checkpoint_path, bpe_path = resolve_sam3_assets(args.checkpoint_dir, args.bpe_path)
    torch_module, model, processor, device = load_sam3(checkpoint_path, bpe_path)
    state = encode_image(torch_module, processor, device, rgb)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "sam3_metadata.json"
    common_metadata = {
        "input_path": str(args.input.resolve()),
        "rgb_bands_1_based": list(args.rgb_bands),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "bpe_path": str(bpe_path.resolve()) if bpe_path is not None else None,
        "device": str(device),
    }

    if has_text:
        instances = predict_text_masks(
            torch_module,
            processor,
            device,
            state,
            args.text_prompt.strip(),
            args.score_threshold,
        )
        metadata_instances = []
        for index, instance in enumerate(instances, start=1):
            mask_path = args.output_dir / f"sam3_instance_{index:03d}.tif"
            write_mask(mask_path, instance["mask"], profile)
            metadata_instances.append(
                {
                    "mask_path": str(mask_path.resolve()),
                    "score": instance["score"],
                    "bbox_pixel": instance["bbox_pixel"],
                    "pixel_count": int(instance["mask"].sum()),
                }
            )
        write_json(
            metadata_path,
            {
                **common_metadata,
                "mode": "text",
                "text_prompt": args.text_prompt.strip(),
                "score_threshold": args.score_threshold,
                "instance_count": len(metadata_instances),
                "instances": metadata_instances,
            },
        )
        print(f"Instances: {len(metadata_instances)}")
    else:
        mask, score = predict_point_mask(
            torch_module,
            model,
            device,
            state,
            args.positive_point,
            args.negative_point,
        )
        mask_path = args.output_dir / "sam3_mask.tif"
        write_mask(mask_path, mask, profile)
        write_json(
            metadata_path,
            {
                **common_metadata,
                "mode": "point",
                "mask_path": str(mask_path.resolve()),
                "score": score,
                "pixel_count": int(mask.sum()),
                "positive_points_col_row": [list(point) for point in args.positive_point],
                "negative_points_col_row": [list(point) for point in args.negative_point],
            },
        )
        print(f"Mask: {mask_path}")
        print(f"Score: {score:.6g}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
