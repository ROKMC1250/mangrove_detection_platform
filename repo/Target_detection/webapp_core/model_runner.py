"""Model execution helpers for web_app routes."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np

from models.band_conversion import convert_to_rgb, get_sensor_for_dataset
from models.webapp_registry import (
    WEBAPP_MODELS,
    predict_model as predict_webapp_model,
    get_loaded_sam_predictor,
)


def _to_points(raw_pts: List[List[int] | Tuple[int, int]]) -> List[Tuple[int, int]]:
    return [(int(x), int(y)) for x, y in raw_pts]


def build_pca_rgb(cube: np.ndarray, loaded_name: str) -> np.ndarray:
    sensor = get_sensor_for_dataset(loaded_name or "")
    return convert_to_rgb(cube, method="pca", sensor=sensor)


def run_single_model(
    *,
    model_name: str,
    cube: np.ndarray,
    loaded_name: str,
    pos_raw: List[List[int] | Tuple[int, int]],
    neg_raw: List[List[int] | Tuple[int, int]],
    gt_mask: np.ndarray | None,
    threshold: float | None,
    progress_callback=None,
) -> Dict[str, Any]:
    H, W, _ = cube.shape
    pos_pts = _to_points(pos_raw)
    neg_pts = _to_points(neg_raw)

    pca_rgb = build_pca_rgb(cube, loaded_name)

    t0 = time.time()
    res = predict_webapp_model(
        model_name=model_name,
        cube=cube,
        pca_rgb=pca_rgb,
        pos_points=pos_pts,
        neg_points=neg_pts,
        gt_mask=gt_mask,
        threshold=threshold,
        shared_sam_predictor=get_loaded_sam_predictor(),
        progress_callback=progress_callback,
    )

    mask_up = cv2.resize(res["mask"], (W, H), interpolation=cv2.INTER_NEAREST)
    score_up = cv2.resize(res["score_map"], (W, H), interpolation=cv2.INTER_LINEAR)
    score_raw = np.asarray(res["score_map"], dtype=np.float32)

    return {
        "model_name": model_name,
        "pos_pts": pos_pts,
        "neg_pts": neg_pts,
        "pca_rgb": pca_rgb,
        "mask_up": mask_up,
        "score_up": score_up,
        "used_threshold": res.get("threshold"),
        "score_min": float(score_raw.min()) if score_raw.size > 0 else 0.0,
        "score_max": float(score_raw.max()) if score_raw.size > 0 else 1.0,
        "state": res.get("state"),
        "elapsed_ms": int((time.time() - t0) * 1000),
    }


def run_all_models(
    *,
    cube: np.ndarray,
    loaded_name: str,
    pos_raw: List[List[int] | Tuple[int, int]],
    neg_raw: List[List[int] | Tuple[int, int]],
    gt_mask: np.ndarray | None,
) -> Dict[str, Any]:
    H, W, _ = cube.shape
    pos_pts = _to_points(pos_raw)
    neg_pts = _to_points(neg_raw)
    pca_rgb = build_pca_rgb(cube, loaded_name)

    model_outputs: Dict[str, Dict[str, Any]] = {}
    for spec in WEBAPP_MODELS:
        model_name = spec.id
        t0 = time.time()
        try:
            res = predict_webapp_model(
                model_name=model_name,
                cube=cube,
                pca_rgb=pca_rgb,
                pos_points=pos_pts,
                neg_points=neg_pts,
                gt_mask=gt_mask,
                threshold=None,
                shared_sam_predictor=get_loaded_sam_predictor(),
            )
            mask_up = cv2.resize(res["mask"], (W, H), interpolation=cv2.INTER_NEAREST)
            score_up = cv2.resize(res["score_map"], (W, H), interpolation=cv2.INTER_LINEAR)
            model_outputs[model_name] = {
                "mask_up": mask_up,
                "score_up": score_up,
                "elapsed_ms": int((time.time() - t0) * 1000),
            }
        except Exception as e:
            model_outputs[model_name] = {"error": str(e)}

    return {
        "model_outputs": model_outputs,
        "pos_pts": pos_pts,
        "neg_pts": neg_pts,
        "pca_rgb": pca_rgb,
    }
