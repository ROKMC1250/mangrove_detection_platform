"""
Web-app model registry and unified prediction entrypoints.

This module is the single source of truth for model names/colors exposed in
web_app.py. Add new models here, then web_app automatically picks them up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.config import CFG


@dataclass(frozen=True)
class WebAppModelSpec:
    id: str
    label: str
    color: str


# Keep this order in sync with UI display order.
WEBAPP_MODELS: List[WebAppModelSpec] = [
    WebAppModelSpec("SAM", "SAM", "#0077BB"),
    WebAppModelSpec("SAM2", "SAM2", "#EE7733"),
    WebAppModelSpec("SAMitizer", "SAMitizer", "#CC3311"),
    WebAppModelSpec("SAMitizer(SAM2)", "SAMitizer(SAM2)", "#009988"),
    WebAppModelSpec("new_method_amf", "new_method_amf", "#EE3377"),
    WebAppModelSpec("new_method_ace", "new_method_ace", "#AA4499"),
    WebAppModelSpec("new_method_mlp", "new_method_mlp", "#228833"),
    WebAppModelSpec("new_method_mlp_ace", "new_method_mlp_ace", "#66AA55"),
    WebAppModelSpec("new_method_bilinear", "new_method_bilinear", "#117733"),
    WebAppModelSpec("new_method_bilinear_ace", "new_method_bilinear_ace", "#44AA99"),
    WebAppModelSpec("ACE", "ACE", "#33BBEE"),
    WebAppModelSpec("AMF", "AMF", "#AA3377"),
    WebAppModelSpec("OSP+AMF", "OSP+AMF", "#CCBB44"),
]

_MODEL_IDS = {m.id for m in WEBAPP_MODELS}
_MODEL_CACHE: Dict[str, object] = {}


def available_models_payload() -> List[Dict[str, str]]:
    return [{"id": m.id, "label": m.label, "color": m.color} for m in WEBAPP_MODELS]


def _build_model(model_name: str):
    if model_name == "SAM":
        from models.sam import SAM

        model = SAM()
        model.setup()
        return model
    if model_name == "SAM2":
        from models.sam2 import SAM2

        model = SAM2()
        model.setup()
        return model
    if model_name == "SAMitizer":
        from models.samitizer import SAMitizer

        model = SAMitizer("osp_amf")
        model.setup()
        return model
    if model_name == "SAMitizer(SAM2)":
        from models.samitizer_sam2 import SAMitizerSAM2

        model = SAMitizerSAM2("osp_amf")
        model.setup()
        return model
    if model_name == "new_method_amf":
        from models.new_method import NewMethodAMF

        return NewMethodAMF()
    if model_name == "new_method_ace":
        from models.new_method import NewMethodACE

        return NewMethodACE()
    if model_name == "new_method_mlp":
        from models.new_method_mlp import NewMethodMLP

        return NewMethodMLP()
    if model_name == "new_method_mlp_ace":
        from models.new_method_mlp import NewMethodMLPACE

        return NewMethodMLPACE()
    if model_name == "new_method_bilinear":
        from models.new_method_bilinear import NewMethodBilinear

        return NewMethodBilinear()
    if model_name == "new_method_bilinear_ace":
        from models.new_method_bilinear import NewMethodBilinearACE

        return NewMethodBilinearACE()
    raise ValueError(f"Unknown web-app model: {model_name}")


def _predict_spectral_model(
    *,
    model_name: str,
    cube: np.ndarray,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    threshold: Optional[float] = None,
) -> Dict[str, np.ndarray]:
    """Run ACE/AMF/OSP+AMF directly via osp_amf core."""
    from models.osp_amf import detect_step

    if model_name == "ACE":
        scoring = "osp_ace"
        use_neg_for_osp = False
    elif model_name == "AMF":
        scoring = "osp_amf"
        use_neg_for_osp = False
    elif model_name == "OSP+AMF":
        scoring = "osp_amf"
        use_neg_for_osp = True
    else:
        raise ValueError(f"Unknown spectral model: {model_name}")

    F = torch.from_numpy(cube.astype(np.float32))
    neg_pts = list(neg_points) if use_neg_for_osp else []
    score_t, mask, state = detect_step(
        F,
        pos_points=pos_points,
        neg_points=neg_pts,
        scoring=scoring,
        cov_method="ledoit_wolf",
        eta=CFG.eta,
        max_cov_samples=CFG.max_cov_samples,
        use_otsu=True,
        use_fpr=False,
        threshold=threshold,
        osp_lambda=CFG.osp_lambda,
    )
    return {
        "mask": mask,
        "score_map": score_t.detach().cpu().numpy().astype(np.float32),
        "threshold": state.get("threshold_used"),
        "state": state,
    }


def get_model(model_name: str):
    if model_name not in _MODEL_IDS:
        raise ValueError(f"Unknown web-app model: {model_name}")
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = _build_model(model_name)
    return _MODEL_CACHE[model_name]


def get_loaded_sam_predictor():
    sam_model = _MODEL_CACHE.get("SAM")
    if sam_model is None:
        return None
    return getattr(sam_model, "_predictor", None)


def predict_model(
    *,
    model_name: str,
    cube: np.ndarray,
    pca_rgb: np.ndarray,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    gt_mask: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
    shared_sam_predictor=None,
    progress_callback=None,
) -> Dict[str, np.ndarray]:
    """
    Run one model with a unified API.

    Returns dict with at least:
      - "mask": (H, W) uint8
      - "score_map": (H, W) float32
    Optional:
      - "threshold": float
    """
    if model_name in ("ACE", "AMF", "OSP+AMF"):
        return _predict_spectral_model(
            model_name=model_name,
            cube=cube,
            pos_points=pos_points,
            neg_points=neg_points,
            threshold=threshold,
        )

    model = get_model(model_name)

    if model_name == "SAM":
        res = model.predict(pca_rgb, pos_points, neg_points)
        return {
            "mask": res["mask"],
            "score_map": res["score_map"],
            "threshold": res.get("threshold"),
        }

    if model_name == "SAM2":
        res = model.predict(pca_rgb, pos_points, neg_points)
        return {
            "mask": res["mask"],
            "score_map": res["score_map"],
            "threshold": res.get("threshold"),
        }

    if model_name == "SAMitizer":
        if shared_sam_predictor is not None and getattr(model, "_predictor", None) is None:
            model._predictor = shared_sam_predictor
        res = model.predict(
            cube, pca_rgb, pos_points, neg_points,
            gt_mask=gt_mask, threshold=threshold,
        )
        return {
            "mask": res["mask"],
            "score_map": res["score_map"],
            "threshold": res.get("threshold"),
        }

    if model_name == "SAMitizer(SAM2)":
        res = model.predict(
            cube, pca_rgb, pos_points, neg_points,
            gt_mask=gt_mask, threshold=threshold,
        )
        return {
            "mask": res["mask"],
            "score_map": res["score_map"],
            "threshold": res.get("threshold"),
        }

    if model_name in ("new_method_amf", "new_method_ace"):
        res = model.predict(
            cube,
            pos_points=pos_points,
            neg_points=neg_points,
            gt_mask=gt_mask,
            threshold=threshold,
        )
        return {
            "mask": res["mask"],
            "score_map": res["score_map"],
            "threshold": res.get("threshold"),
            "state": res.get("state"),
        }

    if model_name in ("new_method_mlp", "new_method_mlp_ace", "new_method_bilinear", "new_method_bilinear_ace"):
        res = model.predict(
            cube,
            pos_points=pos_points,
            neg_points=neg_points,
            gt_mask=gt_mask,
            threshold=threshold,
            progress_callback=progress_callback,
        )
        return {
            "mask": res["mask"],
            "score_map": res["score_map"],
            "threshold": res.get("threshold"),
            "state": res.get("state"),
        }

    raise ValueError(f"Unknown web-app model: {model_name}")
