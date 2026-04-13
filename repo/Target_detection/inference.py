"""
Clean inference API for Platform integration.

Usage from Platform backend:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'repo', 'Target_detection'))
    from inference import detect
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure the Target_detection package root is on sys.path so that
# internal `from models.xxx import ...` statements resolve correctly.
_PACKAGE_ROOT = Path(__file__).resolve().parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))


# ---------------------------------------------------------------------------
# Supported models
# ---------------------------------------------------------------------------

MLP_MODELS = {
    "new_method_mlp": "NewMethodMLP",
    "new_method_mlp_ace": "NewMethodMLPACE",
}


def available_models() -> List[Dict[str, str]]:
    """Return list of model specs available through this API."""
    return [
        {"id": "new_method_mlp", "name": "MLP + AMF", "description": "Learnable MLP projector with AMF scoring"},
        {"id": "new_method_mlp_ace", "name": "MLP + ACE", "description": "Learnable MLP projector with ACE scoring"},
    ]


# ---------------------------------------------------------------------------
# Model cache (singleton per model name)
# ---------------------------------------------------------------------------

_model_cache: Dict[str, Any] = {}


def _get_model(model_name: str):
    """Lazily build and cache a model instance."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    if model_name not in MLP_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available: {list(MLP_MODELS.keys())}"
        )

    from models.new_method_mlp import NewMethodMLP, NewMethodMLPACE

    if model_name == "new_method_mlp":
        model = NewMethodMLP()
    else:
        model = NewMethodMLPACE()

    _model_cache[model_name] = model
    return model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    cube: np.ndarray,
    pos_points: List[Tuple[int, int]],
    neg_points: Optional[List[Tuple[int, int]]] = None,
    model_name: str = "new_method_mlp",
    threshold: Optional[float] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Run MLP-based target detection on a hyperspectral cube.

    Parameters
    ----------
    cube : np.ndarray, shape (H, W, C)
        Input raster data. For Sentinel-2 this is typically 10-12 bands.
    pos_points : list of (x, y) tuples
        Positive target pixel coordinates (col, row).
    neg_points : list of (x, y) tuples, optional
        Negative (non-target) pixel coordinates.
    model_name : str
        One of "new_method_mlp" (AMF) or "new_method_mlp_ace" (ACE).
    threshold : float, optional
        Detection threshold. None = auto (Otsu).
    progress_callback : callable, optional
        Called with progress dict during MLP training.

    Returns
    -------
    dict with keys:
        "mask"      : np.ndarray (H, W) bool — binary detection mask
        "score_map" : np.ndarray (H, W) float32 — detection score map
        "threshold" : float — threshold used
        "state"     : dict — internal state/debug info
    """
    if neg_points is None:
        neg_points = []

    if len(pos_points) == 0:
        raise ValueError("At least one positive point is required")

    model = _get_model(model_name)

    result = model.predict(
        cube=cube,
        pos_points=pos_points,
        neg_points=neg_points,
        gt_mask=None,
        threshold=threshold,
        progress_callback=progress_callback,
    )

    return {
        "mask": result["mask"],
        "score_map": result["score_map"],
        "threshold": result.get("threshold"),
        "state": result.get("state", {}),
    }
