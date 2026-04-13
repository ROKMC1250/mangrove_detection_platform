#!/usr/bin/env python3
"""
Flask web app for interactive point-prompted target detection.

Features:
  - RGB band selector for multi-band images
  - Extensible model selector
  - Otsu auto-thresholding

Usage:
    conda activate target
    python web_app.py [--port 5000]
"""

from __future__ import annotations

import argparse
import base64
import inspect
import io
import os
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import rasterio
import scipy.io
from flask import Flask, jsonify, request, send_file

try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    _HAS_H5PY = False

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.classical_detectors import (
    estimate_background,
    ALL_DETECTORS,
)
from models.osp_amf import generate_mask
from models.webapp_registry import (
    WEBAPP_MODELS,
    available_models_payload,
)
from models.band_conversion import RGB_BAND_MAP, get_sensor_for_dataset
from webapp_core.model_runner import run_single_model, run_all_models
from webapp_core.save_utils import safe_filename
from webapp_core.visualization import (
    _score_map_to_heatmap_rgb,
    _segmentation_eval_rgb,
    _to_heatmap_b64,
    _mask_overlay_rgb,
    _mask_overlay_b64,
)

# Evaluation metrics (for GT mask comparison)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
try:
    from evaluation import compute_roc, compute_pr, pd_at_fixed_pfa, \
        false_alarm_stability, fp_per_megapixel
    _HAS_EVAL = True
except ImportError:
    _HAS_EVAL = False

app = Flask(__name__)
_SEND_FILE_NAME_KWARG = (
    "download_name"
    if "download_name" in inspect.signature(send_file).parameters
    else "attachment_filename"
)

TARGET_SIZE = 224
MAX_DISPLAY = 512

# ===================================================================
# State
# ===================================================================
_state: Dict = {
    "image_raw": None,       # (H, W, C) float32
    "display_rgb": None,     # (H, W, 3) uint8
    "display_bands": None,   # [r, g, b] 0-based indices used for display_rgb
    "display_pct_low": 2.0,
    "display_pct_high": 98.0,
    "img_h": 0, "img_w": 0, "num_ch": 0,
    "gt_mask": None,         # (H, W) uint8 binary mask for active label
    "gt_mask_raw": None,     # (H, W) uint8 raw multi-class mask (0=bg)
    "gt_label_id": None,     # int or None: selected class id (None=all)
    "gt_class_map": {},      # {int: str} class id -> label name
    "last_score_np": None,   # (Fh, Fw) float32 — latest score map
    "last_mask": None,       # (Fh, Fw) uint8 — latest detection mask
    "last_single_result": None,
    "loaded_name": "",       # stem of the uploaded filename (for default save name)
}
_projector_progress: Dict[str, Dict[str, Any]] = {}
_projector_progress_lock = threading.Lock()
_PROJECTOR_PROGRESS_TTL_SEC = 600.0


def _prune_projector_progress(now: float | None = None) -> None:
    ts = float(time.time() if now is None else now)
    stale_ids = []
    for pid, payload in _projector_progress.items():
        updated = float(payload.get("updated_at", ts))
        if ts - updated > _PROJECTOR_PROGRESS_TTL_SEC:
            stale_ids.append(pid)
    for pid in stale_ids:
        _projector_progress.pop(pid, None)


def _set_projector_progress(progress_id: str | None, payload: Dict[str, Any]) -> None:
    if not progress_id:
        return
    now = float(time.time())
    with _projector_progress_lock:
        _prune_projector_progress(now)
        cur = dict(_projector_progress.get(progress_id) or {})
        cur.update(payload)
        cur["updated_at"] = now
        _projector_progress[progress_id] = cur


def _get_projector_progress(progress_id: str) -> Dict[str, Any] | None:
    now = float(time.time())
    with _projector_progress_lock:
        _prune_projector_progress(now)
        payload = _projector_progress.get(progress_id)
        if payload is None:
            return None
        return dict(payload)

# ===================================================================
# I/O helpers
# ===================================================================

def _read_image_file(file_storage) -> np.ndarray | None:
    raw = file_storage.read()
    filename = (file_storage.filename or "").lower()
    if filename.endswith(".npy"):
        import io
        arr = np.load(io.BytesIO(raw))
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        return arr.astype(np.float32)
    if filename.endswith(".mat"):
        return _read_mat(raw)
    if filename.endswith((".tif", ".tiff")):
        return _read_tif(raw)
    buf = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        return _read_tif(raw)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def _find_hsi_array(d: dict) -> np.ndarray | None:
    """Find the largest 3-D (or 2-D) numeric array in a dict of .mat vars.

    Handles plain arrays and MATLAB structs (scipy structured/object arrays)
    by recursively unwrapping nested fields.
    """
    candidates: list[np.ndarray] = []
    _collect_numeric_arrays(d, candidates, depth=0)
    if not candidates:
        return None
    return max(candidates, key=lambda a: a.size)


def _collect_numeric_arrays(
    obj, out: list, depth: int = 0, max_depth: int = 8,
) -> None:
    """Recursively collect numeric 2-D/3-D arrays from nested .mat structures."""
    if depth > max_depth:
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.startswith("__"):
                continue
            _collect_numeric_arrays(v, out, depth + 1, max_depth)
        return

    if not isinstance(obj, np.ndarray):
        return

    # Plain numeric array
    if obj.dtype.kind in ("f", "i", "u") and obj.ndim in (2, 3) and obj.size > 1:
        out.append(obj)
        return

    # MATLAB struct loaded by scipy → structured dtype with named fields
    if obj.dtype.names is not None:
        for name in obj.dtype.names:
            _collect_numeric_arrays(obj[name], out, depth + 1, max_depth)
        return

    # Object array (e.g. 1x1 struct wrapper) — unwrap elements
    if obj.dtype == object:
        for item in obj.flat:
            _collect_numeric_arrays(item, out, depth + 1, max_depth)


def _read_mat(raw_bytes: bytes) -> np.ndarray | None:
    """Read a .mat file containing a hyperspectral image cube.

    Supports both MATLAB v5/v7 (scipy) and v7.3/HDF5 (h5py).
    Automatically finds the largest 3-D array as the HSI data cube.
    Returns (H, W, C) float32, or None on failure.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name

        # Try scipy.io.loadmat first (MATLAB v5 / v7)
        try:
            mat = scipy.io.loadmat(tmp_path)
            arr = _find_hsi_array(mat)
            if arr is not None:
                print(f"[MAT] Found array shape={arr.shape} dtype={arr.dtype}")
                if arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                return arr.astype(np.float32)
        except NotImplementedError:
            pass  # v7.3 HDF5 format — fall through

        # Try h5py for MATLAB v7.3 (HDF5-based)
        if _HAS_H5PY:
            with h5py.File(tmp_path, "r") as hf:
                candidates = {}
                def _visitor(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.ndim in (2, 3):
                        candidates[name] = obj.shape
                hf.visititems(_visitor)
                if not candidates:
                    return None
                best_key = max(candidates, key=lambda k: np.prod(candidates[k]))
                arr = np.array(hf[best_key])
                # HDF5/MATLAB stores as (bands, cols, rows) — transpose
                if arr.ndim == 3 and arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
                    arr = np.transpose(arr, (2, 1, 0))
                elif arr.ndim == 2:
                    arr = arr[:, :, np.newaxis]
                return arr.astype(np.float32)

        return None
    except Exception as e:
        print(f"[MAT] Failed to read .mat file: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _read_tif(raw_bytes: bytes) -> np.ndarray | None:
    try:
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        with rasterio.open(tmp_path) as ds:
            bands = ds.read()
        os.unlink(tmp_path)
    except Exception:
        return None
    return np.moveaxis(bands, 0, -1).astype(np.float32)


def _sanitize_rgb_bands(
    bands: List[int] | None,
    num_ch: int,
) -> List[int] | None:
    """Validate and clamp RGB band indices to [0, num_ch-1]."""
    if bands is None or len(bands) != 3:
        return None
    out: List[int] = []
    for b in bands:
        try:
            bi = int(b)
        except (TypeError, ValueError):
            bi = 0
        out.append(min(max(bi, 0), num_ch - 1))
    return out


def _default_rgb_bands(num_ch: int, loaded_name: str = "") -> List[int] | None:
    """Pick default RGB bands with dataset-aware mapping (0-based)."""
    if num_ch <= 0:
        return None
    if num_ch <= 3:
        return [0, min(1, num_ch - 1), min(2, num_ch - 1)]

    # Try dataset mapping first (e.g., DSTL/WV3 -> [4,2,1]).
    sensor = get_sensor_for_dataset(loaded_name or "")
    mapped = RGB_BAND_MAP.get(sensor)
    mapped = _sanitize_rgb_bands(list(mapped), num_ch) if mapped is not None else None
    if mapped is not None:
        return mapped

    # Fallback for typical 8-band WorldView ordering.
    if num_ch == 8:
        return [4, 2, 1]
    if num_ch <= 15:
        return [3, 2, 1]
    return [int(num_ch * 0.6), int(num_ch * 0.4), int(num_ch * 0.2)]


def _sanitize_display_percentiles(low: float, high: float) -> tuple[float, float]:
    lo = float(low)
    hi = float(high)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("display percentiles must be finite numbers")
    lo = min(max(lo, 0.0), 99.9)
    hi = min(max(hi, 0.1), 100.0)
    if hi <= lo:
        raise ValueError("upper percentile must be greater than lower percentile")
    return lo, hi


def _display_percentiles() -> tuple[float, float]:
    return _sanitize_display_percentiles(
        _state.get("display_pct_low", 2.0),
        _state.get("display_pct_high", 98.0),
    )


def _render_threshold_overlay_rgb(mask_np: np.ndarray) -> np.ndarray:
    display_rgb = _state.get("display_rgb")
    if display_rgb is None:
        raise RuntimeError("No display image available for segmentation preview")
    return _mask_overlay_rgb(display_rgb, mask_np)


def _render_single_seg_rgb(mask_np: np.ndarray) -> np.ndarray:
    H = int(_state.get("img_h") or 0)
    W = int(_state.get("img_w") or 0)
    if H <= 0 or W <= 0:
        raise RuntimeError("No active image geometry for segmentation preview")
    gt_mask = _state.get("gt_mask")
    if gt_mask is not None:
        return _segmentation_eval_rgb(mask_np, gt_mask)
    return _render_threshold_overlay_rgb(mask_np)


def _render_single_seg_b64(mask_np: np.ndarray) -> str:
    return _np_to_b64(_render_single_seg_rgb(mask_np))


def _compute_binary_metrics(mask_np: np.ndarray) -> Dict[str, Any]:
    gt_mask = _state.get("gt_mask")
    if gt_mask is None:
        return {"has_gt": False}

    pred = (np.asarray(mask_np) > 0).astype(bool)
    gt = gt_mask.astype(bool)
    tp = float((pred & gt).sum())
    fp = float((pred & ~gt).sum())
    fn = float((~pred & gt).sum())
    iou = tp / max(tp + fp + fn, 1.0)
    prec = tp / max(tp + fp, 1.0)
    rec = tp / max(tp + fn, 1.0)
    return {
        "has_gt": True,
        "iou": round(iou, 4),
        "prec": round(prec, 4),
        "rec": round(rec, 4),
    }


def _make_display_rgb(
    img: np.ndarray,
    bands: List[int] | None = None,
    pct_low: float = 2.0,
    pct_high: float = 98.0,
) -> np.ndarray:
    """Percentile-stretch selected bands to uint8 RGB for visualization only."""
    H, W, C = img.shape
    bands = _sanitize_rgb_bands(bands, C)
    pct_low, pct_high = _sanitize_display_percentiles(pct_low, pct_high)
    if bands is not None:
        rgb = np.stack([img[:, :, b] for b in bands], axis=-1)
    elif C > 3:
        if C == 8:
            b_r, b_g, b_b = 4, 2, 1
        elif C <= 15:
            b_r = min(3, C - 1)
            b_g = min(2, C - 1)
            b_b = min(1, C - 1)
        else:
            b_r = int(C * 0.6)
            b_g = int(C * 0.4)
            b_b = int(C * 0.2)
        rgb = np.stack([img[:, :, b_r], img[:, :, b_g], img[:, :, b_b]], axis=-1)
    elif C == 3:
        rgb = img[:, :, :3].copy()
    else:
        rgb = np.concatenate([img[:, :, :1]] * 3, axis=-1)

    for c in range(3):
        lo = np.percentile(rgb[:, :, c], pct_low)
        hi = np.percentile(rgb[:, :, c], pct_high)
        min_range = max(abs(lo), abs(hi)) * 0.5 + 1e-6
        if hi - lo < min_range:
            mid = (lo + hi) * 0.5
            lo = mid - min_range * 0.5
            hi = mid + min_range * 0.5
        rgb[:, :, c] = np.clip((rgb[:, :, c] - lo) / (hi - lo), 0, 1)
    rgb = np.clip(rgb * 1.2, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _np_to_b64(img: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode("ascii")


def _encode_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(
        ".png",
        cv2.cvtColor(np.asarray(img, dtype=np.uint8), cv2.COLOR_RGB2BGR),
    )
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


# ===================================================================
# Routes
# ===================================================================

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("image")
    if f is None:
        return jsonify(error="No image"), 400
    img = _read_image_file(f)
    if img is None:
        return jsonify(error="Cannot decode image"), 400

    raw_fname = (f.filename or "image")
    loaded_name = Path(raw_fname).stem.replace(" ", "_")
    H, W, C = img.shape
    default_bands = _default_rgb_bands(C, loaded_name)
    pct_low, pct_high = _display_percentiles()
    display_rgb = _make_display_rgb(
        img,
        default_bands,
        pct_low=pct_low,
        pct_high=pct_high,
    )

    _state["image_raw"] = img
    _state["display_rgb"] = display_rgb
    _state["display_bands"] = default_bands
    _state["display_pct_low"] = pct_low
    _state["display_pct_high"] = pct_high
    _state["img_h"] = H
    _state["img_w"] = W
    _state["num_ch"] = C
    _state["loaded_name"] = loaded_name
    _state["last_score_np"] = None
    _state["last_mask"] = None
    _state["last_single_result"] = None
    _state["last_all_results"] = None

    available_models = available_models_payload()

    return jsonify(
        image=_np_to_b64(display_rgb),
        disp_w=W, disp_h=H,
        orig_w=W, orig_h=H,
        channels=C,
        rgb_bands=default_bands,
        display_percentiles=[pct_low, pct_high],
        models=available_models,
        loaded_name=loaded_name,
    )


@app.route("/set_bands", methods=["POST"])
def set_bands():
    """Re-render the display image only, preserving the latest score cache."""
    data = request.get_json() or {}
    img = _state["image_raw"]
    if img is None:
        return jsonify(error="No image loaded"), 400

    bands = data.get("bands", _state.get("display_bands"))
    bands = _sanitize_rgb_bands(bands, img.shape[2])
    if bands is None:
        bands = _default_rgb_bands(img.shape[2], _state.get("loaded_name", ""))

    try:
        pct_low, pct_high = _sanitize_display_percentiles(
            data.get("percentile_low", _state.get("display_pct_low", 2.0)),
            data.get("percentile_high", _state.get("display_pct_high", 98.0)),
        )
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    display_rgb = _make_display_rgb(
        img,
        bands,
        pct_low=pct_low,
        pct_high=pct_high,
    )
    _state["display_rgb"] = display_rgb
    _state["display_bands"] = bands
    _state["display_pct_low"] = pct_low
    _state["display_pct_high"] = pct_high
    H, W = _state["img_h"], _state["img_w"]

    resp: Dict[str, Any] = {
        "image": _np_to_b64(display_rgb),
        "disp_w": W,
        "disp_h": H,
        "rgb_bands": bands,
        "percentile_low": pct_low,
        "percentile_high": pct_high,
    }
    single = _state.get("last_single_result") or {}
    if single.get("mask_np") is not None:
        resp["seg"] = _render_single_seg_b64(single["mask_np"])
    return jsonify(resp)


@app.route("/projector_progress/<progress_id>", methods=["GET"])

def projector_progress(progress_id: str):
    payload = _get_projector_progress(progress_id)
    if payload is None:
        return jsonify(found=False, state="missing")
    payload.pop("updated_at", None)
    return jsonify(found=True, **payload)


# ===================================================================
# Run ALL benchmark models at once
# ===================================================================

def _feature_meta_to_label(meta: Dict[str, Any]) -> str:
    """Compact human-readable expression for one engineered feature."""
    t = str(meta.get("type", "?"))
    i = meta.get("i")
    j = meta.get("j")
    if t == "identity":
        return f"x[{i}]"
    if t == "diff":
        return f"x[{i}] - x[{j}]"
    if t == "normdiff":
        return f"(x[{i}] - x[{j}])/(x[{i}] + x[{j}])"
    if t == "ratio":
        return f"x[{i}] / x[{j}]"
    if t == "product":
        return f"x[{i}] * x[{j}]"
    if t == "log":
        return f"log(x[{i}])"
    if t == "square":
        return f"x[{i}]^2"
    return str(meta)


def _build_single_analysis_payload(
    model_name: str,
    state: Dict[str, Any] | None,
) -> Dict[str, Any] | None:
    """Extract compact UI payload for new_method models."""
    if state is None:
        return None
    if model_name not in (
        "new_method_amf",
        "new_method_ace",
        "new_method_mlp",
        "new_method_mlp_ace",
        "new_method_bilinear",
        "new_method_bilinear_ace",
    ):
        return None

    selected_idx = [int(v) for v in (state.get("feature_selected_idx") or [])]
    selected_meta = state.get("feature_selected_meta") or []
    selected_labels = state.get("feature_selected_labels") or []
    sep_score = state.get("feature_sep_score") or []
    final_score = state.get("feature_final_score") or []
    feat_weights = state.get("feature_weights")

    rows = []
    max_rows = 80
    for rank, idx in enumerate(selected_idx[:max_rows], start=1):
        meta = selected_meta[rank - 1] if (rank - 1) < len(selected_meta) else {}
        if not isinstance(meta, dict):
            meta = {"raw": str(meta)}
        label = (
            str(selected_labels[rank - 1])
            if (rank - 1) < len(selected_labels)
            else _feature_meta_to_label(meta)
        )
        row: Dict[str, Any] = {
            "rank": rank,
            "band": rank,
            "idx": idx,
            "label": label,
            "meta": meta,
        }
        if 0 <= idx < len(sep_score):
            row["sep_score"] = float(sep_score[idx])
        if 0 <= idx < len(final_score):
            row["final_score"] = float(final_score[idx])
        if isinstance(feat_weights, list) and 0 <= idx < len(feat_weights):
            row["weight"] = float(feat_weights[idx])
        rows.append(row)

    analysis = {
        "enabled": True,
        "model_name": model_name,
        "detector": state.get("detector"),
        "feature_bank_size": int(state.get("feature_bank_size", 0) or 0),
        "feature_selected_dim": len(selected_idx),
        "feature_selection_mode": state.get("feature_selection_mode", ""),
        "feature_norm_method": state.get("feature_norm_method", ""),
        "feature_used_neg_fallback": bool(state.get("feature_used_neg_fallback", False)),
        "feature_rows_total": len(selected_idx),
        "feature_rows": rows,
        "feature_axis_labels": selected_labels,
        "spectra": {
            "display_space": (
                state.get("projector_used") and "projected_detector_space"
            ) or "normalized_selected_preweight",
            "pos": (
                state.get("pos_spectra_proj")
                or state.get("pos_spectra_norm")
                or state.get("pos_spectra_raw")
                or []
            ),
            "neg": (
                state.get("neg_spectra_proj")
                or state.get("neg_spectra_norm")
                or state.get("neg_spectra_raw")
                or []
            ),
            "target": (
                state.get("target_spectrum_proj")
                or state.get("target_spectrum_norm")
                or state.get("target_spectrum")
                or []
            ),
            "mu_b": (
                state.get("mu_bg_spectrum_proj")
                or state.get("mu_bg_spectrum_norm")
                or state.get("mu_B")
                or []
            ),
        },
        "r_inv": {
            "matrix": state.get("r_inv_matrix") or [],
            "dim_full": int(state.get("r_inv_dim", 0) or 0),
            "dim_shown": int(state.get("r_inv_dim_shown", 0) or 0),
        },
        "cov_info": state.get("cov_info") or {},
        "memory_estimate": state.get("memory_estimate") or {},
        "gpu_mem": state.get("gpu_mem") or {},
        "projector_used": bool(state.get("projector_used", False)),
        "projector_train_info": state.get("projector_train_info") or {},
    }
    return analysis

@app.route("/detect_one", methods=["POST"])
def detect_one():
    """Run a SINGLE named benchmark model and return heatmap + seg map + metrics."""
    data = request.get_json() or {}
    model_name = data.get("model", "SAM")
    pos_raw = data.get("pos", [])
    neg_raw = data.get("neg", [])
    progress_id = str(data.get("progress_id") or "").strip() or None
    # Optional explicit threshold override. None -> auto Otsu for thresholdable models.
    raw_thr = data.get("threshold", None)
    threshold = float(raw_thr) if raw_thr is not None else None

    cube = _state.get("image_raw")
    if cube is None:
        return jsonify(error="No image loaded"), 400
    if not pos_raw:
        return jsonify(error="Need at least 1 positive point"), 400

    H, W, _ = cube.shape
    gt_mask = _state.get("gt_mask")
    loaded_name = _state.get("loaded_name", "")
    is_projector_model = model_name in ("new_method_mlp", "new_method_mlp_ace", "new_method_bilinear", "new_method_bilinear_ace")
    if is_projector_model and progress_id is None:
        progress_id = uuid.uuid4().hex

    progress_callback = None
    if is_projector_model:
        _set_projector_progress(progress_id, {
            "state": "starting",
            "model_name": model_name,
            "step": 0,
            "n_steps": 0,
            "loss_history": [],
            "diag_history": [],
            "loss_mode": "pending",
        })

        def _progress_callback(update: Dict[str, Any]) -> None:
            payload = dict(update)
            payload["model_name"] = model_name
            _set_projector_progress(progress_id, payload)

        progress_callback = _progress_callback

    try:
        out = run_single_model(
            model_name=model_name,
            cube=cube,
            loaded_name=loaded_name,
            pos_raw=pos_raw,
            neg_raw=neg_raw,
            gt_mask=gt_mask,
            threshold=threshold,
            progress_callback=progress_callback,
        )
        mask_np = out["mask_up"]
        score_np = out["score_up"]
        elapsed = out["elapsed_ms"]
        pos_pts = out["pos_pts"]
        neg_pts = out["neg_pts"]
        used_thr = out.get("used_threshold")
        score_min = out.get("score_min")
        score_max = out.get("score_max")
        model_state = out.get("state")
    except Exception as e:
        print(f"[detect_one] {model_name}: {e}")
        if is_projector_model:
            _set_projector_progress(progress_id, {
                "state": "error",
                "model_name": model_name,
                "error": str(e),
            })
        return jsonify(error=str(e)), 500

    # Heatmap
    heat = _to_heatmap_b64(score_np, H, W)

    seg = _render_single_seg_b64(mask_np)

    # Metrics
    metrics = {}
    if gt_mask is not None:
        pred = (mask_np > 0).astype(bool)
        gt   = gt_mask.astype(bool)
        tp   = float((pred & gt).sum())
        fp   = float((pred & ~gt).sum())
        fn   = float((~pred & gt).sum())
        iou  = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        metrics = {"iou": round(iou, 4), "prec": round(prec, 4), "rec": round(rec, 4)}
        if _HAS_EVAL:
            s = score_np
            if s.shape != gt_mask.shape:
                s = cv2.resize(s, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            pr = compute_pr(s, gt_mask, n_thresholds=500)
            step = max(1, len(pr["recall"]) // 200)
            metrics["pr"] = {
                "recall":    pr["recall"][::step].tolist(),
                "precision": pr["precision"][::step].tolist(),
                "ap": round(pr["ap"], 4),
            }

    # Cache for save_results
    _state["last_score_np"] = score_np
    _state["last_mask"] = mask_np
    _state["last_single_result"] = {
        "model_name": model_name,
        "score_np": score_np,
        "mask_np": mask_np,
        "pos_pts": pos_pts,
        "neg_pts": neg_pts,
        "used_threshold": float(used_thr) if used_thr is not None else None,
    }

    analysis = _build_single_analysis_payload(model_name, model_state)

    return jsonify(heatmap=heat, seg=seg, ms=elapsed,
                   has_gt=(gt_mask is not None),
                   progress_id=progress_id,
                   used_threshold=float(used_thr) if used_thr is not None else None,
                   threshold_min=float(score_min) if score_min is not None else None,
                   threshold_max=float(score_max) if score_max is not None else None,
                   analysis=analysis,
                   **metrics)


@app.route("/preview_threshold", methods=["POST"])
def preview_threshold():
    data = request.get_json() or {}
    model_name = str(data.get("model") or "").strip()
    raw_threshold = data.get("threshold", None)
    if raw_threshold is None:
        return jsonify(error="Missing threshold"), 400

    single = _state.get("last_single_result") or {}
    if single.get("score_np") is None:
        return jsonify(error="No cached single-model result"), 400
    if model_name and single.get("model_name") != model_name:
        return jsonify(error="Cached result does not match selected model"), 409

    try:
        threshold = float(raw_threshold)
    except Exception:
        return jsonify(error="Invalid threshold"), 400

    score_np = np.asarray(single["score_np"], dtype=np.float32)
    mask_np, used_threshold = generate_mask(
        score_np,
        theta=threshold,
        return_threshold=True,
    )

    _state["last_score_np"] = score_np
    _state["last_mask"] = mask_np
    single["mask_np"] = mask_np
    single["used_threshold"] = float(used_threshold)
    _state["last_single_result"] = single

    resp: Dict[str, Any] = {
        "seg": _render_single_seg_b64(mask_np),
        "used_threshold": float(used_threshold),
    }
    resp.update(_compute_binary_metrics(mask_np))
    return jsonify(resp)


@app.route("/detect_all", methods=["POST"])
def detect_all():
    """Run all benchmark models simultaneously.

    SAM-family models receive PCA-converted RGB.
    Spectral-style models receive raw spectral cube.
    When GT mask is loaded: returns IoU, Precision, Recall per model + PR curve data.
    """
    data = request.get_json() or {}
    pos_raw = data.get("pos", [])
    neg_raw = data.get("neg", [])

    cube = _state.get("image_raw")
    if cube is None:
        return jsonify(error="No image loaded"), 400
    if not pos_raw:
        return jsonify(error="Need at least 1 positive point"), 400

    H, W, _ = cube.shape
    gt_mask = _state.get("gt_mask")
    loaded_name = _state.get("loaded_name", "")

    def _seg_b64(mask_hw, score_np_hw):
        heat = _to_heatmap_b64(score_np_hw, H, W)
        if gt_mask is not None:
            seg = _np_to_b64(_segmentation_eval_rgb(mask_hw, gt_mask))
        else:
            seg = _np_to_b64(_mask_overlay_rgb(_state["display_rgb"], mask_hw))
        return heat, seg

    def _metrics(mask_hw, score_np_hw):
        if gt_mask is None:
            return {}
        pred = (mask_hw > 0).astype(bool)
        gt   = gt_mask.astype(bool)
        tp   = float((pred & gt).sum())
        fp   = float((pred & ~gt).sum())
        fn   = float((~pred & gt).sum())
        union = float((pred | gt).sum())
        iou  = tp / max(union, 1)
        prec = tp / max(tp + fp, 1)
        rec  = tp / max(tp + fn, 1)
        f1   = 2 * prec * rec / max(prec + rec, 1e-10)
        result = {
            "iou":  round(iou, 4),
            "prec": round(prec, 4),
            "rec":  round(rec, 4),
            "f1":   round(f1, 4),
        }
        if _HAS_EVAL:
            s = score_np_hw
            if s.shape != gt_mask.shape:
                s = cv2.resize(s, (gt_mask.shape[1], gt_mask.shape[0]),
                               interpolation=cv2.INTER_LINEAR)
            pr = compute_pr(s, gt_mask, n_thresholds=500)
            step = max(1, len(pr["recall"]) // 200)
            result["pr"] = {
                "recall":    pr["recall"][::step].tolist(),
                "precision": pr["precision"][::step].tolist(),
                "ap": round(pr["ap"], 4),
            }
        return result

    run_out = run_all_models(
        cube=cube,
        loaded_name=loaded_name,
        pos_raw=pos_raw,
        neg_raw=neg_raw,
        gt_mask=gt_mask,
    )
    pos_pts = run_out["pos_pts"]
    neg_pts = run_out["neg_pts"]
    pca_rgb = run_out["pca_rgb"]

    model_results = {}
    for model_name, out in run_out["model_outputs"].items():
        if out.get("error"):
            print(f"[detect_all] {model_name} error: {out['error']}")
            model_results[model_name] = {"error": out["error"]}
            continue
        mask_up = out["mask_up"]
        score_up = out["score_up"]
        heat, seg = _seg_b64(mask_up, score_up)
        model_results[model_name] = {
            "heatmap": heat, "seg": seg,
            "ms": out["elapsed_ms"],
            **_metrics(mask_up, score_up),
        }

    # Cache for save_results
    _state["last_all_results"] = {
        "model_results": model_results,
        "pos_pts": pos_pts,
        "neg_pts": neg_pts,
        "pca_rgb": pca_rgb,
    }

    has_gt = gt_mask is not None
    return jsonify(models=model_results, has_gt=has_gt,
                   gt_pixels=int(gt_mask.sum()) if has_gt else 0)

# ===================================================================
# GT mask upload
# ===================================================================

_DSTL_CLASS_MAP = {
    1: "Buildings", 2: "Misc_Manmade_Structures", 3: "Road",
    4: "Track", 5: "Trees", 6: "Crops", 7: "Waterway",
    8: "Standing_Water", 9: "Vehicle_Large", 10: "Vehicle_Small",
}


def _apply_gt_label(label_id):
    """Set gt_mask from gt_mask_raw for the chosen label_id (None=all)."""
    raw = _state["gt_mask_raw"]
    if raw is None:
        return
    if label_id is None:
        _state["gt_mask"] = (raw > 0).astype(np.uint8)
    else:
        _state["gt_mask"] = (raw == label_id).astype(np.uint8)
    _state["gt_label_id"] = label_id


@app.route("/load_gt_mask", methods=["POST"])
def load_gt_mask():
    """Upload a ground-truth mask (.npy or image) for evaluation.

    Multi-class masks (max > 1) are detected automatically. The response
    includes ``labels`` so the frontend can offer a class selector.
    """
    f = request.files.get("gt_mask")
    if f is None:
        return jsonify(error="No file"), 400

    filename = (f.filename or "").lower()
    raw_bytes = f.read()
    gt = None

    if filename.endswith(".npy"):
        import io
        gt = np.load(io.BytesIO(raw_bytes))
    else:
        buf = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            gt = (img > 127).astype(np.uint8)

    if gt is None:
        return jsonify(error="Cannot decode GT mask"), 400

    H, W = _state["img_h"], _state["img_w"]
    if gt.shape != (H, W):
        gt = cv2.resize(gt, (W, H), interpolation=cv2.INTER_NEAREST)

    is_multiclass = int(gt.max()) > 1
    unique_ids = sorted(int(v) for v in np.unique(gt) if v > 0)

    class_map = {}
    if is_multiclass:
        for uid in unique_ids:
            class_map[uid] = _DSTL_CLASS_MAP.get(uid, f"class_{uid}")

    _state["gt_mask_raw"] = gt.astype(np.uint8)
    _state["gt_class_map"] = class_map

    if is_multiclass:
        _apply_gt_label(unique_ids[0])
    else:
        _state["gt_mask"] = (gt > 0).astype(np.uint8)
        _state["gt_label_id"] = None

    binary = _state["gt_mask"]
    n_pos = int(binary.sum())
    print(f"[GT] Loaded mask: {n_pos} target pixels / {H*W} total"
          f"  (multiclass={is_multiclass}, labels={unique_ids})")

    labels_info = []
    if is_multiclass:
        labels_info.append({"id": 0, "name": "All Classes",
                            "pixels": int((gt > 0).sum())})
        for uid in unique_ids:
            labels_info.append({
                "id": uid,
                "name": class_map.get(uid, f"class_{uid}"),
                "pixels": int((gt == uid).sum()),
            })

    return jsonify(
        ok=True,
        target_pixels=n_pos,
        total_pixels=H * W,
        is_multiclass=is_multiclass,
        labels=labels_info,
        selected_label=int(_state["gt_label_id"]) if _state["gt_label_id"] is not None else 0,
    )


@app.route("/set_gt_label", methods=["POST"])
def set_gt_label():
    """Switch the active GT label for a multi-class mask."""
    data = request.get_json() or {}
    label_id = data.get("label_id", 0)

    if _state.get("gt_mask_raw") is None:
        return jsonify(error="No GT mask loaded"), 400

    if label_id == 0:
        _apply_gt_label(None)
    else:
        _apply_gt_label(int(label_id))

    binary = _state["gt_mask"]
    n_pos = int(binary.sum())
    label_name = _state["gt_class_map"].get(label_id, "All Classes") if label_id else "All Classes"
    print(f"[GT] Switched to label {label_id} ({label_name}): {n_pos} target px")

    return jsonify(ok=True, target_pixels=n_pos, label_name=label_name)


# ===================================================================
# Save results helpers
# ===================================================================

def _draw_points_on_image(
    rgb: np.ndarray,
    ordered_points: list,      # [(x, y, 'pos'|'neg'), ...]  in click order
    include_neg: bool = True,
    size_scale: float = 1.0,
) -> np.ndarray:
    """Draw prompt markers on a copy of the displayed RGB image."""
    out = rgb.copy()
    H, W = out.shape[:2]
    scale = max(size_scale, 0.1)
    arm = max(6, int(round(min(H, W) / 64.0 * scale)))
    outer_thickness = max(3, int(round(2.5 * scale)))
    inner_thickness = max(1, int(round(1.5 * scale)))
    dot_radius = max(2, int(round(2.5 * scale)))
    dot_border = max(1, int(round(1.0 * scale)))

    for x, y, pt_type in ordered_points:
        if pt_type == "neg" and not include_neg:
            continue
        px = int(np.clip(round(x), 0, W - 1))
        py = int(np.clip(round(y), 0, H - 1))
        color_rgb = (0, 220, 0) if pt_type == "pos" else (220, 30, 30)
        cv2.line(out, (px - arm, py), (px + arm, py), (255, 255, 255), outer_thickness)
        cv2.line(out, (px, py - arm), (px, py + arm), (255, 255, 255), outer_thickness)
        cv2.line(out, (px - arm, py), (px + arm, py), color_rgb, inner_thickness)
        cv2.line(out, (px, py - arm), (px, py + arm), color_rgb, inner_thickness)
        cv2.circle(out, (px, py), dot_radius, color_rgb, -1)
        cv2.circle(out, (px, py), dot_radius, (255, 255, 255), dot_border)

    return out


# ===================================================================
# Save results
# ===================================================================

@app.route("/save_results", methods=["POST"])
def save_results():
    """Download the current single-model result as a ZIP to the browser."""
    from datetime import datetime

    data = request.get_json() or {}
    pos_raw = data.get("pos", [])
    neg_raw = data.get("neg", [])
    result_name = str(data.get("result_name", "") or "").strip()
    model_name = str(data.get("model", "") or "").strip()

    cube = _state.get("image_raw")
    display_rgb = _state.get("display_rgb")
    single = _state.get("last_single_result") or {}

    if cube is None:
        return jsonify(error="No image loaded"), 400
    if display_rgb is None:
        return jsonify(error="No display image available"), 400
    if single.get("score_np") is None or single.get("mask_np") is None:
        return jsonify(error="Run a single model first"), 400

    cached_model = str(single.get("model_name") or "")
    if model_name and cached_model and model_name != cached_model:
        return jsonify(error="Cached result does not match the selected model"), 409

    if not result_name:
        base = _state.get("loaded_name") or "interactive"
        model_suffix = safe_filename(cached_model or "result")
        result_name = f"{base}_{model_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if result_name.lower().endswith(".zip"):
        result_name = result_name[:-4]
    result_name = safe_filename(result_name.replace("/", "_").replace(" ", "_"))
    if not result_name:
        result_name = "interactive_result"

    pos_pts = [(int(x), int(y)) for x, y in pos_raw] if pos_raw else list(single.get("pos_pts") or [])
    neg_pts = [(int(x), int(y)) for x, y in neg_raw] if neg_raw else list(single.get("neg_pts") or [])
    ordered_points = (
        [(x, y, "pos") for x, y in pos_pts] +
        [(x, y, "neg") for x, y in neg_pts]
    )

    input_rgb = _draw_points_on_image(
        display_rgb,
        ordered_points,
        include_neg=True,
        size_scale=1.0,
    )
    score_rgb = _score_map_to_heatmap_rgb(np.asarray(single["score_np"], dtype=np.float32))
    threshold_rgb = _render_threshold_overlay_rgb(np.asarray(single["mask_np"], dtype=np.uint8))

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{result_name}_input.png", _encode_png_bytes(input_rgb))
        zf.writestr(f"{result_name}_score_map.png", _encode_png_bytes(score_rgb))
        zf.writestr(f"{result_name}_threshold_map.png", _encode_png_bytes(threshold_rgb))

    zip_buffer.seek(0)
    print(
        f"[save_results] Prepared ZIP download for {cached_model or 'single_result'} "
        f"with 3 images"
    )
    send_kwargs = {
        "mimetype": "application/zip",
        "as_attachment": True,
        _SEND_FILE_NAME_KWARG: f"{result_name}.zip",
    }
    return send_file(zip_buffer, **send_kwargs)


@app.route("/get_pixel_spectrum", methods=["POST"])
def get_pixel_spectrum():
    """Return the raw spectrum at a given pixel (x, y) in original coordinates."""
    data = request.get_json()
    x, y = int(data["x"]), int(data["y"])
    cube = _state.get("image_raw")
    if cube is None:
        return jsonify(error="No image loaded"), 400
    H, W, C = cube.shape
    if not (0 <= y < H and 0 <= x < W):
        return jsonify(error="Pixel out of bounds"), 400
    spec = cube[y, x, :].tolist()
    return jsonify(spectrum=spec, x=x, y=y, bands=C)


@app.route("/detect_classical", methods=["POST"])
def detect_classical():
    """Run a classical spectral target detector (MF/SAM/ACE).

    Expects JSON:
      algorithm: "MF" | "SAM" | "ACE"
      target_pixels: [[x1,y1], [x2,y2], ...]   pixel coords for target spectrum
    """
    data = request.get_json()
    algo = data.get("algorithm", "MF").upper()
    target_pixels = data.get("target_pixels", [])

    allowed_algorithms = {"MF", "SAM", "ACE"}
    if algo not in allowed_algorithms:
        return jsonify(error=f"Unknown algorithm: {algo}"), 400

    cube = _state.get("image_raw")
    if cube is None:
        return jsonify(error="No image loaded"), 400

    H, W, C = cube.shape

    if not target_pixels:
        return jsonify(error="Need at least 1 target pixel"), 400

    # Build target spectrum as mean of selected pixels
    specs = []
    for px in target_pixels:
        x, y = int(px[0]), int(px[1])
        if 0 <= y < H and 0 <= x < W:
            specs.append(cube[y, x, :])
    if not specs:
        return jsonify(error="All target pixels out of bounds"), 400

    target_spectrum = np.mean(specs, axis=0).astype(np.float32)

    t0 = time.time()
    detector_fn = ALL_DETECTORS[algo]
    if algo == "SAM":
        score_np = detector_fn(cube, target_spectrum)
    else:
        mu, R_inv, _ = estimate_background(cube)
        score_np = detector_fn(cube, target_spectrum, mu=mu, R_inv=R_inv)
    elapsed = time.time() - t0

    # Auto-threshold: Otsu on the score map
    s_min, s_max = float(score_np.min()), float(score_np.max())
    if s_max - s_min > 1e-10:
        norm = ((score_np - s_min) / (s_max - s_min) * 255).astype(np.uint8)
        _, mask = cv2.threshold(norm, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        mask = np.zeros((H, W), dtype=np.uint8)

    # Cache for saving
    _state["last_score_np"] = score_np
    _state["last_mask"] = mask

    det_px = int(mask.astype(bool).sum())
    rgb = _state["display_rgb"]
    scale = min(MAX_DISPLAY / W, MAX_DISPLAY / H, 1.0)
    dw, dh = int(W * scale), int(H * scale)

    resp = dict(
        heatmap=_to_heatmap_b64(score_np, dh, dw),
        overlay=_mask_overlay_b64(rgb, mask, dh, dw),
        ms=int(elapsed * 1000),
        det=det_px,
        total=H * W,
        model=f"classical_{algo}",
        algorithm=algo,
        target_spectrum=target_spectrum.tolist(),
    )

    # Evaluation metrics if GT mask is loaded
    gt_mask = _state.get("gt_mask")
    if gt_mask is not None and _HAS_EVAL:
        gt_eval = gt_mask
        score_eval = score_np
        if score_eval.shape != gt_eval.shape:
            score_eval = cv2.resize(
                score_eval, (gt_eval.shape[1], gt_eval.shape[0]),
                interpolation=cv2.INTER_LINEAR)

        roc = compute_roc(score_eval, gt_eval, n_thresholds=500)
        pr = compute_pr(score_eval, gt_eval, n_thresholds=500)
        pd_table = pd_at_fixed_pfa(roc)
        stab = false_alarm_stability(score_eval, gt_eval)
        fp_mp = {}
        for pfa_val in [1e-2, 1e-3, 1e-4]:
            fp_mp[str(pfa_val)] = fp_per_megapixel(
                score_eval, gt_eval, target_pfa=pfa_val)

        step = max(1, len(roc["pfa"]) // 200)
        resp["eval"] = {
            "roc_pfa": roc["pfa"][::step].tolist(),
            "roc_pd": roc["pd"][::step].tolist(),
            "auc": roc["auc"],
            "pr_recall": pr["recall"][::step].tolist(),
            "pr_precision": pr["precision"][::step].tolist(),
            "ap": pr["ap"],
            "pd_at_pfa": {str(k): round(v, 4) for k, v in pd_table.items()},
            "stability_tile_pfa": stab["tile_pfa"].tolist(),
            "stability_target": stab["target_pfa"],
            "fp_per_megapixel": fp_mp,
            "gt_pixels": int(gt_eval.sum()),
        }

    return jsonify(resp)


# ===================================================================
# HTML
# ===================================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Point-Prompted Target Detection</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e0e0e0;min-height:100vh}
header{background:linear-gradient(135deg,#1a1d29,#252836);padding:14px 24px;border-bottom:1px solid #2d3044;display:flex;align-items:center;gap:16px}
header h1{font-size:17px;font-weight:600;color:#fff}

.toolbar{background:#181b24;padding:10px 24px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid #2d3044}
.btn{padding:6px 14px;border:none;border-radius:6px;font-size:12px;cursor:pointer;font-weight:500;transition:.15s}
.btn-primary{background:#3b82f6;color:#fff}.btn-primary:hover{background:#2563eb}
.btn-danger{background:#ef4444;color:#fff}.btn-danger:hover{background:#dc2626}
.btn-outline{background:transparent;color:#a0a8c0;border:1px solid #3a3f55}
.btn-outline:hover{border-color:#5b6280;color:#fff}
.btn:disabled{opacity:.4;cursor:not-allowed}
select,input[type=number]{background:#1e2230;color:#e0e0e0;border:1px solid #3a3f55;border-radius:5px;padding:4px 8px;font-size:12px}
select:focus,input:focus{outline:none;border-color:#3b82f6}
label{font-size:11px;color:#8890a8}
.sep{width:1px;height:24px;background:#2d3044;margin:0 4px}
#file-input{display:none}
.legend{display:flex;gap:14px;align-items:center;font-size:11px;color:#8890a8;margin-left:auto}
.legend-item{display:flex;align-items:center;gap:4px}
.ldot{width:9px;height:9px;border-radius:50%;border:2px solid #fff}
.ldot-pos{background:#22c55e}.ldot-neg{background:#ef4444}
.spinner{display:none;width:18px;height:18px;border:2px solid #3b82f6;border-top-color:transparent;border-radius:50%;animation:spin .6s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}

.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:16px 24px;padding-bottom:50px}
@media(max-width:1400px){.grid{grid-template-columns:repeat(3,1fr)}}
@media(max-width:900px){.grid{grid-template-columns:1fr 1fr}}
@media(max-width:600px){.grid{grid-template-columns:1fr}}
.panel{background:#181b24;border:1px solid #2d3044;border-radius:8px;overflow:hidden;display:flex;flex-direction:column}
.ptitle{padding:8px 12px;font-size:12px;font-weight:600;background:#1e2230;border-bottom:1px solid #2d3044;display:flex;align-items:center;gap:6px}
.ptitle .meta{margin-left:auto;font-weight:400;color:#8890a8;font-size:11px}
.dot{width:7px;height:7px;border-radius:50%}
.pbody{padding:6px;display:flex;align-items:center;justify-content:center;min-height:180px;position:relative}
.pbody img{max-width:100%;border-radius:3px;display:block;cursor:crosshair}
.sync-zoom-media{position:absolute;left:0;top:0;display:block;transform-origin:0 0;max-width:none !important;max-height:none !important;border-radius:3px}
.result-pbody{overflow:hidden;align-items:flex-start;justify-content:flex-start}
.result-zoom-wrap{position:relative;overflow:hidden;display:none;background:#10131b;border-radius:3px;flex:0 0 auto}
.ph{color:#555b72;font-size:13px;text-align:center;padding:30px}
canvas{cursor:crosshair;display:block;border-radius:3px}
.zoom-wrap{position:relative;overflow:hidden;width:100%;height:100%}
.zoom-info{position:absolute;bottom:4px;right:6px;font-size:10px;color:#8890a8;background:rgba(15,17,23,.75);padding:2px 6px;border-radius:3px;pointer-events:none;z-index:2}
.pixel-info{position:absolute;top:4px;left:6px;font-size:10px;color:#c084fc;background:rgba(15,17,23,.8);padding:2px 6px;border-radius:3px;pointer-events:none;z-index:2;font-family:monospace}

.info-bar{background:#181b24;padding:8px 24px;border-top:1px solid #2d3044;font-size:11px;color:#8890a8;display:flex;gap:20px;flex-wrap:wrap;position:fixed;bottom:0;left:0;right:0}
.info-bar span{display:flex;align-items:center;gap:3px}
.info-bar .v{color:#fff;font-weight:600}

.band-group{display:flex;align-items:center;gap:4px}
.band-group input[type=number]{width:42px;text-align:center}
#sel-model{min-width:160px;font-weight:600}

/* Toggle switch */
.toggle-label{display:flex;align-items:center;cursor:pointer;gap:2px}
.toggle-label input{display:none}
.toggle-slider{width:34px;height:18px;background:#3a3f55;border-radius:9px;position:relative;transition:.2s}
.toggle-slider::after{content:'';position:absolute;left:2px;top:2px;width:14px;height:14px;background:#888;border-radius:50%;transition:.2s}
.toggle-label input:checked+.toggle-slider{background:#3b82f6}
.toggle-label input:checked+.toggle-slider::after{transform:translateX(16px);background:#fff}
#table-pd-pfa td{padding:6px 8px;border-bottom:1px solid #1e2230}
#table-pd-pfa tr:hover td{background:#1e2230}
.analysis-wrap{padding:10px 24px 6px 24px}
.analysis-grid{display:grid;grid-template-columns:1.15fr .85fr .95fr;gap:12px}
@media(max-width:900px){.analysis-grid{grid-template-columns:1fr}}
.analysis-card{background:#181b24;border:1px solid #2d3044;border-radius:8px;padding:10px 12px}
.analysis-title{font-size:11px;font-weight:700;color:#9bb5ff;text-transform:uppercase;letter-spacing:.4px;margin-bottom:8px}
.analysis-canvas{position:relative;height:220px}
.analysis-note{font-size:11px;color:#8890a8;margin-top:6px}
.feature-details{margin:4px 16px 8px}
.feature-details details{background:#12151e;border:1px solid #2a2f45;border-radius:8px;padding:8px 10px}
.feature-details summary{cursor:pointer;color:#c4cbde;font-size:12px;font-weight:600}
.feature-summary{font-size:11px;color:#8890a8;margin:8px 0 6px 0}
.feature-table-wrap{max-height:240px;overflow:auto;border:1px solid #2a2f45;border-radius:6px}
.feature-table{width:100%;border-collapse:collapse;font-size:11px}
.feature-table th,.feature-table td{padding:5px 8px;border-bottom:1px solid #1e2230;text-align:left}
.feature-table th{position:sticky;top:0;background:#1a1e2a;color:#8890a8}

</style>
</head>
<body>
<header>
  <h1>Point-Prompted Target Detection</h1>
</header>
<div class="toolbar">
  <input type="file" id="file-input" accept="image/*,.tif,.tiff,.mat,.npy">
  <button class="btn btn-primary" onclick="document.getElementById('file-input').click()">Upload</button>
  <button class="btn btn-danger"   id="btn-clear" onclick="clearPoints()" disabled>Clear</button>
  <button class="btn btn-outline"  id="btn-undo"  onclick="undoPoint()"   disabled>Undo</button>

  <div class="sep"></div>

  <!-- Model selector (visible) -->
  <label style="font-size:12px;color:#8890a8">Model:</label>
  <select id="sel-model" disabled
    style="background:#1e2230;color:#e0e0e0;border:1px solid #3a3f55;border-radius:5px;padding:4px 10px;font-size:13px;font-weight:600;min-width:170px">
  </select>

  <!-- Threshold slider for score-map models -->
  <div id="samitizer-thr-ctrl" style="display:none;align-items:center;gap:6px">
    <div class="sep"></div>
    <label style="font-size:12px;color:#8890a8;white-space:nowrap">Threshold:</label>
    <input type="range" id="sld-sam-thr" min="0" max="1" value="0.5" step="0.01"
      style="width:100px;accent-color:#CC3311">
    <span id="sld-sam-thr-val" style="font-size:12px;color:#e0e0e0;min-width:56px">0.5000</span>
    <span id="sld-sam-thr-used" style="font-size:10px;color:#888;min-width:60px"></span>
  </div>

  <!-- RGB display band selector -->
  <div id="band-ctrl" style="display:none;align-items:center;gap:6px">
    <div class="sep"></div>
    <label>R</label><input type="number" id="band-r" min="0" value="1">
    <label>G</label><input type="number" id="band-g" min="0" value="2">
    <label>B</label><input type="number" id="band-b" min="0" value="3">
    <button class="btn btn-outline" onclick="applyDisplaySettings()" style="padding:4px 10px">Apply Bands</button>
  </div>

  <div id="display-norm-ctrl" style="display:none;align-items:center;gap:6px">
    <div class="sep"></div>
    <label>Low %</label><input type="number" id="pct-low" min="0" max="99.9" step="0.1" value="2.0">
    <label>High %</label><input type="number" id="pct-high" min="0.1" max="100" step="0.1" value="98.0">
    <button class="btn btn-outline" onclick="applyDisplaySettings()" style="padding:4px 10px">Apply View</button>
  </div>

  <div class="sep"></div>

  <!-- GT mask upload -->
  <input type="file" id="gt-input" accept=".npy,.png,.jpg,.bmp,.tif" style="display:none">
  <button class="btn btn-outline" id="btn-gt" onclick="document.getElementById('gt-input').click()" disabled>Load GT Mask</button>
  <select id="gt-label-select" style="display:none;width:180px;background:#1e2230;color:#e0e0e0;border:1px solid #3a3f55;border-radius:5px;padding:4px 8px;font-size:12px" onchange="onGtLabelChange(this.value)"></select>
  <span id="gt-info" style="font-size:10px;color:#555b72"></span>

  <div class="sep"></div>

  <!-- Save Results button -->
  <button class="btn btn-outline" id="btn-save" onclick="saveResults()" disabled title="Download input, score map, and threshold map as a ZIP">Save Results</button>
  <!-- Compare All Models (secondary) -->
  <button class="btn btn-outline" id="btn-run-all" onclick="runAllModels()" disabled style="color:#6366f1;border-color:#6366f1;font-size:11px">Compare All</button>

  <div class="spinner" id="spinner"></div>

  <div class="legend">
    <div class="legend-item"><div class="ldot ldot-pos"></div>Left = Pos</div>
    <div class="legend-item"><div class="ldot ldot-neg"></div>Right = Neg</div>
  </div>
</div>

<!-- hidden classical toolbar for compat -->
<div id="classical-toolbar" style="display:none">
  <input type="checkbox" id="chk-classical-mode">
  <select id="sel-classical-algo"><option value="ACE">ACE</option></select>
  <span id="classical-target-info"></span>
  <button id="btn-classical-clear"></button>
  <button id="btn-classical-run"></button>
</div>
<div id="det-band-ctrl" style="display:none"><input id="det-bands"><span id="det-band-info"></span></div>

<!-- hidden legacy canvases for compat -->
<div style="display:none">
  <canvas id="chart-raw"></canvas><canvas id="chart-projected"></canvas>
  <canvas id="chart-svd"></canvas><canvas id="chart-score-hist"></canvas>
  <canvas id="chart-roc"></canvas><canvas id="chart-stability"></canvas>
  <canvas id="chart-fpmp"></canvas>
  <div id="oblique-metrics"></div><div id="svd-metrics"></div>
  <span id="score-dist-info"></span><span id="eval-auc"></span><span id="eval-ap"></span>
  <span id="eval-gt-info"></span><div id="charts-row"></div>
  <table id="table-pd-pfa"><tbody id="tbody-pd-pfa"></tbody></table>
</div>

<!-- Results: comparison table + PR curve -->
<div id="eval-row" style="display:none;padding:0 24px 16px 24px">
  <div style="font-size:12px;font-weight:700;color:#10b981;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;padding-top:8px;border-top:1px solid #2d3044">
    Comparison Results
    <span id="eval-gt-info-main" style="font-weight:400;color:#6b7394;font-size:10px;margin-left:8px"></span>
  </div>
  <div style="display:grid;grid-template-columns:auto 1fr;gap:20px;align-items:start">
    <!-- Results table -->
    <div style="background:#181b24;border:1px solid #2d3044;border-radius:8px;padding:12px 16px;min-width:340px">
      <div style="font-size:11px;font-weight:700;color:#8890a8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">Model Results</div>
      <table id="tbl-results" style="width:100%;border-collapse:collapse;font-size:12px;color:#c4cbde">
        <thead>
          <tr style="border-bottom:1px solid #3a3f55">
            <th style="padding:5px 8px;text-align:left;color:#8890a8">Model</th>
            <th style="padding:5px 8px;text-align:right;color:#8890a8">IoU</th>
            <th style="padding:5px 8px;text-align:right;color:#8890a8">Prec</th>
            <th style="padding:5px 8px;text-align:right;color:#8890a8">Recall</th>
            <th style="padding:5px 8px;text-align:right;color:#8890a8">ms</th>
          </tr>
        </thead>
        <tbody id="tbody-results"></tbody>
      </table>
    </div>
    <!-- PR Curve (multi-model) -->
    <div style="background:#181b24;border:1px solid #2d3044;border-radius:8px;padding:12px 16px">
      <div style="font-size:11px;font-weight:700;color:#377eb8;text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px">PR Curve</div>
      <div style="position:relative;height:280px">
        <canvas id="chart-pr"></canvas>
      </div>
    </div>
  </div>
</div>

<!-- hidden compat elements -->
<div style="display:none">
  <img id="img-feat"><span id="meta-feat"></span><span id="title-feat"></span>
  <span id="dot-feat"></span><div id="ph-feat"></div><div id="panel-feat"></div>
  <img id="img-score"><span id="meta-score"></span><span id="title-score"></span>
  <span id="dot-score"></span><div id="ph-score"></div>
  <img id="img-overlay"><span id="meta-overlay"></span><span id="title-overlay"></span>
  <span id="dot-overlay"></span><div id="ph-overlay"></div>
  <span id="i-model">-</span>
</div>

<!-- New-method analysis charts (shown only for new_method_amf/new_method_ace) -->
<div id="method-analysis-top" class="analysis-wrap" style="display:none">
  <div class="analysis-grid">
    <div class="analysis-card" id="analysis-card-spectrum">
      <div class="analysis-title">Expanded Feature Spectra</div>
      <div class="analysis-canvas"><canvas id="chart-new-spectrum"></canvas></div>
      <div id="new-spectrum-note" class="analysis-note"></div>
    </div>
    <div class="analysis-card" id="analysis-card-rinv">
      <div class="analysis-title">Inverse Covariance (R^-1)</div>
      <div class="analysis-canvas"><canvas id="canvas-rinv"></canvas></div>
      <div id="rinv-note" class="analysis-note"></div>
    </div>
    <div class="analysis-card" id="analysis-card-projector-loss" style="display:none">
      <div class="analysis-title">Projector Training Loss</div>
      <div class="analysis-canvas"><canvas id="chart-projector-loss"></canvas></div>
      <div id="projector-loss-note" class="analysis-note"></div>
    </div>
  </div>
</div>

<div class="grid" id="main-grid" style="grid-template-columns:repeat(3,1fr)">
  <!-- Input panel -->
  <div class="panel">
    <div class="ptitle"><div class="dot" style="background:#3b82f6"></div>Input Image + Points
      <span class="meta" style="display:flex;gap:4px;align-items:center">
        <button class="btn btn-outline" onclick="zoomInShared()" style="padding:2px 6px;font-size:11px">+</button>
        <button class="btn btn-outline" onclick="zoomOutShared()" style="padding:2px 6px;font-size:11px">−</button>
        <button class="btn btn-outline" onclick="resetZoom()" style="padding:2px 6px;font-size:11px">Reset</button>
      </span>
    </div>
    <div class="pbody" id="pb-canvas" style="overflow:hidden;position:relative">
      <div class="zoom-wrap" id="zoom-wrap">
        <canvas id="canvas" style="display:none;transform-origin:0 0"></canvas>
      </div>
      <div class="zoom-info" id="zoom-info">100%</div>
      <div class="pixel-info" id="pixel-info" style="display:none"></div>
      <div class="ph" id="ph-input">Upload an image to begin</div>
    </div>
  </div>

  <!-- Score Map panel -->
  <div class="panel">
    <div class="ptitle">
      <div class="dot" id="dot-result" style="background:#888"></div>
      <span id="label-result">Score Map</span>
      <span class="meta" id="meta-result-score" style="font-size:11px;color:#8890a8"></span>
    </div>
    <div class="pbody result-pbody" id="pb-result-score">
      <div class="zoom-wrap result-zoom-wrap" id="zoom-wrap-score">
        <img id="img-result-score" class="sync-zoom-media" style="display:none;cursor:crosshair">
      </div>
      <div class="ph" id="ph-result-score">Add a prompt point to run</div>
    </div>
  </div>

  <!-- Seg Map panel -->
  <div class="panel">
    <div class="ptitle">
      <div class="dot" id="dot-result2" style="background:#888"></div>
      <span id="label-result2">Seg Map</span>
      <span class="meta" id="meta-result-seg" style="font-size:11px;color:#8890a8"></span>
    </div>
    <div class="pbody result-pbody" id="pb-result-seg">
      <div class="zoom-wrap result-zoom-wrap" id="zoom-wrap-seg">
        <img id="img-result-seg" class="sync-zoom-media" style="display:none;cursor:crosshair">
      </div>
      <div class="ph" id="ph-result-seg">Add a prompt point to run</div>
    </div>
  </div>
</div>

<!-- New-method feature details (shown under result panels as a toggle) -->
<div id="method-analysis-details" class="feature-details" style="display:none">
  <details id="feature-details-toggle" open>
    <summary>Added Bands / Expanded Features</summary>
    <div id="feature-summary" class="feature-summary"></div>
    <div id="feature-table-wrap" class="feature-table-wrap"></div>
  </details>
</div>

<!-- Metrics bar (shown when GT loaded and results available) -->
<div id="metrics-bar" style="display:none;background:#12151e;border:1px solid #2a2f45;border-radius:8px;margin:8px 16px;padding:10px 20px;display:flex;align-items:center;gap:24px;flex-wrap:wrap">
  <span style="font-size:12px;color:#8890a8;font-weight:600" id="metrics-model-label">—</span>
  <span style="font-size:12px">IoU: <strong id="m-iou" style="color:#22c55e">—</strong></span>
  <span style="font-size:12px">Precision: <strong id="m-prec" style="color:#3b82f6">—</strong></span>
  <span style="font-size:12px">Recall: <strong id="m-rec" style="color:#f59e0b">—</strong></span>
  <span style="font-size:12px">AP: <strong id="m-ap" style="color:#a78bfa">—</strong></span>
  <span id="m-thr-row" style="display:none;font-size:12px">
    Threshold: <strong id="m-thr" style="color:#CC3311">—</strong>
  </span>
  <span style="font-size:12px;color:#555b72" id="m-ms"></span>
  <!-- GT legend -->
  <span style="font-size:11px;color:#8890a8;margin-left:auto">
    <span style="color:#ff00ff">■</span> TP &nbsp;
    <span style="color:#c80000">■</span> FP &nbsp;
    <span style="color:#0000c8">■</span> FN
  </span>
</div>

<!-- PR Curve (shown when GT loaded) -->
<div id="pr-panel" style="display:none;margin:4px 16px 8px">
  <canvas id="chart-pr-single" height="180"></canvas>
</div>

<div class="info-bar">
  <span>Points: <span class="v" id="i-pts">0 pos / 0 neg</span></span>
  <span>Image: <span class="v" id="i-size">-</span></span>
  <span>GT: <span class="v" id="i-det">-</span> target px</span>
</div>

<!-- model panels are generated dynamically from /upload -> available_models -->
<div id="all-models-section" style="display:none">
  <div style="padding:8px 16px;font-size:13px;font-weight:600;color:#8890a8">All Models Comparison</div>
  <div class="grid" id="all-grid" style="grid-template-columns:repeat(4,1fr)"></div>
  <!-- All-models metrics table -->
  <div style="overflow-x:auto;margin:4px 16px 12px">
    <table id="tbl-all" style="width:100%;border-collapse:collapse;font-size:12px;color:#e0e0e0">
      <thead><tr style="color:#8890a8;border-bottom:1px solid #2a2f45">
        <th style="text-align:left;padding:4px 8px">Model</th>
        <th style="padding:4px 8px">IoU</th>
        <th style="padding:4px 8px">Prec</th>
        <th style="padding:4px 8px">Recall</th>
        <th style="padding:4px 8px">ms</th>
      </tr></thead>
      <tbody id="tbl-all-body"></tbody>
    </table>
  </div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const spinner = document.getElementById('spinner');
const selModel = document.getElementById('sel-model');
const samThrCtrl = document.getElementById('samitizer-thr-ctrl');
const samThrSlider = document.getElementById('sld-sam-thr');
const samThrVal = document.getElementById('sld-sam-thr-val');
const samThrUsed = document.getElementById('sld-sam-thr-used');
let imgObj = null;
let excludeOverlayImg = null;
let origW=0, origH=0, dispW=0, dispH=0, numCh=0;
let posPoints=[], negPoints=[], history=[];
let availModels = [];
let curModel = '';
let _gtLoaded = false;
let loadedName = '';
let _samThrManual = false;  // true once user moves the threshold slider manually
let MODEL_PANEL_MAP = {};
let MODEL_COLORS = {};
let _singleResultDirty = true;
let _lastSingleResultModel = '';
let _thresholdPreviewTimer = null;
let _thresholdPreviewInFlight = false;
let _thresholdPreviewPending = false;

function markSingleResultDirty(){
  _singleResultDirty = true;
  _lastSingleResultModel = '';
}

function markSingleResultReady(modelName){
  _singleResultDirty = false;
  _lastSingleResultModel = String(modelName || '');
}

function canPreviewThreshold(){
  const imgScore = document.getElementById('img-result-score');
  return !!imgScore
    && imgScore.style.display !== 'none'
    && !_singleResultDirty
    && _lastSingleResultModel === selModel.value
    && isThresholdTunable(selModel.value);
}

function getSyncedResultWraps(){
  return ['zoom-wrap-score', 'zoom-wrap-seg']
    .map(id => document.getElementById(id))
    .filter(Boolean);
}

function getSyncedMediaElements(){
  return [canvas, document.getElementById('img-result-score'), document.getElementById('img-result-seg')]
    .filter(Boolean);
}

function setSharedCursor(cursor){
  canvas.style.cursor = cursor;
  getSyncedResultWraps().forEach(el => { el.style.cursor = cursor; });
  ['img-result-score', 'img-result-seg'].forEach(id => {
    const el = document.getElementById(id);
    if(el) el.style.cursor = cursor;
  });
}

function syncResultViewports(){
  const inputWrap = document.getElementById('zoom-wrap');
  if(!inputWrap) return;
  const ww = Math.max(inputWrap.clientWidth, 1);
  const wh = Math.max(inputWrap.clientHeight, 1);
  getSyncedResultWraps().forEach(wrap => {
    wrap.style.width = ww + 'px';
    wrap.style.height = wh + 'px';
  });
}

function syncZoomMediaGeometry(){
  ['img-result-score', 'img-result-seg'].forEach(id => {
    const el = document.getElementById(id);
    if(!el) return;
    el.style.width = dispW + 'px';
    el.style.height = dispH + 'px';
  });
}

function _applySyncedViewportLayout(){
  if(!Number.isFinite(zoomLevel) || zoomLevel <= 0){
    resetZoom();
    return;
  }
  syncResultViewports();
  syncZoomMediaGeometry();
  clampPan();
  applyTransform();
}

function showSyncedResultImage(imgId, wrapId, phId, b64){
  const imgEl = document.getElementById(imgId);
  const wrapEl = document.getElementById(wrapId);
  const phEl = document.getElementById(phId);
  if(!imgEl || !wrapEl) return;
  if(!b64){
    wrapEl.style.display = 'none';
    imgEl.style.display = 'none';
    if(phEl){
      phEl.style.display = '';
      phEl.textContent = 'No image data';
    }
    return;
  }

  wrapEl.style.display = 'block';
  imgEl.style.display = 'block';
  if(phEl) phEl.style.display = 'none';
  imgEl.onload = ()=> {
    _applySyncedViewportLayout();
  };
  imgEl.onerror = ()=> {
    wrapEl.style.display = 'none';
    imgEl.style.display = 'none';
    if(phEl){
      phEl.style.display = '';
      phEl.textContent = 'Image render failed';
    }
  };
  imgEl.src = 'data:image/png;base64,' + b64;
  if(imgEl.complete && imgEl.naturalWidth > 0){
    _applySyncedViewportLayout();
  }
}

function viewportToOrig(e, wrapEl){
  const wr = wrapEl.getBoundingClientRect();
  const mx = e.clientX - wr.left;
  const my = e.clientY - wr.top;
  const cx = (mx - panX) / zoomLevel;
  const cy = (my - panY) / zoomLevel;
  return {
    x: Math.round(cx / dispW * origW),
    y: Math.round(cy / dispH * origH),
  };
}

function rememberViewportPointer(e, viewportEl){
  if(!viewportEl) return;
  const rect = viewportEl.getBoundingClientRect();
  lastViewportPointer = {
    wrapId: viewportEl.id || 'zoom-wrap',
    x: e.clientX - rect.left,
    y: e.clientY - rect.top,
  };
}

function zoomAroundViewportPoint(viewportEl, mx, my, nextZoom){
  if(!viewportEl || dispW <= 0 || dispH <= 0) return;
  const rect = viewportEl.getBoundingClientRect();
  if(rect.width <= 0 || rect.height <= 0) return;
  const px = Math.min(Math.max(mx, 0), rect.width);
  const py = Math.min(Math.max(my, 0), rect.height);
  const cxBefore = (px - panX) / zoomLevel;
  const cyBefore = (py - panY) / zoomLevel;
  zoomLevel = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, nextZoom));
  panX = px - cxBefore * zoomLevel;
  panY = py - cyBefore * zoomLevel;
  clampPan();
  applyTransform();
}

function getSharedZoomAnchor(){
  const viewportEl = document.getElementById(lastViewportPointer.wrapId || 'zoom-wrap') || document.getElementById('zoom-wrap');
  if(!viewportEl) return null;
  const rect = viewportEl.getBoundingClientRect();
  if(rect.width <= 0 || rect.height <= 0) return null;
  const mx = Number.isFinite(lastViewportPointer.x) ? lastViewportPointer.x : rect.width / 2;
  const my = Number.isFinite(lastViewportPointer.y) ? lastViewportPointer.y : rect.height / 2;
  return {
    viewportEl,
    mx: Math.min(Math.max(mx, 0), rect.width),
    my: Math.min(Math.max(my, 0), rect.height),
  };
}

function zoomInShared(){
  const anchor = getSharedZoomAnchor();
  if(!anchor) return;
  zoomAroundViewportPoint(anchor.viewportEl, anchor.mx, anchor.my, zoomLevel * ZOOM_STEP);
}

function zoomOutShared(){
  const anchor = getSharedZoomAnchor();
  if(!anchor) return;
  zoomAroundViewportPoint(anchor.viewportEl, anchor.mx, anchor.my, zoomLevel / ZOOM_STEP);
}

function applySharedWheelZoom(e, viewportEl){
  e.preventDefault();
  if(!viewportEl || dispW <= 0 || dispH <= 0) return;
  rememberViewportPointer(e, viewportEl);
  const nextZoom = e.deltaY < 0 ? (zoomLevel * ZOOM_STEP) : (zoomLevel / ZOOM_STEP);
  zoomAroundViewportPoint(viewportEl, lastViewportPointer.x, lastViewportPointer.y, nextZoom);
}

function startSharedPan(e){
  e.preventDefault();
  isPanning = true;
  panStartX = e.clientX;
  panStartY = e.clientY;
  panStartPX = panX;
  panStartPY = panY;
  setSharedCursor('grabbing');
}

function _slugModelId(name){
  const s = String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return s || 'model';
}

function buildAllModelPanels(models){
  const grid = document.getElementById('all-grid');
  if(!grid) return;

  grid.innerHTML = '';
  MODEL_PANEL_MAP = {};
  MODEL_COLORS = {};

  models.forEach((m, idx) => {
    const slug = _slugModelId(m.id) + '-' + idx;
    const ids = {
      img: `img-${slug}`,
      ph: `ph-${slug}`,
      meta: `meta-${slug}`,
    };
    MODEL_PANEL_MAP[m.id] = ids;
    MODEL_COLORS[m.id] = m.color || '#888';

    const panel = document.createElement('div');
    panel.className = 'panel';

    const title = document.createElement('div');
    title.className = 'ptitle';

    const dot = document.createElement('div');
    dot.className = 'dot';
    dot.style.background = MODEL_COLORS[m.id];
    title.appendChild(dot);

    title.appendChild(document.createTextNode(m.label || m.id));

    const meta = document.createElement('span');
    meta.className = 'meta';
    meta.id = ids.meta;
    title.appendChild(meta);

    const body = document.createElement('div');
    body.className = 'pbody';

    const img = document.createElement('img');
    img.id = ids.img;
    img.style.display = 'none';
    img.style.cursor = 'crosshair';
    body.appendChild(img);

    const ph = document.createElement('div');
    ph.className = 'ph';
    ph.id = ids.ph;
    ph.textContent = '—';
    body.appendChild(ph);

    panel.appendChild(title);
    panel.appendChild(body);
    grid.appendChild(panel);
  });
}

function updateSaveBtn(){
  const btn = document.getElementById('btn-save');
  if(btn) btn.disabled = !(!_singleResultDirty && !!_lastSingleResultModel);
}

// ---- Zoom / Pan state ----
let zoomLevel = 1.0;
let panX = 0, panY = 0;
let isPanning = false, panStartX = 0, panStartY = 0, panStartPX = 0, panStartPY = 0;
let lastViewportPointer = { wrapId:'zoom-wrap', x:null, y:null };
const ZOOM_MIN = 0.05, ZOOM_MAX = 20, ZOOM_STEP = 1.15;

function updateCanvasViewport(){
  const wrap = document.getElementById('zoom-wrap');
  const body = document.getElementById('pb-canvas');
  if(!wrap || !body || dispW <= 0 || dispH <= 0) return;
  const bodyWidth = Math.max(body.clientWidth - 12, 1);
  const maxViewportH = Math.max(240, Math.floor(window.innerHeight * 0.65));
  const fitScale = Math.min(bodyWidth / dispW, maxViewportH / dispH, 1.0);
  const viewportW = Math.max(1, Math.round(dispW * fitScale));
  const viewportH = Math.max(1, Math.round(dispH * fitScale));
  wrap.style.width = viewportW + 'px';
  wrap.style.height = viewportH + 'px';
  body.style.minHeight = viewportH + 'px';
  syncResultViewports();
  syncZoomMediaGeometry();
}

function getFitZoom(){
  const wrap = document.getElementById('zoom-wrap');
  if(!wrap || dispW <= 0 || dispH <= 0) return 1.0;
  const ww = Math.max(wrap.clientWidth, 1);
  const wh = Math.max(wrap.clientHeight, 1);
  return Math.min(ww / dispW, wh / dispH, 1.0);
}

function applyTransform(){
  const transform = `translate(${panX}px,${panY}px) scale(${zoomLevel})`;
  getSyncedMediaElements().forEach(el => { el.style.transform = transform; });
  document.getElementById('zoom-info').textContent = Math.round(zoomLevel*100)+'%';
}

function clampPan(){
  const wrap = document.getElementById('zoom-wrap');
  if(!wrap) return;
  const ww = wrap.clientWidth, wh = wrap.clientHeight;
  const cw = dispW * zoomLevel, ch = dispH * zoomLevel;
  if(cw <= ww) panX = (ww - cw)/2;
  else panX = Math.min(0, Math.max(ww - cw, panX));
  if(ch <= wh) panY = (wh - ch)/2;
  else panY = Math.min(0, Math.max(wh - ch, panY));
}

function resetZoom(){
  zoomLevel = getFitZoom();
  panX = 0;
  panY = 0;
  clampPan(); applyTransform();
}

canvas.addEventListener('wheel', (e)=> applySharedWheelZoom(e, document.getElementById('zoom-wrap')), {passive:false});
canvas.addEventListener('mousemove', (e)=> rememberViewportPointer(e, document.getElementById('zoom-wrap')));

canvas.addEventListener('mousedown', (e)=>{
  if(e.button === 1){
    startSharedPan(e);
    return;
  }
});
window.addEventListener('mousemove', (e)=>{
  if(isPanning){
    panX = panStartPX + (e.clientX - panStartX);
    panY = panStartPY + (e.clientY - panStartY);
    clampPan(); applyTransform();
  }
  // Show pixel coordinate under cursor
  if(imgObj && origW > 0){
    const wrap = document.getElementById('zoom-wrap');
    const wr = wrap.getBoundingClientRect();
    const mx = e.clientX - wr.left;
    const my = e.clientY - wr.top;
    if(mx >= 0 && my >= 0 && mx < wr.width && my < wr.height){
      const cx = (mx - panX) / zoomLevel;
      const cy = (my - panY) / zoomLevel;
      const px = Math.floor(cx / dispW * origW);
      const py = Math.floor(cy / dispH * origH);
      const info = document.getElementById('pixel-info');
      if(px >= 0 && px < origW && py >= 0 && py < origH){
        info.style.display = 'block';
        info.textContent = `(${px}, ${py})`;
      } else {
        info.style.display = 'none';
      }
    } else {
      document.getElementById('pixel-info').style.display = 'none';
    }
  }
});
window.addEventListener('mouseup', (e)=>{
  if(isPanning){
    isPanning = false;
    setSharedCursor(classicalMode ? 'cell' : 'crosshair');
  }
});
window.addEventListener('resize', ()=>{
  if(!imgObj) return;
  updateCanvasViewport();
  clampPan();
  applyTransform();
});

// Classical detector state
let classicalTargetPixels = [];
let classicalMode = false;
const chkClassical = document.getElementById('chk-classical-mode');
chkClassical.addEventListener('change', ()=>{
  classicalMode = chkClassical.checked;
  document.getElementById('classical-target-info').textContent =
    classicalMode ? 'Click image to select target pixel(s)' : '';
  setSharedCursor(classicalMode ? 'cell' : 'crosshair');
});

// ---- Model selector ----
function isSAMitizer(name){ return name && name.startsWith('SAMitizer'); }
function isThresholdTunable(name){
  return isSAMitizer(name)
    || name === 'new_method_amf'
    || name === 'new_method_ace'
    || name === 'new_method_mlp'
    || name === 'new_method_mlp_ace'
    || name === 'new_method_bilinear'
    || name === 'new_method_bilinear_ace'
    || name === 'ACE'
    || name === 'AMF'
    || name === 'OSP+AMF';
}

function formatThresholdValue(v){
  const n = Number(v);
  if(!Number.isFinite(n)) return '-';
  const a = Math.abs(n);
  if(a >= 1000 || (a > 0 && a < 0.001)) return n.toExponential(2);
  return n.toFixed(4);
}

function setThresholdSliderRange(minV, maxV, valueV){
  let lo = Number(minV);
  let hi = Number(maxV);
  if(!Number.isFinite(lo) || !Number.isFinite(hi)){
    lo = 0.0;
    hi = 1.0;
  }
  if(hi <= lo){
    const pad = Math.max(Math.abs(lo) * 0.01, 1e-6);
    lo -= pad;
    hi += pad;
  }
  const step = Math.max((hi - lo) / 1000.0, 1e-6);
  const rawValue = Number(valueV);
  const clamped = Number.isFinite(rawValue) ? Math.min(hi, Math.max(lo, rawValue)) : lo;
  samThrSlider.min = String(lo);
  samThrSlider.max = String(hi);
  samThrSlider.step = String(step);
  samThrSlider.value = String(clamped);
  samThrVal.textContent = formatThresholdValue(clamped);
}

function resetThresholdOverride(){
  _samThrManual = false;
  if(_thresholdPreviewTimer){
    clearTimeout(_thresholdPreviewTimer);
    _thresholdPreviewTimer = null;
  }
  _thresholdPreviewPending = false;
  samThrUsed.textContent = '';
  setThresholdSliderRange(0.0, 1.0, 0.5);
}

function updateThrSliderVisibility(){
  const show = isThresholdTunable(selModel.value);
  samThrCtrl.style.display = show ? 'flex' : 'none';
  if(!show){
    resetThresholdOverride();
    const mThrRow = document.getElementById('m-thr-row');
    if(mThrRow) mThrRow.style.display = 'none';
  }
}

async function previewThresholdFromCache(){
  if(!canPreviewThreshold()) return;
  if(_thresholdPreviewInFlight){
    _thresholdPreviewPending = true;
    return;
  }
  _thresholdPreviewInFlight = true;
  try {
    const resp = await fetch('/preview_threshold', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({
        model: selModel.value,
        threshold: parseFloat(samThrSlider.value),
      }),
    });
    const d = await resp.json();
    if(!resp.ok || d.error){
      console.error('[previewThresholdFromCache]', d.error || resp.status);
      markSingleResultDirty();
      scheduleAutoRun();
      return;
    }

    showSyncedResultImage('img-result-seg', 'zoom-wrap-seg', 'ph-result-seg', d.seg);

    samThrVal.textContent = formatThresholdValue(d.used_threshold);
    samThrUsed.textContent = '(manual)';

    const mThrRow = document.getElementById('m-thr-row');
    if(d.used_threshold != null){
      document.getElementById('m-thr').textContent = formatThresholdValue(d.used_threshold);
      if(mThrRow) mThrRow.style.display = '';
    }

    if(d.has_gt && d.iou !== undefined){
      document.getElementById('metrics-bar').style.display = 'flex';
      document.getElementById('m-iou').textContent  = (d.iou*100).toFixed(1)  + '%';
      document.getElementById('m-prec').textContent = (d.prec*100).toFixed(1) + '%';
      document.getElementById('m-rec').textContent  = (d.rec*100).toFixed(1)  + '%';
    }
  } finally {
    _thresholdPreviewInFlight = false;
    if(_thresholdPreviewPending){
      _thresholdPreviewPending = false;
      scheduleThresholdPreview();
    }
  }
}

function scheduleThresholdPreview(){
  if(!canPreviewThreshold()) return false;
  if(_thresholdPreviewTimer) clearTimeout(_thresholdPreviewTimer);
  _thresholdPreviewTimer = setTimeout(()=>{
    _thresholdPreviewTimer = null;
    previewThresholdFromCache();
  }, 40);
  return true;
}

selModel.addEventListener('change', ()=>{
  curModel = selModel.value;
  markSingleResultDirty();
  resetThresholdOverride();
  updateThrSliderVisibility();
  clearResults();
  if(posPoints.length > 0) runOneModel();
});

// Threshold slider
samThrSlider.addEventListener('input', ()=>{
  _samThrManual = true;
  samThrVal.textContent = formatThresholdValue(samThrSlider.value);
  if(!scheduleThresholdPreview()){
    samThrUsed.textContent = '(manual)';
  }
});
samThrSlider.addEventListener('change', ()=>{
  _samThrManual = true;
  if(!scheduleThresholdPreview()) scheduleAutoRun();
});


// ---- Upload ----
document.getElementById('file-input').addEventListener('change', async (e)=>{
  const file = e.target.files[0];
  if(!file) return;
  const form = new FormData();
  form.append('image', file);
  spinner.style.display='block';
  const resp = await fetch('/upload',{method:'POST',body:form});
  const d = await resp.json();
  spinner.style.display='none';
  if(d.error){alert(d.error);return}

  origW=d.orig_w; origH=d.orig_h; dispW=d.disp_w; dispH=d.disp_h;
  numCh = d.channels;
  availModels = d.models || [];
  loadedName = d.loaded_name || '';
  buildAllModelPanels(availModels);

  // Populate model selector
  selModel.innerHTML = '';
  availModels.forEach(m=>{
    const opt = document.createElement('option');
    opt.value = m.id;
    opt.textContent = m.label;
    selModel.appendChild(opt);
  });
  selModel.disabled = false;
  if(availModels.length > 0){
    selModel.value = availModels[0].id;
    selModel.dispatchEvent(new Event('change'));
  }

  // Display controls
  const bandCtrl = document.getElementById('band-ctrl');
  const detBandCtrl = document.getElementById('det-band-ctrl');
  const displayNormCtrl = document.getElementById('display-norm-ctrl');
  displayNormCtrl.style.display = 'flex';
  if(numCh > 3){
    bandCtrl.style.display='flex';
    detBandCtrl.style.display='flex';
    document.getElementById('band-r').max = numCh-1;
    document.getElementById('band-g').max = numCh-1;
    document.getElementById('band-b').max = numCh-1;
    const rb = (Array.isArray(d.rgb_bands) && d.rgb_bands.length === 3)
      ? d.rgb_bands : [0, Math.min(1, numCh-1), Math.min(2, numCh-1)];
    document.getElementById('band-r').value = rb[0];
    document.getElementById('band-g').value = rb[1];
    document.getElementById('band-b').value = rb[2];
    document.getElementById('det-bands').value = '';
    document.getElementById('det-bands').placeholder = '0-' + (numCh-1) + ' (all)';
    detBandIndices = null;
    document.getElementById('det-band-info').textContent = 'all ' + numCh + ' bands';
  } else {
    bandCtrl.style.display='none';
    detBandCtrl.style.display='none';
  }
  const pct = Array.isArray(d.display_percentiles) ? d.display_percentiles : [2.0, 98.0];
  document.getElementById('pct-low').value = pct[0];
  document.getElementById('pct-high').value = pct[1];

  setCanvasImage(d.image);
  posPoints=[]; negPoints=[]; history=[];
  markSingleResultDirty();
  resetThresholdOverride();
  clearResults();
  document.getElementById('i-size').textContent = origW+'\u00d7'+origH+'\u00d7'+numCh+'ch';
  updateInfo();
});

function setCanvasImage(b64, {keepView=false} = {}){
  imgObj = new Image();
  imgObj.onload = ()=>{
    canvas.width = dispW;
    canvas.height = dispH;
    canvas.style.width = dispW+'px';
    canvas.style.height = dispH+'px';
    canvas.style.display='block';
    updateCanvasViewport();
    document.getElementById('ph-input').style.display='none';
    if(!keepView) resetZoom();
    else {
      syncResultViewports();
      syncZoomMediaGeometry();
      clampPan();
      applyTransform();
    }
    setSharedCursor(classicalMode ? 'cell' : 'crosshair');
    redraw();
  };
  imgObj.src = 'data:image/png;base64,' + b64;
}

// ---- Detection band selector ----
let detBandIndices = null; // null = all bands

function parseDetBands(text) {
  text = text.trim();
  if (!text) return null;
  const indices = new Set();
  text.split(',').forEach(part => {
    part = part.trim();
    if (!part) return;
    const m = part.match(/^(\d+)\s*-\s*(\d+)$/);
    if (m) {
      const lo = parseInt(m[1]), hi = parseInt(m[2]);
      for (let i = lo; i <= hi; i++) indices.add(i);
    } else {
      const v = parseInt(part);
      if (!isNaN(v)) indices.add(v);
    }
  });
  const arr = [...indices].filter(i => i >= 0 && i < numCh).sort((a,b) => a-b);
  return arr.length > 0 ? arr : null;
}

function applyDetBands() {
  const text = document.getElementById('det-bands').value;
  detBandIndices = parseDetBands(text);
  const info = document.getElementById('det-band-info');
  if (detBandIndices) {
    info.textContent = detBandIndices.length + '/' + numCh + ' bands';
    info.style.color = '#3b82f6';
  } else {
    info.textContent = 'all ' + numCh + ' bands';
    info.style.color = '#555b72';
  }
  if (posPoints.length > 0) runDetection();
}

// ---- RGB band selector ----
async function applyDisplaySettings(){
  const r = parseInt(document.getElementById('band-r').value)||0;
  const g = parseInt(document.getElementById('band-g').value)||0;
  const b = parseInt(document.getElementById('band-b').value)||0;
  const low = parseFloat(document.getElementById('pct-low').value);
  const high = parseFloat(document.getElementById('pct-high').value);
  spinner.style.display='block';
  const resp = await fetch('/set_bands',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      bands:[r,g,b],
      percentile_low: low,
      percentile_high: high,
    })});
  const d = await resp.json();
  spinner.style.display='none';
  if(d.error){alert(d.error);return}
  dispW=d.disp_w; dispH=d.disp_h;
  if(d.percentile_low != null) document.getElementById('pct-low').value = d.percentile_low;
  if(d.percentile_high != null) document.getElementById('pct-high').value = d.percentile_high;
  setCanvasImage(d.image, {keepView:true});
  if(d.seg){
    showSyncedResultImage('img-result-seg', 'zoom-wrap-seg', 'ph-result-seg', d.seg);
  }
}

function applyBands(){
  applyDisplaySettings();
}

// ---- Auto-run debounce ----
// Whenever a prompt point is added, auto-trigger runOneModel after a short delay.
let _autoRunTimer = null;
function scheduleAutoRun(){
  if(_autoRunTimer) clearTimeout(_autoRunTimer);
  _autoRunTimer = setTimeout(()=>{
    if(posPoints.length > 0) runOneModel();
  }, 500);   // 500 ms debounce
}

// ---- Click handling ----
// Use mousedown instead of click/contextmenu to avoid duplicate events on Linux
function canvasToOrig(e){
  const wrap = document.getElementById('zoom-wrap');
  const wr = wrap.getBoundingClientRect();
  const mx = e.clientX - wr.left;
  const my = e.clientY - wr.top;
  const cx = (mx - panX) / zoomLevel;
  const cy = (my - panY) / zoomLevel;
  const ox = Math.round(cx / dispW * origW);
  const oy = Math.round(cy / dispH * origH);
  return {ox, oy};
}

// Generic helper: convert a click on any <img> panel to original image coords
function imgToOrig(e, el){
  const rect = el.getBoundingClientRect();
  const rx = (e.clientX - rect.left) / rect.width;
  const ry = (e.clientY - rect.top) / rect.height;
  return {
    ox: Math.round(rx * origW),
    oy: Math.round(ry * origH),
  };
}

// Generic helper: handle a prompt click (button 0 = pos, button 2 = neg)
function handlePromptClick(ox, oy, btn){
  if(!origW || !origH) return;
  if(btn === 0){
    posPoints.push({x:ox, y:oy});
    history.push('pos');
  } else if(btn === 2){
    negPoints.push({x:ox, y:oy});
    history.push('neg');
  } else { return; }
  markSingleResultDirty();
  resetThresholdOverride();
  redraw(); updateInfo();
  document.getElementById('btn-run-all').disabled = false;
  scheduleAutoRun();
}

canvas.addEventListener('mousedown', (e)=>{
  if(e.button === 1) return; // middle = pan, handled above
  e.preventDefault();
  const {ox, oy} = canvasToOrig(e);

  if(classicalMode && e.button === 0){
    classicalTargetPixels.push({x:ox, y:oy});
    document.getElementById('classical-target-info').textContent =
      classicalTargetPixels.length + ' target pixel(s) selected — (' + ox + ',' + oy + ')';
    document.getElementById('btn-classical-clear').disabled = false;
    document.getElementById('btn-classical-run').disabled = false;
    redraw();
    return;
  }

  handlePromptClick(ox, oy, e.button);
});
canvas.addEventListener('contextmenu', (e)=> e.preventDefault());

// Legacy result panels: allow clicking to add prompts.
['img-score','img-overlay','img-feat'].forEach(id => {
  const el = document.getElementById(id);
  if(!el) return;
  el.addEventListener('mousedown', (e) => {
    e.preventDefault();
    const {ox, oy} = imgToOrig(e, el);
    handlePromptClick(ox, oy, e.button);
  });
  el.addEventListener('contextmenu', (e)=> e.preventDefault());
});

// Dynamic all-model panels: delegate clicks from the grid container.
const allGridEl = document.getElementById('all-grid');
if(allGridEl){
  allGridEl.addEventListener('mousedown', (e) => {
    const img = e.target.closest('img');
    if(!img) return;
    if(img.style.display === 'none' || !img.src) return;
    e.preventDefault();
    const {ox, oy} = imgToOrig(e, img);
    handlePromptClick(ox, oy, e.button);
  });
  allGridEl.addEventListener('contextmenu', (e) => {
    if(e.target.closest('img')) e.preventDefault();
  });
}

// ---- Drawing ----
function redraw(){
  if(!imgObj) return;
  ctx.drawImage(imgObj, 0, 0, dispW, dispH);
  if(excludeOverlayImg) ctx.drawImage(excludeOverlayImg, 0, 0, dispW, dispH);
  posPoints.forEach(p=> drawPt(p.x/origW*dispW, p.y/origH*dispH, '#22c55e'));
  negPoints.forEach(p=> drawPt(p.x/origW*dispW, p.y/origH*dispH, '#ef4444'));
  classicalTargetPixels.forEach(p=> drawPt(p.x/origW*dispW, p.y/origH*dispH, '#c084fc'));
  document.getElementById('btn-clear').disabled=(posPoints.length+negPoints.length===0);
  document.getElementById('btn-undo').disabled=(posPoints.length+negPoints.length===0);
}
function drawPt(x,y,color){
  const s=8;
  ctx.strokeStyle='#fff'; ctx.lineWidth=3;
  ctx.beginPath(); ctx.moveTo(x-s,y); ctx.lineTo(x+s,y);
  ctx.moveTo(x,y-s); ctx.lineTo(x,y+s); ctx.stroke();
  ctx.strokeStyle=color; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(x-s,y); ctx.lineTo(x+s,y);
  ctx.moveTo(x,y-s); ctx.lineTo(x,y+s); ctx.stroke();
  ctx.beginPath(); ctx.arc(x,y,3,0,Math.PI*2);
  ctx.fillStyle=color; ctx.fill();
  ctx.strokeStyle='#fff'; ctx.lineWidth=1; ctx.stroke();
}

// ---- Detection (single selected model) ----
let _dt=null;
function runDetection(){
  if(_dt) clearTimeout(_dt);
  _dt=setTimeout(()=>{ if(posPoints.length > 0) runOneModel(); }, 80);
}

// ---- Run ALL models ----
let _prChart = null;
let _newSpectrumChart = null;
let _projectorLossChart = null;
let _projectorProgressTimer = null;
let _activeProjectorProgressId = null;

function _escapeHtml(s){
  return String(s)
    .replace(/&/g,'&amp;')
    .replace(/</g,'&lt;')
    .replace(/>/g,'&gt;')
    .replace(/\"/g,'&quot;')
    .replace(/'/g,'&#39;');
}

function _destroyNewMethodCharts(){
  if(_newSpectrumChart){ _newSpectrumChart.destroy(); _newSpectrumChart = null; }
  if(_projectorLossChart){ _projectorLossChart.destroy(); _projectorLossChart = null; }
  const c = document.getElementById('canvas-rinv');
  if(c){
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  }
}

function isProjectorModel(name){
  return name === 'new_method_mlp' || name === 'new_method_mlp_ace' || name === 'new_method_bilinear' || name === 'new_method_bilinear_ace';
}

function _setNewMethodCardVisibility({showSpectrum=true, showRinv=true, showProjectorLoss=false} = {}){
  const spectrumCard = document.getElementById('analysis-card-spectrum');
  const rinvCard = document.getElementById('analysis-card-rinv');
  const lossCard = document.getElementById('analysis-card-projector-loss');
  if(spectrumCard) spectrumCard.style.display = showSpectrum ? '' : 'none';
  if(rinvCard) rinvCard.style.display = showRinv ? '' : 'none';
  if(lossCard) lossCard.style.display = showProjectorLoss ? '' : 'none';
}

function clearNewMethodAnalysisUI(){
  document.getElementById('method-analysis-top').style.display = 'none';
  document.getElementById('method-analysis-details').style.display = 'none';
  document.getElementById('new-spectrum-note').textContent = '';
  document.getElementById('rinv-note').textContent = '';
  document.getElementById('projector-loss-note').textContent = '';
  document.getElementById('feature-summary').textContent = '';
  document.getElementById('feature-table-wrap').innerHTML = '';
  _setNewMethodCardVisibility();
  _destroyNewMethodCharts();
}

function renderProjectorLossChart(trainInfo, isLive=false){
  const noteEl = document.getElementById('projector-loss-note');
  const canvasEl = document.getElementById('chart-projector-loss');
  if(!canvasEl){
    return;
  }
  const lossHistory = Array.isArray(trainInfo && trainInfo.loss_history)
    ? trainInfo.loss_history.map(v => Number(v))
    : [];
  const diagHistory = Array.isArray(trainInfo && trainInfo.diag_history)
    ? trainInfo.diag_history
    : [];
  if(_projectorLossChart){
    _projectorLossChart.destroy();
    _projectorLossChart = null;
  }
  if(lossHistory.length === 0){
    if(noteEl){
      noteEl.textContent = isLive
        ? 'projector training starting...'
        : 'projector training info unavailable';
    }
    return;
  }

  const labels = lossHistory.map((_, i) => i + 1);
  const detLoss = diagHistory.map(d => Number(d && d.loss_pos));
  const datasets = [{
    label: 'total loss',
    data: lossHistory,
    borderColor: 'rgba(34,197,94,0.95)',
    backgroundColor: 'rgba(34,197,94,0.15)',
    borderWidth: 2.0,
    pointRadius: 0,
    fill: false,
    tension: 0.15,
  }];
  if(detLoss.some(v => Number.isFinite(v))){
    datasets.push({
      label: 'pos loss',
      data: detLoss,
      borderColor: 'rgba(251,191,36,0.95)',
      borderDash: [5, 3],
      borderWidth: 1.6,
      pointRadius: 0,
      fill: false,
      tension: 0.15,
    });
  }

  _projectorLossChart = new Chart(canvasEl, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color:'#a0a8c0', font:{size:10}, boxWidth:12, padding:6 } },
        tooltip: {
          backgroundColor:'#1e2230', titleColor:'#fff', bodyColor:'#c4cbde',
        },
      },
      scales: {
        x: { title:{display:true,text:'Training step',color:'#6b7394'},
             ticks:{color:'#555b72'}, grid:{color:'#1e2230'} },
        y: { title:{display:true,text:'Loss',color:'#6b7394'},
             ticks:{color:'#555b72'}, grid:{color:'#1e2230'} },
      },
    },
  });

  const finalDiag = (trainInfo && trainInfo.final) || (diagHistory.length ? diagHistory[diagHistory.length - 1] : {});
  const step = Number.isFinite(Number(trainInfo && trainInfo.step))
    ? Number(trainInfo.step)
    : lossHistory.length;
  const nSteps = Number.isFinite(Number(trainInfo && trainInfo.n_steps))
    ? Number(trainInfo.n_steps)
    : lossHistory.length;
  const lossMode = String((trainInfo && trainInfo.loss_mode) || '-');
  const totalLoss = Number(finalDiag && finalDiag.loss_total);
  const detLossNow = Number(finalDiag && finalDiag.loss_pos);
  const posMean = Number(finalDiag && finalDiag.pos_score_mean);
  const negMean = Number(finalDiag && finalDiag.neg_score_mean);
  const bgMean = Number(finalDiag && finalDiag.bg_score_mean);
  const statusText = isLive ? 'training' : 'final';
  const parts = [
    `${statusText} ${step}/${nSteps}`,
    `mode=${lossMode}`,
  ];
  if(Number.isFinite(totalLoss)) parts.push(`total=${totalLoss.toFixed(4)}`);
  if(Number.isFinite(detLossNow)) parts.push(`pos=${detLossNow.toFixed(4)}`);
  if(Number.isFinite(posMean)) parts.push(`pos=${posMean.toFixed(4)}`);
  if(Number.isFinite(negMean) && lossMode === 'positive_vs_negative') parts.push(`neg=${negMean.toFixed(4)}`);
  if(Number.isFinite(bgMean) && lossMode !== 'positive_vs_negative') parts.push(`bg=${bgMean.toFixed(4)}`);
  if(noteEl) noteEl.textContent = parts.join(' | ');
}

function prepareProjectorProgressUI(modelName){
  clearNewMethodAnalysisUI();
  document.getElementById('method-analysis-top').style.display = 'block';
  document.getElementById('method-analysis-details').style.display = 'none';
  _setNewMethodCardVisibility({ showSpectrum:false, showRinv:false, showProjectorLoss:true });
  renderProjectorLossChart({ loss_history: [], diag_history: [] }, true);
}

function stopProjectorProgressPolling(){
  if(_projectorProgressTimer){
    clearTimeout(_projectorProgressTimer);
    _projectorProgressTimer = null;
  }
  _activeProjectorProgressId = null;
}

function _makeProgressId(){
  if(window.crypto && typeof window.crypto.randomUUID === 'function'){
    return window.crypto.randomUUID();
  }
  return 'progress_' + Date.now().toString(36) + '_' + Math.random().toString(36).slice(2, 10);
}

async function pollProjectorProgress(progressId){
  if(!progressId || _activeProjectorProgressId !== progressId) return;
  try {
    const resp = await fetch('/projector_progress/' + encodeURIComponent(progressId), { cache:'no-store' });
    const d = await resp.json();
    if(_activeProjectorProgressId !== progressId) return;
    if(d && d.found){
      if(d.state === 'error'){
        const noteEl = document.getElementById('projector-loss-note');
        if(noteEl) noteEl.textContent = 'projector training failed: ' + (d.error || 'unknown error');
      } else {
        renderProjectorLossChart(d, d.state !== 'done');
      }
    }
  } catch(err) {
    console.error('[projector_progress]', err);
  }
  if(_activeProjectorProgressId === progressId){
    _projectorProgressTimer = setTimeout(() => pollProjectorProgress(progressId), 200);
  }
}

function renderRinvHeatmap(matrix){
  const c = document.getElementById('canvas-rinv');
  const note = document.getElementById('rinv-note');
  if(!c) return;
  if(!Array.isArray(matrix) || matrix.length === 0 || !Array.isArray(matrix[0])){
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
    if(note) note.textContent = 'R^-1 not available';
    return;
  }

  const h = matrix.length;
  const w = matrix[0].length;
  const flat = [];
  for(let y=0; y<h; y++){
    for(let x=0; x<w; x++) flat.push(Number(matrix[y][x]) || 0);
  }
  let absMax = 0;
  for(const v of flat){ const a = Math.abs(v); if(a > absMax) absMax = a; }
  absMax = Math.max(absMax, 1e-9);

  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const offCtx = off.getContext('2d');
  const img = offCtx.createImageData(w, h);
  for(let y=0; y<h; y++){
    for(let x=0; x<w; x++){
      const v = (Number(matrix[y][x]) || 0) / absMax;
      const t = Math.max(-1, Math.min(1, v));
      let r=0,g=0,b=0;
      if(t >= 0){
        r = 255;
        g = Math.round(255 * (1 - t));
        b = Math.round(255 * (1 - t));
      } else {
        b = 255;
        r = Math.round(255 * (1 + t));
        g = Math.round(255 * (1 + t));
      }
      const p = (y * w + x) * 4;
      img.data[p + 0] = r;
      img.data[p + 1] = g;
      img.data[p + 2] = b;
      img.data[p + 3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);

  const cw = c.clientWidth || 320;
  const ch = c.clientHeight || 220;
  c.width = cw; c.height = ch;
  const ctx = c.getContext('2d');
  ctx.imageSmoothingEnabled = false;
  ctx.clearRect(0, 0, c.width, c.height);
  ctx.drawImage(off, 0, 0, c.width, c.height);
  if(note) note.textContent = `size ${w}x${h} | color: blue(-) ~ red(+)`;
}

function renderNewMethodAnalysis(analysis, modelName){
  const isNewMethod =
    modelName === 'new_method_amf'
    || modelName === 'new_method_ace'
    || modelName === 'new_method_mlp'
    || modelName === 'new_method_mlp_ace'
    || modelName === 'new_method_bilinear'
    || modelName === 'new_method_bilinear_ace';
  if(!isNewMethod || !analysis || !analysis.enabled){
    clearNewMethodAnalysisUI();
    return;
  }

  document.getElementById('method-analysis-top').style.display = 'block';
  document.getElementById('method-analysis-details').style.display = 'block';
  _setNewMethodCardVisibility({
    showSpectrum: true,
    showRinv: true,
    showProjectorLoss: !!(analysis && analysis.projector_used),
  });

  const spectra = analysis.spectra || {};
  const pos = Array.isArray(spectra.pos) ? spectra.pos : [];
  const neg = Array.isArray(spectra.neg) ? spectra.neg : [];
  const target = Array.isArray(spectra.target) ? spectra.target : [];
  const muB = Array.isArray(spectra.mu_b) ? spectra.mu_b : [];
  const displaySpace = spectra.display_space || '';
  const axisLabels = Array.isArray(analysis.feature_axis_labels) ? analysis.feature_axis_labels : [];
  const D = target.length || (pos[0] ? pos[0].length : (muB.length || 0));

  const labels = Array.from({length: D}, (_, i) => i + 1);
  const datasets = [];
  if(muB.length === D){
    datasets.push({
      label: 'background mean', data: muB,
      borderColor: 'rgba(148,163,184,0.9)', borderDash:[5,3],
      borderWidth: 1.8, pointRadius: 0, fill: false, tension: 0.15,
    });
  }
  if(target.length === D){
    datasets.push({
      label: 'target(mean pos)', data: target,
      borderColor: 'rgba(251,191,36,0.95)',
      borderWidth: 2.2, pointRadius: 0, fill: false, tension: 0.15,
    });
  }
  pos.forEach((sp, i) => {
    if(!Array.isArray(sp) || sp.length !== D) return;
    datasets.push({
      label: `pos #${i+1}`, data: sp,
      borderColor: `rgba(34,197,94,${Math.max(0.35, 0.9 - i * 0.1)})`,
      borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0.15,
    });
  });
  neg.forEach((sp, i) => {
    if(!Array.isArray(sp) || sp.length !== D) return;
    datasets.push({
      label: `neg #${i+1}`, data: sp,
      borderColor: `rgba(239,68,68,${Math.max(0.35, 0.9 - i * 0.1)})`,
      borderWidth: 1.6, pointRadius: 0, fill: false, tension: 0.15,
    });
  });

  const spectrumCv = document.getElementById('chart-new-spectrum');
  if(_newSpectrumChart){ _newSpectrumChart.destroy(); _newSpectrumChart = null; }
  if(spectrumCv && D > 0 && datasets.length > 0){
    _newSpectrumChart = new Chart(spectrumCv, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true, maintainAspectRatio: false, animation: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color:'#a0a8c0', font:{size:10}, boxWidth:12, padding:6 } },
          tooltip: {
            backgroundColor:'#1e2230', titleColor:'#fff', bodyColor:'#c4cbde',
            callbacks: {
              title(items){
                const idx = items && items.length ? items[0].dataIndex : 0;
                return `Band ${idx + 1}`;
              },
              afterTitle(items){
                const idx = items && items.length ? items[0].dataIndex : 0;
                return axisLabels[idx] || '';
              },
              label(ctx){
                const y = (ctx && ctx.parsed && Number.isFinite(ctx.parsed.y))
                  ? Number(ctx.parsed.y).toFixed(4)
                  : '-';
                return `${ctx.dataset.label}: ${y}`;
              },
            },
          },
        },
        scales: {
          x: { title:{display:true,text:'Expanded feature index',color:'#6b7394'},
               ticks:{color:'#555b72',maxTicksLimit:20}, grid:{color:'#1e2230'} },
          y: { title:{display:true,text:'Normalized value',color:'#6b7394'},
               ticks:{color:'#555b72'}, grid:{color:'#1e2230'} },
        },
      },
    });
  }

  const bank = analysis.feature_bank_size || 0;
  const selDim = analysis.feature_selected_dim || 0;
  const mode = analysis.feature_selection_mode || '-';
  const norm = analysis.feature_norm_method || '-';
  const fallback = analysis.feature_used_neg_fallback ? 'yes' : 'no';
  const covInfo = analysis.cov_info || {};
  const etaUsed = (covInfo.eta_used !== undefined) ? covInfo.eta_used : '-';
  const covDtype = covInfo.cov_dtype_used || '-';
  const memInfo = analysis.memory_estimate || {};
  const fmMb = (memInfo.feature_map_mb_fp32 !== undefined) ? memInfo.feature_map_mb_fp32 : '-';
  const cmMb = (memInfo.cov_matrix_mb_fp64 !== undefined) ? memInfo.cov_matrix_mb_fp64 : '-';
  const gpuMem = analysis.gpu_mem || {};
  const prepGpu = (gpuMem.prep || {});
  const runGpu = (gpuMem.run || {});
  const prepPeak = (prepGpu.max_reserved_mb !== undefined) ? Number(prepGpu.max_reserved_mb).toFixed(1) : '-';
  const runPeak = (runGpu.max_reserved_mb !== undefined) ? Number(runGpu.max_reserved_mb).toFixed(1) : '-';
  document.getElementById('new-spectrum-note').textContent =
    `bank=${bank}, selected=${selDim}, mode=${mode}, norm=${norm}, display=${displaySpace || 'weighted_selected'}, hover=band meaning, neg-fallback=${fallback}, eta=${etaUsed}, cov-dtype=${covDtype}, feat=${fmMb}MB, cov=${cmMb}MB, gpu-peak(prep/run)=${prepPeak}/${runPeak}MB`;

  if(analysis.projector_used){
    renderProjectorLossChart(analysis.projector_train_info || {}, false);
  } else {
    document.getElementById('projector-loss-note').textContent = '';
    if(_projectorLossChart){
      _projectorLossChart.destroy();
      _projectorLossChart = null;
    }
  }

  const rinv = analysis.r_inv || {};
  renderRinvHeatmap(rinv.matrix || []);
  const fullDim = rinv.dim_full || 0;
  const shownDim = rinv.dim_shown || 0;
  if(fullDim > 0){
    const sfx = (shownDim && shownDim < fullDim) ? ` (shown ${shownDim})` : '';
    document.getElementById('rinv-note').textContent =
      `R^-1 dimension ${fullDim}${sfx}`;
  }

  const rows = Array.isArray(analysis.feature_rows) ? analysis.feature_rows : [];
  document.getElementById('feature-summary').textContent =
    `selected ${selDim} / ${bank} engineered features`;
  const tableWrap = document.getElementById('feature-table-wrap');
  if(rows.length === 0){
    tableWrap.innerHTML = '<div style=\"padding:8px;color:#8890a8;font-size:11px\">No feature rows</div>';
  } else {
    const hasScoreCols = rows.some(r =>
      r.sep_score !== undefined || r.final_score !== undefined || r.weight !== undefined);
    let html = '<table class=\"feature-table\"><thead><tr>' +
      '<th>#</th><th>band</th><th>idx</th><th>feature</th>';
    if(hasScoreCols){
      html += '<th>sep</th><th>final</th><th>w</th>';
    }
    html += '</tr></thead><tbody>';
    rows.forEach(r => {
      html += '<tr>' +
        `<td>${r.rank}</td>` +
        `<td>${r.band}</td>` +
        `<td>${r.idx}</td>` +
        `<td>${_escapeHtml(r.label || '')}</td>`;
      if(hasScoreCols){
        const sep = (r.sep_score !== undefined) ? Number(r.sep_score).toFixed(4) : '-';
        const fin = (r.final_score !== undefined) ? Number(r.final_score).toFixed(4) : '-';
        const w = (r.weight !== undefined) ? Number(r.weight).toFixed(4) : '-';
        html += `<td>${sep}</td><td>${fin}</td><td>${w}</td>`;
      }
      html += '</tr>';
    });
    html += '</tbody></table>';
    tableWrap.innerHTML = html;
  }
}

// ---- Single model run (primary action) ----
let _runOneInFlight = false;
async function runOneModel(){
  if(posPoints.length === 0) return;
  if(_runOneInFlight) return;
  const modelName = selModel.value;
  if(!modelName) return;
  let progressId = null;
  if(isProjectorModel(modelName)){
    progressId = _makeProgressId();
    _activeProjectorProgressId = progressId;
    prepareProjectorProgressUI(modelName);
    pollProjectorProgress(progressId);
  } else {
    stopProjectorProgressPolling();
  }
  _runOneInFlight = true;
  spinner.style.display = 'block';
  try {
    const body = {
      model: modelName,
      pos: posPoints.map(p=>[p.x, p.y]),
      neg: negPoints.map(p=>[p.x, p.y]),
    };
    if(progressId){
      body.progress_id = progressId;
    }
    if(isThresholdTunable(modelName) && _samThrManual){
      body.threshold = parseFloat(samThrSlider.value);
    }
    const resp = await fetch('/detect_one', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(body),
    });
    const d = await resp.json();
    if(!resp.ok || d.error){
      console.error('[runOneModel]', d.error);
      document.getElementById('ph-result-score').textContent = 'Error: ' + (d.error||resp.status);
      document.getElementById('ph-result-score').style.display = '';
      document.getElementById('ph-result-seg').textContent   = 'Error: ' + (d.error||resp.status);
      document.getElementById('ph-result-seg').style.display = '';
      markSingleResultDirty();
      clearNewMethodAnalysisUI();
      return;
    }

    // Get model color
    const mObj = availModels.find(x => x.id === modelName);
    const color = mObj ? mObj.color : '#888';

    // Score map panel
    showSyncedResultImage('img-result-score', 'zoom-wrap-score', 'ph-result-score', d.heatmap);
    document.getElementById('dot-result').style.background = color;
    document.getElementById('label-result').textContent = modelName + ' – Score Map';
    document.getElementById('meta-result-score').textContent = d.ms + ' ms';

    // Seg map panel
    showSyncedResultImage('img-result-seg', 'zoom-wrap-seg', 'ph-result-seg', d.seg);
    document.getElementById('dot-result2').style.background = color;
    document.getElementById('label-result2').textContent = modelName + ' – Seg Map';
    document.getElementById('meta-result-seg').textContent = '';
    _applySyncedViewportLayout();
    markSingleResultReady(modelName);
    renderNewMethodAnalysis(d.analysis || null, modelName);

    updateSaveBtn();

    if(isThresholdTunable(modelName) && d.threshold_min != null && d.threshold_max != null){
      const sliderValue = (_samThrManual && samThrSlider.value !== '')
        ? parseFloat(samThrSlider.value)
        : d.used_threshold;
      setThresholdSliderRange(d.threshold_min, d.threshold_max, sliderValue);
    }
    if(isThresholdTunable(modelName) && d.used_threshold != null && !_samThrManual){
      samThrVal.textContent = formatThresholdValue(d.used_threshold);
      samThrUsed.textContent = '(auto)';
    } else if(isThresholdTunable(modelName) && d.used_threshold != null){
      samThrVal.textContent = formatThresholdValue(samThrSlider.value);
      samThrUsed.textContent = '(manual)';
    }

    // Metrics bar
    const mb = document.getElementById('metrics-bar');
    if(d.has_gt && d.iou !== undefined){
      document.getElementById('metrics-model-label').textContent = modelName;
      document.getElementById('m-iou').textContent  = (d.iou*100).toFixed(1)  + '%';
      document.getElementById('m-prec').textContent = (d.prec*100).toFixed(1) + '%';
      document.getElementById('m-rec').textContent  = (d.rec*100).toFixed(1)  + '%';
      document.getElementById('m-ap').textContent   = d.pr ? (d.pr.ap*100).toFixed(1)+'%' : '—';
      document.getElementById('m-ms').textContent   = d.ms + ' ms';
      // Show used threshold in metrics bar for thresholdable score-map models
      const mThrRow = document.getElementById('m-thr-row');
      if(isThresholdTunable(modelName) && d.used_threshold != null){
        document.getElementById('m-thr').textContent = formatThresholdValue(d.used_threshold);
        if(mThrRow) mThrRow.style.display = '';
      } else {
        if(mThrRow) mThrRow.style.display = 'none';
      }
      mb.style.display = 'flex';
      // PR curve
      if(d.pr && window.Chart){
        document.getElementById('pr-panel').style.display = '';
        const ctx2 = document.getElementById('chart-pr-single').getContext('2d');
        if(_prChart) _prChart.destroy();
        _prChart = new Chart(ctx2, {
          type:'line',
          data:{datasets:[{
            label: modelName + ' (AP=' + (d.pr.ap*100).toFixed(1)+'%)',
            data: d.pr.recall.map((r,i)=>({x:r, y:d.pr.precision[i]})),
            borderColor: color, backgroundColor: color+'33',
            borderWidth:2, pointRadius:0, fill:true, tension:0.3,
          }]},
          options:{
            animation:false, responsive:true,
            scales:{
              x:{type:'linear',min:0,max:1,title:{display:true,text:'Recall',color:'#8890a8'},ticks:{color:'#8890a8'},grid:{color:'#2a2f45'}},
              y:{type:'linear',min:0,max:1,title:{display:true,text:'Precision',color:'#8890a8'},ticks:{color:'#8890a8'},grid:{color:'#2a2f45'}},
            },
            plugins:{legend:{labels:{color:'#e0e0e0'}},},
            backgroundColor:'#0d0f18',
          }
        });
      }
    } else {
      mb.style.display = 'none';
      document.getElementById('pr-panel').style.display = 'none';
    }

  } finally {
    if(progressId && _activeProjectorProgressId === progressId){
      stopProjectorProgressPolling();
    }
    spinner.style.display = 'none';
    _runOneInFlight = false;
  }
}

// Click or zoom on result panels → stay in sync with the input viewport.
['zoom-wrap-score','zoom-wrap-seg'].forEach(id=>{
  const wrap = document.getElementById(id);
  if(!wrap) return;
  wrap.addEventListener('mousemove', (e)=> rememberViewportPointer(e, wrap));
  wrap.addEventListener('wheel', (e)=> applySharedWheelZoom(e, wrap), {passive:false});
  wrap.addEventListener('mousedown', e=>{
    if(e.button === 1){
      startSharedPan(e);
      return;
    }
    e.preventDefault();
    const imgId = id === 'zoom-wrap-score' ? 'img-result-score' : 'img-result-seg';
    const imgEl = document.getElementById(imgId);
    if(!imgEl || !imgEl.src || imgEl.style.display === 'none') return;
    const {x, y} = viewportToOrig(e, wrap);
    handlePromptClick(x, y, e.button);
  });
  wrap.addEventListener('contextmenu', e=> e.preventDefault());
});

let _runAllInFlight = false;
async function runAllModels(){
  if(posPoints.length === 0) return;  // no-op when called from auto-run with no points
  if(_runAllInFlight) return;         // prevent parallel duplicate runs
  _runAllInFlight = true;
  clearNewMethodAnalysisUI();
  spinner.style.display = 'block';
  document.getElementById('btn-run-all').disabled = true;
  try {
    const body = {
      pos: posPoints.map(p=>[p.x, p.y]),
      neg: negPoints.map(p=>[p.x, p.y]),
    };
    const resp = await fetch('/detect_all', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    const d = await resp.json();
    if(d.error){ alert(d.error); return; }

    const models = d.models || {};
    const hasGt = d.has_gt || false;

    // Update each panel
    for(const [name, res] of Object.entries(models)){
      const pm = MODEL_PANEL_MAP[name];
      if(!pm) continue;
      if(res.error){
        document.getElementById(pm.ph).textContent = 'Error: ' + res.error;
        continue;
      }
      // Show seg map (TP/FP/FN if GT loaded, else overlay)
      if(res.seg){ showImg(pm.img, pm.ph, res.seg); }
      const metaEl = document.getElementById(pm.meta);
      if(metaEl){
        let txt = res.ms ? res.ms+'ms' : '';
        if(hasGt && res.iou !== undefined)
          txt += '  IoU:'+res.iou.toFixed(3)+' P:'+res.prec.toFixed(3)+' R:'+res.rec.toFixed(3);
        metaEl.textContent = txt;
      }
    }

    // Show all-models comparison section
    document.getElementById('all-models-section').style.display = 'block';

    // Update results table + PR chart
    renderAllResults(models, hasGt);

  } catch(err){ console.error(err); alert('Detection failed: ' + err); }
  finally {
    _runAllInFlight = false;
    spinner.style.display='none';
    document.getElementById('btn-run-all').disabled=false;
    updateSaveBtn();
  }
}

function renderAllResults(models, hasGt){
  const tbody = document.getElementById('tbl-all-body') || document.getElementById('tbody-results');
  if(!tbody) return;
  tbody.innerHTML = '';
  const prDatasets = [];

  for(const [name, res] of Object.entries(models)){
    const tr = document.createElement('tr');
    tr.style.borderBottom = '1px solid #2a2f45';
    const color = MODEL_COLORS[name] || '#888';
    const iouCell  = hasGt && res.iou  !== undefined ? res.iou.toFixed(3)  : '—';
    const precCell = hasGt && res.prec !== undefined ? res.prec.toFixed(3) : '—';
    const recCell  = hasGt && res.rec  !== undefined ? res.rec.toFixed(3)  : '—';
    const msCell   = res.ms !== undefined ? res.ms+'ms' : '—';
    tr.innerHTML =
      `<td style="padding:5px 8px;color:${color};font-weight:600">${name}</td>`+
      `<td style="padding:5px 8px;text-align:right">${iouCell}</td>`+
      `<td style="padding:5px 8px;text-align:right">${precCell}</td>`+
      `<td style="padding:5px 8px;text-align:right">${recCell}</td>`+
      `<td style="padding:5px 8px;text-align:right;color:#6b7394">${msCell}</td>`;
    if(res.error) tr.style.opacity='0.4';
    tbody.appendChild(tr);

    if(hasGt && res.pr){
      prDatasets.push({
        label: name + ' (AP='+res.pr.ap+')',
        data: res.pr.recall.map((r,i)=>({x:r, y:res.pr.precision[i]})),
        borderColor: color, backgroundColor: 'transparent',
        borderWidth: 2, pointRadius: 0, tension: 0.2, fill: false,
      });
    }
  }

  // PR Chart
  const prCanvas = document.getElementById('chart-pr');
  if(prCanvas && prDatasets.length > 0){
    if(_prChart){ _prChart.destroy(); _prChart = null; }
    _prChart = new Chart(prCanvas, {
      type: 'line',
      data: { datasets: prDatasets },
      options: {
        scales: {
          x: { type:'linear', min:0, max:1, title:{display:true,text:'Recall',color:'#a0a8c0'}, ticks:{color:'#8890a8'}, grid:{color:'#2d3044'} },
          y: { min:0, max:1, title:{display:true,text:'Precision',color:'#a0a8c0'}, ticks:{color:'#8890a8'}, grid:{color:'#2d3044'} },
        },
        plugins: {
          legend: { labels: { color:'#a0a8c0', font:{size:10}, boxWidth:14 } },
          tooltip: { backgroundColor:'#1e2230', titleColor:'#fff', bodyColor:'#c4cbde' },
        },
        animation: false,
      }
    });
  }

  // Show legend for GT seg map
  const infoEl = document.getElementById('eval-gt-info-main');
  if(infoEl){
    infoEl.textContent = hasGt
      ? '🟪 TP  🟥 FP  🟦 FN'
      : '(load GT mask to see IoU / F1 / PR curve)';
  }
}

function showImg(imgId,phId,b64){
  if(!b64) return;
  const el=document.getElementById(imgId);
  if(!el) return;
  el.src='data:image/png;base64,'+b64;
  el.style.display='block';
  if(phId){
    const ph=document.getElementById(phId);
    if(ph) ph.style.display='none';
  }
}

// ═══════ Spectra & Weight charts (Chart.js) ═══════
let rawChart = null;
let projectedChart = null;
let svdChart = null;

const CHART_COLORS_POS = [
  'rgba(34,197,94,0.9)', 'rgba(74,222,128,0.7)', 'rgba(22,163,74,0.7)',
  'rgba(134,239,172,0.6)', 'rgba(21,128,61,0.7)',
];
const CHART_COLORS_NEG = [
  'rgba(239,68,68,0.9)', 'rgba(248,113,113,0.7)', 'rgba(220,38,38,0.7)',
  'rgba(252,165,165,0.6)', 'rgba(185,28,28,0.7)',
];

const CHART_THEME = {
  legend: { position: 'top', labels: { color: '#a0a8c0', font: { size: 10 }, boxWidth: 14, padding: 6 } },
  tooltip: { backgroundColor: '#1e2230', titleColor: '#fff', bodyColor: '#c4cbde', borderColor: '#3a3f55', borderWidth: 1 },
};

function buildLineDatasets(pos, neg, muB) {
  const ds = [];
  if (muB && muB.length > 0) {
    ds.push({
      label: 'μ_B', data: muB,
      borderColor: 'rgba(148,163,184,0.8)', backgroundColor: 'transparent',
      borderWidth: 2, borderDash: [6, 3], pointRadius: 0, tension: 0.2, order: 10,
    });
  }
  pos.forEach((s, i) => ds.push({
    label: 'Pos #' + (i+1), data: s,
    borderColor: CHART_COLORS_POS[i % CHART_COLORS_POS.length],
    backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.2, order: i,
  }));
  neg.forEach((s, i) => ds.push({
    label: 'Neg #' + (i+1), data: s,
    borderColor: CHART_COLORS_NEG[i % CHART_COLORS_NEG.length],
    backgroundColor: 'transparent', borderWidth: 2, pointRadius: 0, tension: 0.2, order: pos.length + i,
  }));
  return ds;
}

function makeLineChart(canvasId, datasets, D, xLabel, yLabel) {
  const labels = Array.from({length: D}, (_, i) => i + 1);
  return new Chart(document.getElementById(canvasId), {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: CHART_THEME,
      scales: {
        x: { title: { display: true, text: xLabel, color: '#6b7394', font: { size: 10 } },
             ticks: { color: '#555b72', font: { size: 9 }, maxTicksLimit: 20 }, grid: { color: '#1e2230' } },
        y: { title: { display: true, text: yLabel, color: '#6b7394', font: { size: 10 } },
             ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' } },
      },
    },
  });
}

function renderRawSpectra(posRaw, negRaw, muB) {
  const row = document.getElementById('charts-row');
  const hasData = (posRaw.length + negRaw.length) > 0;
  if (!hasData) { if (row) row.style.display = 'none'; return; }
  if (row) row.style.display = 'block';

  const D = (posRaw[0] || negRaw[0]).length;
  if (rawChart) { rawChart.destroy(); rawChart = null; }
  rawChart = makeLineChart('chart-raw',
    buildLineDatasets(posRaw, negRaw, muB), D, 'Band', 'Value');
}

function renderProjectedSpectra(ospInfo, posRaw, negRaw) {
  const metricsEl = document.getElementById('oblique-metrics');
  if (!ospInfo || !ospInfo.applied) {
    if (projectedChart) { projectedChart.destroy(); projectedChart = null; }
    if (metricsEl) metricsEl.innerHTML = '<span style="color:#555">no negatives → no projection</span>';
    return;
  }
  const posProj = ospInfo.pos_projected || [];
  const negProj = ospInfo.neg_projected || [];
  const lam = ospInfo['lambda'];
  const r = ospInfo.rank_kept || 0;
  const posPres = ospInfo.pos_preservation || [];
  const negSupp = ospInfo.neg_suppression || [];
  const overlaps = ospInfo.overlap_ratios || [];

  if (metricsEl) {
    const lamStr = lam != null ? (lam < 0.01 ? lam.toExponential(1) : lam.toFixed(2)) : '?';
    const avgPres = posPres.length > 0 ? (posPres.reduce((a,b)=>a+b,0)/posPres.length*100).toFixed(1) : '—';
    const avgSupp = negSupp.length > 0 ? (negSupp.reduce((a,b)=>a+b,0)/negSupp.length*100).toFixed(1) : '—';
    metricsEl.innerHTML =
      `<span style="color:#22c55e">▲ Pos preserved: <b>${avgPres}%</b></span>` +
      ` &nbsp;|&nbsp; <span style="color:#ef4444">▼ Neg suppressed: <b>${avgSupp}%</b></span>` +
      ` &nbsp;|&nbsp; rank=${r}, λ=${lamStr}`;
  }
  if (posProj.length === 0 && negProj.length === 0) return;

  const D = (posProj[0] || negProj[0] || posRaw[0] || []).length;
  const labels = Array.from({length: D}, (_, i) => i + 1);
  const datasets = [];

  // Original positive (dashed green)
  posRaw.forEach((sp, i) => {
    datasets.push({
      label: `Pos #${i+1} orig`,
      data: sp, borderColor: 'rgba(34,197,94,0.3)', borderWidth: 1.5,
      borderDash: [4,3], pointRadius: 0, fill: false,
    });
  });
  // Projected positive (solid green)
  posProj.forEach((sp, i) => {
    datasets.push({
      label: `Pos #${i+1} proj`,
      data: sp, borderColor: 'rgba(34,197,94,1)', borderWidth: 2,
      pointRadius: 2, pointBackgroundColor: 'rgba(34,197,94,1)', fill: false,
    });
  });
  // Original negative (dashed red)
  negRaw.forEach((sp, i) => {
    datasets.push({
      label: `Neg #${i+1} orig`,
      data: sp, borderColor: 'rgba(239,68,68,0.3)', borderWidth: 1.5,
      borderDash: [4,3], pointRadius: 0, fill: false,
    });
  });
  // Projected negative (solid red)
  negProj.forEach((sp, i) => {
    datasets.push({
      label: `Neg #${i+1} proj`,
      data: sp, borderColor: 'rgba(239,68,68,1)', borderWidth: 2,
      pointRadius: 2, pointBackgroundColor: 'rgba(239,68,68,1)', fill: false,
    });
  });

  const cvEl = document.getElementById('chart-projected');
  if (projectedChart) { projectedChart.destroy(); projectedChart = null; }
  projectedChart = new Chart(cvEl, {
    type: 'line',
    data: { labels, datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { display: true, position: 'top',
          labels: { color: '#a0a8c0', font: { size: 8 }, boxWidth: 10, padding: 5,
            filter: (item) => !item.text.includes('orig') } },
        tooltip: { backgroundColor: '#1e2230', titleColor: '#fff', bodyColor: '#c4cbde',
          borderColor: '#3a3f55', borderWidth: 1 },
      },
      scales: {
        x: { title: { display: true, text: 'Band', color: '#6b7394', font: { size: 10 } },
             ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' } },
        y: { title: { display: true, text: 'Value', color: '#6b7394', font: { size: 10 } },
             ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' } },
      },
    },
  });
}

function renderSvdChart(ospInfo) {
  const metricsEl = document.getElementById('svd-metrics');
  if (!ospInfo || !ospInfo.applied || !ospInfo.sv_all) {
    if (svdChart) { svdChart.destroy(); svdChart = null; }
    if (metricsEl) metricsEl.innerHTML = '';
    return;
  }
  const svAll = ospInfo.sv_all || [];
  const rankRaw = ospInfo.rank_raw || svAll.length;
  const rankKept = ospInfo.rank_kept || svAll.length;
  const overlaps = ospInfo.overlap_ratios || [];
  const msg = ospInfo.msg || '';

  if (metricsEl) {
    let html = `<span>${rankKept}/${rankRaw} neg-only directions kept</span>`;
    if (overlaps.length > 0) {
      const avgOvl = (overlaps.reduce((a,b)=>a+b,0)/overlaps.length*100).toFixed(1);
      html += ` &nbsp;|&nbsp; <span style="color:#f59e0b">avg overlap with pos: ${avgOvl}%</span>`;
    }
    if (msg) html += ` &nbsp;|&nbsp; <span style="color:#ef4444">${msg}</span>`;
    metricsEl.innerHTML = html;
  }

  const labels = svAll.map((_, i) => `σ${i+1}`);
  const values = svAll;
  const colors = svAll.map((_, i) =>
    i < rankKept ? 'rgba(168,85,247,0.8)' : 'rgba(100,116,160,0.3)');

  const cvEl = document.getElementById('chart-svd');
  if (svdChart) { svdChart.destroy(); svdChart = null; }
  svdChart = new Chart(cvEl, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Singular value',
        data: values,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace('0.8','1').replace('0.3','0.5')),
        borderWidth: 1,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1e2230', titleColor: '#fff', bodyColor: '#c4cbde',
          borderColor: '#3a3f55', borderWidth: 1 },
      },
      scales: {
        x: { title: { display: true, text: 'Direction', color: '#6b7394', font: { size: 10 } },
             ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' } },
        y: { title: { display: true, text: 'σ', color: '#6b7394', font: { size: 10 } },
             ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' } },
      },
    },
  });
}

let scoreHistChart = null;

function renderScoreHist(sh) {
  const infoEl = document.getElementById('score-dist-info');
  if (!sh || !sh.hist_counts || sh.hist_counts.length === 0) {
    if (scoreHistChart) { scoreHistChart.destroy(); scoreHistChart = null; }
    if (infoEl) infoEl.textContent = '';
    return;
  }

  const histCounts = sh.hist_counts;
  const histCenters = sh.hist_centers;
  const maxC = Math.max(...histCounts, 1);
  if (infoEl) {
    const lo = histCenters[0].toFixed(3);
    const hi = histCenters[histCenters.length-1].toFixed(3);
    infoEl.textContent = `range [${lo}, ${hi}]`;
  }

  const datasets = [{
    type: 'bar',
    label: 'Score',
    data: histCenters.map((c, i) => ({x: c, y: histCounts[i] / maxC})),
    backgroundColor: 'rgba(16,185,129,0.45)',
    borderColor: 'rgba(16,185,129,0.7)',
    borderWidth: 1,
    barPercentage: 1.0,
    categoryPercentage: 1.0,
    order: 2,
  }];

  const cvEl = document.getElementById('chart-score-hist');
  if (scoreHistChart) { scoreHistChart.destroy(); scoreHistChart = null; }
  scoreHistChart = new Chart(cvEl, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: '#1e2230', titleColor: '#fff', bodyColor: '#c4cbde',
          borderColor: '#3a3f55', borderWidth: 1 },
      },
      scales: {
        x: { type: 'linear',
          title: { display: true, text: 'Score', color: '#6b7394', font: { size: 10 } },
          ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' } },
        y: { title: { display: true, text: 'density', color: '#6b7394', font: { size: 10 } },
          ticks: { color: '#555b72', font: { size: 9 } }, grid: { color: '#1e2230' },
          min: 0, max: 1.1 },
      },
    },
  });
}

// (R_inv heatmap removed — replaced by oblique projector monitoring)

// ---- Controls ----
function clearPoints(){
  if(_autoRunTimer){ clearTimeout(_autoRunTimer); _autoRunTimer=null; }
  resetThresholdOverride();
  posPoints=[]; negPoints=[]; history=[];
  redraw(); clearResults(); updateInfo();
}
function undoPoint(){
  if(!history.length) return;
  resetThresholdOverride();
  if(history.pop()==='pos') posPoints.pop(); else negPoints.pop();
  redraw(); updateInfo();
  if(posPoints.length>0) scheduleAutoRun(); else clearResults();
}
// ---- Classical detector functions ----
function clearClassicalTargets(){
  classicalTargetPixels = [];
  document.getElementById('classical-target-info').textContent =
    classicalMode ? 'Click image to select target pixel(s)' : '';
  document.getElementById('btn-classical-clear').disabled = true;
  document.getElementById('btn-classical-run').disabled = true;
  redraw();
}

async function runClassicalDetection(){
  if(classicalTargetPixels.length === 0) return;
  const algo = document.getElementById('sel-classical-algo').value;
  spinner.style.display = 'block';
  const body = {
    algorithm: algo,
    target_pixels: classicalTargetPixels.map(p => [p.x, p.y]),
  };
  try {
    const resp = await fetch('/detect_classical', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body),
    });
    const d = await resp.json();
    if(d.error){ alert(d.error); return; }

    // Show results in the same panels
    if(d.heatmap) showImg('img-score','ph-score', d.heatmap);
    document.getElementById('meta-score').textContent =
      algo + ' ' + d.ms + 'ms';
    if(d.overlay) showImg('img-overlay','ph-overlay', d.overlay);
    const pct = (d.det/d.total*100).toFixed(2);
    document.getElementById('meta-overlay').textContent =
      d.det.toLocaleString()+' px ('+pct+'%)';
    document.getElementById('title-score').textContent = algo + ' Score Map';
    document.getElementById('title-overlay').textContent = algo + ' Detection Overlay';

    // Evaluation if available
    renderEvalCharts(d.eval || null);
    updateSaveBtn();

    // Update info bar
    document.getElementById('i-det').textContent = d.det.toLocaleString();

  } catch(err){ console.error(err); } finally { spinner.style.display='none'; }
}

function clearResults(){
  markSingleResultDirty();
  clearNewMethodAnalysisUI();
  // Primary single-model panels
  ['img-result-score','img-result-seg'].forEach(id=>{
    const el = document.getElementById(id); if(el) el.style.display = 'none';
  });
  ['zoom-wrap-score','zoom-wrap-seg'].forEach(id=>{
    const el = document.getElementById(id); if(el) el.style.display = 'none';
  });
  ['ph-result-score','ph-result-seg'].forEach(id=>{
    const el = document.getElementById(id); if(el){ el.style.display=''; el.textContent='Add a prompt point to run'; }
  });
  document.getElementById('metrics-bar').style.display = 'none';
  document.getElementById('pr-panel').style.display = 'none';
  document.getElementById('btn-save').disabled = true;
  // Reset threshold display (keep the last slider range/value for convenience)
  samThrUsed.textContent = '';
  const mThrRow = document.getElementById('m-thr-row');
  if(mThrRow) mThrRow.style.display = 'none';

  // All-models section
  for(const pm of Object.values(MODEL_PANEL_MAP)){
    const el=document.getElementById(pm.img); if(el) el.style.display='none';
    const ph=document.getElementById(pm.ph); if(ph){ph.style.display='flex';ph.textContent='—';}
    const meta=document.getElementById(pm.meta); if(meta) meta.textContent='';
  }
  ['img-feat','img-score','img-overlay'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.style.display='none';
  });
  excludeOverlayImg = null;
  redraw();
  if (_prChart) { _prChart.destroy(); _prChart = null; }
  if (rocChart) { rocChart.destroy(); rocChart = null; }
  if (prChart) { prChart.destroy(); prChart = null; }
  if (stabilityChart) { stabilityChart.destroy(); stabilityChart = null; }
  if (fpmpChart) { fpmpChart.destroy(); fpmpChart = null; }
  const evalRow = document.getElementById('eval-row');
  if (evalRow) evalRow.style.display = 'none';
  const tbody = document.getElementById('tbody-results');
  if(tbody) tbody.innerHTML = '';
  updateSaveBtn();
  const phFeat=document.getElementById('ph-feat');
  if(phFeat){phFeat.style.display='block'; phFeat.textContent='Select model & click'}
  const phScore=document.getElementById('ph-score');
  if(phScore){phScore.style.display='block'; phScore.textContent='Click to detect'}
  const phOverlay=document.getElementById('ph-overlay');
  if(phOverlay){phOverlay.style.display='block'; phOverlay.textContent='Click to detect'}
  ['meta-feat','meta-score','meta-overlay'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.textContent='';
  });
}
function updateInfo(){
  document.getElementById('i-pts').textContent=posPoints.length+' pos / '+negPoints.length+' neg';
  updateSaveBtn();
}

// ═══════ GT Mask upload ═══════
document.getElementById('gt-input').addEventListener('change', async (e)=>{
  const file = e.target.files[0];
  if(!file) return;
  const form = new FormData();
  form.append('gt_mask', file);
  spinner.style.display='block';
  const resp = await fetch('/load_gt_mask',{method:'POST',body:form});
  const d = await resp.json();
  spinner.style.display='none';
  if(d.error){alert(d.error);return}
  document.getElementById('gt-info').textContent =
    d.target_pixels.toLocaleString()+' target px';
  document.getElementById('gt-info').style.color = '#10b981';
  _gtLoaded = true;
  const sel = document.getElementById('gt-label-select');
  if(d.is_multiclass && d.labels && d.labels.length > 0){
    sel.innerHTML = '';
    d.labels.forEach(lb => {
      const opt = document.createElement('option');
      opt.value = lb.id;
      opt.textContent = lb.name + ' (' + lb.pixels.toLocaleString() + ' px)';
      sel.appendChild(opt);
    });
    sel.value = String(d.selected_label || d.labels[0].id);
    sel.style.display = '';
  } else {
    sel.style.display = 'none';
    sel.innerHTML = '';
  }
  updateSaveBtn();
  if(posPoints.length>0) runDetection();
});

async function onGtLabelChange(labelId){
  spinner.style.display='block';
  const resp = await fetch('/set_gt_label',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({label_id: parseInt(labelId)})
  });
  const d = await resp.json();
  spinner.style.display='none';
  if(d.error){alert(d.error);return}
  document.getElementById('gt-info').textContent =
    d.target_pixels.toLocaleString()+' target px (' + d.label_name + ')';
  document.getElementById('gt-info').style.color = '#10b981';
  if(posPoints.length>0) runDetection();
}

// Enable GT + Run All buttons after image upload, reset label selector
document.getElementById('file-input').addEventListener('change', ()=>{
  setTimeout(()=>{
    document.getElementById('btn-gt').disabled = false;
    document.getElementById('btn-run-all').disabled = false;
    const saveBtnEl = document.getElementById('btn-save');
    if(saveBtnEl) saveBtnEl.disabled = true;  // reset until run
    const sel = document.getElementById('gt-label-select');
    sel.style.display = 'none';
    sel.innerHTML = '';
    document.getElementById('gt-info').textContent = '';
    _gtLoaded = false;
    // Clear result panels
    for(const pm of Object.values(MODEL_PANEL_MAP)){
      const imgEl = document.getElementById(pm.img);
      if(imgEl){ imgEl.style.display='none'; }
      const phEl = document.getElementById(pm.ph);
      if(phEl){ phEl.style.display='flex'; phEl.textContent='Run All Models'; }
      const metaEl = document.getElementById(pm.meta);
      if(metaEl) metaEl.textContent='';
    }
    document.getElementById('eval-row').style.display='none';
    document.getElementById('tbody-results').innerHTML='';
    if(_prChart){ _prChart.destroy(); _prChart=null; }
  }, 500);
});

// ═══════ Evaluation chart rendering ═══════
let rocChart = null, prChart = null, stabilityChart = null, fpmpChart = null;

function renderEvalCharts(ev) {
  // Legacy single-model eval — no-op since we use renderAllResults()
  return;
  if(!ev) {
    return;
  }
  updateSaveBtn();

  // ROC
  document.getElementById('eval-auc').textContent = 'AUC=' + ev.auc.toFixed(4);
  if(rocChart){ rocChart.destroy(); rocChart=null; }
  rocChart = new Chart(document.getElementById('chart-roc'), {
    type: 'line',
    data: {
      labels: ev.roc_pfa,
      datasets: [{
        label: 'ROC',
        data: ev.roc_pfa.map((x,i) => ({x: x, y: ev.roc_pd[i]})),
        borderColor: '#e41a1c', backgroundColor: 'transparent',
        borderWidth: 2, pointRadius: 0, tension: 0.1,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend:{display:false}, tooltip:{backgroundColor:'#1e2230',titleColor:'#fff',bodyColor:'#c4cbde',borderColor:'#3a3f55',borderWidth:1} },
      scales: {
        x: { type:'linear', title:{display:true,text:'PFA',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'}, min:0, max:1 },
        y: { title:{display:true,text:'PD',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'}, min:0, max:1 },
      },
    },
  });

  // PR
  document.getElementById('eval-ap').textContent = 'AP=' + ev.ap.toFixed(4);
  if(prChart){ prChart.destroy(); prChart=null; }
  prChart = new Chart(document.getElementById('chart-pr'), {
    type: 'line',
    data: {
      datasets: [{
        label: 'PR',
        data: ev.pr_recall.map((x,i) => ({x: x, y: ev.pr_precision[i]})),
        borderColor: '#377eb8', backgroundColor: 'transparent',
        borderWidth: 2, pointRadius: 0, tension: 0.1,
      }]
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend:{display:false}, tooltip:{backgroundColor:'#1e2230',titleColor:'#fff',bodyColor:'#c4cbde',borderColor:'#3a3f55',borderWidth:1} },
      scales: {
        x: { type:'linear', title:{display:true,text:'Recall',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'}, min:0, max:1 },
        y: { title:{display:true,text:'Precision',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'}, min:0, max:1 },
      },
    },
  });

  // PD @ PFA Table
  const tbody = document.getElementById('tbody-pd-pfa');
  tbody.innerHTML = '';
  Object.entries(ev.pd_at_pfa).forEach(([pfa, pd]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = '<td style="text-align:left;font-family:monospace">' + parseFloat(pfa).toExponential(0) +
      '</td><td style="text-align:right;font-weight:600;color:#4daf4a">' + pd.toFixed(4) + '</td>';
    tbody.appendChild(tr);
  });

  // Stability
  if(stabilityChart){ stabilityChart.destroy(); stabilityChart=null; }
  const tilePfa = ev.stability_tile_pfa || [];
  if(tilePfa.length > 0) {
    stabilityChart = new Chart(document.getElementById('chart-stability'), {
      type: 'line',
      data: {
        labels: tilePfa.map((_,i)=>i),
        datasets: [
          {
            label: 'Realised PFA',
            data: tilePfa,
            borderColor: '#984ea3', backgroundColor: 'transparent',
            borderWidth: 1.5, pointRadius: 2, tension: 0,
          },
          {
            label: 'Target α',
            data: tilePfa.map(()=>ev.stability_target),
            borderColor: '#fff', backgroundColor: 'transparent',
            borderWidth: 1.5, borderDash: [6,3], pointRadius: 0,
          },
        ]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend:{display:true,position:'top',labels:{color:'#a0a8c0',font:{size:9},boxWidth:12,padding:6}}, tooltip:{backgroundColor:'#1e2230',titleColor:'#fff',bodyColor:'#c4cbde',borderColor:'#3a3f55',borderWidth:1} },
        scales: {
          x: { title:{display:true,text:'Tile',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'} },
          y: { title:{display:true,text:'PFA',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'} },
        },
      },
    });
  }

  // FP per Megapixel
  if(fpmpChart){ fpmpChart.destroy(); fpmpChart=null; }
  const fpmpData = ev.fp_per_megapixel || {};
  const fpmpKeys = Object.keys(fpmpData);
  if(fpmpKeys.length > 0) {
    fpmpChart = new Chart(document.getElementById('chart-fpmp'), {
      type: 'bar',
      data: {
        labels: fpmpKeys.map(k => 'PFA=' + parseFloat(k).toExponential(0)),
        datasets: [{
          label: 'FP/MP',
          data: fpmpKeys.map(k => fpmpData[k]),
          backgroundColor: ['#ff7f00','#e6550d','#a63603'],
          borderWidth: 0,
        }]
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { legend:{display:false}, tooltip:{backgroundColor:'#1e2230',titleColor:'#fff',bodyColor:'#c4cbde',borderColor:'#3a3f55',borderWidth:1} },
        scales: {
          x: { ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'} },
          y: { title:{display:true,text:'FP / Megapixel',color:'#6b7394',font:{size:11}}, ticks:{color:'#555b72',font:{size:10}}, grid:{color:'#1e2230'} },
        },
      },
    });
  }
}

// ═══════ Save Results ═══════
function getDownloadNameFromResponse(resp, fallbackName){
  const cd = resp.headers.get('Content-Disposition') || '';
  const utf8Match = cd.match(/filename\*=UTF-8''([^;]+)/i);
  if(utf8Match && utf8Match[1]){
    try {
      return decodeURIComponent(utf8Match[1]);
    } catch(_err) {}
  }
  const plainMatch = cd.match(/filename="?([^";]+)"?/i);
  if(plainMatch && plainMatch[1]) return plainMatch[1];
  return fallbackName;
}

async function saveResults() {
  if(_singleResultDirty || !_lastSingleResultModel){
    alert('Run a single model first.');
    return;
  }

  const stamp = new Date().toISOString().slice(0,19).replace(/[T:]/g,'_').replace(/-/g,'');
  const baseName = (loadedName || 'interactive') + '_' + (_lastSingleResultModel || selModel.value || 'result') + '_' + stamp;
  const resultName = prompt('Download ZIP name:', baseName);
  if(!resultName) return;

  spinner.style.display='block';
  try {
    const resp = await fetch('/save_results', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        result_name: resultName,
        model: selModel.value,
        pos: posPoints.map(p=>[p.x,p.y]),
        neg: negPoints.map(p=>[p.x,p.y]),
      }),
    });
    if(!resp.ok){
      const d = await resp.json().catch(()=> null);
      alert((d && d.error) ? d.error : ('Save failed (' + resp.status + ')'));
      return;
    }

    const blob = await resp.blob();
    const downloadName = getDownloadNameFromResponse(resp, resultName.endsWith('.zip') ? resultName : (resultName + '.zip'));
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = downloadName;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(()=> URL.revokeObjectURL(url), 1000);
  } catch(err){
    console.error(err);
    alert('Save failed: ' + err);
  }
  finally { spinner.style.display='none'; }
}
</script>
</body>
</html>"""


# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()


    all_ids = [m.id for m in WEBAPP_MODELS]
    print(f"\n  Point-Prompted Target Detection Web App")
    print(f"  Models: {all_ids}")
    print(f"  Open http://localhost:{args.port} in your browser\n")
    app.run(host=args.host, port=args.port, debug=False)
