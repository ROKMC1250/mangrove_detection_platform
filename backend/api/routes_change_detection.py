"""
API routes for Change Detection.

Single endpoint:
  POST /api/change-detection/run  → compares the binary masks of two
  prior analyses (target detection, mangrove segmentation, SAM3, or
  spectral threshold) and returns a single overlay with green = gained
  (B-only) and red = lost (A-only). Pixels with no change are fully
  transparent — the same convention every other analysis uses.

Cloud-mode (GEE results, geo-referenced) and local-mode (uploaded /
local-td/sa results, pixel-aligned, no CRS) are both supported. The
distinction is driven by whether the cached inputs carry a transform/CRS.
"""

import base64
import io as pyio
import threading
import time

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, Request

from .schemas import ChangeDetectionRequest, ChangeDetectionGroup
from ..services.change_detection import (
    compute_signed_change,
    _extract_binary_mask_and_meta,
)
from ..services.visualization import warp_rgb_and_mask_to_aoi
from ..services.gpu_compute import rgb_mask_to_base64_gpu
from ..utils.cache_service import TTLCache

from .routes_mangrove_segmentation import (
    MANGROVE_SEG_CACHE,
    MANGROVE_SEG_CACHE_LOCK,
)
from .routes_target_detection import (
    TARGET_DETECTION_CACHE,
    TARGET_DETECTION_CACHE_LOCK,
)
from .routes_sam3 import (
    SAM3_MASK_CACHE,
    SAM3_MASK_CACHE_LOCK,
)
from .routes_local import (
    _LOCAL_TD_CACHE,
    _LOCAL_TD_CACHE_LOCK,
    _LOCAL_INDEX_CACHE,
    _LOCAL_INDEX_CACHE_LOCK,
)
from ..utils.cache import INDEX_DATA_CACHE, INDEX_CACHE_LOCK


router = APIRouter(prefix="/api", tags=["change-detection"])


CHANGE_DETECTION_CACHE = TTLCache(maxsize=32, ttl=2 * 3600, name="change_detection")
CHANGE_DETECTION_CACHE_LOCK = threading.Lock()


def _lookup_result(result_id: str) -> dict:
    """Find a cached analysis result by id and verify it has a binary mask.

    Searches every analysis cache in turn. Raises:
      - 404 if the id doesn't exist anywhere (likely TTL expiry)
      - 400 if it exists but has no binary mask yet (user must apply a
        threshold first)
    """
    entry = None
    with MANGROVE_SEG_CACHE_LOCK:
        entry = MANGROVE_SEG_CACHE.get(result_id)
    if entry is None:
        with TARGET_DETECTION_CACHE_LOCK:
            entry = TARGET_DETECTION_CACHE.get(result_id)
    if entry is None:
        with SAM3_MASK_CACHE_LOCK:
            entry = SAM3_MASK_CACHE.get(result_id)
    if entry is None:
        with _LOCAL_TD_CACHE_LOCK:
            entry = _LOCAL_TD_CACHE.get(result_id)
    if entry is None:
        with _LOCAL_INDEX_CACHE_LOCK:
            entry = _LOCAL_INDEX_CACHE.get(result_id)
    if entry is None:
        with INDEX_CACHE_LOCK:
            entry = INDEX_DATA_CACHE.get(result_id)

    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Analysis result '{result_id}' not found in cache. It may "
                "have expired (2-hour TTL) or never been computed — re-run "
                "the analysis and retry."
            ),
        )

    # Some legacy entries are bare ndarrays (path-keyed local index cache).
    # Treat those as "no binary mask" — change-detection requires a dict
    # entry with a `binary_mask`/`mask` field.
    if not isinstance(entry, dict) or (
        entry.get('binary_mask') is None and entry.get('mask') is None
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Analysis result '{result_id}' has no binary mask. Apply "
                "a threshold first to create the mask before running "
                "change detection."
            ),
        )

    return entry


def _preview_png_b64(rgb: np.ndarray, mask: np.ndarray, max_dim: int = 256) -> str:
    """Small thumbnail as a base64 data URL. `mask` controls alpha."""
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = np.where(mask, 220, 0).astype(np.uint8)
    img = Image.fromarray(rgba, mode='RGBA')
    img.thumbnail((max_dim, max_dim), Image.LANCZOS)
    buf = pyio.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _render_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    transform,
    crs,
    bbox,
    geometry,
):
    """Render either a geo-warped or native-resolution overlay.

    Returns (overlay_url, overlay_meta_dict).
    """
    is_local_mode = transform is None and crs is None

    if is_local_mode:
        overlay_url = rgb_mask_to_base64_gpu(rgb, mask)
        h, w = rgb.shape[:2]
        meta = {
            'width': int(w),
            'height': int(h),
            'mode': 'local',
        }
        return overlay_url, meta

    if bbox is None:
        raise RuntimeError(
            "No bbox available for change-detection overlay. Pass `bbox` "
            "in the request if neither result has it cached."
        )
    min_lon, min_lat, max_lon, max_lat = bbox

    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        rgb, mask, transform, crs,
        (min_lon, min_lat, max_lon, max_lat),
        scale_m=10, geometry=geometry,
    )
    overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)
    meta = {
        'width': int(aoi_w),
        'height': int(aoi_h),
        'mode': 'geo',
        'bounds': [
            float(min_lat), float(min_lon),
            float(max_lat), float(max_lon),
        ],
    }
    return overlay_url, meta


def _compose_group_to_result(side: str, group: ChangeDetectionGroup) -> dict:
    """Look up each component by id, extract its binary mask, and combine
    left-to-right with the supplied operators. Returns a synthetic
    'analysis result' dict that `compute_signed_change` can consume —
    `binary_mask` plus the geometry/transform/crs copied from the first
    component (all components in a group come from the same slot/AOI, so
    they are expected to share that frame).
    """
    if not group.items or len(group.items) < 2:
        raise HTTPException(
            status_code=400,
            detail=f"Mask group on side {side} needs at least 2 components.",
        )
    if len(group.operators) < len(group.items) - 1:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Mask group on side {side} has {len(group.items)} items but "
                f"only {len(group.operators)} operators "
                f"(expected {len(group.items) - 1})."
            ),
        )

    entries = [_lookup_result(cid) for cid in group.items]
    first_mask, first_meta = _extract_binary_mask_and_meta(entries[0])
    composite = first_mask.copy()

    for k in range(1, len(entries)):
        mk, _meta_k = _extract_binary_mask_and_meta(entries[k])
        if mk.shape != composite.shape:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Shape mismatch inside mask group on side {side}: "
                    f"component {k} has shape {mk.shape}, expected "
                    f"{composite.shape}. All components must come from the "
                    "same AOI / resolution."
                ),
            )
        op = group.operators[k - 1]
        if op == 'inc':
            composite &= mk
        elif op == 'exc':
            composite &= ~mk
        elif op == 'add':
            composite |= mk
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown mask operator '{op}' on side {side}. "
                       "Expected one of: inc, exc, add.",
            )

    return {
        'binary_mask': composite,
        'transform': first_meta['transform'],
        'crs': first_meta['crs'],
        'bbox': entries[0].get('bbox'),
        'geometry': entries[0].get('geometry'),
    }


def _resolve_side(side: str, single_id, group) -> dict:
    if single_id and group:
        raise HTTPException(
            status_code=400,
            detail=f"Side {side}: specify result_{side}_id or result_{side}_group, not both.",
        )
    if group is not None:
        return _compose_group_to_result(side, group)
    if single_id:
        return _lookup_result(single_id)
    raise HTTPException(
        status_code=400,
        detail=f"Side {side}: provide result_{side}_id or result_{side}_group.",
    )


def _describe_source(single_id, group) -> str:
    """Human/diagnostic label for source_a_id / source_b_id in the response."""
    if group is not None:
        sep = {'inc': ' ∩ ', 'exc': ' − ', 'add': ' ∪ '}
        parts = [group.items[0]]
        for k in range(1, len(group.items)):
            op = group.operators[k - 1] if k - 1 < len(group.operators) else 'inc'
            parts.append(sep.get(op, ' ? ') + group.items[k])
        return 'group(' + ''.join(parts) + ')'
    return single_id or ''


def _execute_change_detection_job(req: ChangeDetectionRequest) -> dict:
    """Compare two binary masks and render the gained/lost overlay."""
    result_a = _resolve_side('a', req.result_a_id, req.result_a_group)
    result_b = _resolve_side('b', req.result_b_id, req.result_b_group)

    try:
        diff = compute_signed_change(result_a, result_b)
    except ValueError as e:
        raise RuntimeError(f"Change detection input error: {e}")

    gained = diff['gained_mask']
    lost = diff['lost_mask']
    change_mask = diff['change_mask']

    # Two-color RGB: green for gained, red for lost. Alpha comes from the
    # change mask, so unchanged pixels stay fully transparent.
    h, w = change_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[gained, 1] = 220   # green
    rgb[lost, 0] = 220     # red

    bbox = diff['bbox'] or req.bbox
    overlay_url, overlay_meta = _render_overlay(
        rgb, change_mask,
        diff['transform'], diff['crs'],
        bbox, diff.get('geometry'),
    )

    preview_url = _preview_png_b64(rgb, change_mask)

    change_detection_id = f"chg-{int(time.time() * 1000)}"
    with CHANGE_DETECTION_CACHE_LOCK:
        CHANGE_DETECTION_CACHE[change_detection_id] = {
            'gained_mask': gained,
            'lost_mask': lost,
            'transform': diff['transform'],
            'crs': diff['crs'],
            'bbox': bbox,
            'geometry': diff.get('geometry'),
            'source_types': diff['source_types'],
            'stats': diff['stats'],
        }

    return {
        'change_detection_id': change_detection_id,
        'name': 'Change Detection',
        'source_a_type': diff['source_types'][0],
        'source_b_type': diff['source_types'][1],
        'source_a_id': _describe_source(req.result_a_id, req.result_a_group),
        'source_b_id': _describe_source(req.result_b_id, req.result_b_group),
        'stats': diff['stats'],
        'detection_result': {
            'name': 'Change Mask (gained vs lost)',
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'overlay_meta': overlay_meta,
            'type': 'change_mask',
        },
    }


@router.post("/change-detection/run")
def run_change_detection(req: ChangeDetectionRequest, request: Request):
    """Compare two binary-mask analyses and return a gained/lost overlay."""
    a_desc = _describe_source(req.result_a_id, req.result_a_group)
    b_desc = _describe_source(req.result_b_id, req.result_b_group)
    print(f"CHANGE DETECTION - Running: A={a_desc}, B={b_desc}")

    try:
        return _execute_change_detection_job(req)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
