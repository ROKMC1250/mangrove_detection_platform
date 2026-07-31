"""
API routes for Sentinel-1 flood segmentation (UNet++/ResNet34).

Mirrors the mangrove segmentation surface (run / apply-threshold / status)
so the frontend can reuse the same threshold/overlay UI patterns. Adds an
"eraser" pair (remove-blob / reset-exclusions) that lets the user click a
false-positive water blob on the map to drop it from the binary mask via
connected-component labelling.
"""

import base64
import io as pyio
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np
import pyproj
import rasterio
from PIL import Image
from scipy import ndimage

from fastapi import APIRouter, HTTPException

from .schemas import (
    FloodSegmentationRequest,
    FloodSegmentationThresholdRequest,
    FloodEraseRegionRequest,
    FloodResetExclusionsRequest,
)
from ..services.flood_inference import (
    get_flood_model_status,
    init_flood_model,
    is_flood_model_ready,
    run_flood_inference,
)
from ..services.visualization import (
    create_index_visualization,
    warp_rgb_and_mask_to_aoi,
)
from ..utils.cache import RASTER_CACHE_LOCK, RASTER_FILE_CACHE, bbox_to_cache_key
from ..utils.cache_service import TTLCache


router = APIRouter(prefix="/api", tags=["flood-segmentation"])


# Cache full-scene probability maps. These are large float32 arrays
# (e.g. 16k x 16k ~= 1 GB), so keep the size tight and let TTL evict.
FLOOD_SEG_CACHE = TTLCache(maxsize=16, ttl=2 * 3600, name="flood_seg")
FLOOD_SEG_CACHE_LOCK = threading.Lock()

# 8-connectivity for water-blob labelling: water bodies often touch
# diagonally through single pixels, and the user clicking "this island"
# expects diagonal neighbours to be part of the same blob.
_CC_STRUCTURE = np.ones((3, 3), dtype=bool)


@router.get("/flood-segmentation/status")
def flood_status():
    return get_flood_model_status()


def _latlng_to_source_pixel(raster_path: str, lat: float, lng: float) -> Tuple[int, int]:
    """Convert EPSG:4326 lat/lng to (row, col) in the source raster's pixel grid.

    Mirrors ``backend/api/routes_sam3.py:_latlon_to_pixel`` but returns (row, col)
    in numpy convention rather than (col, row).
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", raster_crs, always_xy=True
    )
    x_proj, y_proj = transformer.transform(lng, lat)
    row, col = rasterio.transform.rowcol(raster_transform, x_proj, y_proj)
    return int(row), int(col)


def _build_binary_mask_response(
    cached: dict,
    binary_mask: np.ndarray,
    segmentation_id: str,
    name_label: str,
    extra: Optional[dict] = None,
) -> dict:
    """Render a binary watermask overlay (warped to AOI) and return the standard
    apply-threshold-shaped response. Used by apply-threshold, remove-blob, and
    reset-exclusions so the frontend can use one overlay-replace code path.
    """
    from ..services.gpu_compute import rgb_mask_to_base64_gpu

    bbox = cached['bbox']
    min_lon, min_lat, max_lon, max_lat = bbox

    # Blue overlay for water.
    mask_rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    mask_rgb[binary_mask, 2] = 255

    aoi_binary, aoi_mask_binary, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        mask_rgb,
        binary_mask,
        cached['transform'],
        cached['crs'],
        (min_lon, min_lat, max_lon, max_lat),
        scale_m=10,
        geometry=cached.get('geometry'),
    )
    mask_overlay_url = rgb_mask_to_base64_gpu(aoi_binary, aoi_mask_binary)

    mask_rgba = np.zeros((*mask_rgb.shape[:2], 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = mask_rgb
    mask_rgba[:, :, 3] = np.where(binary_mask, 255, 0).astype(np.uint8)
    preview = Image.fromarray(mask_rgba, mode='RGBA')
    preview.thumbnail((256, 256), Image.LANCZOS)
    preview_buf = pyio.BytesIO()
    preview.save(preview_buf, format='PNG')
    preview_b64 = (
        f"data:image/png;base64,{base64.b64encode(preview_buf.getvalue()).decode('utf-8')}"
    )

    n_detected = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    detection_percentage = round(100 * n_detected / total_pixels, 2) if total_pixels else 0.0

    body = {
        'segmentation_id': segmentation_id,
        'detected_pixels': n_detected,
        'total_pixels': total_pixels,
        'detection_percentage': detection_percentage,
        'excluded_blobs': int(cached.get('excluded_blobs', 0)),
        'total_excluded_pixels': int(
            cached['exclusion_mask'].sum() if cached.get('exclusion_mask') is not None else 0
        ),
        'mask_result': {
            'name': name_label,
            'preview_url': preview_b64,
            'overlay_url': mask_overlay_url,
            'overlay_meta': {
                'width': int(aoi_w),
                'height': int(aoi_h),
                'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)],
            },
        },
    }
    if extra:
        body.update(extra)
    return body


def _execute_flood_job(req: FloodSegmentationRequest) -> dict:
    result = run_flood_inference(
        image_id=req.image_id,
        bbox=req.bbox,
        geometry=req.geometry,
    )
    if result is None:
        raise RuntimeError("Flood segmentation failed. Check server logs.")

    prob_map = result['prob_map']
    min_val = result['min_val']
    max_val = result['max_val']

    from ..services.gpu_compute import rgb_mask_to_base64_gpu

    prob_rgb, _amin, _amax = create_index_visualization(
        prob_map, cmap='jet', vmin=min_val, vmax=max_val
    )

    min_lon, min_lat, max_lon, max_lat = req.bbox
    data_mask = result.get('data_mask')
    finite_mask = np.isfinite(prob_map)
    valid_mask = (data_mask & finite_mask) if data_mask is not None else finite_mask

    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        prob_rgb, valid_mask,
        result['transform'], result['crs'],
        (min_lon, min_lat, max_lon, max_lat),
        scale_m=10, geometry=req.geometry,
    )

    overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)

    preview_img = Image.fromarray(prob_rgb, mode='RGB')
    preview_img.thumbnail((256, 256), Image.LANCZOS)
    preview_buf = pyio.BytesIO()
    preview_img.save(preview_buf, format='PNG')
    preview_b64 = f"data:image/png;base64,{base64.b64encode(preview_buf.getvalue()).decode('utf-8')}"

    segmentation_id = f"flood-seg-{int(time.time() * 1000)}"
    with FLOOD_SEG_CACHE_LOCK:
        FLOOD_SEG_CACHE[segmentation_id] = {
            'prob_map': prob_map,
            'data_mask': data_mask,
            'transform': result['transform'],
            'crs': result['crs'],
            'bbox': req.bbox,
            'geometry': req.geometry,
            'image_id': req.image_id,
            'min_val': min_val,
            'max_val': max_val,
            'prob_path': result.get('prob_path'),
            'mask_path': result.get('mask_path'),
            # Eraser state — starts empty, accumulated by /remove-blob.
            'exclusion_mask': np.zeros(prob_map.shape, dtype=bool),
            'excluded_blobs': 0,
        }

    return {
        'segmentation_id': segmentation_id,
        'name': 'Flood Segmentation',
        'min_val': min_val,
        'max_val': max_val,
        'preview_url': preview_b64,
        'overlay_url': overlay_url,
        'colormap': {
            'name': 'jet',
            'min_val': min_val,
            'max_val': max_val,
            'label': 'Water Probability',
        },
        'overlay_meta': {
            'width': int(aoi_w),
            'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)],
        },
        'prob_path': result.get('prob_path'),
        'mask_path': result.get('mask_path'),
        'stats': result.get('stats', {}),
    }


@router.post("/flood-segmentation/run")
def run_flood_segmentation(req: FloodSegmentationRequest):
    """Run flood segmentation on a Sentinel-1 image."""
    print(f"FLOOD SEG - Running on {req.image_id}")

    if not is_flood_model_ready():
        if not init_flood_model():
            status = get_flood_model_status()
            raise HTTPException(
                status_code=503,
                detail=f"Flood segmentation model not available: {status.get('error', 'Unknown error')}",
            )

    try:
        return _execute_flood_job(req)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/flood-segmentation/apply-threshold")
def apply_flood_threshold(req: FloodSegmentationThresholdRequest):
    """Apply a new threshold range to a cached flood probability map.

    Honours any previously-recorded user exclusions (via /remove-blob) so
    pixels the user has erased do NOT come back when the threshold is
    re-applied. Use /reset-exclusions to clear them.
    """
    try:
        print(
            f"FLOOD SEG - Threshold {req.min_threshold:.3f}-{req.max_threshold:.3f} "
            f"on {req.segmentation_id}"
        )

        with FLOOD_SEG_CACHE_LOCK:
            cached = FLOOD_SEG_CACHE.get(req.segmentation_id)
        if not cached:
            raise HTTPException(
                status_code=400,
                detail=f"Flood segmentation result not found: {req.segmentation_id}",
            )

        prob_map = cached['prob_map']
        data_mask = cached.get('data_mask')
        exclusion_mask = cached.get('exclusion_mask')

        binary_mask = (prob_map >= req.min_threshold) & (prob_map <= req.max_threshold)
        if data_mask is not None:
            binary_mask = binary_mask & data_mask
        if exclusion_mask is not None:
            binary_mask = binary_mask & ~exclusion_mask

        with FLOOD_SEG_CACHE_LOCK:
            cached['binary_mask'] = binary_mask
            cached['last_min_threshold'] = float(req.min_threshold)
            cached['last_max_threshold'] = float(req.max_threshold)
            FLOOD_SEG_CACHE[req.segmentation_id] = cached

        return _build_binary_mask_response(
            cached,
            binary_mask,
            req.segmentation_id,
            f'Flood Mask ({req.min_threshold:.3f} - {req.max_threshold:.3f})',
            extra={
                'min_threshold': req.min_threshold,
                'max_threshold': req.max_threshold,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"FLOOD SEG - Threshold error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


def _get_cached_raster_path(image_id: str, bbox) -> Optional[str]:
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        path = RASTER_FILE_CACHE.get(cache_key)
    if path and os.path.exists(path):
        return path
    return None


def _latlng_box_to_pixel_bounds(
    raster_path: str,
    lat_min: float,
    lng_min: float,
    lat_max: float,
    lng_max: float,
) -> Tuple[int, int, int, int]:
    """Convert a lat/lng box to a conservative source pixel bbox (row_min,
    col_min, row_max, col_max). All four corners are projected and the outer
    bbox is taken — a lat/lng rectangle is not exactly a rectangle in UTM,
    so we enlarge slightly to be sure we cover the user-drawn region.
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
    transformer = pyproj.Transformer.from_crs("EPSG:4326", raster_crs, always_xy=True)
    corners_lnglat = [
        (lng_min, lat_min),
        (lng_min, lat_max),
        (lng_max, lat_min),
        (lng_max, lat_max),
    ]
    rows: list = []
    cols: list = []
    for lng, lat in corners_lnglat:
        x_proj, y_proj = transformer.transform(lng, lat)
        r, c = rasterio.transform.rowcol(raster_transform, x_proj, y_proj)
        rows.append(int(r))
        cols.append(int(c))
    return min(rows), min(cols), max(rows), max(cols)


@router.post("/flood-segmentation/erase-region")
def erase_flood_region(req: FloodEraseRegionRequest):
    """Erase every connected water blob that has at least one pixel inside
    the user-dragged lat/lng box.

    Pipeline:
      1. Look up cached segmentation.
      2. Resolve source raster path (provides CRS/transform).
      3. Convert the four lat/lng corners to source pixel bounds (outer bbox).
      4. Clip to image extent. Build effective mask = binary & ~exclusion.
      5. `scipy.ndimage.label` (8-connectivity) on the effective mask.
      6. Read the labels inside the pixel bbox; collect unique non-zero
         labels — those are the components to remove (fully, even pixels
         that extend outside the box).
      7. Add every pixel belonging to those labels into exclusion_mask.
      8. Re-render and return the updated binary overlay.
    """
    try:
        with FLOOD_SEG_CACHE_LOCK:
            cached = FLOOD_SEG_CACHE.get(req.segmentation_id)
        if not cached:
            raise HTTPException(
                status_code=400,
                detail=f"Flood segmentation result not found: {req.segmentation_id}",
            )

        image_id = cached.get('image_id')
        if not image_id:
            raise HTTPException(
                status_code=400,
                detail="Cached segmentation is missing image_id (re-run flood segmentation).",
            )
        raster_path = _get_cached_raster_path(image_id, cached['bbox'])
        if not raster_path:
            raise HTTPException(
                status_code=400,
                detail="Source raster no longer cached. Re-run flood segmentation first.",
            )

        binary_mask = cached.get('binary_mask')
        if binary_mask is None:
            raise HTTPException(
                status_code=400,
                detail="Apply a threshold first before erasing.",
            )

        # Normalize box ordering (UI may send corners in any order).
        lat_lo = min(req.lat_min, req.lat_max)
        lat_hi = max(req.lat_min, req.lat_max)
        lng_lo = min(req.lng_min, req.lng_max)
        lng_hi = max(req.lng_min, req.lng_max)

        row_min, col_min, row_max, col_max = _latlng_box_to_pixel_bounds(
            raster_path, lat_lo, lng_lo, lat_hi, lng_hi
        )
        H, W = binary_mask.shape
        row_min = max(0, row_min)
        col_min = max(0, col_min)
        row_max = min(H - 1, row_max)
        col_max = min(W - 1, col_max)

        if row_max < row_min or col_max < col_min:
            raise HTTPException(
                status_code=400,
                detail="Drag rectangle is outside the image extent.",
            )

        exclusion_mask = cached.get('exclusion_mask')
        if exclusion_mask is None:
            exclusion_mask = np.zeros_like(binary_mask, dtype=bool)

        effective = binary_mask & ~exclusion_mask
        if not effective.any():
            raise HTTPException(
                status_code=400,
                detail="Nothing left to erase.",
            )

        labels, _num_labels = ndimage.label(effective, structure=_CC_STRUCTURE)
        box_labels = labels[row_min:row_max + 1, col_min:col_max + 1]
        unique_labels = np.unique(box_labels)
        unique_labels = unique_labels[unique_labels > 0]
        if unique_labels.size == 0:
            raise HTTPException(
                status_code=400,
                detail="No water blobs in the selected region.",
            )

        remove_mask = np.isin(labels, unique_labels)
        removed_pixels = int(remove_mask.sum())
        removed_blobs = int(unique_labels.size)

        exclusion_mask = exclusion_mask | remove_mask
        new_effective = binary_mask & ~exclusion_mask

        with FLOOD_SEG_CACHE_LOCK:
            cached['exclusion_mask'] = exclusion_mask
            cached['excluded_blobs'] = int(cached.get('excluded_blobs', 0)) + removed_blobs
            FLOOD_SEG_CACHE[req.segmentation_id] = cached

        print(
            f"FLOOD SEG - Erased {removed_blobs} blob(s) ({removed_pixels} px) in "
            f"region rows={row_min}..{row_max} cols={col_min}..{col_max}; "
            f"total excluded blobs={cached['excluded_blobs']}"
        )

        min_thr = cached.get('last_min_threshold')
        max_thr = cached.get('last_max_threshold')
        if min_thr is not None and max_thr is not None:
            name_label = f'Flood Mask ({min_thr:.3f} - {max_thr:.3f})'
        else:
            name_label = 'Flood Mask'

        return _build_binary_mask_response(
            cached,
            new_effective,
            req.segmentation_id,
            name_label,
            extra={
                'removed_pixels': removed_pixels,
                'removed_blobs': removed_blobs,
            },
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"FLOOD SEG - erase-region error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/flood-segmentation/reset-exclusions")
def reset_flood_exclusions(req: FloodResetExclusionsRequest):
    """Clear all user-applied blob exclusions for this segmentation_id."""
    try:
        with FLOOD_SEG_CACHE_LOCK:
            cached = FLOOD_SEG_CACHE.get(req.segmentation_id)
        if not cached:
            raise HTTPException(
                status_code=400,
                detail=f"Flood segmentation result not found: {req.segmentation_id}",
            )

        binary_mask = cached.get('binary_mask')
        if binary_mask is None:
            raise HTTPException(
                status_code=400,
                detail="No threshold has been applied yet — nothing to reset.",
            )

        with FLOOD_SEG_CACHE_LOCK:
            cached['exclusion_mask'] = np.zeros_like(binary_mask, dtype=bool)
            cached['excluded_blobs'] = 0
            FLOOD_SEG_CACHE[req.segmentation_id] = cached

        min_thr = cached.get('last_min_threshold')
        max_thr = cached.get('last_max_threshold')
        if min_thr is not None and max_thr is not None:
            name_label = f'Flood Mask ({min_thr:.3f} - {max_thr:.3f})'
        else:
            name_label = 'Flood Mask'

        return _build_binary_mask_response(
            cached,
            binary_mask,
            req.segmentation_id,
            name_label,
        )

    except HTTPException:
        raise
    except Exception as exc:
        print(f"FLOOD SEG - reset-exclusions error: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))
