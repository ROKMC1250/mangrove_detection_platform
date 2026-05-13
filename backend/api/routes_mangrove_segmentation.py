"""
API routes for Mangrove Segmentation functionality.

Provides endpoints for:
- Running mangrove segmentation (returns probability map)
- Applying threshold to cached probability map
- Checking model status
"""

import os
import base64
import time
import threading

import numpy as np
from PIL import Image
import io as pyio

from fastapi import APIRouter, HTTPException, Request

from .schemas import MangroveSegmentationRequest, MangroveSegmentationThresholdRequest
from ..core.config import OUTPUTS_DIR
from ..services.model_inference import (
    run_model1_inference,
    is_model1_ready,
    get_model1_status,
    init_model1,
)
from ..services.visualization import (
    create_index_visualization,
    warp_rgb_and_mask_to_aoi,
)
from ..utils.cache import (
    bbox_to_cache_key,
    RASTER_FILE_CACHE,
    RASTER_CACHE_LOCK,
)


router = APIRouter(prefix="/api", tags=["mangrove-segmentation"])


# Cache for probability maps (keyed by segmentation_id).
# Bounded + TTL so long-running servers do not accumulate prob maps (each
# is a full-resolution float array) indefinitely.
from ..utils.cache_service import TTLCache
MANGROVE_SEG_CACHE = TTLCache(maxsize=32, ttl=2 * 3600, name="mangrove_seg")
MANGROVE_SEG_CACHE_LOCK = threading.Lock()


@router.get("/mangrove-segmentation/status")
def get_segmentation_status():
    """Get current status of the segmentation model."""
    return get_model1_status()


def _execute_segmentation_job(req: MangroveSegmentationRequest, cached_raster_path: str) -> dict:
    """Run the full segmentation pipeline."""
    print(f"MANGROVE SEG - Using raster: {cached_raster_path}")

    result = run_model1_inference(
        image_path=cached_raster_path,
        bbox=req.bbox,
        image_id=req.image_id,
        geometry=req.geometry,
        use_tta=req.use_tta,
    )

    if result is None:
        raise RuntimeError("Segmentation failed. Check server logs for details.")

    prob_map = result['prob_map']
    min_val = result['min_val']
    max_val = result['max_val']

    from ..services.gpu_compute import rgb_mask_to_base64_gpu

    prob_rgb, _actual_min, _actual_max = create_index_visualization(
        prob_map, cmap='jet', vmin=min_val, vmax=max_val
    )

    min_lon, min_lat, max_lon, max_lat = req.bbox
    finite_mask = np.isfinite(prob_map)

    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        prob_rgb, finite_mask,
        result['transform'], result['crs'],
        (min_lon, min_lat, max_lon, max_lat),
        scale_m=10, geometry=req.geometry
    )

    overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)

    preview_img = Image.fromarray(prob_rgb, mode='RGB')
    preview_img.thumbnail((256, 256), Image.LANCZOS)
    preview_buf = pyio.BytesIO()
    preview_img.save(preview_buf, format='PNG')
    preview_b64 = f"data:image/png;base64,{base64.b64encode(preview_buf.getvalue()).decode('utf-8')}"

    segmentation_id = f"mangrove-seg-{int(time.time() * 1000)}"
    with MANGROVE_SEG_CACHE_LOCK:
        MANGROVE_SEG_CACHE[segmentation_id] = {
            'prob_map': prob_map,
            'transform': result['transform'],
            'crs': result['crs'],
            'bbox': req.bbox,
            'geometry': req.geometry,
            'min_val': min_val,
            'max_val': max_val,
        }

    response = {
        'segmentation_id': segmentation_id,
        'name': 'Mangrove Segmentation',
        'min_val': min_val,
        'max_val': max_val,
        'preview_url': preview_b64,
        'overlay_url': overlay_url,
        'colormap': {
            'name': 'jet',
            'min_val': min_val,
            'max_val': max_val,
            'label': 'Mangrove Probability',
        },
        'overlay_meta': {
            'width': int(aoi_w),
            'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        },
    }

    print(f"MANGROVE SEG - Complete. ID: {segmentation_id}, prob range: [{min_val:.4f}, {max_val:.4f}]")
    return response


@router.post("/mangrove-segmentation/run")
def run_segmentation(req: MangroveSegmentationRequest, request: Request):
    """Run mangrove segmentation synchronously and return the result.

    Concurrency is handled by the session gate (only one active user), so no
    request-level queue is needed here.
    """
    print(f"MANGROVE SEG - Running on {req.image_id}")

    if not is_model1_ready():
        if not init_model1():
            status = get_model1_status()
            raise HTTPException(
                status_code=503,
                detail=f"Segmentation model not available: {status.get('error', 'Unknown error')}"
            )

    cache_key = bbox_to_cache_key(req.image_id, req.bbox)
    with RASTER_CACHE_LOCK:
        cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

    if not cached_raster_path or not os.path.exists(cached_raster_path):
        raise HTTPException(
            status_code=400,
            detail="Image not processed yet. Please process the image first."
        )

    try:
        return _execute_segmentation_job(req, cached_raster_path)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mangrove-segmentation/apply-threshold")
def apply_segmentation_threshold(req: MangroveSegmentationThresholdRequest):
    """
    Apply a new threshold range to cached probability map.
    Returns binary mask overlay.
    """
    try:
        print(f"MANGROVE SEG - Applying threshold {req.min_threshold:.3f}-{req.max_threshold:.3f} to {req.segmentation_id}")

        with MANGROVE_SEG_CACHE_LOCK:
            cached = MANGROVE_SEG_CACHE.get(req.segmentation_id)

        if not cached:
            raise HTTPException(
                status_code=400,
                detail=f"Segmentation result not found: {req.segmentation_id}"
            )

        prob_map = cached['prob_map']

        # Apply threshold range
        binary_mask = (prob_map >= req.min_threshold) & (prob_map <= req.max_threshold)

        # Create mask overlay (red for mangrove)
        mask_rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        mask_rgb[binary_mask, 0] = 255  # Red channel

        # Warp to AOI
        min_lon, min_lat, max_lon, max_lat = req.bbox

        aoi_binary, aoi_mask_binary, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            mask_rgb,
            binary_mask,
            cached['transform'],
            cached['crs'],
            (min_lon, min_lat, max_lon, max_lat),
            scale_m=10,
            geometry=cached.get('geometry'),
        )

        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        mask_overlay_url = rgb_mask_to_base64_gpu(aoi_binary, aoi_mask_binary)

        # Preview
        mask_rgba = np.zeros((*mask_rgb.shape[:2], 4), dtype=np.uint8)
        mask_rgba[:, :, :3] = mask_rgb
        mask_rgba[:, :, 3] = np.where(binary_mask, 255, 0).astype(np.uint8)
        mask_preview = Image.fromarray(mask_rgba, mode='RGBA')
        mask_preview.thumbnail((256, 256), Image.LANCZOS)
        mask_buf = pyio.BytesIO()
        mask_preview.save(mask_buf, format='PNG')
        mask_preview_b64 = f"data:image/png;base64,{base64.b64encode(mask_buf.getvalue()).decode('utf-8')}"

        # Persist the freshly computed binary mask so change-detection can
        # use it without re-running threshold logic.
        with MANGROVE_SEG_CACHE_LOCK:
            cached['binary_mask'] = binary_mask
            cached['last_min_threshold'] = float(req.min_threshold)
            cached['last_max_threshold'] = float(req.max_threshold)
            MANGROVE_SEG_CACHE[req.segmentation_id] = cached

        # Statistics
        n_detected = int(np.sum(binary_mask))
        total_pixels = int(binary_mask.size)
        detection_percentage = round(100 * n_detected / total_pixels, 2)

        return {
            'segmentation_id': req.segmentation_id,
            'min_threshold': req.min_threshold,
            'max_threshold': req.max_threshold,
            'detected_pixels': n_detected,
            'total_pixels': total_pixels,
            'detection_percentage': detection_percentage,
            'mask_result': {
                'name': f'Mangrove Mask ({req.min_threshold:.3f} - {req.max_threshold:.3f})',
                'preview_url': mask_preview_b64,
                'overlay_url': mask_overlay_url,
                'overlay_meta': {
                    'width': int(aoi_w),
                    'height': int(aoi_h),
                    'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
                }
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"MANGROVE SEG - Threshold error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
