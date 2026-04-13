"""
API routes for SAM2 (Segment Anything Model 2) segmentation.

Provides endpoints for:
- Checking SAM2 model status
- Encoding images for SAM2
- Predicting segmentation masks with point prompts
- Managing saved masks
"""

import os
import time
import base64
import threading
from typing import Dict

import numpy as np
from PIL import Image
import io as pyio
import rasterio
import pyproj

from fastapi import APIRouter, HTTPException

from .schemas import (
    SAM2EncodeRequest,
    SAM2PredictRequest,
    SAM2SaveMaskRequest,
)
from ..core.config import OUTPUTS_DIR
from ..services.sam2_service import (
    init_sam2,
    is_sam2_ready,
    get_sam2_status,
    encode_image,
    predict_mask,
)
from ..services.visualization import (
    warp_rgb_and_mask_to_aoi,
)
from ..utils.cache import (
    bbox_to_cache_key,
    RASTER_FILE_CACHE,
    RASTER_CACHE_LOCK,
)


router = APIRouter(prefix="/api", tags=["sam2"])

# Cache for SAM2 masks
SAM2_MASK_CACHE: Dict = {}
SAM2_MASK_CACHE_LOCK = threading.Lock()


def _get_cached_raster(image_id: str, bbox: list):
    """Get cached raster path, raising HTTP 400 if not found."""
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

    if not cached_raster_path or not os.path.exists(cached_raster_path):
        raise HTTPException(
            status_code=400,
            detail="No cached raster data found. Please process the image first."
        )
    return cache_key, cached_raster_path


def _extract_rgb_uint8(raster_path: str) -> np.ndarray:
    """Extract RGB bands from raster and convert to uint8 for SAM2.

    Reads B4 (Red), B3 (Green), B2 (Blue) from the raster file.
    Applies 2-98 percentile stretch to uint8 range.

    Returns:
        RGB uint8 array (H, W, 3)
    """
    with rasterio.open(raster_path) as src:
        # Sentinel-2 bands in raster: B2=band1, B3=band2, B4=band3 (0-indexed: 0,1,2)
        # Read as (bands, H, W)
        blue = src.read(1).astype(np.float32)   # B2
        green = src.read(2).astype(np.float32)  # B3
        red = src.read(3).astype(np.float32)    # B4

    rgb = np.stack([red, green, blue], axis=-1)  # (H, W, 3)

    # Percentile stretch to uint8
    valid = rgb[np.isfinite(rgb) & (rgb > 0)]
    if len(valid) > 0:
        p2 = np.percentile(valid, 2)
        p98 = np.percentile(valid, 98)
        if p98 > p2:
            rgb = (rgb - p2) / (p98 - p2)
        else:
            rgb = rgb / (rgb.max() + 1e-8)
    else:
        rgb = rgb / (rgb.max() + 1e-8)

    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb


def _latlon_to_pixel(raster_path: str, lat: float, lng: float):
    """Convert lat/lng to pixel coordinates (col, row) in the raster.

    Returns:
        (x, y) tuple where x=col, y=row (SAM2 convention)
    """
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform

    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", raster_crs, always_xy=True
    )
    x_proj, y_proj = transformer.transform(lng, lat)
    row, col = rasterio.transform.rowcol(raster_transform, x_proj, y_proj)
    return int(col), int(row)  # SAM2 uses (x, y) = (col, row)


def _mask_to_overlay_base64(mask: np.ndarray, raster_path: str, bbox: list) -> tuple:
    """Convert a boolean mask to a base64 overlay PNG.

    Creates a green semi-transparent overlay where mask is True.

    Returns:
        (overlay_url, preview_url, width, height) tuple
    """
    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        raster_crs = src.crs

    # Ensure mask is boolean
    mask = mask.astype(bool)

    # Create green overlay RGB
    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask, 0] = 255  # Red (magenta)
    mask_rgb[mask, 2] = 200  # Blue (magenta)

    min_lon, min_lat, max_lon, max_lat = bbox

    # Warp to AOI bounds
    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        mask_rgb,
        mask,
        raster_transform,
        raster_crs,
        (min_lon, min_lat, max_lon, max_lat),
        scale_m=10
    )

    # Generate overlay as base64
    try:
        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)
    except ImportError:
        # Fallback: manual RGBA composition
        rgba = np.zeros((*aoi_rgb.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = aoi_rgb
        rgba[:, :, 3] = (aoi_mask * 180).astype(np.uint8)  # Semi-transparent
        img = Image.fromarray(rgba, mode='RGBA')
        buf = pyio.BytesIO()
        img.save(buf, format='PNG')
        overlay_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    # Generate preview thumbnail
    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = mask_rgb
    mask_rgba[mask, 3] = 200
    preview_img = Image.fromarray(mask_rgba, mode='RGBA')
    preview_img.thumbnail((256, 256), Image.LANCZOS)
    preview_buf = pyio.BytesIO()
    preview_img.save(preview_buf, format='PNG')
    preview_url = f"data:image/png;base64,{base64.b64encode(preview_buf.getvalue()).decode('utf-8')}"

    return overlay_url, preview_url, int(aoi_w), int(aoi_h)


@router.get("/sam2/status")
def sam2_status():
    """Get the status of the SAM2 model."""
    return get_sam2_status()


@router.post("/sam2/encode")
def sam2_encode(req: SAM2EncodeRequest):
    """Pre-encode an image for SAM2 (runs the expensive image encoder once)."""
    if not is_sam2_ready():
        # Try to initialize
        if not init_sam2():
            raise HTTPException(
                status_code=503,
                detail=f"SAM2 model not available: {get_sam2_status().get('error', 'Unknown error')}"
            )

    try:
        cache_key, raster_path = _get_cached_raster(req.image_id, req.bbox)

        # Extract RGB image
        rgb_image = _extract_rgb_uint8(raster_path)

        # Encode
        encode_image(cache_key, rgb_image)

        return {
            "status": "encoded",
            "cache_key": cache_key,
            "image_shape": list(rgb_image.shape),
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"SAM2 encode error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM2 encoding failed: {str(e)}")


@router.post("/sam2/predict")
def sam2_predict(req: SAM2PredictRequest):
    """Run SAM2 mask prediction with point prompts."""
    if not is_sam2_ready():
        if not init_sam2():
            raise HTTPException(
                status_code=503,
                detail=f"SAM2 model not available: {get_sam2_status().get('error', 'Unknown error')}"
            )

    try:
        cache_key, raster_path = _get_cached_raster(req.image_id, req.bbox)

        # Auto-encode if needed
        from ..services.sam2_service import _CURRENT_ENCODED_KEY
        if _CURRENT_ENCODED_KEY != cache_key:
            rgb_image = _extract_rgb_uint8(raster_path)
            encode_image(cache_key, rgb_image)

        # Convert lat/lng to pixel coordinates
        positive_pixels = []
        for p in req.positive_points:
            x, y = _latlon_to_pixel(raster_path, p.lat, p.lng)
            positive_pixels.append((x, y))

        negative_pixels = []
        if req.negative_points:
            for p in req.negative_points:
                x, y = _latlon_to_pixel(raster_path, p.lat, p.lng)
                negative_pixels.append((x, y))

        print(f"SAM2 - Positive pixels: {positive_pixels}")
        print(f"SAM2 - Negative pixels: {negative_pixels}")

        # Run prediction
        mask, score = predict_mask(
            cache_key,
            positive_pixels,
            negative_pixels if negative_pixels else None,
        )

        # Generate overlay
        overlay_url, preview_url, aoi_w, aoi_h = _mask_to_overlay_base64(
            mask, raster_path, req.bbox
        )

        # Generate mask ID and cache
        mask_id = f"sam2-{int(time.time() * 1000)}"
        min_lon, min_lat, max_lon, max_lat = req.bbox

        with SAM2_MASK_CACHE_LOCK:
            SAM2_MASK_CACHE[mask_id] = {
                "mask": mask,
                "raster_path": raster_path,
                "bbox": req.bbox,
                "overlay_url": overlay_url,
                "preview_url": preview_url,
                "score": score,
                "pixel_count": int(mask.sum()),
                "saved": False,
            }

        return {
            "mask_id": mask_id,
            "score": float(score),
            "overlay_url": overlay_url,
            "preview_url": preview_url,
            "pixel_count": int(mask.sum()),
            "total_pixels": int(mask.size),
            "overlay_meta": {
                "width": aoi_w,
                "height": aoi_h,
                "bounds": [float(min_lat), float(min_lon), float(max_lat), float(max_lon)],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"SAM2 predict error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM2 prediction failed: {str(e)}")


@router.post("/sam2/save-mask")
def sam2_save_mask(req: SAM2SaveMaskRequest):
    """Mark a predicted mask as saved."""
    with SAM2_MASK_CACHE_LOCK:
        if req.mask_id not in SAM2_MASK_CACHE:
            raise HTTPException(status_code=404, detail="Mask not found")
        SAM2_MASK_CACHE[req.mask_id]["saved"] = True

    return {"status": "saved", "mask_id": req.mask_id}


@router.delete("/sam2/masks/{mask_id}")
def sam2_delete_mask(mask_id: str):
    """Delete a saved mask."""
    with SAM2_MASK_CACHE_LOCK:
        if mask_id in SAM2_MASK_CACHE:
            del SAM2_MASK_CACHE[mask_id]

    return {"status": "deleted", "mask_id": mask_id}
