"""
API routes for SAM3 (Segment Anything Model 3) segmentation.

Endpoints
---------
GET    /api/sam3/status                    - model readiness
POST   /api/sam3/encode                    - pre-encode a Sentinel-2 raster
POST   /api/sam3/predict                   - point/box mode (single mask)
POST   /api/sam3/text-predict              - text mode (multi-instance PCS)
POST   /api/sam3/save-mask                 - mark a mask as saved
DELETE /api/sam3/masks/{mask_id}           - delete a saved/preview mask
"""

import os
import time
import base64
import colorsys
import threading
from typing import Dict, Tuple

import numpy as np
from PIL import Image
import io as pyio
import rasterio
import pyproj

from fastapi import APIRouter, HTTPException

from .schemas import (
    SAM3EncodeRequest,
    SAM3PointPredictRequest,
    SAM3TextPredictRequest,
    SAM3SaveMaskRequest,
)
from ..core.config import OUTPUTS_DIR  # noqa: F401  (kept for parity)
from ..services.sam3_service import (
    init_sam3,
    is_sam3_ready,
    get_sam3_status,
    encode_image,
    encode_and_predict,
    encode_and_predict_text,
)
from ..services.visualization import (
    warp_rgb_and_mask_to_aoi,
)
from ..utils.cache import (
    bbox_to_cache_key,
    RASTER_FILE_CACHE,
    RASTER_CACHE_LOCK,
)


router = APIRouter(prefix="/api", tags=["sam3"])

# Cache for SAM3 masks (point-mode previews and text-mode instances)
SAM3_MASK_CACHE: Dict = {}
SAM3_MASK_CACHE_LOCK = threading.Lock()

# Default magenta overlay colour, matches the previous SAM2 visual.
_DEFAULT_OVERLAY_RGB: Tuple[int, int, int] = (255, 0, 200)


# ============================================================================
# Helpers shared with routes_local.py (uploaded GeoTIFF path)
# ============================================================================

def instance_color(index: int, total: int) -> Tuple[int, int, int]:
    """Pick a distinct RGB colour for instance ``index`` of ``total``.

    Uses an evenly-spaced HSV hue so neighbouring instances are easy to tell
    apart, even when ``total`` is large.
    """
    if total <= 0:
        total = 1
    hue = (index % total) / float(total)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _get_cached_raster(image_id: str, bbox: list):
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        cached_raster_path = RASTER_FILE_CACHE.get(cache_key)
    if not cached_raster_path or not os.path.exists(cached_raster_path):
        raise HTTPException(
            status_code=400,
            detail="No cached raster data found. Please process the image first.",
        )
    return cache_key, cached_raster_path


def _extract_rgb_uint8(raster_path: str) -> np.ndarray:
    """Read B4/B3/B2 from a Sentinel-2 raster and percentile-stretch to uint8."""
    with rasterio.open(raster_path) as src:
        blue = src.read(1).astype(np.float32)   # B2
        green = src.read(2).astype(np.float32)  # B3
        red = src.read(3).astype(np.float32)    # B4

    rgb = np.stack([red, green, blue], axis=-1)

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

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _latlon_to_pixel(raster_path: str, lat: float, lng: float):
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        raster_transform = src.transform
    transformer = pyproj.Transformer.from_crs(
        "EPSG:4326", raster_crs, always_xy=True
    )
    x_proj, y_proj = transformer.transform(lng, lat)
    row, col = rasterio.transform.rowcol(raster_transform, x_proj, y_proj)
    return int(col), int(row)


def make_mask_overlay_native(
    mask: np.ndarray,
    rgb_color: Tuple[int, int, int] = _DEFAULT_OVERLAY_RGB,
) -> Tuple[str, str]:
    """Render a coloured mask overlay PNG (no warping) and a thumbnail.

    Used by the local/uploaded path where the image is already in pixel space.
    Returns ``(overlay_url, preview_url)``.
    """
    mask = mask.astype(bool)
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask, 0] = rgb_color[0]
    rgba[mask, 1] = rgb_color[1]
    rgba[mask, 2] = rgb_color[2]
    rgba[mask, 3] = 200

    img = Image.fromarray(rgba, mode="RGBA")
    buf = pyio.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    overlay_url = (
        f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
    )

    preview = img.copy()
    preview.thumbnail((256, 256), Image.LANCZOS)
    pbuf = pyio.BytesIO()
    preview.save(pbuf, format="PNG", optimize=True)
    preview_url = (
        f"data:image/png;base64,{base64.b64encode(pbuf.getvalue()).decode('utf-8')}"
    )
    return overlay_url, preview_url


def make_mask_overlay_aoi(
    mask: np.ndarray,
    raster_path: str,
    bbox: list,
    rgb_color: Tuple[int, int, int] = _DEFAULT_OVERLAY_RGB,
) -> Tuple[str, str, int, int]:
    """Warp a coloured mask overlay to AOI bounds for the cloud path.

    Returns ``(overlay_url, preview_url, aoi_w, aoi_h)``.
    """
    with rasterio.open(raster_path) as src:
        raster_transform = src.transform
        raster_crs = src.crs

    mask = mask.astype(bool)

    mask_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    mask_rgb[mask, 0] = rgb_color[0]
    mask_rgb[mask, 1] = rgb_color[1]
    mask_rgb[mask, 2] = rgb_color[2]

    min_lon, min_lat, max_lon, max_lat = bbox
    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        mask_rgb,
        mask,
        raster_transform,
        raster_crs,
        (min_lon, min_lat, max_lon, max_lat),
        scale_m=10,
    )

    try:
        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)
    except ImportError:
        rgba = np.zeros((*aoi_rgb.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = aoi_rgb
        rgba[:, :, 3] = (aoi_mask * 180).astype(np.uint8)
        img = Image.fromarray(rgba, mode="RGBA")
        buf = pyio.BytesIO()
        img.save(buf, format="PNG")
        overlay_url = (
            f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        )

    mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = mask_rgb
    mask_rgba[mask, 3] = 200
    preview_img = Image.fromarray(mask_rgba, mode="RGBA")
    preview_img.thumbnail((256, 256), Image.LANCZOS)
    preview_buf = pyio.BytesIO()
    preview_img.save(preview_buf, format="PNG")
    preview_url = (
        f"data:image/png;base64,{base64.b64encode(preview_buf.getvalue()).decode('utf-8')}"
    )

    return overlay_url, preview_url, int(aoi_w), int(aoi_h)


def _ensure_ready():
    if not is_sam3_ready() and not init_sam3():
        raise HTTPException(
            status_code=503,
            detail=f"SAM3 model not available: {get_sam3_status().get('error', 'Unknown error')}",
        )


# ============================================================================
# Endpoints
# ============================================================================

@router.get("/sam3/status")
def sam3_status():
    return get_sam3_status()


@router.post("/sam3/encode")
def sam3_encode(req: SAM3EncodeRequest):
    """Pre-encode a Sentinel-2 image for SAM3 (cheap subsequent predicts)."""
    _ensure_ready()
    try:
        cache_key, raster_path = _get_cached_raster(req.image_id, req.bbox)
        rgb_image = _extract_rgb_uint8(raster_path)
        encode_image(cache_key, rgb_image)
        return {
            "status": "encoded",
            "cache_key": cache_key,
            "image_shape": list(rgb_image.shape),
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"SAM3 encode error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM3 encoding failed: {e}")


@router.post("/sam3/predict")
def sam3_predict(req: SAM3PointPredictRequest):
    """Point/box mode: single best mask for the supplied prompts."""
    _ensure_ready()
    try:
        cache_key, raster_path = _get_cached_raster(req.image_id, req.bbox)

        positive_pixels = [
            _latlon_to_pixel(raster_path, p.lat, p.lng) for p in req.positive_points
        ]
        negative_pixels = [
            _latlon_to_pixel(raster_path, p.lat, p.lng) for p in (req.negative_points or [])
        ]

        print(f"SAM3 - positive pixels: {positive_pixels}")
        print(f"SAM3 - negative pixels: {negative_pixels}")

        mask, score = encode_and_predict(
            cache_key,
            lambda: _extract_rgb_uint8(raster_path),
            positive_pixels,
            negative_pixels if negative_pixels else None,
        )

        overlay_url, preview_url, aoi_w, aoi_h = make_mask_overlay_aoi(
            mask, raster_path, req.bbox
        )

        mask_id = f"sam3-{int(time.time() * 1000)}"
        min_lon, min_lat, max_lon, max_lat = req.bbox

        # Read transform/crs once for change-detection alignment.
        with rasterio.open(raster_path) as src:
            sam3_transform = src.transform
            sam3_crs = src.crs

        with SAM3_MASK_CACHE_LOCK:
            SAM3_MASK_CACHE[mask_id] = {
                "mask": mask,
                "binary_mask": mask,  # alias used by change-detection lookup
                "raster_path": raster_path,
                "bbox": req.bbox,
                "transform": sam3_transform,
                "crs": sam3_crs,
                "overlay_url": overlay_url,
                "preview_url": preview_url,
                "score": score,
                "pixel_count": int(mask.sum()),
                "saved": False,
                "mode": "point",
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
                "bounds": [
                    float(min_lat), float(min_lon),
                    float(max_lat), float(max_lon),
                ],
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"SAM3 predict error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM3 prediction failed: {e}")


@router.post("/sam3/text-predict")
def sam3_text_predict(req: SAM3TextPredictRequest):
    """Text mode (PCS): every instance matching ``req.prompt`` in the AOI."""
    _ensure_ready()
    try:
        cache_key, raster_path = _get_cached_raster(req.image_id, req.bbox)

        masks, scores, boxes = encode_and_predict_text(
            cache_key,
            lambda: _extract_rgb_uint8(raster_path),
            req.prompt,
            score_threshold=req.score_threshold or 0.5,
        )

        ts = int(time.time() * 1000)
        min_lon, min_lat, max_lon, max_lat = req.bbox
        n = len(masks)
        total_for_color = max(n, 8)  # smoother hues with at least 8 buckets

        with rasterio.open(raster_path) as src:
            sam3_transform = src.transform
            sam3_crs = src.crs

        instances = []
        for i, (m, sc, bx) in enumerate(zip(masks, scores, boxes)):
            color = instance_color(i, total_for_color)
            overlay_url, preview_url, aoi_w, aoi_h = make_mask_overlay_aoi(
                m, raster_path, req.bbox, rgb_color=color
            )
            mask_id = f"sam3-text-{ts}-{i}"
            with SAM3_MASK_CACHE_LOCK:
                SAM3_MASK_CACHE[mask_id] = {
                    "mask": m,
                    "binary_mask": m,
                    "raster_path": raster_path,
                    "bbox": req.bbox,
                    "transform": sam3_transform,
                    "crs": sam3_crs,
                    "overlay_url": overlay_url,
                    "preview_url": preview_url,
                    "score": sc,
                    "pixel_count": int(m.sum()),
                    "saved": False,
                    "mode": "text",
                    "prompt": req.prompt,
                    "color": color,
                }
            instances.append({
                "mask_id": mask_id,
                "score": float(sc),
                "overlay_url": overlay_url,
                "preview_url": preview_url,
                "pixel_count": int(m.sum()),
                "color_hex": rgb_to_hex(color),
                "bbox_pixel": list(bx),
                "overlay_meta": {
                    "width": aoi_w,
                    "height": aoi_h,
                    "bounds": [
                        float(min_lat), float(min_lon),
                        float(max_lat), float(max_lon),
                    ],
                },
            })

        return {
            "prompt": req.prompt,
            "instance_count": len(instances),
            "instances": instances,
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"SAM3 text-predict error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM3 text prediction failed: {e}")


@router.post("/sam3/save-mask")
def sam3_save_mask(req: SAM3SaveMaskRequest):
    with SAM3_MASK_CACHE_LOCK:
        if req.mask_id not in SAM3_MASK_CACHE:
            raise HTTPException(status_code=404, detail="Mask not found")
        SAM3_MASK_CACHE[req.mask_id]["saved"] = True
    return {"status": "saved", "mask_id": req.mask_id}


@router.delete("/sam3/masks/{mask_id}")
def sam3_delete_mask(mask_id: str):
    with SAM3_MASK_CACHE_LOCK:
        if mask_id in SAM3_MASK_CACHE:
            del SAM3_MASK_CACHE[mask_id]
    return {"status": "deleted", "mask_id": mask_id}
