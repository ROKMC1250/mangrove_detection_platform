"""
API routes for loading and visualizing local TIF files.
Saves full-resolution WebP files to outputs/ and serves via proxy.
Also provides spectral analysis (NDVI, SAVI, cloud mask),
custom visualization, and target detection for local images.
"""

import os
import hashlib
import json
import queue
import time
import base64
import threading
import io as pyio
from typing import Dict, List, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from PIL import Image
import requests as url_requests

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..core.config import OUTPUTS_DIR, PROJECT_ROOT

router = APIRouter(prefix="/api/local", tags=["local"])

# Base directory for the band-registration experiment results browsed in local
# mode. Set LOCAL_BASE_DIR in .env to point at your own directory; relative
# paths are resolved against the project root.
LOCAL_BASE_DIR = os.environ.get("LOCAL_BASE_DIR", "").strip() or os.path.join(
    PROJECT_ROOT, "experiment_results"
)
if not os.path.isabs(LOCAL_BASE_DIR):
    LOCAL_BASE_DIR = os.path.join(PROJECT_ROOT, LOCAL_BASE_DIR)

# Upload directory for user-uploaded GeoTIFFs
UPLOAD_DIR = os.path.join(OUTPUTS_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Metadata cache for uploaded images
_UPLOADED_IMAGE_META: Dict[str, Dict] = {}

# Cache for loaded raster data: key -> numpy array
_LOCAL_RASTER_CACHE: Dict[str, np.ndarray] = {}

# Cache for computed index data (for threshold / pixel value operations)
_LOCAL_INDEX_CACHE: Dict[str, np.ndarray] = {}
_LOCAL_INDEX_CACHE_LOCK = threading.Lock()

# Cache for target detection results
_LOCAL_TD_CACHE: Dict[str, Dict] = {}
_LOCAL_TD_CACHE_LOCK = threading.Lock()

# Default band mapping: band role -> band file pattern
# MS1=Blue, MS2=Green, MS3=Red, MS4=RedEdge1, MS5=RedEdge2, MS6=NIR
LOCAL_BAND_ROLES = {
    'BLUE': 'MS1',
    'GREEN': 'MS2',
    'RED': 'MS3',
    'REDEDGE1': 'MS4',
    'REDEDGE2': 'MS5',
    'NIR': 'MS6',
}


class LocalVizRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    bands: List[str] = Field(..., description="3 band filenames for R,G,B")
    min_val: float = 0
    max_val: float = 3000
    gamma: float = 1.0


class LocalGrayscaleRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    band: str
    min_val: float = 0
    max_val: float = 3000


def _read_band(image_dir: str, algorithm_dir: str, band_file: str,
                target_shape: tuple = None) -> np.ndarray:
    """Read a single band TIF file and return as 2D float32 array.

    If target_shape is provided as (height, width), resample using bicubic
    interpolation to match that resolution.
    """
    cache_key = f"{image_dir}/{algorithm_dir}/{band_file}"
    if target_shape:
        cache_key += f"@{target_shape[0]}x{target_shape[1]}"
    if cache_key in _LOCAL_RASTER_CACHE:
        return _LOCAL_RASTER_CACHE[cache_key]

    path = os.path.join(LOCAL_BASE_DIR, image_dir, algorithm_dir, band_file)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {band_file}")

    with rasterio.open(path) as src:
        if target_shape and (src.height, src.width) != target_shape:
            data = src.read(1, out_shape=target_shape,
                           resampling=Resampling.cubic).astype(np.float32)
        else:
            data = src.read(1).astype(np.float32)

    _LOCAL_RASTER_CACHE[cache_key] = data
    return data


def _get_max_band_shape(image_dir: str, algorithm_dir: str, band_files: list) -> tuple:
    """Scan band files and return the largest (height, width) among them."""
    max_h, max_w = 0, 0
    for bf in band_files:
        path = os.path.join(LOCAL_BASE_DIR, image_dir, algorithm_dir, bf)
        if os.path.exists(path):
            with rasterio.open(path) as src:
                if src.height > max_h or src.width > max_w:
                    max_h = max(max_h, src.height)
                    max_w = max(max_w, src.width)
    return (max_h, max_w)


def _read_all_bands_resampled(image_dir: str, algorithm_dir: str,
                               band_files: list) -> list:
    """Read all bands, resampling to the highest resolution using bicubic."""
    target_shape = _get_max_band_shape(image_dir, algorithm_dir, band_files)
    return [_read_band(image_dir, algorithm_dir, bf, target_shape=target_shape)
            for bf in band_files]


def _stretch_to_uint8(data: np.ndarray, min_val: float, max_val: float, gamma: float = 1.0) -> np.ndarray:
    """Stretch data to uint8 range with optional gamma."""
    arr = np.clip((data - min_val) / max(max_val - min_val, 1e-10), 0, 1)
    if gamma != 1.0:
        arr = np.power(arr, 1.0 / gamma)
    return (arr * 255).astype(np.uint8)


def _save_webp(rgb: np.ndarray, cache_key: str) -> str:
    """Save full-resolution RGB array as WebP, return URL path. Skips if already cached."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    filename = hashlib.md5(cache_key.encode()).hexdigest() + ".webp"
    filepath = os.path.join(OUTPUTS_DIR, filename)

    if not os.path.exists(filepath):
        img = Image.fromarray(rgb)
        img.save(filepath, format='WEBP', quality=90, method=2)

    import requests as _req
    url = f"/api/proxy-file?path={_req.utils.quote('file://' + filepath)}"
    return url


@router.get("/images")
def list_images():
    """List all available image directories."""
    if not os.path.isdir(LOCAL_BASE_DIR):
        raise HTTPException(status_code=404, detail=f"Base directory not found")

    images = []
    for name in sorted(os.listdir(LOCAL_BASE_DIR)):
        if os.path.isdir(os.path.join(LOCAL_BASE_DIR, name)):
            images.append(name)
    return {"images": images}


@router.get("/algorithms/{image_dir}")
def list_algorithms(image_dir: str):
    """List algorithm subdirectories for a given image."""
    dir_path = os.path.join(LOCAL_BASE_DIR, image_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail=f"Image directory not found")

    algorithms = []
    for name in sorted(os.listdir(dir_path)):
        if os.path.isdir(os.path.join(dir_path, name)):
            algorithms.append(name)
    return {"algorithms": algorithms}


@router.get("/bands/{image_dir}/{algorithm_dir}")
def list_bands(image_dir: str, algorithm_dir: str):
    """List available TIF band files in an algorithm directory."""
    dir_path = os.path.join(LOCAL_BASE_DIR, image_dir, algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail=f"Algorithm directory not found")

    bands = []
    for name in sorted(os.listdir(dir_path)):
        if name.endswith('.tif') and name.startswith('after_'):
            bands.append(name)
    return {"bands": bands}


class PercentilesRequest(BaseModel):
    image_dir: str
    algorithm_dir: str


@router.post("/percentiles")
def get_percentiles(req: PercentilesRequest):
    """Get percentile values (1-99) for all bands. Used for percentile stretch."""
    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))

    # Stack all bands to compute global percentiles
    all_data = []
    for bf in band_files:
        data = _read_band(req.image_dir, req.algorithm_dir, bf)
        all_data.append(data.ravel())
    combined = np.concatenate(all_data)
    valid = combined[np.isfinite(combined)]

    percentiles = {}
    for p in range(1, 100):
        percentiles[str(p)] = float(np.percentile(valid, p))

    return {"percentiles": percentiles}


class PreloadRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    min_val: float = 0
    max_val: float = 3000


@router.post("/preload-bands")
def preload_bands(req: PreloadRequest):
    """Read all bands, stretch to uint8, return as base64 data URLs. No file I/O."""
    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    result = {}

    for band_file in band_files:
        data = _read_band(req.image_dir, req.algorithm_dir, band_file)
        gray = _stretch_to_uint8(data, req.min_val, req.max_val)
        rgb = np.stack([gray, gray, gray], axis=-1)
        buf = pyio.BytesIO()
        Image.fromarray(rgb).save(buf, format='JPEG', quality=85)
        result[band_file] = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

    first_data = _read_band(req.image_dir, req.algorithm_dir, band_files[0])
    return {
        "band_urls": result,
        "width": int(first_data.shape[1]),
        "height": int(first_data.shape[0])
    }


# =============================================================================
# Raw band data endpoint — send float32 binary to browser for client-side stretch
# =============================================================================

class RawBandsRequest(BaseModel):
    image_dir: str
    algorithm_dir: str


@router.post("/raw-bands")
def get_raw_bands(req: RawBandsRequest):
    """Read all bands, return raw float32 as base64 binary + shape info.

    Browser receives Float32Arrays and does stretch/percentile locally.
    No JPEG encoding, no stretch — just raw data transfer.
    Also loads to GPU cache so analyze/detection is instant later.
    """
    import time as _time
    from ..services.gpu_compute import load_image_to_gpu, get_gpu_image

    t0 = _time.time()
    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    if not band_files:
        raise HTTPException(status_code=400, detail="No band files found")

    # Read bands from disk (cached in _LOCAL_RASTER_CACHE after first read)
    t1 = _time.time()
    band_arrays = _read_all_bands_resampled(req.image_dir, req.algorithm_dir, band_files)
    print(f"RAW-BANDS - Disk read: {_time.time()-t1:.2f}s ({len(band_files)} bands)")

    h, w = band_arrays[0].shape

    # Encode each band as base64 float32 binary
    t2 = _time.time()
    band_data = {}
    for i, bf in enumerate(band_files):
        raw_bytes = band_arrays[i].tobytes()  # float32 binary
        band_data[bf] = base64.b64encode(raw_bytes).decode('ascii')
    print(f"RAW-BANDS - Base64 encode: {_time.time()-t2:.2f}s")

    # Also load to GPU for later analyze/detection
    gpu_cache_key = f"{req.image_dir}/{req.algorithm_dir}"
    if get_gpu_image(gpu_cache_key) is None:
        t3 = _time.time()
        load_image_to_gpu(gpu_cache_key, band_arrays)
        print(f"RAW-BANDS - GPU upload: {_time.time()-t3:.2f}s")

    print(f"RAW-BANDS - Total: {_time.time()-t0:.2f}s")

    return {
        "band_data": band_data,
        "band_files": band_files,
        "width": w,
        "height": h,
        "dtype": "float32",
    }


# =============================================================================
# GPU-accelerated: Load bands to GPU + compute percentiles in one call
# =============================================================================

class GpuLoadRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    min_val: float = 0
    max_val: float = 3000


@router.post("/gpu-load")
def gpu_load_bands(req: GpuLoadRequest):
    """Read all bands, load to GPU, compute percentiles on GPU, stretch on GPU, return base64.

    Single endpoint replaces separate /percentiles + /preload-bands calls.
    Disk read happens once, everything else runs on GPU.
    """
    import time as _time
    from ..services.gpu_compute import load_image_to_gpu, get_gpu_image, compute_percentiles_gpu, stretch_bands_gpu, clear_gpu_image

    t0 = _time.time()

    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    if not band_files:
        raise HTTPException(status_code=400, detail="No band files found")

    gpu_cache_key = f"{req.image_dir}/{req.algorithm_dir}"

    # Check if already on GPU (e.g. from previous load or analyze)
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is not None:
        print(f"GPU-LOAD - Reusing cached GPU tensor: {gpu_image.shape}")
    else:
        # Step 1: Read bands from disk (only slow part, happens once)
        t1 = _time.time()
        band_arrays = _read_all_bands_resampled(req.image_dir, req.algorithm_dir, band_files)
        print(f"GPU-LOAD - Disk read: {_time.time()-t1:.2f}s ({len(band_files)} bands)")

        # Step 2: Load to GPU (stacks + transfers, cached for later analyze/detection)
        t2 = _time.time()
        gpu_image = load_image_to_gpu(gpu_cache_key, band_arrays)
        print(f"GPU-LOAD - GPU transfer: {_time.time()-t2:.2f}s")

    # Step 3: Compute percentiles on GPU
    t3 = _time.time()
    percentiles = compute_percentiles_gpu(gpu_image)
    print(f"GPU-LOAD - Percentiles: {_time.time()-t3:.2f}s")

    # Step 4: Stretch all bands on GPU + encode to JPEG base64
    t4 = _time.time()
    band_urls_by_idx, w, h = stretch_bands_gpu(gpu_image, req.min_val, req.max_val)
    # Map back to band filenames
    band_urls = {}
    for i, bf in enumerate(band_files):
        band_urls[bf] = band_urls_by_idx[i]
    print(f"GPU-LOAD - Stretch+encode: {_time.time()-t4:.2f}s")

    print(f"GPU-LOAD - Total: {_time.time()-t0:.2f}s")

    return {
        "band_urls": band_urls,
        "width": w,
        "height": h,
        "percentiles": percentiles,
        "band_files": band_files,
    }


class GpuStretchRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    min_val: float = 0
    max_val: float = 3000


@router.post("/gpu-stretch")
def gpu_stretch_bands(req: GpuStretchRequest):
    """Re-stretch bands using GPU-cached tensor. No disk read, no re-upload.

    Called when user adjusts percentile sliders. Sub-second response.
    """
    import time as _time
    from ..services.gpu_compute import get_gpu_image, stretch_bands_gpu

    t0 = _time.time()

    gpu_cache_key = f"{req.image_dir}/{req.algorithm_dir}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        raise HTTPException(status_code=400, detail="Image not loaded on GPU. Call /gpu-load first.")

    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))

    band_urls_by_idx, w, h = stretch_bands_gpu(gpu_image, req.min_val, req.max_val)
    band_urls = {}
    for i, bf in enumerate(band_files):
        band_urls[bf] = band_urls_by_idx[i]

    print(f"GPU-STRETCH - Total: {_time.time()-t0:.2f}s (no disk read)")

    return {
        "band_urls": band_urls,
        "width": w,
        "height": h,
    }


@router.post("/visualize-rgb")
def visualize_rgb(req: LocalVizRequest):
    """Create RGB visualization from 3 local TIF bands. Returns base64."""
    if len(req.bands) != 3:
        raise HTTPException(status_code=400, detail="Exactly 3 bands required for RGB")

    try:
        tgt = _get_max_band_shape(req.image_dir, req.algorithm_dir, req.bands)
        r_data = _read_band(req.image_dir, req.algorithm_dir, req.bands[0], target_shape=tgt)
        g_data = _read_band(req.image_dir, req.algorithm_dir, req.bands[1], target_shape=tgt)
        b_data = _read_band(req.image_dir, req.algorithm_dir, req.bands[2], target_shape=tgt)

        r_uint8 = _stretch_to_uint8(r_data, req.min_val, req.max_val, req.gamma)
        g_uint8 = _stretch_to_uint8(g_data, req.min_val, req.max_val, req.gamma)
        b_uint8 = _stretch_to_uint8(b_data, req.min_val, req.max_val, req.gamma)

        rgb = np.stack([r_uint8, g_uint8, b_uint8], axis=-1)

        buf = pyio.BytesIO()
        Image.fromarray(rgb).save(buf, format='JPEG', quality=85)
        url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "image_url": url,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize-grayscale")
def visualize_grayscale(req: LocalGrayscaleRequest):
    """Create full-resolution grayscale visualization from a single TIF band."""
    try:
        data = _read_band(req.image_dir, req.algorithm_dir, req.band)
        gray = _stretch_to_uint8(data, req.min_val, req.max_val)
        rgb = np.stack([gray, gray, gray], axis=-1)

        buf = pyio.BytesIO()
        Image.fromarray(rgb).save(buf, format='JPEG', quality=85)
        url = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"

        return {
            "image_url": url,
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0])
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class LocalPixelRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    row: int
    col: int


@router.post("/get-pixel-values")
def get_local_pixel_values(req: LocalPixelRequest):
    """Get all band values at a pixel from local TIF files."""
    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    bands = []

    for bf in band_files:
        data = _read_band(req.image_dir, req.algorithm_dir, bf)
        r = max(0, min(req.row, data.shape[0] - 1))
        c = max(0, min(req.col, data.shape[1] - 1))
        val = float(data[r, c])
        name = bf.replace('after_', '').replace('.tif', '')
        bands.append({"name": name, "value": round(val, 2)})

    return {"bands": bands, "row": req.row, "col": req.col}


# =============================================================================
# Helper: find band file by spectral role
# =============================================================================

def _find_band_file(image_dir: str, algorithm_dir: str, role: str) -> Optional[str]:
    """Find band file matching a spectral role (e.g. 'NIR' -> 'after_MS6.tif')."""
    ms_id = LOCAL_BAND_ROLES.get(role.upper())
    if not ms_id:
        return None
    dir_path = os.path.join(LOCAL_BASE_DIR, image_dir, algorithm_dir)
    if not os.path.isdir(dir_path):
        return None
    for f in os.listdir(dir_path):
        if f.startswith('after_') and f.endswith('.tif') and ms_id in f:
            return f
    return None


def _save_overlay_png(rgba: np.ndarray, prefix: str, max_dim: int = 2048) -> str:
    """Save RGBA numpy array as PNG. Downscale on GPU if larger than max_dim."""
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    timestamp = int(time.time() * 1000)
    filename = f'{prefix}_{timestamp}.png'
    filepath = os.path.join(OUTPUTS_DIR, filename)

    h, w = rgba.shape[:2]
    if max(h, w) > max_dim:
        try:
            import torch
            t = torch.from_numpy(rgba).to('cuda').permute(2, 0, 1).unsqueeze(0).float()
            scale = max_dim / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            t_small = torch.nn.functional.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False)
            rgba = t_small.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            del t, t_small
            torch.cuda.empty_cache()
        except Exception:
            img = Image.fromarray(rgba, mode='RGBA')
            scale = max_dim / max(h, w)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            img.save(filepath, format='PNG', compress_level=1)
            return f"/api/proxy-file?path={url_requests.utils.quote('file://' + filepath)}"

    Image.fromarray(rgba, mode='RGBA').save(filepath, format='PNG', compress_level=1)
    return f"/api/proxy-file?path={url_requests.utils.quote('file://' + filepath)}"


def _preview_to_b64(preview_rgba: np.ndarray) -> str:
    """Convert small RGBA numpy array to base64 data URL."""
    img = Image.fromarray(preview_rgba, mode='RGBA')
    buf = pyio.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _make_preview_b64(rgb: np.ndarray, mask: np.ndarray = None, max_size: int = 256) -> str:
    """Create base64-encoded preview thumbnail."""
    if mask is not None:
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
        img = Image.fromarray(rgba, mode='RGBA')
    else:
        img = Image.fromarray(rgb, mode='RGB')
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = pyio.BytesIO()
    img.save(buf, format='PNG')
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# =============================================================================
# Local Analysis Endpoint
# =============================================================================

class LocalAnalyzeRequest(BaseModel):
    image_dir: str
    algorithm_dir: str


@router.post("/analyze")
def analyze_local_image(req: LocalAnalyzeRequest):
    """Run analysis (cloud mask, NDVI, SAVI) on local image bands. GPU-accelerated."""
    from ..services.gpu_compute import (
        compute_normalized_index_gpu, compute_savi_gpu,
        create_index_visualization_gpu, build_rgba_and_preview_gpu,
        rgba_to_base64_gpu, load_image_to_gpu,
    )

    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    # Pre-load all bands to GPU cache for fast target detection later
    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    gpu_cache_key = f"{req.image_dir}/{req.algorithm_dir}"
    band_arrays = _read_all_bands_resampled(req.image_dir, req.algorithm_dir, band_files)
    load_image_to_gpu(gpu_cache_key, band_arrays)

    analysis_results = {}

    nir_file = _find_band_file(req.image_dir, req.algorithm_dir, 'NIR')
    red_file = _find_band_file(req.image_dir, req.algorithm_dir, 'RED')
    blue_file = _find_band_file(req.image_dir, req.algorithm_dir, 'BLUE')

    sample_file = nir_file or red_file or blue_file
    if not sample_file:
        raise HTTPException(status_code=400, detail="No suitable bands found for analysis")

    # Use the max resolution across all bands for consistent analysis
    target_shape = _get_max_band_shape(req.image_dir, req.algorithm_dir, band_files)
    sample_data = _read_band(req.image_dir, req.algorithm_dir, sample_file, target_shape=target_shape)
    h, w = sample_data.shape

    # --- Cloud Mask ---
    if blue_file:
        try:
            blue_data = _read_band(req.image_dir, req.algorithm_dir, blue_file, target_shape=target_shape)
            valid = blue_data[np.isfinite(blue_data)]
            if len(valid) > 0:
                threshold_val = float(np.percentile(valid, 95))
                cloud_mask = blue_data > threshold_val
                dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                rgba, preview = build_rgba_and_preview_gpu(
                    dummy_rgb, cloud_mask, color_override=(255, 165, 0), alpha_val=128
                )
                overlay_url = rgba_to_base64_gpu(rgba)
                preview_url = _preview_to_b64(preview)

                analysis_results['cloud_mask'] = {
                    'name': 'Cloud Mask',
                    'preview_url': preview_url,
                    'overlay_url': overlay_url,
                    'bands': ['Blue (brightness threshold)']
                }
        except Exception as e:
            print(f"Cloud mask error: {e}")

    # --- NDVI + SAVI (share NIR/RED, GPU-accelerated) ---
    nir_data = None
    red_data = None
    if nir_file and red_file:
        nir_data = _read_band(req.image_dir, req.algorithm_dir, nir_file, target_shape=target_shape)
        red_data = _read_band(req.image_dir, req.algorithm_dir, red_file, target_shape=target_shape)

    if nir_data is not None and red_data is not None:
        try:
            ndvi = compute_normalized_index_gpu(nir_data, red_data)
            with _LOCAL_INDEX_CACHE_LOCK:
                _LOCAL_INDEX_CACHE[f"{req.image_dir}/{req.algorithm_dir}/model2"] = ndvi

            ndvi_rgb, actual_min, actual_max = create_index_visualization_gpu(ndvi, 'RdYlGn', -1.0, 1.0)
            rgba, preview = build_rgba_and_preview_gpu(ndvi_rgb, np.isfinite(ndvi))
            overlay_url = rgba_to_base64_gpu(rgba)
            preview_url = _preview_to_b64(preview)

            analysis_results['model2'] = {
                'name': 'NDVI',
                'preview_url': preview_url,
                'overlay_url': overlay_url,
                'bands': ['NIR', 'RED'],
                'colormap': {
                    'name': 'RdYlGn',
                    'min_val': float(actual_min),
                    'max_val': float(actual_max),
                    'label': 'NDVI (NIR-RED)/(NIR+RED)'
                }
            }
        except Exception as e:
            print(f"NDVI error: {e}")

        try:
            savi = compute_savi_gpu(nir_data, red_data)
            with _LOCAL_INDEX_CACHE_LOCK:
                _LOCAL_INDEX_CACHE[f"{req.image_dir}/{req.algorithm_dir}/model4"] = savi

            savi_rgb, actual_min, actual_max = create_index_visualization_gpu(savi, 'viridis')
            rgba, preview = build_rgba_and_preview_gpu(savi_rgb, np.isfinite(savi))
            overlay_url = rgba_to_base64_gpu(rgba)
            preview_url = _preview_to_b64(preview)

            analysis_results['model4'] = {
                'name': 'SAVI',
                'preview_url': preview_url,
                'overlay_url': overlay_url,
                'bands': ['NIR', 'RED'],
                'colormap': {
                    'name': 'viridis',
                    'min_val': float(actual_min),
                    'max_val': float(actual_max),
                    'label': 'SAVI ((NIR-RED)/(NIR+RED+L))*(1+L)'
                }
            }
        except Exception as e:
            print(f"SAVI error: {e}")

    analysis_results['overlay_meta'] = {'width': int(w), 'height': int(h)}
    return {"analysis_results": analysis_results}


# =============================================================================
# Local Spectral Index (on-demand, like satellite mode)
# =============================================================================

class LocalSpectralIndexRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    index_type: str = Field(..., description="ndvi, mvi, ndmi, ndwi, savi, evi, custom")
    band_a: Optional[str] = None
    band_b: Optional[str] = None
    colormap: Optional[str] = None


@router.post("/compute-spectral-index")
def local_compute_spectral_index(req: LocalSpectralIndexRequest):
    """Compute a spectral index on-demand for local images. GPU-accelerated."""
    from ..services.gpu_compute import (
        compute_normalized_index_gpu, compute_savi_gpu,
        create_index_visualization_gpu, build_rgba_and_preview_gpu,
        rgba_to_base64_gpu,
    )
    from ..services.spectral_analysis import INDEX_REGISTRY, safe_divide

    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    index_type = req.index_type.lower()
    print(f"LOCAL SPECTRAL INDEX - Computing {index_type} for {req.image_dir}/{req.algorithm_dir}")

    # Determine max resolution across all bands for consistent resampling
    all_band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    tgt = _get_max_band_shape(req.image_dir, req.algorithm_dir, all_band_files) if all_band_files else None

    # Helper to read band by role
    def read_role(role):
        bf = _find_band_file(req.image_dir, req.algorithm_dir, role)
        if not bf:
            return None
        return _read_band(req.image_dir, req.algorithm_dir, bf, target_shape=tgt)

    # Helper to read band by filename (for custom)
    def read_by_name(band_name):
        """Try to find band file matching band_name (e.g. 'MS1', 'MS6', 'B2', etc.)"""
        band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
        for bf in band_files:
            name_part = bf.replace('after_', '').replace('.tif', '')
            if name_part.upper() == band_name.upper() or band_name.upper() in bf.upper():
                return _read_band(req.image_dir, req.algorithm_dir, bf, target_shape=tgt)
        # Also try role mapping
        role_map = {'B2': 'BLUE', 'B3': 'GREEN', 'B4': 'RED', 'B5': 'REDEDGE1',
                    'B6': 'REDEDGE2', 'B8': 'NIR', 'BLUE': 'BLUE', 'GREEN': 'GREEN',
                    'RED': 'RED', 'NIR': 'NIR'}
        role = role_map.get(band_name.upper())
        if role:
            return read_role(role)
        # Try by index
        try:
            idx = int(band_name.replace('Band', '').replace('band', '').replace('MS', '')) - 1
            if 0 <= idx < len(band_files):
                return _read_band(req.image_dir, req.algorithm_dir, band_files[idx], target_shape=tgt)
        except (ValueError, IndexError):
            pass
        return None

    try:
        if index_type == 'custom':
            if not req.band_a or not req.band_b:
                raise HTTPException(status_code=400, detail="band_a and band_b required for custom index")
            a_data = read_by_name(req.band_a)
            b_data = read_by_name(req.band_b)
            if a_data is None or b_data is None:
                raise HTTPException(status_code=400, detail=f"Could not find bands: {req.band_a}, {req.band_b}")
            index = compute_normalized_index_gpu(a_data, b_data)
            colormap_name = req.colormap or 'viridis'
            vmin, vmax = -1.0, 1.0
            label = f'{req.band_a}-{req.band_b} Index'
            index_name = f'Custom ({req.band_a}/{req.band_b})'
        elif index_type in INDEX_REGISTRY:
            info = INDEX_REGISTRY[index_type]
            colormap_name = req.colormap or info['colormap']
            vmin = info.get('vmin')
            vmax = info.get('vmax')
            label = info['name']
            index_name = info['name']

            nir = read_role('NIR')
            red = read_role('RED')
            green = read_role('GREEN')
            blue = read_role('BLUE')

            if index_type == 'ndvi':
                if nir is None or red is None:
                    raise HTTPException(status_code=400, detail="NIR and RED bands required for NDVI")
                index = compute_normalized_index_gpu(nir, red)
            elif index_type == 'ndmi':
                # NDMI needs SWIR which may not be available in local MS data
                # Fall back to using available bands
                swir = read_role('REDEDGE2')  # Proxy if no SWIR
                if nir is None or swir is None:
                    raise HTTPException(status_code=400, detail="Required bands not available for NDMI")
                index = compute_normalized_index_gpu(nir, swir)
            elif index_type == 'mvi':
                swir1 = read_role('REDEDGE1')  # Proxy
                if nir is None or green is None or swir1 is None:
                    raise HTTPException(status_code=400, detail="Required bands not available for MVI")
                index = safe_divide(nir - green, swir1 - green)
            elif index_type == 'ndwi':
                if green is None or nir is None:
                    raise HTTPException(status_code=400, detail="GREEN and NIR bands required for NDWI")
                index = compute_normalized_index_gpu(green, nir)
            elif index_type == 'savi':
                if nir is None or red is None:
                    raise HTTPException(status_code=400, detail="NIR and RED bands required for SAVI")
                index = compute_savi_gpu(nir, red)
            elif index_type == 'evi':
                if nir is None or red is None or blue is None:
                    raise HTTPException(status_code=400, detail="NIR, RED, and BLUE bands required for EVI")
                from ..services.spectral_analysis import calculate_evi
                index = calculate_evi(nir.astype(np.float32), red.astype(np.float32), blue.astype(np.float32))
            else:
                raise HTTPException(status_code=400, detail=f"Unknown index: {index_type}")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown index type: {index_type}")

        # Visualization
        index_rgb, actual_min, actual_max = create_index_visualization_gpu(index, colormap_name, vmin, vmax)
        finite_mask = np.isfinite(index)

        rgba, preview = build_rgba_and_preview_gpu(index_rgb, finite_mask)
        overlay_url = rgba_to_base64_gpu(rgba)
        preview_url = _preview_to_b64(preview)

        # Cache index data for threshold / pixel operations
        custom_id = f"local-spectral-{index_type}-{int(time.time() * 1000)}"
        with _LOCAL_INDEX_CACHE_LOCK:
            _LOCAL_INDEX_CACHE[f"{req.image_dir}/{req.algorithm_dir}/{custom_id}"] = index

        h, w = index.shape

        print(f"LOCAL SPECTRAL INDEX - {index_name} computed. Range: [{actual_min:.3f}, {actual_max:.3f}]")

        return {
            'name': index_name,
            'index_type': index_type,
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'model_id': custom_id,
            'colormap': {
                'name': colormap_name,
                'min_val': float(actual_min),
                'max_val': float(actual_max),
                'label': label,
            },
            'overlay_meta': {'width': int(w), 'height': int(h)}
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Local spectral index error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Local Threshold Range
# =============================================================================

class LocalThresholdRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    model_id: str
    min_threshold: float
    max_threshold: float
    colormap: Dict = {}


@router.post("/apply-threshold-range")
def local_apply_threshold_range(req: LocalThresholdRequest):
    """Apply threshold range to create binary mask for local image index."""
    cache_key = f"{req.image_dir}/{req.algorithm_dir}/{req.model_id}"
    with _LOCAL_INDEX_CACHE_LOCK:
        index_data = _LOCAL_INDEX_CACHE.get(cache_key)

    if index_data is None:
        raise HTTPException(status_code=400, detail=f"No cached index data for {req.model_id}. Run analyze first.")

    # The path-keyed entry sometimes holds a raw ndarray (legacy) and sometimes
    # a dict — normalize.
    if isinstance(index_data, dict):
        index_array = index_data.get('index_data') or index_data.get('data')
    else:
        index_array = index_data
    if index_array is None:
        raise HTTPException(status_code=400, detail=f"No cached index data for {req.model_id}.")

    mask = (index_array >= req.min_threshold) & (index_array <= req.max_threshold)
    h, w = mask.shape

    # Red overlay for binary mask
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = 255
    rgba[mask, 3] = 255

    from ..services.gpu_compute import rgba_to_base64_gpu
    overlay_url = rgba_to_base64_gpu(rgba)
    preview_url = _make_preview_b64(rgba[:, :, :3], mask)

    # Register a stable id-keyed entry alongside the path-keyed entry so
    # change-detection can look up this mask by id.
    analysis_id = f"local-sa-{int(time.time() * 1000)}"
    with _LOCAL_INDEX_CACHE_LOCK:
        _LOCAL_INDEX_CACHE[analysis_id] = {
            'index_data': index_array,
            'binary_mask': mask,
            'last_min_threshold': float(req.min_threshold),
            'last_max_threshold': float(req.max_threshold),
            'model_id': req.model_id,
            'image_dir': req.image_dir,
            'algorithm_dir': req.algorithm_dir,
        }

    # Mirror TD response shape: pixel counts so the SA card can show stats.
    n_detected = int(mask.sum())
    total_pixels = int(mask.size)
    detection_percentage = round(100.0 * n_detected / total_pixels, 2) if total_pixels else 0.0

    return {
        'analysis_id': analysis_id,
        'overlay_url': overlay_url,
        'preview_url': preview_url,
        'min_threshold': req.min_threshold,
        'max_threshold': req.max_threshold,
        'detected_pixels': n_detected,
        'detection_percentage': detection_percentage,
    }


# =============================================================================
# Local Pixel Value for Index
# =============================================================================

class LocalIndexPixelRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    model_id: str
    row: int
    col: int


@router.post("/get-index-value")
def local_get_index_value(req: LocalIndexPixelRequest):
    """Get index value at pixel coordinates for a local analysis result."""
    cache_key = f"{req.image_dir}/{req.algorithm_dir}/{req.model_id}"
    with _LOCAL_INDEX_CACHE_LOCK:
        index_data = _LOCAL_INDEX_CACHE.get(cache_key)

    if index_data is None:
        return {"error": f"No cached data for {req.model_id}. Run analyze first."}

    r = max(0, min(req.row, index_data.shape[0] - 1))
    c = max(0, min(req.col, index_data.shape[1] - 1))
    val = float(index_data[r, c])

    if np.isnan(val):
        return {"value": "No data"}
    return {"value": round(val, 4)}


# =============================================================================
# Local Custom Visualization
# =============================================================================

class LocalCustomVizRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    custom_visualization: Dict
    custom_name: str = "Custom Visualization"


@router.post("/custom-visualization")
def local_custom_visualization(req: LocalCustomVizRequest):
    """Create custom visualization for local image (RGB composite or index). GPU-accelerated."""
    from ..services.gpu_compute import compute_normalized_index_gpu, create_index_visualization_gpu, rgba_to_base64_gpu

    viz = req.custom_visualization
    viz_type = viz.get('type')

    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    if viz_type == 'composite':
        bands = viz.get('bands', [])
        if len(bands) < 3:
            raise HTTPException(status_code=400, detail="3 bands required for RGB composite")

        tgt = _get_max_band_shape(req.image_dir, req.algorithm_dir, bands)
        r_data = _read_band(req.image_dir, req.algorithm_dir, bands[0], target_shape=tgt)
        g_data = _read_band(req.image_dir, req.algorithm_dir, bands[1], target_shape=tgt)
        b_data = _read_band(req.image_dir, req.algorithm_dir, bands[2], target_shape=tgt)

        def auto_stretch(d):
            valid = d[np.isfinite(d)]
            lo = float(np.percentile(valid, 2)) if len(valid) > 0 else 0
            hi = float(np.percentile(valid, 98)) if len(valid) > 0 else 1
            return _stretch_to_uint8(d, lo, hi)

        rgb = np.stack([auto_stretch(r_data), auto_stretch(g_data), auto_stretch(b_data)], axis=-1)
        h, w = rgb.shape[:2]

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        overlay_url = rgba_to_base64_gpu(rgba)
        preview_url = _make_preview_b64(rgb)

        result = {
            'name': req.custom_name,
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'type': 'composite',
        }

    elif viz_type == 'index':
        band_a_file = viz.get('bandA')
        band_b_file = viz.get('bandB')
        colormap = viz.get('colormap', 'viridis')

        tgt = _get_max_band_shape(req.image_dir, req.algorithm_dir, [band_a_file, band_b_file])
        a_data = _read_band(req.image_dir, req.algorithm_dir, band_a_file, target_shape=tgt)
        b_data = _read_band(req.image_dir, req.algorithm_dir, band_b_file, target_shape=tgt)

        index = compute_normalized_index_gpu(a_data, b_data)

        finite = index[np.isfinite(index)]
        is_normalized = (band_a_file != band_b_file)
        if is_normalized:
            vmin, vmax = -1.0, 1.0
        elif len(finite) > 0:
            vmin = float(np.percentile(finite, 0.5))
            vmax = float(np.percentile(finite, 99.5))
        else:
            vmin, vmax = 0.0, 1.0

        index_rgb, actual_min, actual_max = create_index_visualization_gpu(index, colormap, vmin, vmax)
        finite_mask = np.isfinite(index)
        h, w = index.shape

        # Cache for threshold
        custom_id = f"local-custom-{int(time.time() * 1000)}"
        with _LOCAL_INDEX_CACHE_LOCK:
            _LOCAL_INDEX_CACHE[custom_id] = index

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = index_rgb
        rgba[finite_mask, 3] = 255

        overlay_url = rgba_to_base64_gpu(rgba)
        preview_url = _make_preview_b64(index_rgb, finite_mask)

        a_label = band_a_file.replace('after_', '').replace('.tif', '')
        b_label = band_b_file.replace('after_', '').replace('.tif', '')

        result = {
            'name': req.custom_name,
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'type': 'index',
            'custom_id': custom_id,
            'colormap': {
                'name': colormap,
                'min_val': float(actual_min),
                'max_val': float(actual_max),
                'label': f'{a_label}-{b_label} Index'
            }
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown visualization type: {viz_type}")

    return result


# =============================================================================
# Local Target Detection
# =============================================================================

class LocalTargetDetectionRequest(BaseModel):
    image_dir: str
    algorithm_dir: str
    target_points: List[Dict]  # [{"row": int, "col": int}, ...]
    negative_points: Optional[List[Dict]] = None
    algorithm: str = "SAM"
    selected_bands: Optional[List[int]] = None
    auto_threshold: bool = True
    threshold_percentile: float = 95.0


@router.post("/target-detection/run")
def local_run_target_detection(req: LocalTargetDetectionRequest):
    """Run target detection entirely on GPU using cached tensor. Zero CPU↔GPU transfer."""
    from ..services.gpu_compute import (
        get_gpu_image, load_image_to_gpu, detect_from_gpu_tensor,
        create_index_visualization_gpu, rgba_to_base64_gpu, build_rgba_and_preview_gpu,
    )

    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    # Get GPU-cached image (loaded during analyze, or load now)
    gpu_cache_key = f"{req.image_dir}/{req.algorithm_dir}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
        if not band_files:
            raise HTTPException(status_code=400, detail="No band files found")
        band_arrays = _read_all_bands_resampled(req.image_dir, req.algorithm_dir, band_files)
        gpu_image = load_image_to_gpu(gpu_cache_key, band_arrays)

    h, w, n_bands = gpu_image.shape
    print(f"LOCAL TD - GPU image: {h}x{w}x{n_bands}, device={gpu_image.device}, points={req.target_points}")

    algo_upper = req.algorithm.upper()
    if algo_upper in ('MLP_AMF', 'MLP_ACE'):
        # MLP needs numpy cube + pixel points via _run_mlp_detection
        from ..services.target_detection import _run_mlp_detection, ImageDataLoader

        cube = gpu_image.cpu().numpy()  # (H, W, C)
        if req.selected_bands and len(req.selected_bands) > 0:
            cube = cube[:, :, req.selected_bands]

        loader = ImageDataLoader.__new__(ImageDataLoader)
        loader.data = cube
        loader.transform = None
        loader.crs = None
        loader.raster_path = None

        target_pixels = [(int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0)))) for p in req.target_points]
        neg_pixels = [(int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0)))) for p in (req.negative_points or [])]

        result = _run_mlp_detection(
            loader, target_pixels, neg_pixels,
            algo_upper, req.selected_bands,
        )
    else:
        # Run detection entirely on GPU
        result = detect_from_gpu_tensor(
            gpu_image, req.target_points, req.algorithm,
            selected_bands=req.selected_bands,
        )

    detection_map = result['detection_map']
    threshold = result['threshold']
    min_val = result['min_val']
    max_val = result['max_val']

    # Visualizations
    detection_rgb, _, _ = create_index_visualization_gpu(detection_map, 'jet', min_val, max_val)
    binary_mask = detection_map >= threshold
    valid = np.isfinite(detection_map)

    det_rgba, det_preview = build_rgba_and_preview_gpu(detection_rgb, valid)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[binary_mask, 0] = 255
    mask_rgba, mask_preview = build_rgba_and_preview_gpu(mask_rgb, binary_mask)

    detection_id = f"local-td-{int(time.time() * 1000)}"

    detection_overlay_url = rgba_to_base64_gpu(det_rgba)
    mask_overlay_url = rgba_to_base64_gpu(mask_rgba)

    # Cache
    with _LOCAL_TD_CACHE_LOCK:
        # Initial run does NOT publish a binary_mask. Change-detection
        # requires the user to explicitly apply a threshold first.
        _LOCAL_TD_CACHE[detection_id] = {
            'detection_map': detection_map,
            'algorithm': req.algorithm,
            'threshold': threshold,
            'min_val': min_val,
            'max_val': max_val,
            'image_size': (h, w),
            'target_spectrum': result['target_spectrum'],
        }

    # Previews
    det_preview_url = _preview_to_b64(det_preview)
    mask_preview_url = _preview_to_b64(mask_preview)

    # Statistics
    n_detected = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    detection_percentage = round(100 * n_detected / total_pixels, 2)

    # Charts
    score_dist_chart = _create_local_score_chart(detection_map, threshold, req.algorithm)
    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
    band_labels = [bf.replace('after_', '').replace('.tif', '') for bf in band_files]
    if req.selected_bands:
        band_labels = [band_labels[i] for i in req.selected_bands if i < len(band_labels)]
    spectrum_chart = _create_local_spectrum_chart(
        result['target_spectrum'],
        result['background_mean'],
        result['background_std'],
        band_labels,
        req.algorithm
    )

    return {
        'detection_id': detection_id,
        'algorithm': req.algorithm,
        'threshold': float(threshold),
        'min_val': min_val,
        'max_val': max_val,
        'detected_pixels': n_detected,
        'total_pixels': total_pixels,
        'detection_percentage': detection_percentage,
        'target_spectrum': result['target_spectrum'],
        'detection_result': {
            'name': f'{req.algorithm} Detection Score',
            'preview_url': det_preview_url,
            'overlay_url': detection_overlay_url,
            'type': 'detection_score',
            'colormap': {
                'name': 'jet',
                'min_val': min_val,
                'max_val': max_val,
                'label': f'{req.algorithm} Score'
            },
        },
        'mask_result': {
            'name': f'{req.algorithm} Detection Mask (threshold: {threshold:.4f})',
            'preview_url': mask_preview_url,
            'overlay_url': mask_overlay_url,
            'type': 'detection_mask',
        },
        'charts': {
            'score_distribution': score_dist_chart,
            'spectrum_comparison': spectrum_chart,
        },
        'used_bands': result['used_bands'],
        'training_chart': _make_training_chart(result, req.algorithm),
    }


@router.post("/target-detection/run-stream")
async def local_run_target_detection_stream(req: LocalTargetDetectionRequest):
    """
    SSE streaming variant of `/api/local/target-detection/run`. Emits `training`
    events each MLP step so the frontend can animate the loss curve while the
    model trains, then a final `done` event with the full response payload.
    Non-MLP algorithms go straight to `done` — the endpoint still works but
    without intermediate events.
    """
    from ..services.gpu_compute import (
        get_gpu_image, load_image_to_gpu, detect_from_gpu_tensor,
        create_index_visualization_gpu, rgba_to_base64_gpu, build_rgba_and_preview_gpu,
    )
    from ..services.target_detection import _run_mlp_detection, ImageDataLoader

    algo_upper = req.algorithm.upper()
    is_mlp = algo_upper in ('MLP_AMF', 'MLP_ACE')

    # ---- Pre-checks (raise synchronously so the client sees a clean 4xx) ----
    dir_path = os.path.join(LOCAL_BASE_DIR, req.image_dir, req.algorithm_dir)
    if not os.path.isdir(dir_path):
        raise HTTPException(status_code=404, detail="Directory not found")

    gpu_cache_key = f"{req.image_dir}/{req.algorithm_dir}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        band_files_pre = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))
        if not band_files_pre:
            raise HTTPException(status_code=400, detail="No band files found")
        band_arrays = _read_all_bands_resampled(req.image_dir, req.algorithm_dir, band_files_pre)
        gpu_image = load_image_to_gpu(gpu_cache_key, band_arrays)

    h, w, n_bands = gpu_image.shape
    band_files = sorted(f for f in os.listdir(dir_path) if f.endswith('.tif') and f.startswith('after_'))

    # Prepare MLP loader / points once so the worker thread can reuse them.
    mlp_loader = None
    mlp_target_pixels = None
    mlp_neg_pixels = None
    if is_mlp:
        cube = gpu_image.cpu().numpy()
        if req.selected_bands and len(req.selected_bands) > 0:
            cube = cube[:, :, req.selected_bands]
        mlp_loader = ImageDataLoader.__new__(ImageDataLoader)
        mlp_loader.data = cube
        mlp_loader.transform = None
        mlp_loader.crs = None
        mlp_loader.raster_path = None
        mlp_target_pixels = [
            (int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0))))
            for p in req.target_points
        ]
        mlp_neg_pixels = [
            (int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0))))
            for p in (req.negative_points or [])
        ]

    def event_generator():
        progress_queue = queue.Queue() if is_mlp else None

        def on_progress(data):
            if progress_queue is not None:
                progress_queue.put(data)

        result_holder = [None]
        error_holder = [None]

        def run_in_thread():
            try:
                if is_mlp:
                    result_holder[0] = _run_mlp_detection(
                        mlp_loader, mlp_target_pixels, mlp_neg_pixels,
                        algo_upper, req.selected_bands,
                        progress_callback=on_progress,
                    )
                else:
                    result_holder[0] = detect_from_gpu_tensor(
                        gpu_image, req.target_points, req.algorithm,
                        selected_bands=req.selected_bands,
                    )
            except Exception as e:
                error_holder[0] = e
            finally:
                if progress_queue is not None:
                    progress_queue.put(None)

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()

        # Stream MLP training steps as they arrive.
        if is_mlp and progress_queue is not None:
            while True:
                try:
                    data = progress_queue.get(timeout=60)
                except queue.Empty:
                    break
                if data is None:
                    break
                evt = {
                    "step": data.get("step", 0),
                    "n_steps": data.get("n_steps", 100),
                    "loss": data["loss_history"][-1] if data.get("loss_history") else None,
                    "loss_history": data.get("loss_history", []),
                }
                yield f"event: training\ndata: {json.dumps(evt)}\n\n"

        t.join(timeout=600)

        if error_holder[0]:
            yield f"event: error\ndata: {json.dumps({'detail': str(error_holder[0])})}\n\n"
            return
        result = result_holder[0]
        if result is None:
            yield f"event: error\ndata: {json.dumps({'detail': 'Detection returned no result'})}\n\n"
            return

        try:
            final = _build_local_td_response(
                req, result, (h, w),
                band_labels_source=[bf.replace('after_', '').replace('.tif', '') for bf in band_files],
                selected_bands=req.selected_bands,
            )
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            return
        yield f"event: done\ndata: {json.dumps(final, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _build_local_td_response(
    req: "LocalTargetDetectionRequest | UploadedTargetDetectionRequest",
    result: Dict,
    hw: tuple,
    band_labels_source: List[str],
    selected_bands: Optional[List[int]],
) -> Dict:
    """Shared response builder for local and uploaded target-detection paths.

    Pulled out of the non-streaming endpoints so the streaming variants can
    assemble the same payload after the MLP training events have drained.
    """
    from ..services.gpu_compute import (
        create_index_visualization_gpu, rgba_to_base64_gpu, build_rgba_and_preview_gpu,
    )
    h, w = hw
    detection_map = result['detection_map']
    threshold = result['threshold']
    min_val = result['min_val']
    max_val = result['max_val']

    detection_rgb, _, _ = create_index_visualization_gpu(detection_map, 'jet', min_val, max_val)
    binary_mask = detection_map >= threshold
    valid = np.isfinite(detection_map)

    det_rgba, det_preview = build_rgba_and_preview_gpu(detection_rgb, valid)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[binary_mask, 0] = 255
    mask_rgba, mask_preview = build_rgba_and_preview_gpu(mask_rgb, binary_mask)

    detection_id = f"local-td-{int(time.time() * 1000)}"
    detection_overlay_url = rgba_to_base64_gpu(det_rgba)
    mask_overlay_url = rgba_to_base64_gpu(mask_rgba)

    with _LOCAL_TD_CACHE_LOCK:
        # Initial run does NOT publish a binary_mask. Change-detection
        # requires the user to explicitly apply a threshold first.
        _LOCAL_TD_CACHE[detection_id] = {
            'detection_map': detection_map,
            'algorithm': req.algorithm,
            'threshold': threshold,
            'min_val': min_val,
            'max_val': max_val,
            'image_size': (h, w),
            'target_spectrum': result['target_spectrum'],
        }

    det_preview_url = _preview_to_b64(det_preview)
    mask_preview_url = _preview_to_b64(mask_preview)

    n_detected = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    detection_percentage = round(100 * n_detected / total_pixels, 2)

    band_labels = list(band_labels_source)
    if selected_bands:
        band_labels = [band_labels[i] for i in selected_bands if i < len(band_labels)]

    score_dist_chart = _create_local_score_chart(detection_map, threshold, req.algorithm)
    spectrum_chart = _create_local_spectrum_chart(
        result['target_spectrum'],
        result['background_mean'],
        result['background_std'],
        band_labels,
        req.algorithm,
    )

    return {
        'detection_id': detection_id,
        'algorithm': req.algorithm,
        'threshold': float(threshold),
        'min_val': min_val,
        'max_val': max_val,
        'detected_pixels': n_detected,
        'total_pixels': total_pixels,
        'detection_percentage': detection_percentage,
        'target_spectrum': result['target_spectrum'],
        'detection_result': {
            'name': f'{req.algorithm} Detection Score',
            'preview_url': det_preview_url,
            'overlay_url': detection_overlay_url,
            'type': 'detection_score',
            'colormap': {
                'name': 'jet',
                'min_val': min_val,
                'max_val': max_val,
                'label': f'{req.algorithm} Score',
            },
        },
        'mask_result': {
            'name': f'{req.algorithm} Detection Mask (threshold: {threshold:.4f})',
            'preview_url': mask_preview_url,
            'overlay_url': mask_overlay_url,
            'type': 'detection_mask',
        },
        'charts': {
            'score_distribution': score_dist_chart,
            'spectrum_comparison': spectrum_chart,
        },
        'used_bands': result['used_bands'],
        'training_chart': _make_training_chart(result, req.algorithm),
    }


class LocalTDThresholdRequest(BaseModel):
    detection_id: str
    min_threshold: float
    max_threshold: float


@router.post("/target-detection/apply-threshold")
def local_apply_td_threshold(req: LocalTDThresholdRequest):
    """Apply new threshold to cached local target detection result."""
    with _LOCAL_TD_CACHE_LOCK:
        cached = _LOCAL_TD_CACHE.get(req.detection_id)

    if not cached:
        raise HTTPException(status_code=400, detail=f"Detection result not found: {req.detection_id}")

    detection_map = cached['detection_map']
    h, w = detection_map.shape

    binary_mask = (detection_map >= req.min_threshold) & (detection_map <= req.max_threshold)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[binary_mask, 0] = 255

    mask_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = mask_rgb
    mask_rgba[binary_mask, 3] = 255

    from ..services.gpu_compute import rgba_to_base64_gpu
    mask_overlay_url = rgba_to_base64_gpu(mask_rgba)
    mask_preview = _make_preview_b64(mask_rgb, binary_mask)

    # Persist the freshly computed binary mask so change-detection can
    # use it without re-running threshold logic.
    with _LOCAL_TD_CACHE_LOCK:
        cached['binary_mask'] = binary_mask
        cached['last_min_threshold'] = float(req.min_threshold)
        cached['last_max_threshold'] = float(req.max_threshold)
        _LOCAL_TD_CACHE[req.detection_id] = cached

    n_detected = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    detection_percentage = round(100 * n_detected / total_pixels, 2)

    return {
        'detection_id': req.detection_id,
        'min_threshold': req.min_threshold,
        'max_threshold': req.max_threshold,
        'detected_pixels': n_detected,
        'total_pixels': total_pixels,
        'detection_percentage': detection_percentage,
        'mask_result': {
            'name': f'{cached["algorithm"]} Detection Mask ({req.min_threshold:.3f} - {req.max_threshold:.3f})',
            'preview_url': mask_preview,
            'overlay_url': mask_overlay_url,
            'type': 'detection_mask',
        }
    }


# =============================================================================
# Chart helpers for local target detection
# =============================================================================

def _create_local_score_chart(detection_map, threshold, algorithm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    valid = detection_map[np.isfinite(detection_map)].flatten()
    ax.hist(valid, bins=60, alpha=0.6, color='#3498DB', density=True, edgecolor='white', linewidth=0.5, label='All Pixels')
    target_vals = valid[valid >= threshold]
    if len(target_vals) > 0:
        ax.hist(target_vals, bins=30, alpha=0.7, color='#E74C3C', density=True, edgecolor='white', linewidth=0.5, label='Target Region')
    ax.axvline(x=threshold, color='#2C3E50', linestyle='--', linewidth=2.5, label=f'Threshold: {threshold:.4f}')
    ax.set_xlabel('Detection Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{algorithm} Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = pyio.BytesIO()
    plt.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def _create_local_spectrum_chart(target_spectrum, bg_mean, bg_std, band_labels, algorithm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    n = len(target_spectrum)
    x = np.arange(n)
    bg_m = np.array(bg_mean)
    bg_s = np.array(bg_std)
    tgt = np.array(target_spectrum)
    ax.fill_between(x, bg_m - bg_s, bg_m + bg_s, alpha=0.3, color='#3498DB', label='Background +/- 1 sigma')
    ax.plot(x, bg_m, '--', linewidth=2.5, color='#2980B9', label='Background Mean')
    ax.plot(x, tgt, '-', linewidth=3, color='#E74C3C', label='Target', marker='o', markersize=6)
    ax.set_xlabel('Band', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title(f'Target vs Background Spectrum ({algorithm})', fontsize=14, fontweight='bold')
    if len(band_labels) == n:
        ax.set_xticks(x)
        ax.set_xticklabels(band_labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = pyio.BytesIO()
    plt.savefig(buf, format='png', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


# =============================================================================
# Uploaded GeoTIFF: helpers
# =============================================================================

def _read_uploaded_band(upload_id: str, band_index: int) -> np.ndarray:
    """Read a specific band (1-based) from an uploaded multi-band GeoTIFF."""
    meta = _UPLOADED_IMAGE_META.get(upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail=f"Upload not found: {upload_id}")

    cache_key = f"uploaded/{upload_id}/band_{band_index}"
    if cache_key in _LOCAL_RASTER_CACHE:
        return _LOCAL_RASTER_CACHE[cache_key]

    filepath = meta["filepath"]
    with rasterio.open(filepath) as src:
        if band_index < 1 or band_index > src.count:
            raise HTTPException(status_code=400, detail=f"Band {band_index} out of range (1-{src.count})")
        data = src.read(band_index).astype(np.float32)

    _LOCAL_RASTER_CACHE[cache_key] = data
    return data


# =============================================================================
# Upload GeoTIFF endpoint
# =============================================================================

@router.post("/upload-geotiff")
async def upload_geotiff(file: UploadFile = File(...)):
    """Upload a GeoTIFF, extract geo-metadata, return upload_id + info."""
    upload_id = hashlib.md5(f"{file.filename}-{time.time()}".encode()).hexdigest()
    ext = os.path.splitext(file.filename)[1] or ".tif"
    save_path = os.path.join(UPLOAD_DIR, f"{upload_id}{ext}")

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    try:
        with rasterio.open(save_path) as src:
            crs = src.crs.to_string() if src.crs else None
            bounds = src.bounds
            band_count = src.count
            width = src.width
            height = src.height
            band_names = []
            for i in range(band_count):
                desc = src.descriptions[i] if src.descriptions[i] else f"{i + 1}"
                band_names.append(desc)

            # Reproject bounds to EPSG:4326 for Leaflet
            if src.crs and str(src.crs) != "EPSG:4326":
                from rasterio.warp import transform_bounds
                bounds_4326 = list(transform_bounds(src.crs, "EPSG:4326",
                                                    bounds.left, bounds.bottom, bounds.right, bounds.top))
            else:
                bounds_4326 = [bounds.left, bounds.bottom, bounds.right, bounds.top]
    except Exception as e:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Invalid GeoTIFF: {e}")

    if not crs:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="GeoTIFF has no CRS (coordinate reference system)")

    meta = {
        "upload_id": upload_id,
        "filename": file.filename,
        "filepath": save_path,
        "crs": crs,
        "bounds_4326": bounds_4326,  # [west, south, east, north]
        "band_count": band_count,
        "band_names": band_names,
        "width": width,
        "height": height,
    }
    _UPLOADED_IMAGE_META[upload_id] = meta

    print(f"UPLOAD - {file.filename}: {band_count} bands, {width}x{height}, CRS={crs}, bounds_4326={bounds_4326}")

    return meta


# =============================================================================
# Uploaded GeoTIFF: GPU load / stretch
# =============================================================================

class UploadedGpuLoadRequest(BaseModel):
    upload_id: str
    min_val: float = 0
    max_val: float = 3000


@router.post("/uploaded/gpu-load")
def uploaded_gpu_load(req: UploadedGpuLoadRequest):
    """Load uploaded multi-band GeoTIFF to GPU, compute percentiles, stretch."""
    import time as _time
    from ..services.gpu_compute import load_image_to_gpu, get_gpu_image, compute_percentiles_gpu, stretch_bands_gpu

    t0 = _time.time()

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    gpu_cache_key = f"uploaded/{req.upload_id}"
    gpu_image = get_gpu_image(gpu_cache_key)

    if gpu_image is not None:
        print(f"UPLOADED GPU-LOAD - Reusing cached GPU tensor: {gpu_image.shape}")
    else:
        t1 = _time.time()
        band_arrays = [_read_uploaded_band(req.upload_id, i + 1) for i in range(meta["band_count"])]
        print(f"UPLOADED GPU-LOAD - Disk read: {_time.time()-t1:.2f}s ({meta['band_count']} bands)")

        t2 = _time.time()
        gpu_image = load_image_to_gpu(gpu_cache_key, band_arrays)
        print(f"UPLOADED GPU-LOAD - GPU transfer: {_time.time()-t2:.2f}s")

    t3 = _time.time()
    percentiles = compute_percentiles_gpu(gpu_image)
    print(f"UPLOADED GPU-LOAD - Percentiles: {_time.time()-t3:.2f}s")

    t4 = _time.time()
    band_urls_by_idx, w, h = stretch_bands_gpu(gpu_image, req.min_val, req.max_val)
    band_urls = {}
    band_names = meta["band_names"]
    for i in range(meta["band_count"]):
        band_urls[band_names[i]] = band_urls_by_idx[i]
    print(f"UPLOADED GPU-LOAD - Stretch+encode: {_time.time()-t4:.2f}s")

    print(f"UPLOADED GPU-LOAD - Total: {_time.time()-t0:.2f}s")

    return {
        "band_urls": band_urls,
        "width": w,
        "height": h,
        "percentiles": percentiles,
        "band_names": band_names,
    }


class UploadedGpuStretchRequest(BaseModel):
    upload_id: str
    min_val: float = 0
    max_val: float = 3000


@router.post("/uploaded/gpu-stretch")
def uploaded_gpu_stretch(req: UploadedGpuStretchRequest):
    """Re-stretch uploaded image using GPU cache."""
    import time as _time
    from ..services.gpu_compute import get_gpu_image, stretch_bands_gpu

    t0 = _time.time()

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    gpu_cache_key = f"uploaded/{req.upload_id}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        raise HTTPException(status_code=400, detail="Image not on GPU. Call /uploaded/gpu-load first.")

    band_urls_by_idx, w, h = stretch_bands_gpu(gpu_image, req.min_val, req.max_val)
    band_urls = {}
    band_names = meta["band_names"]
    for i in range(meta["band_count"]):
        band_urls[band_names[i]] = band_urls_by_idx[i]

    print(f"UPLOADED GPU-STRETCH - Total: {_time.time()-t0:.2f}s")

    return {
        "band_urls": band_urls,
        "width": w,
        "height": h,
    }


class UploadedGpuStretchSpec(BaseModel):
    slot: str = Field(..., description="Caller-defined slot id (e.g. 'r', 'g', 'b')")
    band_name: str = Field(..., description="Band name from meta['band_names']")
    min_val: float = 0.0
    max_val: float = 3000.0


class UploadedGpuStretchMultiRequest(BaseModel):
    upload_id: str
    specs: List[UploadedGpuStretchSpec]


@router.post("/uploaded/gpu-stretch-multi")
def uploaded_gpu_stretch_multi(req: UploadedGpuStretchMultiRequest):
    """Re-stretch arbitrary (band, range) tuples — one stretch per spec.

    Used by the symbology UI to compute per-channel R/G/B grayscales when each
    channel may want a different display range. Same band may appear twice with
    different ranges (e.g. R uses Band 4 at [0,3000] while G also uses Band 4
    at [200,2500]); the function stretches twice and returns slot-keyed URLs.
    """
    import time as _time
    from ..services.gpu_compute import get_gpu_image, stretch_bands_gpu_multi

    t0 = _time.time()

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    gpu_cache_key = f"uploaded/{req.upload_id}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        raise HTTPException(status_code=400, detail="Image not on GPU. Call /uploaded/gpu-load first.")

    band_names = meta["band_names"]
    name_to_idx = {name: i for i, name in enumerate(band_names)}

    gpu_specs = []
    for s in req.specs:
        if s.band_name not in name_to_idx:
            raise HTTPException(status_code=400, detail=f"Unknown band: {s.band_name}")
        gpu_specs.append({
            "key": s.slot,
            "band_idx": name_to_idx[s.band_name],
            "min_val": s.min_val,
            "max_val": s.max_val,
        })

    slot_urls, w, h = stretch_bands_gpu_multi(gpu_image, gpu_specs)
    print(f"UPLOADED GPU-STRETCH-MULTI ({len(gpu_specs)} specs) - Total: {_time.time()-t0:.2f}s")

    return {
        "slot_urls": slot_urls,
        "width": w,
        "height": h,
    }


# =============================================================================
# Uploaded GeoTIFF: Spectral Index
# =============================================================================

class UploadedSpectralIndexRequest(BaseModel):
    upload_id: str
    index_type: str
    band_roles: Optional[Dict[str, int]] = None  # {"NIR": 4, "RED": 3, "GREEN": 2, "BLUE": 1} (1-based)
    band_a: Optional[str] = None
    band_b: Optional[str] = None
    colormap: Optional[str] = None


@router.post("/uploaded/compute-spectral-index")
def uploaded_compute_spectral_index(req: UploadedSpectralIndexRequest):
    """Compute spectral index for uploaded GeoTIFF. GPU-accelerated."""
    from ..services.gpu_compute import (
        compute_normalized_index_gpu, compute_savi_gpu,
        create_index_visualization_gpu, build_rgba_and_preview_gpu,
        rgba_to_base64_gpu,
    )
    from ..services.spectral_analysis import INDEX_REGISTRY, safe_divide

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    band_roles = req.band_roles or {}
    index_type = req.index_type.lower()
    print(f"UPLOADED SPECTRAL INDEX - Computing {index_type} for {meta['filename']}")

    def read_role(role):
        idx = band_roles.get(role.upper())
        if idx is None:
            return None
        return _read_uploaded_band(req.upload_id, idx)

    def read_by_name(band_name):
        """Try to read band by name or index for custom index."""
        # Try as band index
        try:
            idx = int(band_name.replace('Band', '').replace('band', '').strip())
            return _read_uploaded_band(req.upload_id, idx)
        except (ValueError, IndexError):
            pass
        # Try matching band name
        for i, name in enumerate(meta["band_names"]):
            if name.upper() == band_name.upper() or band_name.upper() in name.upper():
                return _read_uploaded_band(req.upload_id, i + 1)
        return None

    try:
        if index_type == 'custom':
            if not req.band_a or not req.band_b:
                raise HTTPException(status_code=400, detail="band_a and band_b required for custom index")
            a_data = read_by_name(req.band_a)
            b_data = read_by_name(req.band_b)
            if a_data is None or b_data is None:
                raise HTTPException(status_code=400, detail=f"Could not find bands: {req.band_a}, {req.band_b}")
            index = compute_normalized_index_gpu(a_data, b_data)
            colormap_name = req.colormap or 'viridis'
            vmin, vmax = -1.0, 1.0
            label = f'{req.band_a}-{req.band_b} Index'
            index_name = f'Custom ({req.band_a}/{req.band_b})'
        elif index_type in INDEX_REGISTRY:
            info = INDEX_REGISTRY[index_type]
            colormap_name = req.colormap or info['colormap']
            vmin = info.get('vmin')
            vmax = info.get('vmax')
            label = info['name']
            index_name = info['name']

            nir = read_role('NIR')
            red = read_role('RED')
            green = read_role('GREEN')
            blue = read_role('BLUE')

            if index_type == 'ndvi':
                if nir is None or red is None:
                    raise HTTPException(status_code=400, detail="NIR and RED band roles required for NDVI")
                index = compute_normalized_index_gpu(nir, red)
            elif index_type == 'ndmi':
                swir = read_role('SWIR') or read_role('SWIR1')
                if nir is None or swir is None:
                    raise HTTPException(status_code=400, detail="NIR and SWIR band roles required for NDMI")
                index = compute_normalized_index_gpu(nir, swir)
            elif index_type == 'mvi':
                swir1 = read_role('SWIR') or read_role('SWIR1')
                if nir is None or green is None or swir1 is None:
                    raise HTTPException(status_code=400, detail="NIR, GREEN, SWIR band roles required for MVI")
                index = safe_divide(nir - green, swir1 - green)
            elif index_type == 'ndwi':
                if green is None or nir is None:
                    raise HTTPException(status_code=400, detail="GREEN and NIR band roles required for NDWI")
                index = compute_normalized_index_gpu(green, nir)
            elif index_type == 'savi':
                if nir is None or red is None:
                    raise HTTPException(status_code=400, detail="NIR and RED band roles required for SAVI")
                index = compute_savi_gpu(nir, red)
            elif index_type == 'evi':
                if nir is None or red is None or blue is None:
                    raise HTTPException(status_code=400, detail="NIR, RED, and BLUE band roles required for EVI")
                from ..services.spectral_analysis import calculate_evi
                index = calculate_evi(nir.astype(np.float32), red.astype(np.float32), blue.astype(np.float32))
            else:
                raise HTTPException(status_code=400, detail=f"Unknown index: {index_type}")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown index type: {index_type}")

        index_rgb, actual_min, actual_max = create_index_visualization_gpu(index, colormap_name, vmin, vmax)
        finite_mask = np.isfinite(index)
        rgba, preview = build_rgba_and_preview_gpu(index_rgb, finite_mask)
        overlay_url = rgba_to_base64_gpu(rgba)
        preview_url = _preview_to_b64(preview)

        custom_id = f"uploaded-spectral-{index_type}-{int(time.time() * 1000)}"
        with _LOCAL_INDEX_CACHE_LOCK:
            _LOCAL_INDEX_CACHE[f"uploaded/{req.upload_id}/{custom_id}"] = index

        h, w = index.shape
        print(f"UPLOADED SPECTRAL INDEX - {index_name} computed. Range: [{actual_min:.3f}, {actual_max:.3f}]")

        return {
            'name': index_name,
            'index_type': index_type,
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'model_id': custom_id,
            'colormap': {
                'name': colormap_name,
                'min_val': float(actual_min),
                'max_val': float(actual_max),
                'label': label,
            },
            'overlay_meta': {'width': int(w), 'height': int(h)}
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Uploaded spectral index error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Uploaded GeoTIFF: Threshold
# =============================================================================

class UploadedThresholdRequest(BaseModel):
    upload_id: str
    model_id: str
    min_threshold: float
    max_threshold: float
    colormap: Dict = {}


@router.post("/uploaded/apply-threshold-range")
def uploaded_apply_threshold_range(req: UploadedThresholdRequest):
    """Apply threshold to uploaded image spectral index."""
    cache_key = f"uploaded/{req.upload_id}/{req.model_id}"
    with _LOCAL_INDEX_CACHE_LOCK:
        index_data = _LOCAL_INDEX_CACHE.get(cache_key)

    if index_data is None:
        raise HTTPException(status_code=400, detail=f"No cached index data for {req.model_id}")

    if isinstance(index_data, dict):
        index_array = index_data.get('index_data') or index_data.get('data')
    else:
        index_array = index_data
    if index_array is None:
        raise HTTPException(status_code=400, detail=f"No cached index data for {req.model_id}.")

    mask = (index_array >= req.min_threshold) & (index_array <= req.max_threshold)
    h, w = mask.shape

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask, 0] = 255
    rgba[mask, 3] = 255

    from ..services.gpu_compute import rgba_to_base64_gpu
    overlay_url = rgba_to_base64_gpu(rgba)
    preview_url = _make_preview_b64(rgba[:, :, :3], mask)

    # Register an id-keyed entry alongside the path-keyed one so
    # change-detection can look it up by id.
    analysis_id = f"uploaded-sa-{int(time.time() * 1000)}"
    with _LOCAL_INDEX_CACHE_LOCK:
        _LOCAL_INDEX_CACHE[analysis_id] = {
            'index_data': index_array,
            'binary_mask': mask,
            'last_min_threshold': float(req.min_threshold),
            'last_max_threshold': float(req.max_threshold),
            'model_id': req.model_id,
            'upload_id': req.upload_id,
        }

    n_detected = int(mask.sum())
    total_pixels = int(mask.size)
    detection_percentage = round(100.0 * n_detected / total_pixels, 2) if total_pixels else 0.0

    return {
        'analysis_id': analysis_id,
        'overlay_url': overlay_url,
        'preview_url': preview_url,
        'min_threshold': req.min_threshold,
        'max_threshold': req.max_threshold,
        'detected_pixels': n_detected,
        'detection_percentage': detection_percentage,
    }


# =============================================================================
# Uploaded GeoTIFF: Target Detection
# =============================================================================

class UploadedTargetDetectionRequest(BaseModel):
    upload_id: str
    target_points: List[Dict]  # [{"row": int, "col": int}, ...]
    negative_points: Optional[List[Dict]] = None
    algorithm: str = "SAM"
    selected_bands: Optional[List[int]] = None
    auto_threshold: bool = True
    threshold_percentile: float = 95.0


@router.post("/uploaded/target-detection/run")
def uploaded_run_target_detection(req: UploadedTargetDetectionRequest):
    """Run target detection on uploaded GeoTIFF. GPU-accelerated."""
    from ..services.gpu_compute import (
        get_gpu_image, load_image_to_gpu, detect_from_gpu_tensor,
        create_index_visualization_gpu, rgba_to_base64_gpu, build_rgba_and_preview_gpu,
    )

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    gpu_cache_key = f"uploaded/{req.upload_id}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        band_arrays = [_read_uploaded_band(req.upload_id, i + 1) for i in range(meta["band_count"])]
        gpu_image = load_image_to_gpu(gpu_cache_key, band_arrays)

    h, w, n_bands = gpu_image.shape
    print(f"UPLOADED TD - GPU image: {h}x{w}x{n_bands}, algo={req.algorithm}, points={req.target_points}")

    algo_upper = req.algorithm.upper()
    if algo_upper in ('MLP_AMF', 'MLP_ACE'):
        # MLP needs numpy cube + pixel points via _run_mlp_detection
        from ..services.target_detection import _run_mlp_detection, ImageDataLoader

        cube = gpu_image.cpu().numpy()  # (H, W, C)
        if req.selected_bands and len(req.selected_bands) > 0:
            cube = cube[:, :, req.selected_bands]

        # Build a lightweight loader from numpy data (properties derive from data.shape)
        loader = ImageDataLoader.__new__(ImageDataLoader)
        loader.data = cube
        loader.transform = None
        loader.crs = None
        loader.raster_path = None

        # Convert points to (col, row) tuples
        target_pixels = [(int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0)))) for p in req.target_points]
        neg_pixels = [(int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0)))) for p in (req.negative_points or [])]

        result = _run_mlp_detection(
            loader, target_pixels, neg_pixels,
            algo_upper, req.selected_bands,
        )
    else:
        result = detect_from_gpu_tensor(
            gpu_image, req.target_points, req.algorithm,
            selected_bands=req.selected_bands,
        )

    detection_map = result['detection_map']
    threshold = result['threshold']
    min_val = result['min_val']
    max_val = result['max_val']

    detection_rgb, _, _ = create_index_visualization_gpu(detection_map, 'jet', min_val, max_val)
    binary_mask = detection_map >= threshold
    valid = np.isfinite(detection_map)

    det_rgba, det_preview = build_rgba_and_preview_gpu(detection_rgb, valid)
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    mask_rgb[binary_mask, 0] = 255
    mask_rgba, mask_preview = build_rgba_and_preview_gpu(mask_rgb, binary_mask)

    detection_id = f"uploaded-td-{int(time.time() * 1000)}"

    detection_overlay_url = rgba_to_base64_gpu(det_rgba)
    mask_overlay_url = rgba_to_base64_gpu(mask_rgba)

    with _LOCAL_TD_CACHE_LOCK:
        # Initial run does NOT publish a binary_mask. Change-detection
        # requires the user to explicitly apply a threshold first.
        _LOCAL_TD_CACHE[detection_id] = {
            'detection_map': detection_map,
            'algorithm': req.algorithm,
            'threshold': threshold,
            'min_val': min_val,
            'max_val': max_val,
            'image_size': (h, w),
            'target_spectrum': result['target_spectrum'],
        }

    det_preview_url = _preview_to_b64(det_preview)
    mask_preview_url = _preview_to_b64(mask_preview)

    n_detected = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    detection_percentage = round(100 * n_detected / total_pixels, 2)

    band_labels = meta["band_names"]
    if req.selected_bands:
        band_labels = [band_labels[i] for i in req.selected_bands if i < len(band_labels)]
    score_dist_chart = _create_local_score_chart(detection_map, threshold, req.algorithm)
    spectrum_chart = _create_local_spectrum_chart(
        result['target_spectrum'], result['background_mean'], result['background_std'],
        band_labels, req.algorithm
    )

    return {
        'detection_id': detection_id,
        'algorithm': req.algorithm,
        'threshold': float(threshold),
        'min_val': min_val,
        'max_val': max_val,
        'detected_pixels': n_detected,
        'total_pixels': total_pixels,
        'detection_percentage': detection_percentage,
        'target_spectrum': result['target_spectrum'],
        'detection_result': {
            'name': f'{req.algorithm} Detection Score',
            'preview_url': det_preview_url,
            'overlay_url': detection_overlay_url,
            'type': 'detection_score',
            'colormap': {
                'name': 'jet',
                'min_val': min_val,
                'max_val': max_val,
                'label': f'{req.algorithm} Score'
            },
        },
        'mask_result': {
            'name': f'{req.algorithm} Detection Mask (threshold: {threshold:.4f})',
            'preview_url': mask_preview_url,
            'overlay_url': mask_overlay_url,
            'type': 'detection_mask',
        },
        'charts': {
            'score_distribution': score_dist_chart,
            'spectrum_comparison': spectrum_chart,
        },
        'used_bands': result['used_bands'],
        'training_chart': _make_training_chart(result, req.algorithm),
    }


def _make_training_chart(result, algorithm):
    """Generate training loss chart if loss_history exists."""
    loss_history = result.get('loss_history', [])
    if not loss_history:
        return None
    from .routes_target_detection import create_training_loss_chart
    chart_b64 = create_training_loss_chart(loss_history, algorithm)
    return f"data:image/png;base64,{chart_b64}" if chart_b64 else None


@router.post("/uploaded/target-detection/run-stream")
async def uploaded_run_target_detection_stream(req: UploadedTargetDetectionRequest):
    """
    SSE streaming variant of `/api/local/uploaded/target-detection/run`.
    Emits `training` events for each MLP training step so the loss curve
    animates live in the frontend, followed by a single `done` event with
    the full response. Non-MLP algorithms short-circuit to `done`.
    """
    from ..services.gpu_compute import (
        get_gpu_image, load_image_to_gpu, detect_from_gpu_tensor,
    )
    from ..services.target_detection import _run_mlp_detection, ImageDataLoader

    algo_upper = req.algorithm.upper()
    is_mlp = algo_upper in ('MLP_AMF', 'MLP_ACE')

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    gpu_cache_key = f"uploaded/{req.upload_id}"
    gpu_image = get_gpu_image(gpu_cache_key)
    if gpu_image is None:
        band_arrays = [_read_uploaded_band(req.upload_id, i + 1) for i in range(meta["band_count"])]
        gpu_image = load_image_to_gpu(gpu_cache_key, band_arrays)

    h, w, n_bands = gpu_image.shape

    mlp_loader = None
    mlp_target_pixels = None
    mlp_neg_pixels = None
    if is_mlp:
        cube = gpu_image.cpu().numpy()
        if req.selected_bands and len(req.selected_bands) > 0:
            cube = cube[:, :, req.selected_bands]
        mlp_loader = ImageDataLoader.__new__(ImageDataLoader)
        mlp_loader.data = cube
        mlp_loader.transform = None
        mlp_loader.crs = None
        mlp_loader.raster_path = None
        mlp_target_pixels = [
            (int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0))))
            for p in req.target_points
        ]
        mlp_neg_pixels = [
            (int(p.get('col', p.get('lng', 0))), int(p.get('row', p.get('lat', 0))))
            for p in (req.negative_points or [])
        ]

    def event_generator():
        progress_queue = queue.Queue() if is_mlp else None

        def on_progress(data):
            if progress_queue is not None:
                progress_queue.put(data)

        result_holder = [None]
        error_holder = [None]

        def run_in_thread():
            try:
                if is_mlp:
                    result_holder[0] = _run_mlp_detection(
                        mlp_loader, mlp_target_pixels, mlp_neg_pixels,
                        algo_upper, req.selected_bands,
                        progress_callback=on_progress,
                    )
                else:
                    result_holder[0] = detect_from_gpu_tensor(
                        gpu_image, req.target_points, req.algorithm,
                        selected_bands=req.selected_bands,
                    )
            except Exception as e:
                error_holder[0] = e
            finally:
                if progress_queue is not None:
                    progress_queue.put(None)

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()

        if is_mlp and progress_queue is not None:
            while True:
                try:
                    data = progress_queue.get(timeout=60)
                except queue.Empty:
                    break
                if data is None:
                    break
                evt = {
                    "step": data.get("step", 0),
                    "n_steps": data.get("n_steps", 100),
                    "loss": data["loss_history"][-1] if data.get("loss_history") else None,
                    "loss_history": data.get("loss_history", []),
                }
                yield f"event: training\ndata: {json.dumps(evt)}\n\n"

        t.join(timeout=600)

        if error_holder[0]:
            yield f"event: error\ndata: {json.dumps({'detail': str(error_holder[0])})}\n\n"
            return
        result = result_holder[0]
        if result is None:
            yield f"event: error\ndata: {json.dumps({'detail': 'Detection returned no result'})}\n\n"
            return

        try:
            final = _build_local_td_response(
                req, result, (h, w),
                band_labels_source=meta.get("band_names", []),
                selected_bands=req.selected_bands,
            )
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"
            return
        yield f"event: done\ndata: {json.dumps(final, default=str)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# =============================================================================
# Uploaded GeoTIFF: Pixel value inspection
# =============================================================================

class UploadedPixelValueRequest(BaseModel):
    upload_id: str
    row: int
    col: int


@router.post("/uploaded/get-pixel-values")
def uploaded_get_pixel_values(req: UploadedPixelValueRequest):
    """Get all band values at a pixel for uploaded image."""
    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    bands = []
    for i in range(meta["band_count"]):
        data = _read_uploaded_band(req.upload_id, i + 1)
        r = max(0, min(req.row, data.shape[0] - 1))
        c = max(0, min(req.col, data.shape[1] - 1))
        val = float(data[r, c])
        bands.append({"name": meta["band_names"][i], "value": round(val, 2)})

    return {"bands": bands, "row": req.row, "col": req.col}


class UploadedIndexValueRequest(BaseModel):
    upload_id: str
    model_id: str
    row: int
    col: int


@router.post("/uploaded/get-index-value")
def uploaded_get_index_value(req: UploadedIndexValueRequest):
    """Get spectral index value at a pixel for uploaded image."""
    cache_key = f"uploaded/{req.upload_id}/{req.model_id}"
    with _LOCAL_INDEX_CACHE_LOCK:
        index_data = _LOCAL_INDEX_CACHE.get(cache_key)

    if index_data is None:
        return {"error": "No cached index data"}

    r = max(0, min(req.row, index_data.shape[0] - 1))
    c = max(0, min(req.col, index_data.shape[1] - 1))
    val = float(index_data[r, c])
    return {"value": round(val, 6), "row": req.row, "col": req.col}


# =============================================================================
# Uploaded GeoTIFF: SAM3 Segmentation (point + text modes)
# =============================================================================

from .schemas import UploadedSAM3PointPredictRequest, UploadedSAM3TextPredictRequest


def _read_uploaded_rgb_uint8(meta: dict, rgb_bands):
    """Read & percentile-stretch the uploaded GeoTIFF's RGB bands to uint8."""
    filepath = meta["filepath"]
    h, w = meta["height"], meta["width"]
    n_bands = meta["band_count"]

    if rgb_bands and len(rgb_bands) >= 3:
        r_idx, g_idx, b_idx = rgb_bands[0], rgb_bands[1], rgb_bands[2]
    elif n_bands >= 3:
        r_idx, g_idx, b_idx = 3, 2, 1
    else:
        r_idx = g_idx = b_idx = 1

    with rasterio.open(filepath) as src:
        red = src.read(r_idx).astype(np.float32)
        green = src.read(g_idx).astype(np.float32)
        blue = src.read(b_idx).astype(np.float32)

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
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    return rgb, h, w, n_bands


def _maybe_downscale_for_sam3(rgb: np.ndarray, h: int, w: int):
    """SAM3 max-edge clamp at 2048 px to control memory."""
    MAX_DIM = 2048
    if max(h, w) <= MAX_DIM:
        return rgb, 1.0
    scale = MAX_DIM / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    from PIL import Image as PILImage
    img = PILImage.fromarray(rgb).resize((new_w, new_h), PILImage.LANCZOS)
    print(f"SAM3 UPLOADED - resized {h}x{w} -> {new_h}x{new_w} (scale={scale:.3f})")
    return np.array(img), scale


def _upscale_mask_to_original(mask: np.ndarray, scale_factor: float, w: int, h: int) -> np.ndarray:
    if scale_factor >= 1.0:
        return mask.astype(bool)
    from PIL import Image as PILImage
    mask_pil = PILImage.fromarray(mask.astype(np.uint8) * 255, mode="L")
    mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
    return np.array(mask_pil) > 127


@router.post("/uploaded/sam3/predict")
def uploaded_sam3_predict(req: UploadedSAM3PointPredictRequest):
    """SAM3 single-instance segmentation on an uploaded GeoTIFF (point/box mode)."""
    from ..services.sam3_service import (
        is_sam3_ready, init_sam3, get_sam3_status,
        encode_image, predict_mask,
    )
    from .routes_sam3 import (
        SAM3_MASK_CACHE, SAM3_MASK_CACHE_LOCK, make_mask_overlay_native,
    )

    if not is_sam3_ready() and not init_sam3():
        raise HTTPException(
            status_code=503,
            detail=f"SAM3 model not available: {get_sam3_status().get('error', 'Unknown')}",
        )

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    try:
        rgb, h, w, _ = _read_uploaded_rgb_uint8(meta, req.rgb_bands)
        rgb, scale_factor = _maybe_downscale_for_sam3(rgb, h, w)

        cache_key = f"uploaded/{req.upload_id}"
        encode_image(cache_key, rgb)

        # SAM3 (like SAM2) takes (x, y) = (col, row).
        positive_pixels = [
            (int(p.col * scale_factor), int(p.row * scale_factor))
            for p in req.positive_points
        ]
        negative_pixels = (
            [
                (int(p.col * scale_factor), int(p.row * scale_factor))
                for p in req.negative_points
            ]
            if req.negative_points else None
        )

        print(f"SAM3 UPLOADED - positive: {positive_pixels}")
        print(f"SAM3 UPLOADED - negative: {negative_pixels}")

        mask, score = predict_mask(cache_key, positive_pixels, negative_pixels)
        mask = _upscale_mask_to_original(mask, scale_factor, w, h)

        overlay_url, preview_url = make_mask_overlay_native(mask)

        mask_id = f"sam3-uploaded-{int(time.time() * 1000)}"
        with SAM3_MASK_CACHE_LOCK:
            SAM3_MASK_CACHE[mask_id] = {
                "mask": mask,
                "binary_mask": mask,  # alias used by change-detection lookup
                "upload_id": req.upload_id,
                "overlay_url": overlay_url,
                "preview_url": preview_url,
                "score": score,
                "pixel_count": int(mask.sum()),
                "saved": False,
                "mode": "point",
                # No transform/crs for uploaded images — change-detection
                # treats this as local-mode (pixel-aligned).
                "transform": None,
                "crs": None,
            }

        return {
            "mask_id": mask_id,
            "score": float(score),
            "overlay_url": overlay_url,
            "preview_url": preview_url,
            "pixel_count": int(mask.sum()),
            "total_pixels": int(mask.size),
            "overlay_meta": {"width": w, "height": h},
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"SAM3 uploaded predict error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM3 prediction failed: {e}")


@router.post("/uploaded/sam3/text-predict")
def uploaded_sam3_text_predict(req: UploadedSAM3TextPredictRequest):
    """SAM3 PCS on an uploaded GeoTIFF — every instance matching the text prompt."""
    from ..services.sam3_service import (
        is_sam3_ready, init_sam3, get_sam3_status,
        encode_image, predict_text,
    )
    from .routes_sam3 import (
        SAM3_MASK_CACHE, SAM3_MASK_CACHE_LOCK,
        make_mask_overlay_native, instance_color, rgb_to_hex,
    )

    if not is_sam3_ready() and not init_sam3():
        raise HTTPException(
            status_code=503,
            detail=f"SAM3 model not available: {get_sam3_status().get('error', 'Unknown')}",
        )

    meta = _UPLOADED_IMAGE_META.get(req.upload_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt must be non-empty")

    try:
        rgb, h, w, _ = _read_uploaded_rgb_uint8(meta, req.rgb_bands)
        rgb, scale_factor = _maybe_downscale_for_sam3(rgb, h, w)

        cache_key = f"uploaded/{req.upload_id}"
        encode_image(cache_key, rgb)

        masks, scores, boxes = predict_text(
            cache_key, req.prompt, score_threshold=req.score_threshold or 0.5,
        )

        ts = int(time.time() * 1000)
        n = len(masks)
        total_for_color = max(n, 8)

        instances = []
        for i, (m, sc, bx) in enumerate(zip(masks, scores, boxes)):
            full_mask = _upscale_mask_to_original(m, scale_factor, w, h)
            color = instance_color(i, total_for_color)
            overlay_url, preview_url = make_mask_overlay_native(
                full_mask, rgb_color=color
            )
            mask_id = f"sam3-uploaded-text-{ts}-{i}"
            with SAM3_MASK_CACHE_LOCK:
                SAM3_MASK_CACHE[mask_id] = {
                    "mask": full_mask,
                    "binary_mask": full_mask,
                    "upload_id": req.upload_id,
                    "overlay_url": overlay_url,
                    "preview_url": preview_url,
                    "score": sc,
                    "pixel_count": int(full_mask.sum()),
                    "saved": False,
                    "mode": "text",
                    "prompt": req.prompt,
                    "color": color,
                    "transform": None,
                    "crs": None,
                }
            # Box was returned in (possibly downscaled) prediction space.
            if scale_factor < 1.0:
                bx = tuple(v / scale_factor for v in bx)
            instances.append({
                "mask_id": mask_id,
                "score": float(sc),
                "overlay_url": overlay_url,
                "preview_url": preview_url,
                "pixel_count": int(full_mask.sum()),
                "color_hex": rgb_to_hex(color),
                "bbox_pixel": list(bx),
                "overlay_meta": {"width": w, "height": h},
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
        print(f"SAM3 uploaded text-predict error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"SAM3 text prediction failed: {e}")
