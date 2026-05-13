"""
API routes for Target Detection functionality.

Provides endpoints for:
- Running target detection with various algorithms
- Applying threshold to detection results
- Getting target spectrum information
"""

import os
import json
import time
import base64
import queue
import threading
from typing import Dict, List

import numpy as np
from PIL import Image
import io as pyio
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .schemas import (
    TargetDetectionRequest,
    TargetDetectionThresholdRequest,
    TargetSpectrumRequest,
)
from ..core.config import OUTPUTS_DIR
from ..services.target_detection import (
    run_target_detection,
    get_available_detectors,
    ImageDataLoader,
    MLP_ALGORITHMS,
)
from ..services.visualization import (
    create_index_visualization,
    save_rgba_overlay_png,
    warp_rgb_and_mask_to_aoi,
)
from ..utils.cache import (
    bbox_to_cache_key,
    RASTER_FILE_CACHE,
    RASTER_CACHE_LOCK,
)


router = APIRouter(prefix="/api", tags=["target-detection"])

# Cache for target detection results.
# Bounded + TTL: detection results are heavy (numpy arrays + cached state);
# without this the process RSS grew unboundedly across sessions.
# The `_LOCK` is retained for callers that still wrap access in `with` blocks
# — TTLCache is already thread-safe, so the extra lock is harmless.
from ..utils.cache_service import TTLCache
TARGET_DETECTION_CACHE = TTLCache(maxsize=64, ttl=2 * 3600, name="target_detection")
TARGET_DETECTION_CACHE_LOCK = threading.Lock()

# Sentinel-2 band names
S2_BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']


def create_training_loss_chart(loss_history: list, algorithm: str) -> str:
    """Create training loss curve chart and return as base64 PNG."""
    if not loss_history:
        return ''
    fig, ax = plt.subplots(figsize=(6, 4), dpi=120)
    steps = list(range(1, len(loss_history) + 1))
    ax.plot(steps, loss_history, '-', linewidth=2, color='#7b1fa2', label='Training Loss')
    ax.set_xlabel('Step', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(f'{algorithm} Training Progress', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    buf = pyio.BytesIO()
    plt.savefig(buf, format='png', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_score_distribution_chart(
    detection_map: np.ndarray, 
    threshold: float, 
    algorithm: str,
    target_scores: List[float] = None
) -> str:
    """Create score distribution histogram showing background vs target and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    
    valid_values = detection_map[np.isfinite(detection_map)].flatten()
    
    # Background distribution (all pixels)
    ax.hist(valid_values, bins=60, alpha=0.6, color='#3498DB', 
            density=True, edgecolor='white', linewidth=0.5, label='All Pixels')
    
    # Target region distribution (above threshold)
    target_values = valid_values[valid_values >= threshold]
    if len(target_values) > 0:
        ax.hist(target_values, bins=30, alpha=0.7, color='#E74C3C', 
                density=True, edgecolor='white', linewidth=0.5, label='Target Region')
    
    ax.axvline(x=threshold, color='#2C3E50', linestyle='--', linewidth=2.5,
               label=f'Threshold: {threshold:.4f}')
    
    ax.set_xlabel('Detection Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{algorithm} Score Distribution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    buf = pyio.BytesIO()
    plt.savefig(buf, format='png', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def create_spectrum_chart(
    target_spectrum: List[float],
    background_mean: List[float],
    background_std: List[float],
    used_bands: List[int],
    algorithm: str
) -> str:
    """Create spectrum comparison chart and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    
    n_bands = len(target_spectrum)
    x = np.arange(n_bands)
    
    # Get band labels
    if len(used_bands) == len(S2_BAND_NAMES) and used_bands == list(range(10)):
        labels = S2_BAND_NAMES
    else:
        labels = [S2_BAND_NAMES[i] if i < len(S2_BAND_NAMES) else f'B{i+1}' for i in used_bands]
    
    bg_mean = np.array(background_mean)
    bg_std = np.array(background_std)
    target = np.array(target_spectrum)
    
    # Background shading
    ax.fill_between(x, bg_mean - bg_std, bg_mean + bg_std,
                    alpha=0.3, color='#3498DB', label='Background ± 1σ')
    ax.plot(x, bg_mean, '--', linewidth=2.5, color='#2980B9', label='Background Mean')
    
    # Target spectrum
    ax.plot(x, target, '-', linewidth=3, color='#E74C3C', 
            label='Target', marker='o', markersize=6)
    
    ax.set_xlabel('Band', fontsize=12)
    ax.set_ylabel('Reflectance', fontsize=12)
    ax.set_title(f'Target vs Background Spectrum ({algorithm})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    buf = pyio.BytesIO()
    plt.savefig(buf, format='png', facecolor='white', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@router.get("/target-detection/algorithms")
def get_algorithms():
    """Get list of available target detection algorithms."""
    return {
        'algorithms': get_available_detectors()
    }


@router.post("/target-detection/run-stream")
async def run_detection_stream(req: TargetDetectionRequest):
    """
    Run target detection with SSE streaming for MLP training progress.
    Returns Server-Sent Events: 'training' events during MLP training, then 'done' with full result.
    For non-MLP algorithms, immediately returns 'done' (no training events).
    """
    algorithm_upper = req.algorithm.upper()
    is_mlp = algorithm_upper in MLP_ALGORITHMS

    # Get cached raster path (validate early)
    cache_key = bbox_to_cache_key(req.image_id, req.bbox)
    with RASTER_CACHE_LOCK:
        cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

    if not cached_raster_path or not os.path.exists(cached_raster_path):
        raise HTTPException(status_code=400, detail="No cached raster data found. Please process the image first.")

    target_points_latlon = [(p.lat, p.lng) for p in req.target_points]
    negative_points_latlon = [(p.lat, p.lng) for p in req.negative_points] if req.negative_points else []

    def event_generator():
        progress_queue = queue.Queue() if is_mlp else None

        def on_progress(data):
            """Called from the MLP training thread each step."""
            progress_queue.put(data)

        # Run detection in a background thread so we can stream progress
        result_holder = [None]
        error_holder = [None]

        def run_in_thread():
            try:
                result_holder[0] = run_target_detection(
                    raster_path=cached_raster_path,
                    target_points_latlon=target_points_latlon,
                    negative_points_latlon=negative_points_latlon,
                    algorithm=req.algorithm,
                    threshold_percentile=req.threshold_percentile or 95.0,
                    auto_threshold=req.auto_threshold if req.auto_threshold is not None else True,
                    selected_bands=req.selected_bands,
                    progress_callback=on_progress if is_mlp else None,
                )
            except Exception as e:
                error_holder[0] = e
            finally:
                if progress_queue:
                    progress_queue.put(None)  # sentinel

        t = threading.Thread(target=run_in_thread, daemon=True)
        t.start()

        # Stream training progress for MLP
        if is_mlp and progress_queue:
            while True:
                try:
                    data = progress_queue.get(timeout=60)
                except queue.Empty:
                    break
                if data is None:
                    break  # training done
                evt = {
                    "step": data.get("step", 0),
                    "n_steps": data.get("n_steps", 100),
                    "loss": data["loss_history"][-1] if data.get("loss_history") else None,
                    "loss_history": data.get("loss_history", []),
                }
                yield f"event: training\ndata: {json.dumps(evt)}\n\n"

        t.join(timeout=300)

        if error_holder[0]:
            yield f"event: error\ndata: {json.dumps({'detail': str(error_holder[0])})}\n\n"
            return

        result = result_holder[0]
        if result is None:
            yield f"event: error\ndata: {json.dumps({'detail': 'Detection returned no result'})}\n\n"
            return

        # Build the full response (same logic as run_detection)
        try:
            final = _build_detection_response(req, result)
            yield f"event: done\ndata: {json.dumps(final, default=str)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _build_detection_response(req: TargetDetectionRequest, result: dict) -> dict:
    """Build the JSON response dict from a detection result. Shared by run_detection and run_detection_stream."""
    detection_map = result['detection_map']
    threshold = result['threshold']

    detection_rgb, actual_min, actual_max = create_index_visualization(
        detection_map, cmap='jet', vmin=result['min_val'], vmax=result['max_val']
    )

    binary_mask = detection_map >= threshold
    mask_rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
    mask_rgb[binary_mask, 0] = 255

    min_lon, min_lat, max_lon, max_lat = req.bbox

    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        detection_rgb, np.isfinite(detection_map),
        result['transform'], result['crs'],
        (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=req.geometry
    )

    aoi_binary, aoi_mask_binary, _, _ = warp_rgb_and_mask_to_aoi(
        mask_rgb, binary_mask,
        result['transform'], result['crs'],
        (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=req.geometry
    )

    from ..services.gpu_compute import rgb_mask_to_base64_gpu
    detection_overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)
    mask_overlay_url = rgb_mask_to_base64_gpu(aoi_binary, aoi_mask_binary)

    # Diagnose silent-empty-overlay bugs (observed with MLP_AMF/MLP_ACE).
    # If the warped AOI is entirely masked out, the browser receives a
    # valid-looking data URL that paints nothing.
    valid_pixels = int(np.asarray(aoi_mask).sum()) if aoi_mask is not None else 0
    valid_binary = int(np.asarray(aoi_mask_binary).sum()) if aoi_mask_binary is not None else 0
    print(
        f"TARGET DETECTION - AOI warp: size=({aoi_w}x{aoi_h}), "
        f"score-valid-px={valid_pixels}, mask-valid-px={valid_binary}, "
        f"algorithm={req.algorithm}"
    )
    if valid_pixels == 0:
        print(
            f"TARGET DETECTION - WARNING score overlay has 0 valid pixels. "
            f"bbox={req.bbox}, detection_map shape={detection_map.shape}, "
            f"finite={int(np.isfinite(detection_map).sum())}"
        )

    # Previews
    det_rgba = np.zeros((*detection_rgb.shape[:2], 4), dtype=np.uint8)
    det_rgba[:, :, :3] = detection_rgb
    det_rgba[:, :, 3] = np.where(np.isfinite(detection_map), 255, 0).astype(np.uint8)
    det_preview = Image.fromarray(det_rgba, mode='RGBA')
    det_preview.thumbnail((256, 256), Image.LANCZOS)
    det_buf = pyio.BytesIO(); det_preview.save(det_buf, format='PNG')
    det_preview_b64 = base64.b64encode(det_buf.getvalue()).decode('utf-8')

    mask_rgba = np.zeros((*mask_rgb.shape[:2], 4), dtype=np.uint8)
    mask_rgba[:, :, :3] = mask_rgb; mask_rgba[:, :, 3] = 255; mask_rgba[~binary_mask, 3] = 0
    mask_preview = Image.fromarray(mask_rgba, mode='RGBA')
    mask_preview.thumbnail((256, 256), Image.LANCZOS)
    mask_buf = pyio.BytesIO(); mask_preview.save(mask_buf, format='PNG')
    mask_preview_b64 = base64.b64encode(mask_buf.getvalue()).decode('utf-8')

    # Charts
    score_dist_chart = create_score_distribution_chart(detection_map, threshold, req.algorithm)
    spectrum_chart = create_spectrum_chart(
        result['target_spectrum'], result['background_mean'], result['background_std'], result['used_bands'], req.algorithm
    )

    n_detected = int(np.sum(binary_mask)); total_pixels = int(binary_mask.size)
    detection_percentage = round(100 * n_detected / total_pixels, 2)

    detection_id = f"td-{int(time.time() * 1000)}"
    with TARGET_DETECTION_CACHE_LOCK:
        # Initial run does NOT publish a binary_mask. Change-detection
        # requires the user to explicitly apply a threshold first — that
        # endpoint writes binary_mask + last_min/max_threshold into this
        # entry. The auto-threshold value is still kept around (`threshold`
        # field below) so the UI can pre-populate the slider.
        TARGET_DETECTION_CACHE[detection_id] = {
            'detection_map': detection_map, 'transform': result['transform'],
            'crs': result['crs'], 'algorithm': req.algorithm,
            'bbox': req.bbox, 'geometry': req.geometry,
            'threshold': float(threshold),
        }

    return {
        'detection_id': detection_id, 'algorithm': req.algorithm,
        'threshold': float(threshold),
        'min_val': float(result['min_val']), 'max_val': float(result['max_val']),
        'detected_pixels': n_detected, 'total_pixels': total_pixels,
        'detection_percentage': detection_percentage,
        'detection_result': {
            'name': f'{req.algorithm} Detection Score',
            'preview_url': f"data:image/png;base64,{det_preview_b64}",
            'overlay_url': detection_overlay_url, 'type': 'detection_score',
            'colormap': {'name': 'jet', 'min_val': float(result['min_val']),
                         'max_val': float(result['max_val']), 'label': f'{req.algorithm} Score'},
            'overlay_meta': {'width': int(aoi_w), 'height': int(aoi_h),
                             'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]}
        },
        'mask_result': {
            'name': f'{req.algorithm} Detection Mask (threshold: {threshold:.4f})',
            'preview_url': f"data:image/png;base64,{mask_preview_b64}",
            'overlay_url': mask_overlay_url, 'type': 'detection_mask',
            'overlay_meta': {'width': int(aoi_w), 'height': int(aoi_h),
                             'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]}
        },
        'charts': {
            'score_distribution': f"data:image/png;base64,{score_dist_chart}",
            'spectrum_comparison': f"data:image/png;base64,{spectrum_chart}"
        },
        'used_bands': result['used_bands'],
    }


@router.post("/target-detection/run")
def run_detection(req: TargetDetectionRequest):
    """
    Run target detection on a processed image (non-streaming).
    For MLP algorithms with live training chart, use /target-detection/run-stream instead.
    """
    try:
        print(f"TARGET DETECTION - Running {req.algorithm} on {req.image_id}")

        cache_key = bbox_to_cache_key(req.image_id, req.bbox)
        with RASTER_CACHE_LOCK:
            cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

        if not cached_raster_path or not os.path.exists(cached_raster_path):
            raise HTTPException(status_code=400, detail="No cached raster data found. Please process the image first.")

        target_points_latlon = [(p.lat, p.lng) for p in req.target_points]
        negative_points_latlon = [(p.lat, p.lng) for p in req.negative_points] if req.negative_points else []

        result = run_target_detection(
            raster_path=cached_raster_path,
            target_points_latlon=target_points_latlon,
            negative_points_latlon=negative_points_latlon,
            algorithm=req.algorithm,
            threshold_percentile=req.threshold_percentile or 95.0,
            auto_threshold=req.auto_threshold if req.auto_threshold is not None else True,
            selected_bands=req.selected_bands
        )

        return _build_detection_response(req, result)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in target detection: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/target-detection/apply-threshold")
def apply_detection_threshold(req: TargetDetectionThresholdRequest):
    """
    Apply a new threshold range to existing detection results.
    """
    try:
        print(f"TARGET DETECTION - Applying threshold {req.min_threshold:.3f}-{req.max_threshold:.3f} to {req.detection_id}")
        
        # Get cached detection result
        with TARGET_DETECTION_CACHE_LOCK:
            cached = TARGET_DETECTION_CACHE.get(req.detection_id)
        
        if not cached:
            raise HTTPException(
                status_code=400,
                detail=f"Detection result not found: {req.detection_id}"
            )
        
        detection_map = cached['detection_map']
        
        # Create new binary mask within threshold range
        binary_mask = (detection_map >= req.min_threshold) & (detection_map <= req.max_threshold)
        
        # Create mask overlay (red color)
        mask_rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        mask_rgb[binary_mask, 0] = 255  # Red channel
        
        # Warp to AOI bounds
        min_lon, min_lat, max_lon, max_lat = req.bbox
        
        aoi_binary, aoi_mask_binary, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            mask_rgb,
            binary_mask,
            cached['transform'],
            cached['crs'],
            (min_lon, min_lat, max_lon, max_lat),
            scale_m=10
        )
        
        # Generate overlay base64 (no file I/O)
        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        mask_overlay_url = rgb_mask_to_base64_gpu(aoi_binary, aoi_mask_binary)

        # Preview
        mask_rgba = np.zeros((*mask_rgb.shape[:2], 4), dtype=np.uint8)
        mask_rgba[:, :, :3] = mask_rgb
        mask_rgba[:, :, 3] = 255
        mask_rgba[~binary_mask, 3] = 0
        mask_preview = Image.fromarray(mask_rgba, mode='RGBA')
        mask_preview.thumbnail((256, 256), Image.LANCZOS)
        mask_buf = pyio.BytesIO()
        mask_preview.save(mask_buf, format='PNG')
        mask_preview_b64 = base64.b64encode(mask_buf.getvalue()).decode('utf-8')

        # Persist the freshly computed binary mask so change-detection can
        # use it without re-running threshold logic.
        with TARGET_DETECTION_CACHE_LOCK:
            cached['binary_mask'] = binary_mask
            cached['last_min_threshold'] = float(req.min_threshold)
            cached['last_max_threshold'] = float(req.max_threshold)
            TARGET_DETECTION_CACHE[req.detection_id] = cached

        # Statistics
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
                'preview_url': f"data:image/png;base64,{mask_preview_b64}",
                'overlay_url': mask_overlay_url,
                'type': 'detection_mask',
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
        print(f"Error applying threshold: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/target-detection/get-spectrum")
def get_target_spectrum(req: TargetSpectrumRequest):
    """
    Get spectrum values at a specific pixel location.
    Useful for previewing target pixel before running detection.
    """
    try:
        print(f"TARGET DETECTION - Getting spectrum at ({req.lat}, {req.lng})")
        
        # Get cached raster path
        cache_key = bbox_to_cache_key(req.image_id, req.bbox)
        with RASTER_CACHE_LOCK:
            cached_raster_path = RASTER_FILE_CACHE.get(cache_key)
        
        if not cached_raster_path or not os.path.exists(cached_raster_path):
            raise HTTPException(
                status_code=400,
                detail="No cached raster data found. Please process the image first."
            )
        
        # Load and get spectrum (excluding 60m bands B1/B9 to match detection)
        loader = ImageDataLoader(cached_raster_path)
        from ..services.target_detection import drop_s2_60m_bands
        drop_s2_60m_bands(loader)
        spectrum = loader.get_spectrum_at_latlon(req.lat, req.lng)

        # Get pixel coordinates for reference
        x, y = loader.latlon_to_pixel(req.lat, req.lng)

        # Band names for Sentinel-2 (10 bands, B1 and B9 dropped)
        band_names = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        if len(spectrum) != len(band_names):
            band_names = [f'Band_{i+1}' for i in range(len(spectrum))]
        
        return {
            'lat': req.lat,
            'lng': req.lng,
            'pixel_x': x,
            'pixel_y': y,
            'spectrum': spectrum.tolist(),
            'band_names': band_names,
            'n_bands': len(spectrum)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting spectrum: {e}")
        raise HTTPException(status_code=500, detail=str(e))

