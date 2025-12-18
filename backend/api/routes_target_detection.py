"""
API routes for Target Detection functionality.

Provides endpoints for:
- Running target detection with various algorithms
- Applying threshold to detection results
- Getting target spectrum information
"""

import os
import time
import base64
import threading
from typing import Dict, List

import numpy as np
from PIL import Image
import io as pyio
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from fastapi import APIRouter, HTTPException

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

# Cache for target detection results
TARGET_DETECTION_CACHE: Dict = {}
TARGET_DETECTION_CACHE_LOCK = threading.Lock()

# Sentinel-2 band names
S2_BAND_NAMES = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']


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


@router.post("/target-detection/run")
def run_detection(req: TargetDetectionRequest):
    """
    Run target detection on a processed image.
    
    Requires the image to be processed first (cached raster available).
    """
    try:
        print(f"TARGET DETECTION - Running {req.algorithm} on {req.image_id}")
        print(f"TARGET DETECTION - Target points: {req.target_points}")
        
        # Get cached raster path
        cache_key = bbox_to_cache_key(req.image_id, req.bbox)
        with RASTER_CACHE_LOCK:
            cached_raster_path = RASTER_FILE_CACHE.get(cache_key)
        
        if not cached_raster_path or not os.path.exists(cached_raster_path):
            raise HTTPException(
                status_code=400, 
                detail="No cached raster data found. Please process the image first."
            )
        
        print(f"TARGET DETECTION - Using cached raster: {cached_raster_path}")
        
        # Convert target points to list of (lat, lon) tuples
        target_points_latlon = [(p.lat, p.lng) for p in req.target_points]
        
        # Run detection
        result = run_target_detection(
            raster_path=cached_raster_path,
            target_points_latlon=target_points_latlon,
            algorithm=req.algorithm,
            threshold_percentile=req.threshold_percentile or 95.0,
            auto_threshold=req.auto_threshold if req.auto_threshold is not None else True,
            selected_bands=req.selected_bands
        )
        
        detection_map = result['detection_map']
        threshold = result['threshold']
        
        # Create visualization using jet colormap
        detection_rgb, actual_min, actual_max = create_index_visualization(
            detection_map, 
            cmap='jet',
            vmin=result['min_val'],
            vmax=result['max_val']
        )
        
        # Create binary mask at auto threshold
        binary_mask = detection_map >= threshold
        
        # Create mask overlay (red color for detected pixels)
        mask_rgb = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        mask_rgb[binary_mask, 0] = 255  # Red channel
        
        # Warp to AOI bounds
        min_lon, min_lat, max_lon, max_lat = req.bbox
        
        # Detection map overlay
        aoi_detection, aoi_mask_detection, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            detection_rgb, 
            np.isfinite(detection_map), 
            result['transform'], 
            result['crs'],
            (min_lon, min_lat, max_lon, max_lat), 
            scale_m=10
        )
        
        # Binary mask overlay
        aoi_binary, aoi_mask_binary, _, _ = warp_rgb_and_mask_to_aoi(
            mask_rgb,
            binary_mask,
            result['transform'],
            result['crs'],
            (min_lon, min_lat, max_lon, max_lat),
            scale_m=10
        )
        
        # Generate unique detection ID
        detection_id = f"target-detection-{int(time.time() * 1000)}"
        
        # Save to cache
        with TARGET_DETECTION_CACHE_LOCK:
            TARGET_DETECTION_CACHE[detection_id] = {
                'detection_map': detection_map,
                'transform': result['transform'],
                'crs': result['crs'],
                'bbox': req.bbox,
                'algorithm': req.algorithm,
                'threshold': threshold,
                'min_val': result['min_val'],
                'max_val': result['max_val'],
                'target_spectrum': result['target_spectrum']
            }
        
        # Save overlay files
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        
        # Detection map overlay
        detection_overlay_path = os.path.join(OUTPUTS_DIR, f'{detection_id}_detection.png')
        save_rgba_overlay_png(aoi_detection, aoi_mask_detection, detection_overlay_path)
        
        # Binary mask overlay
        mask_overlay_path = os.path.join(OUTPUTS_DIR, f'{detection_id}_mask.png')
        save_rgba_overlay_png(aoi_binary, aoi_mask_binary, mask_overlay_path)
        
        # Create preview images
        # Detection preview
        detection_preview = Image.fromarray(detection_rgb, mode='RGB')
        detection_preview.thumbnail((256, 256), Image.LANCZOS)
        detection_buf = pyio.BytesIO()
        detection_preview.save(detection_buf, format='PNG')
        detection_preview_b64 = base64.b64encode(detection_buf.getvalue()).decode('utf-8')
        
        # Mask preview
        mask_rgba = np.zeros((*mask_rgb.shape[:2], 4), dtype=np.uint8)
        mask_rgba[:, :, :3] = mask_rgb
        mask_rgba[:, :, 3] = 255
        mask_rgba[~binary_mask, 3] = 0
        
        mask_preview = Image.fromarray(mask_rgba, mode='RGBA')
        mask_preview.thumbnail((256, 256), Image.LANCZOS)
        mask_buf = pyio.BytesIO()
        mask_preview.save(mask_buf, format='PNG')
        mask_preview_b64 = base64.b64encode(mask_buf.getvalue()).decode('utf-8')
        
        # Calculate detected pixel statistics
        n_detected = int(np.sum(binary_mask))
        total_pixels = int(binary_mask.size)
        detection_percentage = round(100 * n_detected / total_pixels, 2)
        
        detection_overlay_url = f"/api/proxy-file?path={requests.utils.quote('file://' + detection_overlay_path)}"
        mask_overlay_url = f"/api/proxy-file?path={requests.utils.quote('file://' + mask_overlay_path)}"
        
        # Generate analysis charts
        score_dist_chart = create_score_distribution_chart(detection_map, threshold, req.algorithm)
        spectrum_chart = create_spectrum_chart(
            result['target_spectrum'],
            result['background_mean'],
            result['background_std'],
            result['used_bands'],
            req.algorithm
        )
        
        return {
            'detection_id': detection_id,
            'algorithm': req.algorithm,
            'threshold': float(threshold),
            'min_val': float(result['min_val']),
            'max_val': float(result['max_val']),
            'detected_pixels': n_detected,
            'total_pixels': total_pixels,
            'detection_percentage': detection_percentage,
            'target_spectrum': result['target_spectrum'],
            'detection_result': {
                'name': f'{req.algorithm} Detection Score',
                'preview_url': f"data:image/png;base64,{detection_preview_b64}",
                'overlay_url': detection_overlay_url,
                'type': 'detection_score',
                'colormap': {
                    'name': 'jet',
                    'min_val': float(result['min_val']),
                    'max_val': float(result['max_val']),
                    'label': f'{req.algorithm} Score'
                },
                'overlay_meta': {
                    'width': int(aoi_w),
                    'height': int(aoi_h),
                    'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
                }
            },
            'mask_result': {
                'name': f'{req.algorithm} Detection Mask (threshold: {threshold:.4f})',
                'preview_url': f"data:image/png;base64,{mask_preview_b64}",
                'overlay_url': mask_overlay_url,
                'type': 'detection_mask',
                'overlay_meta': {
                    'width': int(aoi_w),
                    'height': int(aoi_h),
                    'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
                }
            },
            'charts': {
                'score_distribution': f"data:image/png;base64,{score_dist_chart}",
                'spectrum_comparison': f"data:image/png;base64,{spectrum_chart}"
            },
            'used_bands': result['used_bands']
        }
        
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
    Apply a new threshold to existing detection results.
    """
    try:
        print(f"TARGET DETECTION - Applying threshold {req.threshold} to {req.detection_id}")
        
        # Get cached detection result
        with TARGET_DETECTION_CACHE_LOCK:
            cached = TARGET_DETECTION_CACHE.get(req.detection_id)
        
        if not cached:
            raise HTTPException(
                status_code=400,
                detail=f"Detection result not found: {req.detection_id}"
            )
        
        detection_map = cached['detection_map']
        
        # Create new binary mask at specified threshold
        binary_mask = detection_map >= req.threshold
        
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
        
        # Save overlay
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        timestamp = int(time.time() * 1000)
        mask_overlay_path = os.path.join(OUTPUTS_DIR, f'{req.detection_id}_mask_{timestamp}.png')
        save_rgba_overlay_png(aoi_binary, aoi_mask_binary, mask_overlay_path)
        
        # Create preview
        mask_rgba = np.zeros((*mask_rgb.shape[:2], 4), dtype=np.uint8)
        mask_rgba[:, :, :3] = mask_rgb
        mask_rgba[:, :, 3] = 255
        mask_rgba[~binary_mask, 3] = 0
        
        mask_preview = Image.fromarray(mask_rgba, mode='RGBA')
        mask_preview.thumbnail((256, 256), Image.LANCZOS)
        mask_buf = pyio.BytesIO()
        mask_preview.save(mask_buf, format='PNG')
        mask_preview_b64 = base64.b64encode(mask_buf.getvalue()).decode('utf-8')
        
        # Statistics
        n_detected = int(np.sum(binary_mask))
        total_pixels = int(binary_mask.size)
        detection_percentage = round(100 * n_detected / total_pixels, 2)
        
        mask_overlay_url = f"/api/proxy-file?path={requests.utils.quote('file://' + mask_overlay_path)}"
        
        return {
            'detection_id': req.detection_id,
            'threshold': req.threshold,
            'detected_pixels': n_detected,
            'total_pixels': total_pixels,
            'detection_percentage': detection_percentage,
            'mask_result': {
                'name': f'{cached["algorithm"]} Detection Mask (threshold: {req.threshold:.4f})',
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
        
        # Load and get spectrum
        loader = ImageDataLoader(cached_raster_path)
        spectrum = loader.get_spectrum_at_latlon(req.lat, req.lng)
        
        # Get pixel coordinates for reference
        x, y = loader.latlon_to_pixel(req.lat, req.lng)
        
        # Band names for Sentinel-2 (10 bands in standard order)
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

