"""
API routes for analysis, visualization, and pixel operations.
"""

import os
import re
import time
import base64
from typing import Dict, List, Optional

import numpy as np
import rasterio
from PIL import Image
import io as pyio
import requests

from fastapi import APIRouter, HTTPException

from .schemas import (
    CustomVisualizationRequest,
    PixelValueRequest,
    ThresholdRangeRequest,
)
from ..core.config import PROJECT_ROOT, OUTPUTS_DIR
from ..services.earth_engine import bbox_to_geometry
from ..services.visualization import (
    stretch_uint8,
    create_index_visualization,
    save_rgba_overlay_png,
    save_rgba_overlay_png_with_transparency,
    warp_rgb_and_mask_to_aoi,
    to_png_bytes,
    generate_overlay_path,
)
from ..services.spectral_analysis import safe_divide
from ..utils.cache import (
    get_cached_raster_path,
    get_pixel_value_from_cache,
    get_custom_viz_cache,
    set_custom_viz_cache,
    find_matching_index_cache_key,
    bbox_to_cache_key,
    RASTER_FILE_CACHE,
    RASTER_CACHE_LOCK,
    INDEX_DATA_CACHE,
    INDEX_CACHE_LOCK,
)


router = APIRouter(prefix="/api", tags=["analysis"])


def _create_rgb_composite(custom_viz: Dict, image_id: str, bbox: List[float]) -> Dict:
    """Create RGB composite visualization using cached raster."""
    bands = custom_viz['bands']
    
    # Convert band names
    converted_bands = []
    for band in bands:
        if band.startswith('B0'):
            converted_bands.append(band[1:])
        else:
            converted_bands.append(band)
    
    # Get cached raster
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        cached_raster_path = RASTER_FILE_CACHE.get(cache_key)
    
    if not cached_raster_path or not os.path.exists(cached_raster_path):
        raise HTTPException(status_code=400, detail="No cached raster data found. Please process the image first.")
    
    print(f"RGB COMPOSITE - Using cached raster: {cached_raster_path}")
    
    with rasterio.open(cached_raster_path) as src:
        # Band mapping for 10 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        band_mapping = {'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5, 'B7': 6, 'B8': 7, 'B8A': 8, 'B11': 9, 'B12': 10}
        
        if len(converted_bands) >= 3:
            try:
                r_idx = band_mapping.get(converted_bands[0], 1)
                g_idx = band_mapping.get(converted_bands[1], 2)
                b_idx = band_mapping.get(converted_bands[2], 3)
                r_band = src.read(r_idx)
                g_band = src.read(g_idx)
                b_band = src.read(b_idx)
            except Exception:
                r_band = src.read(1)
                g_band = src.read(2) if src.count >= 2 else src.read(1)
                b_band = src.read(3) if src.count >= 3 else src.read(1)
        else:
            r_band = src.read(1)
            g_band = src.read(1) if len(converted_bands) < 2 else src.read(2)
            b_band = src.read(1) if len(converted_bands) < 3 else src.read(3)
        
        src_transform = src.transform
        src_crs = src.crs
    
    rgb_composite = np.dstack([stretch_uint8(r_band), stretch_uint8(g_band), stretch_uint8(b_band)])
    
    min_lon, min_lat, max_lon, max_lat = bbox
    valid_mask = np.ones(rgb_composite.shape[:2], dtype=bool)
    
    aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
        rgb_composite, valid_mask, src_transform, src_crs,
        (min_lon, min_lat, max_lon, max_lat), scale_m=10
    )
    
    preview_data = to_png_bytes(rgb_composite)
    
    overlay_png_path = generate_overlay_path("rgb_composite_overlay")
    Image.fromarray(aoi_rgb, mode='RGB').save(overlay_png_path, format='PNG')
    
    preview_b64 = base64.b64encode(preview_data).decode('utf-8')
    preview_url = f"data:image/png;base64,{preview_b64}"
    overlay_url = f"/api/proxy-file?path={requests.utils.quote('file://' + overlay_png_path)}"
    
    return {
        'name': f'RGB Composite ({"-".join(converted_bands)})',
        'description': f'RGB composite using bands {", ".join(converted_bands)}',
        'preview_url': preview_url,
        'overlay_url': overlay_url,
        'type': 'composite',
        'overlay_meta': {
            'width': int(aoi_w),
            'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        }
    }


def _create_index_visualization(custom_viz: Dict, image_id: str, bbox: List[float]) -> Dict:
    """Create index visualization using cached raster."""
    band_a = custom_viz['bandA']
    band_b = custom_viz['bandB']
    colormap = custom_viz.get('colormap', 'viridis')
    
    converted_band_a = band_a[1:] if band_a.startswith('B0') else band_a
    converted_band_b = band_b[1:] if band_b.startswith('B0') else band_b
    
    is_normalized_index = (converted_band_a != converted_band_b)
    
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        cached_raster_path = RASTER_FILE_CACHE.get(cache_key)
    
    if not cached_raster_path or not os.path.exists(cached_raster_path):
        raise HTTPException(status_code=400, detail="No cached raster data found. Please process the image first.")
    
    with rasterio.open(cached_raster_path) as src:
        # Band mapping for 10 bands: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        band_mapping = {'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5, 'B7': 6, 'B8': 7, 'B8A': 8, 'B11': 9, 'B12': 10}
        
        try:
            a_idx = band_mapping.get(converted_band_a, 1)
            b_idx = band_mapping.get(converted_band_b, 2)
            a_data = src.read(a_idx).astype(np.float32)
            b_data = src.read(b_idx).astype(np.float32)
        except Exception:
            a_data = src.read(1).astype(np.float32)
            b_data = src.read(2).astype(np.float32)
        
        src_transform = src.transform
        src_crs = src.crs
        
        denominator = a_data + b_data
        mask_valid = denominator != 0
        index = np.zeros_like(a_data)
        index[mask_valid] = (a_data[mask_valid] - b_data[mask_valid]) / denominator[mask_valid]
        
        if is_normalized_index:
            vmin, vmax = -1.0, 1.0
        else:
            finite_mask = np.isfinite(index)
            if np.any(finite_mask):
                vmin = np.percentile(index[finite_mask], 0.5)
                vmax = np.percentile(index[finite_mask], 99.5)
            else:
                vmin, vmax = np.nanmin(index), np.nanmax(index)
        
        index_rgb, actual_min, actual_max = create_index_visualization(index, colormap, vmin, vmax)
        finite_index = np.isfinite(index)
        
        # Cache index data
        custom_id = f"custom-result-{int(time.time() * 1000)}"
        with INDEX_CACHE_LOCK:
            INDEX_DATA_CACHE[custom_id] = {
                'data': index,
                'transform': src_transform,
                'crs': src_crs,
                'bbox': bbox
            }
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            index_rgb, finite_index, src_transform, src_crs,
            (min_lon, min_lat, max_lon, max_lat), scale_m=10
        )
        
        rgba_preview = np.zeros((*index_rgb.shape[:2], 4), dtype=np.uint8)
        rgba_preview[:, :, :3] = index_rgb
        rgba_preview[:, :, 3] = 255
        rgba_preview[~finite_index, 3] = 0
        
        preview_img = Image.fromarray(rgba_preview, mode='RGBA')
        preview_img.thumbnail((512, 512), Image.LANCZOS)
        preview_buf = pyio.BytesIO()
        preview_img.save(preview_buf, format='PNG')
        preview_data = preview_buf.getvalue()
        
        overlay_png_path = generate_overlay_path("index_overlay")
        save_rgba_overlay_png(aoi_rgb, aoi_mask, overlay_png_path)
        
        preview_b64 = base64.b64encode(preview_data).decode('utf-8')
        preview_url = f"data:image/png;base64,{preview_b64}"
        overlay_url = f"/api/proxy-file?path={requests.utils.quote('file://' + overlay_png_path)}"
        
        return {
            'name': f'Index ({converted_band_a}-{converted_band_b})/({converted_band_a}+{converted_band_b})',
            'description': f'Custom index using {converted_band_a} and {converted_band_b}',
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'type': 'index',
            'cache_key': custom_id,
            'colormap': {
                'name': colormap,
                'min_val': float(actual_min),
                'max_val': float(actual_max),
                'label': f'{converted_band_a}-{converted_band_b} Index'
            },
            'overlay_meta': {
                'width': int(aoi_w),
                'height': int(aoi_h),
                'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
            }
        }


def _create_script_visualization(custom_viz: Dict, image_id: str, bbox: List[float]) -> Dict:
    """Create custom script visualization."""
    script = custom_viz['script']
    
    band_pattern = r"['\"]?(B\w+)['\"]?"
    bands_found = re.findall(band_pattern, script)
    
    if not bands_found:
        bands_found = ['B4', 'B3', 'B2']
    
    converted_bands = []
    for band in bands_found:
        if band.startswith('B0'):
            converted_bands.append(band[1:])
        else:
            converted_bands.append(band)
    
    rgb_bands = converted_bands[:3] if len(converted_bands) >= 3 else converted_bands + ['B4', 'B3', 'B2'][:3-len(converted_bands)]
    
    result = _create_rgb_composite({'bands': rgb_bands}, image_id, bbox)
    result['name'] = 'Custom Script Result'
    result['description'] = f'Custom visualization using bands: {", ".join(rgb_bands)}'
    result['type'] = 'script'
    
    return result


@router.post("/custom-visualization")
def create_custom_visualization(req: CustomVisualizationRequest):
    """Create custom visualization based on user-defined parameters."""
    try:
        print(f"Creating custom visualization for image {req.image_id}")
        
        custom_viz = req.custom_visualization
        viz_type = custom_viz.get('type')
        
        if viz_type == 'composite':
            result = _create_rgb_composite(custom_viz, req.image_id, req.bbox)
        elif viz_type == 'index':
            result = _create_index_visualization(custom_viz, req.image_id, req.bbox)
        elif viz_type == 'script':
            result = _create_script_visualization(custom_viz, req.image_id, req.bbox)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown visualization type: {viz_type}")
        
        if req.custom_name:
            result['name'] = req.custom_name
        
        custom_id = result.get('cache_key', f"custom-result-{int(time.time() * 1000)}")
        
        set_custom_viz_cache(custom_id, {
            'image_id': req.image_id,
            'type': viz_type,
            'visualization': custom_viz,
            'result': result,
            'cache_key': custom_id
        })
        
        result['custom_id'] = custom_id
        return result
        
    except Exception as e:
        print(f"Error in custom visualization: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-pixel-value")
def get_pixel_value(req: PixelValueRequest):
    """Get pixel value at specific coordinates from cached data."""
    try:
        print(f"Getting pixel value for {req.image_id} at ({req.lat}, {req.lng}) for {req.model_id}")
        
        cache_key = None
        
        if req.model_id in ['model1', 'model2', 'model3', 'model4']:
            cache_key = find_matching_index_cache_key(req.image_id, req.model_id)
        elif req.model_id.startswith('custom-result-'):
            cache_key = req.model_id
        
        if cache_key:
            value = get_pixel_value_from_cache(cache_key, req.lat, req.lng)
            if value is not None:
                print(f"PIXEL VALUE - Retrieved: {value}")
                return {"value": round(value, 4)}
        
        if req.model_id.startswith('custom-result-'):
            custom_info = get_custom_viz_cache(req.model_id)
            if custom_info:
                if custom_info['type'] == 'composite':
                    return {"value": "RGB composite - no single value"}
                elif custom_info['type'] == 'script':
                    return {"value": "Custom script - no single value"}
        
        return {"error": f"No cached data available for {req.model_id}. Please process the image first."}
        
    except Exception as e:
        print(f"Error getting pixel value: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply-threshold-range")
def apply_threshold_range(req: ThresholdRangeRequest):
    """Apply threshold range to create a binary mask."""
    try:
        print(f"Applying threshold range {req.min_threshold:.3f}-{req.max_threshold:.3f} to {req.model_id}")
        
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        
        cache_key = None
        if req.model_id in ['model2', 'model3', 'model4']:
            cache_key = find_matching_index_cache_key(req.image_id, req.model_id)
        elif req.model_id.startswith('custom-result-'):
            cache_key = req.model_id
        
        cached_data = None
        index_data = None
        if cache_key:
            with INDEX_CACHE_LOCK:
                cached_data = INDEX_DATA_CACHE.get(cache_key)
            if cached_data:
                index_data = cached_data['data']
        
        if index_data is None or cached_data is None:
            raise HTTPException(status_code=400, detail=f"No cached index data for {req.model_id}")
        
        src_transform = cached_data['transform']
        src_crs = cached_data['crs']
        bbox = cached_data['bbox']
        
        mask = (index_data >= req.min_threshold) & (index_data <= req.max_threshold)
        
        threshold_rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
        threshold_rgb[mask, 0] = 255
        
        min_lon, min_lat, max_lon, max_lat = bbox
        
        aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            threshold_rgb, mask, src_transform, src_crs,
            (min_lon, min_lat, max_lon, max_lat), scale_m=10
        )
        
        timestamp = int(time.time() * 1000)
        overlay_filename = f'threshold_range_{req.model_id}_aoi_{timestamp}.png'
        overlay_png_path = os.path.join(OUTPUTS_DIR, overlay_filename)
        save_rgba_overlay_png(aoi_rgb, aoi_mask, overlay_png_path)
        
        preview_rgba = np.zeros((*threshold_rgb.shape[:2], 4), dtype=np.uint8)
        preview_rgba[:, :, :3] = threshold_rgb
        preview_rgba[:, :, 3] = 255
        preview_rgba[~mask, 3] = 0
        
        preview_img = Image.fromarray(preview_rgba, mode='RGBA')
        preview_img.thumbnail((256, 256), Image.LANCZOS)
        preview_filename = f'threshold_range_preview_{req.model_id}_{timestamp}.png'
        preview_png_path = os.path.join(OUTPUTS_DIR, preview_filename)
        preview_img.save(preview_png_path)
        
        preview_url = f"/api/proxy-file?path={requests.utils.quote('file://' + preview_png_path)}"
        overlay_url = f"/api/proxy-file?path={requests.utils.quote('file://' + overlay_png_path)}"
        
        return {
            'name': f'{req.colormap.get("label", "Index")} (Range: {req.min_threshold:.3f}-{req.max_threshold:.3f})',
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'min_threshold': req.min_threshold,
            'max_threshold': req.max_threshold,
            'type': 'threshold_range',
            'overlay_meta': {
                'width': int(aoi_w),
                'height': int(aoi_h),
                'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
            }
        }
        
    except Exception as e:
        print(f"Error applying threshold range: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

