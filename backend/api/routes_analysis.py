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
from pydantic import BaseModel

from .schemas import (
    CustomVisualizationRequest,
    PixelValueRequest,
    ThresholdRangeRequest,
    ComputeSpectralIndexRequest,
)
from ..core.config import PROJECT_ROOT, OUTPUTS_DIR, S2_BAND_MAPPING
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
from ..services.spectral_analysis import (
    safe_divide,
    calculate_spectral_index,
    calculate_custom_normalized_index,
    calculate_savi,
    calculate_evi,
    get_band_data_from_raster,
    INDEX_REGISTRY,
)
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
        r_idx = S2_BAND_MAPPING[converted_bands[0]]
        g_idx = S2_BAND_MAPPING[converted_bands[1]]
        b_idx = S2_BAND_MAPPING[converted_bands[2]]
        r_band = src.read(r_idx)
        g_band = src.read(g_idx)
        b_band = src.read(b_idx)
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
    
    from ..services.gpu_compute import rgb_mask_to_base64_gpu
    overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)

    preview_b64 = base64.b64encode(preview_data).decode('utf-8')
    preview_url = f"data:image/png;base64,{preview_b64}"
    
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
        a_idx = S2_BAND_MAPPING[converted_band_a]
        b_idx = S2_BAND_MAPPING[converted_band_b]
        a_data = src.read(a_idx).astype(np.float32)
        b_data = src.read(b_idx).astype(np.float32)
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
        
        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)

        preview_b64 = base64.b64encode(preview_data).decode('utf-8')
        preview_url = f"data:image/png;base64,{preview_b64}"
        
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


@router.get("/available-indices")
def get_available_indices():
    """Return all supported spectral indices."""
    return {k: {'name': v['name'], 'full_name': v.get('full_name', v['name']),
                'formula': v.get('formula', ''), 'bands': v['bands']}
            for k, v in INDEX_REGISTRY.items()}


@router.post("/compute-spectral-index")
def compute_spectral_index(req: ComputeSpectralIndexRequest):
    """Compute a spectral index on-demand from cached raster data."""
    try:
        index_type = req.index_type.lower()
        print(f"SPECTRAL INDEX - Computing {index_type} for {req.image_id}")

        cache_key = bbox_to_cache_key(req.image_id, req.bbox)
        with RASTER_CACHE_LOCK:
            cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

        if not cached_raster_path or not os.path.exists(cached_raster_path):
            raise HTTPException(status_code=400,
                                detail="No cached raster data found. Please process the image first.")

        with rasterio.open(cached_raster_path) as src:
            data = src.read().astype(np.float32)
            src_transform = src.transform
            src_crs = src.crs
            n_bands = src.count

        # Raster band order (12 bands): B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12
        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']
        if n_bands != len(band_names):
            raise HTTPException(status_code=400, detail=f"Unexpected band count: {n_bands}")
        band_dict = {name: data[i] for i, name in enumerate(band_names)}
        mask_bool = np.ones(data.shape[1:], dtype=bool)

        # Apply mask
        def apply_mask(arr):
            res = arr.copy()
            res[~mask_bool] = np.nan
            return res

        if index_type == 'custom':
            if not req.band_a or not req.band_b:
                raise HTTPException(status_code=400, detail="band_a and band_b required for custom index")
            a_data = apply_mask(band_dict.get(req.band_a, band_dict.get(req.band_a.replace('B0', 'B'), data[0])))
            b_data = apply_mask(band_dict.get(req.band_b, band_dict.get(req.band_b.replace('B0', 'B'), data[1])))
            index = safe_divide(a_data - b_data, a_data + b_data)
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

            if index_type == 'ndvi':
                index = safe_divide(apply_mask(band_dict['B8']) - apply_mask(band_dict['B4']),
                                    apply_mask(band_dict['B8']) + apply_mask(band_dict['B4']))
            elif index_type == 'ndmi':
                index = safe_divide(apply_mask(band_dict['B8']) - apply_mask(band_dict['B12']),
                                    apply_mask(band_dict['B8']) + apply_mask(band_dict['B12']))
            elif index_type == 'mvi':
                index = safe_divide(apply_mask(band_dict['B8']) - apply_mask(band_dict['B3']),
                                    apply_mask(band_dict['B11']) - apply_mask(band_dict['B3']))
            elif index_type == 'ndwi':
                index = safe_divide(apply_mask(band_dict['B3']) - apply_mask(band_dict['B8']),
                                    apply_mask(band_dict['B3']) + apply_mask(band_dict['B8']))
            elif index_type == 'savi':
                index = calculate_savi(apply_mask(band_dict['B8']), apply_mask(band_dict['B4']))
            elif index_type == 'evi':
                index = calculate_evi(apply_mask(band_dict['B8']), apply_mask(band_dict['B4']),
                                      apply_mask(band_dict['B2']))
            else:
                raise HTTPException(status_code=400, detail=f"Unknown index: {index_type}")
        else:
            raise HTTPException(status_code=400, detail=f"Unknown index type: {index_type}")

        # Visualization
        index_rgb, actual_min, actual_max = create_index_visualization(index, colormap_name, vmin, vmax)
        finite_mask = np.isfinite(index) & mask_bool

        # Cache index data for pixel inspection and thresholding
        custom_id = f"spectral-{index_type}-{int(time.time() * 1000)}"
        with INDEX_CACHE_LOCK:
            INDEX_DATA_CACHE[custom_id] = {
                'data': index,
                'transform': src_transform,
                'crs': src_crs,
                'bbox': req.bbox
            }

        min_lon, min_lat, max_lon, max_lat = req.bbox

        aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            index_rgb, finite_mask, src_transform, src_crs,
            (min_lon, min_lat, max_lon, max_lat), scale_m=10,
            geometry=req.geometry
        )

        # Preview thumbnail
        rgba_preview = np.zeros((*index_rgb.shape[:2], 4), dtype=np.uint8)
        rgba_preview[:, :, :3] = index_rgb
        rgba_preview[:, :, 3] = 255
        rgba_preview[~finite_mask, 3] = 0

        preview_img = Image.fromarray(rgba_preview, mode='RGBA')
        preview_img.thumbnail((512, 512), Image.LANCZOS)
        preview_buf = pyio.BytesIO()
        preview_img.save(preview_buf, format='PNG')
        preview_b64 = base64.b64encode(preview_buf.getvalue()).decode('utf-8')
        preview_url = f"data:image/png;base64,{preview_b64}"

        # Overlay
        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)

        print(f"SPECTRAL INDEX - {index_name} computed. Range: [{actual_min:.3f}, {actual_max:.3f}]")

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
            'overlay_meta': {
                'width': int(aoi_w),
                'height': int(aoi_h),
                'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error computing spectral index: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-pixel-value")
def get_pixel_value(req: PixelValueRequest):
    """Get pixel value at specific coordinates from cached data."""
    try:
        print(f"Getting pixel value for {req.image_id} at ({req.lat}, {req.lng}) for {req.model_id}")
        
        # model1 is segmentation - no continuous pixel value
        if req.model_id == 'model1':
            return {"value": "Segmentation model - categorical output"}
        
        cache_key = None
        
        # For spectral indices (model2=NDVI, model3=NDMI, model4=MVI)
        if req.model_id in ['model2', 'model3', 'model4']:
            cache_key = find_matching_index_cache_key(req.image_id, req.model_id)
            
            # Debug: log available cache keys
            with INDEX_CACHE_LOCK:
                available_keys = list(INDEX_DATA_CACHE.keys())
                print(f"PIXEL VALUE - Looking for {req.model_id}, available keys: {available_keys}")
            
        elif req.model_id.startswith('custom-result-') or req.model_id.startswith('spectral-'):
            cache_key = req.model_id

        elif req.model_id.startswith('td-'):
            # Target detection results — look up in TARGET_DETECTION_CACHE
            from .routes_target_detection import TARGET_DETECTION_CACHE, TARGET_DETECTION_CACHE_LOCK
            detection_key = req.model_id[3:]  # strip 'td-' prefix
            with TARGET_DETECTION_CACHE_LOCK:
                cached = TARGET_DETECTION_CACHE.get(detection_key)
            if cached:
                from ..utils.cache import _lat_lng_to_pixel_index
                data = cached['detection_map']
                transform = cached['transform']
                src_crs = cached.get('crs')
                height, width = data.shape

                target_lng, target_lat = req.lng, req.lat
                if src_crs and str(src_crs).upper() != 'EPSG:4326':
                    try:
                        from rasterio.warp import transform as rasterio_transform
                        xs, ys = rasterio_transform('EPSG:4326', src_crs, [req.lng], [req.lat])
                        target_lng, target_lat = xs[0], ys[0]
                    except Exception:
                        pass

                row_idx, col_idx = _lat_lng_to_pixel_index(target_lat, target_lng, transform, height, width)
                if row_idx is not None and col_idx is not None:
                    value = float(data[row_idx, col_idx])
                    if np.isfinite(value):
                        return {"value": round(value, 4)}
                return {"value": "Outside image bounds"}

        if cache_key:
            value = get_pixel_value_from_cache(cache_key, req.lat, req.lng)
            if value is not None:
                print(f"PIXEL VALUE - Retrieved: {value}")
                return {"value": round(value, 4)}
            else:
                # Value might be None if coordinates are outside the image bounds
                return {"value": "Outside image bounds"}
        
        if req.model_id.startswith('custom-result-'):
            custom_info = get_custom_viz_cache(req.model_id)
            if custom_info:
                if custom_info['type'] == 'composite':
                    return {"value": "RGB composite - no single value"}
                elif custom_info['type'] == 'script':
                    return {"value": "Custom script - no single value"}
        
        # More helpful error message
        model_names = {'model2': 'NDVI', 'model3': 'NDMI', 'model4': 'MVI'}
        model_name = model_names.get(req.model_id, req.model_id)
        return {"error": f"No {model_name} data cached. Please run 'Process Image' first."}
        
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
        elif req.model_id.startswith('custom-result-') or req.model_id.startswith('spectral-'):
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
        
        from ..services.gpu_compute import rgb_mask_to_base64_gpu, rgba_to_base64_gpu
        overlay_url = rgb_mask_to_base64_gpu(aoi_rgb, aoi_mask)

        preview_rgba = np.zeros((*threshold_rgb.shape[:2], 4), dtype=np.uint8)
        preview_rgba[:, :, :3] = threshold_rgb
        preview_rgba[:, :, 3] = 255
        preview_rgba[~mask, 3] = 0
        preview_url = rgba_to_base64_gpu(preview_rgba, max_dim=256)

        # Persist the binary mask under the same INDEX_DATA_CACHE entry so
        # change-detection can look it up by `analysis_id` (= cache_key).
        with INDEX_CACHE_LOCK:
            cached_data['binary_mask'] = mask
            cached_data['last_min_threshold'] = float(req.min_threshold)
            cached_data['last_max_threshold'] = float(req.max_threshold)
            INDEX_DATA_CACHE[cache_key] = cached_data

        # Mirror the target-detection contract: report how many source pixels
        # met the threshold so the SA result card can display "X px / Y%"
        # like TD does.
        n_detected = int(mask.sum())
        total_pixels = int(mask.size)
        detection_percentage = round(100.0 * n_detected / total_pixels, 2) if total_pixels else 0.0

        return {
            'analysis_id': cache_key,
            'name': f'{req.colormap.get("label", "Index")} (Range: {req.min_threshold:.3f}-{req.max_threshold:.3f})',
            'preview_url': preview_url,
            'overlay_url': overlay_url,
            'min_threshold': req.min_threshold,
            'max_threshold': req.max_threshold,
            'detected_pixels': n_detected,
            'detection_percentage': detection_percentage,
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


class SpectralValuesRequest(BaseModel):
    image_id: str
    bbox: List[float]
    lat: float
    lng: float


@router.post("/get-spectral-values")
def get_spectral_values(req: SpectralValuesRequest):
    """Get all band values at a pixel location from the cached raster."""
    try:
        cache_key = bbox_to_cache_key(req.image_id, req.bbox)
        with RASTER_CACHE_LOCK:
            cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

        if not cached_raster_path or not os.path.exists(cached_raster_path):
            raise HTTPException(status_code=400, detail="No cached raster. Process the image first.")

        from rasterio.transform import rowcol
        from pyproj import Transformer

        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        with rasterio.open(cached_raster_path) as src:
            if src.crs and str(src.crs).upper() != 'EPSG:4326':
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                x, y = transformer.transform(req.lng, req.lat)
            else:
                x, y = req.lng, req.lat

            row, col = rowcol(src.transform, x, y)
            row, col = int(row), int(col)

            if row < 0 or row >= src.height or col < 0 or col >= src.width:
                return {"bands": [], "error": "Outside image bounds"}

            bands = []
            n_bands = min(src.count, len(band_names))
            for i in range(n_bands):
                val = float(src.read(i + 1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                bands.append({"name": band_names[i] if i < len(band_names) else f"Band_{i+1}", "value": round(val, 2)})

        return {"bands": bands, "lat": req.lat, "lng": req.lng}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AreaSpectralStatsRequest(BaseModel):
    image_id: str
    bbox: List[float]
    points: List[dict]  # [{lat, lng}, ...]


@router.post("/get-area-spectral-stats")
def get_area_spectral_stats(req: AreaSpectralStatsRequest):
    """Get mean and std of all band values across multiple points."""
    try:
        cache_key = bbox_to_cache_key(req.image_id, req.bbox)
        with RASTER_CACHE_LOCK:
            cached_raster_path = RASTER_FILE_CACHE.get(cache_key)

        if not cached_raster_path or not os.path.exists(cached_raster_path):
            raise HTTPException(status_code=400, detail="No cached raster. Process the image first.")

        from rasterio.transform import rowcol
        from pyproj import Transformer

        band_names = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12']

        all_values = []  # list of [band0_val, band1_val, ...] per valid point

        with rasterio.open(cached_raster_path) as src:
            transformer = None
            if src.crs and str(src.crs).upper() != 'EPSG:4326':
                transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

            n_bands = min(src.count, len(band_names))

            for pt in req.points:
                lat, lng = pt.get('lat', 0), pt.get('lng', 0)
                if transformer:
                    x, y = transformer.transform(lng, lat)
                else:
                    x, y = lng, lat

                row, col = rowcol(src.transform, x, y)
                row, col = int(row), int(col)

                if row < 0 or row >= src.height or col < 0 or col >= src.width:
                    continue

                vals = []
                for i in range(n_bands):
                    v = float(src.read(i + 1, window=rasterio.windows.Window(col, row, 1, 1))[0, 0])
                    vals.append(v)
                all_values.append(vals)

        if not all_values:
            return {"mean_bands": [], "std_bands": [], "sample_count": 0, "error": "No valid points in bounds"}

        arr = np.array(all_values)  # shape: (N, n_bands)
        means = np.mean(arr, axis=0)
        stds = np.std(arr, axis=0)

        mean_bands = [{"name": band_names[i] if i < len(band_names) else f"Band_{i+1}", "value": round(float(means[i]), 2)} for i in range(len(means))]
        std_bands = [{"name": band_names[i] if i < len(band_names) else f"Band_{i+1}", "value": round(float(stds[i]), 2)} for i in range(len(stds))]

        return {"mean_bands": mean_bands, "std_bands": std_bands, "sample_count": len(all_values)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

