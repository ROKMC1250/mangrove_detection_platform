"""
Cache management module for raster files, index data, and custom visualizations.
"""

import threading
import numpy as np
from typing import Dict, List, Optional, Tuple


# ===== Raster File Cache =====
# Cache key format: {image_id}_{bbox_hash} to support different AOI bounds
RASTER_FILE_CACHE: Dict[str, str] = {}  # {cache_key: file_path}
RASTER_CACHE_LOCK = threading.Lock()


# ===== Custom Visualization Cache =====
CUSTOM_VIZ_CACHE: Dict[str, Dict] = {}  # {custom_id: {type, bands, etc.}}
CUSTOM_VIZ_LOCK = threading.Lock()


# ===== Index Data Cache =====
# Cache key format: {image_id}_{model_id}
INDEX_DATA_CACHE: Dict[str, Dict] = {}  # {cache_key: {data, transform, crs, bbox}}
INDEX_CACHE_LOCK = threading.Lock()


def bbox_to_cache_key(image_id: str, bbox: List[float]) -> str:
    """Generate a cache key that includes both image_id and bbox hash."""
    # Round bbox values to avoid floating point comparison issues (6 decimal places ~ 10cm precision)
    bbox_rounded = [round(v, 6) for v in bbox]
    bbox_str = "_".join([str(v) for v in bbox_rounded])
    return f"{image_id}_{hash(bbox_str) & 0xFFFFFFFF}"  # Use positive hash


def get_cached_raster_path(image_id: str, bbox: List[float]) -> Optional[str]:
    """Get cached raster file path if exists."""
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        path = RASTER_FILE_CACHE.get(cache_key)
        if path:
            import os
            if os.path.exists(path):
                return path
    return None


def cache_raster_file(image_id: str, bbox: List[float], file_path: str) -> str:
    """Cache a raster file path and return the cache key."""
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        RASTER_FILE_CACHE[cache_key] = file_path
        print(f"CACHE - Stored raster: {cache_key} -> {file_path}")
    return cache_key


def _lat_lng_to_pixel_index(lat: float, lng: float, transform, height: int, width: int) -> Tuple[Optional[int], Optional[int]]:
    """Convert lat/lng coordinates to pixel indices using rasterio transform."""
    try:
        from rasterio.transform import rowcol
        row, col = rowcol(transform, lng, lat)
        
        print(f"COORD TRANSFORM - Input: lat={lat}, lng={lng}")
        print(f"COORD TRANSFORM - Raw row/col: {row}, {col}")
        print(f"COORD TRANSFORM - Image bounds: height={height}, width={width}")
        
        # Convert to integer pixel indices
        row_idx = int(round(row))
        col_idx = int(round(col))
        
        print(f"COORD TRANSFORM - Pixel indices: row_idx={row_idx}, col_idx={col_idx}")
        
        # Always clamp to image bounds for robustness
        clamped_row = max(0, min(row_idx, height - 1))
        clamped_col = max(0, min(col_idx, width - 1))
        
        # Calculate distance from bounds to decide if clamping is reasonable
        max_distance = max(
            abs(row_idx - clamped_row),
            abs(col_idx - clamped_col)
        )
        
        # Allow reasonable distance (up to 20% of smaller dimension or 200 pixels)
        tolerance = max(200, min(height, width) * 0.2)
        
        if max_distance <= tolerance:
            print(f"COORD TRANSFORM - SUCCESS: using clamped indices [{clamped_row}, {clamped_col}] (distance: {max_distance})")
            return clamped_row, clamped_col
        else:
            print(f"COORD TRANSFORM - FAILED: too far from image (distance: {max_distance}, tolerance: {tolerance})")
            return None, None
            
    except Exception as e:
        print(f"COORD TRANSFORM - Error: {e}")
        return None, None


def get_pixel_value_from_cache(cache_key: str, lat: float, lng: float) -> Optional[float]:
    """Get pixel value from cached index data."""
    try:
        with INDEX_CACHE_LOCK:
            cached_data = INDEX_DATA_CACHE.get(cache_key)
        
        if not cached_data:
            return None
        
        data = cached_data['data']
        transform = cached_data['transform']
        src_crs = cached_data.get('crs')
        height, width = data.shape

        # Transform coordinates from EPSG:4326 to image CRS if needed
        target_lng, target_lat = lng, lat
        if src_crs and str(src_crs).upper() != 'EPSG:4326':
            try:
                from rasterio.warp import transform as rasterio_transform
                xs, ys = rasterio_transform('EPSG:4326', src_crs, [lng], [lat])
                target_lng, target_lat = xs[0], ys[0]
                print(f"CRS TRANSFORM - Original: ({lng}, {lat}) -> Target CRS: ({target_lng}, {target_lat})")
            except Exception as e:
                print(f"CRS TRANSFORM - Failed: {e}, using original coordinates")
                target_lng, target_lat = lng, lat
        else:
            print(f"CRS TRANSFORM - No transformation needed, CRS: {src_crs}")

        # Convert transformed coordinates to pixel indices
        row_idx, col_idx = _lat_lng_to_pixel_index(target_lat, target_lng, transform, height, width)
        print(f"INDEX CACHE - Pixel indices: row_idx={row_idx}, col_idx={col_idx}")
        print(f"INDEX CACHE - Data shape: {data.shape}")
        print(f"INDEX CACHE - Transform: {transform}")
        
        if row_idx is not None and col_idx is not None:
            value = data[row_idx, col_idx]
            print(f"INDEX CACHE - Raw value at [{row_idx}, {col_idx}]: {value}")
            # Check if value is valid (not NaN)
            if np.isfinite(value):
                print(f"INDEX CACHE - Valid value: {value}")
                return float(value)
            else:
                print(f"INDEX CACHE - Invalid value (NaN/Inf): {value}")
        else:
            print(f"INDEX CACHE - Invalid pixel indices")
        
        return None
    except Exception as e:
        print(f"Error getting pixel value from cache: {e}")
        return None


def cache_index_data(image_id: str, model_id: str, data: np.ndarray, 
                     transform, crs, bbox: List[float]) -> str:
    """Cache index data for pixel value computation."""
    cache_key = f"{image_id}_{model_id}"
    with INDEX_CACHE_LOCK:
        INDEX_DATA_CACHE[cache_key] = {
            'data': data,
            'transform': transform,
            'crs': crs,
            'bbox': bbox
        }
        print(f"INDEX CACHE - Stored: {cache_key}")
    return cache_key


def find_matching_index_cache_key(image_id: str, model_id: str) -> Optional[str]:
    """Find a matching cache key for the given image and model."""
    exact_key = f"{image_id}_{model_id}"
    
    with INDEX_CACHE_LOCK:
        if exact_key in INDEX_DATA_CACHE:
            return exact_key
        
        # Try to find a key that ends with the model_id
        matching_keys = [k for k in INDEX_DATA_CACHE.keys() if k.endswith(f"_{model_id}")]
        if matching_keys:
            print(f"INDEX CACHE - Using partial match key: {matching_keys[0]}")
            return matching_keys[0]
    
    return None


def get_custom_viz_cache(custom_id: str) -> Optional[Dict]:
    """Get custom visualization from cache."""
    with CUSTOM_VIZ_LOCK:
        return CUSTOM_VIZ_CACHE.get(custom_id)


def set_custom_viz_cache(custom_id: str, data: Dict) -> None:
    """Set custom visualization in cache."""
    with CUSTOM_VIZ_LOCK:
        CUSTOM_VIZ_CACHE[custom_id] = data
        print(f"CUSTOM VIZ CACHE - Stored: {custom_id}")

