"""
Spectral analysis module for calculating vegetation and water indices.
"""

import numpy as np
from typing import Dict, Tuple, Optional

from ..core.config import SAFE_DIV_EPS


def safe_divide(numer: np.ndarray, denom: np.ndarray, eps: float = SAFE_DIV_EPS) -> np.ndarray:
    """Safe division that handles zero denominators."""
    d_copy = denom.copy()
    d_copy[d_copy == 0] = eps
    return numer / d_copy


def calculate_ndvi(nir: np.ndarray, red: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Vegetation Index.
    
    NDVI = (NIR - RED) / (NIR + RED)
    """
    return safe_divide(nir - red, nir + red)


def calculate_ndmi(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Moisture Index.
    
    NDMI = (NIR - SWIR) / (NIR + SWIR)
    """
    return safe_divide(nir - swir, nir + swir)


def calculate_mvi(nir: np.ndarray, green: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Calculate Mangrove Vegetation Index.
    
    MVI = (NIR - GREEN) / (SWIR1 - GREEN)
    """
    return safe_divide(nir - green, swir1 - green)


def calculate_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Water Index (McFeeters).
    
    NDWI = (GREEN - NIR) / (GREEN + NIR)
    """
    return safe_divide(green - nir, green + nir)


def calculate_spectral_index(data: Dict[str, np.ndarray], index_type: str, 
                              mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Calculate spectral index from raster band data.
    
    Args:
        data: Dictionary mapping band names to numpy arrays
              Expected keys: 'B2' (Blue), 'B3' (Green), 'B4' (Red), 
                           'B8' (NIR), 'B11' (SWIR1), 'B12' (SWIR2)
        index_type: Type of index ('ndvi', 'ndmi', 'mvi', 'ndwi')
        mask: Optional mask array (True = valid pixels)
        
    Returns:
        Calculated index array with NaN for invalid pixels
    """
    # Apply mask to bands if provided
    def apply_mask(arr):
        if mask is not None:
            result = arr.copy().astype(np.float32)
            result[~mask] = np.nan
            return result
        return arr.astype(np.float32)
    
    index_type = index_type.lower()
    
    if index_type == 'ndvi':
        nir = apply_mask(data['B8'])
        red = apply_mask(data['B4'])
        return calculate_ndvi(nir, red)
    
    elif index_type == 'ndmi':
        nir = apply_mask(data['B8'])
        swir2 = apply_mask(data['B12'])
        return calculate_ndmi(nir, swir2)
    
    elif index_type == 'mvi':
        nir = apply_mask(data['B8'])
        green = apply_mask(data['B3'])
        swir1 = apply_mask(data['B11'])
        return calculate_mvi(nir, green, swir1)
    
    elif index_type == 'ndwi':
        green = apply_mask(data['B3'])
        nir = apply_mask(data['B8'])
        return calculate_ndwi(green, nir)
    
    else:
        raise ValueError(f"Unknown index type: {index_type}")


def calculate_custom_normalized_index(band_a: np.ndarray, band_b: np.ndarray) -> np.ndarray:
    """Calculate custom normalized difference index.
    
    Index = (A - B) / (A + B)
    """
    return safe_divide(band_a - band_b, band_a + band_b)


def calculate_mangrove_area_km2(mask: np.ndarray, pixel_size_m: float = 10.0) -> float:
    """Calculate mangrove area in km² from binary mask.
    
    Args:
        mask: Binary mask or RGB/RGBA mask array
        pixel_size_m: Pixel size in meters (default 10m for Sentinel-2)
        
    Returns:
        Area in square kilometers
    """
    if mask is None or len(mask) == 0:
        return 0.0
    
    # Handle different mask formats
    if mask.ndim == 2:
        mangrove_pixels = np.sum(mask > 0)
    elif mask.ndim == 3:
        if mask.shape[-1] == 3:
            # RGB mask - count pixels where red channel is 255
            mangrove_pixels = np.sum(mask[:, :, 0] == 255)
        elif mask.shape[-1] == 4:
            # RGBA mask
            red_pixels = (mask[:, :, 0] == 255)
            alpha_pixels = (mask[:, :, 3] > 0)
            mangrove_pixels = np.sum(red_pixels & alpha_pixels)
        else:
            mangrove_pixels = np.sum(np.any(mask > 0, axis=-1))
    else:
        mangrove_pixels = np.sum(mask > 0)
    
    print(f"AREA CALCULATION - Mask shape: {mask.shape}, Mangrove pixels: {mangrove_pixels}")
    
    # Calculate area
    area_m2 = mangrove_pixels * (pixel_size_m ** 2)
    area_km2 = area_m2 / 1_000_000
    
    return float(area_km2)


def get_band_data_from_raster(raster_data: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract band data from raster array.
    
    Assumes standard band order: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
    
    Args:
        raster_data: Raster data array of shape (bands, height, width)
        
    Returns:
        Dictionary mapping band names to arrays
    """
    return {
        'B2': raster_data[0],   # Blue
        'B3': raster_data[1],   # Green
        'B4': raster_data[2],   # Red
        'B5': raster_data[3],   # Red Edge 1
        'B6': raster_data[4],   # Red Edge 2
        'B7': raster_data[5],   # Red Edge 3
        'B8': raster_data[6],   # NIR
        'B8A': raster_data[7],  # NIR narrow
        'B11': raster_data[8],  # SWIR1
        'B12': raster_data[9],  # SWIR2
    }


def get_index_visualization_params(index_type: str) -> Tuple[str, float, float]:
    """Get visualization parameters for an index type.
    
    Returns:
        Tuple of (colormap_name, vmin, vmax)
    """
    params = {
        'ndvi': ('RdYlGn', -1, 1),
        'ndmi': ('RdYlGn', -1, 1),
        'mvi': ('viridis', None, None),  # Dynamic range
        'ndwi': ('RdYlBu', -1, 1),
    }
    return params.get(index_type.lower(), ('viridis', None, None))

