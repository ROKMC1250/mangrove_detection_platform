"""
Visualization module for creating image overlays and PNG outputs.
"""

import os
import io
import numpy as np
from typing import Dict, List, Optional, Tuple

import rasterio
from rasterio.warp import reproject, transform_bounds
from rasterio.transform import from_bounds
from rasterio.enums import Resampling
from PIL import Image
import matplotlib.pyplot as plt

from ..core.config import PROJECT_ROOT


def stretch_uint8(arr: np.ndarray, max_val: float = 10000.0, gamma: float = 3.0) -> np.ndarray:
    """Stretch array values to uint8 (0-255) for visualization.
    
    Args:
        arr: Input array (typically Sentinel-2 reflectance values)
        max_val: Maximum expected value for scaling
        gamma: Gamma correction factor
        
    Returns:
        Uint8 array (0-255)
    """
    x = np.asarray(arr, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    maxv = float(np.max(x)) if x.size > 0 else 1.0
    
    if maxv > 1.01:
        x = x / max_val * gamma
    
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def create_index_visualization(index_data: np.ndarray, cmap: str = 'viridis', 
                                vmin: Optional[float] = None, 
                                vmax: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
    """Creates a colormapped RGB visualization for a spectral index.
    
    Args:
        index_data: 2D array of index values
        cmap: Matplotlib colormap name
        vmin: Minimum value for colormap (uses percentile if None)
        vmax: Maximum value for colormap (uses percentile if None)
        
    Returns:
        Tuple of (RGB array uint8, actual_min, actual_max)
    """
    nan_mask = np.isnan(index_data)

    min_val = vmin if vmin is not None else np.percentile(index_data[~nan_mask], 0.5)
    max_val = vmax if vmax is not None else np.percentile(index_data[~nan_mask], 99.5)

    denominator = max_val - min_val
    if denominator < 1e-6: 
        denominator = 1e-6

    normalized_data = np.clip(index_data, min_val, max_val)
    normalized_data = (normalized_data - min_val) / denominator

    colored_image = (plt.get_cmap(cmap)(normalized_data)[:, :, :3] * 255).astype(np.uint8)
    colored_image[nan_mask] = [0, 0, 0]

    return colored_image, min_val, max_val


def save_rgba_overlay_png(rgb: np.ndarray, valid_mask: Optional[np.ndarray], 
                          out_path: str) -> None:
    """Save overlay PNG with full opacity (for regular visualizations)."""
    h, w, _ = rgb.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    
    if valid_mask is not None:
        invalid_mask = ~valid_mask
        rgba[invalid_mask, :3] = [0, 0, 0]
    
    rgba[..., 3] = 255
    
    img = Image.fromarray(rgba, mode='RGBA')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format='PNG')


def save_rgba_overlay_png_with_transparency(rgb: np.ndarray, valid_mask: Optional[np.ndarray], 
                                            out_path: str) -> None:
    """Save overlay PNG with transparency for background (for masks)."""
    h, w, _ = rgb.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    
    if valid_mask is not None:
        rgba[..., 3] = np.where(valid_mask, 255, 0)
    else:
        is_black = (rgb[:, :, 0] < 10) & (rgb[:, :, 1] < 10) & (rgb[:, :, 2] < 10)
        rgba[..., 3] = np.where(~is_black, 255, 0)
    
    img = Image.fromarray(rgba, mode='RGBA')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    img.save(out_path, format='PNG')


def save_float_geotiff(arr: np.ndarray, ref_profile: dict, out_path: str, 
                       nodata_value: float = -9999.0) -> None:
    """Save float32 array as GeoTIFF."""
    prof = ref_profile.copy()
    h, w = arr.shape
    prof.update({
        'driver': 'GTiff',
        'height': h,
        'width': w,
        'count': 1,
        'dtype': 'float32',
        'nodata': nodata_value,
        'tiled': False,
        'compress': 'deflate',
        'interleave': 'band'
    })
    data = arr.astype(np.float32)
    data = np.nan_to_num(data, nan=nodata_value)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, 'w', **prof) as dst:
        dst.write(data, 1)


def resize_rgb_to_dims(rgb: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize RGB array to target dimensions."""
    src_h, src_w = rgb.shape[0], rgb.shape[1]
    if src_w == target_width and src_h == target_height:
        return rgb
    pil = Image.fromarray(rgb, mode='RGB')
    resized = pil.resize((int(target_width), int(target_height)), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def resize_mask_to_dims(mask: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize mask to target dimensions using nearest neighbor."""
    src_h, src_w = mask.shape[0], mask.shape[1]
    if src_w == target_width and src_h == target_height:
        return mask
    pil = Image.fromarray(mask.astype(np.uint8), mode='L')
    resized = pil.resize((int(target_width), int(target_height)), Image.NEAREST)
    return (np.asarray(resized) > 0).astype(np.uint8)


def to_png_bytes(rgb: np.ndarray, mask: Optional[np.ndarray] = None, 
                 thumbnail_size: int = 512) -> bytes:
    """Convert RGB array to PNG bytes, optionally with mask for transparency.
    
    Args:
        rgb: RGB array (H, W, 3)
        mask: Optional mask array (H, W) where True = visible
        thumbnail_size: Maximum dimension for thumbnail
        
    Returns:
        PNG bytes
    """
    if rgb.dtype != np.uint8:
        rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    
    if mask is not None:
        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = (mask * 255).astype(np.uint8)
        img = Image.fromarray(rgba, mode='RGBA')
    else:
        img = Image.fromarray(rgb, mode='RGB')
    
    img.thumbnail((thumbnail_size, thumbnail_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def warp_rgb_and_mask_to_aoi(rgb: np.ndarray, valid_mask: Optional[np.ndarray],
                              src_transform, src_crs, 
                              aoi_bbox: Tuple[float, float, float, float],
                              scale_m: int = 10, 
                              geometry: Optional[Dict] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int], any]:
    """Warp RGB and mask to AOI-aligned grid.
    
    Args:
        rgb: RGB array (H, W, 3)
        valid_mask: Valid data mask (H, W)
        src_transform: Source rasterio transform
        src_crs: Source CRS
        aoi_bbox: (min_lon, min_lat, max_lon, max_lat)
        scale_m: Target scale in meters
        geometry: Optional GeoJSON geometry for masking
        
    Returns:
        Tuple of (warped_rgb, warped_mask, (width, height), dst_transform)
    """
    from .downloader import estimate_pixels_from_bounds
    
    min_lon, min_lat, max_lon, max_lat = aoi_bbox
    dst_w, dst_h = estimate_pixels_from_bounds(min_lon, min_lat, max_lon, max_lat, scale_m)
    dst_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, dst_w, dst_h)
    dst_crs = 'EPSG:4326'
    
    # Prepare outputs
    dst_r = np.zeros((dst_h, dst_w), dtype=np.uint8)
    dst_g = np.zeros((dst_h, dst_w), dtype=np.uint8)
    dst_b = np.zeros((dst_h, dst_w), dtype=np.uint8)
    
    # Reproject channels
    reproject(
        source=rgb[..., 0], destination=dst_r,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    reproject(
        source=rgb[..., 1], destination=dst_g,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    reproject(
        source=rgb[..., 2], destination=dst_b,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )
    
    dst_rgb = np.stack([dst_r, dst_g, dst_b], axis=-1)
    
    # Reproject mask
    if valid_mask is not None:
        src_mask = valid_mask.astype(np.uint8)
        dst_mask_proj = np.zeros((dst_h, dst_w), dtype=np.uint8)
        reproject(
            source=src_mask, destination=dst_mask_proj,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=dst_transform, dst_crs=dst_crs,
            resampling=Resampling.nearest
        )
        dst_mask = (dst_mask_proj > 0)
    else:
        dst_mask = np.ones((dst_h, dst_w), dtype=bool)

    # Apply geometry mask if provided
    if geometry:
        try:
            from rasterio.features import geometry_mask
            from shapely.geometry import shape, MultiPolygon
            
            geom_shape = shape(geometry)
            
            if isinstance(geom_shape, MultiPolygon):
                geometries = list(geom_shape.geoms)
            else:
                geometries = [geom_shape]
            
            geo_mask = geometry_mask(
                geometries,
                out_shape=(dst_h, dst_w),
                transform=dst_transform,
                invert=True,
                all_touched=True
            )
            
            dst_mask = dst_mask & geo_mask
            print(f"WARP - Geometry mask applied, valid pixels: {np.sum(geo_mask)}")
            
        except Exception as e:
            print(f"WARP - Error creating geometry mask: {e}")
            import traceback
            traceback.print_exc()

    return dst_rgb, dst_mask, (dst_w, dst_h), dst_transform


def generate_overlay_path(prefix: str, suffix: str = ".png") -> str:
    """Generate a unique path for overlay files."""
    import time
    stamp = int(time.time())
    filename = f"{prefix}_aoi_{stamp}{suffix}"
    return os.path.join(PROJECT_ROOT, 'outputs', filename)

