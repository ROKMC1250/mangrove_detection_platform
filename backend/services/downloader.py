"""
Image download module - Unified download logic for Earth Engine images.

This module consolidates the download logic that was previously duplicated across
multiple functions (process_image, change_monitoring, process_spectral_image, etc.)
"""

import os
import io
import math
import time
import zipfile
import threading
import queue as _queue
from typing import List, Tuple, Optional

import numpy as np
import ee
import rasterio
from rasterio.io import MemoryFile
from rasterio.merge import merge as rio_merge

from ..core.config import (
    HTTP_SESSION,
    GEE_DL_MAX_BYTES,
    GEE_DL_SAFETY,
    PARALLEL_TILE_WORKERS,
    PROJECT_ROOT,
    OUTPUTS_DIR,
)
from ..core.progress import PROGRESS_TRACKER


def estimate_pixels_from_bounds(min_lon: float, min_lat: float, max_lon: float, max_lat: float, scale: int) -> Tuple[int, int]:
    """Estimate pixel dimensions from geographic bounds."""
    center_lat = (min_lat + max_lat) / 2.0
    lon_m_per_deg = 111320.0 * max(0.0, math.cos(math.radians(center_lat)))
    lat_m_per_deg = 110574.0
    width_m = max(0.0, (max_lon - min_lon)) * lon_m_per_deg
    height_m = max(0.0, (max_lat - min_lat)) * lat_m_per_deg
    px_w = max(1, int(math.ceil(width_m / scale)))
    px_h = max(1, int(math.ceil(height_m / scale)))
    return px_w, px_h


def estimate_file_size(aoi: ee.Geometry, scale: int, num_bands: int, bytes_per_sample: int = 2) -> Tuple[int, int, float]:
    """Estimate file size and pixel dimensions for a given AOI.
    
    Returns:
        Tuple of (px_width, px_height, estimated_mb)
    """
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]
    
    px_w, px_h = estimate_pixels_from_bounds(min_lon, min_lat, max_lon, max_lat, scale)
    est_bytes = px_w * px_h * num_bands * bytes_per_sample
    est_mb = est_bytes / 1e6
    
    return px_w, px_h, est_mb


def should_use_mosaic(est_bytes: float) -> bool:
    """Determine if mosaic download should be used based on estimated size."""
    return est_bytes > GEE_DL_MAX_BYTES * GEE_DL_SAFETY


def download_tile_get_url(img: ee.Image, bands: list, region_geom: ee.Geometry, 
                           scale: int, as_float: bool = False) -> bytes:
    """Download a single tile from GEE using getDownloadURL.
    
    Args:
        img: Earth Engine image
        bands: List of band names
        region_geom: Region geometry
        scale: Scale in meters
        as_float: If True, keep as Float32 (for S1 dB data). If False, convert to Uint16 (for S2).
        
    Returns:
        Raw bytes of the downloaded TIFF
    """
    t_start = time.time()
    print(f"DOWNLOAD - Starting getDownloadURL (as_float={as_float})...")
    
    nodata_val = -9999 if as_float else 0
    selected = img.select(bands).unmask(nodata_val)
    
    if as_float:
        download_img = selected.toFloat()
    else:
        download_img = selected.round().toUint16()
    
    url = download_img.getDownloadURL({
        "region": region_geom,
        "scale": scale,
        "format": "GEO_TIFF",
        "filePerBand": False,
    })
    
    t_url = time.time()
    print(f"DOWNLOAD - getDownloadURL took {t_url - t_start:.2f}s")
    
    r = HTTP_SESSION.get(url, timeout=600)
    r.raise_for_status()
    content = r.content
    
    t_end = time.time()
    print(f"DOWNLOAD - HTTP request took {t_end - t_url:.2f}s, total: {t_end - t_start:.2f}s")
    
    # Detect TIFF magic or ZIP
    if len(content) >= 4 and (content[:4] in (b'II*\x00', b'MM\x00*')):
        return content
    
    ct = (r.headers.get('Content-Type') or '').lower()
    if content[:2] == b'PK' or 'zip' in ct:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            tif_infos = [(n, zf.getinfo(n).file_size) for n in zf.namelist() 
                        if n.lower().endswith(('.tif', '.tiff'))]
            if not tif_infos:
                raise RuntimeError("ZIP from getDownloadURL has no .tif inside")
            tif_infos.sort(key=lambda x: x[1], reverse=True)
            return zf.read(tif_infos[0][0])
    
    raise RuntimeError(f"Unexpected payload from getDownloadURL (ct={ct}, size={len(content)})")


def mosaic_tiles_to_file(img: ee.Image, bands: list, aoi: ee.Geometry, scale: int, 
                         max_tile_px: int, out_path: str, as_float: bool = False) -> None:
    """Download and mosaic tiles to a file.
    
    Args:
        img: Earth Engine image
        bands: List of band names
        aoi: Area of interest geometry
        scale: Scale in meters
        max_tile_px: Maximum tile size in pixels
        out_path: Output file path
        as_float: If True, download as Float32 (for S1 dB data). If False, download as Uint16 (for S2).
    """
    bounds = aoi.bounds().getInfo()['coordinates'][0]
    min_lon, min_lat = bounds[0]
    max_lon, max_lat = bounds[2]

    datasets = []
    memfiles = []

    # Pre-split into a grid
    px_w_total, px_h_total = estimate_pixels_from_bounds(min_lon, min_lat, max_lon, max_lat, scale)
    bytes_limit = int(GEE_DL_MAX_BYTES * GEE_DL_SAFETY)
    num_bands = max(1, len(bands))
    bytes_per_sample = 4 if as_float else 2
    max_samples = max(1, int(bytes_limit / (num_bands * bytes_per_sample)))
    target_side_px = int(math.floor(math.sqrt(max_samples)))
    target_side_px = max(512, min(target_side_px, max_tile_px))
    
    ncols = max(1, int(math.ceil(px_w_total / target_side_px)))
    nrows = max(1, int(math.ceil(px_h_total / target_side_px)))
    lon_step = (max_lon - min_lon) / float(ncols)
    lat_step = (max_lat - min_lat) / float(nrows)
    
    task_queue = _queue.Queue()
    for r in range(nrows):
        for c in range(ncols):
            t_min_lon = min_lon + c * lon_step
            t_max_lon = min_lon + (c + 1) * lon_step
            t_min_lat = min_lat + r * lat_step
            t_max_lat = min_lat + (r + 1) * lat_step
            task_queue.put((t_min_lon, t_min_lat, t_max_lon, t_max_lat))
    
    list_lock = threading.Lock()
    min_tile_px_local = 512

    def _worker():
        while True:
            try:
                b = task_queue.get_nowait()
            except _queue.Empty:
                break
            
            cur_min_lon, cur_min_lat, cur_max_lon, cur_max_lat = b
            px_w, px_h = estimate_pixels_from_bounds(cur_min_lon, cur_min_lat, cur_max_lon, cur_max_lat, scale)
            
            try:
                tile_geom = ee.Geometry.Rectangle([cur_min_lon, cur_min_lat, cur_max_lon, cur_max_lat], 
                                                  proj=None, geodesic=False)
                blob = download_tile_get_url(img, bands, tile_geom, scale, as_float=as_float)
                mem = MemoryFile(blob)
                ds = mem.open()
                
                if ds.width <= 0 or ds.height <= 0:
                    raise RuntimeError("Invalid TIFF dimensions (zero)")
                if ds.count != len(bands):
                    raise RuntimeError(f"Unexpected band count {ds.count} != {len(bands)}")
                
                with list_lock:
                    datasets.append(ds)
                    memfiles.append(mem)
                
                print(f"TILE OK [{cur_min_lon:.6f},{cur_min_lat:.6f},{cur_max_lon:.6f},{cur_max_lat:.6f}] ~ {px_w}x{px_h} px")
                
            except Exception as e:
                msg = str(e)
                if px_w <= min_tile_px_local and px_h <= min_tile_px_local:
                    print(f"TILE FAIL (min size reached) {b}: {msg}")
                else:
                    mid_lon = (cur_min_lon + cur_max_lon) / 2.0
                    mid_lat = (cur_min_lat + cur_max_lat) / 2.0
                    children = [
                        (cur_min_lon, cur_min_lat, mid_lon, mid_lat),
                        (mid_lon, cur_min_lat, cur_max_lon, mid_lat),
                        (cur_min_lon, mid_lat, mid_lon, cur_max_lat),
                        (mid_lon, mid_lat, cur_max_lon, cur_max_lat),
                    ]
                    print(f"SUBDIVIDE -> 4 tiles from ~{px_w}x{px_h} px region due to: {msg}")
                    for child in children:
                        task_queue.put(child)
            finally:
                task_queue.task_done()

    threads = []
    for _ in range(PARALLEL_TILE_WORKERS):
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        threads.append(t)

    task_queue.join()
    for t in threads:
        try:
            t.join(timeout=0)
        except Exception:
            pass

    if not datasets:
        raise RuntimeError("No tiles downloaded via getDownloadURL (all subdivisions failed)")

    # Merge tiles
    out_dtype = datasets[0].dtypes[0] if hasattr(datasets[0], 'dtypes') else datasets[0].profile.get('dtype', 'uint16')
    is_float = 'float' in str(out_dtype).lower()
    nodata_val = -9999 if is_float else 0
    
    mosaic, out_transform = rio_merge(datasets, nodata=nodata_val, dtype=out_dtype, method='first')
    
    profile = datasets[0].profile
    profile.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': out_transform,
        'count': mosaic.shape[0],
        'dtype': out_dtype,
        'nodata': nodata_val,
        'compress': 'lzw'
    })
    
    print(f"MOSAIC - Merging {len(datasets)} tiles, dtype={out_dtype}, nodata={nodata_val}")
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(mosaic)

    # Cleanup
    for ds in datasets:
        try:
            ds.close()
        except Exception:
            pass
    for mem in memfiles:
        try:
            mem.close()
        except Exception:
            pass


def download_ee_image(img: ee.Image, bands: List[str], aoi: ee.Geometry, 
                      scale: int, output_path: str, job_id: Optional[str] = None,
                      as_float: bool = False, max_tile: int = 4096) -> str:
    """
    Unified download function for Earth Engine images.
    
    Automatically determines whether to use single tile or mosaic download
    based on estimated file size.
    
    Args:
        img: Earth Engine image to download
        bands: List of band names to download
        aoi: Area of interest geometry
        scale: Scale in meters (e.g., 10 for Sentinel-2)
        output_path: Path to save the output file
        job_id: Optional job ID for progress tracking
        as_float: If True, download as Float32 (for S1 dB data)
        max_tile: Maximum tile size in pixels for mosaic
        
    Returns:
        Path to the downloaded file
    """
    t_start = time.time()
    
    # Estimate file size
    aoi_rect = aoi.bounds()
    bytes_per_sample = 4 if as_float else 2
    px_w, px_h, est_mb = estimate_file_size(aoi_rect, scale, len(bands), bytes_per_sample)
    est_bytes = px_w * px_h * len(bands) * bytes_per_sample
    
    print(f"DOWNLOAD - Estimated: {px_w}x{px_h} pixels, {est_mb:.2f}MB")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Choose download method
    use_mosaic = should_use_mosaic(est_bytes)
    
    if use_mosaic:
        print(f"DOWNLOAD - Using mosaic download: {est_mb:.2f}MB")
        if job_id:
            PROGRESS_TRACKER.update_phase(job_id, "Download", 0, 
                                         f'Downloading large image as mosaic - {est_mb:.1f}MB')
        mosaic_tiles_to_file(img, bands, aoi_rect, scale, max_tile, output_path, as_float=as_float)
    else:
        print(f"DOWNLOAD - Using single tile download: {est_mb:.2f}MB")
        if job_id:
            PROGRESS_TRACKER.update_phase(job_id, "Download", 0, 
                                         f'Downloading single tile - {est_mb:.1f}MB', 0.1)
        
        blob = download_tile_get_url(img, bands, aoi_rect, scale, as_float=as_float)
        
        if job_id:
            PROGRESS_TRACKER.update_phase(job_id, "Download", 0, 'Writing to file', 0.8)
        
        with open(output_path, 'wb') as f:
            f.write(blob)
    
    download_time = time.time() - t_start
    print(f"DOWNLOAD - Completed in {download_time:.2f}s")
    
    if job_id:
        PROGRESS_TRACKER.complete_phase(job_id, "Download", 
                                       f'Downloaded in {download_time:.1f}s')
    
    return output_path


def generate_output_path(prefix: str, item_id: str, suffix: str = ".tif") -> str:
    """Generate a unique output path for downloaded files."""
    stamp = int(time.time())
    # Sanitize item_id for filename
    safe_id = item_id.replace("/", "_").replace("\\", "_")
    filename = f"{prefix}_{safe_id}_{stamp}{suffix}"
    return os.path.join(OUTPUTS_DIR, filename)

