"""
API routes for tile and download operations.
"""

import os
import tempfile
import zipfile
import io
from typing import Optional

import numpy as np
import rasterio
import ee
import requests

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .schemas import (
    GetTileUrlRequest,
    S1TileRequest,
    S2TileRequest,
    DownloadImageRequest,
    EmitTileRequest,
)
from ..core.config import GEE_DL_MAX_BYTES, GEE_DL_SAFETY
from ..services.earth_engine import (
    bbox_to_geometry,
    resolve_item_to_image,
    get_visualization_params,
)
from ..services.downloader import (
    download_tile_get_url,
    mosaic_tiles_to_file,
    estimate_pixels_from_bounds,
)


router = APIRouter(prefix="/api", tags=["download"])


# Cache for map IDs (to avoid repeated getMapId calls)
_MAP_ID_CACHE = {}
_MAP_ID_CACHE_TTL = 3600  # 1 hour TTL


@router.post("/get-gee-tile")
async def get_gee_tile(req: GetTileUrlRequest):
    """Get tile URL for a Sentinel-2 image with caching."""
    import time
    try:
        t0 = time.time()
        cache_key = f"{req.item_id}_{hash(str(req.bbox))}"
        
        # Check cache first
        if cache_key in _MAP_ID_CACHE:
            cached = _MAP_ID_CACHE[cache_key]
            if time.time() - cached['timestamp'] < _MAP_ID_CACHE_TTL:
                print(f"TILE - Using cached mapId for {req.item_id}")
                return {"tile_template": cached['tile_template'], "bounds": cached['bounds']}
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        base_img = resolve_item_to_image(req.item_id).clip(aoi)
        vis_params = get_visualization_params()
        vis_img = base_img.unmask(0).visualize(**vis_params)
        
        t1 = time.time()
        m = vis_img.getMapId()
        t2 = time.time()
        
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        # Cache the result
        _MAP_ID_CACHE[cache_key] = {
            'tile_template': tile_template,
            'bounds': bounds,
            'timestamp': time.time()
        }

        print(f"TILE - getMapId took {t2-t1:.2f}s, total {t2-t0:.2f}s for {req.item_id}")
        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tile-proxy/{z}/{x}/{y}")
async def proxy_tile(z: int, x: int, y: int, url: str):
    """Proxy tiles from GEE to handle rate limiting and network issues."""
    from fastapi.responses import Response
    import time
    
    max_retries = 3
    retry_delay = 1.0
    
    # Construct the actual tile URL
    tile_url = url.replace("{z}", str(z)).replace("{x}", str(x)).replace("{y}", str(y))
    
    for attempt in range(max_retries):
        try:
            response = requests.get(tile_url, timeout=30)
            if response.status_code == 200:
                return Response(
                    content=response.content,
                    media_type="image/png",
                    headers={
                        "Cache-Control": "public, max-age=86400",  # Cache for 1 day
                        "Access-Control-Allow-Origin": "*"
                    }
                )
            elif response.status_code == 429:  # Rate limited
                print(f"TILE PROXY - Rate limited for {z}/{x}/{y}, waiting...")
                time.sleep(retry_delay * (attempt + 1))
                continue
            else:
                print(f"TILE PROXY - Error {response.status_code} for {z}/{x}/{y}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    continue
                raise HTTPException(status_code=response.status_code, detail="Tile fetch failed")
        except requests.Timeout:
            print(f"TILE PROXY - Timeout for {z}/{x}/{y} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise HTTPException(status_code=504, detail="Tile fetch timeout")
        except Exception as e:
            print(f"TILE PROXY - Error for {z}/{x}/{y}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise HTTPException(status_code=500, detail=str(e))
    
    raise HTTPException(status_code=500, detail="Failed to fetch tile after retries")


@router.post("/get-titiler-url")
async def get_titiler_url(req: GetTileUrlRequest):
    """Alias for get-gee-tile for backward compatibility."""
    return await get_gee_tile(req)


@router.post("/get-s1-tile")
async def get_s1_tile(req: S1TileRequest):
    """Get tile URL for Sentinel-1 GRD image with custom visualization."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        
        item_id = req.item_id
        if item_id.startswith("COPERNICUS/S1_GRD/"):
            base_img = ee.Image(item_id)
        else:
            base_img = ee.Image(f"COPERNICUS/S1_GRD/{item_id}")
        
        base_img = base_img.clip(aoi)
        
        bands = req.bands if req.bands else ['VV', 'VH', 'VV']
        vis_params = {
            'bands': bands,
            'min': req.min if req.min is not None else -25,
            'max': req.max if req.max is not None else 0,
        }
        
        vis_img = base_img.visualize(**vis_params)
        
        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-s2-tile-custom")
async def get_s2_tile_custom(req: S2TileRequest):
    """Get tile URL for Sentinel-2 image with custom visualization bands."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        
        item_id = req.item_id
        if item_id.startswith("COPERNICUS/S2"):
            # Convert old S2_SR paths to S2_SR_HARMONIZED
            if "COPERNICUS/S2_SR/" in item_id and "HARMONIZED" not in item_id:
                item_id = item_id.replace("COPERNICUS/S2_SR/", "COPERNICUS/S2_SR_HARMONIZED/")
            base_img = ee.Image(item_id)
        else:
            base_img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{item_id}")
        
        base_img = base_img.clip(aoi)
        
        bands = req.bands if req.bands else ['B4', 'B3', 'B2']
        vis_params = {
            'bands': bands,
            'min': req.min if req.min is not None else 0,
            'max': req.max if req.max is not None else 3000,
        }
        
        vis_img = base_img.visualize(**vis_params)
        
        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-emit-tile")
async def get_emit_tile(req: GetTileUrlRequest):
    """Get tile URL for EMIT hyperspectral image with default RGB visualization."""
    try:
        from ..services.earth_engine import bbox_to_geometry, resolve_emit_image
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        
        base_img = resolve_emit_image(req.item_id).clip(aoi)
        
        # Get band names
        band_names = base_img.bandNames().getInfo()
        
        # Default RGB: try to find bands around 650nm (red), 550nm (green), 450nm (blue)
        # EMIT bands are typically named with wavelength
        r_band = None
        g_band = None
        b_band = None
        
        # Try to find closest bands to target wavelengths
        target_r = 650  # Red
        target_g = 550  # Green
        target_b = 450  # Blue
        
        def find_closest_band(target_wl):
            closest = None
            min_diff = float('inf')
            for band_name in band_names:
                try:
                    # Try to extract wavelength from band name
                    if '_' in band_name:
                        wl_str = band_name.split('_')[-1]
                        wl = float(wl_str)
                    elif band_name.replace('.', '').isdigit():
                        wl = float(band_name)
                    else:
                        continue
                    
                    diff = abs(wl - target_wl)
                    if diff < min_diff:
                        min_diff = diff
                        closest = band_name
                except (ValueError, AttributeError):
                    continue
            return closest
        
        r_band = find_closest_band(target_r) or (band_names[len(band_names) // 2] if band_names else None)
        g_band = find_closest_band(target_g) or (band_names[len(band_names) // 3] if band_names else None)
        b_band = find_closest_band(target_b) or (band_names[0] if band_names else None)
        
        # Fallback: use first 3 bands if wavelength extraction fails
        if not r_band or not g_band or not b_band:
            if len(band_names) >= 3:
                r_band = band_names[len(band_names) // 2]
                g_band = band_names[len(band_names) // 3]
                b_band = band_names[0]
            else:
                # Use same band for all if not enough bands
                r_band = g_band = b_band = band_names[0] if band_names else None
        
        if not r_band or not g_band or not b_band:
            raise HTTPException(status_code=400, detail="Could not determine RGB bands for EMIT image")
        
        # Select RGB bands
        vis_img = base_img.select([r_band, g_band, b_band])
        
        # Visualize with appropriate scaling (EMIT reflectance is typically 0-1 or 0-10000)
        vis_params = {
            'bands': [r_band, g_band, b_band],
            'min': 0,
            'max': 10000,
        }
        
        vis_img = vis_img.visualize(**vis_params)
        
        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-emit-tile-custom")
async def get_emit_tile_custom(req: EmitTileRequest):
    """Get tile URL for EMIT hyperspectral image with custom RGB bands and visualization parameters."""
    try:
        from ..services.earth_engine import bbox_to_geometry, resolve_emit_image
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        base_img = resolve_emit_image(req.item_id).clip(aoi)
        
        # Use provided bands or default
        if not req.bands or len(req.bands) != 3:
            raise HTTPException(status_code=400, detail="Exactly 3 bands required for RGB visualization")
        
        r_band, g_band, b_band = req.bands
        
        # Select RGB bands
        vis_img = base_img.select([r_band, g_band, b_band])
        
        # Visualize with custom parameters
        vis_params = {
            'bands': [r_band, g_band, b_band],
            'min': req.min or 0,
            'max': req.max or 10000,
        }
        
        vis_img = vis_img.visualize(**vis_params)
        
        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        return {"tile_template": tile_template, "bounds": bounds}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download-s1-image")
async def download_s1_image(req: DownloadImageRequest):
    """Download Sentinel-1 GRD image as GeoTIFF or visualization PNG."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        aoi_rect = aoi.bounds()
        
        item_id = req.item_id
        if item_id.startswith("COPERNICUS/S1_GRD/"):
            full_id = item_id
        else:
            full_id = f"COPERNICUS/S1_GRD/{item_id}"
        
        print(f"S1 DOWNLOAD - Loading image: {full_id}")
        base_img = ee.Image(full_id).clip(aoi)
        
        if req.as_visualization:
            bands = req.bands if req.bands else ['VV', 'VH', 'VV']
            vis_min = req.min if req.min is not None else -25
            vis_max = req.max if req.max is not None else 0
            
            vis_img = base_img.visualize(bands=bands, min=vis_min, max=vis_max)
            
            filename = f"S1_GRD_{item_id.split('/')[-1]}_vis.png"
            temp_dir = tempfile.mkdtemp(prefix="s1_vis_download_")
            out_path = os.path.join(temp_dir, filename)
            
            url = vis_img.getThumbURL({
                'region': aoi_rect,
                'dimensions': 2048,
                'format': 'png'
            })
            
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            
            return FileResponse(
                out_path, media_type='image/png', filename=filename,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        # Raw download
        bands = ['VV', 'VH']
        filename = f"S1_GRD_{item_id.split('/')[-1]}.tif"
        temp_dir = tempfile.mkdtemp(prefix="s1_download_")
        out_path = os.path.join(temp_dir, filename)
        
        bounds_info = aoi_rect.getInfo()['coordinates'][0]
        min_lon, min_lat = bounds_info[0]
        max_lon, max_lat = bounds_info[2]
        
        px_w, px_h = estimate_pixels_from_bounds(min_lon, min_lat, max_lon, max_lat, 10)
        num_pixels = px_w * px_h
        
        max_tile = 2048
        if num_pixels > max_tile * max_tile:
            mosaic_tiles_to_file(base_img, bands, aoi_rect, 10, max_tile, out_path, as_float=True)
        else:
            blob = download_tile_get_url(base_img, bands, aoi_rect, 10, as_float=True)
            with open(out_path, 'wb') as f:
                f.write(blob)
        
        return FileResponse(
            out_path, media_type='image/tiff', filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"S1 DOWNLOAD ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download-s2-image")
async def download_s2_image(req: DownloadImageRequest):
    """Download Sentinel-2 image as GeoTIFF or visualization PNG."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        aoi_rect = aoi.bounds()
        
        item_id = req.item_id
        if item_id.startswith("COPERNICUS/S2"):
            # Convert old S2_SR paths to S2_SR_HARMONIZED
            if "COPERNICUS/S2_SR/" in item_id and "HARMONIZED" not in item_id:
                item_id = item_id.replace("COPERNICUS/S2_SR/", "COPERNICUS/S2_SR_HARMONIZED/")
            base_img = ee.Image(item_id)
        else:
            base_img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{item_id}")
        
        base_img = base_img.clip(aoi)
        print(f"S2 DOWNLOAD - Loading image: {item_id}")
        
        if req.as_visualization:
            bands = req.bands if req.bands else ['B4', 'B3', 'B2']
            vis_min = req.min if req.min is not None else 0
            vis_max = req.max if req.max is not None else 3000
            
            vis_img = base_img.visualize(bands=bands, min=vis_min, max=vis_max)
            
            filename = f"S2_SR_HARMONIZED_{item_id.split('/')[-1]}_vis.png"
            temp_dir = tempfile.mkdtemp(prefix="s2_vis_download_")
            out_path = os.path.join(temp_dir, filename)
            
            url = vis_img.getThumbURL({
                'region': aoi_rect,
                'dimensions': 2048,
                'format': 'png'
            })
            
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            
            with open(out_path, 'wb') as f:
                f.write(resp.content)
            
            return FileResponse(
                out_path, media_type='image/png', filename=filename,
                headers={"Content-Disposition": f"attachment; filename={filename}"}
            )
        
        # Raw download - all 13 bands (12 spectral + SCL)
        # B1(60m), B2-B4(10m), B5-B7(20m), B8(10m), B8A(20m), B9(60m), B11-B12(20m), SCL(20m)
        bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12', 'SCL']
        filename = f"S2_SR_HARMONIZED_{item_id.split('/')[-1]}_13bands.tif"
        temp_dir = tempfile.mkdtemp(prefix="s2_download_")
        out_path = os.path.join(temp_dir, filename)
        
        print(f"S2 DOWNLOAD - Downloading 13 bands (S2_SR_HARMONIZED): {bands}")
        
        # Resample all bands to 10m resolution for consistency
        # 60m bands (B1, B9) and 20m bands (B5-B7, B8A, B11, B12, SCL) will be resampled
        base_img_resampled = base_img.select(bands).reproject(
            crs=base_img.select('B2').projection(),
            scale=10
        )
        
        bounds_info = aoi_rect.getInfo()['coordinates'][0]
        min_lon, min_lat = bounds_info[0]
        max_lon, max_lat = bounds_info[2]
        
        px_w, px_h = estimate_pixels_from_bounds(min_lon, min_lat, max_lon, max_lat, 10)
        num_pixels = px_w * px_h
        
        max_tile = 2048
        if num_pixels > max_tile * max_tile:
            mosaic_tiles_to_file(base_img_resampled, bands, aoi_rect, 10, max_tile, out_path)
        else:
            blob = download_tile_get_url(base_img_resampled, bands, aoi_rect, 10)
            with open(out_path, 'wb') as f:
                f.write(blob)
        
        return FileResponse(
            out_path, media_type='image/tiff', filename=filename,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"S2 DOWNLOAD ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

