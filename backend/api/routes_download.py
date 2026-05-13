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
    StretchStatsRequest,
    StretchStatsResponse,
)
from ..core.config import GEE_DL_MAX_BYTES, GEE_DL_SAFETY
from ..services.earth_engine import (
    bbox_to_geometry,
    resolve_item_to_image,
    get_visualization_params,
    get_s1_visualization_params,
    compute_band_stretch_stats,
)
from ..services.downloader import (
    download_tile_get_url,
    mosaic_tiles_to_file,
    estimate_pixels_from_bounds,
)
from ..utils.cache import cache_key_for_images
from ..utils.cache_service import TTLCache


router = APIRouter(prefix="/api", tags=["download"])


# Cache for map IDs (to avoid repeated getMapId calls).
# Bounded LRU + 1h TTL; evicting stale entries happens on access.
_MAP_ID_CACHE = TTLCache(maxsize=256, ttl=3600, name="map_id")

# Cache for AOI-band stretch statistics. Keyed by (item_id, bbox, bands, pct_low, pct_high).
_STRETCH_STATS_CACHE = TTLCache(maxsize=256, ttl=3600, name="stretch_stats")


def _coerce_to_list(val, n: int) -> Optional[list]:
    """Normalize a min/max payload into an n-element list (or None)."""
    if val is None:
        return None
    if isinstance(val, (list, tuple)):
        out = [float(v) for v in val]
        if len(out) == 1:
            return out * n
        if len(out) == n:
            return out
        # Otherwise pad/truncate gracefully by repeating the first element.
        return [out[0]] * n
    return [float(val)] * n


def _resolve_stretch_min_max(
    base_img: ee.Image,
    aoi: ee.Geometry,
    bands: list,
    stretch_mode: Optional[str],
    explicit_min,
    explicit_max,
    pct_low: Optional[float],
    pct_high: Optional[float],
    default_min: float,
    default_max: float,
):
    """Decide (min, max) for `ee.Image.visualize()`. Both can be returned as scalars
    (uniform) or per-band lists (one entry per band, in the request order).

    - 'percentile': reduceRegion percentile per band → per-band lists.
    - 'minmax' or unset: honor caller's payload (scalar or list); fall back to defaults.
    """
    nb = len(bands) if bands else 3

    if stretch_mode == "percentile" and bands:
        lo = 2.0 if pct_low is None else float(pct_low)
        hi = 98.0 if pct_high is None else float(pct_high)
        if hi <= lo:
            hi = min(100.0, lo + 1.0)
        stats = compute_band_stretch_stats(base_img, aoi, list(bands), lo, hi)
        mins, maxs = [], []
        for b in bands:
            row = stats.get(b) or {}
            p_low = row.get("p_low")
            p_high = row.get("p_high")
            if p_low is None or p_high is None or p_high <= p_low:
                # Fall back to default for that band only.
                mins.append(float(default_min))
                maxs.append(float(default_max))
            else:
                mins.append(float(p_low))
                maxs.append(float(p_high))
        return mins, maxs

    use_min = _coerce_to_list(explicit_min, nb) or [float(default_min)] * nb
    use_max = _coerce_to_list(explicit_max, nb) or [float(default_max)] * nb
    # Guard against degenerate ranges per band.
    use_max = [
        (mx if mx > mn else mn + 1.0)
        for mn, mx in zip(use_min, use_max)
    ]
    return use_min, use_max


def _gee_tile_cache_key(req: GetTileUrlRequest) -> str:
    """Cache key includes symbology overrides so different settings get distinct entries."""
    base = cache_key_for_images([req.item_id], req.bbox)
    parts = [
        base,
        ",".join(req.bands) if req.bands else "",
        f"{req.min}" if req.min is not None else "",
        f"{req.max}" if req.max is not None else "",
        req.stretch_mode or "",
        f"{req.pct_low}" if req.pct_low is not None else "",
        f"{req.pct_high}" if req.pct_high is not None else "",
    ]
    return "|".join(parts)


@router.post("/get-gee-tile")
def get_gee_tile(req: GetTileUrlRequest):
    """Get tile URL for a Sentinel-2 image with optional symbology overrides + caching."""
    import time
    try:
        t0 = time.time()
        cache_key = _gee_tile_cache_key(req)

        # Check cache first
        cached = _MAP_ID_CACHE.get(cache_key)
        if cached is not None:
            print(f"TILE - Using cached mapId for {req.item_id}")
            return {"tile_template": cached['tile_template'], "bounds": cached['bounds']}

        aoi = bbox_to_geometry(req.bbox, req.geometry)
        base_img = resolve_item_to_image(req.item_id).clip(aoi).unmask(0)

        bands = list(req.bands) if req.bands else ["B4", "B3", "B2"]
        vis_min, vis_max = _resolve_stretch_min_max(
            base_img, aoi, bands, req.stretch_mode,
            req.min, req.max, req.pct_low, req.pct_high,
            default_min=0.0, default_max=3000.0,
        )
        vis_params = get_visualization_params(bands=bands, min_val=vis_min, max_val=vis_max)
        vis_img = base_img.visualize(**vis_params)

        t1 = time.time()
        m = vis_img.getMapId()
        t2 = time.time()

        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        # Cache the result (TTL handled by TTLCache)
        _MAP_ID_CACHE[cache_key] = {
            'tile_template': tile_template,
            'bounds': bounds,
        }

        print(f"TILE - getMapId took {t2-t1:.2f}s, total {t2-t0:.2f}s for {req.item_id}")
        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tile-proxy/{z}/{x}/{y}")
def proxy_tile(z: int, x: int, y: int, url: str):
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
def get_titiler_url(req: GetTileUrlRequest):
    """Alias for get-gee-tile for backward compatibility."""
    return get_gee_tile(req)


def _resolve_s1_image(item_id: str) -> ee.Image:
    if item_id.startswith("COPERNICUS/S1_GRD/"):
        return ee.Image(item_id)
    return ee.Image(f"COPERNICUS/S1_GRD/{item_id}")


def _resolve_s2_image(item_id: str) -> ee.Image:
    if item_id.startswith("COPERNICUS/S2"):
        if "COPERNICUS/S2_SR/" in item_id and "HARMONIZED" not in item_id:
            item_id = item_id.replace("COPERNICUS/S2_SR/", "COPERNICUS/S2_SR_HARMONIZED/")
        return ee.Image(item_id)
    return ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{item_id}")


@router.post("/get-s1-tile")
def get_s1_tile(req: S1TileRequest):
    """Get tile URL for Sentinel-1 GRD image with custom visualization."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        base_img = _resolve_s1_image(req.item_id).clip(aoi)

        bands = list(req.bands) if req.bands else ['VV', 'VH', 'VV']
        vis_min, vis_max = _resolve_stretch_min_max(
            base_img, aoi, bands, req.stretch_mode,
            req.min, req.max, req.pct_low, req.pct_high,
            default_min=-25.0, default_max=0.0,
        )
        vis_params = get_s1_visualization_params(bands=bands, min_val=vis_min, max_val=vis_max)
        vis_img = base_img.visualize(**vis_params)

        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-s2-tile-custom")
def get_s2_tile_custom(req: S2TileRequest):
    """Get tile URL for Sentinel-2 image with custom visualization bands."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        base_img = _resolve_s2_image(req.item_id).clip(aoi).unmask(0)

        bands = list(req.bands) if req.bands else ['B4', 'B3', 'B2']
        vis_min, vis_max = _resolve_stretch_min_max(
            base_img, aoi, bands, req.stretch_mode,
            req.min, req.max, req.pct_low, req.pct_high,
            default_min=0.0, default_max=3000.0,
        )
        vis_params = get_visualization_params(bands=bands, min_val=vis_min, max_val=vis_max)
        vis_img = base_img.visualize(**vis_params)

        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compute-stretch-stats", response_model=StretchStatsResponse)
def compute_stretch_stats(req: StretchStatsRequest):
    """Compute per-band min/max/percentile statistics for the requested image+AOI.

    Used by the symbology UI to populate slider initial values and the "Auto" button.
    """
    try:
        if not req.bands:
            raise HTTPException(status_code=400, detail="bands must not be empty")

        pct_low = 2.0 if req.pct_low is None else float(req.pct_low)
        pct_high = 98.0 if req.pct_high is None else float(req.pct_high)

        cache_key = "|".join([
            req.sensor,
            req.item_id,
            ",".join(f"{v:.6f}" for v in req.bbox),
            ",".join(req.bands),
            f"{pct_low}",
            f"{pct_high}",
        ])
        cached = _STRETCH_STATS_CACHE.get(cache_key)
        if cached is not None:
            return cached

        aoi = bbox_to_geometry(req.bbox, req.geometry)
        if req.sensor == "s1":
            base_img = _resolve_s1_image(req.item_id).clip(aoi)
            scale = 10
        else:
            base_img = _resolve_s2_image(req.item_id).clip(aoi).unmask(0)
            scale = 10

        stats = compute_band_stretch_stats(base_img, aoi, list(req.bands), pct_low, pct_high, scale=scale)
        result = {"bands": stats, "pct_low": pct_low, "pct_high": pct_high}
        _STRETCH_STATS_CACHE[cache_key] = result
        return result
    except HTTPException:
        raise
    except Exception as e:
        print(f"STRETCH STATS ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/download-s1-image")
def download_s1_image(req: DownloadImageRequest):
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
def download_s2_image(req: DownloadImageRequest):
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

