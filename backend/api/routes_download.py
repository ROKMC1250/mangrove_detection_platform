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


@router.post("/get-gee-tile")
async def get_gee_tile(req: GetTileUrlRequest):
    """Get tile URL for a Sentinel-2 image."""
    try:
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        base_img = resolve_item_to_image(req.item_id).clip(aoi)
        vis_params = get_visualization_params()
        vis_img = base_img.unmask(0).visualize(**vis_params)
        
        m = vis_img.getMapId()
        tile_template = m["tile_fetcher"].url_format
        min_lon, min_lat, max_lon, max_lat = req.bbox
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]

        return {"tile_template": tile_template, "bounds": bounds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        
        # Raw download - all 10m + 20m bands
        bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
        filename = f"S2_SR_HARMONIZED_{item_id.split('/')[-1]}.tif"
        temp_dir = tempfile.mkdtemp(prefix="s2_download_")
        out_path = os.path.join(temp_dir, filename)
        
        bounds_info = aoi_rect.getInfo()['coordinates'][0]
        min_lon, min_lat = bounds_info[0]
        max_lon, max_lat = bounds_info[2]
        
        px_w, px_h = estimate_pixels_from_bounds(min_lon, min_lat, max_lon, max_lat, 10)
        num_pixels = px_w * px_h
        
        max_tile = 2048
        if num_pixels > max_tile * max_tile:
            mosaic_tiles_to_file(base_img, bands, aoi_rect, 10, max_tile, out_path)
        else:
            blob = download_tile_get_url(base_img, bands, aoi_rect, 10)
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

