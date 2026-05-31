"""
API routes for image processing operations.
"""

import os
import time
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import ee
import requests
from rasterio.io import MemoryFile

from fastapi import APIRouter, HTTPException

from .schemas import ProcessImageRequest
from ..services.flood_inference import ensure_s1_raster_cached
from ..core.config import S2_BANDS
from ..core.progress import PROGRESS_TRACKER
from ..services.earth_engine import (
    bbox_to_geometry,
    resolve_item_to_image,
    get_model_names,
)
from ..services.downloader import (
    download_ee_image,
    generate_output_path,
)
from ..services.gpu_compute import load_image_to_gpu
from ..services.visualization import (
    stretch_uint8,
    warp_rgb_and_mask_to_aoi,
    to_png_bytes,
)
from ..utils.cache import (
    cache_raster_file,
    bbox_to_cache_key,
)


router = APIRouter(prefix="/api", tags=["process"])


def _create_cloud_mask(image_path: str, bbox: List[float], item_id: str, 
                       aoi: ee.Geometry) -> Tuple[Optional[bytes], Optional[str], Optional[Dict]]:
    """Create cloud mask overlay using s2cloudless data."""
    try:
        print(f"CLOUD MASK - Generating cloud mask for {item_id}")
        
        base_img = resolve_item_to_image(item_id)
        s2_index = base_img.get('system:index').getInfo()
        
        s2cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        cloudless_col = s2cloudless_col.filter(ee.Filter.eq('system:index', s2_index))
        
        if cloudless_col.size().getInfo() == 0:
            print(f"CLOUD MASK - No cloud probability image found for {item_id}")
            return None, None, None
        
        cloudless = cloudless_col.first()
        cloud_prob = cloudless.select('probability')
        cloud_prob_aoi = cloud_prob.clip(aoi)
        cloud_mask_ee = cloud_prob_aoi.gt(30)
        
        min_lon, min_lat, max_lon, max_lat = bbox
        cloud_mask_url = cloud_mask_ee.getDownloadURL({
            'region': aoi,
            'scale': 20,
            'format': 'GEO_TIFF'
        })
        
        response = requests.get(cloud_mask_url, timeout=300)
        response.raise_for_status()
        
        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                cloud_mask_data = src.read(1)
                cloud_transform = src.transform
                cloud_crs = src.crs
        
        print(f"CLOUD MASK - Downloaded mask shape: {cloud_mask_data.shape}")
        
        with rasterio.open(image_path) as src:
            b4 = src.read(1)
            b3 = src.read(2)
            b2 = src.read(3)
            src_transform = src.transform
            src_crs = src.crs
        
        base_rgb = np.dstack([stretch_uint8(b4), stretch_uint8(b3), stretch_uint8(b2)])
        
        from PIL import Image
        if cloud_mask_data.shape != b4.shape:
            cloud_mask_pil = Image.fromarray(cloud_mask_data.astype(np.uint8), mode='L')
            cloud_mask_pil = cloud_mask_pil.resize((b4.shape[1], b4.shape[0]), Image.NEAREST)
            cloud_mask_data = np.array(cloud_mask_pil)
        
        cloud_overlay = np.zeros_like(base_rgb, dtype=np.uint8)
        cloud_overlay[..., 0] = 255
        cloud_overlay[..., 1] = 165
        cloud_overlay[..., 2] = 0
        
        alpha = 0.5
        cloud_pixels = (cloud_mask_data > 0)
        composed = base_rgb.copy()
        composed[cloud_pixels] = (base_rgb[cloud_pixels] * (1 - alpha) + cloud_overlay[cloud_pixels] * alpha).astype(np.uint8)
        
        preview_bytes = to_png_bytes(composed)
        
        cloud_rgb = np.zeros_like(base_rgb, dtype=np.uint8)
        cloud_rgb[cloud_pixels, 0] = 255
        cloud_rgb[cloud_pixels, 1] = 165
        cloud_rgb[cloud_pixels, 2] = 0
        
        cloud_aoi_rgb, cloud_aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            cloud_rgb, cloud_pixels, src_transform, src_crs,
            (min_lon, min_lat, max_lon, max_lat), scale_m=10
        )
        
        from ..services.gpu_compute import rgb_mask_to_base64_gpu
        overlay_b64 = rgb_mask_to_base64_gpu(cloud_aoi_rgb, cloud_aoi_mask)

        overlay_meta = {
            'width': int(aoi_w),
            'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        }

        print(f"CLOUD MASK - Created overlay (base64, no file)")
        return preview_bytes, overlay_b64, overlay_meta
        
    except Exception as e:
        print(f"CLOUD MASK - Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def _download_and_process_alphaearth(item_id: str, bbox: List[float], aoi: ee.Geometry,
                                      src_transform, src_crs) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """Download AlphaEarth Satellite Embedding data and create RGB visualization."""
    try:
        print(f"ALPHAEARTH - Processing embedding data for {item_id}")
        
        base_img = resolve_item_to_image(item_id)
        img_date = base_img.get('system:time_start').getInfo()
        img_datetime = datetime.fromtimestamp(img_date / 1000)
        year = img_datetime.year
        
        if year >= 2025:
            year = 2024
            print(f"ALPHAEARTH - Year {img_datetime.year} not available, using 2024")
        
        embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
        start_date = ee.Date.fromYMD(year, 1, 1)
        end_date = start_date.advance(1, 'year')
        
        filtered_embeddings = embeddings \
            .filter(ee.Filter.date(start_date, end_date)) \
            .filter(ee.Filter.bounds(aoi))
        
        embeddings_image = filtered_embeddings.mosaic().clip(aoi)
        selected_bands = embeddings_image.select(['A01', 'A16', 'A32'])
        
        download_url = selected_bands.getDownloadURL({
            'region': aoi,
            'scale': 10,
            'format': 'GEO_TIFF',
            'bands': ['A01', 'A16', 'A32']
        })
        
        response = requests.get(download_url, timeout=600)
        response.raise_for_status()
        
        with MemoryFile(response.content) as memfile:
            with memfile.open() as src:
                pca_data = src.read()
                emb_transform = src.transform
                emb_crs = src.crs
        
        h, w = pca_data.shape[1], pca_data.shape[2]
        bands_image_array = pca_data.transpose(1, 2, 0)
        valid_mask = np.any(np.isfinite(pca_data), axis=0)
        
        pca_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for i in range(3):
            channel = bands_image_array[:, :, i]
            channel = np.nan_to_num(channel, nan=0.0, posinf=0.0, neginf=0.0)
            channel_norm = np.clip((channel + 0.3) / 0.6, 0, 1)
            pca_rgb[:, :, i] = (channel_norm * 255).astype(np.uint8)
        
        metadata = {
            'transform': emb_transform,
            'crs': emb_crs,
            'shape': (h, w),
            'year': year,
            'mask': valid_mask
        }
        
        return pca_rgb, metadata
        
    except Exception as e:
        print(f"ALPHAEARTH - Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None


@router.post("/process-image")
def process_image(req: ProcessImageRequest):
    """Process a satellite image with all analysis models.

    Declared as plain `def` so FastAPI offloads it to its threadpool. The body
    makes heavy blocking calls (GEE download, rasterio, GPU upload) — keeping
    this on the event loop would serialise every concurrent request.
    """
    try:
        t0 = time.time()
        job_id = req.job_id or f"job-{int(t0)}"
        print(f"PROCESS IMAGE - Using job_id: {job_id}")
        
        # Initialize progress tracking
        phases = [
            ("Initialization", 5.0),
            ("Download", 40.0),
            ("Model Inference", 30.0),
            ("GPU Loading", 10.0),
            ("Visualization", 10.0),
            ("Finalization", 5.0)
        ]
        PROGRESS_TRACKER.create_job(job_id, phases)
        
        # Initialization phase
        PROGRESS_TRACKER.start_phase(job_id, "Initialization", total_steps=3)
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 1, "Setting up AOI geometry")
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        aoi_rect = aoi.bounds()
        
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 2, "Resolving image reference")
        base_img = resolve_item_to_image(req.item_id).clip(aoi)
        
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 3, "Preparing for download")
        PROGRESS_TRACKER.complete_phase(job_id, "Initialization", "Ready for download")
        
        # Prepare bands. Use bicubic resampling so 20m/60m bands (B5-B7, B8A, B11, B12, B1, B9)
        # are smoothly upsampled to 10m instead of nearest-neighbor blocks.
        s2_bands = S2_BANDS
        img = base_img.select(s2_bands).resample('bicubic').unmask(0)
        download_bands = s2_bands
        
        # Generate output path
        out_path = generate_output_path("process", req.item_id)
        
        # Download phase
        PROGRESS_TRACKER.start_phase(job_id, "Download", total_steps=1)
        download_ee_image(img, download_bands, aoi_rect, 10, out_path, job_id)
        
        # Cache the raster file
        cache_raster_file(req.item_id, req.bbox, out_path)
        
        # Model Inference phase (cloud mask only; mangrove segmentation runs on-demand)
        PROGRESS_TRACKER.start_phase(job_id, "Model Inference", total_steps=1)
        PROGRESS_TRACKER.update_phase(job_id, "Model Inference", 0, 'Generating cloud mask', 0.1)

        cloud_preview, cloud_overlay_png_path, cloud_overlay_meta = _create_cloud_mask(
            out_path, req.bbox, req.item_id, aoi_rect
        )

        PROGRESS_TRACKER.complete_phase(job_id, "Model Inference", 'Cloud mask completed')
        
        # Read raster data for analysis
        with rasterio.open(out_path) as ds:
            data = ds.read().astype(np.float32)
            src_transform = ds.transform
            src_crs = ds.crs
            if ds.crs and ds.crs.to_string().upper() != 'EPSG:4326':
                l, b, r, t = transform_bounds(ds.crs, 'EPSG:4326', *ds.bounds, densify_pts=21)
            else:
                l, b, r, t = ds.bounds

        # GPU Loading phase — load all 12 bands (B1-B12) to GPU for on-demand analysis
        PROGRESS_TRACKER.start_phase(job_id, "GPU Loading", total_steps=2)
        PROGRESS_TRACKER.update_phase(job_id, "GPU Loading", 1, 'Loading bands to GPU')

        gpu_cache_key = bbox_to_cache_key(req.item_id, req.bbox)
        band_arrays = [data[i] for i in range(12)]
        load_image_to_gpu(gpu_cache_key, band_arrays)

        PROGRESS_TRACKER.update_phase(job_id, "GPU Loading", 2, 'GPU loading complete')
        PROGRESS_TRACKER.complete_phase(job_id, "GPU Loading", 'All bands loaded to GPU')

        # AlphaEarth processing
        alphaearth_pca_rgb, alphaearth_meta = _download_and_process_alphaearth(
            req.item_id, req.bbox, aoi_rect, src_transform, src_crs
        )
        
        # Visualization phase
        PROGRESS_TRACKER.start_phase(job_id, "Visualization", total_steps=3)
        PROGRESS_TRACKER.update_phase(job_id, "Visualization", 1, 'Aligning overlays to AOI')

        min_lon, min_lat, max_lon, max_lat = req.bbox
        mask_bool = np.ones(data.shape[1:], dtype=bool)

        from ..services.gpu_compute import rgb_mask_to_base64_gpu

        # AlphaEarth overlay
        alphaearth_overlay_b64 = None
        if alphaearth_pca_rgb is not None and alphaearth_meta is not None:
            valid_mask = alphaearth_meta.get('mask', np.ones(alphaearth_pca_rgb.shape[:2], dtype=bool))
            alphaearth_aoi_rgb, alphaearth_aoi_mask, _, _ = warp_rgb_and_mask_to_aoi(
                alphaearth_pca_rgb, valid_mask,
                alphaearth_meta['transform'], alphaearth_meta['crs'],
                (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=req.geometry
            )
            mask_for_save = (alphaearth_aoi_mask > 0)
            alphaearth_overlay_b64 = rgb_mask_to_base64_gpu(alphaearth_aoi_rgb, mask_for_save)
        
        PROGRESS_TRACKER.update_phase(job_id, "Visualization", 2, 'Generating thumbnails')

        # Build thumbnails (RGB = B4(Red), B3(Green), B2(Blue))
        thumbs = {}
        if cloud_preview:
            thumbs['cloud_mask'] = cloud_preview
        if alphaearth_pca_rgb is not None:
            thumbs['model5'] = to_png_bytes(alphaearth_pca_rgb)

        model_names = get_model_names()

        analysis_results = {}
        for mid, png in thumbs.items():
            b64 = base64.b64encode(png).decode('ascii')
            analysis_results[mid] = {
                'name': model_names.get(mid, mid),
                'preview_url': f'data:image/png;base64,{b64}'
            }

        # Add overlay URLs
        def to_outputs_url(path):
            """Convert absolute path to /outputs/filename URL (legacy for model1)"""
            return f"/outputs/{os.path.basename(path)}"

        if cloud_overlay_png_path and cloud_overlay_meta:
            analysis_results['cloud_mask']['overlay_url'] = cloud_overlay_png_path  # already base64

        if alphaearth_overlay_b64:
            analysis_results['model5']['overlay_url'] = alphaearth_overlay_b64

        # Compute overlay_meta from AlphaEarth warp or model1
        aoi_w = aoi_h = 0
        if alphaearth_pca_rgb is not None and alphaearth_meta is not None:
            # Already computed above
            pass
        analysis_results['overlay_meta'] = cloud_overlay_meta or {
            'width': int(aoi_w), 'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        }
        
        PROGRESS_TRACKER.complete_phase(job_id, "Visualization", 'Visualizations completed')
        
        # Finalization
        PROGRESS_TRACKER.start_phase(job_id, "Finalization", total_steps=1)
        PROGRESS_TRACKER.update_phase(job_id, "Finalization", 1, 'Preparing final results')
        
        total = time.time() - t0
        print(f"TIMING - total process_image: {total:.2f}s")
        
        file_uri = f'file://{out_path}'
        resp = {
            'original_cog_uri': file_uri,
            'original_10m_gcs_uri': file_uri,
            'original_20m_gcs_uri': file_uri,
            'analysis_results': analysis_results,
            'gpu_cache_key': gpu_cache_key,
            'export_started': True,
            'message': 'Processing completed'
        }
        
        PROGRESS_TRACKER.complete_job(job_id, f'Processing completed in {total:.1f}s')
        return resp
        
    except Exception as e:
        PROGRESS_TRACKER.error_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-s1-image")
def process_s1_image(req: ProcessImageRequest):
    """
    Download a Sentinel-1 GRD (VV+VH) image and cache it for flood segmentation.

    Mirrors the user-visible flow of /api/process-image but skips the
    S2-specific spectral pipeline (NDVI, AlphaEarth, cloud mask). The
    cached GeoTIFF is then consumed by /api/flood-segmentation/run.
    """
    t0 = time.time()
    job_id = req.job_id or f"job-{int(t0)}"
    print(f"PROCESS S1 IMAGE - Using job_id: {job_id}")

    phases = [
        ("Initialization", 5.0),
        ("Download", 90.0),
        ("Finalization", 5.0),
    ]
    PROGRESS_TRACKER.create_job(job_id, phases)

    try:
        PROGRESS_TRACKER.start_phase(job_id, "Initialization", total_steps=1)
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 1, "Preparing S1 download")
        PROGRESS_TRACKER.complete_phase(job_id, "Initialization", "Ready")

        PROGRESS_TRACKER.start_phase(job_id, "Download", total_steps=1)
        cached_path = ensure_s1_raster_cached(
            image_id=req.item_id,
            bbox=req.bbox,
            geometry=req.geometry,
            job_id=job_id,
        )

        PROGRESS_TRACKER.start_phase(job_id, "Finalization", total_steps=1)
        with rasterio.open(cached_path) as src:
            height, width = src.height, src.width
            n_bands = src.count
        PROGRESS_TRACKER.complete_phase(job_id, "Finalization", "Cached raster ready")

        total = time.time() - t0
        PROGRESS_TRACKER.complete_job(job_id, f'S1 ready in {total:.1f}s')

        return {
            'item_id': req.item_id,
            'cached_path': cached_path,
            'width': width,
            'height': height,
            'bands': n_bands,
            'bbox': req.bbox,
            'analysis_results': {},  # No S2-style spectral overlays for S1
            'satellite': 'Sentinel-1',
        }

    except Exception as e:
        PROGRESS_TRACKER.error_job(job_id, str(e))
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

