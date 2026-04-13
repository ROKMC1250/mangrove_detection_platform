"""
API routes for image processing operations.
"""

import os
import time
import base64
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import ee
import requests
from rasterio.io import MemoryFile

from fastapi import APIRouter, HTTPException

from .schemas import ProcessImageRequest, ChangeMonitoringRequest, ProcessSpectralImageRequest, ProcessEmitImageRequest
from ..core.config import PROJECT_ROOT, S2_BANDS, OUTPUTS_DIR
from ..core.progress import PROGRESS_TRACKER, estimate_download_time, estimate_processing_time
from ..services.earth_engine import (
    bbox_to_geometry,
    resolve_item_to_image,
    resolve_emit_image,
    get_model_names,
)
from ..services.downloader import (
    download_ee_image,
    download_tile_get_url,
    mosaic_tiles_to_file,
    estimate_file_size,
    should_use_mosaic,
    generate_output_path,
)
from ..services.spectral_analysis import (
    safe_divide,
    get_band_data_from_raster,
    calculate_mangrove_area_km2,
)
from ..services.gpu_compute import load_image_to_gpu
from ..services.visualization import (
    stretch_uint8,
    save_rgba_overlay_png_with_transparency,
    warp_rgb_and_mask_to_aoi,
    to_png_bytes,
    generate_overlay_path,
)
from ..services.model_inference import (
    is_model1_ready,
)
from ..utils.cache import (
    cache_raster_file,
    get_cached_raster_path,
    bbox_to_cache_key,
    RASTER_FILE_CACHE,
    RASTER_CACHE_LOCK,
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
async def process_image(req: ProcessImageRequest):
    """Process a satellite image with all analysis models."""
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
        
        # Prepare bands and mask
        s2_bands = S2_BANDS
        mask = ee.Image.constant(1).clip(aoi).toFloat().rename('mask')
        img = base_img.select(s2_bands).addBands(mask).unmask(0)
        download_bands = s2_bands + ['mask']
        
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

        # Band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, mask
        B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 = (
            data[0], data[1], data[2], data[3], data[4], data[5],
            data[6], data[7], data[8], data[9], data[10], data[11]
        )
        MASK = data[12]

        # GPU Loading phase — load all bands to GPU for on-demand analysis
        PROGRESS_TRACKER.start_phase(job_id, "GPU Loading", total_steps=2)
        PROGRESS_TRACKER.update_phase(job_id, "GPU Loading", 1, 'Loading bands to GPU')

        gpu_cache_key = bbox_to_cache_key(req.item_id, req.bbox)
        band_arrays = [data[i] for i in range(12)]  # B1-B12 (exclude mask)
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
        mask_bool = (MASK > 0)

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


@router.post("/change-monitoring")
async def change_monitoring(req: ChangeMonitoringRequest):
    """Run change monitoring analysis on multiple images."""
    try:
        t0 = time.time()
        job_id = req.job_id or f"change-{int(t0)}"
        
        phases = [
            ("Initialization", 5.0),
            ("Image Discovery", 15.0),
            ("Composite Generation & Analysis", 75.0),
            ("Finalization", 5.0)
        ]
        PROGRESS_TRACKER.create_job(job_id, phases)
        
        PROGRESS_TRACKER.start_phase(job_id, "Initialization", total_steps=2)
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 1, "Setting up AOI geometry")
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        aoi_rect = aoi.bounds()
        
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 2, "Resolving selected images")
        
        image_date_map = []
        
        PROGRESS_TRACKER.start_phase(job_id, "Image Discovery", total_steps=len(req.image_ids))
        
        for idx, img_id in enumerate(req.image_ids):
            PROGRESS_TRACKER.update_phase(job_id, "Image Discovery", idx, f"Resolving image {img_id}")
            try:
                img = resolve_item_to_image(img_id)
                date_ts = img.get('system:time_start').getInfo()
                date_str = datetime.fromtimestamp(date_ts / 1000).strftime('%Y-%m-%d')
                image_date_map.append({'id': img_id, 'date': date_str, 'ts': date_ts})
            except Exception as e:
                print(f"Error resolving image {img_id}: {e}")
        
        image_date_map.sort(key=lambda x: x['ts'])
        
        date_groups = defaultdict(list)
        for item in image_date_map:
            date_groups[item['date']].append(item['id'])
        
        sorted_dates = sorted(date_groups.keys())
        print(f"CHANGE MONITORING - Found {len(sorted_dates)} time points from {len(req.image_ids)} images")
        
        PROGRESS_TRACKER.complete_phase(job_id, "Image Discovery", f"Resolved {len(sorted_dates)} unique dates")
        
        results = []
        PROGRESS_TRACKER.start_phase(job_id, "Composite Generation & Analysis", total_steps=len(sorted_dates))
        
        for i, date_str in enumerate(sorted_dates):
            img_ids = date_groups[date_str]
            PROGRESS_TRACKER.update_phase(job_id, "Composite Generation & Analysis", i, f'Processing {date_str}')
            
            try:
                ee_images = [resolve_item_to_image(iid).clip(aoi) for iid in img_ids]
                
                if len(ee_images) > 1:
                    composite = ee.ImageCollection(ee_images).mosaic()
                else:
                    composite = ee_images[0]
                
                mask_band = ee.Image.constant(1).clip(aoi).toFloat().rename('mask')
                composite = composite.addBands(mask_band)
                s2_bands_with_mask = S2_BANDS + ["mask"]
                composite = composite.select(s2_bands_with_mask)
                
                temp_path = generate_output_path(f"change_{date_str}", img_ids[0])
                
                px_w, px_h, est_mb = estimate_file_size(aoi_rect, 10, len(s2_bands_with_mask), 2)
                
                if est_mb < 30:
                    blob = download_tile_get_url(composite, s2_bands_with_mask, aoi_rect, 10)
                    with open(temp_path, 'wb') as f:
                        f.write(blob)
                else:
                    mosaic_tiles_to_file(composite, s2_bands_with_mask, aoi_rect, 10, 2048, temp_path)
                
                # Cache the downloaded TIF
                first_image_id = img_ids[0]
                cache_raster_file(first_image_id, req.bbox, temp_path)
                
                # Analyze
                # Band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, mask
                with rasterio.open(temp_path) as src:
                    data = src.read()
                    B3 = data[2].astype(np.float32)   # Green
                    B4 = data[3].astype(np.float32)   # Red
                    B8 = data[7].astype(np.float32)   # NIR
                    MASK = data[12]
                    
                    mask_bool = (MASK > 0)
                    
                    if req.model_id == 'model3':
                        numerator = (B3 - B8)
                        denominator = (B3 + B8 + 1e-6)
                        index_name = 'NDWI'
                    else:
                        numerator = (B8 - B4)
                        denominator = (B8 + B4 + 1e-6)
                        index_name = 'NDVI'
                    
                    spectral_index = numerator / denominator
                    spectral_index[~mask_bool] = np.nan
                    
                    valid_pixels = np.sum(mask_bool)
                    area_km2 = (valid_pixels * 100) / 1000000.0
                    
                    if valid_pixels > 0:
                        valid_data = spectral_index[mask_bool]
                        p1, p99 = np.percentile(valid_data, [1, 99])
                        valid_data = valid_data[(valid_data >= p1) & (valid_data <= p99)]
                        mean_spectral = float(np.mean(valid_data)) if len(valid_data) > 0 else 0.0
                    else:
                        mean_spectral = 0.0
                
                results.append({
                    'period': date_str,
                    'date': date_str,
                    'image_id': first_image_id,
                    'image_ids': img_ids,
                    'area_km2': area_km2,
                    'mean_spectral': mean_spectral,
                    'index_name': index_name,
                    'tif_cached': True
                })
                
            except Exception as e:
                print(f"Error analyzing composite {date_str}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        PROGRESS_TRACKER.complete_phase(job_id, "Composite Generation & Analysis", f"Analyzed {len(results)} time points")
        
        PROGRESS_TRACKER.start_phase(job_id, "Finalization", total_steps=1)
        PROGRESS_TRACKER.update_phase(job_id, "Finalization", 1, "Finalizing results")
        
        if results:
            areas = [r['area_km2'] for r in results]
            spectrals = [r['mean_spectral'] for r in results]
            stats = {
                'total_change_km2': areas[-1] - areas[0] if len(areas) > 1 else 0,
                'max_area_km2': max(areas),
                'min_area_km2': min(areas),
                'avg_spectral': sum(spectrals) / len(spectrals)
            }
        else:
            stats = {}
        
        PROGRESS_TRACKER.complete_job(job_id, {"status": "success", "results": len(results)})
        
        return {
            "job_id": job_id,
            "status": "success",
            "results": results,
            "statistics": stats
        }
        
    except Exception as e:
        PROGRESS_TRACKER.error_job(job_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-spectral-image")
async def process_spectral_image(req: ProcessSpectralImageRequest):
    """Generate a spectral analysis image (NDVI, NDMI, MVI) from cached or downloaded raster data."""
    try:
        image_id = req.item_id
        spectral_method = req.spectral_method.lower()
        bbox = req.bbox
        
        method_to_model = {'ndvi': 'model2', 'ndmi': 'model3', 'mvi': 'model4'}
        if spectral_method not in method_to_model:
            raise HTTPException(status_code=400, detail=f"Unsupported spectral method: {spectral_method}")
        
        # Check cache or download
        cached_raster_path = get_cached_raster_path(image_id, bbox)
        
        if not cached_raster_path:
            print(f"SPECTRAL IMAGE - No cached raster, downloading for {image_id}")
            aoi = bbox_to_geometry(bbox, req.geometry)
            aoi_rect = aoi.bounds()
            base_img = resolve_item_to_image(image_id).clip(aoi)
            
            s2_bands = S2_BANDS
            mask = ee.Image.constant(1).clip(aoi).toFloat().rename('mask')
            img = base_img.select(s2_bands).addBands(mask).unmask(0)
            download_bands = s2_bands + ['mask']
            
            out_path = generate_output_path(f"spectral_{spectral_method}", image_id)
            download_ee_image(img, download_bands, aoi_rect, 10, out_path)
            
            cached_raster_path = out_path
            cache_raster_file(image_id, bbox, out_path)
        
        # Process
        with rasterio.open(cached_raster_path) as src:
            B4 = src.read(1).astype(np.float32)
            B3 = src.read(2).astype(np.float32)
            B8 = src.read(4).astype(np.float32)
            B11 = src.read(5).astype(np.float32)
            B12 = src.read(6).astype(np.float32)
            MASK = src.read(7).astype(np.float32)
            
            src_transform = src.transform
            src_crs = src.crs
            src_bounds = src.bounds
            
            min_lon, min_lat, max_lon, max_lat = bbox
            mask_bool = (MASK > 0)
            
            def apply_mask(arr):
                res = arr.copy()
                res[~mask_bool] = np.nan
                return res
            
            if spectral_method == 'ndvi':
                index = safe_divide(apply_mask(B8) - apply_mask(B4), apply_mask(B8) + apply_mask(B4))
                colormap = 'RdYlGn'
                vmin, vmax = -1, 1
            elif spectral_method == 'ndmi':
                index = safe_divide(apply_mask(B12) - apply_mask(B3), apply_mask(B12) + apply_mask(B3))
                colormap = 'RdYlGn'
                vmin, vmax = -1, 1
            elif spectral_method == 'mvi':
                index = safe_divide(apply_mask(B8) - apply_mask(B3), apply_mask(B11) - apply_mask(B3))
                colormap = 'viridis'
                vmin, vmax = None, None
            
            index_rgb, actual_min, actual_max = create_index_visualization(index, colormap, vmin, vmax)
            valid_mask = np.isfinite(index) & mask_bool
            
            aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
                index_rgb, valid_mask, src_transform, src_crs,
                (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=req.geometry
            )
        
        # Save
        timestamp = int(time.time() * 1000)
        png_filename = f"{image_id}_{spectral_method}_overlay_{timestamp}.png"
        png_path = os.path.join(OUTPUTS_DIR, png_filename)
        save_rgba_overlay_png_with_transparency(aoi_rgb, aoi_mask, png_path)
        
        png_url = f"/outputs/{png_filename}"
        
        return {
            'url': png_url,
            'tile_url': png_url,
            'static_url': png_url,
            'bounds': [min_lon, min_lat, max_lon, max_lat],
            'method': spectral_method,
            'range': {'min': float(actual_min), 'max': float(actual_max)}
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"SPECTRAL IMAGE ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process-emit-image")
async def process_emit_image(req: ProcessEmitImageRequest):
    """Process EMIT hyperspectral image with selected bands for custom visualization only."""
    try:
        t0 = time.time()
        job_id = req.job_id or f"emit-{int(t0)}"
        print(f"EMIT PROCESS - Starting with job_id: {job_id}, bands: {req.selected_bands}")
        
        # Validate band selection
        if len(req.selected_bands) == 0:
            raise HTTPException(status_code=400, detail="At least one band must be selected")
        
        if len(req.selected_bands) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 bands can be selected")
        
        # Initialize progress tracking
        phases = [
            ("Initialization", 10.0),
            ("Download", 50.0),
            ("Visualization", 30.0),
            ("Finalization", 10.0)
        ]
        PROGRESS_TRACKER.create_job(job_id, phases)
        
        # Initialization phase
        PROGRESS_TRACKER.start_phase(job_id, "Initialization", total_steps=3)
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 1, "Setting up AOI geometry")
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        aoi_rect = aoi.bounds()
        
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 2, "Resolving EMIT image")
        base_img = resolve_emit_image(req.item_id).clip(aoi)
        
        # Select only requested bands
        selected_bands = req.selected_bands
        img = base_img.select(selected_bands)
        
        PROGRESS_TRACKER.update_phase(job_id, "Initialization", 3, "Preparing for download")
        PROGRESS_TRACKER.complete_phase(job_id, "Initialization", "Ready for download")
        
        # Generate output path
        out_path = generate_output_path("emit", req.item_id)
        
        # Download phase
        PROGRESS_TRACKER.start_phase(job_id, "Download", total_steps=1)
        download_ee_image(img, selected_bands, aoi_rect, 60, out_path, job_id)  # EMIT uses 60m resolution
        
        # Cache the raster file
        cache_raster_file(req.item_id, req.bbox, out_path)
        
        PROGRESS_TRACKER.complete_phase(job_id, "Download", "Download complete")
        
        # Visualization phase
        PROGRESS_TRACKER.start_phase(job_id, "Visualization", total_steps=1)
        
        min_lon, min_lat, max_lon, max_lat = req.bbox
        
        with rasterio.open(out_path) as ds:
            data = ds.read().astype(np.float32)
            src_transform = ds.transform
            src_crs = ds.crs
            height, width = ds.height, ds.width
        
        # Create mask (all valid pixels)
        mask = np.ones((height, width), dtype=bool)
        for i in range(data.shape[0]):
            mask &= np.isfinite(data[i])
        
        visualization_results = {}
        
        if req.visualization_type == "rgb":
            # RGB composite visualization
            if req.rgb_bands is None or len(req.rgb_bands) != 3:
                raise HTTPException(status_code=400, detail="RGB visualization requires exactly 3 bands")
            
            # Find band indices
            r_idx = selected_bands.index(req.rgb_bands[0]) if req.rgb_bands[0] in selected_bands else None
            g_idx = selected_bands.index(req.rgb_bands[1]) if req.rgb_bands[1] in selected_bands else None
            b_idx = selected_bands.index(req.rgb_bands[2]) if req.rgb_bands[2] in selected_bands else None
            
            if r_idx is None or g_idx is None or b_idx is None:
                raise HTTPException(status_code=400, detail="RGB bands must be in selected_bands list")
            
            # Extract RGB bands
            r_band = data[r_idx]
            g_band = data[g_idx]
            b_band = data[b_idx]
            
            # Stretch to uint8
            r_stretched = stretch_uint8(r_band)
            g_stretched = stretch_uint8(g_band)
            b_stretched = stretch_uint8(b_band)
            
            # Create RGB composite
            rgb_composite = np.dstack([r_stretched, g_stretched, b_stretched])
            rgb_composite[~mask] = 0
            
            # Create preview thumbnail
            from PIL import Image
            import io as pyio
            img_pil = Image.fromarray(rgb_composite, mode='RGB')
            img_pil.thumbnail((512, 512), Image.LANCZOS)
            buf = pyio.BytesIO()
            img_pil.save(buf, format='PNG')
            preview_bytes = buf.getvalue()
            
            # Create AOI-aligned overlay
            aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
                rgb_composite, mask, src_transform, src_crs,
                (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=req.geometry
            )
            
            # Save overlay
            overlay_png_path = generate_overlay_path("emit_rgb_overlay")
            save_rgba_overlay_png_with_transparency(aoi_rgb, aoi_mask, overlay_png_path)
            
            visualization_results['preview'] = base64.b64encode(preview_bytes).decode('utf-8')
            visualization_results['overlay_url'] = f"/api/proxy-file?path={requests.utils.quote('file://' + overlay_png_path)}"
            visualization_results['overlay_meta'] = {
                'width': int(aoi_w),
                'height': int(aoi_h),
                'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
            }
            
        elif req.visualization_type == "index":
            # Custom index visualization
            if req.index_bands is None or len(req.index_bands) != 2:
                raise HTTPException(status_code=400, detail="Index visualization requires exactly 2 bands")
            
            # Find band indices
            a_idx = selected_bands.index(req.index_bands[0]) if req.index_bands[0] in selected_bands else None
            b_idx = selected_bands.index(req.index_bands[1]) if req.index_bands[1] in selected_bands else None
            
            if a_idx is None or b_idx is None:
                raise HTTPException(status_code=400, detail="Index bands must be in selected_bands list")
            
            # Calculate normalized difference index
            band_a = data[a_idx]
            band_b = data[b_idx]
            
            index = safe_divide(band_a - band_b, band_a + band_b)
            
            # Create visualization
            index_rgb, actual_min, actual_max = create_index_visualization(
                index, req.colormap or "RdYlGn", vmin=-1, vmax=1
            )
            valid_mask = np.isfinite(index) & mask
            
            # Create preview thumbnail
            from PIL import Image
            import io as pyio
            img_pil = Image.fromarray(index_rgb, mode='RGB')
            img_pil.thumbnail((512, 512), Image.LANCZOS)
            buf = pyio.BytesIO()
            img_pil.save(buf, format='PNG')
            preview_bytes = buf.getvalue()
            
            # Create AOI-aligned overlay
            aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
                index_rgb, valid_mask, src_transform, src_crs,
                (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=req.geometry
            )
            
            # Save overlay
            overlay_png_path = generate_overlay_path("emit_index_overlay")
            save_rgba_overlay_png_with_transparency(aoi_rgb, aoi_mask, overlay_png_path)
            
            visualization_results['preview'] = base64.b64encode(preview_bytes).decode('utf-8')
            visualization_results['overlay_url'] = f"/api/proxy-file?path={requests.utils.quote('file://' + overlay_png_path)}"
            visualization_results['overlay_meta'] = {
                'width': int(aoi_w),
                'height': int(aoi_h),
                'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
            }
            visualization_results['range'] = {'min': float(actual_min), 'max': float(actual_max)}
        else:
            raise HTTPException(status_code=400, detail=f"Unknown visualization_type: {req.visualization_type}")
        
        PROGRESS_TRACKER.complete_phase(job_id, "Visualization", "Visualization complete")
        
        # Finalization
        PROGRESS_TRACKER.start_phase(job_id, "Finalization", total_steps=1)
        PROGRESS_TRACKER.complete_phase(job_id, "Finalization", "Processing complete")
        PROGRESS_TRACKER.complete_job(job_id)
        
        return {
            'job_id': job_id,
            'visualization': visualization_results,
            'selected_bands': selected_bands,
            'visualization_type': req.visualization_type,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"EMIT PROCESS ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

