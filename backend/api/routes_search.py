"""
API routes for image search operations.
"""

import time
from typing import List

from fastapi import APIRouter, HTTPException

from .schemas import SearchImagesRequest
from ..services.earth_engine import (
    bbox_to_geometry,
    get_s2_collection,
    get_s1_collection,
    get_emit_collection,
    resolve_emit_image,
    get_emit_band_metadata,
    build_feature_collection_simple,
    build_feature_collection_simple_s1,
    build_feature_collection_simple_emit,
)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search-images")
async def search_images(req: SearchImagesRequest):
    """Search for Sentinel-2 SR images."""
    try:
        print(f"SEARCH - Starting search with params: start={req.start_date}, end={req.end_date}, cloud_max={req.cloud_cover_max}, limit={req.limit}")
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        user_cloud_max = req.cloud_cover_max or 100
        limit = min(req.limit or 20, 100)
        
        col = get_s2_collection(aoi, req.start_date, req.end_date, user_cloud_max)
        fc = build_feature_collection_simple(col, aoi)
        
        fc = fc.sort('cloud_cover')
        fc = fc.sort('aoi_overlap', False)
        
        lst = fc.limit(limit).toList(limit)
        
        print(f"SEARCH - Fetching {limit} results from Earth Engine...")
        features = lst.getInfo()
        print(f"SEARCH - Got {len(features)} features")

        results: List[dict] = []
        for f in features:
            props = f.get('properties', {})
            image_id = props.get('id')
            time_start = props.get('datetime')
            dt_iso = None
            if time_start is not None:
                try:
                    dt_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(time_start) / 1000))
                except Exception:
                    dt_iso = None
            cloud = props.get('cloud_cover')
            overlap = props.get('aoi_overlap')
            info = {
                'id': image_id,
                'datetime': dt_iso,
                'cloud_cover': float(cloud) if cloud is not None else None,
                'collection': 'Sentinel-2',
                'assets': {},
                'aoi_overlap': float(overlap) if overlap is not None else None,
            }
            results.append(info)

        print(f"SEARCH - Returning {len(results)} results")
        return {"images": results}
    except Exception as e:
        print(f"SEARCH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-s1-images")
async def search_s1_images(req: SearchImagesRequest):
    """Search for Sentinel-1 GRD images."""
    try:
        print(f"S1 SEARCH - Starting search with params: start={req.start_date}, end={req.end_date}, limit={req.limit}")
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        limit = min(req.limit or 20, 100)
        
        col = get_s1_collection(aoi, req.start_date, req.end_date)
        fc = build_feature_collection_simple_s1(col, aoi)
        
        fc = fc.sort('datetime', False)
        fc = fc.sort('aoi_overlap', False)
        
        lst = fc.limit(limit).toList(limit)
        
        print(f"S1 SEARCH - Fetching {limit} results from Earth Engine...")
        features = lst.getInfo()
        print(f"S1 SEARCH - Got {len(features)} features")

        results: List[dict] = []
        for f in features:
            props = f.get('properties', {})
            image_id = props.get('id')
            time_start = props.get('datetime')
            dt_iso = None
            if time_start is not None:
                try:
                    dt_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(time_start) / 1000))
                except Exception:
                    dt_iso = None
            orbit = props.get('orbit')
            overlap = props.get('aoi_overlap')
            info = {
                'id': image_id,
                'datetime': dt_iso,
                'cloud_cover': None,
                'orbit': orbit,
                'collection': 'Sentinel-1',
                'assets': {},
                'aoi_overlap': float(overlap) if overlap is not None else None,
            }
            results.append(info)

        print(f"S1 SEARCH - Returning {len(results)} results")
        return {"images": results}
    except Exception as e:
        print(f"S1 SEARCH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search-emit-images")
async def search_emit_images(req: SearchImagesRequest):
    """Search for EMIT L2A Reflectance hyperspectral images.
    
    Note: EMIT data may not be available in all regions or the collection 
    may not be accessible. Returns empty results gracefully on error.
    """
    try:
        print(f"EMIT SEARCH - Starting search with params: start={req.start_date}, end={req.end_date}, limit={req.limit}")
        
        aoi = bbox_to_geometry(req.bbox, req.geometry)
        limit = min(req.limit or 20, 100)
        
        try:
            col = get_emit_collection(aoi, req.start_date, req.end_date)
            fc = build_feature_collection_simple_emit(col, aoi)
            
            fc = fc.sort('datetime', False)
            fc = fc.sort('aoi_overlap', False)
            
            lst = fc.limit(limit).toList(limit)
            
            print(f"EMIT SEARCH - Fetching {limit} results from Earth Engine...")
            features = lst.getInfo()
            print(f"EMIT SEARCH - Got {len(features)} features")
        except Exception as collection_error:
            # EMIT collection may not exist or be accessible
            error_msg = str(collection_error)
            if "not found" in error_msg.lower() or "does not exist" in error_msg.lower() or "not have access" in error_msg.lower():
                print(f"EMIT SEARCH - Collection not accessible: {error_msg}")
                return {
                    "images": [],
                    "message": "EMIT hyperspectral data is not currently available. The NASA EMIT collection may not be accessible in your region or with your credentials."
                }
            raise

        results: List[dict] = []
        for f in features:
            props = f.get('properties', {})
            image_id = props.get('id')
            time_start = props.get('datetime')
            dt_iso = None
            if time_start is not None:
                try:
                    dt_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(time_start) / 1000))
                except Exception:
                    dt_iso = None
            band_count = props.get('band_count')
            overlap = props.get('aoi_overlap')
            
            # Get band metadata for the first image (sample)
            band_metadata = []
            if image_id:
                try:
                    img = resolve_emit_image(image_id)
                    band_metadata = get_emit_band_metadata(img)
                except Exception as e:
                    print(f"EMIT SEARCH - Could not get band metadata for {image_id}: {e}")
            
            info = {
                'id': image_id,
                'datetime': dt_iso,
                'cloud_cover': None,  # EMIT doesn't have cloud cover metadata
                'collection': 'EMIT',
                'band_count': int(band_count) if band_count is not None else None,
                'band_metadata': band_metadata[:50] if band_metadata else [],  # Limit to first 50 for response size
                'assets': {},
                'aoi_overlap': float(overlap) if overlap is not None else None,
            }
            results.append(info)

        print(f"EMIT SEARCH - Returning {len(results)} results")
        return {"images": results}
    except Exception as e:
        print(f"EMIT SEARCH ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty results instead of raising an error
        return {
            "images": [],
            "message": f"EMIT search failed: {str(e)}"
        }

