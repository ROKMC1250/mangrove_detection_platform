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
    build_feature_collection_simple,
    build_feature_collection_simple_s1,
    require_earth_engine,
)

router = APIRouter(prefix="/api", tags=["search"])


@router.post("/search-images")
def search_images(req: SearchImagesRequest):
    """Search for Sentinel-2 SR images."""
    require_earth_engine()  # outside the try: a 503 must not become a 500
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
def search_s1_images(req: SearchImagesRequest):
    """Search for Sentinel-1 GRD images."""
    require_earth_engine()  # outside the try: a 503 must not become a 500
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



