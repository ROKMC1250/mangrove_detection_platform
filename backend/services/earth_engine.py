"""
Earth Engine utilities for initialization, geometry handling, and collection management.
"""

import time
from typing import Dict, List, Optional, Tuple
from datetime import timedelta

import ee
from google.cloud import storage

from ..core.config import (
    SERVICE_ACCOUNT_EMAIL,
    SERVICE_ACCOUNT_KEY_PATH,
    GCS_BUCKET,
    validate_service_account,
)

# Global EE initialization flag
_EE_INITIALIZED = False


def init_earth_engine() -> bool:
    """Initialize Earth Engine with service account credentials."""
    global _EE_INITIALIZED
    if _EE_INITIALIZED:
        return True
    
    validate_service_account()
    credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT_EMAIL, SERVICE_ACCOUNT_KEY_PATH)
    ee.Initialize(credentials)
    _EE_INITIALIZED = True
    print("Earth Engine initialized successfully")
    return True


def bbox_to_geometry(bbox: List[float], geometry: Optional[Dict] = None) -> ee.Geometry:
    """Convert bbox or GeoJSON geometry to ee.Geometry."""
    if geometry:
        try:
            return ee.Geometry(geometry)
        except Exception as e:
            print(f"Error parsing geometry: {e}, falling back to bbox")
            
    if len(bbox) != 4:
        raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat]")
    min_lon, min_lat, max_lon, max_lat = bbox
    return ee.Geometry.Rectangle([min_lon, min_lat, max_lon, max_lat], proj=None, geodesic=False)


def resolve_item_to_image(item_id: str) -> ee.Image:
    """Resolve item ID to an Earth Engine Image (uses SR_HARMONIZED for Sentinel-2)."""
    try:
        if "/" in item_id or item_id.startswith("COPERNICUS/"):
            # If it's an old S2_SR path, convert to S2_SR_HARMONIZED
            if "COPERNICUS/S2_SR/" in item_id and "HARMONIZED" not in item_id:
                item_id = item_id.replace("COPERNICUS/S2_SR/", "COPERNICUS/S2_SR_HARMONIZED/")
            return ee.Image(item_id)
        else:
            return ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{item_id}")
    except Exception as e:
        raise ValueError(f"Invalid item_id: {e}")


def get_s2_collection(aoi: ee.Geometry, start: str, end: str, cc_max: int) -> ee.ImageCollection:
    """Get Sentinel-2 SR Harmonized collection filtered by AOI, date, and cloud cover."""
    return (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cc_max))
    )


def get_s1_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    """Get Sentinel-1 GRD collection filtered by AOI and date range."""
    return (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq('instrumentMode', 'IW'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
        .filter(ee.Filter.inList('orbitProperties_pass', ['ASCENDING', 'DESCENDING']))
    )


def _normalize_minmax(val, default):
    """Pass through scalar or list. None falls back to default scalar."""
    if val is None:
        return default
    if isinstance(val, (list, tuple)):
        return [float(v) for v in val]
    return float(val)


def get_visualization_params(
    bands: Optional[List[str]] = None,
    min_val=None,
    max_val=None,
) -> dict:
    """RGB visualization parameters for Sentinel-2 SR.

    `min_val` / `max_val` may be scalars (applied to all bands) or 3-element lists
    matching the bands order. Gamma is intentionally omitted — it lives in the
    client-side rendering layer (CSS/SVG filter) so users can adjust it without
    re-issuing a getMapId call.
    """
    return {
        "bands": list(bands) if bands else ["B4", "B3", "B2"],
        "min": _normalize_minmax(min_val, 0.0),
        "max": _normalize_minmax(max_val, 3000.0),
    }


def get_s1_visualization_params(
    bands: Optional[List[str]] = None,
    min_val=None,
    max_val=None,
) -> dict:
    """Visualization parameters for Sentinel-1 GRD (VV, VH bands in dB).

    `min_val` / `max_val` may be scalars or per-band lists.
    """
    return {
        "bands": list(bands) if bands else ["VV", "VH", "VV"],
        "min": _normalize_minmax(min_val, -25.0),
        "max": _normalize_minmax(max_val, 0.0),
    }


def compute_band_stretch_stats(
    image: ee.Image,
    aoi: ee.Geometry,
    bands: List[str],
    pct_low: float = 2.0,
    pct_high: float = 98.0,
    scale: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute per-band min, max, and percentile values inside AOI.

    Returns: { "B4": {"min": .., "max": .., "p_low": .., "p_high": ..}, ... }
    """
    if not bands:
        return {}

    pct_low = float(max(0.0, min(100.0, pct_low)))
    pct_high = float(max(0.0, min(100.0, pct_high)))
    if pct_high < pct_low:
        pct_low, pct_high = pct_high, pct_low

    img = image.select(bands)
    reducer = (
        ee.Reducer.minMax()
        .combine(ee.Reducer.percentile([pct_low, pct_high]), sharedInputs=True)
    )
    stats = img.reduceRegion(
        reducer=reducer,
        geometry=aoi,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4,
    ).getInfo() or {}

    out: Dict[str, Dict[str, float]] = {}
    p_low_key = f"p{int(pct_low) if pct_low.is_integer() else pct_low}"
    p_high_key = f"p{int(pct_high) if pct_high.is_integer() else pct_high}"
    for b in bands:
        def _g(suffix: str) -> Optional[float]:
            v = stats.get(f"{b}_{suffix}")
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        out[b] = {
            "min": _g("min"),
            "max": _g("max"),
            "p_low": _g(p_low_key),
            "p_high": _g(p_high_key),
        }
    return out


def get_model_names() -> dict:
    """Get mapping of model IDs to display names."""
    return {
        "model1": "Segmentation Mask",
        "model2": "NDVI",
        "model3": "NDMI",
        "model4": "MVI",
        "model5": "AlphaEarth Embedding",
        "cloud_mask": "Cloud Mask",
    }


def compute_aoi_overlap_ratio(img: ee.Image, aoi: ee.Geometry) -> Optional[float]:
    """Compute ratio of image footprint overlapping with AOI."""
    try:
        aoi_area = aoi.area(1)
        footprint = img.geometry(1)
        inter = footprint.intersection(aoi, 1)
        inter_area = inter.area(1)
        ratio = inter_area.divide(aoi_area).getInfo()
        if ratio is None:
            return None
        return max(0.0, min(1.0, float(ratio)))
    except Exception:
        return None


def calculate_aoi_cloud_coverage(img: ee.Image, aoi: ee.Geometry) -> ee.Number:
    """Calculate cloud coverage within AOI using s2cloudless."""
    s2_index = img.get('system:index')
    s2cloudless_col = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
    cloudless = s2cloudless_col.filter(ee.Filter.eq('system:index', s2_index)).first()
    
    def calculate_cloud_prob():
        cloud_prob = cloudless.select('probability')
        cloud_prob_aoi = cloud_prob.clip(aoi)
        stats = cloud_prob_aoi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi,
            scale=20,
            maxPixels=1e9,
            bestEffort=True,
            tileScale=4
        )
        mean_cloud = ee.Number(stats.get('probability'))
        return ee.Algorithms.If(
            mean_cloud,
            mean_cloud,
            img.get('CLOUDY_PIXEL_PERCENTAGE')
        )
    
    return ee.Algorithms.If(
        cloudless,
        ee.Number(calculate_cloud_prob()),
        ee.Number(img.get('CLOUDY_PIXEL_PERCENTAGE'))
    )


def build_feature_collection_simple(col: ee.ImageCollection, aoi: ee.Geometry) -> ee.FeatureCollection:
    """Lightweight feature collection builder using metadata cloud cover."""
    aoi_area = aoi.area(1)
    
    def _map(img):
        footprint = img.geometry(1)
        inter = footprint.intersection(aoi, 1)
        inter_area = inter.area(1)
        ratio = inter_area.divide(aoi_area)
        cloud_cover = img.get('CLOUDY_PIXEL_PERCENTAGE')
        
        return ee.Feature(None, {
            'id': img.id(),
            'datetime': img.get('system:time_start'),
            'cloud_cover': cloud_cover,
            'aoi_overlap': ratio
        })
    
    return ee.FeatureCollection(col.map(_map))


def build_feature_collection_simple_s1(col: ee.ImageCollection, aoi: ee.Geometry) -> ee.FeatureCollection:
    """Lightweight feature collection builder for Sentinel-1."""
    aoi_area = aoi.area(1)
    
    def _map(img):
        footprint = img.geometry(1)
        inter = footprint.intersection(aoi, 1)
        inter_area = inter.area(1)
        ratio = inter_area.divide(aoi_area)
        
        return ee.Feature(None, {
            'id': img.id(),
            'datetime': img.get('system:time_start'),
            'orbit': img.get('orbitProperties_pass'),
            'aoi_overlap': ratio
        })
    
    return ee.FeatureCollection(col.map(_map))


def create_best_image_composite(collection: ee.ImageCollection, aoi: Optional[ee.Geometry] = None) -> Tuple[ee.Image, dict]:
    """Select the image with lowest cloud coverage from collection."""
    bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
    best_image = collection.first()
    
    image_info = best_image.getInfo()
    image_date = image_info['properties'].get('system:time_start')
    
    if aoi is not None:
        try:
            cloud_cover = calculate_aoi_cloud_coverage(best_image, aoi).getInfo()
        except:
            cloud_cover = image_info['properties'].get('CLOUDY_PIXEL_PERCENTAGE', 'Unknown')
    else:
        cloud_cover = image_info['properties'].get('CLOUDY_PIXEL_PERCENTAGE', 'Unknown')
    
    if image_date:
        from datetime import datetime
        image_date_str = datetime.fromtimestamp(image_date / 1000).strftime('%Y-%m-%d')
    else:
        image_date_str = 'Unknown'
    
    result = best_image.select(bands).unmask(0)
    
    metadata = {
        'date': image_date_str,
        'cloud_coverage': cloud_cover,
        'timestamp': image_date
    }
    
    return result, metadata


def gcs_signed_url(gcs_uri: str, expire_seconds: int = 3600) -> Optional[str]:
    """Generate a signed URL for a GCS object."""
    if not gcs_uri.startswith("gs://"):
        return None
    if not GCS_BUCKET:
        return None
    try:
        client = storage.Client()
        bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        return blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expire_seconds),
            method="GET",
        )
    except Exception:
        return None


def image_properties_to_dict(img: ee.Image) -> dict:
    """Convert EE Image properties to dictionary."""
    try:
        image_id = img.id().getInfo()
    except Exception:
        image_id = None
    try:
        time_start = img.get('system:time_start').getInfo()
    except Exception:
        time_start = None
    try:
        cloud = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
    except Exception:
        cloud = None

    dt_iso = None
    if time_start is not None:
        try:
            dt_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(int(time_start) / 1000))
        except Exception:
            dt_iso = None

    return {
        "id": image_id,
        "datetime": dt_iso,
        "cloud_cover": float(cloud) if cloud is not None else None,
        "collection": "Sentinel-2",
        "assets": {},
    }

