"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class SearchImagesRequest(BaseModel):
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    start_date: str
    end_date: str
    cloud_cover_max: Optional[int] = 20
    limit: Optional[int] = 20


class GetTileUrlRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None


class ProcessImageRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    intensity_multiplier: Optional[float] = 1.0
    max_pixels: Optional[float] = None
    job_id: Optional[str] = None


class ChangeMonitoringRequest(BaseModel):
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    image_ids: List[str]
    job_id: Optional[str] = None
    model_id: Optional[str] = None


class CustomVisualizationRequest(BaseModel):
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    custom_visualization: Dict
    custom_name: str = "Custom Visualization"


class PixelValueRequest(BaseModel):
    image_id: str
    lat: float
    lng: float
    model_id: str


class ThresholdRequest(BaseModel):
    image_id: str
    model_id: str
    threshold: float
    colormap: Dict


class ThresholdRangeRequest(BaseModel):
    image_id: str
    model_id: str
    min_threshold: float
    max_threshold: float
    colormap: Dict


class ProcessSpectralImageRequest(BaseModel):
    item_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    spectral_method: str = "ndvi"


class S1TileRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: Optional[List[str]] = None
    min: Optional[float] = -25
    max: Optional[float] = 0


class S2TileRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: Optional[List[str]] = None
    min: Optional[float] = 0
    max: Optional[float] = 3000


class EmitTileRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: Optional[List[str]] = None
    min: Optional[float] = 0
    max: Optional[float] = 10000


class DownloadImageRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: Optional[List[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    as_visualization: Optional[bool] = False


# =============================================================================
# Target Detection Schemas
# =============================================================================

class TargetPoint(BaseModel):
    lat: float
    lng: float


class TargetDetectionRequest(BaseModel):
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    target_points: List[TargetPoint] = Field(..., description="List of target point coordinates")
    algorithm: str = Field(default="SAM", description="Detection algorithm: SAM, ACE, RXD, CEM")
    threshold_percentile: Optional[float] = Field(default=95.0, description="Percentile for auto-threshold")
    auto_threshold: Optional[bool] = Field(default=True, description="Automatically find optimal threshold")
    selected_bands: Optional[List[int]] = Field(default=None, description="List of band indices to use (0-based). None means all bands.")


class TargetDetectionThresholdRequest(BaseModel):
    detection_id: str
    min_threshold: float
    max_threshold: float
    bbox: List[float]


class TargetSpectrumRequest(BaseModel):
    image_id: str
    bbox: List[float]
    lat: float
    lng: float


class ProcessEmitImageRequest(BaseModel):
    item_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    selected_bands: List[str] = Field(..., description="List of band names to process")
    visualization_type: str = Field(default="rgb", description="Type: 'rgb' or 'index'")
    rgb_bands: Optional[List[str]] = Field(default=None, description="For RGB: [R, G, B] band names")
    index_bands: Optional[List[str]] = Field(default=None, description="For index: [A, B] band names")
    colormap: Optional[str] = Field(default="RdYlGn", description="Colormap for index visualization")
    job_id: Optional[str] = None

