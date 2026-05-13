"""
Pydantic schemas for API request/response models.
"""

from typing import List, Optional, Dict, Union
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
    # Optional symbology overrides. If all None, server defaults are used
    # (backward compatible with callers that don't supply them). `min`/`max` may be a
    # scalar (applied to all RGB bands) or a 3-element list (per-band, R/G/B order).
    bands: Optional[List[str]] = None
    min: Optional[Union[List[float], float]] = None
    max: Optional[Union[List[float], float]] = None
    stretch_mode: Optional[str] = Field(default=None, description="'minmax' | 'percentile'")
    pct_low: Optional[float] = None
    pct_high: Optional[float] = None


class ProcessImageRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    intensity_multiplier: Optional[float] = 1.0
    max_pixels: Optional[float] = None
    job_id: Optional[str] = None


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


class S1TileRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: Optional[List[str]] = None
    min: Optional[Union[List[float], float]] = -25
    max: Optional[Union[List[float], float]] = 0
    stretch_mode: Optional[str] = Field(default=None, description="'minmax' | 'percentile'")
    pct_low: Optional[float] = None
    pct_high: Optional[float] = None


class S2TileRequest(BaseModel):
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: Optional[List[str]] = None
    min: Optional[Union[List[float], float]] = 0
    max: Optional[Union[List[float], float]] = 3000
    stretch_mode: Optional[str] = Field(default=None, description="'minmax' | 'percentile'")
    pct_low: Optional[float] = None
    pct_high: Optional[float] = None


class StretchStatsRequest(BaseModel):
    """Request per-band min/max/percentile statistics within an AOI.

    `sensor`: 's2' | 's1' selects the EE asset prefix used to resolve item_id.
    """
    item_id: str
    bbox: List[float]
    geometry: Optional[Dict] = None
    bands: List[str]
    sensor: str = Field(default="s2", description="'s2' | 's1'")
    pct_low: Optional[float] = 2.0
    pct_high: Optional[float] = 98.0


class StretchStatsResponse(BaseModel):
    bands: Dict[str, Dict[str, Optional[float]]]
    pct_low: float
    pct_high: float


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
    target_points: List[TargetPoint] = Field(..., description="List of positive target point coordinates")
    negative_points: Optional[List[TargetPoint]] = Field(default=None, description="List of negative (non-target) point coordinates")
    algorithm: str = Field(default="SAM", description="Detection algorithm: SAM, ACE, RXD, CEM, MF, MLP_AMF, MLP_ACE")
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


# =============================================================================
# Mangrove Segmentation Schemas
# =============================================================================

class MangroveSegmentationRequest(BaseModel):
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    use_tta: Optional[bool] = Field(default=False, description="Use Test Time Augmentation (more accurate but 4x slower)")


class MangroveSegmentationThresholdRequest(BaseModel):
    segmentation_id: str
    min_threshold: float
    max_threshold: float
    bbox: List[float]


# =============================================================================
# Change Detection Schemas
# =============================================================================

class ChangeDetectionGroup(BaseModel):
    """A composed binary mask: items[0] seeded with operators[k-1] joining items[k].
    Operators: 'inc' (∩), 'exc' (−), 'add' (∪). Mirrors the layer panel's group model."""
    items: List[str] = Field(..., description="Cache ids of the binary-mask analyses to compose, in order")
    operators: List[str] = Field(default_factory=list, description="One operator per join — length = len(items) - 1")


class ChangeDetectionRequest(BaseModel):
    result_a_id: Optional[str] = Field(default=None, description="Single-analysis id for slot A")
    result_b_id: Optional[str] = Field(default=None, description="Single-analysis id for slot B")
    result_a_group: Optional[ChangeDetectionGroup] = Field(default=None, description="Composite mask (from a layer-panel group) for slot A; mutually exclusive with result_a_id")
    result_b_group: Optional[ChangeDetectionGroup] = Field(default=None, description="Composite mask (from a layer-panel group) for slot B; mutually exclusive with result_b_id")
    bbox: Optional[List[float]] = Field(default=None, description="Fallback bbox used for AOI warp when cached entries lack one (cloud mode only)")


class QueuedJobResponse(BaseModel):
    job_id: str
    status: str = Field(..., description="queued | running | completed | error | cancelled")
    queue_position: Optional[int] = Field(default=None, description="1-based position; 0 if running")
    ahead_count: Optional[int] = Field(default=None, description="Jobs ahead (waiting + running)")
    estimated_wait_seconds: Optional[int] = None
    message: Optional[str] = None


class ComputeSpectralIndexRequest(BaseModel):
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None
    index_type: str = Field(..., description="Index type: ndvi, mvi, ndmi, ndwi, savi, evi, custom")
    band_a: Optional[str] = Field(default=None, description="Band A for custom index")
    band_b: Optional[str] = Field(default=None, description="Band B for custom index")
    colormap: Optional[str] = Field(default=None, description="Colormap name override")


# =============================================================================
# SAM3 Segmentation Schemas
# =============================================================================

class SAM3EncodeRequest(BaseModel):
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    geometry: Optional[Dict] = None


class SAM3PointPredictRequest(BaseModel):
    """Cloud (GEE) single-instance segmentation from point prompts (SAM2-style)."""
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    positive_points: List[TargetPoint] = Field(..., description="Positive point prompts (lat/lng)")
    negative_points: Optional[List[TargetPoint]] = Field(default=None, description="Negative point prompts (lat/lng)")
    geometry: Optional[Dict] = None


class SAM3TextPredictRequest(BaseModel):
    """Cloud (GEE) Promptable Concept Segmentation from a text noun phrase."""
    image_id: str
    bbox: List[float] = Field(..., description="[min_lon, min_lat, max_lon, max_lat]")
    prompt: str = Field(..., description="Open-vocabulary noun phrase, e.g. 'mangrove'")
    geometry: Optional[Dict] = None
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)


class SAM3SaveMaskRequest(BaseModel):
    mask_id: str


class PixelPoint(BaseModel):
    row: int
    col: int


class UploadedSAM3PointPredictRequest(BaseModel):
    """Local (uploaded GeoTIFF) single-instance segmentation from point prompts."""
    upload_id: str
    positive_points: List[PixelPoint] = Field(..., description="Positive points in pixel coords (row/col)")
    negative_points: Optional[List[PixelPoint]] = Field(default=None, description="Negative points in pixel coords")
    rgb_bands: Optional[List[int]] = Field(default=None, description="RGB band indices (1-based). e.g. [3,2,1]")


class UploadedSAM3TextPredictRequest(BaseModel):
    """Local (uploaded GeoTIFF) PCS from a text noun phrase."""
    upload_id: str
    prompt: str = Field(..., description="Open-vocabulary noun phrase, e.g. 'mangrove'")
    rgb_bands: Optional[List[int]] = Field(default=None, description="RGB band indices (1-based). e.g. [3,2,1]")
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)



