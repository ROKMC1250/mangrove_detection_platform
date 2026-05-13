"""
Services module containing business logic for Earth Engine, downloads, analysis, and visualization.
"""

from .earth_engine import (
    init_earth_engine,
    bbox_to_geometry,
    resolve_item_to_image,
    get_s2_collection,
    get_s1_collection,
    get_visualization_params,
    get_s1_visualization_params,
    compute_band_stretch_stats,
    compute_aoi_overlap_ratio,
    calculate_aoi_cloud_coverage,
    build_feature_collection_simple,
    build_feature_collection_simple_s1,
    create_best_image_composite,
)

from .downloader import (
    download_ee_image,
    estimate_pixels_from_bounds,
    download_tile_get_url,
)

from .spectral_analysis import (
    calculate_spectral_index,
    safe_divide,
    calculate_mangrove_area_km2,
)

from .visualization import (
    create_index_visualization,
    save_rgba_overlay_png,
    save_rgba_overlay_png_with_transparency,
    save_float_geotiff,
    warp_rgb_and_mask_to_aoi,
    to_png_bytes,
    stretch_uint8,
)

from .model_inference import (
    init_model1,
    is_model1_ready,
    run_model1_inference,
    get_model1_status,
)

