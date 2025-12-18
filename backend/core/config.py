"""
Configuration module - Central location for all settings, constants, and paths.
"""

import os
import sys
import requests
from requests.adapters import HTTPAdapter

# ===== Path Configuration =====
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, os.pardir))
FRONTEND_DIR = os.path.join(PROJECT_ROOT, "frontend")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
STATIC_MOUNT = "/static"

# Model paths
MODEL_ROOT = os.environ.get('MODEL_ROOT', '/home/hjh1037/Mangrove_segmentation')
if MODEL_ROOT and MODEL_ROOT not in sys.path:
    sys.path.append(MODEL_ROOT)

# Ensure outputs directory exists
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ===== Earth Engine Configuration =====
SERVICE_ACCOUNT_EMAIL = os.environ.get("EE_SERVICE_ACCOUNT", "hjh1037@gmail.com")
SERVICE_ACCOUNT_KEY_PATH = os.environ.get(
    "EE_SERVICE_ACCOUNT_KEY", os.path.join(BASE_DIR, "ee-service-account-key.json")
)

# ===== GCS Configuration =====
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")
GCS_BUCKET_REGION = os.environ.get("GCS_BUCKET_REGION", "")
GCS_EXPORT_PREFIX = "mangrove_analysis"

# ===== Download Constants =====
GEE_DL_MAX_BYTES = 40 * 1024 * 1024  # 40 MiB hard limit
GEE_DL_SAFETY = 0.90  # Safety multiplier
MAX_PIXELS_DEFAULT = 1e13  # Default max pixels (10 trillion)

# Bands to export for Sentinel-2 (10m + 20m resolution bands)
# B2=Blue, B3=Green, B4=Red, B5-B7=RedEdge, B8=NIR, B8A=NIR narrow, B11-B12=SWIR
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
SELECTED_EXPORT_BANDS = S2_BANDS

# Band index mapping (1-indexed for rasterio)
S2_BAND_MAPPING = {
    'B2': 1, 'B3': 2, 'B4': 3, 'B5': 4, 'B6': 5,
    'B7': 6, 'B8': 7, 'B8A': 8, 'B11': 9, 'B12': 10
}

# ===== HTTP Session Pool =====
HTTP_POOL_SIZE = int(os.environ.get('HTTP_POOL', '32'))
HTTP_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=HTTP_POOL_SIZE, pool_maxsize=HTTP_POOL_SIZE, max_retries=0)
HTTP_SESSION.mount('https://', _adapter)
HTTP_SESSION.mount('http://', _adapter)

# ===== Model Configuration =====
MODEL1_LOG_DIR = os.environ.get('MODEL1_LOG_DIR', '')
MODEL1_GPUS = os.environ.get('MODEL1_GPUS', '')
MODEL1_PATCH_SIZE = int(os.environ.get('MODEL1_PATCH_SIZE', '256'))
MODEL1_OVERLAP = float(os.environ.get('MODEL1_OVERLAP', '0.25'))

# ===== Parallel Processing =====
PARALLEL_TILE_WORKERS = int(os.environ.get('PARALLEL_TILE_WORKERS', '6'))

# ===== Epsilon for safe division =====
SAFE_DIV_EPS = 1e-6


def validate_service_account():
    """Validate that the service account key exists."""
    if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        raise RuntimeError(
            f"Service account key not found at {SERVICE_ACCOUNT_KEY_PATH}. "
            f"Please place your JSON key at backend/ee-service-account-key.json or set EE_SERVICE_ACCOUNT_KEY."
        )
    return True

