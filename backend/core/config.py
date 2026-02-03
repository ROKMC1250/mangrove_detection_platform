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

# Bands to export for Sentinel-2 (all 12 bands for 13-channel model input)
# B1=Coastal, B2=Blue, B3=Green, B4=Red, B5-B7=RedEdge, B8=NIR, B8A=NIR narrow, B9=WaterVapor, B11-B12=SWIR
# Note: B1 and B9 are 60m resolution but will be resampled to 10m by GEE
# Note: B10 (Cirrus) is not available in Sentinel-2 SR products
# Total: 12 bands + 1 mask = 13 channels (matching model in_channels=13)
S2_BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
SELECTED_EXPORT_BANDS = S2_BANDS

# Band index mapping (1-indexed for rasterio)
S2_BAND_MAPPING = {
    'B1': 1, 'B2': 2, 'B3': 3, 'B4': 4, 'B5': 5, 'B6': 6,
    'B7': 7, 'B8': 8, 'B8A': 9, 'B9': 10, 'B11': 11, 'B12': 12
}

# ===== HTTP Session Pool =====
HTTP_POOL_SIZE = int(os.environ.get('HTTP_POOL', '32'))
HTTP_SESSION = requests.Session()
_adapter = HTTPAdapter(pool_connections=HTTP_POOL_SIZE, pool_maxsize=HTTP_POOL_SIZE, max_retries=0)
HTTP_SESSION.mount('https://', _adapter)
HTTP_SESSION.mount('http://', _adapter)

# ===== Model Configuration =====
# Load from model_config.yaml file for easy configuration
MODEL_CONFIG_PATH = os.path.join(BASE_DIR, 'model_config.yaml')

def _load_model_config():
    """Load model configuration from YAML file."""
    default_config = {
        'model_dir': '',
        'checkpoint': 'last.pt',
        'gpus': '',
        'patch_size': 256,
        'overlap': 0.5,
        'use_tta': False,
        'default_model': {
            'name': 'Segformer',
            'encoder_name': 'mit_b2',
            'in_channels': 13,
            'classes': 1
        }
    }
    
    if os.path.exists(MODEL_CONFIG_PATH):
        try:
            import yaml
            with open(MODEL_CONFIG_PATH, 'r', encoding='utf-8') as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    default_config.update(loaded)
            print(f"✅ Model config loaded from: {MODEL_CONFIG_PATH}")
        except Exception as e:
            print(f"⚠️  Failed to load model_config.yaml: {e}, using defaults")
    else:
        print(f"⚠️  model_config.yaml not found at {MODEL_CONFIG_PATH}, using defaults")
    
    return default_config

_MODEL_CONFIG = _load_model_config()

# Export model settings
MODEL_DIR = _MODEL_CONFIG.get('model_dir', '')
MODEL_CHECKPOINT = _MODEL_CONFIG.get('checkpoint', 'last.pt')
MODEL1_GPUS = str(_MODEL_CONFIG.get('gpus', ''))
MODEL1_PATCH_SIZE = int(_MODEL_CONFIG.get('patch_size', 256))
MODEL1_OVERLAP = float(_MODEL_CONFIG.get('overlap', 0.5))
MODEL1_USE_TTA = bool(_MODEL_CONFIG.get('use_tta', False))
DEFAULT_MODEL_PARAMS = _MODEL_CONFIG.get('default_model', {})

# Legacy support
MODEL1_LOG_DIR = MODEL_DIR

# ===== Parallel Processing =====
PARALLEL_TILE_WORKERS = int(os.environ.get('PARALLEL_TILE_WORKERS', '6'))

# ===== Epsilon for safe division =====
SAFE_DIV_EPS = 1e-6

# ===== EMIT Hyperspectral Configuration =====
EMIT_COLLECTION = "NASA/EMIT/L2A_RFL"  # EMIT L2A Reflectance collection
EMIT_BAND_COUNT = 285  # Approximate number of bands in EMIT
EMIT_MAX_BANDS_TO_DOWNLOAD = 50  # Limit for band selection to prevent huge downloads


def validate_service_account():
    """Validate that the service account key exists."""
    if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        raise RuntimeError(
            f"Service account key not found at {SERVICE_ACCOUNT_KEY_PATH}. "
            f"Please place your JSON key at backend/ee-service-account-key.json or set EE_SERVICE_ACCOUNT_KEY."
        )
    return True

