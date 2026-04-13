"""
SAM2 (Segment Anything Model 2) service module.
Handles model loading, image encoding, and mask prediction.
"""

import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.config import PROJECT_ROOT

# Try to import SAM2 dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    build_sam2 = None
    SAM2ImagePredictor = None


# Global model state
_SAM2_PREDICTOR = None
_SAM2_DEVICE = None
_SAM2_READY = False
_SAM2_ERROR = None
_SAM2_MODEL_NAME = None
_SAM2_LOCK = threading.Lock()

# Image embedding cache: cache_key -> True (predictor holds the embedding internally)
_ENCODED_IMAGES: Dict[str, bool] = {}
_CURRENT_ENCODED_KEY: Optional[str] = None


# Model configuration mapping
SAM2_MODEL_CONFIGS = {
    "tiny": ("sam2_hiera_t.yaml", "sam2_hiera_tiny.pt"),
    "small": ("sam2_hiera_s.yaml", "sam2_hiera_small.pt"),
    "base_plus": ("sam2_hiera_b+.yaml", "sam2_hiera_base_plus.pt"),
    "large": ("sam2_hiera_l.yaml", "sam2_hiera_large.pt"),
}


def init_sam2() -> bool:
    """Initialize SAM2 model.

    Configuration via environment variables:
    - SAM2_MODEL_SIZE: Model size ('tiny', 'small', 'base_plus', 'large'). Default: 'small'
    - SAM2_CHECKPOINT_DIR: Directory containing SAM2 checkpoints. Default: PROJECT_ROOT/models/sam2/

    Returns:
        True if model loaded successfully, False otherwise
    """
    global _SAM2_PREDICTOR, _SAM2_DEVICE, _SAM2_READY, _SAM2_ERROR, _SAM2_MODEL_NAME

    if _SAM2_READY:
        return True

    with _SAM2_LOCK:
        if _SAM2_READY:
            return True

        if not TORCH_AVAILABLE:
            _SAM2_ERROR = "PyTorch not available"
            print(f"⚠️  SAM2 - {_SAM2_ERROR}")
            return False

        if not SAM2_AVAILABLE:
            _SAM2_ERROR = "sam2 package not installed. Install with: pip install sam2"
            print(f"⚠️  SAM2 - {_SAM2_ERROR}")
            return False

        model_size = os.environ.get("SAM2_MODEL_SIZE", "small").lower()
        checkpoint_dir = os.environ.get(
            "SAM2_CHECKPOINT_DIR",
            os.path.join(PROJECT_ROOT, "repo", "sam2")
        )

        if model_size not in SAM2_MODEL_CONFIGS:
            _SAM2_ERROR = f"Unknown model size: {model_size}. Options: {list(SAM2_MODEL_CONFIGS.keys())}"
            print(f"⚠️  SAM2 - {_SAM2_ERROR}")
            return False

        config_name, checkpoint_name = SAM2_MODEL_CONFIGS[model_size]
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

        if not os.path.exists(checkpoint_path):
            _SAM2_ERROR = (
                f"SAM2 checkpoint not found at {checkpoint_path}. "
                f"Download from https://github.com/facebookresearch/segment-anything-2"
            )
            print(f"⚠️  SAM2 - {_SAM2_ERROR}")
            return False

        try:
            # Determine device
            if torch.cuda.is_available():
                _SAM2_DEVICE = torch.device("cuda")
            else:
                _SAM2_DEVICE = torch.device("cpu")

            print(f"🔄 SAM2 - Loading {model_size} model on {_SAM2_DEVICE}...")

            sam2_model = build_sam2(config_name, checkpoint_path, device=_SAM2_DEVICE)
            _SAM2_PREDICTOR = SAM2ImagePredictor(sam2_model)
            _SAM2_MODEL_NAME = model_size
            _SAM2_READY = True

            print(f"✅ SAM2 - Model loaded successfully ({model_size} on {_SAM2_DEVICE})")
            return True

        except Exception as e:
            _SAM2_ERROR = f"Failed to load SAM2 model: {str(e)}"
            print(f"❌ SAM2 - {_SAM2_ERROR}")
            return False


def is_sam2_ready() -> bool:
    """Check if SAM2 model is loaded and ready."""
    return _SAM2_READY


def get_sam2_status() -> dict:
    """Get SAM2 model status information."""
    return {
        "ready": _SAM2_READY,
        "model": _SAM2_MODEL_NAME,
        "device": str(_SAM2_DEVICE) if _SAM2_DEVICE else None,
        "error": _SAM2_ERROR,
    }


def encode_image(cache_key: str, rgb_image: np.ndarray) -> str:
    """Encode an image using SAM2's image encoder.

    Args:
        cache_key: Unique key for caching the embedding
        rgb_image: RGB uint8 image array (H, W, 3)

    Returns:
        The cache_key for reference

    Raises:
        RuntimeError: If SAM2 is not ready
    """
    global _CURRENT_ENCODED_KEY

    if not _SAM2_READY or _SAM2_PREDICTOR is None:
        raise RuntimeError("SAM2 model not loaded")

    if _CURRENT_ENCODED_KEY == cache_key:
        return cache_key

    with _SAM2_LOCK:
        if _CURRENT_ENCODED_KEY == cache_key:
            return cache_key

        print(f"🔄 SAM2 - Encoding image ({rgb_image.shape})...")
        _SAM2_PREDICTOR.set_image(rgb_image)
        _CURRENT_ENCODED_KEY = cache_key
        print(f"✅ SAM2 - Image encoded (key: {cache_key[:20]}...)")

    return cache_key


def predict_mask(
    cache_key: str,
    positive_points: List[Tuple[int, int]],
    negative_points: Optional[List[Tuple[int, int]]] = None,
) -> Tuple[np.ndarray, float]:
    """Predict a segmentation mask using point prompts.

    Args:
        cache_key: Must match a previously encoded image
        positive_points: List of (x, y) pixel coordinates for positive prompts
        negative_points: Optional list of (x, y) pixel coordinates for negative prompts

    Returns:
        Tuple of (mask_array, score):
            - mask_array: Boolean mask (H, W)
            - score: Confidence score for the mask

    Raises:
        RuntimeError: If SAM2 is not ready or image not encoded
    """
    if not _SAM2_READY or _SAM2_PREDICTOR is None:
        raise RuntimeError("SAM2 model not loaded")

    if _CURRENT_ENCODED_KEY != cache_key:
        raise RuntimeError(f"Image not encoded. Current: {_CURRENT_ENCODED_KEY}, requested: {cache_key}")

    # Build point coordinates and labels
    all_points = list(positive_points)
    all_labels = [1] * len(positive_points)

    if negative_points:
        all_points.extend(negative_points)
        all_labels.extend([0] * len(negative_points))

    point_coords = np.array(all_points, dtype=np.float32)
    point_labels = np.array(all_labels, dtype=np.int32)

    print(f"🔄 SAM2 - Predicting mask with {len(positive_points)} positive, {len(negative_points or [])} negative points...")

    masks, scores, logits = _SAM2_PREDICTOR.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Select the mask with the highest confidence score
    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx].astype(bool)  # Ensure boolean (H, W)
    best_score = float(scores[best_idx])

    print(f"✅ SAM2 - Mask predicted (score: {best_score:.3f}, pixels: {best_mask.sum()})")

    return best_mask, best_score


def clear_encoding():
    """Clear the current image encoding to free memory."""
    global _CURRENT_ENCODED_KEY

    with _SAM2_LOCK:
        if _SAM2_PREDICTOR is not None:
            _SAM2_PREDICTOR.reset_predictor()
        _CURRENT_ENCODED_KEY = None
        print("🗑 SAM2 - Image encoding cleared")
