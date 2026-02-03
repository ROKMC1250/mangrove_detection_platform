"""
Model inference module for deep learning segmentation.
Optimized for Segformer model with Gaussian-weighted blending.
"""

import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from PIL import Image
import io as pyio

from ..core.config import (
    MODEL_ROOT,
    MODEL_DIR,
    MODEL1_LOG_DIR,
    MODEL1_GPUS,
    MODEL1_PATCH_SIZE,
    MODEL1_OVERLAP,
    MODEL1_USE_TTA,
    MODEL_CHECKPOINT,
    DEFAULT_MODEL_PARAMS,
    PROJECT_ROOT,
)
from ..utils.cache import cache_index_data

# Try to import torch and model dependencies
try:
    import torch
    import yaml
    from model.smp_models import create_model
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    yaml = None
    create_model = None


# Global model cache
_MODEL1 = None
_MODEL1_DEVICE = None
_MODEL1_CFG = None
_MODEL1_READY = False
_MODEL1_ERROR = None
_MODEL1_LOCK = threading.Lock()

# ===== Segmentation Model Default Configuration =====
# Default model parameters loaded from model_config.yaml (via config.py)
# Modify backend/model_config.yaml to change these settings
def _get_default_model_config():
    """Build default model config from model_config.yaml settings."""
    return {
        'model': {
            'name': DEFAULT_MODEL_PARAMS.get('name', 'Segformer'),
            'args': {
                'encoder_name': DEFAULT_MODEL_PARAMS.get('encoder_name', 'mit_b2'),
                'in_channels': DEFAULT_MODEL_PARAMS.get('in_channels', 13),
                'classes': DEFAULT_MODEL_PARAMS.get('classes', 1),
                'encoder_weights': DEFAULT_MODEL_PARAMS.get('encoder_weights', None)
            }
        }
    }


def _get_device_from_env(gpus_str: str):
    """Get the appropriate device based on environment settings."""
    if not TORCH_AVAILABLE or torch is None:
        return 'cpu'
    if torch.cuda.is_available() and gpus_str:
        try:
            gpu_ids = [int(g) for g in gpus_str.split(',') if g.strip().isdigit()]
            if gpu_ids:
                return torch.device(f"cuda:{gpu_ids[0]}")
        except Exception:
            pass
    return torch.device('cpu')


def _create_gaussian_window(size: int, sigma_ratio: float = 0.25) -> np.ndarray:
    """
    Create a 2D Gaussian window for smooth patch blending.
    
    This eliminates visible seams at patch boundaries by giving higher weight
    to the center of each patch and lower weight to edges.
    
    Args:
        size: Window size (assumes square)
        sigma_ratio: Ratio of sigma to size (smaller = more peaked center)
        
    Returns:
        2D Gaussian weight array normalized to [0, 1]
    """
    sigma = size * sigma_ratio
    
    # Create 1D Gaussian
    x = np.arange(size) - (size - 1) / 2
    gauss_1d = np.exp(-x**2 / (2 * sigma**2))
    
    # Create 2D Gaussian via outer product
    gauss_2d = np.outer(gauss_1d, gauss_1d)
    
    # Normalize to [0, 1] with minimum value to avoid zero weights at edges
    gauss_2d = gauss_2d / gauss_2d.max()
    gauss_2d = np.clip(gauss_2d, 0.1, 1.0)
    
    return gauss_2d.astype(np.float32)


def init_model1() -> bool:
    """Initialize segmentation model for mangrove inference.
    
    Model path priority:
    1. MODEL_DIR environment variable (recommended)
    2. MODEL1_LOG_DIR environment variable (legacy)
    
    Returns:
        True if model loaded successfully, False otherwise
    """
    global _MODEL1, _MODEL1_DEVICE, _MODEL1_CFG, _MODEL1_READY, _MODEL1_ERROR
    
    if _MODEL1_READY:
        return True
    
    with _MODEL1_LOCK:
        if _MODEL1_READY:
            return True
        
        if not TORCH_AVAILABLE:
            _MODEL1_ERROR = "PyTorch not available - segmentation model will be disabled"
            print(f"⚠️  MODEL - {_MODEL1_ERROR}")
            print("   Note: Other features (NDVI, NDMI, MVI, etc.) will still work.")
            return False
        
        # Determine model directory (MODEL_DIR takes priority, then MODEL1_LOG_DIR)
        log_dir = MODEL_DIR if MODEL_DIR else MODEL1_LOG_DIR
        if not log_dir:
            _MODEL1_ERROR = "No model directory configured - set MODEL_DIR environment variable"
            print(f"⚠️  MODEL - {_MODEL1_ERROR}")
            print("   Other features (NDVI, NDMI, MVI, etc.) will still work.")
            return False
        
        # Load config or use default config from model_config.yaml
        cfg_path = os.path.join(log_dir, 'config.yaml')
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r') as f:
                    cfg = yaml.safe_load(f)
                print(f"MODEL - loaded config from {cfg_path}")
            except Exception as e:
                print(f"⚠️  MODEL - config load error, using defaults from model_config.yaml: {e}")
                cfg = _get_default_model_config()
        else:
            print(f"MODEL - config.yaml not found in {log_dir}, using model_config.yaml defaults")
            cfg = _get_default_model_config()
        
        try:
            # Extract model parameters
            model_name = cfg['model']['name']
            model_args = cfg['model']['args']
            encoder_name = model_args.get('encoder_name', 'mit_b2')
            in_channels = int(model_args.get('in_channels', 13))
            out_classes = int(model_args.get('classes', 1))
            encoder_weights = model_args.get('encoder_weights', None)
            
            device = _get_device_from_env(MODEL1_GPUS)
            
            print(f"MODEL - Creating model: {model_name}")
            print(f"  encoder: {encoder_name}, in_channels: {in_channels}, classes: {out_classes}")
            
            model = create_model(
                model_name=model_name, 
                encoder_name=encoder_name,
                in_channels=in_channels, 
                classes=out_classes, 
                encoder_weights=encoder_weights
            )
            
            # Load checkpoint
            ckpt = os.path.join(log_dir, 'weights', MODEL_CHECKPOINT)
            if not os.path.exists(ckpt):
                # Try alternative path without 'weights' subdirectory
                ckpt_alt = os.path.join(log_dir, MODEL_CHECKPOINT)
                if os.path.exists(ckpt_alt):
                    ckpt = ckpt_alt
                else:
                    _MODEL1_ERROR = f"Checkpoint not found at {ckpt} - segmentation model will be disabled"
                    print(f"⚠️  MODEL - {_MODEL1_ERROR}")
                    print("   Note: Other features (NDVI, NDMI, MVI, etc.) will still work.")
                    return False
            
            print(f"MODEL - Loading checkpoint: {ckpt}")
            state = torch.load(ckpt, map_location=device, weights_only=False)
            model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
            model.to(device)
            model.eval()
            
            _MODEL1 = model
            _MODEL1_DEVICE = device
            _MODEL1_CFG = cfg
            _MODEL1_READY = True
            _MODEL1_ERROR = None
            
            print(f"✅ MODEL - {model_name} loaded successfully on {device}")
            return True
            
        except Exception as e:
            _MODEL1_ERROR = f"{str(e)} - segmentation model will be disabled"
            print(f"⚠️  MODEL - load error: {e}")
            print("   Note: Other features (NDVI, NDMI, MVI, etc.) will still work.")
            import traceback
            traceback.print_exc()
            return False


def is_model1_ready() -> bool:
    """Check if Model1 is ready for inference."""
    return _MODEL1_READY


def get_model1_status() -> Dict:
    """Get current status of Model1."""
    return {
        'loaded': bool(_MODEL1_READY),
        'device': str(_MODEL1_DEVICE) if _MODEL1_DEVICE is not None else None,
        'error': _MODEL1_ERROR,
    }


def _predict_large_image_mask(model, image_path: str, patch_size: int, 
                               overlap: float, device, 
                               expected_channels: int = 13,
                               use_tta: bool = False) -> Tuple[np.ndarray, dict]:
    """Run inference on a large image using sliding window with Gaussian blending.
    
    Uses Gaussian weighted averaging for smooth transitions between patches,
    which eliminates visible seams at patch boundaries.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        patch_size: Size of patches for inference
        overlap: Overlap ratio between patches
        device: PyTorch device
        expected_channels: Number of channels the model expects (default: 13 for Segformer)
        use_tta: Use Test Time Augmentation (flip averaging) for better results
        
    Returns:
        Tuple of (mask array, rasterio profile)
    """
    with rasterio.open(image_path) as src:
        # Read all available bands
        n_bands = src.count
        image = src.read()  # (C, H, W)
        profile = src.profile
    
    _, height, width = image.shape
    
    # Handle channel mismatch - pad with zeros if needed, or truncate
    if n_bands < expected_channels:
        padding = np.zeros((expected_channels - n_bands, height, width), dtype=image.dtype)
        image = np.concatenate([image, padding], axis=0)
        print(f"  Padded image from {n_bands} to {expected_channels} channels")
    elif n_bands > expected_channels:
        image = image[:expected_channels]
        print(f"  Using first {expected_channels} of {n_bands} available channels")
    
    # Normalize for Sentinel-2 (values typically 0-10000)
    img = image.astype(np.float32) / 10000.0
    
    stride = max(1, int(patch_size * (1.0 - overlap)))
    
    # Calculate necessary padding
    pad_h = (stride - (height - patch_size) % stride) % stride
    pad_w = (stride - (width - patch_size) % stride) % stride
    
    # Use reflect padding for better edge handling
    padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='reflect')
    ph, pw = padded.shape[1], padded.shape[2]
    
    # Create Gaussian weight window for smooth blending
    weight_window = _create_gaussian_window(patch_size, sigma_ratio=0.3)
    weight_tensor = torch.from_numpy(weight_window).to(device).unsqueeze(0).unsqueeze(0)
    
    # Create placeholders for weighted averaging
    pred_map = torch.zeros((1, 1, ph, pw), device=device, dtype=torch.float32)
    weight_map = torch.zeros((1, 1, ph, pw), device=device, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch = padded[:, y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.from_numpy(patch).float().to(device).unsqueeze(0)
                
                # Basic prediction
                out = model(patch_tensor)
                prediction = torch.sigmoid(out)
                
                # Optional: Test Time Augmentation (horizontal + vertical flip)
                if use_tta:
                    # Horizontal flip
                    patch_hflip = torch.flip(patch_tensor, dims=[3])
                    pred_hflip = torch.sigmoid(model(patch_hflip))
                    pred_hflip = torch.flip(pred_hflip, dims=[3])
                    
                    # Vertical flip
                    patch_vflip = torch.flip(patch_tensor, dims=[2])
                    pred_vflip = torch.sigmoid(model(patch_vflip))
                    pred_vflip = torch.flip(pred_vflip, dims=[2])
                    
                    # Both flips
                    patch_hvflip = torch.flip(patch_tensor, dims=[2, 3])
                    pred_hvflip = torch.sigmoid(model(patch_hvflip))
                    pred_hvflip = torch.flip(pred_hvflip, dims=[2, 3])
                    
                    # Average all predictions
                    prediction = (prediction + pred_hflip + pred_vflip + pred_hvflip) / 4.0
                
                # Apply Gaussian weight to prediction
                weighted_pred = prediction * weight_tensor
                
                # Accumulate weighted predictions and weights
                pred_map[:, :, y:y+patch_size, x:x+patch_size] += weighted_pred
                weight_map[:, :, y:y+patch_size, x:x+patch_size] += weight_tensor
    
    # Weighted average
    avg = pred_map / (weight_map + 1e-6)
    
    # Crop back to original size
    final = avg[:, :, :height, :width]
    mask = (final > 0.5).squeeze().detach().cpu().numpy().astype(np.uint8)
    
    # Debug: log prediction statistics
    prob_np = final.squeeze().detach().cpu().numpy()
    print(f"MODEL - Prediction stats: min={prob_np.min():.4f}, max={prob_np.max():.4f}, mean={prob_np.mean():.4f}")
    print(f"MODEL - Mask stats: total={mask.size}, positive={mask.sum()}, ratio={100*mask.sum()/mask.size:.2f}%")
    
    return mask, profile


def run_model1_inference(image_path: str, bbox: List[float], image_id: str,
                          geometry: Optional[Dict] = None,
                          use_tta: Optional[bool] = None) -> Tuple[Optional[bytes], Optional[str], Optional[Dict]]:
    """Run segmentation model inference on an image.
    
    Args:
        image_path: Path to input raster file
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        image_id: Image identifier for caching
        geometry: Optional GeoJSON geometry for masking
        use_tta: Use Test Time Augmentation for better results (slower)
        
    Returns:
        Tuple of (preview_bytes, overlay_path, overlay_meta) or (None, None, None) on failure
    """
    from .visualization import (
        stretch_uint8, 
        warp_rgb_and_mask_to_aoi, 
        save_rgba_overlay_png_with_transparency,
        generate_overlay_path,
    )
    
    if not _MODEL1_READY:
        if not init_model1():
            return None, None, None
    
    if not _MODEL1_READY or _MODEL1 is None or _MODEL1_DEVICE is None:
        print(f"MODEL - not ready: {_MODEL1_ERROR}")
        return None, None, None
    
    try:
        # Get expected channels from config
        expected_channels = 13  # Default
        if _MODEL1_CFG:
            expected_channels = int(_MODEL1_CFG['model']['args'].get('in_channels', 13))
        
        # Use config default for TTA if not explicitly specified
        actual_use_tta = use_tta if use_tta is not None else MODEL1_USE_TTA
        
        print(f"MODEL - Running inference on {image_path}")
        print(f"  patch_size={MODEL1_PATCH_SIZE}, overlap={MODEL1_OVERLAP}, channels={expected_channels}, TTA={actual_use_tta}")
        
        # Run prediction with Gaussian blending
        mask, profile = _predict_large_image_mask(
            _MODEL1, image_path, MODEL1_PATCH_SIZE, MODEL1_OVERLAP, _MODEL1_DEVICE,
            expected_channels=expected_channels, use_tta=actual_use_tta
        )
        
        # Read RGB bands for preview (B4=Red, B3=Green, B2=Blue)
        # New band order: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
        # So B4=band4, B3=band3, B2=band2
        with rasterio.open(image_path) as src:
            b4, b3, b2 = src.read(4), src.read(3), src.read(2)  # Red, Green, Blue
            src_transform = src.transform
            src_crs = src.crs
        
        # Create preview overlay
        base = np.dstack([stretch_uint8(b4), stretch_uint8(b3), stretch_uint8(b2)])
        overlay = np.zeros_like(base, dtype=np.uint8)
        overlay[..., 0] = 255  # Red channel
        
        alpha = 0.4
        m = (mask > 0)
        comp = base.copy()
        comp[m] = (base[m] * (1 - alpha) + overlay[m] * alpha).astype(np.uint8)
        
        # Create preview thumbnail
        img = Image.fromarray(comp, mode='RGB')
        img.thumbnail((512, 512), Image.LANCZOS)
        buf = pyio.BytesIO()
        img.save(buf, format='PNG')
        preview = buf.getvalue()
        
        # Cache mask data for pixel value computation
        cache_index_data(image_id, 'model1', mask.astype(np.float32), 
                        src_transform, src_crs, bbox)
        
        # Create AOI-aligned overlay
        min_lon, min_lat, max_lon, max_lat = bbox
        
        overlay_rgb = np.zeros((profile['height'], profile['width'], 3), dtype=np.uint8)
        overlay_rgb[mask == 1] = [255, 0, 0]  # Red for mangrove
        
        aoi_rgb, aoi_mask, (aoi_w, aoi_h), _ = warp_rgb_and_mask_to_aoi(
            overlay_rgb, mask, profile['transform'], profile['crs'],
            (min_lon, min_lat, max_lon, max_lat), scale_m=10, geometry=geometry
        )
        
        aoi_rgb[~aoi_mask] = 0
        
        # Save overlay
        overlay_png_path = generate_overlay_path("model1_overlay")
        # Note: aoi_mask is already the reprojected mangrove mask from warp_rgb_and_mask_to_aoi
        # Using >= 128 instead of == 255 because bilinear resampling can change exact values
        is_mangrove = (aoi_rgb[:, :, 0] >= 128)
        final_mask = is_mangrove & aoi_mask
        
        # Log mask statistics for debugging
        print(f"MODEL - Mask stats: total={final_mask.size}, mangrove={final_mask.sum()}, ratio={100*final_mask.sum()/final_mask.size:.2f}%")
        
        save_rgba_overlay_png_with_transparency(aoi_rgb, final_mask, overlay_png_path)
        
        overlay_meta = {
            'width': int(aoi_w),
            'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        }
        
        return preview, overlay_png_path, overlay_meta
        
    except Exception as e:
        print(f"MODEL - inference error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

