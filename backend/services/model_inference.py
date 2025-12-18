"""
Model inference module for deep learning segmentation.
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
    MODEL1_LOG_DIR,
    MODEL1_GPUS,
    MODEL1_PATCH_SIZE,
    MODEL1_OVERLAP,
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


def init_model1() -> bool:
    """Initialize Model1 (segmentation model) for inference.
    
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
            _MODEL1_ERROR = "PyTorch not available"
            print(f"MODEL1 - {_MODEL1_ERROR}")
            return False
        
        log_dir = MODEL1_LOG_DIR
        if not log_dir:
            _MODEL1_ERROR = "MODEL1_LOG_DIR environment variable not set"
            print(f"MODEL1 - {_MODEL1_ERROR}")
            return False
        
        cfg_path = os.path.join(log_dir, 'config.yaml')
        if not os.path.exists(cfg_path):
            _MODEL1_ERROR = f"Config file not found at {cfg_path}"
            print(f"MODEL1 - {_MODEL1_ERROR}")
            return False
        
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            
            encoder_weights = cfg['model']['args'].get('encoder_weights', None)
            in_ch = int(cfg['model']['args'].get('in_channels', 3))
            out_classes = int(cfg['model']['args'].get('classes', 1))
            encoder_name = cfg['model']['args'].get('encoder_name', 'resnet34')
            
            device = _get_device_from_env(MODEL1_GPUS)
            model_name = cfg['model']['name']
            
            model = create_model(
                model_name=model_name, 
                encoder_weights=encoder_weights, 
                in_channels=in_ch, 
                classes=out_classes, 
                encoder_name=encoder_name
            )
            
            ckpt = os.path.join(log_dir, 'weights', 'last.pt')
            if not os.path.exists(ckpt):
                _MODEL1_ERROR = f"Checkpoint not found at {ckpt}"
                print(f"MODEL1 - {_MODEL1_ERROR}")
                return False
            
            state = torch.load(ckpt, map_location=device)
            model.load_state_dict(state['model'] if isinstance(state, dict) and 'model' in state else state)
            model.to(device)
            model.eval()
            
            _MODEL1 = model
            _MODEL1_DEVICE = device
            _MODEL1_CFG = cfg
            _MODEL1_READY = True
            _MODEL1_ERROR = None
            
            print(f"MODEL1 - loaded successfully on {device}")
            return True
            
        except Exception as e:
            _MODEL1_ERROR = str(e)
            print(f"MODEL1 - load error: {e}")
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
                               overlap: float, device) -> Tuple[np.ndarray, dict]:
    """Run inference on a large image using sliding window.
    
    Args:
        model: PyTorch model
        image_path: Path to input image
        patch_size: Size of patches for inference
        overlap: Overlap ratio between patches
        device: PyTorch device
        
    Returns:
        Tuple of (mask array, rasterio profile)
    """
    with rasterio.open(image_path) as src:
        image = src.read(indexes=[1, 2, 3, 4, 5, 6])  # (C, H, W)
        profile = src.profile
    
    _, height, width = image.shape
    img = image.astype(np.float32) / 10000.0
    
    stride = max(1, int(patch_size * (1.0 - overlap)))
    pad_h = (stride - (height - patch_size) % stride) % stride
    pad_w = (stride - (width - patch_size) % stride) % stride
    padded = np.pad(img, ((0, 0), (0, pad_h), (0, pad_w)), mode='constant')
    
    ph, pw = padded.shape[1], padded.shape[2]
    
    pred_map = torch.zeros((1, 1, ph, pw), device=device, dtype=torch.float32)
    cnt_map = torch.zeros((1, 1, ph, pw), device=device, dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        for y in range(0, ph - patch_size + 1, stride):
            for x in range(0, pw - patch_size + 1, stride):
                patch = padded[:, y:y+patch_size, x:x+patch_size]
                patch_tensor = torch.from_numpy(patch).float().to(device).unsqueeze(0)
                out = model(patch_tensor)
                prob = torch.sigmoid(out)
                pred_map[:, :, y:y+patch_size, x:x+patch_size] += prob
                cnt_map[:, :, y:y+patch_size, x:x+patch_size] += 1
    
    avg = pred_map / (cnt_map + 1e-6)
    final = avg[:, :, :height, :width]
    mask = (final > 0.5).squeeze().detach().cpu().numpy().astype(np.uint8)
    
    return mask, profile


def run_model1_inference(image_path: str, bbox: List[float], image_id: str,
                          geometry: Optional[Dict] = None) -> Tuple[Optional[bytes], Optional[str], Optional[Dict]]:
    """Run Model1 segmentation inference on an image.
    
    Args:
        image_path: Path to input raster file
        bbox: Bounding box [min_lon, min_lat, max_lon, max_lat]
        image_id: Image identifier for caching
        geometry: Optional GeoJSON geometry for masking
        
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
        print(f"MODEL1 - not ready: {_MODEL1_ERROR}")
        return None, None, None
    
    try:
        # Run prediction
        mask, profile = _predict_large_image_mask(
            _MODEL1, image_path, MODEL1_PATCH_SIZE, MODEL1_OVERLAP, _MODEL1_DEVICE
        )
        
        # Read RGB bands for preview
        with rasterio.open(image_path) as src:
            b4, b3, b2 = src.read(1), src.read(2), src.read(3)
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
        is_mangrove = (aoi_rgb[:, :, 0] == 255)
        final_mask = is_mangrove & aoi_mask
        save_rgba_overlay_png_with_transparency(aoi_rgb, final_mask, overlay_png_path)
        
        overlay_meta = {
            'width': int(aoi_w),
            'height': int(aoi_h),
            'bounds': [float(min_lat), float(min_lon), float(max_lat), float(max_lon)]
        }
        
        return preview, overlay_png_path, overlay_meta
        
    except Exception as e:
        print(f"MODEL1 - inference error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

