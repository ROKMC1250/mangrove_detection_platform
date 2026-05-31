"""
Flood segmentation inference (Sentinel-1 VV, UNet++/ResNet34).

Ports the full-scene tile-and-blend logic from the research repo's
trainer/base_trainer.py:inference() while reusing EarthScope's existing
SMP factory (repo/mangrove_segmentation/model/smp_models.create_model)
and S1 download infrastructure.
"""

import math
import os
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio

from ..core.config import (
    FLOOD_BATCH_SIZE,
    FLOOD_BLEND_MODE,
    FLOOD_CHECKPOINT,
    FLOOD_MODEL_DIR,
    FLOOD_MODEL_PARAMS,
    FLOOD_OVERLAP_PX,
    FLOOD_PATCH_SIZE,
    FLOOD_SAR_DB_MAX,
    FLOOD_SAR_DB_MIN,
    MODEL1_GPUS,
    OUTPUTS_DIR,
)
from ..utils.cache import (
    RASTER_CACHE_LOCK,
    RASTER_FILE_CACHE,
    bbox_to_cache_key,
    cache_raster_file,
)
from .gpu_compute import _GPU_LOCK

try:
    import torch
    import yaml
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    yaml = None

try:
    from model.smp_models import create_model
    SMP_MODEL_AVAILABLE = True
except ImportError:
    SMP_MODEL_AVAILABLE = False
    create_model = None


# ===== Global state =====
_FLOOD_MODEL = None
_FLOOD_DEVICE = None
_FLOOD_CFG = None
_FLOOD_READY = False
_FLOOD_ERROR = None
_FLOOD_LOCK = threading.Lock()


def _get_device(gpus_str: str):
    if not TORCH_AVAILABLE:
        return 'cpu'
    if torch.cuda.is_available() and gpus_str:
        try:
            gpu_ids = [int(g) for g in gpus_str.split(',') if g.strip().isdigit()]
            if gpu_ids:
                return torch.device(f"cuda:{gpu_ids[0]}")
        except Exception:
            pass
    return torch.device('cpu')


def _resolve_model_config() -> Dict:
    """Load the flood checkpoint's config.yaml if present, else fall back to defaults."""
    cfg_path = os.path.join(FLOOD_MODEL_DIR, 'config.yaml')
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as exc:
            print(f"⚠️  FLOOD - failed to load {cfg_path}: {exc}; using defaults")
            cfg = {}
    else:
        cfg = {}

    model_section = cfg.get('model') or {}
    encoder_name = model_section.get('encoder_name') or FLOOD_MODEL_PARAMS.get('encoder_name', 'resnet34')
    in_channels = int(model_section.get('in_channels') or FLOOD_MODEL_PARAMS.get('in_channels', 3))
    classes = int(model_section.get('classes') or FLOOD_MODEL_PARAMS.get('classes', 1))
    model_name = FLOOD_MODEL_PARAMS.get('name', 'UnetPlusPlus')

    encoder_weights = model_section.get('encoder_weights')
    if isinstance(encoder_weights, str) and encoder_weights.lower() in {'none', 'null', ''}:
        encoder_weights = None

    return {
        'name': model_name,
        'encoder_name': encoder_name,
        'in_channels': in_channels,
        'classes': classes,
        'encoder_weights': encoder_weights,
    }


def init_flood_model() -> bool:
    """Load the flood UNet++ checkpoint. Idempotent; graceful failure."""
    global _FLOOD_MODEL, _FLOOD_DEVICE, _FLOOD_CFG, _FLOOD_READY, _FLOOD_ERROR

    if _FLOOD_READY:
        return True

    with _FLOOD_LOCK:
        if _FLOOD_READY:
            return True

        if not TORCH_AVAILABLE:
            _FLOOD_ERROR = "PyTorch not available - flood segmentation disabled"
            print(f"⚠️  FLOOD - {_FLOOD_ERROR}")
            return False
        if not SMP_MODEL_AVAILABLE:
            _FLOOD_ERROR = "segmentation_models_pytorch not available - flood segmentation disabled"
            print(f"⚠️  FLOOD - {_FLOOD_ERROR}")
            return False
        if not FLOOD_MODEL_DIR:
            _FLOOD_ERROR = "FLOOD_MODEL_DIR not configured"
            print(f"⚠️  FLOOD - {_FLOOD_ERROR}")
            return False

        cfg = _resolve_model_config()

        try:
            device = _get_device(MODEL1_GPUS)
            print(
                f"FLOOD - Creating {cfg['name']} "
                f"(encoder={cfg['encoder_name']}, in_channels={cfg['in_channels']}, classes={cfg['classes']})"
            )
            model = create_model(
                model_name=cfg['name'],
                encoder_name=cfg['encoder_name'],
                in_channels=cfg['in_channels'],
                classes=cfg['classes'],
                encoder_weights=cfg['encoder_weights'],
            )

            ckpt_path = os.path.join(FLOOD_MODEL_DIR, 'weights', FLOOD_CHECKPOINT)
            if not os.path.exists(ckpt_path):
                alt = os.path.join(FLOOD_MODEL_DIR, FLOOD_CHECKPOINT)
                if os.path.exists(alt):
                    ckpt_path = alt
                else:
                    _FLOOD_ERROR = f"Checkpoint not found at {ckpt_path}"
                    print(f"⚠️  FLOOD - {_FLOOD_ERROR}")
                    return False

            print(f"FLOOD - Loading checkpoint: {ckpt_path}")
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            # QuickTorch saves as {'model': state_dict, ...}; fall back to raw state dict.
            state_dict = state['model'] if isinstance(state, dict) and 'model' in state else state
            # Research repo wraps smp model as SegmentationModel.net = smp.UnetPlusPlus(...),
            # so the saved keys are prefixed with "net.". Strip it for direct smp loading.
            if any(k.startswith('net.') for k in state_dict.keys()):
                state_dict = {k[len('net.'):]: v for k, v in state_dict.items() if k.startswith('net.')}
                print("FLOOD - Stripped 'net.' prefix from state_dict keys")
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()

            _FLOOD_MODEL = model
            _FLOOD_DEVICE = device
            _FLOOD_CFG = cfg
            _FLOOD_READY = True
            _FLOOD_ERROR = None
            print(f"✅ FLOOD - {cfg['name']} loaded on {device}")
            return True

        except Exception as exc:
            _FLOOD_ERROR = f"{exc}"
            print(f"⚠️  FLOOD - load error: {exc}")
            import traceback
            traceback.print_exc()
            return False


def is_flood_model_ready() -> bool:
    return _FLOOD_READY


def get_flood_model_status() -> Dict:
    return {
        'loaded': bool(_FLOOD_READY),
        'device': str(_FLOOD_DEVICE) if _FLOOD_DEVICE is not None else None,
        'error': _FLOOD_ERROR,
        'model_dir': FLOOD_MODEL_DIR,
        'checkpoint': FLOOD_CHECKPOINT,
    }


# ===== Blending & normalization (ported from research repo) =====

def _create_blend_weight(height: int, width: int, mode: str = 'gaussian') -> np.ndarray:
    """2D blending weight, peaked at center. Mirrors base_trainer._create_blend_weight."""
    if mode == 'gaussian':
        sigma_h = height / 4.0
        sigma_w = width / 4.0
        y = np.linspace(-(height - 1) / 2.0, (height - 1) / 2.0, height)
        x = np.linspace(-(width - 1) / 2.0, (width - 1) / 2.0, width)
        yy, xx = np.meshgrid(y, x, indexing='ij')
        weight = np.exp(-0.5 * ((yy / sigma_h) ** 2 + (xx / sigma_w) ** 2))
    elif mode == 'hanning':
        h_win = np.hanning(height) if height > 1 else np.ones(1)
        w_win = np.hanning(width) if width > 1 else np.ones(1)
        weight = np.outer(h_win, w_win)
    elif mode == 'linear':
        h_win = np.minimum(np.arange(height), np.arange(height)[::-1]) + 1
        w_win = np.minimum(np.arange(width), np.arange(width)[::-1]) + 1
        h_win = h_win / h_win.max()
        w_win = w_win / w_win.max()
        weight = np.outer(h_win, w_win)
    else:
        weight = np.ones((height, width))
    return np.clip(weight, 0.01, 1.0).astype(np.float32)


def _normalize_sar_db(arr: np.ndarray, nan_mask: np.ndarray) -> np.ndarray:
    """dB clip [-30, +10] -> [0, 1]. nan/inf -> 0. Matches sentinel1_real_data._normalize_sar."""
    clipped = np.clip(arr, FLOOD_SAR_DB_MIN, FLOOD_SAR_DB_MAX)
    normalized = (clipped - FLOOD_SAR_DB_MIN) / (FLOOD_SAR_DB_MAX - FLOOD_SAR_DB_MIN)
    normalized[nan_mask] = 0.0
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32)


def _pad_amount(dim: int, patch_size: int, stride: int) -> int:
    """Reflect-pad amount so the dimension fits an integer number of strides."""
    if dim <= patch_size:
        return patch_size - dim
    rem = (dim - patch_size) % stride
    return 0 if rem == 0 else stride - rem


def _predict_full_scene_flood(
    image_path: str,
    progress_cb=None,
) -> Dict:
    """
    Run UNet++ inference over a full SAR scene using the **mangrove-style**
    reflect-pad + uniform-tile + Gaussian-weighted-blend stitching.

    Algorithm (mirrors backend/services/model_inference._predict_large_image_mask):
      1. Read VV (band 1) for the whole scene.
      2. Build a boolean `data_mask` from the source's nodata/nan.
      3. Normalize VV to [0, 1] via dB clip [-30, +10]; nodata -> 0.
      4. Reflect-pad to (full_h + pad_h, full_w + pad_w) so all patches are
         exactly FLOOD_PATCH_SIZE × FLOOD_PATCH_SIZE — no per-tile resize,
         no slice-based edge handling.
      5. Iterate uniform full-size patches with stride = patch_size - overlap.
         For each patch: repeat VV to 3 channels, run model, multiply by a
         fixed (patch_size, patch_size) Gaussian weight, accumulate into
         (ph, pw) sum and weight buffers.
      6. Weighted average -> crop back to (full_h, full_w).
      7. Zero out probabilities at no-data pixels using `data_mask`.
      8. Binarize at 0.5, save GeoTIFFs preserving CRS/transform.
    """
    if not _FLOOD_READY:
        raise RuntimeError("Flood model not loaded")

    patch_size = FLOOD_PATCH_SIZE
    stride = max(1, patch_size - FLOOD_OVERLAP_PX)

    with rasterio.open(image_path) as src:
        if src.count < 1:
            raise ValueError(f"{image_path} has no bands")
        full_h, full_w = src.height, src.width
        meta = src.profile.copy()
        nodata = src.nodata
        vv = src.read(1).astype(np.float32)  # (H, W)

    # Source-extent valid mask (independent of model prediction).
    data_mask = np.isfinite(vv)
    if nodata is not None:
        data_mask &= (vv != nodata)
    nan_mask = ~data_mask

    # dB normalize -> [0, 1]; nodata pixels go to 0.
    vv = np.nan_to_num(vv, nan=0.0, posinf=0.0, neginf=0.0)
    vv_norm = _normalize_sar_db(vv, nan_mask)
    del vv

    # Reflect-pad so every patch is patch_size × patch_size.
    pad_h = _pad_amount(full_h, patch_size, stride)
    pad_w = _pad_amount(full_w, patch_size, stride)
    padded = np.pad(vv_norm, ((0, pad_h), (0, pad_w)), mode='reflect')
    del vv_norm
    ph, pw = padded.shape

    print(
        f"FLOOD - Scene {full_h}x{full_w} -> padded {ph}x{pw} "
        f"(patch={patch_size}, stride={stride}, overlap={FLOOD_OVERLAP_PX})"
    )

    # Fixed (patch_size, patch_size) Gaussian weight — every patch uses it.
    weight_window = _create_blend_weight(patch_size, patch_size, FLOOD_BLEND_MODE)
    weight_tensor = torch.from_numpy(weight_window).to(_FLOOD_DEVICE).unsqueeze(0).unsqueeze(0)

    # CPU accumulators so we don't OOM the GPU on huge scenes (e.g. 16k × 16k).
    arr_sum = np.zeros((ph, pw), dtype=np.float32)
    weight_sum = np.zeros((ph, pw), dtype=np.float32)

    # Enumerate patch starts; all patches are full-sized in this scheme.
    starts: List[Tuple[int, int]] = []
    for y in range(0, ph - patch_size + 1, stride):
        for x in range(0, pw - patch_size + 1, stride):
            starts.append((y, x))
    total_tiles = len(starts)

    batch_patches: List[np.ndarray] = []
    batch_pos: List[Tuple[int, int]] = []
    processed = 0

    def _flush_batch():
        nonlocal processed
        if not batch_patches:
            return
        x_np = np.stack(batch_patches, axis=0)  # (B, 3, ps, ps)
        x_t = torch.from_numpy(x_np).to(_FLOOD_DEVICE)
        with _GPU_LOCK:
            with torch.no_grad():
                logits = _FLOOD_MODEL(x_t)
                probs = torch.sigmoid(logits) * weight_tensor  # (B, 1, ps, ps)
        probs_np = probs.detach().cpu().numpy().astype(np.float32)

        for i, (yy, xx) in enumerate(batch_pos):
            arr_sum[yy:yy + patch_size, xx:xx + patch_size] += probs_np[i, 0]
            weight_sum[yy:yy + patch_size, xx:xx + patch_size] += weight_window
            processed += 1
            if progress_cb is not None and processed % 4 == 0:
                progress_cb(processed, total_tiles)

        batch_patches.clear()
        batch_pos.clear()

    for (y, x) in starts:
        patch = padded[y:y + patch_size, x:x + patch_size]
        patch3 = np.repeat(patch[None, ...], 3, axis=0).astype(np.float32)  # (3, ps, ps)
        batch_patches.append(patch3)
        batch_pos.append((y, x))
        if len(batch_patches) >= FLOOD_BATCH_SIZE:
            _flush_batch()
    _flush_batch()
    del padded

    # Crop back to original size and divide.
    arr_prob = arr_sum[:full_h, :full_w] / np.clip(weight_sum[:full_h, :full_w], 1e-6, None)
    arr_prob = arr_prob.astype(np.float32)
    # Wipe predictions at GEE-clip nodata fill (UTM-vs-AOI border) so the
    # overlay does not show a probability bleed there.
    arr_prob[~data_mask] = 0.0
    arr_binary = (arr_prob > 0.5).astype(np.float32)

    stem = os.path.splitext(os.path.basename(image_path))[0]
    out_meta = meta.copy()
    out_meta.update({"count": 1, "dtype": "float32", "nodata": 0.0})
    prob_path = os.path.join(OUTPUTS_DIR, f"{stem}_flood_probability.tif")
    mask_path = os.path.join(OUTPUTS_DIR, f"{stem}_flood_watermask.tif")
    with rasterio.open(prob_path, "w", **out_meta) as dst:
        dst.write(arr_prob.astype(np.float32), 1)
    with rasterio.open(mask_path, "w", **out_meta) as dst:
        dst.write(arr_binary, 1)
    print(f"FLOOD - Saved {prob_path} and {mask_path}")

    return {
        'prob_map': arr_prob,
        'binary_mask': arr_binary,
        'data_mask': data_mask,
        'prob_path': prob_path,
        'mask_path': mask_path,
        'transform': meta['transform'],
        'crs': meta['crs'],
        'height': full_h,
        'width': full_w,
        'min_val': float(arr_prob.min()),
        'max_val': float(arr_prob.max()),
        'stats': {
            'water_pixels': int(arr_binary.sum()),
            'total_pixels': int(arr_binary.size),
        },
    }


def ensure_s1_raster_cached(
    image_id: str,
    bbox: List[float],
    geometry: Optional[Dict] = None,
    job_id: Optional[str] = None,
) -> str:
    """
    Return a cached path to a VV+VH Float32 GeoTIFF for the given S1 image.
    Downloads via EarthScope's standard S1 path if not cached. If a job_id
    is passed, the download phase reports progress through PROGRESS_TRACKER.
    """
    cache_key = bbox_to_cache_key(image_id, bbox)
    with RASTER_CACHE_LOCK:
        cached = RASTER_FILE_CACHE.get(cache_key)
    if cached and os.path.exists(cached):
        print(f"FLOOD - S1 raster cache hit: {cached}")
        return cached

    import ee
    from .earth_engine import bbox_to_geometry
    from .downloader import download_ee_image, generate_output_path

    full_id = image_id if image_id.startswith("COPERNICUS/S1_GRD/") else f"COPERNICUS/S1_GRD/{image_id}"
    aoi = bbox_to_geometry(bbox, geometry)
    aoi_rect = aoi.bounds()
    img = ee.Image(full_id).clip(aoi).select(['VV', 'VH'])

    out_path = generate_output_path("s1_flood", image_id)
    print(f"FLOOD - Downloading S1 GRD VV+VH for {image_id} -> {out_path}")
    download_ee_image(img, ['VV', 'VH'], aoi_rect, 10, out_path, job_id=job_id, as_float=True)

    cache_raster_file(image_id, bbox, out_path)
    return out_path


def run_flood_inference(
    image_id: str,
    bbox: List[float],
    geometry: Optional[Dict] = None,
    progress_cb=None,
) -> Optional[Dict]:
    """
    Top-level entrypoint: ensure S1 raster is cached, then run full-scene inference.
    Returns the same dict shape as _predict_full_scene_flood plus bbox/image_id.
    """
    if not _FLOOD_READY:
        if not init_flood_model():
            return None

    try:
        image_path = ensure_s1_raster_cached(image_id, bbox, geometry)
        result = _predict_full_scene_flood(image_path, progress_cb=progress_cb)
        result['bbox'] = bbox
        result['image_id'] = image_id
        return result
    except Exception as exc:
        print(f"FLOOD - inference error: {exc}")
        import traceback
        traceback.print_exc()
        return None
