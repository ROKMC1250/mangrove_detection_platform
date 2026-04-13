"""
GPU-accelerated computation module using PyTorch CUDA.

Provides drop-in replacements for CPU-heavy numpy operations.
All functions automatically fall back to CPU if CUDA fails (OOM, etc.).
"""

import io
import base64
import numpy as np
from typing import Tuple, Optional

try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
    if HAS_CUDA:
        _DEVICE = torch.device('cuda')
        torch.zeros(1, device=_DEVICE)
    else:
        _DEVICE = torch.device('cpu')
except ImportError:
    HAS_CUDA = False
    _DEVICE = None
    torch = None

print(f"GPU Compute: CUDA={'available' if HAS_CUDA else 'not available'}")


def _gpu_cleanup():
    """Free GPU memory after computation."""
    if HAS_CUDA:
        torch.cuda.empty_cache()


# =============================================================================
# GPU Resident Image Cache
# =============================================================================

_GPU_IMAGE_CACHE = {}  # cache_key -> torch.Tensor (H, W, Bands) on CUDA


def load_image_to_gpu(cache_key: str, band_arrays: list) -> 'torch.Tensor':
    """Stack band arrays and load to GPU once. Returns GPU tensor (H, W, Bands)."""
    if cache_key in _GPU_IMAGE_CACHE:
        return _GPU_IMAGE_CACHE[cache_key]

    stacked = np.stack(band_arrays, axis=-1)
    if HAS_CUDA:
        t = torch.from_numpy(stacked).to(_DEVICE, dtype=torch.float32)
    else:
        t = torch.from_numpy(stacked.astype(np.float32))
    _GPU_IMAGE_CACHE[cache_key] = t
    print(f"GPU CACHE - Loaded {cache_key}: {t.shape}, {t.device}, {t.element_size()*t.nelement()/1024/1024:.0f}MB")
    return t


def get_gpu_image(cache_key: str):
    """Get cached GPU tensor, or None."""
    return _GPU_IMAGE_CACHE.get(cache_key)


def clear_gpu_image(cache_key: str = None):
    """Clear GPU image cache (specific key or all)."""
    if cache_key:
        t = _GPU_IMAGE_CACHE.pop(cache_key, None)
        if t is not None:
            del t
    else:
        _GPU_IMAGE_CACHE.clear()
    _gpu_cleanup()


def compute_percentiles_gpu(gpu_image: 'torch.Tensor', percentile_list: list = None,
                            max_samples: int = 2_000_000) -> dict:
    """Compute percentiles on GPU using stride-based sampling.

    Avoids large intermediate allocations (isfinite mask, randperm, valid copy).
    Instead, takes every N-th pixel — no extra GPU memory needed beyond the sample.

    Args:
        gpu_image: (H, W, Bands) float32 tensor on CUDA
        percentile_list: list of percentile values (1-99). Default: 1..99
        max_samples: max pixels to sample (default 2M)
    Returns:
        dict: {"1": val, "2": val, ..., "99": val}
    """
    if percentile_list is None:
        percentile_list = list(range(1, 100))

    flat = gpu_image.reshape(-1)
    n = flat.numel()

    # Stride sampling: take every N-th element (no extra allocation)
    step = max(1, n // max_samples)
    sampled = flat[::step]  # view with stride, ~max_samples elements

    # Filter NaN/Inf on the small sample only
    finite_mask = torch.isfinite(sampled)
    valid = sampled[finite_mask]

    if valid.numel() == 0:
        return {str(p): 0.0 for p in percentile_list}

    quantiles = torch.tensor([p / 100.0 for p in percentile_list],
                             device=valid.device, dtype=torch.float32)
    values = torch.quantile(valid, quantiles)

    result = {}
    for i, p in enumerate(percentile_list):
        result[str(p)] = float(values[i].item())

    del valid, sampled, finite_mask, quantiles, values
    _gpu_cleanup()
    return result


def stretch_bands_gpu(gpu_image: 'torch.Tensor', min_val: float, max_val: float) -> dict:
    """Stretch each band to uint8 on GPU and return as base64 JPEG data URLs.

    Processes one band at a time to minimize peak GPU memory.

    Args:
        gpu_image: (H, W, Bands) float32 tensor on CUDA
        min_val: stretch minimum
        max_val: stretch maximum
    Returns:
        dict: {"band_index": base64_data_url, ...}, width, height
    """
    from PIL import Image as PILImage

    h, w, nb = gpu_image.shape
    denom = max(max_val - min_val, 1e-10)

    result = {}
    for b in range(nb):
        # Stretch one band at a time on GPU to save memory
        band = gpu_image[:, :, b]
        stretched = torch.clamp((band - min_val) / denom, 0.0, 1.0)
        gray = (stretched * 255).to(torch.uint8).cpu().numpy()
        del stretched

        rgb = np.stack([gray, gray, gray], axis=-1)
        buf = io.BytesIO()
        PILImage.fromarray(rgb).save(buf, format='JPEG', quality=85)
        result[b] = f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        del gray, rgb

    _gpu_cleanup()
    return result, w, h


def detect_from_gpu_tensor(
    gpu_image: 'torch.Tensor',
    target_points: list,
    algorithm: str,
    selected_bands: list = None,
    n_bg_samples: int = 10000,
) -> dict:
    """Run target detection entirely on GPU using cached tensor. No CPU↔GPU transfer.

    Args:
        gpu_image: (H, W, Bands) float32 tensor on CUDA
        target_points: [{"row": int, "col": int}, ...]
        algorithm: SAM/ACE/CEM/MF
        selected_bands: optional list of band indices
        n_bg_samples: number of background samples
    Returns:
        dict with detection_map (numpy), threshold, stats, target_spectrum, bg stats
    """
    from .target_detection import find_optimal_threshold_otsu

    img = gpu_image
    if selected_bands and len(selected_bands) > 0:
        img = img[:, :, selected_bands]

    h, w, nb = img.shape

    # Extract target spectra on GPU
    target_spectra = []
    target_pixels = []
    for pt in target_points:
        r = max(0, min(int(pt.get('row', pt.get('lat', 0))), h - 1))
        c = max(0, min(int(pt.get('col', pt.get('lng', 0))), w - 1))
        target_pixels.append((c, r))
        target_spectra.append(img[r, c, :])

    target_spectrum = torch.stack(target_spectra).mean(dim=0)  # on GPU

    # Background sampling on GPU
    # Create exclusion mask
    exclude = torch.zeros(h, w, dtype=torch.bool, device=img.device)
    for x, y in target_pixels:
        y0, y1 = max(0, y-5), min(h, y+6)
        x0, x1 = max(0, x-5), min(w, x+6)
        exclude[y0:y1, x0:x1] = True

    valid_idx = (~exclude).nonzero(as_tuple=False)  # (N, 2) — [row, col]
    n_bg = min(n_bg_samples, len(valid_idx))
    perm = torch.randperm(len(valid_idx), device=img.device)[:n_bg]
    bg_idx = valid_idx[perm]
    bg_spectra = img[bg_idx[:, 0], bg_idx[:, 1], :]  # (n_bg, nb) on GPU

    # Background statistics on GPU
    bg_mean = bg_spectra.mean(dim=0)
    centered = bg_spectra - bg_mean
    n = bg_spectra.shape[0]
    bg_cov = (centered.T @ centered) / (n - 1)
    reg = 1e-6 * bg_cov.trace() / bg_cov.shape[0]
    bg_cov += reg * torch.eye(nb, device=img.device)
    bg_cov_inv = torch.linalg.inv(bg_cov)

    # Reshape for detection
    pixels = img.reshape(-1, nb)  # (H*W, nb) — still on GPU, no copy

    algo = algorithm.upper()
    if algo == 'SAM':
        t_norm = torch.clamp(target_spectrum.norm(), min=1e-10)
        p_norms = torch.clamp(pixels.norm(dim=1), min=1e-10)
        cos_a = torch.clamp((pixels @ target_spectrum) / (p_norms * t_norm), -1, 1)
        angles = torch.acos(cos_a)
        det = angles.max() - angles
    elif algo == 'ACE':
        pc = pixels - bg_mean
        tc = target_spectrum - bg_mean
        wt = bg_cov_inv @ tc
        num = (pc @ wt) ** 2
        tt = tc @ wt
        wp = pc @ bg_cov_inv
        pp = (wp * pc).sum(dim=1)
        det = torch.clamp(num / torch.clamp(tt * pp, min=1e-10), 0, 1)
        del pc, wt, num, wp, pp
    elif algo == 'CEM':
        Rd = bg_cov_inv @ target_spectrum
        dRd = torch.clamp(target_spectrum @ Rd, min=1e-10)
        fw = Rd / dRd
        det = pixels @ fw
        del Rd, fw
    elif algo == 'MF':
        pc = pixels - bg_mean
        tc = target_spectrum - bg_mean
        Ct = bg_cov_inv @ tc
        tCt = torch.clamp(tc @ Ct, min=1e-10)
        fw = Ct / tCt
        det = pc @ fw
        del pc, Ct, fw
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    detection_map = det.view(h, w).cpu().numpy()
    del det, pixels

    valid = np.isfinite(detection_map)
    min_val = float(detection_map[valid].min()) if valid.any() else 0.0
    max_val = float(detection_map[valid].max()) if valid.any() else 1.0

    threshold = find_optimal_threshold_otsu(detection_map)

    return {
        'detection_map': detection_map,
        'threshold': threshold,
        'min_val': min_val,
        'max_val': max_val,
        'target_spectrum': target_spectrum.cpu().numpy().tolist(),
        'background_mean': bg_mean.cpu().numpy().tolist(),
        'background_std': bg_spectra.std(dim=0).cpu().numpy().tolist(),
        'target_points_pixel': target_pixels,
        'used_bands': selected_bands if selected_bands else list(range(int(nb))),
    }


def rgba_to_base64_gpu(rgba: np.ndarray, max_dim: int = 2048) -> str:
    """Convert RGBA numpy array to base64 data URL. No file I/O.

    GPU-accelerates the downscaling for large images, then encodes
    the small result as PNG in memory.
    """
    from PIL import Image as PILImage
    h, w = rgba.shape[:2]

    if max(h, w) > max_dim and HAS_CUDA:
        try:
            scale = max_dim / max(h, w)
            nh, nw = int(h * scale), int(w * scale)
            t = torch.from_numpy(rgba).to(_DEVICE).permute(2, 0, 1).unsqueeze(0).float()
            t_small = torch.nn.functional.interpolate(t, size=(nh, nw), mode='bilinear', align_corners=False)
            rgba = t_small.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            del t, t_small
            _gpu_cleanup()
        except Exception:
            img = PILImage.fromarray(rgba, mode='RGBA')
            scale = max_dim / max(h, w)
            img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
            rgba = np.array(img)
    elif max(h, w) > max_dim:
        img = PILImage.fromarray(rgba, mode='RGBA')
        scale = max_dim / max(h, w)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
        rgba = np.array(img)

    buf = io.BytesIO()
    PILImage.fromarray(rgba, mode='RGBA').save(buf, format='PNG', compress_level=1)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


def rgb_mask_to_base64_gpu(rgb: np.ndarray, mask: np.ndarray, max_dim: int = 2048) -> str:
    """Convert RGB + valid mask to RGBA base64 data URL. No file I/O.

    Drop-in replacement for save_rgba_overlay_png_with_transparency() + URL generation.
    Transparent where mask is False.
    """
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    if mask is not None:
        rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    else:
        is_black = (rgb[:, :, 0] < 10) & (rgb[:, :, 1] < 10) & (rgb[:, :, 2] < 10)
        rgba[:, :, 3] = np.where(is_black, 0, 255).astype(np.uint8)
    return rgba_to_base64_gpu(rgba, max_dim)


# =============================================================================
# GPU-accelerated RGBA overlay + downscaled preview generation
# =============================================================================

def build_rgba_and_preview_gpu(
    rgb: np.ndarray, mask: np.ndarray,
    color_override: tuple = None,
    alpha_val: int = 255,
    preview_size: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """Build RGBA overlay and small preview entirely on GPU.

    Args:
        rgb: (H,W,3) uint8 - colormap result
        mask: (H,W) bool - valid pixel mask
        color_override: if set, override RGB with this (R,G,B) where mask is True
        alpha_val: alpha value for valid pixels
        preview_size: max dimension for preview

    Returns:
        (rgba full-size uint8, rgba preview uint8)
    """
    h, w = rgb.shape[:2]

    if not HAS_CUDA:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if color_override:
            rgba[mask, 0] = color_override[0]
            rgba[mask, 1] = color_override[1]
            rgba[mask, 2] = color_override[2]
        else:
            rgba[:, :, :3] = rgb
        rgba[mask, 3] = alpha_val
        # CPU preview via PIL
        from PIL import Image as PILImage
        img = PILImage.fromarray(rgba, mode='RGBA')
        img.thumbnail((preview_size, preview_size), PILImage.LANCZOS)
        preview = np.array(img)
        return rgba, preview

    try:
        t_rgba = torch.zeros(h, w, 4, dtype=torch.uint8, device=_DEVICE)
        m = torch.from_numpy(mask).to(_DEVICE)  # (H, W) bool

        if color_override:
            # Set color channels where mask is True
            # Use nonzero indices for proper indexing
            idx = m.nonzero(as_tuple=True)  # (rows, cols)
            t_rgba[idx[0], idx[1], 0] = color_override[0]
            t_rgba[idx[0], idx[1], 1] = color_override[1]
            t_rgba[idx[0], idx[1], 2] = color_override[2]
            t_rgba[idx[0], idx[1], 3] = alpha_val
        else:
            t_rgb = torch.from_numpy(rgb).to(_DEVICE)
            t_rgba[:, :, :3] = t_rgb
            # Set alpha where mask is True
            alpha_plane = torch.zeros(h, w, dtype=torch.uint8, device=_DEVICE)
            alpha_plane[m] = alpha_val
            t_rgba[:, :, 3] = alpha_plane
            del t_rgb, alpha_plane

        rgba = t_rgba.cpu().numpy()

        # GPU downscale for preview
        scale = preview_size / max(h, w)
        if scale < 1.0:
            ph, pw = int(h * scale), int(w * scale)
            t_for_resize = t_rgba.permute(2, 0, 1).unsqueeze(0).float()
            t_small = torch.nn.functional.interpolate(t_for_resize, size=(ph, pw), mode='bilinear', align_corners=False)
            preview = t_small.squeeze(0).permute(1, 2, 0).to(torch.uint8).cpu().numpy()
            del t_for_resize, t_small
        else:
            preview = rgba.copy()

        del t_rgba, m
        _gpu_cleanup()
        return rgba, preview

    except Exception as e:
        print(f"GPU build_rgba failed ({e}), using CPU")
        _gpu_cleanup()
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if color_override:
            rgba[mask, 0] = color_override[0]
            rgba[mask, 1] = color_override[1]
            rgba[mask, 2] = color_override[2]
        else:
            rgba[:, :, :3] = rgb
        rgba[mask, 3] = alpha_val
        from PIL import Image as PILImage
        img = PILImage.fromarray(rgba, mode='RGBA')
        img.thumbnail((preview_size, preview_size), PILImage.LANCZOS)
        return rgba, np.array(img)


# =============================================================================
# CPU fallback implementations
# =============================================================================

def _cpu_normalized_index(a, b, eps=1e-6):
    denom = a + b
    mask = np.abs(denom) > eps
    out = np.full_like(a, np.nan, dtype=np.float32)
    out[mask] = (a[mask] - b[mask]) / denom[mask]
    return out


def _cpu_savi(nir, red, L=0.5):
    denom = nir + red + L
    mask = np.abs(denom) > 1e-6
    out = np.full_like(nir, np.nan, dtype=np.float32)
    out[mask] = ((nir[mask] - red[mask]) / denom[mask]) * (1 + L)
    return out


def _cpu_colormap(index_data, cmap, vmin, vmax):
    import matplotlib.pyplot as plt
    nan_mask = np.isnan(index_data)
    min_val = vmin if vmin is not None else float(np.percentile(index_data[~nan_mask], 0.5))
    max_val = vmax if vmax is not None else float(np.percentile(index_data[~nan_mask], 99.5))
    denom = max(max_val - min_val, 1e-6)
    norm = np.clip(index_data, min_val, max_val)
    norm = (norm - min_val) / denom
    colored = (plt.get_cmap(cmap)(norm)[:, :, :3] * 255).astype(np.uint8)
    colored[nan_mask] = [0, 0, 0]
    return colored, min_val, max_val


# =============================================================================
# Colormap LUT cache
# =============================================================================

_CMAP_LUT_CACHE = {}


def _get_cmap_lut(cmap_name, n=256):
    if cmap_name in _CMAP_LUT_CACHE:
        return _CMAP_LUT_CACHE[cmap_name]
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cmap_name)
    x = np.linspace(0, 1, n)
    lut_np = (cm(x)[:, :3] * 255).astype(np.uint8)
    lut = torch.from_numpy(lut_np).to(_DEVICE)
    _CMAP_LUT_CACHE[cmap_name] = lut
    return lut


# =============================================================================
# Public API - all with automatic GPU->CPU fallback
# =============================================================================

def compute_normalized_index_gpu(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if not HAS_CUDA:
        return _cpu_normalized_index(a, b, eps)
    try:
        ta = torch.from_numpy(a.astype(np.float32)).to(_DEVICE)
        tb = torch.from_numpy(b.astype(np.float32)).to(_DEVICE)
        denom = ta + tb
        valid = denom.abs() > eps
        result = torch.full_like(ta, float('nan'))
        result[valid] = (ta[valid] - tb[valid]) / denom[valid]
        out = result.cpu().numpy()
        del ta, tb, denom, valid, result
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU normalized_index failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu_normalized_index(a, b, eps)


def compute_savi_gpu(nir: np.ndarray, red: np.ndarray, L: float = 0.5) -> np.ndarray:
    if not HAS_CUDA:
        return _cpu_savi(nir, red, L)
    try:
        tn = torch.from_numpy(nir.astype(np.float32)).to(_DEVICE)
        tr = torch.from_numpy(red.astype(np.float32)).to(_DEVICE)
        denom = tn + tr + L
        valid = denom.abs() > 1e-6
        result = torch.full_like(tn, float('nan'))
        result[valid] = ((tn[valid] - tr[valid]) / denom[valid]) * (1 + L)
        out = result.cpu().numpy()
        del tn, tr, denom, valid, result
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU SAVI failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu_savi(nir, red, L)


def create_index_visualization_gpu(
    index_data: np.ndarray, cmap: str = 'viridis',
    vmin: Optional[float] = None, vmax: Optional[float] = None
) -> Tuple[np.ndarray, float, float]:
    nan_mask = np.isnan(index_data)
    valid_data = index_data[~nan_mask]
    min_val = vmin if vmin is not None else float(np.percentile(valid_data, 0.5))
    max_val = vmax if vmax is not None else float(np.percentile(valid_data, 99.5))
    denom = max(max_val - min_val, 1e-6)

    if not HAS_CUDA:
        return _cpu_colormap(index_data, cmap, min_val, max_val)

    try:
        t = torch.from_numpy(index_data.astype(np.float32)).to(_DEVICE)
        t = torch.clamp(t, min_val, max_val)
        t = (t - min_val) / denom
        indices = (t * 255).to(torch.long).clamp(0, 255)
        lut = _get_cmap_lut(cmap)
        h, w = index_data.shape
        rgb = lut[indices.view(-1)].view(h, w, 3)
        nan_t = torch.from_numpy(nan_mask).to(_DEVICE)
        rgb[nan_t] = 0
        out = rgb.cpu().numpy()
        del t, indices, rgb, nan_t
        _gpu_cleanup()
        return out, min_val, max_val
    except Exception as e:
        print(f"GPU colormap failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu_colormap(index_data, cmap, min_val, max_val)


def detect_sam_gpu(image_data: np.ndarray, target_spectrum: np.ndarray) -> np.ndarray:
    h, w, nb = image_data.shape

    def _cpu():
        pixels = image_data.reshape(-1, nb)
        t_norm = max(np.linalg.norm(target_spectrum), 1e-10)
        p_norms = np.maximum(np.linalg.norm(pixels, axis=1), 1e-10)
        cos_a = np.clip(np.dot(pixels, target_spectrum) / (p_norms * t_norm), -1, 1)
        angles = np.arccos(cos_a).reshape(h, w)
        return float(np.max(angles)) - angles

    if not HAS_CUDA:
        return _cpu()
    try:
        pixels = torch.from_numpy(image_data.reshape(-1, nb)).to(_DEVICE, dtype=torch.float32)
        target = torch.from_numpy(target_spectrum).to(_DEVICE, dtype=torch.float32)
        t_norm = torch.clamp(target.norm(), min=1e-10)
        p_norms = torch.clamp(pixels.norm(dim=1), min=1e-10)
        cos_a = torch.clamp((pixels @ target) / (p_norms * t_norm), -1, 1)
        angles = torch.acos(cos_a)
        result = angles.max() - angles
        out = result.view(h, w).cpu().numpy()
        del pixels, target, p_norms, cos_a, angles, result
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU SAM failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu()


def detect_ace_gpu(image_data: np.ndarray, target_spectrum: np.ndarray,
                   bg_mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    h, w, nb = image_data.shape

    def _cpu():
        pixels = image_data.reshape(-1, nb) - bg_mean
        tc = target_spectrum - bg_mean
        wt = cov_inv @ tc
        num = (pixels @ wt) ** 2
        tt = tc @ wt
        wp = pixels @ cov_inv
        pp = np.sum(wp * pixels, axis=1)
        return np.clip(num / np.maximum(tt * pp, 1e-10), 0, 1).reshape(h, w)

    if not HAS_CUDA:
        return _cpu()
    try:
        # Transfer raw data to GPU, then center on GPU (avoid 2GB CPU allocation)
        pixels = torch.from_numpy(image_data.reshape(-1, nb)).to(_DEVICE, dtype=torch.float32)
        t_bg = torch.from_numpy(bg_mean.astype(np.float32)).to(_DEVICE)
        pixels = pixels - t_bg  # centering on GPU
        tc = torch.from_numpy(target_spectrum.astype(np.float32)).to(_DEVICE) - t_bg
        cinv = torch.from_numpy(cov_inv.astype(np.float32)).to(_DEVICE)
        del t_bg
        wt = cinv @ tc
        num = (pixels @ wt) ** 2
        tt = tc @ wt
        wp = pixels @ cinv
        pp = (wp * pixels).sum(dim=1)
        result = torch.clamp(num / torch.clamp(tt * pp, min=1e-10), 0, 1)
        out = result.view(h, w).cpu().numpy()
        del pixels, tc, cinv, wt, num, wp, pp, result
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU ACE failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu()


def detect_cem_gpu(image_data: np.ndarray, target_spectrum: np.ndarray,
                   cov_inv: np.ndarray) -> np.ndarray:
    h, w, nb = image_data.shape

    def _cpu():
        pixels = image_data.reshape(-1, nb)
        Rd = cov_inv @ target_spectrum
        dRd = max(target_spectrum @ Rd, 1e-10)
        fw = Rd / dRd
        return (pixels @ fw).reshape(h, w)

    if not HAS_CUDA:
        return _cpu()
    try:
        pixels = torch.from_numpy(image_data.reshape(-1, nb)).to(_DEVICE, dtype=torch.float32)
        target = torch.from_numpy(target_spectrum).to(_DEVICE, dtype=torch.float32)
        cinv = torch.from_numpy(cov_inv).to(_DEVICE, dtype=torch.float32)
        Rd = cinv @ target
        dRd = torch.clamp(target @ Rd, min=1e-10)
        fw = Rd / dRd
        result = pixels @ fw
        out = result.view(h, w).cpu().numpy()
        del pixels, target, cinv, Rd, fw, result
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU CEM failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu()


def detect_mf_gpu(image_data: np.ndarray, target_spectrum: np.ndarray,
                  bg_mean: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    h, w, nb = image_data.shape

    def _cpu():
        pixels = image_data.reshape(-1, nb) - bg_mean
        tc = target_spectrum - bg_mean
        Ct = cov_inv @ tc
        tCt = max(tc @ Ct, 1e-10)
        fw = Ct / tCt
        return (pixels @ fw).reshape(h, w)

    if not HAS_CUDA:
        return _cpu()
    try:
        # Transfer raw data to GPU, then center on GPU
        pixels = torch.from_numpy(image_data.reshape(-1, nb)).to(_DEVICE, dtype=torch.float32)
        t_bg = torch.from_numpy(bg_mean.astype(np.float32)).to(_DEVICE)
        pixels = pixels - t_bg
        tc = torch.from_numpy(target_spectrum.astype(np.float32)).to(_DEVICE) - t_bg
        cinv = torch.from_numpy(cov_inv.astype(np.float32)).to(_DEVICE)
        del t_bg
        Ct = cinv @ tc
        tCt = torch.clamp(tc @ Ct, min=1e-10)
        fw = Ct / tCt
        result = pixels @ fw
        out = result.view(h, w).cpu().numpy()
        del pixels, tc, cinv, Ct, fw, result
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU MF failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu()


def fit_background_gpu(bg_spectra: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def _cpu():
        mean = np.mean(bg_spectra, axis=0)
        cov = np.cov(bg_spectra, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        reg = 1e-6 * np.trace(cov) / cov.shape[0]
        cov += reg * np.eye(cov.shape[0])
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov)
        return mean, cov, cov_inv

    if not HAS_CUDA:
        return _cpu()
    try:
        t = torch.from_numpy(bg_spectra.astype(np.float32)).to(_DEVICE)
        mean = t.mean(dim=0)
        centered = t - mean
        n = t.shape[0]
        cov = (centered.T @ centered) / (n - 1)
        reg = 1e-6 * cov.trace() / cov.shape[0]
        cov += reg * torch.eye(cov.shape[0], device=_DEVICE)
        cov_inv = torch.linalg.inv(cov)
        out = mean.cpu().numpy(), cov.cpu().numpy(), cov_inv.cpu().numpy()
        del t, mean, centered, cov, cov_inv
        _gpu_cleanup()
        return out
    except Exception as e:
        print(f"GPU fit_background failed ({e}), using CPU")
        _gpu_cleanup()
        return _cpu()
