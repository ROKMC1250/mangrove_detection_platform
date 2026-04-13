"""
Multispectral-to-3-band conversion for feeding imagery to SAM / SAM2.

SAM expects (H, W, 3) uint8 RGB input. When the source image has >3 bands
(e.g., 4-band Landsat, 8-band WorldView, 10-band Sentinel-2), we need a
principled reduction to 3 channels.

Three strategies:
    1. PCA   - Top-3 principal components (captures most variance)
    2. RGB   - Select dataset-specific R, G, B band indices
    3. Random - Pick 3 random bands (ablation baseline)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# Known RGB band indices for common sensors
RGB_BAND_MAP: Dict[str, Tuple[int, int, int]] = {
    "landsat8":    (2, 1, 0),
    "38-cloud":    (0, 1, 2),
    "worldview3":  (4, 2, 1),
    "spacenet2":   (4, 2, 1),
    "sentinel2":   (3, 2, 1),
    "dstl":        (4, 2, 1),
    "rit18":       (2, 1, 0),
    "default":     (0, 1, 2),
}


def _percentile_stretch(arr: np.ndarray, lo: float = 2, hi: float = 98) -> np.ndarray:
    """Per-channel percentile stretch to [0, 255] uint8."""
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        p_lo, p_hi = np.percentile(arr[:, :, c], [lo, hi])
        if p_hi - p_lo < 1e-10:
            p_hi = p_lo + 1
        out[:, :, c] = np.clip((arr[:, :, c] - p_lo) / (p_hi - p_lo), 0, 1)
    return (out * 255).astype(np.uint8)


def convert_pca(cube: np.ndarray, seed: int = 42) -> np.ndarray:
    """Reduce to 3 channels via PCA on the image pixels."""
    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float64)
    rng = np.random.RandomState(seed)
    n_fit = min(X.shape[0], 50000)
    idx = rng.choice(X.shape[0], n_fit, replace=False)
    X_fit = X[idx]
    mu = X_fit.mean(axis=0)
    X_c = X_fit - mu
    cov = (X_c.T @ X_c) / max(n_fit - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    top3 = eigvecs[:, -3:][:, ::-1]
    X_proj = (X - mu) @ top3
    rgb = X_proj.reshape(H, W, 3).astype(np.float32)
    return _percentile_stretch(rgb)


def convert_rgb(cube: np.ndarray, sensor: str = "default") -> np.ndarray:
    """Select R, G, B bands by sensor-specific indices."""
    H, W, B = cube.shape
    r_idx, g_idx, b_idx = RGB_BAND_MAP.get(sensor, RGB_BAND_MAP["default"])
    r_idx = min(r_idx, B - 1)
    g_idx = min(g_idx, B - 1)
    b_idx = min(b_idx, B - 1)
    rgb = np.stack([cube[:, :, r_idx],
                    cube[:, :, g_idx],
                    cube[:, :, b_idx]], axis=-1).astype(np.float32)
    return _percentile_stretch(rgb)


def convert_random(cube: np.ndarray, seed: int = 42) -> np.ndarray:
    """Pick 3 random bands."""
    H, W, B = cube.shape
    rng = np.random.RandomState(seed)
    indices = sorted(rng.choice(B, 3, replace=False))
    rgb = cube[:, :, indices].astype(np.float32)
    return _percentile_stretch(rgb)


def convert_to_rgb(
    cube: np.ndarray,
    method: str = "pca",
    sensor: str = "default",
    seed: int = 42,
) -> np.ndarray:
    """Convert multispectral cube to (H, W, 3) uint8 for SAM."""
    H, W, B = cube.shape
    if B == 3:
        return _percentile_stretch(cube.astype(np.float32))
    if B < 3:
        padded = np.zeros((H, W, 3), dtype=cube.dtype)
        padded[:, :, :B] = cube
        return _percentile_stretch(padded.astype(np.float32))
    if method == "pca":
        return convert_pca(cube, seed=seed)
    elif method == "rgb":
        return convert_rgb(cube, sensor=sensor)
    elif method == "random":
        return convert_random(cube, seed=seed)
    else:
        raise ValueError(f"Unknown band conversion method: {method}")


def get_sensor_for_dataset(dataset_name: str) -> str:
    """Map dataset name to sensor key for RGB band selection."""
    mapping = {
        "38-cloud": "38-cloud",
        "spacenet2": "spacenet2",
        "sentinel2": "sentinel2",
        "marida": "sentinel2",
        "dstl": "dstl",
        "rit": "rit18",
        "campus": "default",
        "abu": "default",
        "sandiego": "default",
    }
    for key, sensor in mapping.items():
        if key in dataset_name.lower():
            return sensor
    return "default"
