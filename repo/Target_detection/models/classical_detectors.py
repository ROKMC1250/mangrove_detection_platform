"""
Classical spectral target detection algorithms.

All functions operate on a multispectral image cube (H, W, B) and a target
spectrum (B,), returning a per-pixel score map (H, W).

Algorithms:
    - MF   : Matched Filter (Kelly/Reed-Yu)
    - SAM  : Spectral Angle Mapper
    - ACE  : Adaptive Cosine/Coherence Estimator
    - CEM  : Constrained Energy Minimization

Background statistics (mean, covariance inverse) can be precomputed and
passed in to avoid redundant work across detectors.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def estimate_background(
    cube: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eta: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate background mean, covariance inverse, and raw covariance.

    Parameters
    ----------
    cube : (H, W, B) array
    mask : (H, W) array, optional – nonzero pixels are target; zero is background
    eta  : regularisation added to diagonal of covariance

    Returns
    -------
    mu    : (B,)  background mean
    R_inv : (B,B) inverse of regularised covariance
    R     : (B,B) regularised covariance
    """
    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float64)

    if mask is not None:
        bg_idx = mask.ravel() == 0
        X = X[bg_idx]

    mu = X.mean(axis=0)
    X_c = X - mu
    N = X_c.shape[0]
    R = X_c.T @ X_c / max(N - 1, 1) + eta * np.eye(B, dtype=np.float64)
    R_inv = np.linalg.inv(R)
    return mu.astype(np.float64), R_inv, R


def detect_mf(
    cube: np.ndarray,
    target: np.ndarray,
    mu: Optional[np.ndarray] = None,
    R_inv: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    eta: float = 1e-6,
) -> np.ndarray:
    """Matched Filter detector.

    Parameters
    ----------
    cube   : (H, W, B)
    target : (B,)
    mu, R_inv : precomputed background stats (optional)
    mask   : (H, W) optional
    eta    : regularisation

    Returns
    -------
    scores : (H, W) float32
    """
    H, W, B = cube.shape
    if mu is None or R_inv is None:
        mu, R_inv, _ = estimate_background(cube, mask, eta)

    t = (target - mu).astype(np.float64)
    Rt = R_inv @ t
    denom = np.sqrt(np.clip(t @ Rt, 1e-30, None))

    X = cube.reshape(-1, B).astype(np.float64) - mu
    scores = (X @ Rt) / denom
    return scores.reshape(H, W).astype(np.float32)


def detect_sam(
    cube: np.ndarray,
    target: np.ndarray,
    **_kwargs,
) -> np.ndarray:
    """Spectral Angle Mapper.

    Returns a z-score-like measure: (mean_angle - angle) / std_angle,
    so higher values indicate better match to the target.

    Parameters
    ----------
    cube   : (H, W, B)
    target : (B,)

    Returns
    -------
    scores : (H, W) float32
    """
    H, W, B = cube.shape
    t = target.astype(np.float64)
    t_norm = np.linalg.norm(t)
    if t_norm < 1e-30:
        return np.zeros((H, W), dtype=np.float32)

    X = cube.reshape(-1, B).astype(np.float64)
    x_norm = np.linalg.norm(X, axis=1, keepdims=True)
    x_norm = np.clip(x_norm, 1e-30, None)

    cos_theta = np.clip(X @ t / (x_norm.ravel() * t_norm), -1.0, 1.0)
    angle = np.arccos(cos_theta)

    mu_angle = angle.mean()
    std_angle = angle.std()
    if std_angle < 1e-30:
        std_angle = 1.0

    scores = (mu_angle - angle) / std_angle
    return scores.reshape(H, W).astype(np.float32)


def detect_ace(
    cube: np.ndarray,
    target: np.ndarray,
    mu: Optional[np.ndarray] = None,
    R_inv: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    eta: float = 1e-6,
) -> np.ndarray:
    """Adaptive Cosine/Coherence Estimator.

    Parameters
    ----------
    cube   : (H, W, B)
    target : (B,)
    mu, R_inv : precomputed background stats (optional)
    mask   : (H, W) optional
    eta    : regularisation

    Returns
    -------
    scores : (H, W) float32
    """
    H, W, B = cube.shape
    if mu is None or R_inv is None:
        mu, R_inv, _ = estimate_background(cube, mask, eta)

    t = (target - mu).astype(np.float64)
    Rt = R_inv @ t
    tRt = t @ Rt
    if tRt < 1e-30:
        return np.zeros((H, W), dtype=np.float32)

    X = cube.reshape(-1, B).astype(np.float64) - mu
    xRt = X @ Rt
    xRx = np.einsum('ni,ij,nj->n', X, R_inv, X)
    xRx = np.clip(xRx, 1e-30, None)

    scores = xRt ** 2 / (xRx * tRt)
    return scores.reshape(H, W).astype(np.float32)


def detect_cem(
    cube: np.ndarray,
    target: np.ndarray,
    mu: Optional[np.ndarray] = None,
    R_inv: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    eta: float = 1e-6,
) -> np.ndarray:
    """Constrained Energy Minimization.

    Uses autocorrelation matrix (not centred covariance) so *mu* and *R_inv*
    from ``estimate_background`` are **not** reused here.

    Parameters
    ----------
    cube   : (H, W, B)
    target : (B,)
    mask   : (H, W) optional
    eta    : regularisation

    Returns
    -------
    scores : (H, W) float32
    """
    H, W, B = cube.shape
    X = cube.reshape(-1, B).astype(np.float64)

    if mask is not None:
        bg_idx = mask.ravel() == 0
        X_bg = X[bg_idx]
    else:
        X_bg = X

    # Autocorrelation matrix (not mean-centred)
    N = X_bg.shape[0]
    R_corr = X_bg.T @ X_bg / max(N - 1, 1) + eta * np.eye(B, dtype=np.float64)
    R_corr_inv = np.linalg.inv(R_corr)

    d = target.astype(np.float64)
    Rd = R_corr_inv @ d
    dRd = d @ Rd
    if dRd < 1e-30:
        return np.zeros((H, W), dtype=np.float32)

    w = Rd / dRd
    scores = X @ w
    return scores.reshape(H, W).astype(np.float32)


ALL_DETECTORS = {
    "MF": detect_mf,
    "SAM": detect_sam,
    "ACE": detect_ace,
    "CEM": detect_cem,
}


def run_all(
    cube: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    eta: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Run all detectors and return a dict of score maps."""
    mu, R_inv, _ = estimate_background(cube, mask, eta)
    results = {}
    for name, fn in ALL_DETECTORS.items():
        if name == "SAM":
            results[name] = fn(cube, target)
        elif name == "CEM":
            results[name] = fn(cube, target, mask=mask, eta=eta)
        else:
            results[name] = fn(cube, target, mu=mu, R_inv=R_inv)
    return results
