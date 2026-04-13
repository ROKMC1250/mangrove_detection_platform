"""
Target-Aware Regularized Oblique Projector + AMF / ACE Detector
for point-prompted target detection.

Pipeline:
    1. Estimate background covariance (with target-pixel exclusion)
    2. Target-aware oblique projection:
       a. Form positive subspace T and negative subspace U
       b. Extract negative-only directions: U_⊥ = (I − T T⁺) U
       c. SVD-truncate U_⊥ for low-rank stability
       d. Build regularized oblique projector on U_⊥:
          P_λ = I − U_⊥r (U_⊥rᵀ R⁻¹ U_⊥r + λI)⁻¹ U_⊥rᵀ R⁻¹
    3. Re-estimate covariance on cleaned data
    4. Score every pixel with AMF or ACE in the cleaned space

This ensures directions shared by positive and negative spectra are never
suppressed, and only negative-specific directions are attenuated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from models.config import CFG


# ===================================================================
# 1.  Background covariance estimation  (multiple methods)
# ===================================================================

def _cov_info(D: int, N: int, R: torch.Tensor, eta: float,
              method: str) -> Dict[str, Any]:
    trace_R = round(float(R.trace().item()), 4)
    try:
        cond_R = float(torch.linalg.cond(R).item())
    except Exception:
        cond_R = float("nan")
    return {"D": D, "N_cov": N, "trace_R": trace_R,
            "eta": eta, "cond_R": cond_R, "cov_method": method}


# ---------- (a) Sample covariance + diagonal loading (default) ------

@torch.no_grad()
def _cov_sample(X: torch.Tensor, eta: float,
                dtype: torch.dtype | None = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if dtype is None:
        dtype = getattr(torch, CFG.cov_dtype)
    X = X.to(dtype)
    N, D = X.shape
    mu_B = X.mean(dim=0)
    X_c = X - mu_B.unsqueeze(0)
    S = (X_c.t() @ X_c) / max(N - 1, 1)
    R = S + eta * torch.eye(D, dtype=dtype, device=X.device)
    R_inv = torch.linalg.inv(R)
    return R_inv, mu_B, _cov_info(D, N, R, eta, "sample")


# ---------- (b) Ledoit-Wolf shrinkage ---------------------------------

@torch.no_grad()
def _cov_ledoit_wolf(X: torch.Tensor, eta: float,
                     dtype: torch.dtype | None = None,
                     ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if dtype is None:
        dtype = getattr(torch, CFG.cov_dtype)
    X = X.to(dtype)
    N, D = X.shape
    mu_B = X.mean(dim=0)
    X_c = X - mu_B.unsqueeze(0)
    S = (X_c.t() @ X_c) / max(N - 1, 1)

    tr_S = S.trace()
    target = (tr_S / D) * torch.eye(D, dtype=dtype, device=X.device)

    # Ledoit-Wolf optimal shrinkage (Oracle Approximating Shrinkage variant)
    S2 = S @ S
    tr_S2 = S2.trace()
    rho_num = ((1.0 - 2.0 / D) * tr_S2 + tr_S * tr_S)
    rho_den = ((N + 1.0 - 2.0 / D) * (tr_S2 - tr_S * tr_S / D))
    if abs(float(rho_den)) < 1e-30:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, float(rho_num / rho_den)))

    R = (1.0 - alpha) * S + alpha * target
    R += eta * torch.eye(D, dtype=dtype, device=X.device)
    R_inv = torch.linalg.inv(R)
    info = _cov_info(D, N, R, eta, "ledoit_wolf")
    info["lw_alpha"] = round(alpha, 4)
    return R_inv, mu_B, info


# ---------- (c) Tyler's M-estimator (robust scatter) ------------------

@torch.no_grad()
def _cov_tyler(X: torch.Tensor, eta: float,
               dtype: torch.dtype | None = None,
               max_iter: int = CFG.tyler_max_iter,
               tol: float = CFG.tyler_tol,
               ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if dtype is None:
        dtype = getattr(torch, CFG.cov_dtype)
    X = X.to(dtype)
    N, D = X.shape
    mu_B = X.mean(dim=0)
    X_c = X - mu_B.unsqueeze(0)

    I_D = torch.eye(D, dtype=dtype, device=X.device)
    M = I_D.clone()

    for it in range(max_iter):
        M_reg = M + eta * I_D
        M_inv = torch.linalg.inv(M_reg)
        dists = (X_c @ M_inv * X_c).sum(dim=1).clamp(min=1e-30)   # (N,)
        weights = D / dists                                         # (N,)
        M_new = (weights.unsqueeze(1) * X_c).t() @ X_c / N
        M_new = M_new * (D / M_new.trace().clamp(min=1e-30))
        delta = (M_new - M).norm() / M.norm().clamp(min=1e-30)
        M = M_new
        if float(delta) < tol:
            break

    R = M + eta * I_D
    R_inv = torch.linalg.inv(R)
    info = _cov_info(D, N, R, eta, "tyler")
    info["tyler_iters"] = it + 1
    return R_inv, mu_B, info


# ---------- (d) Neural Cholesky (gradient-optimised SPD) ---------------

def _cov_neural_cholesky(
    X: torch.Tensor, eta: float,
    dtype: torch.dtype | None = None,
    n_steps: int = CFG.neural_cov_steps,
    lr: float = CFG.neural_cov_lr,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if dtype is None:
        dtype = getattr(torch, CFG.cov_dtype)
    X = X.to(dtype)
    N, D = X.shape
    mu_B = X.mean(dim=0)
    X_c = X - mu_B.unsqueeze(0)
    S = (X_c.t() @ X_c) / max(N - 1, 1)

    I_D = torch.eye(D, dtype=dtype, device=X.device)
    L_init = torch.linalg.cholesky(S + eta * I_D)
    L_raw = L_init.clone().detach().requires_grad_(True)

    optimiser = torch.optim.Adam([L_raw], lr=lr)
    tril_idx = torch.tril_indices(D, D, device=X.device)

    best_loss = float("inf")
    best_L = L_init.clone()

    for step in range(n_steps):
        L = torch.zeros(D, D, dtype=dtype, device=X.device)
        L[tril_idx[0], tril_idx[1]] = L_raw[tril_idx[0], tril_idx[1]]
        diag_idx = torch.arange(D, device=X.device)
        L[diag_idx, diag_idx] = L_raw[diag_idx, diag_idx].abs().clamp(min=1e-6)
        Sigma = L @ L.t()

        # Negative log-likelihood core:  log|Σ| + tr(Σ^{-1} S)
        loss = torch.logdet(Sigma) + torch.linalg.solve(Sigma, S).trace()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            with torch.no_grad():
                best_L = L.detach().clone()

    with torch.no_grad():
        R = best_L @ best_L.t() + eta * I_D
        R_inv = torch.linalg.inv(R)
    info = _cov_info(D, N, R.detach(), eta, "neural_cholesky")
    info["final_loss"] = round(best_loss, 6)
    return R_inv, mu_B.detach(), info


# ---------- (e) Neural Low-rank + Diagonal -----------------------------

def _cov_neural_lowrank(
    X: torch.Tensor, eta: float,
    dtype: torch.dtype | None = None,
    rank: int = CFG.neural_lowrank_rank,
    n_steps: int = CFG.neural_cov_steps,
    lr: float = CFG.neural_cov_lr,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if dtype is None:
        dtype = getattr(torch, CFG.cov_dtype)
    X = X.to(dtype)
    N, D = X.shape
    mu_B = X.mean(dim=0)
    X_c = X - mu_B.unsqueeze(0)
    S = (X_c.t() @ X_c) / max(N - 1, 1)

    I_D = torch.eye(D, dtype=dtype, device=X.device)

    # Initialise from top-r eigenvectors of S
    eigvals, eigvecs = torch.linalg.eigh(S)
    top_r = min(rank, D)
    U_init = eigvecs[:, -top_r:] * eigvals[-top_r:].clamp(min=1e-8).sqrt().unsqueeze(0)
    d_init = S.diag().clamp(min=1e-8).log()

    d_raw = d_init.clone().detach().requires_grad_(True)
    U_raw = U_init.clone().detach().requires_grad_(True)

    optimiser = torch.optim.Adam([d_raw, U_raw], lr=lr)

    best_loss = float("inf")
    best_d, best_U = d_raw.detach().clone(), U_raw.detach().clone()

    for step in range(n_steps):
        D_diag = d_raw.exp()  # softplus-like via exp to guarantee positive
        Sigma = torch.diag(D_diag) + U_raw @ U_raw.t()

        loss = torch.logdet(Sigma) + torch.linalg.solve(Sigma, S).trace()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            with torch.no_grad():
                best_d = d_raw.detach().clone()
                best_U = U_raw.detach().clone()

    with torch.no_grad():
        R = torch.diag(best_d.exp()) + best_U @ best_U.t() + eta * I_D
        R_inv = torch.linalg.inv(R)
    info = _cov_info(D, N, R.detach(), eta, "neural_lowrank")
    info["rank"] = top_r
    info["final_loss"] = round(best_loss, 6)
    return R_inv, mu_B.detach(), info


# ---------- dispatcher -------------------------------------------------

def estimate_covariance(
    X: torch.Tensor,
    eta: float = CFG.eta,
    method: str = "sample",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Unified covariance estimation dispatcher."""
    if method == "ledoit_wolf":
        return _cov_ledoit_wolf(X, eta)
    elif method == "tyler":
        return _cov_tyler(X, eta, **kwargs)
    elif method == "neural_cholesky":
        return _cov_neural_cholesky(X, eta, **kwargs)
    elif method == "neural_lowrank":
        return _cov_neural_lowrank(X, eta, **kwargs)
    else:
        return _cov_sample(X, eta)


@torch.no_grad()
def _compute_R_inv_sqrt(R_inv: torch.Tensor) -> torch.Tensor:
    """R^{-1/2} via eigendecomposition of R^{-1}."""
    eigvals, eigvecs = torch.linalg.eigh(R_inv)
    return eigvecs @ torch.diag(eigvals.clamp(min=1e-12).sqrt()) @ eigvecs.t()


@torch.no_grad()
def _exclude_target_pixels(
    X: torch.Tensor,
    pos_vecs: torch.Tensor,
    R_inv_sqrt: torch.Tensor,
    mu_B: torch.Tensor,
    thresh: float = CFG.target_exclude_thresh,
) -> torch.Tensor:
    """Remove pixels similar to positive clicks in whitened space.

    Whitens all pixels and the positive references, then removes any pixel
    whose cosine similarity to any positive reference exceeds `thresh`.

    Returns the filtered pixel matrix (N', D).
    """
    N, D = X.shape
    dtype, device = R_inv_sqrt.dtype, R_inv_sqrt.device

    X_w = (X.to(dtype) - mu_B.unsqueeze(0)) @ R_inv_sqrt.t()    # (N, D)
    X_w_norm = X_w / X_w.norm(dim=1, keepdim=True).clamp(min=1e-12)

    P_w = (pos_vecs.to(dtype) - mu_B.unsqueeze(0)) @ R_inv_sqrt.t()  # (m, D)
    P_w_norm = P_w / P_w.norm(dim=1, keepdim=True).clamp(min=1e-12)

    cos_sim = X_w_norm @ P_w_norm.t()          # (N, m)
    max_sim = cos_sim.max(dim=1).values         # (N,)

    keep = max_sim < thresh
    return X[keep]


@torch.no_grad()
def _compute_exclude_mask(
    F: torch.Tensor,
    pos_vecs: torch.Tensor,
    R_inv_sqrt: torch.Tensor,
    mu_B: torch.Tensor,
    thresh: float = CFG.target_exclude_thresh,
) -> np.ndarray:
    """Return (H, W) boolean mask: True = pixel excluded from background."""
    H, W, D = F.shape
    dtype, device = R_inv_sqrt.dtype, R_inv_sqrt.device

    X = F.reshape(-1, D).to(dtype)
    X_w = (X - mu_B.unsqueeze(0)) @ R_inv_sqrt.t()
    X_w_norm = X_w / X_w.norm(dim=1, keepdim=True).clamp(min=1e-12)

    P_w = (pos_vecs.to(dtype) - mu_B.unsqueeze(0)) @ R_inv_sqrt.t()
    P_w_norm = P_w / P_w.norm(dim=1, keepdim=True).clamp(min=1e-12)

    cos_sim = X_w_norm @ P_w_norm.t()
    max_sim = cos_sim.max(dim=1).values

    excluded = (max_sim >= thresh).cpu().numpy().reshape(H, W)
    return excluded


# ===================================================================
# 2.  Feature sampling helpers
# ===================================================================

def _sample(F: torch.Tensor, x: int, y: int) -> torch.Tensor:
    """Sample the feature vector at pixel (x=col, y=row), clamped."""
    H, W, _ = F.shape
    y = max(0, min(y, H - 1))
    x = max(0, min(x, W - 1))
    return F[y, x, :]


# ===================================================================
# 3.  Mask generation  (Otsu / FPR / fixed threshold)
# ===================================================================

def _otsu_threshold(s: np.ndarray) -> float:
    lo, hi = float(s.min()), float(s.max())
    if hi - lo < 1e-10:
        return lo
    norm = ((s - lo) / (hi - lo) * 255).astype(np.uint8)
    otsu_val, _ = cv2.threshold(
        norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return lo + (otsu_val / 255.0) * (hi - lo)


def generate_mask(
    score_map: torch.Tensor | np.ndarray,
    *,
    use_otsu: bool = False,
    use_fpr: bool = True,
    fpr: float = CFG.fpr,
    theta: float | None = None,
    return_threshold: bool = False,
) -> np.ndarray | Tuple[np.ndarray, float]:
    """Binary mask from a score map.

    Priority: theta > otsu > fpr > default 0.5.
    Returns (H, W) uint8  (0 or 255).
    """
    s = (
        score_map.detach().cpu().numpy()
        if isinstance(score_map, torch.Tensor)
        else np.asarray(score_map)
    )

    if theta is not None:
        thr = theta
    elif use_otsu:
        thr = _otsu_threshold(s)
    elif use_fpr:
        thr = float(np.quantile(s.ravel().astype(np.float64), 1.0 - fpr))
    else:
        thr = 0.5

    mask = (s >= thr).astype(np.uint8) * 255
    if return_threshold:
        return mask, float(thr)
    return mask


# ===================================================================
# 4.  Target-aware regularized oblique projector
# ===================================================================

@torch.no_grad()
def _build_oblique_projector(
    pos_vecs: List[torch.Tensor],
    neg_vecs: List[torch.Tensor],
    mu_B: torch.Tensor,
    R_inv: torch.Tensor,
    *,
    lam: float = CFG.osp_lambda,
    sv_thresh: float = CFG.osp_sv_thresh,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    r"""Target-aware regularized oblique projector.

    1.  T = [t₁−μ, t₂−μ, …]  (D × m)  — positive subspace
        U = [u₁−μ, u₂−μ, …]  (D × k)  — negative subspace

    2.  Project out the positive subspace from U to get negative-only dirs:
        Q_T = T (TᵀT)⁻¹ Tᵀ            (orthogonal projector onto span(T))
        U_⊥ = (I − Q_T) U              (negative-only component)

    3.  SVD-truncate U_⊥: keep σᵢ ≥ sv_thresh · σ_max  →  U_⊥r  (D × r)

    4.  Regularized oblique projector on negative-only directions:
        P_λ = I − U_⊥r (U_⊥rᵀ R⁻¹ U_⊥r + λ I_r)⁻¹ U_⊥rᵀ R⁻¹

    This never suppresses the positive subspace — only negative-specific
    directions are attenuated.

    Returns (P, info_dict).
    """
    dtype, device = mu_B.dtype, mu_B.device
    D = mu_B.shape[0]

    T = torch.stack([v - mu_B for v in pos_vecs], dim=1).to(dtype)   # (D, m)
    U = torch.stack([v - mu_B for v in neg_vecs], dim=1).to(dtype)   # (D, k)

    # --- Step 2: remove positive component from negative subspace ---
    TtT = T.t() @ T + 1e-10 * torch.eye(T.shape[1], dtype=dtype, device=device)
    Q_T = T @ torch.linalg.solve(TtT, T.t())       # (D, D) projector onto span(T)
    U_perp = U - Q_T @ U                             # (D, k) negative-only

    # Measure how much overlap was removed
    U_norms = U.norm(dim=0)                           # (k,)
    Up_norms = U_perp.norm(dim=0)                     # (k,)
    overlap_ratios = (1.0 - Up_norms / U_norms.clamp(min=1e-12)).tolist()

    # --- Step 3: SVD truncation for low-rank stability ---
    Uf, S_vals, Vf = torch.linalg.svd(U_perp, full_matrices=False)
    s_max = max(S_vals[0].item(), 1e-12)
    keep = S_vals >= sv_thresh * s_max
    r = max(int(keep.sum().item()), 1)

    # Guard: if all singular values are near-zero, U was entirely in span(T)
    if S_vals[0].item() < 1e-10:
        info = {
            "rank_raw": U.shape[1], "rank_kept": 0,
            "sv_all": S_vals.tolist(), "sv_max": 0.0, "lambda": lam,
            "overlap_ratios": overlap_ratios,
            "msg": "negatives fully overlap with positives — no projection applied",
        }
        return torch.eye(D, dtype=dtype, device=device), info

    U_r = Uf[:, :r] * S_vals[:r].unsqueeze(0)        # (D, r) scaled basis

    # --- Step 4: regularized oblique projector ---
    RiU = R_inv @ U_r                                  # (D, r)
    K = U_r.t() @ RiU                                  # (r, r)
    K += lam * torch.eye(r, dtype=dtype, device=device)
    K_inv_UtRi = torch.linalg.solve(K, RiU.t())       # (r, D)
    P = torch.eye(D, dtype=dtype, device=device) - U_r @ K_inv_UtRi

    info = {
        "rank_raw": U.shape[1],
        "rank_kept": r,
        "sv_max": round(s_max, 6),
        "sv_all": S_vals.tolist(),
        "lambda": lam,
        "overlap_ratios": overlap_ratios,
    }
    return P, info


# ===================================================================
# 5.  AMF / ACE scoring  (works on raw or OSP-cleaned data)
# ===================================================================

@torch.no_grad()
def _score_amf(
    F: torch.Tensor,
    target: torch.Tensor,
    R_inv: torch.Tensor,
    mu_B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Adaptive Matched Filter score map.

    s(x) = (x−μ)ᵀ R⁻¹ (t−μ)  /  √[(t−μ)ᵀ R⁻¹ (t−μ)]

    Returns (score_map (H,W), w_effective (D,)).
    """
    H, W, D = F.shape
    dtype, device = R_inv.dtype, R_inv.device

    d = (target - mu_B).to(dtype=dtype, device=device)        # (D,)
    Rd = R_inv @ d                                              # (D,)
    denom = torch.sqrt((d @ Rd).clamp(min=1e-30))
    w = Rd / denom                                              # (D,)

    X = F.reshape(-1, D).to(dtype=dtype, device=device)
    scores = (X - mu_B.unsqueeze(0)) @ w                       # (N,)
    return scores.float().reshape(H, W), w


@torch.no_grad()
def _score_ace(
    F: torch.Tensor,
    target: torch.Tensor,
    R_inv: torch.Tensor,
    mu_B: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Adaptive Cosine Estimator (ACE) score map.

    s(x) = [(x−μ)ᵀ R⁻¹ (t−μ)]²  /  {[(t−μ)ᵀ R⁻¹ (t−μ)] · [(x−μ)ᵀ R⁻¹ (x−μ)]}

    This is the squared cosine similarity in whitened space.
    Returns (score_map (H,W), w_effective (D,)).

    w_effective is the AMF weight (for state reporting); the actual ACE score
    is a nonlinear function (squared cosine), not a simple linear projection.
    """
    H, W, D = F.shape
    dtype, device = R_inv.dtype, R_inv.device

    d = (target - mu_B).to(dtype=dtype, device=device)         # (D,)
    Rd = R_inv @ d                                              # (D,)
    dRd = (d @ Rd).clamp(min=1e-30)

    X = F.reshape(-1, D).to(dtype=dtype, device=device)
    Xc = X - mu_B.unsqueeze(0)                                 # (N, D)
    num = (Xc @ Rd)                                             # (N,)
    xRx = (Xc @ R_inv * Xc).sum(dim=1).clamp(min=1e-30)       # (N,)
    scores = (num * num) / (dRd * xRx)                          # (N,)

    w_eff = Rd / torch.sqrt(dRd)
    return scores.float().reshape(H, W), w_eff


# ===================================================================
# 6.  Oblique projection + AMF/ACE  (full pipeline)
# ===================================================================

@torch.no_grad()
def _apply_oblique_and_score(
    F: torch.Tensor,
    pos_vecs: List[torch.Tensor],
    neg_vecs: List[torch.Tensor],
    R_inv: torch.Tensor,
    mu_B: torch.Tensor,
    *,
    score_fn: str = "amf",
    eta: float = CFG.eta,
    cov_method: str = "sample",
    osp_lambda: float = CFG.osp_lambda,
    osp_sv_thresh: float = CFG.osp_sv_thresh,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    r"""Regularized oblique projection → re-estimate background → AMF/ACE.

    When neg_vecs is empty, skips projection and scores directly.
    Returns (score_map, w_effective, oblique_info_dict).
    """
    H, W, D = F.shape
    dtype, device = R_inv.dtype, R_inv.device

    target = torch.stack(pos_vecs).mean(dim=0)
    oblique_info: Dict[str, Any] = {"applied": False, "n_neg": 0}

    if len(neg_vecs) > 0:
        P, proj_info = _build_oblique_projector(
            pos_vecs, neg_vecs, mu_B, R_inv,
            lam=osp_lambda, sv_thresh=osp_sv_thresh,
        )
        oblique_info["applied"] = True
        oblique_info["n_neg"] = len(neg_vecs)
        oblique_info["n_pos"] = len(pos_vecs)
        oblique_info.update(proj_info)

        # Monitoring: projected spectra + preservation/suppression metrics
        pos_proj = [(P @ (v - mu_B) + mu_B) for v in pos_vecs]
        neg_proj = [(P @ (v - mu_B) + mu_B) for v in neg_vecs]

        oblique_info["pos_projected"] = [
            v.detach().cpu().float().tolist() for v in pos_proj]
        oblique_info["neg_projected"] = [
            v.detach().cpu().float().tolist() for v in neg_proj]

        # Per-spectrum energy preservation: ‖proj - μ‖ / ‖orig - μ‖
        oblique_info["pos_preservation"] = [
            round(float(
                (pp - mu_B).norm() / (pv - mu_B).norm().clamp(min=1e-12)
            ), 4) for pp, pv in zip(pos_proj, pos_vecs)
        ]
        oblique_info["neg_suppression"] = [
            round(float(
                1.0 - (np_ - mu_B).norm() / (nv - mu_B).norm().clamp(min=1e-12)
            ), 4) for np_, nv in zip(neg_proj, neg_vecs)
        ]

        X = F.reshape(-1, D).to(dtype=dtype, device=device)
        Xc = X - mu_B.unsqueeze(0)
        X_clean = Xc @ P.t() + mu_B.unsqueeze(0)
        F = X_clean.reshape(H, W, D)

        target = P @ (target - mu_B) + mu_B

        for i, v in enumerate(pos_vecs):
            pos_vecs[i] = P @ (v - mu_B) + mu_B

        N = X_clean.shape[0]
        max_cov = CFG.max_cov_samples
        if max_cov > 0 and N > max_cov:
            idx = torch.randperm(N, device=device)[:max_cov]
            X_cov = X_clean[idx]
        else:
            X_cov = X_clean

        R_inv, mu_B, cov_info_clean = estimate_covariance(
            X_cov, eta=eta, method=cov_method)
        oblique_info["cov_info_clean"] = cov_info_clean

    scorer = _score_ace if score_fn == "ace" else _score_amf
    score_map, w_eff = scorer(F, target, R_inv, mu_B)
    return score_map, w_eff, oblique_info


# ===================================================================
# 7.  detect_step  — public API
# ===================================================================

def detect_step(
    F: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    prev_state: Optional[Dict[str, Any]] = None,
    *,
    scoring: str = "osp_ace",
    cov_method: str = "sample",
    eta: float = CFG.eta,
    max_cov_samples: int = CFG.max_cov_samples,
    use_otsu: bool = False,
    use_fpr: bool = True,
    fpr: float = CFG.fpr,
    threshold: Optional[float] = None,
    eps: float = CFG.eps,
    target_exclude_thresh: float = CFG.target_exclude_thresh,
    osp_lambda: float = CFG.osp_lambda,
    external_exclude_mask: Optional[np.ndarray] = None,
    **_kwargs,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """One interaction step — Regularized Oblique + AMF / ACE detection.

    Parameters
    ----------
    scoring : 'osp_ace' | 'osp_amf'
        osp_ace — oblique projection on negatives, then ACE on cleaned data
        osp_amf — oblique projection on negatives, then AMF on cleaned data
    osp_lambda : float
        Regularization strength for the oblique projector.
        0 = hard null, larger = softer attenuation.

    Flow
    ----
    1. Initial covariance from all pixels  →  R0, mu0
    2. Whiten positive clicks with R0^{-1/2}
    3. Exclude target-like pixels (cosine sim > thresh in whitened space)
    4. Re-estimate covariance from filtered background  →  R, mu_B
    5. Regularized oblique projection on negative subspace (if negatives given)
    6. Re-estimate covariance on cleaned data
    7. Score with AMF or ACE  →  score_map, mask
    """
    n_pos = len(pos_points)
    n_neg = len(neg_points)
    dtype = getattr(torch, CFG.cov_dtype)

    H_, W_, D = F.shape
    X_all = F.reshape(-1, D).to(dtype)

    # --- Subsample for covariance if needed ---
    N = X_all.shape[0]
    if max_cov_samples > 0 and N > max_cov_samples:
        idx = torch.randperm(N, device=X_all.device)[:max_cov_samples]
        X_cov = X_all[idx]
    else:
        X_cov = X_all

    # Step 1: initial covariance (sample, for exclusion mask computation)
    R_inv_0, mu_B_0, _ = _cov_sample(X_cov, eta=eta)
    R_inv_sqrt_0 = _compute_R_inv_sqrt(R_inv_0)

    # Step 2-3: exclude target-like pixels for clean background covariance
    exclude_mask_hw: Optional[np.ndarray] = None
    N_excluded = 0

    if external_exclude_mask is not None:
        # --- SAM-guided exclusion: use the provided foreground mask directly ---
        H_, W_, _ = F.shape
        # Resize to feature-map resolution if needed
        if external_exclude_mask.shape != (H_, W_):
            ext_mask = cv2.resize(
                external_exclude_mask.astype(np.uint8), (W_, H_),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            ext_mask = external_exclude_mask.astype(bool)

        mask_flat = torch.from_numpy(ext_mask.reshape(-1)).to(X_all.device)
        X_bg_all = X_all[~mask_flat]
        N_excluded = int(mask_flat.sum().item())
        exclude_mask_hw = ext_mask

        # Subsample background if still large
        N_bg = X_bg_all.shape[0]
        if max_cov_samples > 0 and N_bg > max_cov_samples:
            idx = torch.randperm(N_bg, device=X_bg_all.device)[:max_cov_samples]
            X_bg = X_bg_all[idx]
        else:
            X_bg = X_bg_all

        if X_bg.shape[0] > D:
            R_inv, mu_B, cov_info = estimate_covariance(
                X_bg, eta=eta, method=cov_method)
        else:
            R_inv, mu_B, cov_info = R_inv_0, mu_B_0, {}
            N_excluded = 0
            exclude_mask_hw = None

    elif n_pos > 0 and target_exclude_thresh < 1.0:
        # --- Original whitening-based exclusion ---
        pos_vecs = torch.stack([
            _sample(F, x, y).to(dtype) for x, y in pos_points
        ])
        X_bg = _exclude_target_pixels(
            X_cov, pos_vecs, R_inv_sqrt_0, mu_B_0,
            thresh=target_exclude_thresh,
        )
        N_excluded = X_cov.shape[0] - X_bg.shape[0]

        exclude_mask_hw = _compute_exclude_mask(
            F, pos_vecs, R_inv_sqrt_0, mu_B_0,
            thresh=target_exclude_thresh,
        )

        # Step 4: re-estimate from filtered background
        if X_bg.shape[0] > D:
            R_inv, mu_B, cov_info = estimate_covariance(
                X_bg, eta=eta, method=cov_method)
        else:
            R_inv, mu_B, cov_info = R_inv_0, mu_B_0, {}
            N_excluded = 0
            exclude_mask_hw = None
    else:
        R_inv, mu_B, cov_info = estimate_covariance(
            X_cov, eta=eta, method=cov_method)

    cov_info["N_excluded"] = N_excluded

    # --- Collect spectra ---
    dtype_cov = R_inv.dtype
    device_cov = R_inv.device

    pos_spectra_raw: List[List[float]] = []
    neg_spectra_raw: List[List[float]] = []
    pos_vecs_t: List[torch.Tensor] = []
    neg_vecs_t: List[torch.Tensor] = []

    for x, y in pos_points:
        s_vec = _sample(F, x, y)
        pos_spectra_raw.append(s_vec.detach().cpu().float().tolist())
        pos_vecs_t.append(s_vec.to(dtype=dtype_cov, device=device_cov))
    for x, y in neg_points:
        s_vec = _sample(F, x, y)
        neg_spectra_raw.append(s_vec.detach().cpu().float().tolist())
        neg_vecs_t.append(s_vec.to(dtype=dtype_cov, device=device_cov))

    # =================================================================
    # Scoring:  Oblique projection (if negatives) → AMF or ACE
    # =================================================================
    score_fn = "ace" if "ace" in scoring else "amf"
    score_map, w_vec, osp_info = _apply_oblique_and_score(
        F, pos_vecs_t, neg_vecs_t,
        R_inv, mu_B,
        score_fn=score_fn,
        eta=eta,
        cov_method=cov_method,
        osp_lambda=osp_lambda,
    )

    mask, used_threshold = generate_mask(
        score_map,
        use_otsu=use_otsu,
        use_fpr=use_fpr,
        fpr=fpr,
        theta=threshold,
        return_threshold=True,
    )

    mu_list = mu_B.detach().cpu().float().tolist()

    # --- Score histogram for visualisation ---
    score_hist: Dict[str, Any] = {}
    s_np = score_map.detach().cpu().float()
    n_bins = 80
    lo_h = float(s_np.min())
    hi_h = float(s_np.max())
    if hi_h - lo_h > 1e-12:
        hist_counts = torch.histc(s_np.flatten(), bins=n_bins, min=lo_h, max=hi_h)
        bin_width = (hi_h - lo_h) / n_bins
        hist_centers = [lo_h + (i + 0.5) * bin_width for i in range(n_bins)]
        score_hist = {
            "hist_counts": hist_counts.tolist(),
            "hist_centers": hist_centers,
        }

    state: Dict[str, Any] = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "scoring": scoring,
        "cov_info": cov_info,
        "osp_info": osp_info,
        "mu_B": mu_list,
        "pos_spectra_raw": pos_spectra_raw,
        "neg_spectra_raw": neg_spectra_raw,
        "score_hist": score_hist,
        "exclude_mask": exclude_mask_hw,
        "threshold_used": float(used_threshold),
    }
    return score_map, mask, state
