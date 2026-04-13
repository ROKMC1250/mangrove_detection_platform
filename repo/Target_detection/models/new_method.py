"""new_method: positive-guided soft weighting + direct AMF/ACE."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from models.config import CFG
from models.osp_amf import (
    _compute_R_inv_sqrt,
    _compute_exclude_mask,
    _exclude_target_pixels,
    _sample,
    _score_ace,
    _score_amf,
    estimate_covariance,
    generate_mask,
)


_MAX_RINV_VIS_DIM = 64
_PAIR_CHUNK = 4096
_MAX_PAIR_BASE = 64
_MAX_SELECTED_DIM = 256


def _get_compute_device(src_device: torch.device) -> torch.device:
    """Prefer GPU for expansion/covariance/scoring throughput."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return src_device


def _cov_dtype_for_device(device: torch.device) -> torch.dtype:
    """Use fp32 on GPU for speed, configured dtype on CPU."""
    if device.type == "cuda":
        return torch.float32
    return getattr(torch, CFG.cov_dtype)


def _cuda_mem_stats(device: torch.device) -> Dict[str, float]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    didx = device.index if device.index is not None else torch.cuda.current_device()
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(didx) / (1024.0 * 1024.0)),
        "reserved_mb": float(torch.cuda.memory_reserved(didx) / (1024.0 * 1024.0)),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(didx) / (1024.0 * 1024.0)),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(didx) / (1024.0 * 1024.0)),
    }


def _feature_label(meta: Dict[str, Any]) -> str:
    t = str(meta.get("type", "?"))
    i = meta.get("i")
    j = meta.get("j")
    if t == "identity":
        return f"x[{i}]"
    if t == "diff":
        return f"x[{i}] - x[{j}]"
    if t == "normdiff":
        return f"(x[{i}] - x[{j}])/(x[{i}] + x[{j}])"
    if t == "ratio":
        return f"x[{i}] / x[{j}]"
    if t == "product":
        return f"x[{i}] * x[{j}]"
    if t == "log":
        return f"log(x[{i}])"
    if t == "square":
        return f"x[{i}]^2"
    return str(meta)


def _feature_types(feature_meta: List[Dict[str, Any]]) -> List[str]:
    return [str(meta.get("type", "")) for meta in feature_meta]


def _feature_type_mask(feature_meta: List[Dict[str, Any]], feature_type: str) -> List[bool]:
    return [str(meta.get("type", "")) == feature_type for meta in feature_meta]


def _summary_stats(x: torch.Tensor) -> Dict[str, float]:
    x_f = x.detach().float()
    if x_f.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "mean": float(x_f.mean().item()),
        "std": float(x_f.std(unbiased=False).item()),
        "min": float(x_f.min().item()),
        "max": float(x_f.max().item()),
    }


def _summarize_norm_stats(norm_stats: Dict[str, Any]) -> Dict[str, Any]:
    center = norm_stats.get("center")
    scale = norm_stats.get("scale")
    if not isinstance(center, torch.Tensor) or not isinstance(scale, torch.Tensor):
        return {"method": str(norm_stats.get("method", ""))}
    return {
        "method": str(norm_stats.get("method", "")),
        "center": _summary_stats(center),
        "scale": _summary_stats(scale),
    }


def _try_condition_number(mat: torch.Tensor) -> float:
    try:
        return float(torch.linalg.cond(mat.to(torch.float64)).item())
    except Exception:
        return float("nan")


def _pair_base_indices(
    n_channels: int,
    *,
    max_base: int,
    device: torch.device,
) -> torch.Tensor:
    if n_channels <= 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if max_base <= 0 or n_channels <= max_base:
        return torch.arange(n_channels, dtype=torch.long, device=device)

    idx = torch.linspace(0, n_channels - 1, steps=max_base, device=device)
    idx = idx.round().to(torch.long)
    idx = torch.unique(idx, sorted=True)
    if idx.numel() == max_base:
        return idx

    mask = torch.ones(n_channels, dtype=torch.bool, device=device)
    mask[idx] = False
    extra = torch.arange(n_channels, dtype=torch.long, device=device)[mask]
    need = max_base - int(idx.numel())
    idx = torch.cat([idx, extra[:need]], dim=0)
    return torch.sort(idx).values


@torch.no_grad()
def build_feature_bank(
    F: torch.Tensor,
    *,
    eps: float = 1e-6,
    include_identity: bool = True,
    include_diff: bool = True,
    include_normdiff: bool = False,
    include_ratio: bool = False,
    include_product: bool = False,
    include_log: bool = False,
    include_square: bool = False,
    clip_ratio: float = 10.0,
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Build engineered feature bank from (H, W, D).

    Raw identity channels are always available by default.
    Pairwise feature families use at most `_MAX_PAIR_BASE` base channels to
    bound combinatorial growth.
    """
    if F.ndim != 3:
        raise ValueError(f"F must be (H,W,D), got shape={tuple(F.shape)}")

    Ff = F.to(torch.float32)
    H, W, D = Ff.shape

    feats: List[torch.Tensor] = []
    feature_meta: List[Dict[str, Any]] = []

    if include_identity:
        feats.append(Ff)
        feature_meta.extend({"type": "identity", "i": int(i)} for i in range(D))

    pairwise_enabled = include_diff or include_normdiff or include_ratio or include_product
    if pairwise_enabled and D >= 2:
        pair_base = _pair_base_indices(D, max_base=_MAX_PAIR_BASE, device=Ff.device)
        if pair_base.numel() >= 2:
            i_loc, j_loc = torch.triu_indices(
                int(pair_base.numel()),
                int(pair_base.numel()),
                offset=1,
                device=Ff.device,
            )
            i_und = pair_base.index_select(0, i_loc)
            j_und = pair_base.index_select(0, j_loc)
        else:
            i_und = torch.empty(0, dtype=torch.long, device=Ff.device)
            j_und = torch.empty(0, dtype=torch.long, device=Ff.device)
    else:
        pair_base = torch.empty(0, dtype=torch.long, device=Ff.device)
        i_und = torch.empty(0, dtype=torch.long, device=Ff.device)
        j_und = torch.empty(0, dtype=torch.long, device=Ff.device)

    if include_diff and i_und.numel() > 0:
        n_pairs = int(i_und.numel())
        out_diff = torch.empty((H, W, n_pairs), dtype=Ff.dtype, device=Ff.device)
        for s in range(0, n_pairs, _PAIR_CHUNK):
            e = min(s + _PAIR_CHUNK, n_pairs)
            ri = i_und[s:e]
            rj = j_und[s:e]
            out_diff[:, :, s:e] = Ff.index_select(2, ri) - Ff.index_select(2, rj)
            feature_meta.extend(
                {"type": "diff", "i": int(i), "j": int(j)}
                for i, j in zip(ri.tolist(), rj.tolist())
            )
        feats.append(out_diff)

    if include_normdiff and i_und.numel() > 0:
        n_pairs = int(i_und.numel())
        out_nd = torch.empty((H, W, n_pairs), dtype=Ff.dtype, device=Ff.device)
        for s in range(0, n_pairs, _PAIR_CHUNK):
            e = min(s + _PAIR_CHUNK, n_pairs)
            ri = i_und[s:e]
            rj = j_und[s:e]
            Xi = Ff.index_select(2, ri)
            Xj = Ff.index_select(2, rj)
            out_nd[:, :, s:e] = (Xi - Xj) / (Xi + Xj + float(eps))
            feature_meta.extend(
                {"type": "normdiff", "i": int(i), "j": int(j)}
                for i, j in zip(ri.tolist(), rj.tolist())
            )
        feats.append(out_nd)

    if include_ratio and pair_base.numel() >= 2:
        all_idx = pair_base
        ri_all = all_idx.repeat_interleave(int(all_idx.numel()))
        rj_all = all_idx.repeat(int(all_idx.numel()))
        keep = ri_all != rj_all
        ri_all = ri_all[keep]
        rj_all = rj_all[keep]

        n_ratio = int(ri_all.numel())
        out_ratio = torch.empty((H, W, n_ratio), dtype=Ff.dtype, device=Ff.device)
        for s in range(0, n_ratio, _PAIR_CHUNK):
            e = min(s + _PAIR_CHUNK, n_ratio)
            ri = ri_all[s:e]
            rj = rj_all[s:e]
            Xi = Ff.index_select(2, ri)
            Xj = Ff.index_select(2, rj)
            out_ratio[:, :, s:e] = torch.clamp(
                Xi / (Xj + float(eps)),
                -float(clip_ratio),
                float(clip_ratio),
            )
            feature_meta.extend(
                {"type": "ratio", "i": int(i), "j": int(j)}
                for i, j in zip(ri.tolist(), rj.tolist())
            )
        feats.append(out_ratio)

    if include_product and i_und.numel() > 0:
        n_pairs = int(i_und.numel())
        out_prod = torch.empty((H, W, n_pairs), dtype=Ff.dtype, device=Ff.device)
        for s in range(0, n_pairs, _PAIR_CHUNK):
            e = min(s + _PAIR_CHUNK, n_pairs)
            ri = i_und[s:e]
            rj = j_und[s:e]
            out_prod[:, :, s:e] = Ff.index_select(2, ri) * Ff.index_select(2, rj)
            feature_meta.extend(
                {"type": "product", "i": int(i), "j": int(j)}
                for i, j in zip(ri.tolist(), rj.tolist())
            )
        feats.append(out_prod)

    if include_log:
        feats.append(torch.log(torch.clamp(Ff, min=float(eps))))
        feature_meta.extend({"type": "log", "i": int(i)} for i in range(D))

    if include_square:
        feats.append(Ff * Ff)
        feature_meta.extend({"type": "square", "i": int(i)} for i in range(D))

    if not feats:
        feats = [Ff]
        feature_meta = [{"type": "identity", "i": int(i)} for i in range(D)]

    F_bank = torch.cat(feats, dim=2)
    return F_bank, feature_meta


@torch.no_grad()
def normalize_feature_bank(
    F_bank: torch.Tensor,
    method: str = "robust",
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Per-channel normalization for heterogeneous engineered features.

    robust: z = (x - median) / (MAD + eps)
    zscore: z = (x - mean) / (std + eps)
    """
    if F_bank.ndim != 3:
        raise ValueError(f"F_bank must be (H,W,M), got shape={tuple(F_bank.shape)}")

    H, W, M = F_bank.shape
    X = F_bank.reshape(-1, M).to(torch.float32)
    mode = str(method).lower()

    if mode == "robust":
        center = X.median(dim=0).values
        scale = (X - center.unsqueeze(0)).abs().median(dim=0).values
        scale = scale.clamp(min=float(eps))
    elif mode == "zscore":
        center = X.mean(dim=0)
        scale = X.std(dim=0, unbiased=False).clamp(min=float(eps))
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    Xn = (X - center.unsqueeze(0)) / scale.unsqueeze(0)
    Xn = torch.nan_to_num(Xn, nan=0.0, posinf=0.0, neginf=0.0)
    return Xn.reshape(H, W, M), {
        "method": mode,
        "center": center,
        "scale": scale,
    }


@torch.no_grad()
def prepare_expanded_features(
    F: torch.Tensor,
    *,
    eps: float = 1e-6,
    feature_norm_method: str = "robust",
    clip_ratio: float = 10.0,
    include_identity: bool = True,
    include_diff: bool = True,
    include_normdiff: bool = False,
    include_ratio: bool = False,
    include_product: bool = False,
    include_log: bool = False,
    include_square: bool = False,
) -> Dict[str, Any]:
    """Build + normalize expanded features once (cache-friendly)."""
    F_bank, feature_meta = build_feature_bank(
        F,
        eps=eps,
        include_identity=include_identity,
        include_diff=include_diff,
        include_normdiff=include_normdiff,
        include_ratio=include_ratio,
        include_product=include_product,
        include_log=include_log,
        include_square=include_square,
        clip_ratio=clip_ratio,
    )
    F_norm, norm_stats = normalize_feature_bank(
        F_bank,
        method=feature_norm_method,
        eps=eps,
    )

    feature_labels = [_feature_label(meta) for meta in feature_meta]
    feature_types = _feature_types(feature_meta)
    identity_mask = _feature_type_mask(feature_meta, "identity")
    diff_mask = _feature_type_mask(feature_meta, "diff")

    return {
        "F_bank_norm": F_norm,
        "feature_meta": feature_meta,
        "feature_labels": feature_labels,
        "feature_types": feature_types,
        "feature_bank_size": int(F_norm.shape[2]),
        "feature_norm_method": str(norm_stats.get("method", feature_norm_method)),
        "feature_norm_summary": _summarize_norm_stats(norm_stats),
        "feature_recipe": {
            "include_identity": bool(include_identity),
            "include_diff": bool(include_diff),
            "include_normdiff": bool(include_normdiff),
            "include_ratio": bool(include_ratio),
            "include_product": bool(include_product),
            "include_log": bool(include_log),
            "include_square": bool(include_square),
            "feature_norm_method": str(feature_norm_method).lower(),
            "feature_clip_ratio": float(clip_ratio),
            "pair_base_limit": int(_MAX_PAIR_BASE),
        },
        "feature_type_mask_identity_full": identity_mask,
        "feature_type_mask_diff_full": diff_mask,
        "n_identity_features": int(sum(identity_mask)),
        "n_diff_features": int(sum(diff_mask)),
    }


@torch.no_grad()
def score_feature_bank_positive_bg(
    F_bank_norm: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: Optional[List[Tuple[int, int]]] = None,
    *,
    eps: float = 1e-6,
    beta: float = 0.5,
    exclude_similar_background: bool = False,
    background_exclude_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """Score each normalized channel using positive vs background separation."""
    _ = neg_points
    if F_bank_norm.ndim != 3:
        raise ValueError(f"F_bank_norm must be (H,W,M), got {tuple(F_bank_norm.shape)}")
    if len(pos_points) == 0:
        raise ValueError("Need at least one positive point for feature scoring")

    H, W, M = F_bank_norm.shape
    X_all = F_bank_norm.reshape(-1, M).to(torch.float32)
    device = X_all.device

    pos_vecs = torch.stack(
        [_sample(F_bank_norm, x, y).to(dtype=torch.float32, device=device) for x, y in pos_points],
        dim=0,
    )

    bg_keep = torch.ones(H * W, dtype=torch.bool, device=device)
    for x, y in pos_points:
        xc = max(0, min(int(x), W - 1))
        yc = max(0, min(int(y), H - 1))
        bg_keep[yc * W + xc] = False

    if exclude_similar_background and background_exclude_thresh is not None:
        Xn = X_all / X_all.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Pn = pos_vecs / pos_vecs.norm(dim=1, keepdim=True).clamp(min=1e-12)
        max_sim = (Xn @ Pn.t()).max(dim=1).values
        bg_keep &= max_sim < float(background_exclude_thresh)

    if bool(bg_keep.any()):
        X_bg = X_all[bg_keep]
    else:
        X_bg = X_all

    mu_pos = pos_vecs.mean(dim=0)
    mu_bg = X_bg.mean(dim=0)
    var_pos = ((pos_vecs - mu_pos.unsqueeze(0)) ** 2).mean(dim=0)
    var_bg = ((X_bg - mu_bg.unsqueeze(0)) ** 2).mean(dim=0)

    score = (mu_pos - mu_bg).pow(2) / (var_pos + float(beta) * var_bg + float(eps))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)

    bg_mode = "global"
    if exclude_similar_background and background_exclude_thresh is not None:
        bg_mode = "excluded_similar"

    return {
        "mu_pos": mu_pos,
        "mu_bg": mu_bg,
        "var_pos": var_pos,
        "var_bg": var_bg,
        "score": score,
        "used_negative_for_scoring": False,
        "beta_bg": float(beta),
        "background_mode": bg_mode,
        "n_pos_samples": int(pos_vecs.shape[0]),
        "n_bg_samples": int(X_bg.shape[0]),
    }


@torch.no_grad()
def score_feature_bank(
    F_bank_norm: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    *,
    eps: float = 1e-6,
    beta: float = 0.5,
    exclude_similar_background: bool = False,
    background_exclude_thresh: Optional[float] = None,
) -> Dict[str, Any]:
    """Compatibility wrapper: negatives are intentionally ignored."""
    return score_feature_bank_positive_bg(
        F_bank_norm,
        pos_points,
        neg_points=neg_points,
        eps=eps,
        beta=beta,
        exclude_similar_background=exclude_similar_background,
        background_exclude_thresh=background_exclude_thresh,
    )


@torch.no_grad()
def select_feature_bank(
    feature_scores: Dict[str, Any],
    feature_meta: List[Dict[str, Any]],
    *,
    feature_labels: Optional[List[str]] = None,
    mode: str = "soft",
    topk: Optional[int] = None,
    max_selected_dim: Optional[int] = None,
) -> Dict[str, Any]:
    """Diagnostic ranking helper.

    Default path keeps soft weighting semantics; hard top-k is compatibility-only.
    """
    if feature_labels is None:
        feature_labels = [_feature_label(meta) for meta in feature_meta]

    score = feature_scores["score"].reshape(-1).to(torch.float32)
    rank_idx = torch.argsort(score, descending=True)
    n_total = int(score.numel())

    mode_l = str(mode).lower()
    keep = n_total
    cap_applied = False
    selection_mode = "soft_weighting"

    if mode_l == "topk":
        keep = int(topk if topk is not None else (max_selected_dim or _MAX_SELECTED_DIM))
        keep = max(1, min(keep, n_total))
        cap_applied = keep < n_total
        selection_mode = f"topk_{keep}"
    elif max_selected_dim is not None and n_total > int(max_selected_dim):
        keep = int(max_selected_dim)
        cap_applied = True
        selection_mode = f"soft_weighting_cap_{keep}"

    selected_idx_t = rank_idx[:keep]
    selected_idx = [int(v) for v in selected_idx_t.tolist()]
    selected_meta = [feature_meta[idx] for idx in selected_idx]
    selected_labels = [feature_labels[idx] for idx in selected_idx]
    selected_types = [str(meta.get("type", "")) for meta in selected_meta]

    return {
        "ranked_idx": [int(v) for v in rank_idx.tolist()],
        "selected_idx_t": selected_idx_t,
        "selected_idx": selected_idx,
        "selected_meta": selected_meta,
        "selected_labels": selected_labels,
        "selected_types": selected_types,
        "selected_dim": int(len(selected_idx)),
        "selection_mode": selection_mode,
        "cap_applied": cap_applied,
    }


@torch.no_grad()
def apply_feature_weighting(
    F_bank_norm: torch.Tensor,
    feature_scores: Dict[str, Any],
    feature_meta: List[Dict[str, Any]],
    *,
    feature_labels: Optional[List[str]] = None,
    mode: str = "soft",
    topk: Optional[int] = None,
    w_min: float = 0.25,
    gamma: float = 1.5,
    eps: float = 1e-6,
    max_selected_dim: int = _MAX_SELECTED_DIM,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Soft weighting of normalized features with optional compatibility cap."""
    if F_bank_norm.ndim != 3:
        raise ValueError(f"F_bank_norm must be (H,W,M), got {tuple(F_bank_norm.shape)}")

    if feature_labels is None:
        feature_labels = [_feature_label(meta) for meta in feature_meta]

    score = feature_scores["score"].reshape(-1).to(torch.float32)
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    score_norm = score / (score.max() + float(eps))
    score_norm = torch.nan_to_num(score_norm, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    weights = torch.clamp(score_norm.pow(float(gamma)), min=float(w_min), max=1.0)
    weights = torch.nan_to_num(weights, nan=float(w_min), posinf=1.0, neginf=float(w_min))

    selection = select_feature_bank(
        feature_scores,
        feature_meta,
        feature_labels=feature_labels,
        mode=mode,
        topk=topk,
        max_selected_dim=max_selected_dim,
    )

    F_weighted_full = F_bank_norm * weights.view(1, 1, -1)
    selected_idx_t = selection["selected_idx_t"].to(device=F_bank_norm.device)
    F_selected = F_weighted_full.index_select(2, selected_idx_t)

    selected_score = score.index_select(0, selected_idx_t)
    selected_score_norm = score_norm.index_select(0, selected_idx_t)
    selected_weights = weights.index_select(0, selected_idx_t)
    selected_type_mask_identity = [t == "identity" for t in selection["selected_types"]]
    selected_type_mask_diff = [t == "diff" for t in selection["selected_types"]]

    return F_selected, {
        "mode": str(mode).lower(),
        "selection_mode": selection["selection_mode"],
        "cap_applied": bool(selection["cap_applied"]),
        "selected_idx": selection["selected_idx"],
        "selected_meta": selection["selected_meta"],
        "selected_labels": selection["selected_labels"],
        "selected_types": selection["selected_types"],
        "selected_dim": int(F_selected.shape[2]),
        "weights": weights,
        "score_norm": score_norm,
        "selected_score": selected_score,
        "selected_score_norm": selected_score_norm,
        "selected_weights": selected_weights,
        "weight_floor": float(w_min),
        "weight_gamma": float(gamma),
        "feature_type_mask_identity": selected_type_mask_identity,
        "feature_type_mask_diff": selected_type_mask_diff,
    }


@torch.no_grad()
def apply_feature_selection(
    F_bank_norm: torch.Tensor,
    feature_scores: Dict[str, Any],
    feature_meta: List[Dict[str, Any]],
    *,
    feature_labels: Optional[List[str]] = None,
    mode: str = "soft",
    topk: Optional[int] = None,
    w_min: float = 0.25,
    gamma: float = 1.5,
    eps: float = 1e-6,
    max_selected_dim: int = _MAX_SELECTED_DIM,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compatibility wrapper for the old helper name."""
    return apply_feature_weighting(
        F_bank_norm,
        feature_scores,
        feature_meta,
        feature_labels=feature_labels,
        mode=mode,
        topk=topk,
        w_min=w_min,
        gamma=gamma,
        eps=eps,
        max_selected_dim=max_selected_dim,
    )


@torch.no_grad()
def _subsample_rows(X: torch.Tensor, max_rows: int) -> torch.Tensor:
    if max_rows <= 0 or X.shape[0] <= max_rows:
        return X
    idx = torch.randperm(X.shape[0], device=X.device)[:max_rows]
    return X.index_select(0, idx)


@torch.no_grad()
def _estimate_covariance_stable(
    X: torch.Tensor,
    *,
    method: str,
    eta: float,
    dtype_hint: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Robust covariance inverse estimation for highly expanded features.

    Strategy:
      1) Remove non-finite rows.
      2) Retry inversion with increasing diagonal loading.
      3) If needed, upgrade dtype to float64.
      4) Last resort: pseudo-inverse fallback.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be (N,D), got {tuple(X.shape)}")
    if X.shape[0] == 0:
        raise ValueError("X is empty")

    finite_rows = torch.isfinite(X).all(dim=1)
    if bool(finite_rows.any()):
        X_use = X[finite_rows]
    else:
        X_use = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    eta0 = float(max(float(eta), 1e-4))
    eta_schedule = [eta0, eta0 * 10.0, eta0 * 100.0, eta0 * 1000.0]
    dtype_order: List[torch.dtype] = [dtype_hint]
    if torch.float64 not in dtype_order:
        dtype_order.append(torch.float64)

    last_err: Optional[Exception] = None
    for dtry in dtype_order:
        Xd = X_use.to(dtype=dtry)
        for etry in eta_schedule:
            try:
                R_inv, mu_B, info = estimate_covariance(
                    Xd,
                    eta=etry,
                    method=method,
                    dtype=dtry,
                )
                if not (torch.isfinite(R_inv).all() and torch.isfinite(mu_B).all()):
                    raise RuntimeError("non-finite covariance outputs")
                info = dict(info or {})
                info["eta_used"] = float(etry)
                info["cov_dtype_used"] = str(dtry).replace("torch.", "")
                info["cov_retries"] = int(eta_schedule.index(etry))
                info["finite_rows"] = int(X_use.shape[0])
                return R_inv, mu_B, info
            except Exception as e:  # noqa: BLE001
                last_err = e

    Xd = torch.nan_to_num(X_use.to(dtype=torch.float64), nan=0.0, posinf=0.0, neginf=0.0)
    N, D = Xd.shape
    mu_B = Xd.mean(dim=0)
    Xc = Xd - mu_B.unsqueeze(0)
    S = (Xc.t() @ Xc) / max(N - 1, 1)
    eta_fb = float(max(eta_schedule[-1], 1.0))
    R = S + eta_fb * torch.eye(D, dtype=Xd.dtype, device=Xd.device)
    R_inv = torch.linalg.pinv(R)
    info_fb: Dict[str, Any] = {
        "cov_method": f"{method}_pinv_fallback",
        "eta_used": eta_fb,
        "cov_dtype_used": "float64",
        "cov_retries": len(eta_schedule),
        "finite_rows": int(X_use.shape[0]),
        "fallback_reason": str(last_err) if last_err is not None else "unknown",
    }
    return R_inv, mu_B, info_fb


@torch.no_grad()
def detect_step_new_method(
    F: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    *,
    scoring: str = "amf",
    cov_method: str = "ledoit_wolf",
    eta: float = CFG.eta,
    max_cov_samples: int = CFG.max_cov_samples,
    use_otsu: bool = True,
    use_fpr: bool = False,
    fpr: float = CFG.fpr,
    threshold: Optional[float] = None,
    eps: float = 1e-6,
    target_exclude_thresh: float = CFG.target_exclude_thresh,
    feature_expand_mode: str = "soft",
    feature_norm_method: str = "robust",
    feature_clip_ratio: float = 10.0,
    feature_score_beta_bg: float = 0.5,
    feature_weight_floor: float = 0.25,
    feature_weight_gamma: float = 1.5,
    include_identity: bool = True,
    include_diff: bool = True,
    include_normdiff: bool = False,
    include_ratio: bool = False,
    include_product: bool = False,
    include_log: bool = False,
    include_square: bool = False,
    external_exclude_mask: Optional[np.ndarray] = None,
    prepared: Optional[Dict[str, Any]] = None,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
    """Expanded-feature AMF/ACE detection in weighted feature space."""
    if prepared is None and F.ndim != 3:
        raise ValueError(f"F must be (H,W,D), got shape={tuple(F.shape)}")
    if len(pos_points) == 0:
        raise ValueError("Need at least one positive point")

    score_mode = str(scoring).lower()
    if score_mode not in ("amf", "ace"):
        raise ValueError(f"scoring must be 'amf' or 'ace', got: {scoring}")

    if prepared is None:
        compute_device = _get_compute_device(F.device)
        try:
            F_work = F.to(device=compute_device, dtype=torch.float32, non_blocking=True)
        except Exception:
            F_work = F.to(dtype=torch.float32)
        prepared = prepare_expanded_features(
            F_work,
            eps=eps,
            feature_norm_method=feature_norm_method,
            clip_ratio=feature_clip_ratio,
            include_identity=include_identity,
            include_diff=include_diff,
            include_normdiff=include_normdiff,
            include_ratio=include_ratio,
            include_product=include_product,
            include_log=include_log,
            include_square=include_square,
        )
        prepared["prep_gpu_mem"] = _cuda_mem_stats(F_work.device)

    F_bank_norm = prepared["F_bank_norm"]
    if F_bank_norm.device.type == "cuda":
        didx = F_bank_norm.device.index if F_bank_norm.device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(didx)

    feature_scores = score_feature_bank_positive_bg(
        F_bank_norm,
        pos_points,
        neg_points=neg_points,
        eps=eps,
        beta=feature_score_beta_bg,
        exclude_similar_background=False,
        background_exclude_thresh=None,
    )

    F_exp, apply_info = apply_feature_weighting(
        F_bank_norm,
        feature_scores,
        prepared["feature_meta"],
        feature_labels=prepared["feature_labels"],
        mode=feature_expand_mode,
        topk=_MAX_SELECTED_DIM if str(feature_expand_mode).lower() == "topk" else None,
        w_min=feature_weight_floor,
        gamma=feature_weight_gamma,
        eps=eps,
        max_selected_dim=_MAX_SELECTED_DIM,
    )
    selected_idx_t = torch.tensor(
        apply_info["selected_idx"],
        dtype=torch.long,
        device=F_bank_norm.device,
    )
    F_norm_selected = F_bank_norm.index_select(2, selected_idx_t)
    pos_spectra_norm_selected: List[List[float]] = []
    neg_spectra_norm_selected: List[List[float]] = []
    pos_vecs_norm_selected: List[torch.Tensor] = []
    for x, y in pos_points:
        s_norm = _sample(F_norm_selected, x, y).to(dtype=torch.float32, device=F_norm_selected.device)
        pos_vecs_norm_selected.append(s_norm)
        pos_spectra_norm_selected.append(s_norm.detach().cpu().float().tolist())
    for x, y in neg_points:
        s_norm = _sample(F_norm_selected, x, y).to(dtype=torch.float32, device=F_norm_selected.device)
        neg_spectra_norm_selected.append(s_norm.detach().cpu().float().tolist())
    target_spectrum_norm = torch.stack(pos_vecs_norm_selected, dim=0).mean(dim=0)
    mu_bg_norm_selected = feature_scores["mu_bg"].index_select(0, selected_idx_t)

    H_, W_, D_exp = F_exp.shape
    cov_dtype = _cov_dtype_for_device(F_exp.device)
    n_pix = int(H_ * W_)
    feature_map_mb = (n_pix * D_exp * 4.0) / (1024.0 * 1024.0)
    cov_mat_mb_fp64 = (D_exp * D_exp * 8.0) / (1024.0 * 1024.0)
    if feature_map_mb > 512.0 or cov_mat_mb_fp64 > 256.0:
        print(
            f"[new_method] memory estimate | D_exp={D_exp}, "
            f"feature_map={feature_map_mb:.1f}MB, cov(fp64)={cov_mat_mb_fp64:.1f}MB"
        )

    X_all = F_exp.reshape(-1, D_exp).to(torch.float32)
    X_cov = _subsample_rows(X_all, int(max_cov_samples))

    R_inv_0, mu_B_0, cov_info_0 = _estimate_covariance_stable(
        X_cov,
        method="sample",
        eta=eta,
        dtype_hint=cov_dtype,
    )
    R_inv_sqrt_0 = _compute_R_inv_sqrt(R_inv_0)

    exclude_mask_hw: Optional[np.ndarray] = None
    N_excluded = 0

    if external_exclude_mask is not None:
        if external_exclude_mask.shape != (H_, W_):
            ext_mask = cv2.resize(
                external_exclude_mask.astype(np.uint8),
                (W_, H_),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        else:
            ext_mask = external_exclude_mask.astype(bool)

        mask_flat = torch.from_numpy(ext_mask.reshape(-1)).to(X_all.device)
        X_bg_all = X_all[~mask_flat]
        N_excluded = int(mask_flat.sum().item())
        exclude_mask_hw = ext_mask

        X_bg = _subsample_rows(X_bg_all, int(max_cov_samples))
        if X_bg.shape[0] > D_exp:
            R_inv, mu_B, cov_info = _estimate_covariance_stable(
                X_bg,
                method=cov_method,
                eta=eta,
                dtype_hint=cov_dtype,
            )
        else:
            R_inv, mu_B, cov_info = R_inv_0, mu_B_0, dict(cov_info_0)
            N_excluded = 0
            exclude_mask_hw = None
    elif target_exclude_thresh < 1.0:
        pos_vecs_for_excl = torch.stack(
            [
                _sample(F_exp, x, y).to(dtype=cov_dtype, device=F_exp.device)
                for x, y in pos_points
            ],
            dim=0,
        )
        X_bg = _exclude_target_pixels(
            X_cov,
            pos_vecs_for_excl,
            R_inv_sqrt_0,
            mu_B_0,
            thresh=target_exclude_thresh,
        )
        N_excluded = int(X_cov.shape[0] - X_bg.shape[0])
        exclude_mask_hw = _compute_exclude_mask(
            F_exp,
            pos_vecs_for_excl,
            R_inv_sqrt_0,
            mu_B_0,
            thresh=target_exclude_thresh,
        )

        if X_bg.shape[0] > D_exp:
            R_inv, mu_B, cov_info = _estimate_covariance_stable(
                X_bg,
                method=cov_method,
                eta=eta,
                dtype_hint=cov_dtype,
            )
        else:
            R_inv, mu_B, cov_info = R_inv_0, mu_B_0, dict(cov_info_0)
            N_excluded = 0
            exclude_mask_hw = None
    else:
        R_inv, mu_B, cov_info = _estimate_covariance_stable(
            X_cov,
            method=cov_method,
            eta=eta,
            dtype_hint=cov_dtype,
        )

    cov_info["N_excluded"] = int(N_excluded)
    cov_condition_number = _try_condition_number(R_inv)
    if np.isfinite(cov_condition_number):
        cov_info["cond_R"] = cov_condition_number
    cov_info["weighted_dim"] = int(D_exp)

    score_dtype = torch.float32 if F_exp.device.type == "cuda" else R_inv.dtype
    R_inv_score = R_inv.to(dtype=score_dtype)
    mu_B_score = mu_B.to(dtype=score_dtype)

    pos_vecs_t: List[torch.Tensor] = []
    pos_spectra_exp: List[List[float]] = []
    for x, y in pos_points:
        s_vec = _sample(F_exp, x, y).to(dtype=score_dtype, device=F_exp.device)
        pos_vecs_t.append(s_vec)
        pos_spectra_exp.append(s_vec.detach().cpu().float().tolist())

    target = torch.stack(pos_vecs_t, dim=0).mean(dim=0)
    scorer = _score_ace if score_mode == "ace" else _score_amf
    score_map, w_vec = scorer(F_exp, target, R_inv_score, mu_B_score)

    mask, used_threshold = generate_mask(
        score_map,
        use_otsu=use_otsu,
        use_fpr=use_fpr,
        fpr=fpr,
        theta=threshold,
        return_threshold=True,
    )

    R_inv_vis = R_inv.detach().cpu().float()
    if R_inv_vis.shape[0] > _MAX_RINV_VIS_DIM:
        R_inv_vis = R_inv_vis[:_MAX_RINV_VIS_DIM, :_MAX_RINV_VIS_DIM]

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

    full_identity_mask = prepared["feature_type_mask_identity_full"]
    full_diff_mask = prepared["feature_type_mask_diff_full"]
    weights_full = apply_info["weights"].detach().cpu().float()
    if any(full_identity_mask):
        mean_weight_identity = float(weights_full[torch.tensor(full_identity_mask)].mean().item())
    else:
        mean_weight_identity = 0.0
    if any(full_diff_mask):
        mean_weight_diff = float(weights_full[torch.tensor(full_diff_mask)].mean().item())
    else:
        mean_weight_diff = 0.0

    state: Dict[str, Any] = {
        "n_pos": len(pos_points),
        "n_neg": len(neg_points),
        "scoring": f"new_method_{score_mode}",
        "detector": score_mode,
        "cov_info": cov_info,
        "cov_condition_number": cov_condition_number,
        "mu_B": mu_B.detach().cpu().float().tolist(),
        "w_effective": w_vec.detach().cpu().float().tolist(),
        "target_spectrum": target.detach().cpu().float().tolist(),
        "pos_spectra_raw": pos_spectra_exp,
        "target_spectrum_norm": target_spectrum_norm.detach().cpu().float().tolist(),
        "pos_spectra_norm": pos_spectra_norm_selected,
        "neg_spectra_norm": neg_spectra_norm_selected,
        "mu_bg_spectrum_norm": mu_bg_norm_selected.detach().cpu().float().tolist(),
        "neg_spectra_raw": [],
        "score_hist": score_hist,
        "exclude_mask": exclude_mask_hw,
        "threshold_used": float(used_threshold),
        "r_inv_matrix": R_inv_vis.tolist(),
        "r_inv_dim": int(R_inv.shape[0]),
        "r_inv_dim_shown": int(R_inv_vis.shape[0]),
        "use_feature_expansion": True,
        "feature_meta": prepared["feature_meta"],
        "feature_labels": prepared["feature_labels"],
        "feature_types": prepared["feature_types"],
        "feature_bank_size": int(prepared["feature_bank_size"]),
        "feature_selected_idx": apply_info["selected_idx"],
        "feature_selected_meta": apply_info["selected_meta"],
        "feature_selected_labels": apply_info["selected_labels"],
        "feature_selected_types": apply_info["selected_types"],
        "feature_selection_mode": apply_info["selection_mode"],
        "feature_selected_dim": int(apply_info["selected_dim"]),
        "feature_norm_method": prepared["feature_norm_method"],
        "feature_norm_summary": prepared["feature_norm_summary"],
        "feature_used_neg_fallback": False,
        "feature_scoring_mode": "positive_vs_background",
        "feature_weighting_mode": "power_floor",
        "feature_expand_mode": str(feature_expand_mode).lower(),
        "feature_weight_floor": float(feature_weight_floor),
        "feature_weight_gamma": float(feature_weight_gamma),
        "feature_beta_bg": float(feature_score_beta_bg),
        "feature_score": feature_scores["score"].detach().cpu().float().tolist(),
        "feature_score_normalized": apply_info["score_norm"].detach().cpu().float().tolist(),
        "feature_weights": weights_full.tolist(),
        "feature_sep_score": feature_scores["score"].detach().cpu().float().tolist(),
        "feature_final_score": apply_info["score_norm"].detach().cpu().float().tolist(),
        "feature_mu_pos": feature_scores["mu_pos"].detach().cpu().float().tolist(),
        "feature_mu_bg": feature_scores["mu_bg"].detach().cpu().float().tolist(),
        "feature_var_pos": feature_scores["var_pos"].detach().cpu().float().tolist(),
        "feature_var_bg": feature_scores["var_bg"].detach().cpu().float().tolist(),
        "used_negative_for_scoring": bool(feature_scores["used_negative_for_scoring"]),
        "feature_background_mode": feature_scores["background_mode"],
        "n_identity_features": int(prepared["n_identity_features"]),
        "n_diff_features": int(prepared["n_diff_features"]),
        "feature_type_mask_identity": apply_info["feature_type_mask_identity"],
        "feature_type_mask_diff": apply_info["feature_type_mask_diff"],
        "feature_type_mask_identity_full": full_identity_mask,
        "feature_type_mask_diff_full": full_diff_mask,
        "mean_weight_identity": mean_weight_identity,
        "mean_weight_diff": mean_weight_diff,
        "memory_estimate": {
            "n_pixels": n_pix,
            "feature_bank_dim": int(prepared["feature_bank_size"]),
            "d_exp": int(D_exp),
            "feature_dim_cap": int(_MAX_SELECTED_DIM),
            "feature_map_mb_fp32": float(round(feature_map_mb, 3)),
            "cov_matrix_mb_fp64": float(round(cov_mat_mb_fp64, 3)),
        },
        "gpu_mem": {
            "prep": prepared.get("prep_gpu_mem") or {},
            "run": _cuda_mem_stats(F_exp.device),
        },
    }
    return score_map, mask, state


class _NewMethodBase:
    """Webapp wrapper with expanded-feature cache per uploaded cube."""

    name = "new_method_base"

    def __init__(
        self,
        *,
        detector: str,
        cov_method: str = "ledoit_wolf",
        feature_expand_mode: str = "soft",
        feature_norm_method: str = "robust",
        feature_clip_ratio: float = 10.0,
        feature_score_beta_bg: float = 0.5,
        feature_weight_floor: float = 0.25,
        feature_weight_gamma: float = 1.5,
        include_identity: bool = True,
        include_diff: bool = True,
        include_normdiff: bool = False,
        include_ratio: bool = False,
        include_product: bool = False,
        include_log: bool = False,
        include_square: bool = False,
    ):
        self.detector = str(detector).lower()
        if self.detector not in ("amf", "ace"):
            raise ValueError(f"detector must be amf/ace, got {detector}")

        self.cov_method = cov_method
        self.feature_expand_mode = str(feature_expand_mode).lower()
        self.feature_norm_method = str(feature_norm_method).lower()
        self.feature_clip_ratio = float(feature_clip_ratio)
        self.feature_score_beta_bg = float(feature_score_beta_bg)
        self.feature_weight_floor = float(feature_weight_floor)
        self.feature_weight_gamma = float(feature_weight_gamma)

        self.include_identity = bool(include_identity)
        self.include_diff = bool(include_diff)
        self.include_normdiff = bool(include_normdiff)
        self.include_ratio = bool(include_ratio)
        self.include_product = bool(include_product)
        self.include_log = bool(include_log)
        self.include_square = bool(include_square)

        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_prepared: Optional[Dict[str, Any]] = None

    def _prepared_recipe(self) -> Dict[str, Any]:
        return {
            "include_identity": self.include_identity,
            "include_diff": self.include_diff,
            "include_normdiff": self.include_normdiff,
            "include_ratio": self.include_ratio,
            "include_product": self.include_product,
            "include_log": self.include_log,
            "include_square": self.include_square,
            "feature_norm_method": self.feature_norm_method,
            "feature_clip_ratio": self.feature_clip_ratio,
            "pair_base_limit": _MAX_PAIR_BASE,
        }

    def _prepared_matches_recipe(self, prepared: Dict[str, Any]) -> bool:
        return prepared.get("feature_recipe") == self._prepared_recipe()

    def _cube_key(self, cube: np.ndarray) -> Tuple[Any, ...]:
        ai = cube.__array_interface__
        ptr = int(ai["data"][0])
        if cube.size > 0:
            flat = cube.reshape(-1)
            head = float(flat[0])
            tail = float(flat[-1])
        else:
            head = 0.0
            tail = 0.0
        recipe = (
            "identity_diff_weighted_v2",
            self.feature_norm_method,
            round(self.feature_clip_ratio, 8),
            self.include_identity,
            self.include_diff,
            self.include_normdiff,
            self.include_ratio,
            self.include_product,
            self.include_log,
            self.include_square,
            _MAX_PAIR_BASE,
        )
        return (ptr, cube.shape, cube.strides, cube.dtype.str, head, tail, recipe)

    @torch.no_grad()
    def _get_or_build_prepared(self, cube: np.ndarray) -> Dict[str, Any]:
        key = self._cube_key(cube)
        if (
            self._cache_key == key
            and self._cache_prepared is not None
            and self._prepared_matches_recipe(self._cache_prepared)
        ):
            return self._cache_prepared

        F = torch.from_numpy(cube.astype(np.float32, copy=False))
        device = _get_compute_device(F.device)
        try:
            F = F.to(device=device, dtype=torch.float32, non_blocking=True)
        except Exception:
            F = F.to(dtype=torch.float32)

        if F.device.type == "cuda":
            didx = F.device.index if F.device.index is not None else torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(didx)

        prepared = prepare_expanded_features(
            F,
            eps=CFG.eps,
            feature_norm_method=self.feature_norm_method,
            clip_ratio=self.feature_clip_ratio,
            include_identity=self.include_identity,
            include_diff=self.include_diff,
            include_normdiff=self.include_normdiff,
            include_ratio=self.include_ratio,
            include_product=self.include_product,
            include_log=self.include_log,
            include_square=self.include_square,
        )
        prepared["prep_gpu_mem"] = _cuda_mem_stats(F.device)

        self._cache_key = key
        self._cache_prepared = prepared
        return prepared

    @torch.no_grad()
    def predict(
        self,
        cube: np.ndarray,
        pos_points: List[Tuple[int, int]],
        neg_points: Optional[List[Tuple[int, int]]] = None,
        gt_mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        _ = gt_mask
        if neg_points is None:
            neg_points = []

        prepared = self._get_or_build_prepared(cube)
        F = torch.from_numpy(cube.astype(np.float32, copy=False))

        score_t, mask, state = detect_step_new_method(
            F,
            pos_points=pos_points,
            neg_points=neg_points,
            scoring=self.detector,
            cov_method=self.cov_method,
            eta=CFG.eta,
            max_cov_samples=CFG.max_cov_samples,
            use_otsu=True,
            use_fpr=False,
            fpr=CFG.fpr,
            threshold=threshold,
            eps=CFG.eps,
            target_exclude_thresh=CFG.target_exclude_thresh,
            feature_expand_mode=self.feature_expand_mode,
            feature_norm_method=self.feature_norm_method,
            feature_clip_ratio=self.feature_clip_ratio,
            feature_score_beta_bg=self.feature_score_beta_bg,
            feature_weight_floor=self.feature_weight_floor,
            feature_weight_gamma=self.feature_weight_gamma,
            include_identity=self.include_identity,
            include_diff=self.include_diff,
            include_normdiff=self.include_normdiff,
            include_ratio=self.include_ratio,
            include_product=self.include_product,
            include_log=self.include_log,
            include_square=self.include_square,
            external_exclude_mask=None,
            prepared=prepared,
        )
        return {
            "mask": mask,
            "score_map": score_t.detach().cpu().numpy().astype(np.float32),
            "state": state,
            "threshold": state.get("threshold_used"),
        }


class NewMethodAMF(_NewMethodBase):
    name = "new_method_amf"

    def __init__(self, **kwargs):
        super().__init__(detector="amf", **kwargs)


class NewMethodACE(_NewMethodBase):
    name = "new_method_ace"

    def __init__(self, **kwargs):
        super().__init__(detector="ace", **kwargs)
