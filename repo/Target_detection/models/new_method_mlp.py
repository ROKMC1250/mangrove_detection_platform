"""new_method_mlp: detector-aware bottleneck projector + direct AMF/ACE."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as Fnn
from torch import nn

from models.config import CFG
from models.new_method import (
    _MAX_RINV_VIS_DIM,
    _cuda_mem_stats,
    _cov_dtype_for_device,
    _estimate_covariance_stable,
    _get_compute_device,
    _subsample_rows,
    _summarize_norm_stats,
    _try_condition_number,
    normalize_feature_bank,
)
from models.osp_amf import (
    _compute_R_inv_sqrt,
    _compute_exclude_mask,
    _exclude_target_pixels,
    _sample,
    _score_ace,
    _score_amf,
    generate_mask,
)

PROJECTOR_OUT_DIM = 10


def _projected_feature_label(idx: int) -> str:
    return f"proj[{idx}]"


def _set_seed(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _point_flat_indices(
    H: int,
    W: int,
    points: List[Tuple[int, int]],
    *,
    device: torch.device,
) -> torch.Tensor:
    if len(points) == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    idxs = []
    for x, y in points:
        xc = max(0, min(int(x), W - 1))
        yc = max(0, min(int(y), H - 1))
        idxs.append(yc * W + xc)
    return torch.tensor(idxs, dtype=torch.long, device=device)


def _background_rows_for_points(
    F_proj: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W, r = F_proj.shape
    X_all = F_proj.reshape(-1, r)
    device = X_all.device
    keep = torch.ones(H * W, dtype=torch.bool, device=device)

    pos_idx = _point_flat_indices(H, W, pos_points, device=device)
    neg_idx = _point_flat_indices(H, W, neg_points, device=device)
    if pos_idx.numel() > 0:
        keep[pos_idx] = False
    if neg_idx.numel() > 0:
        keep[neg_idx] = False

    if bool(keep.any()):
        return X_all[keep], keep
    return X_all, torch.ones(H * W, dtype=torch.bool, device=device)


def _sample_rows_limit(X: torch.Tensor, limit: int) -> torch.Tensor:
    if limit <= 0 or X.shape[0] <= limit:
        return X
    idx = torch.randperm(X.shape[0], device=X.device)[:limit]
    return X.index_select(0, idx)


@torch.no_grad()
def _prepare_raw_projector_input(
    F: torch.Tensor,
    *,
    eps: float = 1e-6,
    feature_norm_method: str = "robust",
) -> Dict[str, Any]:
    """Normalize raw input channels directly for projector training."""
    if F.ndim != 3:
        raise ValueError(f"F must be (H,W,D), got shape={tuple(F.shape)}")

    F_raw = F.to(torch.float32)
    F_input_norm, norm_stats = normalize_feature_bank(
        F_raw,
        method=feature_norm_method,
        eps=eps,
    )
    raw_dim = int(F_input_norm.shape[2])
    raw_meta = [{"type": "identity", "i": int(i)} for i in range(raw_dim)]

    return {
        "F_input_norm": F_input_norm,
        "feature_meta": raw_meta,
        "feature_labels": [f"x[{i}]" for i in range(raw_dim)],
        "feature_types": ["identity"] * raw_dim,
        "feature_bank_size": raw_dim,
        "raw_input_dim": raw_dim,
        "feature_norm_method": str(norm_stats.get("method", feature_norm_method)).lower(),
        "feature_norm_summary": _summarize_norm_stats(norm_stats),
        "feature_recipe": {
            "input_mode": "raw_input",
            "feature_norm_method": str(feature_norm_method).lower(),
        },
        "n_identity_features": raw_dim,
        "n_diff_features": 0,
    }


def _project_feature_map_impl(
    F_input_norm: torch.Tensor,
    projector: nn.Module,
    *,
    batch_rows: int,
) -> torch.Tensor:
    if F_input_norm.ndim != 3:
        raise ValueError(f"F_input_norm must be (H,W,D), got {tuple(F_input_norm.shape)}")
    H, W, M = F_input_norm.shape
    X = F_input_norm.reshape(-1, M)
    outs: List[torch.Tensor] = []
    n_rows = X.shape[0]
    step = max(int(batch_rows), 1)
    for s in range(0, n_rows, step):
        e = min(s + step, n_rows)
        outs.append(projector(X[s:e]))
    X_proj = torch.cat(outs, dim=0)
    return X_proj.reshape(H, W, -1)


class LearnableProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = PROJECTOR_OUT_DIM,
        use_layernorm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {in_dim}")
        if hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("hidden_dim/out_dim must be > 0")

        act_name = str(activation).lower()
        if act_name == "relu":
            act = nn.ReLU()
        elif act_name == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.out_dim = int(out_dim)
        self.use_layernorm = bool(use_layernorm)
        self.activation_name = act_name

        self.input_norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.input_norm(x)
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        return y


@torch.no_grad()
def apply_projector_to_feature_map(
    F_input_norm: torch.Tensor,
    projector: nn.Module,
    batch_rows: int = 65536,
) -> torch.Tensor:
    """Apply a trained projector to a normalized feature map."""
    return _project_feature_map_impl(
        F_input_norm,
        projector,
        batch_rows=batch_rows,
    )


def _score_rows_with_detector(
    X_rows: torch.Tensor,
    target: torch.Tensor,
    R_inv: torch.Tensor,
    mu_B: torch.Tensor,
    *,
    score_fn: str,
) -> torch.Tensor:
    if X_rows.numel() == 0:
        return torch.empty(0, dtype=X_rows.dtype, device=X_rows.device)

    dtype = X_rows.dtype
    device = X_rows.device
    score_mode = str(score_fn).lower()
    R = R_inv.to(dtype=dtype, device=device)
    mu = mu_B.to(dtype=dtype, device=device)
    d = (target - mu).to(dtype=dtype, device=device)
    Rd = R @ d

    Xc = X_rows.to(dtype=dtype, device=device) - mu.unsqueeze(0)
    if score_mode == "ace":
        dRd = (d @ Rd).clamp(min=1e-30)
        num = Xc @ Rd
        xRx = (Xc @ R * Xc).sum(dim=1).clamp(min=1e-30)
        return (num * num) / (dRd * xRx)
    if score_mode != "amf":
        raise ValueError(f"score_fn must be 'amf' or 'ace', got {score_fn}")

    denom = torch.sqrt((d @ Rd).clamp(min=1e-30))
    w = Rd / denom
    return Xc @ w


def compute_detector_scores_for_sets(
    F_proj: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    R_inv: torch.Tensor,
    mu_B: torch.Tensor,
    *,
    score_fn: str = "amf",
    bg_sample_limit: int = 4096,
) -> Dict[str, torch.Tensor]:
    if F_proj.ndim != 3:
        raise ValueError(f"F_proj must be (H,W,r), got {tuple(F_proj.shape)}")
    if len(pos_points) == 0:
        raise ValueError("Need at least one positive point")

    pos_vecs = torch.stack([_sample(F_proj, x, y) for x, y in pos_points], dim=0)
    neg_vecs = (
        torch.stack([_sample(F_proj, x, y) for x, y in neg_points], dim=0)
        if len(neg_points) > 0
        else torch.empty(0, F_proj.shape[2], dtype=F_proj.dtype, device=F_proj.device)
    )
    bg_rows_all, _ = _background_rows_for_points(F_proj, pos_points, neg_points)
    bg_rows = _sample_rows_limit(bg_rows_all, int(bg_sample_limit))
    target = pos_vecs.mean(dim=0)

    return {
        "target": target,
        "pos_vecs": pos_vecs,
        "neg_vecs": neg_vecs,
        "bg_vecs": bg_rows,
        "pos_scores": _score_rows_with_detector(pos_vecs, target, R_inv, mu_B, score_fn=score_fn),
        "neg_scores": _score_rows_with_detector(neg_vecs, target, R_inv, mu_B, score_fn=score_fn),
        "bg_scores": _score_rows_with_detector(bg_rows, target, R_inv, mu_B, score_fn=score_fn),
    }


def pairwise_ranking_loss(
    pos_scores: torch.Tensor,
    other_scores: torch.Tensor,
    *,
    margin: float = 0.2,
    chunk_size: int = 16384,
) -> torch.Tensor:
    if pos_scores.numel() == 0:
        raise ValueError("Need at least one positive score for ranking loss")
    if other_scores.numel() == 0:
        return torch.zeros((), dtype=pos_scores.dtype, device=pos_scores.device)

    pos = pos_scores.reshape(-1, 1)
    other = other_scores.reshape(-1)
    total = torch.zeros((), dtype=pos_scores.dtype, device=pos_scores.device)
    n_pairs = 0
    step = max(int(chunk_size), 1)
    for start in range(0, int(other.numel()), step):
        stop = min(start + step, int(other.numel()))
        other_chunk = other[start:stop].reshape(1, -1)
        diff = pos - other_chunk - float(margin)
        total = total + Fnn.softplus(-diff).sum()
        n_pairs += int(diff.numel())
    if n_pairs <= 0:
        return torch.zeros((), dtype=pos_scores.dtype, device=pos_scores.device)
    return total / float(n_pairs)


def bt_offdiag_loss(
    Y: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float], torch.Tensor]:
    if Y.ndim != 2:
        raise ValueError(f"Y must be (N,r), got {tuple(Y.shape)}")
    if Y.shape[0] == 0:
        C = torch.zeros((Y.shape[1], Y.shape[1]), dtype=Y.dtype, device=Y.device)
        return (
            torch.zeros((), dtype=Y.dtype, device=Y.device),
            {
                "corr_offdiag_mean": 0.0,
                "corr_offdiag_max": 0.0,
                "corr_diag_mean": 0.0,
            },
            C,
        )

    Yc = Y - Y.mean(dim=0, keepdim=True)
    Y_std = Yc.std(dim=0, keepdim=True, unbiased=False).clamp(min=float(eps))
    Y_norm = Yc / Y_std
    N = max(int(Y.shape[0]), 1)
    C = (Y_norm.t() @ Y_norm) / float(N)
    off = C - torch.diag(torch.diagonal(C))
    off_abs = off.abs()
    if off_abs.numel() > 0:
        eye_mask = ~torch.eye(off_abs.shape[0], dtype=torch.bool, device=off_abs.device)
        off_vals = off_abs[eye_mask]
    else:
        off_vals = torch.empty(0, dtype=Y.dtype, device=Y.device)
    diag_vals = torch.diagonal(C)
    return (
        (off * off).sum(),
        {
            "corr_offdiag_mean": float(off_vals.mean().item()) if off_vals.numel() > 0 else 0.0,
            "corr_offdiag_max": float(off_vals.max().item()) if off_vals.numel() > 0 else 0.0,
            "corr_diag_mean": float(diag_vals.mean().item()) if diag_vals.numel() > 0 else 0.0,
        },
        C,
    )


def projector_training_loss(
    F_proj: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    *,
    cov_method: str = "ledoit_wolf",
    eta: float = CFG.eta,
    score_fn: str = "amf",
    ranking_margin: float = 0.2,
    lambda_decorr: float = 0.05,
    lambda_var: float = 0.05,
    lambda_reg: float = 1e-4,
    projector: Optional[nn.Module] = None,
    bg_sample_limit: int = 4096,
    var_floor: float = 0.1,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if F_proj.ndim != 3:
        raise ValueError(f"F_proj must be (H,W,r), got {tuple(F_proj.shape)}")
    if len(pos_points) == 0:
        raise ValueError("Need at least one positive point")

    X_bg_all, _ = _background_rows_for_points(F_proj, pos_points, neg_points)
    if X_bg_all.shape[0] == 0:
        X_bg_all = F_proj.reshape(-1, F_proj.shape[2])

    cov_dtype = _cov_dtype_for_device(F_proj.device)
    R_inv, mu_B, cov_info = _estimate_covariance_stable(
        X_bg_all.detach(),
        method=cov_method,
        eta=eta,
        dtype_hint=cov_dtype,
    )

    score_sets = compute_detector_scores_for_sets(
        F_proj,
        pos_points,
        neg_points,
        R_inv,
        mu_B,
        score_fn=score_fn,
        bg_sample_limit=bg_sample_limit,
    )
    pos_scores = score_sets["pos_scores"]
    neg_scores = score_sets["neg_scores"]
    bg_scores = score_sets["bg_scores"]
    bg_vecs = score_sets["bg_vecs"]

    if neg_scores.numel() > 0:
        ranking_other = neg_scores
    else:
        if bg_scores.numel() == 0:
            bg_scores = torch.zeros(1, dtype=pos_scores.dtype, device=pos_scores.device)
        ranking_other = bg_scores
    L_det = pairwise_ranking_loss(
        pos_scores,
        ranking_other,
        margin=ranking_margin,
    )

    Y = bg_vecs if bg_vecs.numel() > 0 else F_proj.reshape(-1, F_proj.shape[2])
    Y = _sample_rows_limit(Y, int(bg_sample_limit))
    if Y.shape[0] > 0:
        Yc = Y - Y.mean(dim=0, keepdim=True)
        cov_Y = (Yc.t() @ Yc) / max(int(Y.shape[0]) - 1, 1)
        L_decorr, corr_stats, _ = bt_offdiag_loss(Y, eps=eps)
        var_y = Y.var(dim=0, unbiased=False)
        floor = torch.full_like(var_y, float(var_floor))
        L_var = torch.relu(floor - var_y).pow(2).sum()
        proj_cov_cond = _try_condition_number(cov_Y.detach() + float(eps) * torch.eye(
            cov_Y.shape[0], dtype=cov_Y.dtype, device=cov_Y.device,
        ))
    else:
        L_decorr = torch.zeros((), dtype=F_proj.dtype, device=F_proj.device)
        L_var = torch.zeros((), dtype=F_proj.dtype, device=F_proj.device)
        proj_cov_cond = float("nan")
        corr_stats = {
            "corr_offdiag_mean": 0.0,
            "corr_offdiag_max": 0.0,
            "corr_diag_mean": 0.0,
        }

    if projector is not None:
        L_reg = torch.zeros((), dtype=F_proj.dtype, device=F_proj.device)
        for p in projector.parameters():
            L_reg = L_reg + p.pow(2).sum()
    else:
        L_reg = torch.zeros((), dtype=F_proj.dtype, device=F_proj.device)

    total_loss = (
        L_det
        + float(lambda_decorr) * L_decorr
        + float(lambda_var) * L_var
        + float(lambda_reg) * L_reg
    )

    diagnostics = {
        "loss_total": float(total_loss.detach().item()),
        "loss_det": float(L_det.detach().item()),
        "loss_decorr": float(L_decorr.detach().item()),
        "loss_var": float(L_var.detach().item()),
        "loss_reg": float(L_reg.detach().item()),
        "pos_score_mean": float(pos_scores.detach().mean().item()) if pos_scores.numel() > 0 else 0.0,
        "neg_score_mean": float(neg_scores.detach().mean().item()) if neg_scores.numel() > 0 else 0.0,
        "bg_score_mean": float(bg_scores.detach().mean().item()) if bg_scores.numel() > 0 else 0.0,
        "proj_cov_cond": float(proj_cov_cond),
        "corr_offdiag_mean": float(corr_stats["corr_offdiag_mean"]),
        "corr_offdiag_max": float(corr_stats["corr_offdiag_max"]),
        "corr_diag_mean": float(corr_stats["corr_diag_mean"]),
        "ranking_margin": float(ranking_margin),
        "cov_info": cov_info,
    }
    return total_loss, diagnostics


def train_projector_for_scene(
    F_input_norm: torch.Tensor,
    pos_points: List[Tuple[int, int]],
    neg_points: List[Tuple[int, int]],
    *,
    hidden_dim: int = 32,
    out_dim: int = PROJECTOR_OUT_DIM,
    n_steps: int = 100,
    lr: float = 1e-3,
    cov_method: str = "ledoit_wolf",
    eta: float = CFG.eta,
    score_fn: str = "amf",
    ranking_margin: float = 0.2,
    lambda_decorr: float = 0.05,
    lambda_var: float = 0.05,
    lambda_reg: float = 1e-4,
    var_floor: float = 0.1,
    seed: int = 0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    if F_input_norm.ndim != 3:
        raise ValueError(f"F_input_norm must be (H,W,D), got {tuple(F_input_norm.shape)}")
    if len(pos_points) == 0:
        raise ValueError("Need at least one positive point")

    _set_seed(int(seed))
    in_dim = int(F_input_norm.shape[2])
    device = F_input_norm.device
    requested_out_dim = int(out_dim)
    projector = LearnableProjector(
        in_dim=in_dim,
        hidden_dim=int(hidden_dim),
        out_dim=requested_out_dim,
    ).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.Adam(projector.parameters(), lr=float(lr))

    loss_history: List[float] = []
    diag_history: List[Dict[str, float]] = []
    last_diag: Dict[str, Any] = {}
    loss_mode = (
        "positive_vs_negative"
        if len(neg_points) > 0
        else "positive_vs_background"
    )
    projector.train()

    total_steps = max(int(n_steps), 1)
    if progress_callback is not None:
        progress_callback({
            "state": "running",
            "step": 0,
            "n_steps": int(total_steps),
            "loss_history": [],
            "diag_history": [],
            "loss_mode": loss_mode,
            "ranking_margin": float(ranking_margin),
        })

    for _step in range(total_steps):
        optimizer.zero_grad(set_to_none=True)
        F_proj = _project_feature_map_impl(
            F_input_norm,
            projector,
            batch_rows=65536,
        )
        loss, diag = projector_training_loss(
            F_proj,
            pos_points,
            neg_points,
            cov_method=cov_method,
            eta=eta,
            score_fn=score_fn,
            ranking_margin=ranking_margin,
            lambda_decorr=lambda_decorr,
            lambda_var=lambda_var,
            lambda_reg=lambda_reg,
            projector=projector,
            bg_sample_limit=4096,
            var_floor=var_floor,
        )
        loss.backward()
        optimizer.step()

        last_diag = dict(diag)
        loss_history.append(float(loss.detach().item()))
        diag_history.append({
            "loss_total": float(diag["loss_total"]),
            "loss_det": float(diag["loss_det"]),
            "loss_decorr": float(diag["loss_decorr"]),
            "loss_var": float(diag["loss_var"]),
            "loss_reg": float(diag["loss_reg"]),
            "pos_score_mean": float(diag["pos_score_mean"]),
            "neg_score_mean": float(diag["neg_score_mean"]),
            "bg_score_mean": float(diag["bg_score_mean"]),
            "corr_offdiag_mean": float(diag["corr_offdiag_mean"]),
            "corr_offdiag_max": float(diag["corr_offdiag_max"]),
            "corr_diag_mean": float(diag["corr_diag_mean"]),
        })
        if progress_callback is not None:
            progress_callback({
                "state": "running",
                "step": int(_step + 1),
                "n_steps": int(total_steps),
                "loss_history": list(loss_history),
                "diag_history": list(diag_history),
                "loss_mode": loss_mode,
                "final": dict(last_diag),
                "ranking_margin": float(ranking_margin),
            })

    projector.eval()
    final_layer_weight_norm = float(projector.fc2.weight.detach().norm().item())
    train_info: Dict[str, Any] = {
        "seed": int(seed),
        "in_dim": in_dim,
        "hidden_dim": int(hidden_dim),
        "out_dim": int(requested_out_dim),
        "n_steps": int(max(int(n_steps), 1)),
        "lr": float(lr),
        "ranking_margin": float(ranking_margin),
        "loss_history": loss_history,
        "diag_history": diag_history,
        "loss_mode": loss_mode,
        "final": last_diag,
        "final_layer_weight_norm": final_layer_weight_norm,
        "param_count": int(sum(p.numel() for p in projector.parameters())),
        "loss_det": float(last_diag.get("loss_det", 0.0)),
        "loss_decorr": float(last_diag.get("loss_decorr", 0.0)),
        "loss_var": float(last_diag.get("loss_var", 0.0)),
        "loss_reg": float(last_diag.get("loss_reg", 0.0)),
        "proj_cov_cond": float(last_diag.get("proj_cov_cond", float("nan"))),
        "corr_offdiag_mean": float(last_diag.get("corr_offdiag_mean", 0.0)),
        "corr_offdiag_max": float(last_diag.get("corr_offdiag_max", 0.0)),
        "corr_diag_mean": float(last_diag.get("corr_diag_mean", 0.0)),
    }
    if requested_out_dim != PROJECTOR_OUT_DIM:
        train_info["requested_out_dim"] = requested_out_dim
    if progress_callback is not None:
        progress_callback({
            "state": "done",
            "step": int(total_steps),
            "n_steps": int(total_steps),
            "loss_history": list(loss_history),
            "diag_history": list(diag_history),
            "loss_mode": loss_mode,
            "final": dict(last_diag),
            "final_layer_weight_norm": final_layer_weight_norm,
            "ranking_margin": float(ranking_margin),
        })
    return projector, train_info


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
    projector_hidden_dim: int = 32,
    projector_out_dim: int = PROJECTOR_OUT_DIM,
    projector_steps: int = 100,
    projector_lr: float = 1e-3,
    projector_lambda_neg: float = 1.0,
    projector_lambda_bg: float = 0.5,
    projector_lambda_decorr: float = 0.05,
    projector_lambda_var: float = 0.05,
    projector_lambda_reg: float = 1e-4,
    projector_var_floor: float = 0.1,
    ranking_margin: float = 0.2,
    external_exclude_mask: Optional[np.ndarray] = None,
    prepared: Optional[Dict[str, Any]] = None,
    seed: int = 0,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> Tuple[torch.Tensor, np.ndarray, Dict[str, Any]]:
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

        prepared = _prepare_raw_projector_input(
            F_work,
            eps=eps,
            feature_norm_method=feature_norm_method,
        )
        prepared["prep_gpu_mem"] = _cuda_mem_stats(F_work.device)

    F_input_norm = prepared["F_input_norm"]
    if F_input_norm.device.type == "cuda":
        didx = F_input_norm.device.index if F_input_norm.device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(didx)

    with torch.enable_grad():
        projector, train_info = train_projector_for_scene(
            F_input_norm,
            pos_points,
            neg_points,
            hidden_dim=projector_hidden_dim,
            out_dim=projector_out_dim,
            n_steps=projector_steps,
            lr=projector_lr,
            cov_method=cov_method,
            eta=eta,
            score_fn=score_mode,
            ranking_margin=ranking_margin,
            lambda_decorr=projector_lambda_decorr,
            lambda_var=projector_lambda_var,
            lambda_reg=projector_lambda_reg,
            var_floor=projector_var_floor,
            seed=seed,
            progress_callback=progress_callback,
        )

    F_exp = apply_projector_to_feature_map(
        F_input_norm,
        projector,
        batch_rows=65536,
    )

    projected_dim = int(F_exp.shape[2])
    if projected_dim != projector_out_dim:
        raise RuntimeError(f"Projected dim must be {projector_out_dim}, got {projected_dim}")
    projected_labels = [_projected_feature_label(i) for i in range(projected_dim)]
    projected_meta = [{"type": "projected", "i": int(i)} for i in range(projected_dim)]

    pos_spectra_proj: List[List[float]] = []
    neg_spectra_proj: List[List[float]] = []
    pos_vecs_proj: List[torch.Tensor] = []
    for x, y in pos_points:
        s_proj = _sample(F_exp, x, y).to(dtype=torch.float32, device=F_exp.device)
        pos_vecs_proj.append(s_proj)
        pos_spectra_proj.append(s_proj.detach().cpu().float().tolist())
    for x, y in neg_points:
        s_proj = _sample(F_exp, x, y).to(dtype=torch.float32, device=F_exp.device)
        neg_spectra_proj.append(s_proj.detach().cpu().float().tolist())
    target_spectrum_proj = torch.stack(pos_vecs_proj, dim=0).mean(dim=0)

    H_, W_, D_exp = F_exp.shape
    cov_dtype = _cov_dtype_for_device(F_exp.device)
    n_pix = int(H_ * W_)
    feature_map_mb = (n_pix * D_exp * 4.0) / (1024.0 * 1024.0)
    cov_mat_mb_fp64 = (D_exp * D_exp * 8.0) / (1024.0 * 1024.0)

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
            [_sample(F_exp, x, y).to(dtype=cov_dtype, device=F_exp.device) for x, y in pos_points],
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
    cov_info["projected_dim"] = int(D_exp)
    cov_condition_number = _try_condition_number(R_inv)
    if np.isfinite(cov_condition_number):
        cov_info["cond_R"] = cov_condition_number

    mu_bg_spectrum_proj = mu_B.detach().cpu().float().tolist()
    score_dtype = torch.float32 if F_exp.device.type == "cuda" else R_inv.dtype
    R_inv_score = R_inv.to(dtype=score_dtype)
    mu_B_score = mu_B.to(dtype=score_dtype)

    pos_vecs_score: List[torch.Tensor] = []
    pos_spectra_projected_for_state: List[List[float]] = []
    for x, y in pos_points:
        s_vec = _sample(F_exp, x, y).to(dtype=score_dtype, device=F_exp.device)
        pos_vecs_score.append(s_vec)
        pos_spectra_projected_for_state.append(s_vec.detach().cpu().float().tolist())

    target = torch.stack(pos_vecs_score, dim=0).mean(dim=0)
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
    lo_h = float(s_np.min())
    hi_h = float(s_np.max())
    if hi_h - lo_h > 1e-12:
        n_bins = 80
        hist_counts = torch.histc(s_np.flatten(), bins=n_bins, min=lo_h, max=hi_h)
        bin_width = (hi_h - lo_h) / n_bins
        hist_centers = [lo_h + (i + 0.5) * bin_width for i in range(n_bins)]
        score_hist = {
            "hist_counts": hist_counts.tolist(),
            "hist_centers": hist_centers,
        }

    final_diag = dict(train_info.get("final") or {})
    state: Dict[str, Any] = {
        "n_pos": len(pos_points),
        "n_neg": len(neg_points),
        "scoring": f"new_method_mlp_{score_mode}",
        "detector": score_mode,
        "cov_info": cov_info,
        "cov_condition_number": cov_condition_number,
        "mu_B": mu_B.detach().cpu().float().tolist(),
        "w_effective": w_vec.detach().cpu().float().tolist(),
        "target_spectrum": target.detach().cpu().float().tolist(),
        "target_spectrum_proj": target_spectrum_proj.detach().cpu().float().tolist(),
        "pos_spectra_raw": pos_spectra_projected_for_state,
        "pos_spectra_proj": pos_spectra_proj,
        "neg_spectra_proj": neg_spectra_proj,
        "mu_bg_spectrum_proj": mu_bg_spectrum_proj,
        "neg_spectra_raw": [],
        "score_hist": score_hist,
        "exclude_mask": exclude_mask_hw,
        "threshold_used": float(used_threshold),
        "r_inv_matrix": R_inv_vis.tolist(),
        "r_inv_dim": int(R_inv.shape[0]),
        "r_inv_dim_shown": int(R_inv_vis.shape[0]),
        "use_feature_expansion": False,
        "projector_used": True,
        "projector_hidden_dim": int(projector_hidden_dim),
        "projector_out_dim": int(projected_dim),
        "projector_steps": int(projector_steps),
        "projector_lr": float(projector_lr),
        "ranking_margin": float(ranking_margin),
        "projector_train_info": train_info,
        "projector_final_layer_weight_norm": float(train_info["final_layer_weight_norm"]),
        "projector_last_layer_weight_norm": float(train_info["final_layer_weight_norm"]),
        "feature_bank_size": int(prepared["feature_bank_size"]),
        "raw_input_dim": int(prepared["raw_input_dim"]),
        "projected_dim": int(projected_dim),
        "feature_selected_idx": list(range(projected_dim)),
        "feature_selected_meta": projected_meta,
        "feature_selected_labels": projected_labels,
        "feature_selected_types": ["projected"] * projected_dim,
        "feature_selection_mode": "learnable_projector_raw_input",
        "feature_selected_dim": int(projected_dim),
        "feature_norm_method": prepared["feature_norm_method"],
        "feature_norm_summary": prepared["feature_norm_summary"],
        "norm_stats_summary": prepared["feature_norm_summary"],
        "feature_used_neg_fallback": False,
        "feature_scoring_mode": "pairwise_ranking_detector_aware_projector",
        "feature_weighting_mode": "projector",
        "feature_expand_mode": "raw_input_projector",
        "n_identity_features": int(prepared["n_identity_features"]),
        "n_diff_features": int(prepared["n_diff_features"]),
        "loss_det": float(final_diag.get("loss_det", 0.0)),
        "loss_decorr": float(final_diag.get("loss_decorr", 0.0)),
        "loss_var": float(final_diag.get("loss_var", 0.0)),
        "loss_reg": float(final_diag.get("loss_reg", 0.0)),
        "proj_cov_cond": float(final_diag.get("proj_cov_cond", float("nan"))),
        "corr_offdiag_mean": float(final_diag.get("corr_offdiag_mean", 0.0)),
        "corr_offdiag_max": float(final_diag.get("corr_offdiag_max", 0.0)),
        "corr_diag_mean": float(final_diag.get("corr_diag_mean", 0.0)),
        "memory_estimate": {
            "n_pixels": n_pix,
            "feature_bank_dim": int(prepared["feature_bank_size"]),
            "d_exp": int(D_exp),
            "projected_dim": int(projected_dim),
            "feature_map_mb_fp32": float(round(feature_map_mb, 3)),
            "cov_matrix_mb_fp64": float(round(cov_mat_mb_fp64, 3)),
        },
        "gpu_mem": {
            "prep": prepared.get("prep_gpu_mem") or {},
            "run": _cuda_mem_stats(F_exp.device),
        },
        "deprecated_params_ignored": {
            "feature_expand_mode": str(feature_expand_mode),
            "feature_clip_ratio": float(feature_clip_ratio),
            "feature_score_beta_bg": float(feature_score_beta_bg),
            "feature_weight_floor": float(feature_weight_floor),
            "feature_weight_gamma": float(feature_weight_gamma),
            "include_identity": bool(include_identity),
            "include_diff": bool(include_diff),
            "include_normdiff": bool(include_normdiff),
            "include_ratio": bool(include_ratio),
            "include_product": bool(include_product),
            "include_log": bool(include_log),
            "include_square": bool(include_square),
            "projector_lambda_neg": float(projector_lambda_neg),
            "projector_lambda_bg": float(projector_lambda_bg),
            "projector_out_dim_requested": int(projector_out_dim),
        },
    }
    return score_map, mask, state


class _NewMethodMLPBase:
    """Webapp wrapper with normalized raw-input cache per uploaded cube."""

    name = "new_method_mlp_base"

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
        projector_hidden_dim: int = 32,
        projector_out_dim: int = PROJECTOR_OUT_DIM,
        projector_steps: int = 100,
        projector_lr: float = 1e-3,
        projector_lambda_neg: float = 1.0,
        projector_lambda_bg: float = 0.5,
        projector_lambda_decorr: float = 0.05,
        projector_lambda_var: float = 0.05,
        projector_lambda_reg: float = 1e-4,
        projector_var_floor: float = 0.1,
        ranking_margin: float = 0.2,
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

        self.projector_hidden_dim = int(projector_hidden_dim)
        self.projector_out_dim = int(projector_out_dim)
        self.projector_out_dim_requested = int(projector_out_dim)
        self.projector_steps = int(projector_steps)
        self.projector_lr = float(projector_lr)
        self.projector_lambda_neg = float(projector_lambda_neg)
        self.projector_lambda_bg = float(projector_lambda_bg)
        self.projector_lambda_decorr = float(projector_lambda_decorr)
        self.projector_lambda_var = float(projector_lambda_var)
        self.projector_lambda_reg = float(projector_lambda_reg)
        self.projector_var_floor = float(projector_var_floor)
        self.ranking_margin = float(ranking_margin)

        self._cache_key: Optional[Tuple[Any, ...]] = None
        self._cache_prepared: Optional[Dict[str, Any]] = None

    def _prepared_recipe(self) -> Dict[str, Any]:
        return {
            "input_mode": "raw_input",
            "feature_norm_method": self.feature_norm_method,
        }

    def _prepared_matches_recipe(self, prepared: Dict[str, Any]) -> bool:
        return dict(prepared.get("feature_recipe") or {}) == self._prepared_recipe()

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
            "raw_input_projector_v2",
            self.feature_norm_method,
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

        prepared = _prepare_raw_projector_input(
            F,
            eps=CFG.eps,
            feature_norm_method=self.feature_norm_method,
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
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
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
            projector_hidden_dim=self.projector_hidden_dim,
            projector_out_dim=self.projector_out_dim,
            projector_steps=self.projector_steps,
            projector_lr=self.projector_lr,
            projector_lambda_neg=self.projector_lambda_neg,
            projector_lambda_bg=self.projector_lambda_bg,
            projector_lambda_decorr=self.projector_lambda_decorr,
            projector_lambda_var=self.projector_lambda_var,
            projector_lambda_reg=self.projector_lambda_reg,
            projector_var_floor=self.projector_var_floor,
            ranking_margin=self.ranking_margin,
            external_exclude_mask=None,
            prepared=prepared,
            progress_callback=progress_callback,
        )
        return {
            "mask": mask,
            "score_map": score_t.detach().cpu().numpy().astype(np.float32),
            "state": state,
            "threshold": state.get("threshold_used"),
        }


class NewMethodMLP(_NewMethodMLPBase):
    name = "new_method_mlp"

    def __init__(self, **kwargs):
        super().__init__(detector="amf", **kwargs)


class NewMethodMLPACE(_NewMethodMLPBase):
    name = "new_method_mlp_ace"

    def __init__(self, **kwargs):
        super().__init__(detector="ace", **kwargs)
