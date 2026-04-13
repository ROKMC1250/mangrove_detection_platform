"""new_method_bilinear: detector-aware bilinear projector + direct AMF/ACE."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import nn

from models.config import CFG
from models.new_method import (
    _MAX_RINV_VIS_DIM,
    _cuda_mem_stats,
    _cov_dtype_for_device,
    _estimate_covariance_stable,
    _get_compute_device,
    _subsample_rows,
    _try_condition_number,
)
from models.new_method_mlp import (
    PROJECTOR_OUT_DIM as _MLP_PROJECTOR_OUT_DIM,
    _prepare_raw_projector_input,
    _project_feature_map_impl,
    _projected_feature_label,
    _set_seed,
    apply_projector_to_feature_map,
    projector_training_loss,
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

PROJECTOR_OUT_DIM = _MLP_PROJECTOR_OUT_DIM


class LearnableProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        out_dim: int = PROJECTOR_OUT_DIM,
        use_layernorm: bool = True,
        activation: str = "relu",
        bilinear_rank: int = 16,
        bilinear_alpha: float = 1.0,
    ):
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got {in_dim}")
        if hidden_dim <= 0 or out_dim <= 0:
            raise ValueError("hidden_dim/out_dim must be > 0")
        if bilinear_rank <= 0:
            raise ValueError(f"bilinear_rank must be > 0, got {bilinear_rank}")

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
        self.bilinear_rank = int(bilinear_rank)
        self.bilinear_alpha = float(bilinear_alpha)

        self.input_norm = nn.LayerNorm(in_dim) if use_layernorm else nn.Identity()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        # Low-rank bilinear branch captures multiplicative channel relations.
        self.bilinear_q = nn.Linear(in_dim, self.bilinear_rank, bias=False)
        self.bilinear_k = nn.Linear(in_dim, self.bilinear_rank, bias=False)
        self.bilinear_out = nn.Linear(self.bilinear_rank, hidden_dim, bias=True)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.input_norm(x)
        h_lin = self.fc1(x_norm)

        # Fuse a low-rank relation term before the output bottleneck.
        q = self.bilinear_q(x_norm)
        k = self.bilinear_k(x_norm)
        r = q * k
        h_bil = self.bilinear_out(r)

        h = h_lin + self.bilinear_alpha * h_bil
        y = self.act(h)
        z = self.fc2(y)
        return z


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
    bilinear_rank: int = 16,
    bilinear_alpha: float = 1.0,
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
        bilinear_rank=int(bilinear_rank),
        bilinear_alpha=float(bilinear_alpha),
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
        "bilinear_rank": int(bilinear_rank),
        "bilinear_alpha": float(bilinear_alpha),
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
    projector_bilinear_rank: int = 16,
    projector_bilinear_alpha: float = 1.0,
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
            bilinear_rank=projector_bilinear_rank,
            bilinear_alpha=projector_bilinear_alpha,
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
        "scoring": f"new_method_bilinear_{score_mode}",
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
        "projector_bilinear_rank": int(projector_bilinear_rank),
        "projector_bilinear_alpha": float(projector_bilinear_alpha),
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
        "feature_selection_mode": "learnable_bilinear_projector_raw_input",
        "feature_selected_dim": int(projected_dim),
        "feature_norm_method": prepared["feature_norm_method"],
        "feature_norm_summary": prepared["feature_norm_summary"],
        "norm_stats_summary": prepared["feature_norm_summary"],
        "feature_used_neg_fallback": False,
        "feature_scoring_mode": "pairwise_ranking_detector_aware_bilinear_projector",
        "feature_weighting_mode": "projector",
        "feature_expand_mode": "raw_input_bilinear_projector",
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


class _NewMethodBilinearBase:
    """Webapp wrapper with normalized raw-input cache per uploaded cube."""

    name = "new_method_bilinear_base"

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
        projector_bilinear_rank: int = 16,
        projector_bilinear_alpha: float = 1.0,
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
        self.projector_bilinear_rank = int(projector_bilinear_rank)
        self.projector_bilinear_alpha = float(projector_bilinear_alpha)
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
            "raw_input_bilinear_projector_v1",
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
            projector_bilinear_rank=self.projector_bilinear_rank,
            projector_bilinear_alpha=self.projector_bilinear_alpha,
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


class NewMethodBilinear(_NewMethodBilinearBase):
    name = "new_method_bilinear"

    def __init__(self, **kwargs):
        super().__init__(detector="amf", **kwargs)


class NewMethodBilinearACE(_NewMethodBilinearBase):
    name = "new_method_bilinear_ace"

    def __init__(self, **kwargs):
        super().__init__(detector="ace", **kwargs)
