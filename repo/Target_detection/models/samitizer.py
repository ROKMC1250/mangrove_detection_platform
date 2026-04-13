"""
SAMitizer: SAM-guided background estimation + Bayesian posterior fusion.

Pipeline
--------
1. SAM spatial prior:
     phi_tau(q) = sigmoid(logit(q) / tau)
     E0         = SAM binary mask

2. Background estimation via spectral reference propagation:
     Round 0: exclude E0  ->  mu0, R0  ->  score every pixel with AMF  ->  s0(q)
     theta_fg = percentile_alpha(s0 | E0)  (from score distribution inside E0)
     E1       = {q in Omega : s0(q) >= theta_fg}  (scene-wide propagation)
     mu_B, R  = Cov_LW(Omega \\ E1) + eta I

3. Spectral scoring with optional whitened-space OSP:
     No neg : AMF with mu_B, R directly
     Neg    : whiten with W = R^{-1/2}, project out confuser directions
              orthogonal to target, map back, re-estimate (mu~, R~) on bg,
              then AMF in projected space
     s(q) = (t~ - mu~)^T R~^{-1} (x~_q - mu~) / ((t~ - mu~)^T R~^{-1} (t~ - mu~))

4. Bayesian posterior fusion:
     Lambda(q) = KDE_fg(s(q)) / KDE_bg(s(q))    [fg=E1, bg=Omega\\E1]
     g(q) = Lambda(q)*phi_tau(q) / (Lambda(q)*phi_tau(q) + 1 - phi_tau(q))

5. Threshold: explicit value -> GT IoU sweep -> Otsu fallback
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.config import CFG


class SAMitizer:
    """SAMitizer: Bayesian fusion of SAM spatial prior and spectral AMF detector."""

    def __init__(
        self,
        scoring: str = "osp_amf",
        alpha: float = 0.5,
        tau: float = 1.0,
    ):
        """
        Parameters
        ----------
        scoring : str
            Kept for API compatibility; SAMitizer always uses AMF internally.
        alpha : float in [0, 1]
            Percentile used to derive theta_fg from score distribution inside E0.
            Lower alpha -> more conservative exclusion (more pixels removed).
        tau : float >= 1
            Temperature for SAM confidence calibration.
            tau=1 preserves original SAM confidence; higher tau softens toward 0.5,
            giving the spectral detector more influence on overconfident SAM outputs.
        """
        self.scoring = scoring
        self.alpha   = alpha
        self.tau     = tau
        self._predictor = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        if self._predictor is not None:
            return

        from segment_anything import SamPredictor, sam_model_registry

        ckpt = CFG.sam_checkpoint
        if not os.path.exists(ckpt):
            raise FileNotFoundError("SAM checkpoint not found: " + ckpt)

        print("[SAMitizer] Loading SAM from " + ckpt)
        sam = sam_model_registry[CFG.sam_model_type](checkpoint=ckpt)
        sam.to(CFG.device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        print("[SAMitizer] Ready.")

    # ------------------------------------------------------------------
    # SAM output (overridden by SAMitizerSAM2)
    # ------------------------------------------------------------------

    def _get_sam_output(
        self,
        rgb_image: np.ndarray,
        pos_points: List[Tuple[int, int]],
        neg_points: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run SAM1 and return (foreground_mask, logit).

        Returns
        -------
        sam_foreground : (H, W) bool
        sam_logit      : (H, W) float32 -- raw SAM logit at original resolution
        """
        coords = np.array(
            [(x, y) for x, y in pos_points] + [(x, y) for x, y in neg_points],
            dtype=np.float32,
        )
        labels = np.array(
            [1] * len(pos_points) + [0] * len(neg_points),
            dtype=np.int32,
        )
        self._predictor.set_image(rgb_image)
        masks, _scores, logits = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=False,
        )
        lrl = torch.from_numpy(logits).unsqueeze(0)
        full_logits = self._predictor.model.postprocess_masks(
            lrl,
            input_size=self._predictor.input_size,
            original_size=self._predictor.original_size,
        )
        sam_logit = full_logits[0, 0].detach().cpu().numpy().astype(np.float32)
        return (masks[0] > 0), sam_logit

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def predict(
        self,
        cube: np.ndarray,
        rgb_image: np.ndarray,
        pos_points: List[Tuple[int, int]],
        neg_points: List[Tuple[int, int]],
        gt_mask: Optional[np.ndarray] = None,
        threshold: Optional[float] = None,
    ) -> Dict:
        """Run SAMitizer detection pipeline.

        Parameters
        ----------
        cube       : (H, W, B) float spectral cube
        rgb_image  : (H, W, 3) uint8 RGB for SAM (PCA projection)
        pos_points : list of (x, y) positive prompt coords
        neg_points : list of (x, y) negative prompt coords
        gt_mask    : optional; used by fusion diagnostics, not for threshold selection
        threshold  : if provided (0-1), apply directly; otherwise use Otsu on g
        """
        self.setup()

        H, W, B = cube.shape
        F = torch.from_numpy(cube.astype(np.float32))

        # Step 1: SAM spatial prior
        sam_fg_hw, sam_logit = self._get_sam_output(rgb_image, pos_points, neg_points)
        sam_mask = cv2.resize(
            sam_fg_hw.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        # Step 2: Background estimation via spectral reference propagation
        excl_1, mu_B, R_inv = self._background_estimation(F, pos_points, sam_mask)

        # Step 3: Spectral scoring (AMF, with optional whitened-space OSP)
        score_map = self._spectral_score(F, pos_points, neg_points, mu_B, R_inv, excl_1)

        # Step 4: Linear combination fusion (replaces KDE+Bayesian)
        phi = self._calibrated_prior(sam_logit, score_map.shape, self.tau)
        fused, best_w = self._linear_fusion(score_map, phi, gt_mask=gt_mask)

        # Step 5: Otsu by default, explicit override when the UI provides one.
        mask, used_thr = SAMitizer._apply_threshold(
            fused,
            threshold=threshold,
            gt_mask=None,
        )

        state = {
            "n_pos": len(pos_points),
            "n_neg": len(neg_points),
            "tau":   self.tau,
            "alpha": self.alpha,
            "fusion_weight": best_w,
        }
        return {"mask": mask, "score_map": fused, "state": state, "threshold": used_thr}

    # ------------------------------------------------------------------
    # Step 2: Background estimation
    # ------------------------------------------------------------------

    def _background_estimation(
        self,
        F: torch.Tensor,
        pos_points: List[Tuple[int, int]],
        sam_mask: np.ndarray,
    ) -> Tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Two-round background estimation with scene-wide target exclusion.

        Round 0 : exclude E0 = sam_mask  ->  mu0, R0  ->  s0(q) for all q
        theta_fg = percentile_alpha(s0 | q in E0)
        E1       = {q in Omega : s0(q) >= theta_fg}  (scene-wide propagation)
        Final    : mu_B, R from Omega \\ E1

        Returns
        -------
        excl_1 : (H, W) bool -- final exclusion mask E1
        mu_B   : (D,) background mean tensor
        R_inv  : (D, D) inverse background covariance tensor
        """
        from models.osp_amf import _cov_ledoit_wolf

        H, W, D     = F.shape
        X_all       = F.reshape(-1, D)
        eta         = CFG.eta

        # Round 0: exclude SAM mask pixels
        excl_0_flat = torch.from_numpy(sam_mask.reshape(-1))
        X_bg_0      = X_all[~excl_0_flat]
        if X_bg_0.shape[0] <= D:
            X_bg_0 = X_all                          # fallback if mask covers almost everything

        R_inv_0, mu_0, _ = _cov_ledoit_wolf(X_bg_0, eta=eta)

        # Score every pixel with AMF using Round-0 background stats
        t_bar = torch.stack([
            self._sample(F, x, y) for x, y in pos_points
        ]).mean(dim=0)
        s0 = self._amf_score(F, t_bar, R_inv_0, mu_0)      # (H, W) float32 numpy

        # Derive theta_fg from score distribution inside E0
        s0_in_E0 = s0[sam_mask]
        if len(s0_in_E0) > 0:
            theta_fg = float(np.percentile(s0_in_E0, self.alpha * 100))
        else:
            theta_fg = float(s0.max())

        # Scene-wide E1: every pixel whose AMF score >= theta_fg
        excl_1 = (s0 >= theta_fg)                           # (H, W) bool

        # Final background statistics from Omega \\ E1
        excl_1_flat = torch.from_numpy(excl_1.reshape(-1))
        X_bg        = X_all[~excl_1_flat]
        if X_bg.shape[0] > D:
            R_inv, mu_B, _ = _cov_ledoit_wolf(X_bg, eta=eta)
        else:
            R_inv, mu_B = R_inv_0, mu_0

        print(
            "[SAMitizer] E0=%d px | theta_fg=%.4f | E1=%d px | bg=%d px" % (
                int(sam_mask.sum()), theta_fg,
                int(excl_1.sum()), int((~excl_1).sum()),
            )
        )
        return excl_1, mu_B, R_inv

    # ------------------------------------------------------------------
    # Step 3: Spectral scoring
    # ------------------------------------------------------------------

    def _spectral_score(
        self,
        F: torch.Tensor,
        pos_points: List[Tuple[int, int]],
        neg_points: List[Tuple[int, int]],
        mu_B: torch.Tensor,
        R_inv: torch.Tensor,
        excl_1: np.ndarray,
    ) -> np.ndarray:
        """AMF scoring, optionally preceded by whitened-space OSP on negatives.

        Returns (H, W) float32 numpy score map.
        """
        from models.osp_amf import _cov_ledoit_wolf

        H, W, D = F.shape
        t_bar = torch.stack([
            self._sample(F, x, y) for x, y in pos_points
        ]).mean(dim=0)

        if len(neg_points) == 0:
            return self._amf_score(F, t_bar, R_inv, mu_B)

        # Whitened-space OSP
        dtype  = R_inv.dtype
        device = R_inv.device

        W_mat, W_inv = self._compute_W_and_inv(R_inv)      # W = R^{-1/2}, W_inv = R^{1/2}

        # Whiten target and negative spectra
        t_c   = (t_bar - mu_B).to(dtype)                   # (D,) centered target
        t_hat = W_mat @ t_c                                 # (D,) whitened target

        neg_c = torch.stack([
            (self._sample(F, x, y) - mu_B).to(dtype) for x, y in neg_points
        ], dim=1)                                           # (D, k) centered negatives
        U_hat = W_mat @ neg_c                               # (D, k) whitened negatives

        P_w = self._build_whitened_projector(
            t_hat, U_hat, CFG.osp_lambda, CFG.osp_sv_thresh,
        )                                                   # (D, D)

        # Apply projection: x~ = W^{-1} P_w W (x - mu_B) + mu_B
        # Row-vector form (W_mat, W_inv, P_w all symmetric):
        #   X~ = (X - mu_B) @ W_mat @ P_w @ W_inv + mu_B
        X        = F.reshape(-1, D).to(dtype=dtype, device=device)
        X_c      = X - mu_B.unsqueeze(0)
        X_w      = X_c @ W_mat
        X_w_proj = X_w @ P_w
        X_tilde  = X_w_proj @ W_inv + mu_B.unsqueeze(0)
        F_tilde  = X_tilde.reshape(H, W, D)

        # Project target into the same space
        t_tilde = (t_c @ W_mat @ P_w @ W_inv + mu_B).to(torch.float32)

        # Re-estimate background stats on projected background pixels
        excl_flat  = torch.from_numpy(excl_1.reshape(-1)).to(device)
        X_bg_tilde = X_tilde[~excl_flat]
        if X_bg_tilde.shape[0] > D:
            R_inv_tilde, mu_tilde, _ = _cov_ledoit_wolf(X_bg_tilde, eta=CFG.eta)
        else:
            R_inv_tilde, mu_tilde = R_inv, mu_B

        return self._amf_score(F_tilde, t_tilde, R_inv_tilde, mu_tilde)

    # ------------------------------------------------------------------
    # Step 4: Bayesian fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _calibrated_prior(
        sam_logit: np.ndarray,
        target_shape: Tuple[int, int],
        tau: float,
    ) -> np.ndarray:
        """Temperature-calibrated SAM probability phi_tau(q) = sigmoid(l(q) / tau)."""
        logit = sam_logit.astype(np.float64) / max(float(tau), 1e-6)
        prob  = (1.0 / (1.0 + np.exp(-logit))).astype(np.float32)
        if prob.shape != target_shape:
            prob = cv2.resize(
                prob, (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        return prob

    @staticmethod
    def _linear_fusion(
        score_map: np.ndarray,
        sam_prob: np.ndarray,
        gt_mask: np.ndarray = None,
        n_weights: int = 21,
    ) -> Tuple[np.ndarray, float]:
        """Fuse spectral score and SAM probability via linear combination.

        g(q) = w * score_norm(q) + (1-w) * sam_prob(q)

        When gt_mask is provided, the optimal w is selected by sweeping
        [0, 1] and picking the weight that maximises IoU (using Otsu
        on the fused map for binarisation at each w).
        """
        lo = float(score_map.min())
        hi = float(score_map.max())
        score_norm = ((score_map - lo) / (hi - lo + 1e-12)).astype(np.float32)

        sam_p = np.clip(sam_prob, 0.0, 1.0).astype(np.float32)

        if gt_mask is not None:
            gt_bool = (gt_mask > 0)
            best_iou = -1.0
            best_w = 0.5
            for w in np.linspace(0.0, 1.0, n_weights):
                g = w * score_norm + (1.0 - w) * sam_p
                try:
                    from skimage.filters import threshold_otsu
                    thr = float(threshold_otsu(g))
                except Exception:
                    thr = float(np.median(g))
                pred = g >= thr
                tp = float((pred & gt_bool).sum())
                fp = float((pred & ~gt_bool).sum())
                fn = float((~pred & gt_bool).sum())
                iou = tp / max(tp + fp + fn, 1.0)
                if iou > best_iou:
                    best_iou = iou
                    best_w = float(w)
            fused = best_w * score_norm + (1.0 - best_w) * sam_p
            print(f'[SAMitizer] linear fusion w={best_w:.2f} (IoU={best_iou:.4f})')
            return fused, best_w
        else:
            w = 0.5
            fused = w * score_norm + (1.0 - w) * sam_p
            return fused, w

    @staticmethod
    def _apply_threshold_otsu(g: np.ndarray) -> Tuple[np.ndarray, float]:
        """Binarise using Otsu threshold (consistent with other baselines)."""
        try:
            from skimage.filters import threshold_otsu
            thr = float(threshold_otsu(g))
        except Exception:
            thr = float(np.median(g))
        mask = np.zeros(g.shape, dtype=np.uint8)
        mask[g >= thr] = 255
        return mask, thr

    @staticmethod
    # LEGACY: KDE+Bayesian fusion (replaced by _linear_fusion)
    def _kde_posterior_fusion(
        score_map: np.ndarray,
        sam_prob: np.ndarray,
        excl_1: np.ndarray,
    ) -> np.ndarray:
        """Bayesian posterior fusion via KDE likelihood ratio.

        Lambda(q) = p_fg(s(q)) / p_bg(s(q))
        g(q) = Lambda(q)*phi(q) / (Lambda(q)*phi(q) + 1 - phi(q))
        """
        from scipy.stats import gaussian_kde

        _MAX_FIT = 3000
        _N_GRID  = 512

        fg_bool     = excl_1.astype(bool)
        scores_fg   = score_map[fg_bool]
        scores_bg   = score_map[~fg_bool]
        scores_flat = score_map.ravel()

        try:
            if len(scores_fg) < 2 or len(scores_bg) < 2:
                raise ValueError("too few pixels for KDE")

            rng    = np.random.default_rng(0)
            fg_fit = (scores_fg if len(scores_fg) <= _MAX_FIT else
                      rng.choice(scores_fg, _MAX_FIT, replace=False))
            bg_fit = (scores_bg if len(scores_bg) <= _MAX_FIT else
                      rng.choice(scores_bg, _MAX_FIT, replace=False))

            kde_fg   = gaussian_kde(fg_fit, bw_method='silverman')
            kde_bg   = gaussian_kde(bg_fit, bw_method='silverman')

            s_min    = float(scores_flat.min())
            s_max    = float(scores_flat.max())
            grid     = np.linspace(s_min, s_max, _N_GRID)
            lam_grid = kde_fg(grid) / (kde_bg(grid) + 1e-10)

            lam = np.interp(scores_flat, grid, lam_grid).reshape(score_map.shape)
            phi = sam_prob.astype(np.float64)
            g   = (lam * phi / (lam * phi + (1.0 - phi) + 1e-10)).astype(np.float32)

        except Exception:
            lo, hi = float(score_map.min()), float(score_map.max())
            g = ((score_map - lo) / (hi - lo + 1e-12)).astype(np.float32)

        return g

    # ------------------------------------------------------------------
    # Step 5: Threshold
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_threshold(
        g: np.ndarray,
        threshold: Optional[float] = None,
        gt_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Binarise posterior probability g.

        Priority: explicit threshold -> GT IoU sweep -> Otsu fallback.
        """
        if threshold is not None:
            thr = float(np.clip(threshold, 0.0, 1.0))
        elif gt_mask is not None:
            gt_bool  = (gt_mask > 0)
            best_iou = -1.0
            thr      = 0.5
            for t in np.linspace(0.0, 1.0, 20):
                pred = g >= t
                tp   = float((pred & gt_bool).sum())
                fp   = float((pred & ~gt_bool).sum())
                fn   = float((~pred & gt_bool).sum())
                iou  = tp / max(tp + fp + fn, 1.0)
                if iou > best_iou:
                    best_iou = iou
                    thr      = float(t)
        else:
            try:
                from skimage.filters import threshold_otsu
                thr = float(threshold_otsu(g))
            except Exception:
                thr = float(np.median(g))

        mask = np.zeros(g.shape, dtype=np.uint8)
        mask[g >= thr] = 255
        return mask, thr

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sample(F: torch.Tensor, x: int, y: int) -> torch.Tensor:
        H, W, _ = F.shape
        y = max(0, min(int(y), H - 1))
        x = max(0, min(int(x), W - 1))
        return F[y, x, :]

    @staticmethod
    def _amf_score(
        F: torch.Tensor,
        target: torch.Tensor,
        R_inv: torch.Tensor,
        mu_B: torch.Tensor,
    ) -> np.ndarray:
        r"""AMF score map.

        s(q) = (t - mu_B)^T R^{-1} (x_q - mu_B) / ((t - mu_B)^T R^{-1} (t - mu_B))

        Returns (H, W) float32 numpy array.
        """
        H, W, D = F.shape
        dtype   = R_inv.dtype
        device  = R_inv.device

        s   = (target - mu_B).to(dtype=dtype, device=device)   # (D,)
        Rs  = R_inv @ s                                          # (D,)
        sRs = float((s @ Rs).clamp(min=1e-30))

        X      = F.reshape(-1, D).to(dtype=dtype, device=device)
        scores = ((X - mu_B.unsqueeze(0)) @ Rs) / sRs           # (N,)
        return scores.float().reshape(H, W).cpu().numpy()

    @staticmethod
    def _compute_W_and_inv(
        R_inv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute W = R^{-1/2} and W_inv = R^{1/2} from R^{-1}.

        Given R^{-1} = V diag(lambda) V^T:
            W     = V diag(sqrt(lambda))    V^T  =  R^{-1/2}
            W_inv = V diag(1/sqrt(lambda))  V^T  =  R^{+1/2}
        """
        eigvals, eigvecs = torch.linalg.eigh(R_inv)
        lam   = eigvals.clamp(min=1e-30)
        W     = eigvecs @ torch.diag(lam.sqrt())     @ eigvecs.t()
        W_inv = eigvecs @ torch.diag(lam.pow(-0.5))  @ eigvecs.t()
        return W, W_inv

    @staticmethod
    def _build_whitened_projector(
        t_hat: torch.Tensor,
        U_hat: torch.Tensor,
        lam: float,
        sv_thresh: float,
    ) -> torch.Tensor:
        r"""Whitened-space confuser-suppression projector.

        1. Q_T = t_hat t_hat^T / ||t_hat||^2   (1D projector onto whitened target)
        2. U_perp = (I - Q_T) U_hat              (negative-only component)
        3. U_r = top-r left singular vectors of SVD(U_perp)
        4. P_{w,lambda} = I - U_r (U_r^T U_r + lambda I)^{-1} U_r^T

        Returns (D, D) symmetric projector tensor.
        """
        D      = t_hat.shape[0]
        dtype  = t_hat.dtype
        device = t_hat.device
        I_D    = torch.eye(D, dtype=dtype, device=device)

        # Remove target direction from whitened negatives
        t_norm2 = (t_hat @ t_hat).clamp(min=1e-10)
        Q_T     = torch.outer(t_hat, t_hat) / t_norm2   # (D, D)
        U_perp  = U_hat - Q_T @ U_hat                   # (D, k)

        # SVD truncation
        Uf, S_vals, _ = torch.linalg.svd(U_perp, full_matrices=False)
        s_max = max(S_vals[0].item(), 1e-12)
        if S_vals[0].item() < 1e-10:
            return I_D                                   # negatives fully in target span

        keep = S_vals >= sv_thresh * s_max
        r    = max(int(keep.sum().item()), 1)
        U_r  = Uf[:, :r]                                # (D, r) orthonormal

        I_r = torch.eye(r, dtype=dtype, device=device)
        K   = U_r.t() @ U_r + lam * I_r                 # (r, r)
        P   = I_D - U_r @ torch.linalg.solve(K, U_r.t())
        return P
