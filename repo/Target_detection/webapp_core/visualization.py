"""Visualization helpers shared by web_app routes."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

SEGMENTATION_OVERLAY_RGB = (255, 0, 255)
SEGMENTATION_TP_RGB = (255, 0, 255)
SEGMENTATION_FP_RGB = (200, 0, 0)
SEGMENTATION_FN_RGB = (0, 0, 200)


def _rgb_to_b64(img_rgb: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return base64.b64encode(buf).decode("ascii")


def _score_map_to_heatmap_rgb(score_map: np.ndarray) -> np.ndarray:
    """Convert a score map to an RGB heatmap at the same resolution."""
    s = np.asarray(score_map, dtype=np.float32)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    smin = float(s.min()) if s.size else 0.0
    smax = float(s.max()) if s.size else 1.0
    if smax - smin < 1e-10:
        norm_u8 = np.zeros_like(s, dtype=np.uint8)
    else:
        norm = (s - smin) / (smax - smin)
        norm_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)


def _to_heatmap_b64(score_map: np.ndarray, out_h: int, out_w: int) -> str:
    """Convert score map -> JET heatmap -> base64 PNG."""
    heat_rgb = _score_map_to_heatmap_rgb(score_map)
    if (heat_rgb.shape[0], heat_rgb.shape[1]) != (out_h, out_w):
        heat_rgb = cv2.resize(heat_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return _rgb_to_b64(heat_rgb)


def _mask_overlay_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int] = SEGMENTATION_OVERLAY_RGB,
    alpha: float = 0.42,
) -> np.ndarray:
    """Overlay a binary mask on an RGB image at the base image resolution."""
    base = np.asarray(rgb, dtype=np.uint8).copy()
    m = (np.asarray(mask) > 0)
    if m.shape != base.shape[:2]:
        m = cv2.resize(
            m.astype(np.uint8),
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    color = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    blended = base.astype(np.float32)
    blended[m] = (1.0 - alpha) * blended[m] + alpha * color
    return np.clip(blended, 0, 255).astype(np.uint8)


def _mask_overlay_b64(
    rgb: np.ndarray,
    mask: np.ndarray,
    out_h: int,
    out_w: int,
    color_rgb: tuple[int, int, int] = SEGMENTATION_OVERLAY_RGB,
    alpha: float = 0.42,
) -> str:
    """Overlay binary mask on RGB image and return base64 PNG."""
    out = _mask_overlay_rgb(rgb, mask, color_rgb=color_rgb, alpha=alpha)
    if (out.shape[0], out.shape[1]) != (out_h, out_w):
        out = cv2.resize(out, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    return _rgb_to_b64(out)


def _segmentation_eval_rgb(mask_np: np.ndarray, gt_mask: np.ndarray) -> np.ndarray:
    """Render a TP/FP/FN segmentation visualization at GT resolution."""
    pred = (np.asarray(mask_np) > 0)
    gt = (np.asarray(gt_mask) > 0)
    if pred.shape != gt.shape:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (gt.shape[1], gt.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    vis = np.full((gt.shape[0], gt.shape[1], 3), 200, dtype=np.uint8)
    vis[pred & gt] = SEGMENTATION_TP_RGB
    vis[pred & ~gt] = SEGMENTATION_FP_RGB
    vis[~pred & gt] = SEGMENTATION_FN_RGB
    return vis


def _save_seg_map_image_web(
    mask_np: np.ndarray,
    gt_mask: np.ndarray,
    save_path: Path,
    tp_fn_size_scale: int = 3,  # kept for API compatibility
) -> None:
    """Save TP/FP/FN visualization image."""
    _ = tp_fn_size_scale
    vis = _segmentation_eval_rgb(mask_np, gt_mask)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(save_path), vis)
