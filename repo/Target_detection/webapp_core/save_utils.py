"""Save/image helpers used by web_app save_results route."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_score_map_image(score_map: np.ndarray, save_path: Path, title: str = "") -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    vmin = float(score_map.min())
    vmax = float(score_map.max())
    if vmax - vmin < 1e-10:
        vmax = vmin + 1.0
    im = ax.imshow(score_map, cmap="jet", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_gt_mask_image(gt_mask: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gt_mask, cmap="gray", vmin=0, vmax=max(int(gt_mask.max()), 1))
    ax.axis("off")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_path), dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def safe_filename(label: str) -> str:
    return (
        label
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("+", "plus")
        .replace("/", "_")
    )
