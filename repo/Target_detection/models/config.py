"""Minimal runtime config for web_app model stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = ROOT / "checkpoints"
REPOS_DIR = ROOT / "repos"


@dataclass
class RuntimeConfig:
    # Detector params
    eta: float = 1e-6
    fpr: float = 0.01
    max_cov_samples: int = 0  # 0 or less -> use all pixels for covariance
    cov_dtype: str = "float64"
    eps: float = 1e-8

    # OSP params
    osp_lambda: float = 0.1
    osp_sv_thresh: float = 0.1
    target_exclude_thresh: float = 0.95
    cov_method: str = "sample"
    tyler_max_iter: int = 30
    tyler_tol: float = 1e-6
    neural_cov_steps: int = 100
    neural_cov_lr: float = 0.01
    neural_lowrank_rank: int = 5

    # Device / checkpoints
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    sam_checkpoint: str = str(CHECKPOINTS_DIR / "sam_vit_b_01ec64.pth")
    sam_model_type: str = "vit_b"


CFG = RuntimeConfig()
