"""
SAM2 (Segment Anything Model 2) wrapper.

SAM2 from Meta's facebookresearch/sam2 extends SAM to video, but also
works on single images via SAM2ImagePredictor.

Requires: pip install sam-2
  or clone: git clone https://github.com/facebookresearch/sam2.git repos/sam2
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.config import CHECKPOINTS_DIR, REPOS_DIR, CFG

SAM2_REPO = REPOS_DIR / "sam2"
SAM2_CHECKPOINTS = {
    "sam2.1_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
    "sam2.1_hiera_base_plus": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "sam2.1_hiera_small": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "sam2.1_hiera_tiny": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml",
    },
}

DEFAULT_SAM2 = "sam2.1_hiera_large"


class SAM2:
    """SAM2 model for point-prompted segmentation."""

    name = "SAM2"

    def __init__(
        self,
        model_name: str = DEFAULT_SAM2,
        device: str = CFG.device,
    ):
        self.model_name = model_name
        self.device = device
        self._predictor = None

    @staticmethod
    def clone_repo():
        """Clone SAM2 repo if not present."""
        if SAM2_REPO.exists():
            print("[SAM2] Repository already cloned.")
            return
        REPOS_DIR.mkdir(parents=True, exist_ok=True)
        import subprocess
        print("[SAM2] Cloning facebookresearch/sam2 ...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/facebookresearch/sam2.git",
             str(SAM2_REPO)],
            check=True,
        )
        print("[SAM2] Clone complete.")

    def _download_checkpoint(self) -> str:
        """Download SAM2 checkpoint if not present."""
        info = SAM2_CHECKPOINTS[self.model_name]
        ckpt_path = CHECKPOINTS_DIR / f"{self.model_name}.pt"
        if ckpt_path.exists():
            return str(ckpt_path)

        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[SAM2] Downloading {self.model_name} checkpoint ...")
        from urllib.request import urlretrieve
        urlretrieve(info["url"], str(ckpt_path))
        print(f"[SAM2] Saved to {ckpt_path}")
        return str(ckpt_path)

    def setup(self):
        """Load SAM2 model."""
        if self._predictor is not None:
            return

        self.clone_repo()

        if str(SAM2_REPO) not in sys.path:
            sys.path.insert(0, str(SAM2_REPO))

        ckpt_path = self._download_checkpoint()
        config = SAM2_CHECKPOINTS[self.model_name]["config"]

        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                config_file=config,
                ckpt_path=ckpt_path,
                device=self.device,
            )
            self._predictor = SAM2ImagePredictor(model)
            print(f"[SAM2] Loaded {self.model_name}")
        except ImportError:
            print("[SAM2] Failed to import sam2. Install: pip install sam-2")
            print("  or: cd repos/sam2 && pip install -e .")
            raise

    def predict(
        self,
        image_rgb: np.ndarray,
        pos_points: List[Tuple[int, int]],
        neg_points: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run SAM2 prediction with point prompts.

        Parameters
        ----------
        image_rgb  : (H, W, 3) uint8 RGB
        pos_points : list of (x, y) positive prompt coordinates
        neg_points : list of (x, y) negative prompt coordinates

        Returns
        -------
        dict with:
            "mask"      : (H, W) uint8 binary mask (0/255)
            "score_map" : (H, W) float32 logits
            "score"     : float confidence
        """
        self.setup()

        if neg_points is None:
            neg_points = []

        self._predictor.set_image(image_rgb)

        coords = np.array(
            [(x, y) for x, y in pos_points]
            + [(x, y) for x, y in neg_points],
            dtype=np.float32,
        )
        labels = np.array(
            [1] * len(pos_points) + [0] * len(neg_points),
            dtype=np.int32,
        )

        full_res_logits, scores, _low_res_logits = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=False,
            return_logits=True,
        )

        score_map = full_res_logits[0].astype(np.float32)
        mask = (score_map > float(self._predictor.mask_threshold)).astype(np.uint8) * 255

        return {
            "mask": mask,
            "score_map": score_map,
            "score": float(scores[0]),
        }
