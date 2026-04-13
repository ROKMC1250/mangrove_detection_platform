"""
SAM (Segment Anything Model) wrapper.

Wraps SAM prediction in a unified interface
for the benchmark framework. SAM only accepts (H, W, 3) uint8 RGB input,
so multispectral images must be converted via band_conversion first.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.config import CFG


class SAM:
    """SAM model for point-prompted segmentation."""

    name = "SAM"

    def __init__(
        self,
        model_type: str = CFG.sam_model_type,
        checkpoint: Optional[str] = None,
        device: str = CFG.device,
    ):
        self.model_type = model_type
        self.checkpoint = checkpoint or CFG.sam_checkpoint
        self.device = device
        self._predictor = None

    def setup(self):
        """Load SAM model. Call once before running predictions."""
        if self._predictor is not None:
            return

        from segment_anything import SamPredictor, sam_model_registry

        if not os.path.exists(self.checkpoint):
            print(f"[SAM] Checkpoint not found: {self.checkpoint}")
            print("[SAM] Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
            raise FileNotFoundError(self.checkpoint)

        print(f"[SAM] Loading {self.model_type} from {self.checkpoint}")
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint)
        sam.to(self.device)
        sam.eval()
        self._predictor = SamPredictor(sam)
        print("[SAM] Ready.")

    def predict(
        self,
        image_rgb: np.ndarray,
        pos_points: List[Tuple[int, int]],
        neg_points: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, np.ndarray]:
        """Run SAM prediction with point prompts.

        Parameters
        ----------
        image_rgb  : (H, W, 3) uint8 RGB
        pos_points : list of (x, y) positive prompt coordinates
        neg_points : list of (x, y) negative prompt coordinates (SAM supports these)

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

        masks, scores, low_res_logits = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=False,
        )

        mask = (masks[0].astype(np.uint8)) * 255

        import torch
        lrl = torch.from_numpy(low_res_logits).unsqueeze(0)
        full_logits = self._predictor.model.postprocess_masks(
            lrl,
            input_size=self._predictor.input_size,
            original_size=self._predictor.original_size,
        )
        score_map = full_logits[0, 0].numpy().astype(np.float32)

        return {
            "mask": mask,
            "score_map": score_map,
            "score": float(scores[0]),
        }
