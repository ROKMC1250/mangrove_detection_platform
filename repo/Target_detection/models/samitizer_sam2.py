"""
SAMitizer(SAM2): same pipeline as SAMitizer but uses SAM2ImagePredictor.

Only setup() and _get_sam_output() are overridden; the full detection pipeline
(background estimation, spectral scoring, Bayesian fusion, thresholding) is
inherited unchanged from SAMitizer (samitizer.py).
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from models.samitizer import SAMitizer


class SAMitizerSAM2(SAMitizer):
    """SAMitizer backed by SAM2 instead of SAM1.

    Inherits the full pipeline from SAMitizer; only overrides
    setup() to load a SAM2ImagePredictor and _get_sam_output() to call it.
    """

    def __init__(self, scoring: str = "osp_amf", alpha: float = 0.5, tau: float = 1.0):
        super().__init__(scoring=scoring, alpha=alpha, tau=tau)

    def setup(self):
        if self._predictor is not None:
            return

        from models.sam2 import SAM2, SAM2_REPO
        if str(SAM2_REPO) not in sys.path:
            sys.path.insert(0, str(SAM2_REPO))

        sam2 = SAM2()
        sam2.setup()
        self._predictor = sam2._predictor   # SAM2ImagePredictor
        print("[SAMitizer(SAM2)] Ready.")

    def _get_sam_output(
        self,
        rgb_image: np.ndarray,
        pos_points: List[Tuple[int, int]],
        neg_points: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """SAM2 version: use SAM2ImagePredictor API with inference_mode."""
        import torch as _torch

        coords = np.array(
            [(x, y) for x, y in pos_points] + [(x, y) for x, y in neg_points],
            dtype=np.float32,
        )
        labels = np.array(
            [1] * len(pos_points) + [0] * len(neg_points),
            dtype=np.int32,
        )
        with _torch.inference_mode():
            self._predictor.set_image(rgb_image)
            full_res_logits, _scores, _low_res_logits = self._predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=False,
                return_logits=True,
            )

        sam_logit = full_res_logits[0].astype(np.float32)
        sam_fg = sam_logit > float(self._predictor.mask_threshold)
        return sam_fg, sam_logit
