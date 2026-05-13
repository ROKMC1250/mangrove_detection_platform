"""
SAM3 (Segment Anything Model 3) service module.

Two operating modes share the same loaded model:
  * Point/box mode  - SAM2-style single-instance segmentation via
                      ``model.predict_inst``. Backwards compatible with the
                      old SAM2 controller flow (positive/negative clicks).
  * Text  mode (PCS) - Promptable Concept Segmentation via
                      ``processor.set_text_prompt``. A single noun phrase
                      returns every matching instance's mask, score, and
                      bounding box.

Both modes share the same image embedding cache (``cache_key`` -> state).
"""

import contextlib
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from ..core.config import PROJECT_ROOT
# Share the global GPU lock so SAM3 serialises with the rest of the GPU work.
# RLock allows nested acquisition within a single thread (e.g. encode -> predict).
from .gpu_compute import _GPU_LOCK


def _amp_ctx():
    """Autocast context matching the SAM3 reference notebook.

    SAM3's checkpoints are stored in bfloat16 (or expect bf16 activations on
    CUDA), so every encode / predict call must run under autocast otherwise
    the bias/weight buffers and the float32 input collide with
    `mat1 and mat2 must have the same dtype, but got BFloat16 and Float`.
    On CPU we skip autocast — sam3 inference there isn't really viable but at
    least won't crash.
    """
    if torch is not None and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    build_sam3_image_model = None
    Sam3Processor = None


# Global model state
_SAM3_MODEL = None
_SAM3_PROCESSOR = None
_SAM3_DEVICE = None
_SAM3_READY = False
_SAM3_ERROR = None
_SAM3_MODEL_NAME = None
_SAM3_INIT_LOCK = threading.Lock()

# Per-image inference states. Unlike SAM2 (where the predictor held the
# embedding internally), SAM3's processor returns a per-image ``state`` object.
# Keeping the most recently used state under its cache_key lets us reuse the
# embedding for both point-mode and text-mode without re-encoding.
_ENCODED_STATES: Dict[str, Any] = {}
_CURRENT_ENCODED_KEY: Optional[str] = None


def init_sam3() -> bool:
    """Initialize the SAM3 image model.

    Configuration via environment variables:
    - ``SAM3_CHECKPOINT_DIR`` - directory containing SAM3 checkpoints.
                                Default: ``PROJECT_ROOT/repo/sam3/``
    - ``SAM3_BPE_PATH``       - path to ``bpe_simple_vocab_16e6.txt.gz``.
                                Default: ``<sam3 package>/assets/...``

    Returns True on success.
    """
    global _SAM3_MODEL, _SAM3_PROCESSOR, _SAM3_DEVICE, _SAM3_READY
    global _SAM3_ERROR, _SAM3_MODEL_NAME

    if _SAM3_READY:
        return True

    with _SAM3_INIT_LOCK:
        if _SAM3_READY:
            return True

        if not TORCH_AVAILABLE:
            _SAM3_ERROR = "PyTorch not available"
            print(f"⚠️  SAM3 - {_SAM3_ERROR}")
            return False

        if not SAM3_AVAILABLE:
            _SAM3_ERROR = (
                "sam3 package not installed. Install via: "
                "pip install git+https://github.com/facebookresearch/sam3.git"
            )
            print(f"⚠️  SAM3 - {_SAM3_ERROR}")
            return False

        checkpoint_dir = os.environ.get(
            "SAM3_CHECKPOINT_DIR",
            os.path.join(PROJECT_ROOT, "repo", "sam3"),
        )

        bpe_path = os.environ.get("SAM3_BPE_PATH")
        if not bpe_path:
            try:
                import sam3 as _sam3_pkg
                pkg_root = os.path.dirname(_sam3_pkg.__file__)
                # The package keeps assets next to the source tree.
                candidate = os.path.join(
                    pkg_root, "..", "assets", "bpe_simple_vocab_16e6.txt.gz"
                )
                if os.path.exists(candidate):
                    bpe_path = os.path.abspath(candidate)
            except Exception:
                bpe_path = None

        try:
            if torch.cuda.is_available():
                _SAM3_DEVICE = torch.device("cuda")
            else:
                _SAM3_DEVICE = torch.device("cpu")

            print(f"🔄 SAM3 - Loading model on {_SAM3_DEVICE}...")

            # Enable TF32 / bf16 friendly settings on Ampere+ GPUs.
            if _SAM3_DEVICE.type == "cuda":
                try:
                    if torch.cuda.get_device_properties(0).major >= 8:
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                except Exception:
                    pass

            os.environ.setdefault("SAM3_CHECKPOINT_DIR", checkpoint_dir)

            kwargs = {"enable_inst_interactivity": True}
            if bpe_path:
                kwargs["bpe_path"] = bpe_path

            _SAM3_MODEL = build_sam3_image_model(**kwargs)
            try:
                _SAM3_MODEL.to(_SAM3_DEVICE)
            except Exception:
                # Some builders wire device internally; non-fatal.
                pass
            _SAM3_PROCESSOR = Sam3Processor(_SAM3_MODEL)
            _SAM3_MODEL_NAME = "sam3"
            _SAM3_READY = True

            print(f"✅ SAM3 - Model loaded successfully on {_SAM3_DEVICE}")
            return True

        except Exception as e:
            _SAM3_ERROR = f"Failed to load SAM3 model: {e}"
            print(f"❌ SAM3 - {_SAM3_ERROR}")
            return False


def is_sam3_ready() -> bool:
    return _SAM3_READY


def get_sam3_status() -> dict:
    return {
        "ready": _SAM3_READY,
        "model": _SAM3_MODEL_NAME,
        "device": str(_SAM3_DEVICE) if _SAM3_DEVICE else None,
        "error": _SAM3_ERROR,
    }


def _to_pil(rgb: np.ndarray) -> Image.Image:
    if isinstance(rgb, Image.Image):
        return rgb
    arr = np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB" if arr.ndim == 3 else "L")


def _to_numpy(x):
    """Convert any tensor / sequence-of-tensors / ndarray to a numpy array.

    Handles SAM3 outputs that may be bfloat16 CUDA tensors which numpy
    cannot consume directly (`numpy doesn't support BFloat16`).
    """
    if x is None:
        return None
    if torch is not None and isinstance(x, torch.Tensor):
        if x.dtype in (torch.bfloat16, torch.float16):
            x = x.float()
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)) and len(x) > 0 and torch is not None and isinstance(x[0], torch.Tensor):
        return np.stack([_to_numpy(t) for t in x], axis=0)
    return np.asarray(x)


def encode_image(cache_key: str, rgb_image: np.ndarray) -> str:
    """Encode an image with SAM3's image encoder. Idempotent on the same key.

    Thread-safety: takes ``_GPU_LOCK`` for the full check + set_image.
    """
    global _CURRENT_ENCODED_KEY

    if not _SAM3_READY or _SAM3_PROCESSOR is None:
        raise RuntimeError("SAM3 model not loaded")

    with _GPU_LOCK:
        if _CURRENT_ENCODED_KEY == cache_key and cache_key in _ENCODED_STATES:
            return cache_key

        pil_image = _to_pil(rgb_image)
        print(f"🔄 SAM3 - Encoding image ({pil_image.size})...")
        with _amp_ctx(), torch.inference_mode():
            state = _SAM3_PROCESSOR.set_image(pil_image)
        _ENCODED_STATES[cache_key] = state
        _CURRENT_ENCODED_KEY = cache_key
        print(f"✅ SAM3 - Image encoded (key: {cache_key[:24]}...)")

    return cache_key


def predict_mask(
    cache_key: str,
    positive_points: List[Tuple[int, int]],
    negative_points: Optional[List[Tuple[int, int]]] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, float]:
    """Single-instance prediction from point and/or box prompts (SAM2-style).

    ``box`` is in ``(x0, y0, x1, y1)`` pixel coordinates if provided.

    Returns ``(best_mask: bool ndarray, best_score: float)``.
    """
    if not _SAM3_READY or _SAM3_MODEL is None:
        raise RuntimeError("SAM3 model not loaded")

    with _GPU_LOCK:
        state = _ENCODED_STATES.get(cache_key)
        if state is None or _CURRENT_ENCODED_KEY != cache_key:
            raise RuntimeError(
                f"Image not encoded. Current: {_CURRENT_ENCODED_KEY}, "
                f"requested: {cache_key}"
            )

        all_points: List[Tuple[int, int]] = list(positive_points or [])
        all_labels: List[int] = [1] * len(all_points)
        if negative_points:
            all_points.extend(negative_points)
            all_labels.extend([0] * len(negative_points))

        point_coords = (
            np.asarray(all_points, dtype=np.float32) if all_points else None
        )
        point_labels = (
            np.asarray(all_labels, dtype=np.int32) if all_labels else None
        )
        box_arr = np.asarray(box, dtype=np.float32) if box is not None else None

        print(
            f"🔄 SAM3 - predict_inst with "
            f"{len(positive_points or [])}+ / {len(negative_points or [])}- points, "
            f"box={'yes' if box_arr is not None else 'no'}"
        )

        with _amp_ctx(), torch.inference_mode():
            masks, scores, _logits = _SAM3_MODEL.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box_arr,
                multimask_output=True,
            )

        masks = _to_numpy(masks)
        scores = _to_numpy(scores).reshape(-1)
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        best_idx = int(np.argmax(scores))
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        print(
            f"✅ SAM3 - mask predicted (score: {best_score:.3f}, "
            f"pixels: {int(best_mask.sum())})"
        )
        return best_mask, best_score


def predict_text(
    cache_key: str,
    prompt: str,
    score_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[float], List[Tuple[float, float, float, float]]]:
    """Promptable Concept Segmentation: return every instance matching ``prompt``.

    Output:
      masks  - list of bool ndarrays (H, W), one per instance
      scores - list of confidences in the same order
      boxes  - list of (x0, y0, x1, y1) pixel boxes in the same order
    """
    if not _SAM3_READY or _SAM3_PROCESSOR is None:
        raise RuntimeError("SAM3 model not loaded")
    if not prompt or not prompt.strip():
        raise ValueError("text prompt must be non-empty")

    with _GPU_LOCK:
        state = _ENCODED_STATES.get(cache_key)
        if state is None or _CURRENT_ENCODED_KEY != cache_key:
            raise RuntimeError(
                f"Image not encoded. Current: {_CURRENT_ENCODED_KEY}, "
                f"requested: {cache_key}"
            )

        print(f"🔄 SAM3 - set_text_prompt: '{prompt}'")
        with _amp_ctx(), torch.inference_mode():
            output = _SAM3_PROCESSOR.set_text_prompt(state=state, prompt=prompt)

        masks_raw = output.get("masks") if isinstance(output, dict) else getattr(output, "masks", None)
        scores_raw = output.get("scores") if isinstance(output, dict) else getattr(output, "scores", None)
        boxes_raw = output.get("boxes") if isinstance(output, dict) else getattr(output, "boxes", None)

        if masks_raw is None:
            raise RuntimeError("SAM3 text prediction returned no masks")

        masks_arr = _to_numpy(masks_raw)
        if masks_arr.ndim == 4:
            masks_arr = masks_arr.squeeze(1)
        if masks_arr.ndim == 2:
            masks_arr = masks_arr[None, ...]

        scores_arr = (
            _to_numpy(scores_raw).reshape(-1)
            if scores_raw is not None
            else np.ones(masks_arr.shape[0], dtype=np.float32)
        )
        boxes_arr = (
            _to_numpy(boxes_raw).reshape(-1, 4)
            if boxes_raw is not None
            else np.zeros((masks_arr.shape[0], 4), dtype=np.float32)
        )

        masks_out: List[np.ndarray] = []
        scores_out: List[float] = []
        boxes_out: List[Tuple[float, float, float, float]] = []
        for i in range(masks_arr.shape[0]):
            score = float(scores_arr[i]) if i < len(scores_arr) else 0.0
            if score < score_threshold:
                continue
            m = masks_arr[i].astype(bool)
            if not m.any():
                continue
            masks_out.append(m)
            scores_out.append(score)
            x0, y0, x1, y1 = (float(v) for v in boxes_arr[i])
            boxes_out.append((x0, y0, x1, y1))

        print(
            f"✅ SAM3 - text mode produced {len(masks_out)} instance(s) "
            f"(threshold={score_threshold})"
        )
        return masks_out, scores_out, boxes_out


def encode_and_predict(
    cache_key: str,
    rgb_image_or_getter,
    positive_points: List[Tuple[int, int]],
    negative_points: Optional[List[Tuple[int, int]]] = None,
    box: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[np.ndarray, float]:
    """Atomic encode-if-needed + point/box predict under one lock.

    ``rgb_image_or_getter`` may be an ndarray or a zero-arg callable (used to
    avoid an expensive raster read when the embedding is already current).
    """
    with _GPU_LOCK:
        if (
            _CURRENT_ENCODED_KEY != cache_key
            or cache_key not in _ENCODED_STATES
        ):
            rgb = (
                rgb_image_or_getter()
                if callable(rgb_image_or_getter)
                else rgb_image_or_getter
            )
            encode_image(cache_key, rgb)
        return predict_mask(cache_key, positive_points, negative_points, box=box)


def encode_and_predict_text(
    cache_key: str,
    rgb_image_or_getter,
    prompt: str,
    score_threshold: float = 0.5,
) -> Tuple[List[np.ndarray], List[float], List[Tuple[float, float, float, float]]]:
    """Atomic encode-if-needed + text predict under one lock."""
    with _GPU_LOCK:
        if (
            _CURRENT_ENCODED_KEY != cache_key
            or cache_key not in _ENCODED_STATES
        ):
            rgb = (
                rgb_image_or_getter()
                if callable(rgb_image_or_getter)
                else rgb_image_or_getter
            )
            encode_image(cache_key, rgb)
        return predict_text(cache_key, prompt, score_threshold=score_threshold)


def clear_encoding():
    """Drop all cached embeddings to free memory."""
    global _CURRENT_ENCODED_KEY

    with _GPU_LOCK:
        _ENCODED_STATES.clear()
        _CURRENT_ENCODED_KEY = None
        print("🗑 SAM3 - encodings cleared")
