"""
Change detection service.

Compares the **binary masks** produced by two prior analyses (target
detection, mangrove segmentation, SAM3, or spectral threshold) and
returns a signed pixel-wise change:

  gained  =  ~mask_A &  mask_B   (B-only — appeared)
  lost    =   mask_A & ~mask_B   (A-only — disappeared)
  change  =  gained | lost       (any change)

Inputs A and B must be on the same pixel grid (same shape and, when
present, the same CRS / affine transform). Analyses run on the same AOI
+ resolution already satisfy this.
"""

from typing import Dict, Tuple

import numpy as np


def _extract_binary_mask_and_meta(cache_entry: Dict) -> Tuple[np.ndarray, Dict]:
    """Return (binary_mask, meta) from any analysis cache entry.

    The mask field is `binary_mask` for TD/MS/spectral/SAM3 entries.
    Older SAM2-style entries (now SAM3) also expose `mask` as a fallback.
    Raises ValueError if the entry has no binary mask yet — the caller
    should turn this into an HTTP 400.
    """
    mask = cache_entry.get('binary_mask')
    if mask is None:
        mask = cache_entry.get('mask')  # SAM-style direct mask field
    if mask is None:
        raise ValueError(
            "This analysis has no binary mask yet — apply a threshold "
            "first (or run SAM3) before using it for change detection."
        )

    if 'prob_map' in cache_entry:
        source_type = 'segmentation'
    elif 'detection_map' in cache_entry:
        source_type = 'detection'
    elif 'index_data' in cache_entry or 'data' in cache_entry:
        source_type = 'spectral'
    else:
        # SAM3 entries don't carry any of the above float maps.
        source_type = 'sam3'

    meta = {
        'source_type': source_type,
        'transform': cache_entry.get('transform'),
        'crs': cache_entry.get('crs'),
        'bbox': cache_entry.get('bbox'),
        'geometry': cache_entry.get('geometry'),
    }
    return np.asarray(mask, dtype=bool), meta


def compute_signed_change(result_a: Dict, result_b: Dict) -> Dict:
    """Compute gained / lost / change masks between two binary results.

    Returns a dict with:
        gained_mask:  bool HxW  (~A & B)
        lost_mask:    bool HxW  ( A & ~B)
        change_mask:  bool HxW  (gained | lost)
        transform / crs / bbox / geometry: copied from input A
        source_types: (type_a, type_b)
        stats:  {gained_px, lost_px, unchanged_px, total_px,
                 gained_pct, lost_pct, unchanged_pct}

    Raises ValueError on shape / CRS / transform mismatch, or if either
    input is missing its binary mask.
    """
    mask_a, meta_a = _extract_binary_mask_and_meta(result_a)
    mask_b, meta_b = _extract_binary_mask_and_meta(result_b)

    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Shape mismatch: A={mask_a.shape} vs B={mask_b.shape}. "
            "Change detection requires two masks computed on the same AOI "
            "and resolution."
        )

    if meta_a['crs'] is not None and meta_b['crs'] is not None:
        if str(meta_a['crs']) != str(meta_b['crs']):
            raise ValueError(
                f"CRS mismatch: A={meta_a['crs']} vs B={meta_b['crs']}."
            )
    if meta_a['transform'] is not None and meta_b['transform'] is not None:
        ta = np.asarray(list(meta_a['transform'])[:6], dtype=np.float64)
        tb = np.asarray(list(meta_b['transform'])[:6], dtype=np.float64)
        if not np.allclose(ta, tb, rtol=1e-6, atol=1e-6):
            raise ValueError(
                "Affine transform mismatch between A and B. Inputs must be "
                "pixel-aligned on the same grid."
            )

    gained = (~mask_a) & mask_b
    lost = mask_a & (~mask_b)
    change = gained | lost

    total = int(mask_a.size)
    gained_px = int(gained.sum())
    lost_px = int(lost.sum())
    unchanged_px = max(0, total - gained_px - lost_px)

    def _pct(n):
        return round(100.0 * n / total, 3) if total > 0 else 0.0

    stats = {
        'gained_px': gained_px,
        'lost_px': lost_px,
        'unchanged_px': unchanged_px,
        'total_px': total,
        'gained_pct': _pct(gained_px),
        'lost_pct': _pct(lost_px),
        'unchanged_pct': _pct(unchanged_px),
    }

    return {
        'gained_mask': gained,
        'lost_mask': lost,
        'change_mask': change,
        'transform': meta_a['transform'],
        'crs': meta_a['crs'],
        'bbox': meta_a['bbox'] or meta_b['bbox'],
        'geometry': meta_a['geometry'] or meta_b['geometry'],
        'source_types': (meta_a['source_type'], meta_b['source_type']),
        'stats': stats,
    }
