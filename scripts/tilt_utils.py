"""Shared ECC pre-processing helpers (tilt fit / 2pi / center crop).

Used by compute_pos_shifts.py, correct_0pergluc.py, grid_subtract.py, and
compute_drift_online.py so all four scripts apply identical crop logic.
"""
from __future__ import annotations

import numpy as np

from channel_crop import extract_rect_roi


def tilt_fit_crop(img_f64, cy, cx, crop_w, ecc_crop_h, tilt_crop_h,
                  fit_right: bool = False):
    """Return ECC-ready crop after background tilt removal, or None if OOB.

    Extract a ``tilt_crop_h``-wide ROI centred at (cy, cx), fit a linear
    slope+intercept on the background side (left 1/3 when ``fit_right=False``,
    right 1/3 otherwise), subtract it, and return the central ``ecc_crop_h``
    columns (shape ``(crop_w, ecc_crop_h)``).

    Returns ``None`` when the wide ROI would require zero-padding. Callers
    must skip such channels in any ECC aggregation rather than fall back to
    a different crop shape.
    """
    h, w = img_f64.shape
    if (cx - tilt_crop_h // 2) < 0 or (cx + tilt_crop_h // 2) > w:
        return None
    big = extract_rect_roi(img_f64, cy, cx, crop_w, tilt_crop_h).astype(np.float64)
    x = np.arange(tilt_crop_h, dtype=np.float64)
    prof = big.mean(axis=0)
    fit_n = max(1, tilt_crop_h // 3)
    if fit_right:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    corrected = big - (a * x + b)[np.newaxis, :]
    start = (tilt_crop_h - ecc_crop_h) // 2
    return corrected[:, start : start + ecc_crop_h]


def apply_2pi_tilt_crop(img_large, out_crop_h, tilt_crop_h,
                        fit_right: bool = False):
    """Apply 2pi offset + linear tilt removal + center crop to a wide image.

    ``img_large`` is expected to have shape ``(crop_w, tilt_crop_h)`` (already
    extracted by the caller; typically a grid-subtracted delta). The returned
    array has shape ``(crop_w, out_crop_h)``.

    Pipeline:
      1. Compute mean over the background 1/3 (left when ``fit_right=False``,
         right otherwise); subtract ``round(mean / 2pi) * 2pi`` globally.
      2. Linear slope+intercept fit on the same 1/3; subtract the trend.
      3. Center-crop ``out_crop_h`` columns.
    """
    fit_n = max(1, tilt_crop_h // 3)
    if fit_right:
        bg_mean = float(np.mean(img_large[:, -fit_n:]))
    else:
        bg_mean = float(np.mean(img_large[:, :fit_n]))
    k = int(round(bg_mean / (2.0 * np.pi)))
    if k != 0:
        img_large = img_large - k * 2.0 * np.pi

    x = np.arange(tilt_crop_h, dtype=np.float64)
    prof = img_large.mean(axis=0)
    if fit_right:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    img_large = img_large - (a * x + b)[np.newaxis, :]

    start = (tilt_crop_h - out_crop_h) // 2
    return img_large[:, start : start + out_crop_h]
