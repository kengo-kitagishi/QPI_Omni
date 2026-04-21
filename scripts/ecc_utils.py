"""Shared ECC pre-processing helpers.

Provides the canonical implementations of tilt correction, uint8
normalisation, ECC alignment, and MAD outlier removal used by every
QPI analysis script.  All scripts must import from here so that the
preprocessing pipeline is provably identical.

Functions
---------
tilt_fit_crop        -- background-tilt removal + center crop
apply_2pi_tilt_crop  -- 2-pi unwrap + tilt removal (grid-subtracted delta)
to_uint8             -- fixed [vmin, vmax] -> uint8 for ECC input
ecc_align            -- cv2.findTransformECC wrapper (TRANSLATION)
mad                  -- median absolute deviation
remove_outliers_mad  -- boolean outlier mask based on MAD
extract_rect_roi     -- re-exported from channel_crop
"""
from __future__ import annotations

import cv2
import numpy as np

from channel_crop import extract_rect_roi  # noqa: F401  (re-export)

# ====================================================================
# Tilt correction
# ====================================================================

def tilt_fit_crop(img_f64, cy, cx, crop_w, ecc_crop_h, tilt_crop_h,
                  fit_right: bool = False):
    """Return ECC-ready crop after background tilt removal, or None if OOB.

    Extract a ``tilt_crop_h``-wide ROI centred at (cy, cx), fit a linear
    slope+intercept on the background side (left 1/3 when ``fit_right=False``,
    right 1/3 otherwise), subtract it, and return the central ``ecc_crop_h``
    columns (shape ``(crop_w, ecc_crop_h)``).

    Returns ``None`` when the final ECC crop would contain zero-padded
    pixels (X: cx ± ecc_crop_h/2, Y: cy ± crop_w/2).  The wider tilt
    crop (tilt_crop_h) may include zero-padding at its edges -- that is
    acceptable because the tilt fit uses only the background 1/3 and the
    ECC sees only the centre.  Callers must skip ``None`` channels from
    any ECC aggregation rather than fall back to a different crop shape.
    """
    h, w = img_f64.shape
    if (cx - ecc_crop_h // 2) < 0 or (cx + ecc_crop_h // 2) > w:
        return None
    if (cy - crop_w // 2) < 0 or (cy + crop_w // 2) > h:
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
    """Apply 2-pi offset + linear tilt removal + center crop to a wide image.

    ``img_large`` is expected to have shape ``(crop_w, tilt_crop_h)`` (already
    extracted by the caller; typically a grid-subtracted delta).  The returned
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


# ====================================================================
# uint8 normalisation for ECC input
# ====================================================================

def to_uint8(img, vmin, vmax):
    """Linearly map *img* from [vmin, vmax] to uint8 [0, 255].

    No default arguments -- callers must pass vmin/vmax explicitly so that
    the normalisation range is always visible at the call site.
    """
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


# ====================================================================
# ECC alignment
# ====================================================================

def ecc_align(ref_u8, tl_u8, max_iter=50000, epsilon=1e-8):
    """ECC translation alignment between two uint8 images.

    Returns ``(tx, ty, correlation)`` on success, or ``None`` if ECC fails
    to converge.
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                max_iter, epsilon)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria,
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except cv2.error:
        return None


# ====================================================================
# MAD outlier removal
# ====================================================================

def mad(arr):
    """Median absolute deviation of *arr*."""
    m = np.median(arr)
    return float(np.median(np.abs(arr - m)))


def remove_outliers_mad(values, thresh):
    """Return a boolean mask (True = outlier) using MAD-based criterion."""
    arr = np.array(values, dtype=np.float64)
    md = mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - np.median(arr)) > thresh * md
