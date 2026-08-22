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
# Canonical ECC defaults (single source of truth)
# ====================================================================

# ECC correlation threshold: channels whose findTransformECC score falls below
# this are excluded from the inter-channel average (0 disables the filter).
# With float ECC input, cell-bearing channels drop below 0.99, so the average
# collapses to the cell-free channels.  Every script imports this default;
# JSON configs (drift_config.json) carry their own copy that compute_drift_online
# reads at runtime, so update those separately.
ECC_MIN_CORR = 0.994

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


def to_ecc_input(img, vmin, vmax):
    """Clip *img* to [vmin, vmax] and return float32 (no 8-bit quantisation).

    Float ECC input. Identical to ``to_uint8`` except the final quantisation to
    256 levels is removed: cv2.findTransformECC accepts single-channel float32
    (CV_32F) and is affine-intensity invariant, so the [0,255] rescale is
    unnecessary. The uint8 quantisation introduced a systematic ~+0.087 px X
    bias (verified by ground-truth bench, scripts/bench_ecc_vs_sgpeak.py);
    feeding clipped float32 removes it (X RMSE ~0.143 px -> ~0.015 px). The clip
    is kept for the same outlier robustness as to_uint8.
    """
    return np.clip(img, vmin, vmax).astype(np.float32)


# ====================================================================
# ECC alignment
# ====================================================================

def ecc_align(ref_u8, tl_u8, max_iter=20000, epsilon=1e-8):
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

# ====================================================================
# Gaussian-2D alignment (NCC surface + rotated 2-D Gaussian subpixel peak)
# ====================================================================
# Measured on 260819 against ECC-float (scripts/bench_subpix_methods.py,
# bench_cellbias_estimators.py):
#   ground-truth precision   4.2 nm vs 5.7 nm   (same-image shifts)
#   cell-content bias       178 nm vs 248 nm   (cell-bearing channel vs
#                                               cell-free grid reference)
# The peak finder is a verbatim copy of the one benchmarked there; the sign
# convention is adapted here so the return value matches ecc_align exactly
# (tx = column shift, ty = row shift). test_gaussian2d_align.py asserts both
# the agreement with the bench implementation and the sign against ecc_align.

# Search half-window (px). This is also the capture range, and it trades against
# precision: the template is (crop_w - 2m) x (ecc_crop_h - 2m), and on 260819
# channel crops the RMSE goes 4.0 nm (m=4) / 4.3 nm (m=8) / 14.5 nm (m=10) /
# 24.3 nm (m=14) -- past m=8 the template is too thin and the estimator becomes
# worse than ECC (5.2 nm). Shifts beyond the range are reported as None rather
# than clipped, so a startup frame drops the channel instead of biasing it.
NCC_MARGIN = 8
NCC_MIN_PEAK = 0.9925  # cell/cell-free threshold for this score (430-channel fit)


def _gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    return g.ravel()


def peak_gaussian_2d(response_map, window_size=3):
    """Subpixel peak (row, col) of *response_map* by a rotated 2-D Gaussian fit."""
    from scipy.optimize import curve_fit
    iy, ix = np.unravel_index(int(np.argmax(response_map)), response_map.shape)
    h, w = response_map.shape
    hw = window_size // 2
    if iy < hw or iy >= h - hw or ix < hw or ix >= w - hw:
        return float(iy), float(ix)
    y0, y1 = iy - hw, iy + hw + 1
    x0, x1 = ix - hw, ix + hw + 1
    patch = response_map[y0:y1, x0:x1]
    yy, xx = np.indices(patch.shape)
    try:
        p0 = (patch[hw, hw], hw, hw, 1.0, 1.0, 0, float(np.min(patch)))
        popt, _ = curve_fit(_gaussian_2d, (xx.ravel(), yy.ravel()), patch.ravel(),
                            p0=p0, maxfev=500)
        fy, fx = y0 + popt[2], x0 + popt[1]
        if abs(fy - iy) > 1.5 or abs(fx - ix) > 1.5:
            return float(iy), float(ix)
        return fy, fx
    except Exception:
        return float(iy), float(ix)


def gaussian2d_align(ref_f, tl_f, margin=NCC_MARGIN):
    """Translation between *ref_f* and *tl_f* via the NCC surface peak.

    Drop-in replacement for :func:`ecc_align`: returns ``(tx, ty, score)`` in the
    SAME sign convention (tx = column shift, ty = row shift), or ``None`` when
    the crop is too small for the margin. ``score`` is the NCC peak value, which
    plays the role ecc_align's correlation plays for channel selection -- note
    it needs its own threshold (NCC_MIN_PEAK), not ECC_MIN_CORR.
    """
    h, w = tl_f.shape
    if h <= 2 * margin or w <= 2 * margin:
        return None
    template = tl_f[margin:h - margin, margin:w - margin].astype(np.float32)
    surf = cv2.matchTemplate(ref_f.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    if surf.shape[0] < 3 or surf.shape[1] < 3:
        return None
    iy, ix = np.unravel_index(int(np.argmax(surf)), surf.shape)
    if iy in (0, surf.shape[0] - 1) or ix in (0, surf.shape[1] - 1):
        return None          # true shift is outside +-margin; do not clip it
    py, px = peak_gaussian_2d(surf.astype(np.float32))
    # The raw NCC peak offset runs opposite to the ECC warp; negate so callers
    # can swap estimators without touching any downstream sign handling.
    return -(px - margin), -(py - margin), float(surf.max())


ALIGNERS = {"ecc_float": ecc_align, "gaussian2d": gaussian2d_align}
MIN_SCORE = {"ecc_float": ECC_MIN_CORR, "gaussian2d": NCC_MIN_PEAK}


def get_aligner(name):
    """Return ``(align_fn, default_min_score)`` for an estimator name."""
    if name not in ALIGNERS:
        raise ValueError(f"unknown estimator {name!r}; have {sorted(ALIGNERS)}")
    return ALIGNERS[name], MIN_SCORE[name]
