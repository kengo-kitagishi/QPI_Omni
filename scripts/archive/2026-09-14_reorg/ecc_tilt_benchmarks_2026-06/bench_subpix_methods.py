"""bench_subpix_methods.py -- Subpixel shift precision, many methods (Qiita-style)

Reproduces the comparison figure of the Qiita article
(https://qiita.com/Cartelet/items/7ef90eebcb9e89dbfd63): a 2-D error scatter
plus an |error| histogram, comparing many subpixel-shift estimators on a
ground-truth artificial-shift benchmark built from real 260517 channel crops.

Methods compared
----------------
Direct image registration:
  ECC-uint8 : production path -- to_uint8([-5,2]) + cv2.findTransformECC
  ECC-float : same ECC on float32 input

Subpixel peak finders on the NCC surface (cv2.matchTemplate, TM_CCOEFF_NORMED):
  SG-2D       : 2-D Savitzky-Golay derivative peak (the article's proposal)
  Centroid    : intensity-weighted centroid of the 3x3 around the peak
  Parabola    : 1-D parabolic fit per axis
  Biquadratic : 2-D quadratic (6-term) fit, Newton step to the vertex
  Gaussian-1D : 1-D log-parabola (Gaussian) fit per axis
  Gaussian-2D : rotated 2-D Gaussian fit (scipy curve_fit)
  LogGaussian : 2-D quadratic fit on log(surface)
  Spline      : RectBivariateSpline + L-BFGS-B maximization

All peak finders are faithful reproductions of the reference article code.

Test design (ground truth available)
-------------------------------------
1. Take real tilt-corrected channel crops (the exact float image production ECC
   sees), but wide so a subpixel shift leaves no boundary artifact.
2. Apply a known subpixel shift (dy, dx); center-crop ref and moved to the ECC
   window (crop_w x ECC_CROP_H).
3. Estimate the shift with each method; error = estimate - truth, per axis.
4. Shift generated with BOTH a Fourier shift and a 5th-order spline shift.

Cell channels (0,3,5,9,10) are EXCLUDED by default -- they carry glucose-
dependent cell content that biases the estimators (see project_cell_ch_ecc_bias).

Sign handling
-------------
Every estimator is wrapped by calibrate_sign (per-axis end-to-end sign from
known spline shifts) -- raises rather than silently returning a flipped result.

Usage
-----
    python scripts/bench_subpix_methods.py
    python scripts/bench_subpix_methods.py --n-frames 5 --n-shifts 40 --max-shift 2.0
    python scripts/bench_subpix_methods.py --exclude-ch ""     # use all channels
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import cv2
from scipy.ndimage import shift as nd_shift
from scipy.ndimage import fourier_shift
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import RectBivariateSpline

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure
from ecc_utils import tilt_fit_crop, to_uint8, ecc_align


# --------------------------------------------------------------------------
# Defaults (260517 timelapse, current drift_config)
# --------------------------------------------------------------------------
DEFAULT_PHASE_DIR = r"E:\260517\2per_0055per_0per_2per\Pos1\z000\output_phase"
DEFAULT_ROIS_JSON = r"E:\260517\grid_2pergluc_2\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
OUT_DIR = r"E:\260517\ecc_sg_ab"

# Cell-bearing channels (0-based roi index) -- excluded from the bench content.
DEFAULT_EXCLUDE_CH = "0,3,5,9,10"

# Production ECC preprocessing constants (drift_config.json)
TILT_CROP_H = 270
ECC_CROP_H = 80
ECC_VMIN = -5.0
ECC_VMAX = 2.0
PIXEL_SCALE_UM = 0.34567514677103717

# NCC parameters
NCC_MARGIN = 6      # search half-window for matchTemplate (px); shifts must be < this
SG_WIN = 5
SG_DEG = 6


# ==========================================================================
# 2-D Savitzky-Golay subpixel peak finder (Qiita: Cartelet)
# ==========================================================================

def sg_filter_2d(m, N):
    """SG derivative kernels for a (2N+1)x(2N+1) window, degree m."""
    mask = (np.c_[:m + 1] + np.r_[:m + 1]).flatten() <= m
    X = np.zeros(((2 * N + 1) ** 2, mask.sum()))
    for i, j in np.indices((2 * N + 1, 2 * N + 1)).T.reshape(-1, 2):
        X[i * (2 * N + 1) + j] = ((i - N) ** np.c_[:m + 1] * (j - N) ** np.r_[:m + 1]).flatten()[mask]
    coords = np.mgrid[:m + 1, :m + 1].transpose(1, 2, 0).reshape(-1, 2)[mask]
    order = np.array(sorted(range(coords.shape[0]),
                            key=lambda i: (coords[i, 0] + coords[i, 1], coords[i].min())))
    C = np.linalg.pinv(X[:, order]).reshape(-1, 2 * N + 1, 2 * N + 1)
    return C


def sg_peak_subpix(image, win_size=SG_WIN, degrees=SG_DEG, max_iter=15,
                   interpolation=cv2.INTER_CUBIC, eps=1e-7, lr=0.7):
    """Subpixel peak (row, col) of a 2-D surface via SG derivatives + Newton."""
    SGs = sg_filter_2d(degrees, win_size // 2)[1:6]
    peak = np.asarray(np.unravel_index(image.argmax(), image.shape), dtype=np.float64)
    mg = np.mgrid[-(win_size // 2): win_size // 2: 1j * win_size,
                  -(win_size // 2): win_size // 2: 1j * win_size]
    for _ in range(max_iter):
        AOI = cv2.remap(image, *(mg + peak[:, None, None]).astype(np.float32)[::-1],
                        interpolation=interpolation)
        Ix, Iy, Ixx, Iyy, Ixy = np.einsum('ijk, jk -> i', SGs, AOI)
        detH = (Ixx + 1e-15) * (Iyy + 1e-15) - Ixy ** 2
        dx = -(Iyy * Ix - Ixy * Iy) / detH
        dy = -(Ixx * Iy - Ixy * Ix) / detH
        peak = peak + np.clip(np.array([dy, dx]), -1, 1) * lr
        if np.hypot(dx, dy) < eps:
            break
    return peak  # (row, col)


# ==========================================================================
# Other subpixel peak finders on a response surface (Qiita: Cartelet)
# Each returns (row, col) in surface coordinates.
# ==========================================================================

def get_peak_integer(response_map):
    iy, ix = np.unravel_index(int(np.argmax(response_map)), response_map.shape)
    h, w = response_map.shape
    is_edge = (iy == 0 or iy == h - 1 or ix == 0 or ix == w - 1)
    return (int(iy), int(ix)), is_edge


def find_peak_centroid(response_map):
    (iy, ix), is_edge = get_peak_integer(response_map)
    if is_edge:
        return float(iy), float(ix)
    patch = np.maximum(response_map[iy - 1:iy + 2, ix - 1:ix + 2], 0)
    total = patch.sum()
    if total == 0:
        return float(iy), float(ix)
    yy, xx = np.indices((3, 3)) - 1
    dy = (yy * patch).sum() / total
    dx = (xx * patch).sum() / total
    return iy + dy, ix + dx


def find_peak_parabolic(response_map):
    (iy, ix), is_edge = get_peak_integer(response_map)
    if is_edge:
        return float(iy), float(ix)
    z0 = response_map[iy, ix]
    ym1, yp1 = response_map[iy - 1, ix], response_map[iy + 1, ix]
    dy_den = 2 * z0 - ym1 - yp1
    dy = (ym1 - yp1) / (2 * dy_den) if dy_den > 0 else 0.0
    xm1, xp1 = response_map[iy, ix - 1], response_map[iy, ix + 1]
    dx_den = 2 * z0 - xm1 - xp1
    dx = (xm1 - xp1) / (2 * dx_den) if dx_den > 0 else 0.0
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return float(iy), float(ix)
    return iy + dy, ix + dx


def find_peak_biquadratic(response_map):
    (iy, ix), is_edge = get_peak_integer(response_map)
    if is_edge:
        return float(iy), float(ix)
    z = response_map[iy - 1:iy + 2, ix - 1:ix + 2].flatten()
    x = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=float)
    y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=float)
    A = np.vstack([x ** 2, y ** 2, x * y, x, y, np.ones(9)]).T
    try:
        a, b, c, d, e, _ = np.linalg.lstsq(A, z, rcond=None)[0]
    except Exception:
        return float(iy), float(ix)
    det = 4 * a * b - c * c
    if det <= 0:
        return float(iy), float(ix)
    try:
        delta = np.linalg.solve(np.array([[2 * a, c], [c, 2 * b]]),
                                np.array([-d, -e]))
        dx, dy = delta[0], delta[1]
    except Exception:
        return float(iy), float(ix)
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return float(iy), float(ix)
    return iy + dy, ix + dx


def find_peak_gaussian_1d(response_map):
    (iy, ix), is_edge = get_peak_integer(response_map)
    if is_edge:
        return float(iy), float(ix)

    def parabolic_1d(v_m1, v_0, v_p1):
        den = 2 * v_0 - v_m1 - v_p1
        return (v_m1 - v_p1) / (2 * den) if den > 0 else 0.0

    ym1, y0, yp1 = response_map[iy - 1, ix], response_map[iy, ix], response_map[iy + 1, ix]
    if ym1 > 1e-6 and y0 > 1e-6 and yp1 > 1e-6:
        lv = np.log([ym1, y0, yp1])
        den = 2 * (2 * lv[1] - lv[2] - lv[0])
        dy = (lv[2] - lv[0]) / den if den > 0 else parabolic_1d(ym1, y0, yp1)
    else:
        dy = parabolic_1d(ym1, y0, yp1)
    xm1, x0, xp1 = response_map[iy, ix - 1], response_map[iy, ix], response_map[iy, ix + 1]
    if xm1 > 1e-6 and x0 > 1e-6 and xp1 > 1e-6:
        lv = np.log([xm1, x0, xp1])
        den = 2 * (2 * lv[1] - lv[2] - lv[0])
        dx = (lv[2] - lv[0]) / den if den > 0 else parabolic_1d(xm1, x0, xp1)
    else:
        dx = parabolic_1d(xm1, x0, xp1)
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return float(iy), float(ix)
    return iy + dy, ix + dx


def _gaussian_2d(xy, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    x, y = xy
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(
        -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
    return g.ravel()


def find_peak_gaussian_2d(response_map, window_size=3):
    (iy, ix), _ = get_peak_integer(response_map)
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


def find_peak_log_gaussian(response_map):
    (iy, ix), is_edge = get_peak_integer(response_map)
    if is_edge:
        return float(iy), float(ix)
    patch = response_map[iy - 1:iy + 2, ix - 1:ix + 2]
    if np.any(patch <= 1e-6):
        return find_peak_parabolic(response_map)
    lp = np.log(patch.flatten())
    x = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=float)
    y = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=float)
    A = np.vstack([x ** 2, y ** 2, x, y, np.ones(9)]).T
    try:
        a, b, c, d, _ = np.linalg.lstsq(A, lp, rcond=None)[0]
    except Exception:
        return float(iy), float(ix)
    if a >= 0 or b >= 0:
        return float(iy), float(ix)
    dx, dy = -c / (2 * a), -d / (2 * b)
    if abs(dy) > 1.0 or abs(dx) > 1.0:
        return float(iy), float(ix)
    return iy + dy, ix + dx


def find_peak_spline(response_map):
    (iy, ix), is_edge = get_peak_integer(response_map)
    if is_edge:
        return float(iy), float(ix)
    yr = np.arange(iy - 1, iy + 2)
    xr = np.arange(ix - 1, ix + 2)
    patch = response_map[iy - 1:iy + 2, ix - 1:ix + 2]
    try:
        f = RectBivariateSpline(yr, xr, patch, kx=2, ky=2)
        res = minimize(lambda p: -float(f(p[0], p[1], grid=False)),
                       x0=[iy, ix], method="L-BFGS-B",
                       bounds=((iy - 0.5, iy + 0.5), (ix - 0.5, ix + 0.5)))
        if res.success:
            ys, xs = res.x
            if (iy - 1.0 < ys < iy + 1.0) and (ix - 1.0 < xs < ix + 1.0):
                return ys, xs
        return float(iy), float(ix)
    except Exception:
        return float(iy), float(ix)


# RectBivariateSpline needs kx<=2 on a 3x3 grid (kx<m). The article used kx=3 on
# a larger patch; on a 3x3 neighborhood degree 2 is the faithful maximum.

NCC_PEAK_FINDERS = {
    "SG-2D": lambda s: sg_peak_subpix(s.astype(np.float32)),
    "Centroid": find_peak_centroid,
    "Parabola": find_peak_parabolic,
    "Biquadratic": find_peak_biquadratic,
    "Gaussian-1D": find_peak_gaussian_1d,
    "Gaussian-2D": find_peak_gaussian_2d,
    "LogGaussian": find_peak_log_gaussian,
    "Spline": find_peak_spline,
}


# ==========================================================================
# Shift generators (ground truth)
# ==========================================================================

def shift_fourier(arr, dy, dx):
    f = np.fft.fftn(arr)
    return np.real(np.fft.ifftn(fourier_shift(f, (dy, dx))))


def shift_spline(arr, dy, dx):
    return nd_shift(arr, (dy, dx), order=5, mode="reflect", prefilter=True)


SHIFT_GENERATORS = {"fourier": shift_fourier, "spline": shift_spline}


def center_crop_cols(wide, out_w):
    start = (wide.shape[1] - out_w) // 2
    return wide[:, start:start + out_w]


# ==========================================================================
# Direct ECC estimators (no NCC surface)
# ==========================================================================

def est_ecc_uint8(ref_f, mov_f):
    res = ecc_align(to_uint8(ref_f, ECC_VMIN, ECC_VMAX),
                    to_uint8(mov_f, ECC_VMIN, ECC_VMAX))
    if res is None:
        return None
    tx, ty, _ = res
    return np.array([ty, tx], dtype=np.float64)


def est_ecc_float(ref_f, mov_f):
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20000, 1e-8)
    try:
        _, warp = cv2.findTransformECC(ref_f.astype(np.float32), mov_f.astype(np.float32),
                                       warp, cv2.MOTION_TRANSLATION, crit)
    except cv2.error:
        return None
    return np.array([float(warp[1, 2]), float(warp[0, 2])], dtype=np.float64)


def make_ncc_estimator(peak_finder):
    """NCC (TM_CCOEFF_NORMED) surface + a subpixel peak finder -> (row, col)."""
    def est(ref_f, mov_f):
        m = NCC_MARGIN
        template = mov_f[m:mov_f.shape[0] - m, m:mov_f.shape[1] - m].astype(np.float32)
        surf = cv2.matchTemplate(ref_f.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
        if surf.shape[0] < SG_WIN or surf.shape[1] < SG_WIN:
            return None
        py, px = peak_finder(surf.astype(np.float32))
        return np.array([py - m, px - m], dtype=np.float64)
    return est


def build_estimators():
    est = {"ECC-uint8": est_ecc_uint8, "ECC-float": est_ecc_float}
    for name, finder in NCC_PEAK_FINDERS.items():
        est[name] = make_ncc_estimator(finder)
    return est


# ==========================================================================
# Sign calibration (end-to-end, per axis)
# ==========================================================================

def calibrate_sign(estimator, wide_ref, name):
    gts = [(0.0, 1.5), (0.0, -1.5), (1.5, 0.0), (-1.5, 0.0), (1.0, 1.0), (-1.0, -1.0)]
    raw, truth = [], []
    for dy, dx in gts:
        mov_wide = shift_spline(wide_ref, dy, dx)
        ref80 = center_crop_cols(wide_ref, ECC_CROP_H)
        mov80 = center_crop_cols(mov_wide, ECC_CROP_H)
        est = estimator(ref80, mov80)
        if est is None:
            continue
        raw.append(est)
        truth.append([dy, dx])
    raw = np.array(raw)
    truth = np.array(truth)
    if len(raw) < 4:
        raise RuntimeError(f"{name}: sign calibration failed (too few successes)")
    sign = np.ones(2)
    for ax in range(2):
        sel = np.abs(truth[:, ax]) > 0.5
        if sel.sum() == 0:
            continue
        ratio = np.mean(raw[sel, ax] / truth[sel, ax])
        if not (0.3 < abs(ratio) < 2.0):
            raise RuntimeError(
                f"{name}: axis {ax} slope {ratio:.3f} off from +-1 -- estimator/sign broken")
        sign[ax] = np.sign(ratio)
    resid = (raw * sign) - truth
    med = np.median(np.abs(resid))
    print(f"  sign-calibrated {name:12s}: sign={sign.tolist()}  median|resid|={med:.3f}px")
    return sign


# ==========================================================================
# Main benchmark
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-dir", default=DEFAULT_PHASE_DIR)
    p.add_argument("--rois-json", default=DEFAULT_ROIS_JSON)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--exclude-ch", default=DEFAULT_EXCLUDE_CH,
                   help="comma-separated 0-based channel indices to exclude (cell channels)")
    p.add_argument("--n-frames", type=int, default=5)
    p.add_argument("--n-shifts", type=int, default=40)
    p.add_argument("--max-shift", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(0)
    exclude = {int(s) for s in args.exclude_ch.split(",") if s.strip() != ""}

    rois = json.loads(Path(args.rois_json).read_text(encoding="utf-8"))
    phase_paths = sorted(Path(args.phase_dir).glob("img_*_phase.tif"))
    if not phase_paths:
        raise FileNotFoundError(f"no phase images in {args.phase_dir}")
    idx = np.linspace(0, len(phase_paths) - 1, args.n_frames).astype(int)
    frame_paths = [phase_paths[i] for i in idx]
    kept_ch = [i for i in range(len(rois)) if i not in exclude]
    print(f"Frames: {len(frame_paths)}  Channels total: {len(rois)}  "
          f"Excluded(cell): {sorted(exclude)}  Kept: {kept_ch}")
    print(f"Shifts/crop: {args.n_shifts}  Generators: {list(SHIFT_GENERATORS)}")

    # Build wide tilt-corrected crops from non-cell channels only.
    wide_crops = []
    for path in frame_paths:
        img = tifffile.imread(str(path)).astype(np.float64)
        for ci, roi in enumerate(rois):
            if ci in exclude:
                continue
            wide = tilt_fit_crop(img, roi["cy"], roi["cx"], roi["crop_w"],
                                 ecc_crop_h=TILT_CROP_H, tilt_crop_h=TILT_CROP_H)
            if wide is not None:
                wide_crops.append(wide.astype(np.float32))
    if not wide_crops:
        raise RuntimeError("no valid crops (all OOB or all excluded)")
    print(f"Valid wide crops: {len(wide_crops)}")

    estimators = build_estimators()
    print("\nSign calibration:")
    signs = {name: calibrate_sign(fn, wide_crops[0], name) for name, fn in estimators.items()}

    # errors[method] -> list of (err_y, err_x), pooled over both generators
    errors = {m: [] for m in estimators}
    fails = {m: 0 for m in estimators}

    for wide in wide_crops:
        ref80 = center_crop_cols(wide, ECC_CROP_H)
        shifts = rng.uniform(-args.max_shift, args.max_shift, size=(args.n_shifts, 2))
        for dy, dx in shifts:
            for gfn in SHIFT_GENERATORS.values():
                mov80 = center_crop_cols(gfn(wide, dy, dx), ECC_CROP_H)
                for mname, mfn in estimators.items():
                    est = mfn(ref80, mov80)
                    if est is None:
                        fails[mname] += 1
                        continue
                    est = est * signs[mname]
                    errors[mname].append([est[0] - dy, est[1] - dx])

    # ---- Statistics ----
    print("\n=== Results (error = estimate - truth, px) ===")
    stats = {}
    rows = []
    for mname in estimators:
        e = np.array(errors[mname])
        if len(e) == 0:
            continue
        mag = np.hypot(e[:, 0], e[:, 1])
        bias = e.mean(axis=0)
        sd = e.std(axis=0)
        rmse = np.sqrt((e ** 2).mean(axis=0))
        stats[mname] = dict(bias_x=bias[1], bias_y=bias[0], std_x=sd[1], std_y=sd[0],
                            rmse_x=rmse[1], rmse_y=rmse[0],
                            mean_abs=float(mag.mean()), std_abs=float(mag.std()),
                            n=len(e))
        rows.append((mname, mag.mean(), mag.std(), rmse[1], rmse[0], bias[1], bias[0]))
    rows.sort(key=lambda r: r[1])
    print(f"{'method':12s} {'mean|err|':>9s} {'std|err|':>9s} "
          f"{'RMSE_X':>8s} {'RMSE_Y':>8s} {'bias_X':>8s} {'bias_Y':>8s}  (px)")
    for name, ma, sa, rx, ry, bx, by in rows:
        print(f"{name:12s} {ma:9.4f} {sa:9.4f} {rx:8.4f} {ry:8.4f} {bx:+8.4f} {by:+8.4f}")
    print(f"\nFailures: {fails}")

    # ---- Figure (article layout: one column per method;
    #      top row = 2-D error scatter, bottom row = |error| histogram) ----
    order = [r[0] for r in rows]  # best (lowest mean|err|) first
    n = len(order)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.8, "font.family": "DejaVu Sans"})
    nm_px = PIXEL_SCALE_UM * 1000.0
    # Common axis ranges so panels are directly comparable (article style).
    sc_lim = 0.3      # scatter half-range (px)
    h_max = 0.3       # histogram |error| max (px)
    bins = np.linspace(0, h_max, 50)

    fig, axes = plt.subplots(2, n, figsize=(1.9 * n, 4.6),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})

    for j, name in enumerate(order):
        e = np.array(errors[name])
        st = stats[name]

        # Top: 2-D error scatter (X error vs Y error).
        ax = axes[0, j]
        es = e
        if len(es) > 800:
            es = e[rng.choice(len(e), 800, replace=False)]
        ax.scatter(es[:, 1], es[:, 0], s=4, alpha=0.30, color="#1f77b4", edgecolors="none")
        ax.axhline(0, color="k", lw=0.4, ls=":")
        ax.axvline(0, color="k", lw=0.4, ls=":")
        ax.set_xlim(-sc_lim, sc_lim)
        ax.set_ylim(-sc_lim, sc_lim)
        ax.set_aspect("equal")
        ax.set_title(f"{name}\nmean={st['mean_abs']:.3f} sd={st['std_abs']:.3f}px", fontsize=8)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel("Y error (px)")
        ax.set_xlabel("X error (px)", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Bottom: |error| histogram.
        ax = axes[1, j]
        mag = np.hypot(e[:, 0], e[:, 1])
        frac_in = float((mag <= h_max).mean())
        ax.hist(np.clip(mag, 0, h_max), bins=bins, color="#1f77b4", alpha=0.8)
        ax.axvline(st["mean_abs"], color="#d62728", lw=1.0)
        ax.set_xlim(0, h_max)
        ax.set_title(f"{frac_in*100:.0f}% within range", fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.set_ylabel("Count")
        ax.set_xlabel("|error| (px)", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Subpixel shift precision -- {n} methods, sorted best -> worst   "
        f"(260517 Pos1, non-cell ch; {len(wide_crops)} crops x {args.n_shifts} shifts x 2 gen; "
        f"1 px = {nm_px:.1f} nm)",
        fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # Save to the requested directory.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "subpix_method_comparison.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"\nSaved figure: {out_png}")

    # Provenance logging.
    flat_params = {f"{m}_{k}": float(v) for m, d in stats.items() for k, v in d.items()}
    save_figure(
        fig,
        params={**flat_params, "n_crops": len(wide_crops),
                "n_shifts_per_crop": args.n_shifts, "max_shift_px": args.max_shift,
                "pixel_scale_um": PIXEL_SCALE_UM,
                "excluded_channels": sorted(exclude)},
        description=(
            "Ground-truth subpixel shift precision across many estimators "
            "(ECC-uint8/float + NCC surface peak finders: SG-2D, Centroid, "
            "Parabola, Biquadratic, Gaussian-1D/2D, LogGaussian, Spline), "
            "Qiita-style 2D error scatter + |error| histogram. Non-cell channels."),
        data={f"err_{m}": np.array(errors[m]) for m in estimators if errors[m]},
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
