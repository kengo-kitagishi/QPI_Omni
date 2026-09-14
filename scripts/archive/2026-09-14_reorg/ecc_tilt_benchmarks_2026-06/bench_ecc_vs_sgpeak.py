"""bench_ecc_vs_sgpeak.py -- Subpixel shift precision: ECC vs SG-NCC

Ground-truth benchmark comparing our production ECC translation estimator
against the 2-D Savitzky-Golay subpixel peak finder
(Qiita: https://qiita.com/Cartelet/items/7ef90eebcb9e89dbfd63) paired with a
normalized cross-correlation (template matching) surface.

Test design (Test A, ground truth available)
--------------------------------------------
1. Take real tilt-corrected channel crops -- the exact float image the
   production ECC sees (ecc_utils.tilt_fit_crop), but wide (tilt_crop_h) so a
   subpixel shift leaves no boundary artifact in the central ECC window.
2. Apply a known subpixel shift (dy, dx) and center-crop both reference and
   moved image to the ECC window (crop_w x ecc_crop_h).
3. Estimate the shift with each method and compare to the ground truth:
   error = estimate - truth, per axis. We report bias (mean error =
   systematic offset) AND precision (std = scatter), since only a ground-truth
   test can separate the two.
4. The shift is generated with BOTH a Fourier shift and a 5th-order spline
   shift. Fourier interpolation slightly favors Fourier-domain methods, spline
   slightly favors gradient methods (ECC); a method that wins under both is
   robustly better.

Methods compared
----------------
  ECC-uint8 : production path -- to_uint8([-5,2]) + cv2.findTransformECC
  ECC-float : same ECC on float32 input (isolates the uint8 quantization cost)
  SG-NCC    : cv2.matchTemplate(TM_CCOEFF_NORMED) surface + SG subpixel peak

Sign handling
-------------
All estimators are wrapped by an end-to-end sign calibration (calibrate_sign)
that derives the per-axis sign from known spline shifts on a real crop, so the
image/array axis and ECC warp-sign conventions cannot silently flip the result
(a recurring failure mode -- see feedback_opencv_ecc_sign).

Usage
-----
    python scripts/bench_ecc_vs_sgpeak.py
    python scripts/bench_ecc_vs_sgpeak.py --n-frames 5 --n-shifts 40 --max-shift 2.0
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

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure
from ecc_utils import tilt_fit_crop, to_uint8, ecc_align


# --------------------------------------------------------------------------
# Defaults (260517 timelapse, current drift_config)
# --------------------------------------------------------------------------
DEFAULT_PHASE_DIR = r"E:\260517\2per_0055per_0per_2per\Pos1\z000\output_phase"
DEFAULT_ROIS_JSON = r"E:\260517\grid_2pergluc_2\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"

# Production ECC preprocessing constants (drift_config.json)
TILT_CROP_H = 270
ECC_CROP_H = 80
ECC_VMIN = -5.0
ECC_VMAX = 2.0
PIXEL_SCALE_UM = 0.34567514677103717

# SG-NCC parameters
NCC_MARGIN = 6      # search half-window for matchTemplate (px); shifts must be < this
SG_WIN = 5
SG_DEG = 6


# ==========================================================================
# 2-D Savitzky-Golay subpixel peak finder (Qiita: Cartelet)
# ==========================================================================

def sg_filter_2d(m, N):
    """Return SG derivative kernels for a (2N+1)x(2N+1) window, degree m.

    C[k] is the convolution kernel that, applied to the window, yields the
    k-th polynomial coefficient at the center. Ordering (after the internal
    sort) is: [value, Ix, Iy, Ixx, Iyy, Ixy, ...]. SG_peakSubPix uses C[1:6] =
    (Ix, Iy, Ixx, Iyy, Ixy). Faithful reproduction of the reference code.
    """
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
    """Subpixel peak (row, col) of a 2-D surface via SG derivatives + Newton.

    Faithful reproduction of SG_peakSubPix from the reference article. The
    kernel order returned by sg_filter_2d is (Ix, Iy, Ixx, Iyy, Ixy); the
    Hessian is [[Ixx, Ixy], [Ixy, Iyy]].
    """
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
    return peak  # (row, col) in surface coordinates


# ==========================================================================
# Shift generators (ground truth)
# ==========================================================================

def shift_fourier(arr, dy, dx):
    """Exact band-limited shift of content by (+dy, +dx). Real part returned."""
    f = np.fft.fftn(arr)
    return np.real(np.fft.ifftn(fourier_shift(f, (dy, dx))))


def shift_spline(arr, dy, dx):
    """5th-order spline shift of content by (+dy, +dx)."""
    return nd_shift(arr, (dy, dx), order=5, mode="reflect", prefilter=True)


SHIFT_GENERATORS = {"fourier": shift_fourier, "spline": shift_spline}


def center_crop_cols(wide, out_w):
    """Center-crop along axis 1 (the tilt/ECC X axis)."""
    start = (wide.shape[1] - out_w) // 2
    return wide[:, start:start + out_w]


# ==========================================================================
# Estimators: (ref_f, mov_f) float32 (crop_w, ECC_CROP_H) -> (est_y, est_x)
# Raw sign is fixed afterwards by calibrate_sign.
# ==========================================================================

def est_ecc_uint8(ref_f, mov_f):
    res = ecc_align(to_uint8(ref_f, ECC_VMIN, ECC_VMAX),
                    to_uint8(mov_f, ECC_VMIN, ECC_VMAX))
    if res is None:
        return None
    tx, ty, _ = res
    return np.array([ty, tx], dtype=np.float64)  # (row=Y, col=X)


def est_ecc_float(ref_f, mov_f):
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20000, 1e-8)
    try:
        _, warp = cv2.findTransformECC(ref_f.astype(np.float32), mov_f.astype(np.float32),
                                       warp, cv2.MOTION_TRANSLATION, crit)
    except cv2.error:
        return None
    return np.array([float(warp[1, 2]), float(warp[0, 2])], dtype=np.float64)


def est_sg_ncc(ref_f, mov_f):
    m = NCC_MARGIN
    template = mov_f[m:mov_f.shape[0] - m, m:mov_f.shape[1] - m].astype(np.float32)
    surf = cv2.matchTemplate(ref_f.astype(np.float32), template, cv2.TM_CCOEFF_NORMED)
    if surf.shape[0] < SG_WIN or surf.shape[1] < SG_WIN:
        return None
    py, px = sg_peak_subpix(surf.astype(np.float32))
    # Zero-shift peak sits at (m, m); offset from there is the raw shift.
    return np.array([py - m, px - m], dtype=np.float64)


ESTIMATORS = {
    "ECC-uint8": est_ecc_uint8,
    "ECC-float": est_ecc_float,
    "SG-NCC": est_sg_ncc,
}


# ==========================================================================
# Sign calibration (end-to-end, per axis)
# ==========================================================================

def calibrate_sign(estimator, wide_ref, name):
    """Derive per-axis sign so estimator output matches ground truth.

    Uses known spline shifts on a real crop. Asserts the sign-corrected
    estimate tracks the truth (|slope-1| small, low residual); raises on
    failure rather than silently returning a flipped result.
    """
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
        if not (0.5 < abs(ratio) < 1.6):
            raise RuntimeError(
                f"{name}: axis {ax} slope {ratio:.3f} off from +-1 -- estimator/sign broken")
        sign[ax] = np.sign(ratio)
    resid = (raw * sign) - truth
    med = np.median(np.abs(resid))
    if med > 0.4:
        raise RuntimeError(f"{name}: post-sign median residual {med:.3f}px too large")
    print(f"  sign-calibrated {name}: sign={sign.tolist()}  median|resid|={med:.3f}px")
    return sign


# ==========================================================================
# Main benchmark
# ==========================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase-dir", default=DEFAULT_PHASE_DIR)
    p.add_argument("--rois-json", default=DEFAULT_ROIS_JSON)
    p.add_argument("--n-frames", type=int, default=5, help="source frames sampled across the timelapse")
    p.add_argument("--n-shifts", type=int, default=40, help="random ground-truth shifts per crop")
    p.add_argument("--max-shift", type=float, default=2.0, help="uniform |shift| range (px)")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(0)

    rois = json.loads(Path(args.rois_json).read_text(encoding="utf-8"))
    phase_paths = sorted(Path(args.phase_dir).glob("img_*_phase.tif"))
    if not phase_paths:
        raise FileNotFoundError(f"no phase images in {args.phase_dir}")
    # Sample frames evenly across the timelapse for content diversity.
    idx = np.linspace(0, len(phase_paths) - 1, args.n_frames).astype(int)
    frame_paths = [phase_paths[i] for i in idx]
    print(f"Frames: {len(frame_paths)}  Channels: {len(rois)}  "
          f"Shifts/crop: {args.n_shifts}  Generators: {list(SHIFT_GENERATORS)}")

    # Build wide tilt-corrected crops (the exact production ECC float image, wide).
    wide_crops = []
    for path in frame_paths:
        img = tifffile.imread(str(path)).astype(np.float64)
        for roi in rois:
            wide = tilt_fit_crop(img, roi["cy"], roi["cx"], roi["crop_w"],
                                 ecc_crop_h=TILT_CROP_H, tilt_crop_h=TILT_CROP_H)
            if wide is not None:
                wide_crops.append(wide.astype(np.float32))
    if not wide_crops:
        raise RuntimeError("no valid crops (all OOB)")
    print(f"Valid wide crops: {len(wide_crops)}")

    # Calibrate signs on the first crop.
    signs = {name: calibrate_sign(fn, wide_crops[0], name) for name, fn in ESTIMATORS.items()}

    # errors[gen][method] -> list of (err_y, err_x)
    errors = {g: {m: [] for m in ESTIMATORS} for g in SHIFT_GENERATORS}
    fails = {m: 0 for m in ESTIMATORS}

    for wide in wide_crops:
        ref80 = center_crop_cols(wide, ECC_CROP_H)
        shifts = rng.uniform(-args.max_shift, args.max_shift, size=(args.n_shifts, 2))
        for dy, dx in shifts:
            for gname, gfn in SHIFT_GENERATORS.items():
                mov80 = center_crop_cols(gfn(wide, dy, dx), ECC_CROP_H)
                for mname, mfn in ESTIMATORS.items():
                    est = mfn(ref80, mov80)
                    if est is None:
                        fails[mname] += 1
                        continue
                    est = est * signs[mname]
                    errors[gname][mname].append([est[0] - dy, est[1] - dx])

    # ---- Statistics ----
    stats = {}
    print("\n=== Results (error = estimate - truth, px) ===")
    for gname in SHIFT_GENERATORS:
        print(f"\n[{gname} shift]")
        for mname in ESTIMATORS:
            e = np.array(errors[gname][mname])
            if len(e) == 0:
                continue
            bias = e.mean(axis=0)
            sd = e.std(axis=0)
            rmse = np.sqrt((e ** 2).mean(axis=0))
            stats[(gname, mname)] = dict(
                bias_y=bias[0], bias_x=bias[1], std_y=sd[0], std_x=sd[1],
                rmse_y=rmse[0], rmse_x=rmse[1], n=len(e))
            print(f"  {mname:10s} N={len(e):5d}  "
                  f"bias=({bias[1]:+.4f},{bias[0]:+.4f})  "
                  f"std=({sd[1]:.4f},{sd[0]:.4f})  "
                  f"RMSE=({rmse[1]:.4f},{rmse[0]:.4f}) px  "
                  f"[X,Y]")
    print(f"\nFailures: {fails}")

    # ---- Figure ----
    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                         "font.family": "DejaVu Sans"})
    colors = {"ECC-uint8": "#d62728", "ECC-float": "#ff7f0e", "SG-NCC": "#1f77b4"}
    gens = list(SHIFT_GENERATORS)
    fig, axes = plt.subplots(len(gens), 3, figsize=(13, 4.2 * len(gens)))
    if len(gens) == 1:
        axes = axes[np.newaxis, :]
    bins = np.linspace(-0.25, 0.25, 60)

    for r, gname in enumerate(gens):
        # Col 0: estimated vs true (X axis), scatter near y=x
        ax = axes[r, 0]
        for mname in ESTIMATORS:
            e = np.array(errors[gname][mname])
            if len(e) == 0:
                continue
            # truth recovered as estimate - error; plot estimate vs truth for X
            # (we only stored errors; reconstruct truth_x is not kept, so show
            #  error vs nothing -> instead show error density via the histograms)
        # Use col0 for a combined error histogram (X+Y) per method.
        for mname in ESTIMATORS:
            e = np.array(errors[gname][mname])
            if len(e) == 0:
                continue
            allerr = e.ravel()
            ax.hist(allerr, bins=bins, histtype="step", lw=1.6,
                    color=colors[mname], density=True,
                    label=f"{mname} (RMSE={np.sqrt((e**2).mean()):.3f})")
        ax.set_xlabel("Error = estimate - truth (px)")
        ax.set_ylabel("Density")
        ax.set_title(f"[{gname} shift] combined X+Y error")
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Col 1: X error histogram with bias/std annotation
        ax = axes[r, 1]
        for mname in ESTIMATORS:
            e = np.array(errors[gname][mname])
            if len(e) == 0:
                continue
            ax.hist(e[:, 1], bins=bins, histtype="step", lw=1.6,
                    color=colors[mname], density=True,
                    label=f"{mname}: bias={e[:,1].mean():+.3f} sd={e[:,1].std():.3f}")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("X error (px)")
        ax.set_ylabel("Density")
        ax.set_title(f"[{gname}] X-axis error")
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        # Col 2: Y error histogram with bias/std annotation
        ax = axes[r, 2]
        for mname in ESTIMATORS:
            e = np.array(errors[gname][mname])
            if len(e) == 0:
                continue
            ax.hist(e[:, 0], bins=bins, histtype="step", lw=1.6,
                    color=colors[mname], density=True,
                    label=f"{mname}: bias={e[:,0].mean():+.3f} sd={e[:,0].std():.3f}")
        ax.axvline(0, color="k", lw=0.6, ls=":")
        ax.set_xlabel("Y error (px)")
        ax.set_ylabel("Density")
        ax.set_title(f"[{gname}] Y-axis error")
        ax.legend(frameon=False, fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        f"Subpixel shift precision: ECC vs SG-NCC  "
        f"(pixel scale {PIXEL_SCALE_UM*1000:.1f} nm/px,  "
        f"{len(wide_crops)} crops x {args.n_shifts} shifts)",
        fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    flat_params = {f"{g}_{m}_{k}": float(v)
                   for (g, m), d in stats.items() for k, v in d.items()}
    save_figure(
        fig,
        params={**flat_params,
                "n_crops": len(wide_crops),
                "n_shifts_per_crop": args.n_shifts,
                "max_shift_px": args.max_shift,
                "pixel_scale_um": PIXEL_SCALE_UM},
        description=(
            "Ground-truth subpixel shift precision: production ECC (uint8/float) "
            "vs 2D Savitzky-Golay + normalized cross-correlation (Qiita method). "
            "Error = estimate - known shift, under Fourier and spline shift models."),
        data={f"err_{g}_{m}": np.array(errors[g][m])
              for g in SHIFT_GENERATORS for m in ESTIMATORS if errors[g][m]},
    )
    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
