# %%
"""
test_iarpls_tilt_subtract.py
----------------------------
TEST ONLY -- compare the current linear tilt correction in grid_subtract
against a 1D iARPLS (smooth baseline) variant, on Pos75 ch02 of 260517.

Background
----------
The production tilt step (ecc_utils.apply_2pi_tilt_crop) builds a 1D profile
of the grid-subtracted delta (mean over the cell axis, axis=0), fits a *linear*
slope+intercept on the background 1/3, and subtracts that straight line from
every row.  If the delta background has a slight curvature (roundness) -- e.g.
from a small focus / position mismatch between timelapse and grid -- a straight
line cannot follow it and leaves a residual after subtraction.

This script swaps ONLY that 1D baseline: linear (a*x+b) -> iARPLS smooth
baseline.  Same dimensionality (varies along axis=1, broadcast over rows),
no new warp / no re-reconstruction.  Everything else (residual subpixel warp,
grid selection, 2pi integer offset, crops) is the *unmodified* production code
imported from grid_subtract / ecc_utils.

It writes two parallel output trees OUTSIDE the crop_sub tree:
    OUT/linear/chNN/   (faithful reproduction of current pipeline)
    OUT/iarpls/chNN/   (iARPLS baseline variant)
plus a comparison figure via figure_logger.save_figure.

Pos75 parameters are taken from the existing logs:
    grid_dir   = E:/260517/grid_2pergluc_2
    grid_z     = 8
    pos_split  = 53  -> Pos75 >= 53 -> fit_right=True
    calibration= grid_calibration_Pos75.json
    tilt_crop_h= 270 (== out crop_h, so start=0, full 270 kept)
"""
import sys
import json
import glob
from pathlib import Path

import numpy as np
import tifffile
from pybaselines.whittaker import iarpls

sys.path.insert(0, str(Path(__file__).parent))
import grid_subtract as gs
from ecc_utils import apply_2pi_tilt_crop as apply_2pi_tilt_crop_linear  # keep original ref

# ============================================================
# Pos75 configuration (from existing logs / pos_shifts)
# ============================================================
TL_PRERECON_DIR = r"E:\260517\2per_0055per_0per_2per\Pos75\z000\output_phase_raw"
GRID_DIR        = r"E:\260517\grid_2pergluc_2"
BASE_LABEL      = "Pos75"
SHIFTS_JSON     = r"E:\260517\2per_0055per_0per_2per_crop_sub\Pos75\output_phase\channels\pos_shifts_cal_online.json"
ROIS_JSON       = r"E:\260517\2per_0055per_0per_2per_crop_sub\Pos75\output_phase\channels\channel_rois.json"
CAL_JSON        = r"E:\260517\grid_2pergluc_2\grid_calibration_Pos75.json"

GRID_Z_INDEX = 8
TILT_CROP_H  = 270
POS_NUM      = 75
POS_SPLIT    = 53
FIT_RIGHT    = POS_NUM >= POS_SPLIT  # True for Pos75

# Output OUTSIDE the crop_sub tree (user request)
OUT_ROOT = Path(r"E:\260517\_iarpls_tilt_test\Pos75")

# iARPLS smoothness penalty (larger -> straighter baseline).
IARPLS_LAM = 1e4

# Cell mask threshold [rad] for the both-ends background fit (relative to the
# background-side median).  Pixels beyond this are treated as cell and excluded.
CELL_THRESH = 0.5
# Polynomial degree for the both-ends cell-masked channel-axis background fit.
BGMASK_DEG = 2

# Representative frame for the diagnostic figure, and frame sampling for the
# two output trees (spread across the 0% glucose window 2019..2884).
REP_FRAME    = 2500
N_PICK       = 30
PICK_LO      = 2019
PICK_HI      = 2884

# pixel scale (same constants as grid_subtract)
PIXEL_SCALE_UM = (gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION
                  * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6)


# ============================================================
# iARPLS variant of the tilt step (same signature as production)
# ============================================================
def apply_2pi_iarpls_crop(img_large, out_crop_h, tilt_crop_h, fit_right=False):
    """Drop-in replacement for apply_2pi_tilt_crop using a 1D iARPLS baseline.

    Steps 1 (2pi integer offset) and 3 (center crop) are identical to the
    linear version; only step 2 changes: instead of a straight-line fit on the
    background 1/3, an asymmetric smooth baseline (iARPLS) is fit on the FULL
    profile -- the asymmetric weighting automatically ignores the (positive)
    cell bump while following gentle background curvature on both ends.
    """
    fit_n = max(1, tilt_crop_h // 3)
    if fit_right:
        bg_mean = float(np.mean(img_large[:, -fit_n:]))
    else:
        bg_mean = float(np.mean(img_large[:, :fit_n]))
    k = int(round(bg_mean / (2.0 * np.pi)))
    if k != 0:
        img_large = img_large - k * 2.0 * np.pi

    prof = img_large.mean(axis=0)
    baseline, _ = iarpls(prof, lam=IARPLS_LAM)
    img_large = img_large - baseline[np.newaxis, :]

    start = (tilt_crop_h - out_crop_h) // 2
    return img_large[:, start:start + out_crop_h]


def apply_2pi_bgmask_crop(img_large, out_crop_h, tilt_crop_h, fit_right=False):
    """Robust drop-in: cell-masked, BOTH-ends channel-axis background fit.

    The production tilt fits a straight line on ONE background 1/3 and
    extrapolates to the far (cell) side, which leaves a ~0.07 rad level offset
    on the far background when the true background is slightly non-linear.

    This variant keeps steps 1 (2pi offset) and 3 (center crop) identical, but
    for step 2 it (a) masks the cell (|value - bg_median| > CELL_THRESH),
    (b) builds a cell-free column profile (nan-mean over rows), and
    (c) fits a degree-``BGMASK_DEG`` polynomial over the FULL channel using
    every cell-free column -- so both the near and far background ends pin the
    baseline.  Still a pure function of column broadcast over rows (no new warp).
    """
    fit_n = max(1, tilt_crop_h // 3)
    bg_side = img_large[:, -fit_n:] if fit_right else img_large[:, :fit_n]
    bg_mean = float(np.mean(bg_side))
    k = int(round(bg_mean / (2.0 * np.pi)))
    if k != 0:
        img_large = img_large - k * 2.0 * np.pi
        bg_side = img_large[:, -fit_n:] if fit_right else img_large[:, :fit_n]

    med = float(np.median(bg_side))
    cell = np.abs(img_large - med) > CELL_THRESH
    bg = np.where(cell, np.nan, img_large)
    bgcol = np.nanmean(bg, axis=0)
    x = np.arange(tilt_crop_h, dtype=np.float64)
    valid = ~np.isnan(bgcol)
    if int(valid.sum()) < (BGMASK_DEG + 5):
        raise RuntimeError(f"too few background columns ({int(valid.sum())}) for deg-{BGMASK_DEG} fit")
    coef = np.polyfit(x[valid], bgcol[valid], BGMASK_DEG)
    base = np.polyval(coef, x)
    img_large = img_large - base[np.newaxis, :]

    start = (tilt_crop_h - out_crop_h) // 2
    return img_large[:, start:start + out_crop_h]


# ============================================================
# Helpers
# ============================================================
def load_inputs():
    rois = json.loads(Path(ROIS_JSON).read_text(encoding="utf-8"))
    shifts = json.loads(Path(SHIFTS_JSON).read_text(encoding="utf-8"))
    frame_results = shifts["frame_results"]
    grid_cal = gs.load_grid_calibration(CAL_JSON)
    pos_map = gs.scan_grid_positions(GRID_DIR, BASE_LABEL)
    tl_frames = sorted(glob.glob(str(Path(TL_PRERECON_DIR) / "img_*_phase.tif")))
    return rois, frame_results, grid_cal, pos_map, tl_frames


_grid_cache = {}
def get_grid_image(pos_map, xi, yi):
    key = (xi, yi)
    if key not in _grid_cache:
        pos_dir = pos_map[key]
        p = pos_dir / "output_phase_raw" / f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
        _grid_cache[key] = tifffile.imread(str(p)).astype(np.float64)
    return _grid_cache[key]


def select_for_frame(fr, grid_cal, pos_map):
    sx = fr["shift_x_avg"]
    sy = fr["shift_y_avg"]
    out = gs.select_grid(sx, sy, pos_map, grid_cal, PIXEL_SCALE_UM,
                         x_step=0.1, y_step=0.1,
                         shift_sign_x=1, shift_sign_y=1)
    xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, res_x, res_y = out
    return sx, sy, xi, yi, cal_dx, cal_dy, res_x, res_y


def run_tree(tilt_func, rois, frame_results, grid_cal, pos_map, tl_frames,
             pick, out_dir):
    """Reproduce grid_subtract per-frame output using the given tilt function."""
    gs.apply_2pi_tilt_crop = tilt_func  # monkeypatch the name used inside _raw_subtract_correct
    n_ch = len(rois)
    for ch in range(n_ch):
        (out_dir / f"ch{ch:02d}").mkdir(parents=True, exist_ok=True)
    for t in pick:
        fr = frame_results[t]
        sx, sy, xi, yi, cal_dx, cal_dy, res_x, res_y = select_for_frame(fr, grid_cal, pos_map)
        tl_img = tifffile.imread(tl_frames[t]).astype(np.float64)
        grid_img = get_grid_image(pos_map, xi, yi)
        per_ch, _ = gs.process_single_frame(
            tl_img, sx, sy, rois,
            cal_dx, cal_dy, res_x, res_y, grid_img,
            output_crop_h_override=None,
            tilt_crop_h_raw=TILT_CROP_H,
            use_raw_phase=True,
            apply_subpixel_correction=True,
            fit_right=FIT_RIGHT,
            apply_inverse_shift=False,
        )
        name = Path(tl_frames[t]).name.replace("_phase.tif", ".tif")
        for ch in range(n_ch):
            tifffile.imwrite(str(out_dir / f"ch{ch:02d}" / name),
                             per_ch[ch].astype(np.float32))
    print(f"  wrote {len(pick)} frames x {n_ch} ch -> {out_dir}")


def build_sub_large(tl_img, grid_img, roi, cal_dx, cal_dy, res_x, res_y):
    """Replicate grid_subtract's pre-tilt delta (sub_large) for diagnostics."""
    tl_warped = gs.apply_inverse_shift_warp(tl_img, res_x, res_y)
    crop_cx = int(round(roi["cx"] + cal_dx))
    crop_cy = int(round(roi["cy"] + cal_dy))
    tl_large = gs.extract_rect_roi(tl_warped, crop_cy, crop_cx, roi["crop_w"], TILT_CROP_H)
    grid_large = gs.extract_rect_roi(grid_img, crop_cy, crop_cx, roi["crop_w"], TILT_CROP_H)
    return (tl_large - grid_large).astype(np.float64)


# ============================================================
# Main
# ============================================================
def main():
    rois, frame_results, grid_cal, pos_map, tl_frames = load_inputs()
    print(f"rois={len(rois)}  frames={len(frame_results)}  grid_cells={len(pos_map)}  tl_prerecon={len(tl_frames)}")
    print(f"fit_right={FIT_RIGHT}  iarpls_lam={IARPLS_LAM:g}")

    pick = sorted(set(np.linspace(PICK_LO, PICK_HI, N_PICK).astype(int).tolist()))

    # ---- validate: linear reproduction must match existing crop_sub output ----
    ch = 2
    fr = frame_results[REP_FRAME]
    sx, sy, xi, yi, cal_dx, cal_dy, res_x, res_y = select_for_frame(fr, grid_cal, pos_map)
    tl_img = tifffile.imread(tl_frames[REP_FRAME]).astype(np.float64)
    grid_img = get_grid_image(pos_map, xi, yi)

    gs.apply_2pi_tilt_crop = apply_2pi_tilt_crop_linear
    per_ch_lin, _ = gs.process_single_frame(
        tl_img, sx, sy, rois, cal_dx, cal_dy, res_x, res_y, grid_img,
        output_crop_h_override=None, tilt_crop_h_raw=TILT_CROP_H,
        use_raw_phase=True, apply_subpixel_correction=True,
        fit_right=FIT_RIGHT, apply_inverse_shift=False)
    lin_crop = per_ch_lin[ch]

    existing_path = Path(r"E:\260517\2per_0055per_0per_2per_crop_sub\Pos75\output_phase\channels"
                         r"\crop_sub_rawraw\z000\ch02") / f"img_{REP_FRAME:09d}_ph_000.tif"
    if existing_path.exists():
        existing = tifffile.imread(str(existing_path)).astype(np.float64)
        diff = float(np.max(np.abs(existing - lin_crop)))
        print(f"[validate] frame {REP_FRAME} ch02 linear-vs-existing max|diff|={diff:.4e}")
    else:
        print(f"[validate] existing frame not found: {existing_path}")

    # ---- diagnostic profiles on the representative frame, ch02 ----
    sub_large = build_sub_large(tl_img, grid_img, rois[ch], cal_dx, cal_dy, res_x, res_y)
    # apply 2pi step the same way both methods do, to get the profile they fit
    fit_n = max(1, TILT_CROP_H // 3)
    bg_mean = float(np.mean(sub_large[:, -fit_n:])) if FIT_RIGHT else float(np.mean(sub_large[:, :fit_n]))
    k = int(round(bg_mean / (2 * np.pi)))
    sub_2pi = sub_large - k * 2 * np.pi if k != 0 else sub_large
    prof = sub_2pi.mean(axis=0)
    x = np.arange(TILT_CROP_H)
    if FIT_RIGHT:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    lin_base = a * x + b
    # both-ends cell-masked background baseline (the variant that actually helps)
    bg_side = sub_2pi[:, -fit_n:] if FIT_RIGHT else sub_2pi[:, :fit_n]
    med = float(np.median(bg_side))
    cell_mask = np.abs(sub_2pi - med) > CELL_THRESH
    bgcol = np.nanmean(np.where(cell_mask, np.nan, sub_2pi), axis=0)
    valid = ~np.isnan(bgcol)
    bgmask_base = np.polyval(np.polyfit(x[valid], bgcol[valid], BGMASK_DEG), x)

    bgmask_crop = apply_2pi_bgmask_crop(sub_large.copy(), TILT_CROP_H, TILT_CROP_H, fit_right=FIT_RIGHT)

    # ---- residual metric across picked frames: far (cell-side) bg level offset ----
    # Cell-masked background column profile; report |mean| over the FAR third
    # (left for FIT_RIGHT), where linear single-side extrapolation leaves a bias.
    far_lo, far_hi = (5, 88) if FIT_RIGHT else (TILT_CROP_H - 88, TILT_CROP_H - 5)
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    def far_offset(corrected):
        m = np.abs(corrected - np.median(corrected[:, -fit_n:] if FIT_RIGHT else corrected[:, :fit_n])) > CELL_THRESH
        bc = np.nanmean(np.where(m, np.nan, corrected), axis=0)
        return float(np.nanmean(bc[far_lo:far_hi]))

    res_lin, res_bg = [], []
    for t in pick:
        frr = frame_results[t]
        s = select_for_frame(frr, grid_cal, pos_map)
        _sx, _sy, _xi, _yi, _cdx, _cdy, _rx, _ry = s
        _tl = tifffile.imread(tl_frames[t]).astype(np.float64)
        _g = get_grid_image(pos_map, _xi, _yi)
        sl = build_sub_large(_tl, _g, rois[ch], _cdx, _cdy, _rx, _ry)
        cl = apply_2pi_tilt_crop_linear(sl.copy(), TILT_CROP_H, TILT_CROP_H, fit_right=FIT_RIGHT)
        cb = apply_2pi_bgmask_crop(sl.copy(), TILT_CROP_H, TILT_CROP_H, fit_right=FIT_RIGHT)
        res_lin.append(abs(far_offset(cl)))
        res_bg.append(abs(far_offset(cb)))
    res_lin = np.array(res_lin); res_bg = np.array(res_bg)
    print(f"[metric] far-bg |offset|  linear={res_lin.mean():.4f}  bgmask-deg{BGMASK_DEG}={res_bg.mean():.4f}")

    # ---- write the two output trees ----
    print("writing linear tree...")
    run_tree(apply_2pi_tilt_crop_linear, rois, frame_results, grid_cal, pos_map,
             tl_frames, pick, OUT_ROOT / "linear")
    print("writing bgmask tree...")
    run_tree(apply_2pi_bgmask_crop, rois, frame_results, grid_cal, pos_map,
             tl_frames, pick, OUT_ROOT / "bgmask")

    # ---- figure ----
    try:
        from figure_logger import save_figure
    except Exception:
        save_figure = None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bgmask_crop_disp = bgmask_crop
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    ax = axs[0, 0]
    ax.plot(x, prof, color="k", lw=1.0, label="delta profile (mean over rows)")
    ax.plot(x, lin_base, color="tab:red", lw=1.5, label="current: linear, right 1/3 only")
    ax.plot(x, bgmask_base, color="tab:green", lw=1.5,
            label=f"robust: cell-masked both-ends deg{BGMASK_DEG}")
    fl = (far_lo, far_hi)
    ax.axvspan(fl[0], fl[1], color="tab:blue", alpha=0.08, label="far bg (metric)")
    bg_lo = TILT_CROP_H - fit_n if FIT_RIGHT else 0
    ax.axvspan(bg_lo, bg_lo + fit_n, color="tab:gray", alpha=0.10, label="fit-side 1/3")
    ax.set_xlabel("column (px, channel axis)")
    ax.set_ylabel("phase (rad)")
    ax.set_title(f"Pos75 ch02 frame {REP_FRAME}: baseline fit")
    ax.legend(fontsize=7)

    vlim = np.percentile(np.abs(lin_crop), 99)
    ax = axs[0, 1]
    im = ax.imshow(lin_crop, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title("current linear-tilt subtracted")
    ax.set_xlabel("column (px)"); ax.set_ylabel("row (px)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[1, 0]
    im = ax.imshow(bgmask_crop_disp, cmap="RdBu_r", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title(f"robust bgmask-deg{BGMASK_DEG} subtracted")
    ax.set_xlabel("column (px)"); ax.set_ylabel("row (px)")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax = axs[1, 1]
    ax.plot(pick, res_lin, "o-", color="tab:red", ms=3, label=f"current linear (mean {res_lin.mean():.3f})")
    ax.plot(pick, res_bg, "o-", color="tab:green", ms=3, label=f"bgmask-deg{BGMASK_DEG} (mean {res_bg.mean():.3f})")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("frame index")
    ax.set_ylabel("far-bg |level offset| (rad)")
    ax.set_title("far (cell-side) background bias (lower = better)")
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save_figure is not None:
        save_figure(
            fig,
            params={"pos": "Pos75", "ch": ch, "rep_frame": REP_FRAME,
                    "cell_thresh": CELL_THRESH, "bgmask_deg": BGMASK_DEG,
                    "fit_right": FIT_RIGHT, "tilt_crop_h": TILT_CROP_H, "n_pick": len(pick)},
            description="Compare current linear (single-side 1/3) vs robust cell-masked "
                        "both-ends polynomial tilt correction in grid_subtract on 260517 "
                        "Pos75 ch02. Top-left: delta column profile with both baselines. "
                        "Right/bottom-left: subtracted crops. Bottom-right: far (cell-side) "
                        "background level offset across frames (target 0).",
            data={"x": x, "prof": prof, "lin_base": lin_base, "bgmask_base": bgmask_base,
                  "lin_crop": lin_crop, "bgmask_crop": bgmask_crop_disp,
                  "pick": np.array(pick), "res_lin": res_lin, "res_bg": res_bg},
        )
    else:
        out_png = OUT_ROOT / "bgmask_vs_linear_ch02.png"
        fig.savefig(str(out_png), dpi=150)
        print(f"figure saved: {out_png}")
    plt.close(fig)
    print("Done")


if __name__ == "__main__":
    main()
