# %%
"""
gen_tilt_timelapse_compare.py
-----------------------------
TEST ONLY -- generate timelapse TIF stacks of the grid-subtracted delta for
Pos75 ch02/ch03, frames 0..N, under three channel-axis background-removal
methods, so they can be scrubbed/compared in ImageJ:

    method 'linear'  : current production (linear fit on one background 1/3)
    method 'bg1d2'   : cell-masked, both-ends degree-2 polynomial (1D)
    method 'surf2'   : cell-masked, 2D quadratic surface (extrapolates under cell)

Everything except the channel-axis baseline (step 2) is the unmodified
grid_subtract math (residual subpixel warp, grid selection via calibration,
2pi integer offset, crop).  No new warp / no re-reconstruction.

Output (OUTSIDE the crop_sub tree):
    E:/260517/_iarpls_tilt_test/Pos75/timelapse/ch{02,03}_{linear,bg1d2,surf2}.tif
Each is an ImageJ stack (T, H=40, W=270), float32.
"""
import sys
import json
import glob
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).parent))
import grid_subtract as gs

# ---- config (Pos75, from existing logs) ----
TLP  = r"E:\260517\2per_0055per_0per_2per\Pos75\z000\output_phase_raw"
GRID = r"E:\260517\grid_2pergluc_2"
CAL  = r"E:\260517\grid_2pergluc_2\grid_calibration_Pos75.json"
SH   = r"E:\260517\2per_0055per_0per_2per_crop_sub\Pos75\output_phase\channels\pos_shifts_cal_online.json"
ROI  = r"E:\260517\2per_0055per_0per_2per_crop_sub\Pos75\output_phase\channels\channel_rois.json"
OUT  = Path(r"E:\260517\_iarpls_tilt_test\Pos75\timelapse")

GRID_Z = 8
TH = 270            # tilt/output crop length along channel axis (== out crop_h)
FIT_RIGHT = True    # Pos75 >= POS_SPLIT(53)
CELL_THRESH = 0.5
DILATION = 2          # morphological dilation of the cell mask (mask2d)
VALID_ERODE = 1       # erode valid mask by Npx at the OOB boundary: valid_out (warp-ones)
                      # keeps the first in-frame column, but that column sits at the 511-frame
                      # edge and carries FFT-reconstruction edge artifacts -> drop it too.
MASK2D_MIN_BG = 100   # min background pixels for the mask2d fit; else fall back to linear
CHANNELS = [2, 3]
T_START, T_END = 0, 1000   # inclusive

PIXEL_SCALE_UM = (gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION
                  * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6)


def make_grid_axes(crop_w):
    cols, rows = np.meshgrid(np.arange(TH), np.arange(crop_w))
    return cols.astype(np.float64), rows.astype(np.float64)


# All three fit ONLY the POS_SPLIT-designated background third (right for
# FIT_RIGHT) -- cell-free by design, so no cell mask is needed -- and
# extrapolate the model across the whole channel.

def _bg_cols():
    """Column slice of the background third (POS_SPLIT-dependent)."""
    fn = TH // 3
    return slice(TH - fn, TH) if FIT_RIGHT else slice(0, fn)


def correct_linear(s, fn, x):
    prof = s.mean(axis=0)
    if FIT_RIGHT:
        a, b = np.polyfit(x[-fn:], prof[-fn:], 1)
    else:
        a, b = np.polyfit(x[:fn], prof[:fn], 1)
    return s - (a * x + b)[np.newaxis, :]


def correct_curve(s, fn, x):
    """1D degree-2 fit on the background-third column profile, extrapolated."""
    prof = s.mean(axis=0)
    if FIT_RIGHT:
        coef = np.polyfit(x[-fn:], prof[-fn:], 2)
    else:
        coef = np.polyfit(x[:fn], prof[:fn], 2)
    return s - np.polyval(coef, x)[np.newaxis, :]


def correct_surf2(s, fn, cols, rows):
    """2D quadratic surface fit on the background-third pixels, extrapolated."""
    cw = s.shape[0]
    csl = _bg_cols()
    c = (cols[:, csl] / TH).ravel()
    r = (rows[:, csl] / cw).ravel()
    z = s[:, csl].ravel()
    A = np.c_[np.ones_like(c), c, r, c * c, r * r, c * r]
    co, *_ = np.linalg.lstsq(A, z, rcond=None)
    cc = cols / TH
    rr = rows / cw
    base = co[0] + co[1] * cc + co[2] * rr + co[3] * cc * cc + co[4] * rr * rr + co[5] * cc * rr
    return s - base


def correct_mask2d(s, fn, cols, rows, valid_out):
    """Cell-masked, both-ends 2D quadratic surface.  OOB pixels excluded from
    the fit via valid_out.  Returns None if too few background pixels (caller
    falls back to linear and counts it -- no silent fallback)."""
    cw = s.shape[0]
    med = np.median(s[:, -fn:] if FIT_RIGHT else s[:, :fn])
    cell = np.abs(s - med) > CELL_THRESH
    if DILATION:
        cell = binary_dilation(cell, iterations=DILATION)
    bgm = (~cell) & valid_out
    if int(bgm.sum()) < MASK2D_MIN_BG:
        return None
    c = cols[bgm] / TH
    r = rows[bgm] / cw
    z = s[bgm]
    A = np.c_[np.ones_like(c), c, r, c * c, r * r, c * r]
    co, *_ = np.linalg.lstsq(A, z, rcond=None)
    cc = cols / TH
    rr = rows / cw
    base = co[0] + co[1] * cc + co[2] * rr + co[3] * cc * cc + co[4] * rr * rr + co[5] * cc * rr
    return s - base


def main():
    rois = json.loads(Path(ROI).read_text(encoding="utf-8"))
    frame_results = json.loads(Path(SH).read_text(encoding="utf-8"))["frame_results"]
    grid_cal = gs.load_grid_calibration(CAL)
    pos_map = gs.scan_grid_positions(GRID, "Pos75")
    tl_frames = sorted(glob.glob(str(Path(TLP) / "img_*_phase.tif")))
    OUT.mkdir(parents=True, exist_ok=True)

    fn = TH // 3
    x = np.arange(TH, dtype=np.float64)
    grid_cache = {}

    def get_grid(xi, yi):
        key = (xi, yi)
        if key not in grid_cache:
            p = pos_map[key] / "output_phase_raw" / f"img_000000000_ph_{GRID_Z:03d}_phase.tif"
            grid_cache[key] = tifffile.imread(str(p)).astype(np.float64)
        return grid_cache[key]

    frames = list(range(T_START, T_END + 1))
    print(f"frames {frames[0]}..{frames[-1]} ({len(frames)})  channels={CHANNELS}")

    # one folder per (channel, method); 1001 individual TIFs each
    methods = ["linear", "curve", "surf2", "mask2d"]
    mask2d_fallbacks = 0
    out_dirs = {}
    for ch in CHANNELS:
        for m in methods:
            d = OUT / f"ch{ch:02d}_{m}"
            d.mkdir(parents=True, exist_ok=True)
            out_dirs[(ch, m)] = d

    axes_cache = {}
    for k, t in enumerate(frames):
        f = frame_results[t]
        o = gs.select_grid(f["shift_x_avg"], f["shift_y_avg"], pos_map, grid_cal,
                           PIXEL_SCALE_UM, x_step=0.1, y_step=0.1,
                           shift_sign_x=1, shift_sign_y=1)
        xi, yi, _, _, _, cdx, cdy, rx, ry = o
        tl = tifffile.imread(tl_frames[t]).astype(np.float64)
        tlw = gs.apply_inverse_shift_warp(tl, rx, ry)
        # OOB / valid mask, same as grid_subtract.process_single_frame: warp(ones)
        # then crop (extract_rect_roi zero-pads out-of-frame -> valid_out False there).
        valid_full = gs.apply_inverse_shift_warp(np.ones_like(tl, dtype=np.float32), rx, ry)
        grid_img = get_grid(xi, yi)
        fname = Path(tl_frames[t]).name.replace("_phase.tif", ".tif")  # img_{t:09d}_ph_000.tif
        for ch in CHANNELS:
            roi = rois[ch]
            cw = roi["crop_w"]
            ccx = int(round(roi["cx"] + cdx))
            ccy = int(round(roi["cy"] + cdy))
            tll = gs.extract_rect_roi(tlw, ccy, ccx, cw, TH)
            gll = gs.extract_rect_roi(grid_img, ccy, ccx, cw, TH)
            valid_out = gs.extract_rect_roi(valid_full, ccy, ccx, cw, TH) > 0.999
            if VALID_ERODE:
                # border_value=1 -> only erode True pixels adjacent to the interior
                # OOB region, not the genuinely-valid outer crop edges.
                valid_out = binary_erosion(valid_out, iterations=VALID_ERODE, border_value=1)
            s = tll - gll
            bm = np.mean(s[:, -fn:] if FIT_RIGHT else s[:, :fn])
            kk = int(round(bm / (2 * np.pi)))
            if kk != 0:
                s = s - kk * 2 * np.pi
            if cw not in axes_cache:
                axes_cache[cw] = make_grid_axes(cw)
            cols, rows = axes_cache[cw]
            m2 = correct_mask2d(s.copy(), fn, cols, rows, valid_out)
            if m2 is None:
                m2 = correct_linear(s.copy(), fn, x)
                mask2d_fallbacks += 1
            outs = {
                "linear": correct_linear(s.copy(), fn, x),
                "curve": correct_curve(s.copy(), fn, x),
                "surf2": correct_surf2(s.copy(), fn, cols, rows),
                "mask2d": m2,
            }
            for m, img in outs.items():
                img = img * valid_out  # zero OOB/padded pixels, same as production
                tifffile.imwrite(str(out_dirs[(ch, m)] / fname), img.astype(np.float32))
        if k % 200 == 0:
            print(f"  {k}/{len(frames)}")

    for (ch, m), d in out_dirs.items():
        print(f"saved {len(frames)} tifs -> {d}")
    print(f"mask2d linear-fallbacks (too few bg px): {mask2d_fallbacks} / {len(frames) * len(CHANNELS)}")
    print("Done")


if __name__ == "__main__":
    main()
