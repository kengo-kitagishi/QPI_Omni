# %%
"""
apply_final_2d_flatten.py
-------------------------
FINAL pipeline step, run AFTER correct_0pergluc.py.

Applies a cell-masked, both-ends 2D quadratic background flattening (the
"mask2d" method) to each per-frame channel crop.  This is intentionally the
LAST step: mask2d assumes the background should be flat, which only holds once
the medium difference has been physically removed by correct_0pergluc.  Running
it earlier (on the 0% glucose period, where the channel still carries the real
~-0.1 rad 0%-medium phase) would erase real signal.

Pipeline position:
    compute_pos_shifts -> grid_subtract (linear tilt) -> correct_0pergluc
    -> apply_final_2d_flatten (THIS, 2D background polish)

What it does per (frame, channel):
  1. Re-derive the SAME valid (non-OOB) mask grid_subtract used, from the
     stored per-frame grid cell + residual + calibration, then erode 1 px at
     the OOB boundary (drops the FFT reconstruction-edge column).
  2. Cell mask: |img - bg_median(fit side)| > CELL_THRESH, dilated.
  3. Fit a 2D quadratic surface (1, c, r, c^2, r^2, c*r) on the background
     pixels = (NOT cell) AND valid, using BOTH channel ends (no one-sided
     extrapolation), and subtract it.
  4. Re-zero the OOB pixels (img * valid_out).
  Graceful: if too few background pixels, leave the frame unchanged (identity)
  and count it -- no silent distortion.

The linear tilt inside grid_subtract / correct_0pergluc is left UNTOUCHED; this
only adds a final 2D polish that also removes the width-direction (row) bow the
1D linear tilt cannot.

Output: a sibling tree ``<crop_sub_rawraw>/<z>_flat/chNN/`` (non-destructive).
Set IN_PLACE=True to overwrite the correct_0pergluc output in place instead.
"""
import sys
import json
import re
import glob
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent))
import grid_subtract as gs
from grid_subtract import (
    apply_inverse_shift_warp, extract_rect_roi, load_grid_calibration,
)

# ============================================================
# Parameters
# ============================================================
# Batch: session root that contains PosN/output_phase/channels/...
PH_SESSION_ROOT = r"E:\260517\2per_0055per_0per_2per_crop_sub"
POS_NUMBERS_TO_RUN = list(range(1, 105))   # [] -> single-Pos mode below
GRID_2PER_DIR = r"E:\260517\grid_2pergluc_2"   # for grid_calibration_PosN.json

# Per-Pos sub-paths (under PosN/output_phase/channels/)
DATA_SUBDIR   = "crop_sub_rawraw/z000"     # contains chNN/ (correct_0pergluc output)
SHIFTS_NAME   = "pos_shifts_cal_online.json"
ROIS_NAME     = "channel_rois.json"

# Single-Pos mode (used only when POS_NUMBERS_TO_RUN is empty)
SINGLE_DATA_DIR = r"E:\260517\2per_0055per_0per_2per_crop_sub\Pos75\output_phase\channels\crop_sub_rawraw\z000"
SINGLE_BASE_LABEL = "Pos75"

# Output: sibling "<z>_flat" tree by default; True -> overwrite in place.
IN_PLACE = False
OUTPUT_SUFFIX = "_flat"

# mask2d parameters
CELL_THRESH = 0.5     # rad, relative to fit-side background median
DILATION    = 2       # cell-mask dilation (px)
MIN_BG      = 100     # min background pixels; else leave frame unchanged
VALID_ERODE_PX = 1    # erode valid mask at OOB boundary (drop reconstruction edge)
POS_SPLIT   = 53
FIT_RIGHT_OVERRIDE = None   # None -> auto from Pos number vs POS_SPLIT

RECON_DIM = gs.RECONSTRUCTED_DIM   # 511
N_PARALLEL_FRAMES = 8
# ============================================================


def _resolve_fit_right(base_label):
    if FIT_RIGHT_OVERRIDE is not None:
        return bool(FIT_RIGHT_OVERRIDE)
    m = re.match(r"Pos(\d+)", str(base_label))
    return bool(m and int(m.group(1)) >= POS_SPLIT)


def _nominal_cal(xi, yi, shifts):
    """Nominal (cal_dx, cal_dy) fallback, matching grid_subtract.select_grid."""
    pixel_scale_um = (gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION
                      * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6)
    x_step = shifts.get("x_step_um", 0.1)
    y_step = shifts.get("y_step_um", 0.1)
    ssx = shifts.get("shift_sign_x", gs.SHIFT_SIGN_X)
    ssy = shifts.get("shift_sign_y", gs.SHIFT_SIGN_Y)
    return (ssy * yi * y_step / pixel_scale_um,
            ssx * xi * x_step / pixel_scale_um)


def flatten_2d(img, valid_out, fit_right):
    """Cell-masked, both-ends 2D quadratic flatten.  Returns (out, ok)."""
    crop_w, out_h = img.shape
    fn = out_h // 3
    bg_side = img[:, -fn:] if fit_right else img[:, :fn]
    med = float(np.median(bg_side))
    cell = np.abs(img - med) > CELL_THRESH
    if DILATION:
        cell = binary_dilation(cell, iterations=DILATION)
    bgm = (~cell) & valid_out
    if int(bgm.sum()) < MIN_BG:
        return img, False
    cols, rows = np.meshgrid(np.arange(out_h), np.arange(crop_w))
    c = cols[bgm] / out_h
    r = rows[bgm] / crop_w
    z = img[bgm]
    A = np.c_[np.ones_like(c), c, r, c * c, r * r, c * r]
    co, *_ = np.linalg.lstsq(A, z, rcond=None)
    cc = cols / out_h
    rr = rows / crop_w
    base = (co[0] + co[1] * cc + co[2] * rr
            + co[3] * cc * cc + co[4] * rr * rr + co[5] * cc * rr)
    return img - base, True


def run_pos(data_dir, shifts_json, rois_json, cal_json, base_label):
    data_dir = Path(data_dir)
    fit_right = _resolve_fit_right(base_label)

    shifts = json.loads(Path(shifts_json).read_text(encoding="utf-8"))
    frame_results = shifts.get("frame_results") or shifts.get("alignment_results")
    rois = json.loads(Path(rois_json).read_text(encoding="utf-8"))
    n_ch = len(rois)
    grid_cal = load_grid_calibration(str(cal_json)) if Path(cal_json).exists() else {}

    # filenames from ch00 (same across channels, sorted == frame order)
    ch0_files = sorted((data_dir / "ch00").glob("*.tif"))
    filenames = [f.name for f in ch0_files]
    if len(filenames) != len(frame_results):
        print(f"  [WARN] {base_label}: file count {len(filenames)} != frame_results {len(frame_results)}")

    out_root = data_dir if IN_PLACE else data_dir.parent / (data_dir.name + OUTPUT_SUFFIX)
    for ch in range(n_ch):
        (out_root / f"ch{ch:02d}").mkdir(parents=True, exist_ok=True)

    fallback = [0]
    ones_full = np.ones((RECON_DIM, RECON_DIM), dtype=np.float32)

    def _process(idx):
        entry = frame_results[idx]
        if entry is None:
            return
        xi, yi = int(entry["grid_xi"]), int(entry["grid_yi"])
        rx = float(entry.get("residual_x_px", 0.0))
        ry = float(entry.get("residual_y_px", 0.0))
        if (xi, yi) in grid_cal:
            cal_dx, cal_dy = grid_cal[(xi, yi)]
        else:
            cal_dx, cal_dy = _nominal_cal(xi, yi, shifts)
        valid_full = apply_inverse_shift_warp(ones_full, rx, ry)
        fname = filenames[idx]
        for ch in range(n_ch):
            roi = rois[ch] if ch < len(rois) else rois[-1]
            tif_path = data_dir / f"ch{ch:02d}" / fname
            if not tif_path.exists():
                continue
            img = tifffile.imread(str(tif_path)).astype(np.float64)
            crop_w, out_h = img.shape
            crop_cx = int(round(roi["cx"] + cal_dx))
            crop_cy = int(round(roi["cy"] + cal_dy))
            valid_out = extract_rect_roi(valid_full, crop_cy, crop_cx, crop_w, out_h) > 0.999
            if VALID_ERODE_PX > 0:
                valid_out = binary_erosion(valid_out, iterations=VALID_ERODE_PX, border_value=1)
            out, ok = flatten_2d(img, valid_out, fit_right)
            if not ok:
                fallback[0] += 1
            out = out * valid_out
            tifffile.imwrite(str(out_root / f"ch{ch:02d}" / fname), out.astype(np.float32))

    with ThreadPoolExecutor(max_workers=N_PARALLEL_FRAMES) as ex:
        futs = [ex.submit(_process, i) for i in range(len(filenames))]
        for f in tqdm(as_completed(futs), total=len(filenames), desc=f"{base_label} 2D-flatten"):
            f.result()

    log = {
        "input_dir": str(data_dir),
        "output_dir": str(out_root),
        "in_place": IN_PLACE,
        "fit_right": fit_right,
        "cell_thresh": CELL_THRESH,
        "dilation": DILATION,
        "valid_erode_px": VALID_ERODE_PX,
        "min_bg": MIN_BG,
        "method": "mask2d_both_ends_quadratic_surface",
        "n_frames": len(filenames),
        "n_channels": n_ch,
        "identity_fallbacks": fallback[0],
        "note": "Final 2D background polish after correct_0pergluc. OOB re-derived from "
                "stored grid cell + residual + calibration, eroded 1px, excluded from fit and re-zeroed.",
    }
    (out_root / "final_2d_flatten_log.json").write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  {base_label}: wrote {len(filenames)} frames x {n_ch} ch -> {out_root}  "
          f"(identity-fallbacks {fallback[0]})")


def main():
    if POS_NUMBERS_TO_RUN:
        session = Path(PH_SESSION_ROOT)
        grid2 = Path(GRID_2PER_DIR)
        for n in POS_NUMBERS_TO_RUN:
            label = f"Pos{n}"
            ch_dir = session / label / "output_phase" / "channels"
            data_dir = ch_dir / DATA_SUBDIR
            if not (data_dir / "ch00").is_dir():
                print(f"SKIP {label}: {data_dir}/ch00 not found")
                continue
            shifts_json = ch_dir / SHIFTS_NAME
            rois_json = ch_dir / ROIS_NAME
            cal_json = grid2 / f"grid_calibration_{label}.json"
            if not shifts_json.exists() or not rois_json.exists():
                print(f"SKIP {label}: missing {SHIFTS_NAME} or {ROIS_NAME}")
                continue
            print("=" * 60)
            print(f"  final 2D flatten - {label}")
            run_pos(data_dir, shifts_json, rois_json, cal_json, label)
    else:
        ch_dir = Path(SINGLE_DATA_DIR).parents[1]  # .../channels
        run_pos(
            SINGLE_DATA_DIR,
            ch_dir / SHIFTS_NAME,
            ch_dir / ROIS_NAME,
            Path(GRID_2PER_DIR) / f"grid_calibration_{SINGLE_BASE_LABEL}.json",
            SINGLE_BASE_LABEL,
        )
    print("Done")


if __name__ == "__main__":
    main()
