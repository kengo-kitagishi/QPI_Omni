"""
batch_quadfit_crop_sub_260908_whitelist.py
-------------------------------------------
Full-timelapse quad-fit crop_sub for 260908, using the whitelist-averaged
offline alignment (pos_shifts_cal_whitelist.json) instead of the online
production shift. Part of the 2026-10-13 realignment: the online average
mixed in cell-bearing channels and produced a cross-channel mass
co-fluctuation (r=+0.260) within a Pos.

Every pixel-level operation here is copied UNCHANGED from
scripts/_qc_outsidech_quadfit_260908.py (quad_fit_outside, the inside/outside/
valid masking, the 2pi offset correction, the output crop) -- that script is
the one the existing 260908_outside_quad lineage review / division QC was
calibrated against (mask_from="grid" arm; see its own docstring for why).
Nothing about the tilt-correction math is new here.

What IS new is the driver: a per-frame loop (the source script processes one
hardcoded --frame across all Pos; this processes the full 0-100h window
across the whitelist-covered Pos only), and the bridge from
compute_pos_shifts.py's shift_x_avg/shift_y_avg (a continuous px value) to
the (grid_xi, grid_yi, residual_x_px, residual_y_px) decomposition the
quad-fit math needs to pick a grid reference node and inverse-warp the
residual onto it. That bridge is also not new math: it is the exact nearest-
grid-node lookup compute_pos_shifts.py already uses internally
(_nearest_from_preloaded) plus grid_subtract.apply_inverse_shift_warp, the
same functions (and the same pos_shifts_cal_online.json fields,
grid_xi/residual_x_px) the online pipeline already produces this way -- see
compute_pos_shifts.py lines ~634-700 for the identical pattern per channel.

Scope: the 58 Pos in channel_whitelist_260908.json, frame 2-1202 (0-100h,
t=(frame-2)/12), ALL channels of each covered Pos (not just the whitelist
channels themselves -- the whitelist only chose which channels anchor the
Pos-level correction; the channels we actually want to see corrected are the
cell-bearing ones).

Output (new tree, nothing existing touched):
  E:/260908_whitelist_realign/crop_sub/PosN/output_phase/channels/
      crop_sub_rawraw/z000/chNN/img_{frame:09d}_ph_000.tif
"""
import sys
import json
import argparse
import subprocess
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))

import grid_subtract as gs                                    # noqa: E402
from ecc_utils import extract_rect_roi                        # noqa: E402
from compute_pos_shifts import (                               # noqa: E402
    load_grid_calibration, _nearest_from_preloaded,
)

# ============================================================
TL_PHASE_ROOT  = Path(r"D:\AquisitionData\Kitagishi\260908\ph_zstack_2")
WHITELIST_JSON = Path(r"D:\AquisitionData\Kitagishi\260908\_qc\channel_whitelist_260908.json")
GRID_DIR       = Path(r"E:\260908\ye_grid_0p05_2")
GRID_Z         = 5
OUT_ROOT       = Path(r"E:\260908_whitelist_realign\crop_sub")

POS_SPLIT   = 52          # tilt-fit direction boundary (fit_right = pos>=POS_SPLIT);
                           # matches datasets/260908_outside_quad.yaml pos_split and
                           # _qc_outsidech_quadfit_260908.py's own --pos-split default.
                           # NOT the raw-hologram crop_before/after boundary (a
                           # different concept, 51 for 260908). Only Pos51 itself
                           # differs between 51/52; outside this run's 58-Pos
                           # whitelist coverage either way.
FRAME_START = 2            # 0-100h window: t=(frame-2)/12
FRAME_END   = 1202         # inclusive

# Unchanged from _qc_outsidech_quadfit_260908.py's own defaults
MASK_FROM   = "grid"       # production arm: grid's output_phase holds no cells
CH_THRESH   = -1.0
DILATE      = 2
MIN_BG      = 500
VALID_ERODE = 1
TILT_H      = 270
OUT_H       = 240


def quad_fit_outside(delta, outside):
    """Verbatim from _qc_outsidech_quadfit_260908.py."""
    h, w = delta.shape
    c, r = np.meshgrid(np.arange(w) / w, np.arange(h) / h)
    terms = [np.ones_like(c), c, r, c * c, r * r, c * r]
    A = np.stack([t[outside] for t in terms], axis=1)
    co, *_ = np.linalg.lstsq(A, delta[outside], rcond=None)
    return delta - sum(k * t for k, t in zip(co, terms))


def available_grid_nodes(grid_dir, label, grid_cal):
    """Grid (xi,yi) nodes that are both calibrated AND have the raw z=GRID_Z
    image on disk, mirroring compute_pos_shifts.py's own preload-success set."""
    avail = set()
    for (xi, yi) in grid_cal.keys():
        node = grid_dir / f"{label}_x{xi:+d}_y{yi:+d}"
        p = node / "output_phase_raw" / f"img_000000000_ph_{GRID_Z:03d}_phase.tif"
        if p.exists():
            avail.add((xi, yi))
    return avail


def run_pos(pos_label):
    pos_num = int(pos_label[3:])
    tl_raw_dir = TL_PHASE_ROOT / pos_label / "z000" / "output_phase_raw"
    tl_ph_dir = TL_PHASE_ROOT / pos_label / "z000" / "output_phase"
    shifts_json = tl_ph_dir / "channels" / "pos_shifts_cal_whitelist.json"
    rois_json = tl_ph_dir / "channels" / "channel_rois.json"
    cal_json = GRID_DIR / f"grid_calibration_{pos_label}.json"
    if not (shifts_json.exists() and rois_json.exists() and cal_json.exists()):
        print(f"  [SKIP] {pos_label}: missing shifts/rois/calibration")
        return "skip_missing"

    out_dir_base = OUT_ROOT / pos_label / "output_phase" / "channels" / "crop_sub_rawraw" / "z000"
    rois = json.loads(rois_json.read_text(encoding="utf-8"))
    for ch in range(len(rois)):
        (out_dir_base / f"ch{ch:02d}").mkdir(parents=True, exist_ok=True)
    done_marker = out_dir_base / "_DONE"
    if done_marker.exists():
        print(f"  [SKIP] {pos_label}: _DONE marker present")
        return "skip_done"

    shifts = json.loads(shifts_json.read_text(encoding="utf-8"))
    frame_results = shifts["frame_results"]
    grid_cal = load_grid_calibration(str(cal_json))
    avail = available_grid_nodes(GRID_DIR, pos_label, grid_cal)
    if not avail:
        print(f"  [ERROR] {pos_label}: no available grid nodes found")
        return "err_no_grid_nodes"

    fit_right = pos_num >= POS_SPLIT
    px = gs.SENSOR_PIXEL_SIZE / gs.MAGNIFICATION * gs.ORIGINAL_DIM / gs.RECONSTRUCTED_DIM * 1e6

    grid_raw_cache = {}
    grid_mask_cache = {}

    def get_grid_imgs(xi, yi):
        key = (xi, yi)
        if key not in grid_raw_cache:
            node = GRID_DIR / f"{pos_label}_x{xi:+d}_y{yi:+d}"
            g_raw = node / "output_phase_raw" / f"img_000000000_ph_{GRID_Z:03d}_phase.tif"
            grid_raw_cache[key] = tifffile.imread(str(g_raw)).astype(np.float64)
            if MASK_FROM == "grid":
                g_ph = node / "output_phase" / f"img_000000000_ph_{GRID_Z:03d}_phase.tif"
                grid_mask_cache[key] = tifffile.imread(str(g_ph)).astype(np.float64)
        return grid_raw_cache[key], grid_mask_cache.get(key)

    n_frames_done = 0
    n_frames_skipped = 0
    n_fallback = 0
    bg_frac = []
    for frame in range(FRAME_START, FRAME_END + 1):
        fr = frame_results[frame] if frame < len(frame_results) else None
        sx = fr.get("shift_x_avg") if fr else None
        sy = fr.get("shift_y_avg") if fr else None
        if sx is None or sy is None:
            n_frames_skipped += 1
            continue

        tl_raw = tl_raw_dir / f"img_{frame:09d}_ph_000_phase.tif"
        if not tl_raw.exists():
            n_frames_skipped += 1
            continue

        xi, yi = _nearest_from_preloaded(sx, sy, grid_cal, avail)
        cal_dx, cal_dy = grid_cal[(xi, yi)]
        rx, ry = sx - cal_dx, sy - cal_dy

        tl_img = tifffile.imread(str(tl_raw)).astype(np.float64)
        valid_full = np.ones_like(tl_img, dtype=np.float32)
        if rx or ry:
            tl_img = gs.apply_inverse_shift_warp(tl_img, rx, ry)
            valid_full = gs.apply_inverse_shift_warp(valid_full, rx, ry)

        grid_img, mask_img = get_grid_imgs(xi, yi)
        if MASK_FROM == "tl":
            tl_ph_path = tl_ph_dir / f"img_{frame:09d}_ph_000_phase.tif"
            mask_img = tifffile.imread(str(tl_ph_path)).astype(np.float64)

        full = tl_img - grid_img
        ones = np.ones_like(tl_img, dtype=np.float64)

        for ch, roi in enumerate(rois):
            cy, cx = int(round(roi["cy"] + cal_dy)), int(round(roi["cx"] + cal_dx))
            delta = extract_rect_roi(full, cy, cx, roi["crop_w"], TILT_H)
            mask_w = extract_rect_roi(mask_img, cy, cx, roi["crop_w"], TILT_H)
            inside = mask_w < CH_THRESH
            if DILATE:
                inside = binary_dilation(inside, iterations=DILATE)
            valid_w = extract_rect_roi(valid_full.astype(np.float64), cy, cx,
                                       roi["crop_w"], TILT_H) > 0.999
            if VALID_ERODE:
                valid_w = binary_erosion(valid_w, iterations=VALID_ERODE, border_value=1)
            valid_w &= extract_rect_roi(ones, cy, cx, roi["crop_w"], TILT_H) > 0.5
            outside = (~inside) & valid_w

            fn = max(1, TILT_H // 3)
            bg = delta[:, -fn:] if fit_right else delta[:, :fn]
            k = int(round(float(np.mean(bg)) / (2.0 * np.pi)))
            if k:
                delta = delta - k * 2.0 * np.pi

            bg_frac.append(outside.mean())
            if outside.sum() >= MIN_BG:
                flat = quad_fit_outside(delta, outside)
            else:
                flat = delta
                n_fallback += 1
            start = (TILT_H - OUT_H) // 2
            crop = flat[:, start:start + OUT_H] * valid_w[:, start:start + OUT_H]
            out_path = out_dir_base / f"ch{ch:02d}" / f"img_{frame:09d}_ph_000.tif"
            tifffile.imwrite(str(out_path), crop.astype(np.float32))
        n_frames_done += 1

    done_marker.write_text(
        json.dumps({"n_frames_done": n_frames_done, "n_frames_skipped": n_frames_skipped,
                    "n_channels": len(rois), "n_fallback_quadfit": n_fallback,
                    "mean_bg_frac": float(np.mean(bg_frac)) if bg_frac else None},
                   indent=2), encoding="utf-8")
    print(f"  {pos_label}: {n_frames_done} frames done, {n_frames_skipped} skipped, "
          f"{n_fallback} quad-fit fallbacks, bg_frac mean={np.mean(bg_frac) if bg_frac else float('nan'):.3f}",
          flush=True)
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=str, default=None)
    args = ap.parse_args()

    whitelist = json.loads(WHITELIST_JSON.read_text(encoding="utf-8"))["whitelist"]

    if args.pos is not None:
        try:
            r = run_pos(args.pos)
        except Exception as e:
            r = f"EXC:{e}"
            print(f"  [EXC] {args.pos}: {e}", flush=True)
            import traceback
            traceback.print_exc()
        print(f"[single] {args.pos} -> {r}", flush=True)
        sys.exit(0 if str(r).startswith(("ok", "skip")) else 1)

    pos_labels = sorted(whitelist.keys(), key=lambda p: int(p[3:]))
    print(f"quadfit crop_sub batch (260908, whitelist-only, 0-100h): "
          f"{len(pos_labels)} Pos [subprocess-per-Pos]", flush=True)
    counts = {}
    for pos_label in pos_labels:
        print(f"\n=== {pos_label} (subprocess) ===", flush=True)
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             "--pos", pos_label]).returncode
        key = "ok" if rc == 0 else f"fail(rc={rc})"
        counts[key] = counts.get(key, 0) + 1
        print(f"[{pos_label}] subprocess rc={rc}", flush=True)
    print("\nSummary:", counts, flush=True)


if __name__ == "__main__":
    main()
