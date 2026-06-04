"""bench_subtract_ab.py -- ECC vs SG-NCC end-to-end subtraction A/B

Produce final grid-subtracted crop folders for the SAME small timelapse using
two (or three) alignment estimators, so the result can be judged by eye.

The alignment method enters the pipeline ONLY through the per-frame shift in
pos_shifts.json. We therefore reuse the production scripts verbatim and only
swap the per-channel estimator:

  ECC-uint8 (current production)  : compute_pos_shifts.ecc_align  on uint8
  SG-NCC    (Qiita method)        : matchTemplate(NCC) + SG subpixel peak
  ECC-float (optional)            : same ECC but fed float32 (to_uint8 bypassed)

For each method we:
  1. drive compute_pos_shifts.main() -> pos_shifts_<tag>.json
  2. drive grid_subtract.main()      -> crop_sub_<tag>/  (final subtracted crops)

No new warp / crop / subtract code is written here -- the math is 100% the
existing pipeline; only the translation estimator differs between arms, so any
visible difference in the subtractions is attributable to the estimator alone.

Sign note: ecc_align(ref, mov) returns (+shift); the SG-NCC adapter negates its
raw peak offset to match that convention. A self-test asserts the two agree on
a known shift before any pipeline run (no silent sign flip).

Usage
-----
    python scripts/bench_subtract_ab.py --n-frames 30 --start 0 --methods ecc sg
    python scripts/bench_subtract_ab.py --n-frames 30 --methods ecc sg ecc_float
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from ecc_utils import ecc_align, to_uint8
from bench_ecc_vs_sgpeak import sg_peak_subpix, SG_WIN

DEFAULT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"
NCC_MARGIN = 8  # search half-window for matchTemplate (px)


# ==========================================================================
# SG-NCC adapter: drop-in replacement for ecc_utils.ecc_align
# Signature/sign match ecc_align: (ref_u8, tl_u8) -> (tx, ty, corr)
# tx = X(col) shift, ty = Y(row) shift, same sign as ecc_align.
# ==========================================================================

def sg_ncc_align(ref_u8, tl_u8, margin=NCC_MARGIN):
    ref = ref_u8.astype(np.float32)
    mov = tl_u8.astype(np.float32)
    m = margin
    th, tw = mov.shape[0] - 2 * m, mov.shape[1] - 2 * m
    if th < SG_WIN or tw < SG_WIN:
        return None
    template = mov[m:m + th, m:m + tw]
    try:
        surf = cv2.matchTemplate(ref, template, cv2.TM_CCOEFF_NORMED)
    except cv2.error:
        return None
    if surf.shape[0] < SG_WIN or surf.shape[1] < SG_WIN:
        return None
    corr = float(surf.max())
    py, px = sg_peak_subpix(surf.astype(np.float32))
    # Zero-shift peak sits at (m, m); raw offset = peak - m is -(true shift),
    # so negate to match ecc_align's (+shift) convention.
    tx = -(px - m)
    ty = -(py - m)
    return float(tx), float(ty), corr


def selftest_sign(ref_u8):
    """Assert sg_ncc_align matches ecc_align sign/value on a known shift."""
    H, W = ref_u8.shape
    M = np.float32([[1, 0, 1.7], [0, 1, -1.3]])  # shift content by (+1.7 col, -1.3 row)
    mov_u8 = cv2.warpAffine(ref_u8, M, (W, H), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT)
    e = ecc_align(ref_u8, mov_u8)
    s = sg_ncc_align(ref_u8, mov_u8)
    if e is None or s is None:
        raise RuntimeError("sign self-test: estimator returned None")
    print(f"  self-test  ecc=({e[0]:+.3f},{e[1]:+.3f})  "
          f"sg-ncc=({s[0]:+.3f},{s[1]:+.3f})  (expect ~(+1.7,-1.3))")
    if np.sign(e[0]) != np.sign(s[0]) or np.sign(e[1]) != np.sign(s[1]):
        raise RuntimeError("sign self-test FAILED: ecc and sg-ncc disagree on sign")
    if abs(e[0] - s[0]) > 0.3 or abs(e[1] - s[1]) > 0.3:
        raise RuntimeError(f"sign self-test FAILED: ecc/sg-ncc differ > 0.3px {e[:2]} vs {s[:2]}")


# ==========================================================================
# Pipeline drivers (mirror batch_pipeline_all_pos step2 / step3)
# ==========================================================================

def configure_cps(cps, cfg, pos, channels_dir, rois_json, n_frames, out_json,
                  ecc_min_corr):
    grid_dir = cfg["grid_dir"]
    cps.CHANNELS_DIR = str(channels_dir)
    cps.CHANNEL_ROIS_JSON = str(rois_json)
    cps.GRID_DIR = grid_dir
    cps.GRID_BASE_LABEL = f"Pos{pos}"
    cps.POS_SPLIT = cfg["pos_split"]
    cps.GRID_Z_INDEX = cfg["raw_grid_z_index"]
    cps.GRID_CALIBRATION_JSON = str(Path(grid_dir) / f"grid_calibration_Pos{pos}.json")
    cps.ECC_CROP_H = cfg["ecc_crop_h"]
    cps.TILT_CROP_H = cfg["tilt_crop_h"]
    cps.USE_SLOPE_CORRECTION = True
    cps.USE_GRID_REFERENCE = True
    cps.USE_INCREMENTAL_TRACKING = True
    cps.USE_SECOND_PASS_ECC = True
    cps.USE_THIRD_PASS_ECC = True
    cps.OUTLIER_MAD_THRESH = 5.0
    cps.OUTLIER_TIMESERIES_THRESH = 0.0
    cps.ECC_MIN_CORR = ecc_min_corr  # estimator-appropriate (NCC corr scale differs)
    cps.VMIN, cps.VMAX = cfg["ecc_vmin"], cfg["ecc_vmax"]
    cps.MAX_FRAMES = n_frames
    cps.N_WORKERS = 8
    cps.TL_Z_INDEX = 0
    cps.SHIFT_SIGN_X = 1
    cps.SHIFT_SIGN_Y = 1
    cps.OUTPUT_JSON = out_json
    cps.SAVE_CORR_DATA = False
    cps.APPLY_BACKSUB_TO_GRID_REF = True
    cps.TILT_FIT_RIGHT = pos >= cfg["pos_split"]


def configure_gs(gs, cfg, pos, tl_pos_dir, channels_dir, rois_json, shifts_json,
                 out_dir, n_frames):
    grid_dir = cfg["grid_dir"]
    crop = cfg["crop_before"] if pos < cfg["pos_split"] else cfg["crop_after"]
    gs.TIMELAPSE_DIR = str(tl_pos_dir)
    gs.SHIFTS_JSON = str(shifts_json)
    gs.CHANNEL_ROIS_JSON = str(rois_json)
    gs.GRID_DIR = grid_dir
    gs.BASE_LABEL = f"Pos{pos}"
    gs.GRID_CALIBRATION_JSON = str(Path(grid_dir) / f"grid_calibration_Pos{pos}.json")
    gs.OUTPUT_DIR = str(out_dir)
    gs.TL_Z_INDEX = 0
    gs.GRID_Z_INDEX = cfg["raw_grid_z_index"]
    gs.MAX_FRAMES = n_frames
    gs.PICK_FRAMES = None
    gs.APPLY_SUBPIXEL_CORRECTION = True
    gs.APPLY_INVERSE_SHIFT = False
    gs.OUTPUT_CROP_H = cfg["tilt_crop_h_raw"]
    gs.OUTPUT_SAVE_FULL_FRAME = False
    gs.USE_RAW_PHASE = True
    raw_phase_dir = tl_pos_dir / "output_phase_raw"
    gs.TL_PHASE_DIR = str(raw_phase_dir) if raw_phase_dir.exists() else None
    gs.RAW_CROP = tuple(crop)
    gs.RAW_TL_Z_INDEX = cfg["raw_tl_z_index"]
    gs.RAW_GRID_Z_INDEX = cfg["raw_grid_z_index"]
    gs.TILT_CROP_H_RAW = cfg["tilt_crop_h_raw"]
    gs.SHIFT_SIGN_X = -1
    gs.SHIFT_SIGN_Y = -1
    gs.X_STEP = cfg["crop_sub_x_step_um"]
    gs.Y_STEP = cfg["crop_sub_y_step_um"]


METHODS = {
    # tag: (patch_ecc_align, patch_to_uint8_passthrough, ecc_min_corr)
    "ecc":       (None,         False, 0.50),
    "sg":        (sg_ncc_align, False, 0.30),
    "ecc_float": (None,         True,  0.50),
}


def run_method(tag, cfg, pos, tl_pos_dir, channels_dir, rois_json, n_frames, out_root):
    import importlib
    import compute_pos_shifts as cps
    import grid_subtract as gs
    importlib.reload(cps)  # fresh module globals per arm
    importlib.reload(gs)

    patch_align, patch_float, ecc_min_corr = METHODS[tag]
    # pos_shifts json (small file) stays next to phase in channels/.
    # crop_sub output folders go to a separate benchmark root, NOT inside the
    # production crop_sub tree (per user constraint).
    shifts_json = channels_dir / f"pos_shifts_{tag}.json"
    out_dir = out_root / f"Pos{pos}" / f"crop_sub_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_cps(cps, cfg, pos, channels_dir, rois_json, n_frames,
                  out_json=shifts_json.name, ecc_min_corr=ecc_min_corr)
    if patch_align is not None:
        cps.ecc_align = patch_align
    if patch_float:
        # Feed float32 to ecc_align instead of uint8 (isolate quantization).
        cps.to_uint8 = lambda img, vmin=None, vmax=None: img.astype(np.float32)

    print(f"\n=== [{tag}] compute_pos_shifts ({n_frames} frames) ===")
    cps.main()

    print(f"\n=== [{tag}] grid_subtract -> {out_dir.name} ===")
    configure_gs(gs, cfg, pos, tl_pos_dir, channels_dir, rois_json,
                 shifts_json, out_dir, n_frames)
    gs.main()
    return out_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--zsub", default="z000", help="z subdir under PosN (z-stack)")
    p.add_argument("--n-frames", type=int, default=30)
    p.add_argument("--methods", nargs="+", default=["ecc", "sg"],
                   choices=list(METHODS))
    p.add_argument("--out-root", default=None,
                   help="benchmark output root (default: <save_dir>/../ecc_sg_ab). "
                        "crop_sub folders are written here, NOT in the production tree.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pos = args.pos
    grid_dir = Path(cfg["grid_dir"])

    tl_pos_dir = Path(cfg["save_dir"]) / f"Pos{pos}" / args.zsub
    channels_dir = tl_pos_dir / "output_phase" / "channels"
    channels_dir.mkdir(parents=True, exist_ok=True)
    out_root = Path(args.out_root) if args.out_root else Path(cfg["save_dir"]).parent / "ecc_sg_ab"

    # channel_rois.json must come from the grid (per project convention).
    grid_rois = grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
    if not grid_rois.exists():
        raise FileNotFoundError(f"grid channel_rois.json not found: {grid_rois}")
    rois_json = channels_dir / "channel_rois.json"
    shutil.copyfile(grid_rois, rois_json)

    phase_dir = tl_pos_dir / "output_phase"
    n_avail = len(list(phase_dir.glob("img_*_ph_000_phase.tif")))
    print(f"Timelapse: {tl_pos_dir}")
    print(f"  phase frames available: {n_avail},  using first {args.n_frames}")
    print(f"  channel_rois (from grid): {grid_rois}")
    print(f"  methods: {args.methods}")

    # Sign self-test on a real tilt-corrected uint8 crop.
    import tifffile
    from ecc_utils import tilt_fit_crop
    rois = json.loads(rois_json.read_text(encoding="utf-8"))
    img0 = tifffile.imread(str(sorted(phase_dir.glob("img_*_ph_000_phase.tif"))[0])).astype(np.float64)
    crop0 = tilt_fit_crop(img0, rois[5]["cy"], rois[5]["cx"], rois[5]["crop_w"],
                          cfg["ecc_crop_h"], cfg["tilt_crop_h"])
    if crop0 is None:
        crop0 = tilt_fit_crop(img0, rois[0]["cy"], rois[0]["cx"], rois[0]["crop_w"],
                              cfg["ecc_crop_h"], cfg["tilt_crop_h"])
    print("Sign self-test:")
    selftest_sign(to_uint8(crop0, cfg["ecc_vmin"], cfg["ecc_vmax"]))

    print(f"  output root (isolated): {out_root}")
    outputs = {}
    for tag in args.methods:
        outputs[tag] = run_method(tag, cfg, pos, tl_pos_dir, channels_dir,
                                   rois_json, args.n_frames, out_root)

    print("\n==== DONE ====")
    for tag, out in outputs.items():
        print(f"  {tag:10s} -> {out}")
    print("\nCompare ch00/.. TIFs between the crop_sub_* folders by eye.")


if __name__ == "__main__":
    main()
