"""bench_subtract_ab.py -- ECC vs SG-NCC end-to-end subtraction A/B (mode A)

Produce final grid-subtracted crop folders for the SAME small timelapse using
several alignment estimators, so the result can be judged by eye.

Why a standalone shift loop (mode A)
------------------------------------
compute_pos_shifts.py computes the per-frame shift with a ProcessPoolExecutor;
its workers re-import the module in fresh processes, so a parent-process
monkeypatch of ecc_align never reaches them. Instead of editing production
code, we compute the per-frame shifts here, IN-PROCESS, reusing the production
building blocks unchanged:

  - tilt_fit_crop  (ecc_utils)            : the exact ECC input crop
  - ecc_align / sg_ncc_align              : the per-channel estimator
  - _frame_result_from_per_channel (cps)  : the exact MAD-outlier + mean
                                            aggregation, imported and reused

Shift convention matches what grid_subtract consumes: shift_x_avg/shift_y_avg
are the shift of each frame relative to grid(0,0) (= pass-1 of compute_pos_shifts).
grid_subtract then picks the nearest grid (xi,yi) and applies the residual warp,
exactly as in production. The only difference from production is the 2nd/3rd
incremental ECC refinement pass, negligible for this drift-corrected data
(<1px for all but the first ~2 startup frames) and applied identically to every
arm, so any visible difference is attributable to the estimator alone.

Estimator inputs (mirror the bench numbers 49 / 9 / 5 nm):
  ecc       : uint8 (current production input)         -> ECC X-RMSE ~49 nm
  sg        : float32 + NCC + SG subpixel peak (Qiita) -> ~9 nm, isotropic
  ecc_float : float32 (uint8 quantization bypassed)    -> ~5 nm

Outputs go to an isolated root (<save_dir>/../ecc_sg_ab) -- NOT into the
production crop_sub tree, and nothing is written into the production/grid dirs.

Usage
-----
    python scripts/bench_subtract_ab.py --n-frames 300 --methods ecc sg ecc_float
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
from ecc_utils import ecc_align, to_uint8, tilt_fit_crop
from bench_ecc_vs_sgpeak import sg_peak_subpix, SG_WIN

DEFAULT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"
SG_MARGIN = 14   # NCC search half-window; must exceed the largest real shift (~10px startup)


# ==========================================================================
# SG-NCC estimator -- same (tx, ty, corr) convention/sign as ecc_align
# ==========================================================================

def sg_ncc_align(ref, mov, margin=SG_MARGIN):
    if ref is None or mov is None:
        return None
    ref = np.asarray(ref, dtype=np.float32)
    mov = np.asarray(mov, dtype=np.float32)
    if ref.ndim != 2 or mov.ndim != 2:
        return None
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
    # zero-shift peak at (m, m); raw offset = -(true shift) -> negate to match ecc_align (+shift)
    return float(-(px - m)), float(-(py - m)), corr


def selftest_sign(ref_u8):
    """Lock the sign convention: each estimator must recover a known shift."""
    H, W = ref_u8.shape
    gt_tx, gt_ty = 1.7, -1.3
    M = np.float32([[1, 0, gt_tx], [0, 1, gt_ty]])
    mov_u8 = cv2.warpAffine(ref_u8, M, (W, H), flags=cv2.INTER_CUBIC,
                            borderMode=cv2.BORDER_REFLECT)
    for name, fn, a, b in [("ecc", ecc_align, ref_u8, mov_u8),
                           ("sg-ncc", sg_ncc_align,
                            ref_u8.astype(np.float32), mov_u8.astype(np.float32))]:
        r = fn(a, b)
        if r is None:
            raise RuntimeError(f"sign self-test: {name} returned None")
        tx, ty = r[0], r[1]
        print(f"  self-test  {name}=({tx:+.3f},{ty:+.3f})  (expect ~(+1.7,-1.3))")
        if np.sign(tx) != np.sign(gt_tx) or np.sign(ty) != np.sign(gt_ty):
            raise RuntimeError(f"sign self-test FAILED: {name} sign wrong")
        if abs(tx - gt_tx) > 0.5 or abs(ty - gt_ty) > 0.5:
            raise RuntimeError(f"sign self-test FAILED: {name} off truth >0.5px")


# tag: (estimator, use_float_input, min_corr)
# min_corr high enough to drop cell-bearing channels (low corr vs the cell-free
# grid background) and average only cell-free channels -> removes the
# glucose-dependent cell-channel ECC bias. ECC-uint8's corr scale runs ~0.01
# lower than float here (quantization + grid(0,0) far reference), so it needs
# 0.98 to select the same cell-free set [1,2,4,6,7,8] that 0.99 selects for
# float/SG. Override with --min-corr.
METHODS = {
    "ecc":       (ecc_align,    False, 0.98),
    "sg":        (sg_ncc_align, True,  0.99),
    "ecc_float": (ecc_align,    True,  0.99),
}


# ==========================================================================
# Per-frame shift computation (in-process; reuses cps aggregation verbatim)
# ==========================================================================

def build_ref_crops(grid_img, rois, cfg, fit_right, use_float):
    """Per-channel grid(0,0) tilt-corrected reference crops (None if OOB)."""
    out = []
    for roi in rois:
        g = tilt_fit_crop(grid_img, roi["cy"], roi["cx"], roi["crop_w"],
                          cfg["ecc_crop_h"], cfg["tilt_crop_h"], fit_right=fit_right)
        if g is None:
            out.append(None)
        else:
            out.append(g.astype(np.float32) if use_float
                       else to_uint8(g, cfg["ecc_vmin"], cfg["ecc_vmax"]))
    return out


def compute_shifts(tag, cfg, rois, ref_crops, phase_paths, fit_right, min_corr_override=None):
    """Return frame_results list (cps schema) for the given estimator."""
    import compute_pos_shifts as cps
    cps.OUTLIER_MAD_THRESH = 5.0  # same as production aggregation
    estimator, use_float, min_corr = METHODS[tag]
    if min_corr_override is not None:
        min_corr = min_corr_override

    def proc(t):
        img = tifffile.imread(str(phase_paths[t])).astype(np.float64)
        per_channel = []
        for ch, roi in enumerate(rois):
            ref = ref_crops[ch]
            rec = {"channel": ch, "shift_x": 0.0, "shift_y": 0.0,
                   "corr": 0.0, "excluded": True, "exclude_reason": "oob"}
            if ref is not None:
                c = tilt_fit_crop(img, roi["cy"], roi["cx"], roi["crop_w"],
                                  cfg["ecc_crop_h"], cfg["tilt_crop_h"], fit_right=fit_right)
                if c is not None:
                    mov = c.astype(np.float32) if use_float else to_uint8(
                        c, cfg["ecc_vmin"], cfg["ecc_vmax"])
                    r = estimator(ref, mov)
                    if r is not None:
                        tx, ty, corr = r
                        excl = corr < min_corr
                        rec = {"channel": ch, "shift_x": tx, "shift_y": ty,
                               "corr": corr, "excluded": excl,
                               "exclude_reason": "low_corr" if excl else None}
            per_channel.append(rec)
        fr, _, _ = cps._frame_result_from_per_channel(t, per_channel)
        return t, fr

    results = [None] * len(phase_paths)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for t, fr in ex.map(proc, range(len(phase_paths))):
            results[t] = fr
    return results


# ==========================================================================
# grid_subtract driver (mirror batch_pipeline_all_pos step3)
# ==========================================================================

def run_grid_subtract(gs, cfg, pos, tl_pos_dir, rois_json, shifts_json, out_dir, grid_z, grid_cal):
    grid_dir = cfg["grid_dir"]
    crop = cfg["crop_before"] if pos < cfg["pos_split"] else cfg["crop_after"]
    gs.TIMELAPSE_DIR = str(tl_pos_dir)
    gs.SHIFTS_JSON = str(shifts_json)
    gs.CHANNEL_ROIS_JSON = str(rois_json)
    gs.GRID_DIR = grid_dir
    gs.BASE_LABEL = f"Pos{pos}"
    gs.GRID_CALIBRATION_JSON = str(grid_cal)
    gs.OUTPUT_DIR = str(out_dir)
    gs.TL_Z_INDEX = 0
    gs.GRID_Z_INDEX = grid_z
    gs.MAX_FRAMES = None
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
    gs.RAW_GRID_Z_INDEX = grid_z
    gs.TILT_CROP_H_RAW = cfg["tilt_crop_h_raw"]
    gs.SHIFT_SIGN_X = -1
    gs.SHIFT_SIGN_Y = -1
    gs.X_STEP = cfg["crop_sub_x_step_um"]
    gs.Y_STEP = cfg["crop_sub_y_step_um"]
    gs.main()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--zsub", default="z000")
    p.add_argument("--n-frames", type=int, default=300)
    p.add_argument("--methods", nargs="+", default=["ecc", "sg", "ecc_float"],
                   choices=list(METHODS))
    p.add_argument("--grid-z", type=int, default=None,
                   help="grid z plane index to subtract against. Default: config "
                        "raw_grid_z_index. NOTE production correct_0pergluc used 8 "
                        "for 260517 (drift_config's 5 is the wrong plane).")
    p.add_argument("--grid-cal", default=None,
                   help="grid_calibration json override (default: production "
                        "grid_dir/grid_calibration_Pos{N}.json). Use an isolated "
                        "recalibration here to avoid touching production.")
    p.add_argument("--per-arm-cal", action="store_true",
                   help="each arm uses the calibration made with its OWN estimator: "
                        "<out_root>/../grid_calibration_Pos{N}_z{Z}_{tag}.json "
                        "(generate via bench_recalibrate.py). Overrides --grid-cal.")
    p.add_argument("--out-suffix", default="",
                   help="suffix appended to crop_sub_<tag> output dirs, to keep "
                        "results from different calibrations side by side.")
    p.add_argument("--min-corr", type=float, default=None,
                   help="override per-arm ECC/NCC corr threshold for all arms. "
                        "High (0.98-0.99) drops cell channels -> cell-free average.")
    p.add_argument("--out-root", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pos = args.pos
    grid_dir = Path(cfg["grid_dir"])
    fit_right = pos >= cfg["pos_split"]

    grid_z = args.grid_z if args.grid_z is not None else cfg["raw_grid_z_index"]
    grid_cal = (Path(args.grid_cal) if args.grid_cal
                else grid_dir / f"grid_calibration_Pos{pos}.json")

    tl_pos_dir = Path(cfg["save_dir"]) / f"Pos{pos}" / args.zsub
    phase_dir = tl_pos_dir / "output_phase"
    phase_paths = sorted(phase_dir.glob("img_*_ph_000_phase.tif"))[:args.n_frames]
    if not phase_paths:
        raise FileNotFoundError(f"no timelapse phase frames in {phase_dir}")

    # channel_rois.json from the grid (per project convention).
    grid_rois = grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
    rois = json.loads(grid_rois.read_text(encoding="utf-8"))

    # grid(0,0) reference image at the z plane that matches the timelapse focus.
    grid_ref_path = (grid_dir / f"Pos{pos}_x+0_y+0" / "output_phase"
                     / f"img_000000000_ph_{grid_z:03d}_phase.tif")
    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)

    # Isolated output root: nothing written to production or grid dirs.
    out_root = (Path(args.out_root) if args.out_root
                else Path(cfg["save_dir"]).parent / "ecc_sg_ab") / f"Pos{pos}"
    iso_channels = out_root / "channels"
    iso_channels.mkdir(parents=True, exist_ok=True)
    rois_json = iso_channels / "channel_rois.json"
    shutil.copyfile(grid_rois, rois_json)

    print(f"Timelapse: {tl_pos_dir}  ({len(phase_paths)} frames)")
    print(f"Grid ref:  {grid_ref_path.name}  (grid z index {grid_z})")
    print(f"Channels:  {len(rois)}   fit_right={fit_right}")
    print(f"Grid cal:  {grid_cal}")
    print(f"Output (isolated): {out_root}  suffix='{args.out_suffix}'")
    print(f"Methods:   {args.methods}\n")

    print("Sign self-test:")
    crop0 = tilt_fit_crop(grid_img, rois[5]["cy"], rois[5]["cx"], rois[5]["crop_w"],
                          cfg["ecc_crop_h"], cfg["tilt_crop_h"], fit_right=fit_right)
    selftest_sign(to_uint8(crop0, cfg["ecc_vmin"], cfg["ecc_vmax"]))

    import grid_subtract as gs
    outputs = {}
    for tag in args.methods:
        _, use_float, _ = METHODS[tag]
        print(f"\n=== [{tag}] computing shifts ({len(phase_paths)} frames) ===")
        ref_crops = build_ref_crops(grid_img, rois, cfg, fit_right, use_float)
        frame_results = compute_shifts(tag, cfg, rois, ref_crops, phase_paths, fit_right,
                                       min_corr_override=args.min_corr)
        n_ok = sum(1 for fr in frame_results if fr and fr["shift_x_avg"] is not None)
        nu = [fr["n_channels_used"] for fr in frame_results[2:] if fr and fr["shift_x_avg"] is not None]
        print(f"  channels used (cell-free): median {int(np.median(nu)) if nu else 0}/{len(rois)}")
        sx = [fr["shift_x_avg"] for fr in frame_results[:4] if fr]
        print(f"  shifts ok: {n_ok}/{len(frame_results)}   shift_x[:4]={[round(v,3) if v is not None else None for v in sx]}")

        shifts_json = iso_channels / f"pos_shifts_{tag}.json"
        shifts_json.write_text(json.dumps(
            {"n_frames": len(frame_results), "frame_results": frame_results},
            ensure_ascii=False), encoding="utf-8")

        if args.per_arm_cal:
            grid_cal_tag = out_root.parent / f"grid_calibration_Pos{pos}_z{grid_z}_{tag}.json"
            if not grid_cal_tag.exists():
                raise FileNotFoundError(
                    f"per-arm calibration missing: {grid_cal_tag}\n"
                    f"  run: python scripts/bench_recalibrate.py --estimator {tag} --grid-z {grid_z}")
        else:
            grid_cal_tag = grid_cal

        out_dir = out_root / f"crop_sub_{tag}{args.out_suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== [{tag}] grid_subtract -> {out_dir.name}  (cal={Path(grid_cal_tag).name}) ===")
        run_grid_subtract(gs, cfg, pos, tl_pos_dir, rois_json, shifts_json, out_dir, grid_z, grid_cal_tag)
        outputs[tag] = out_dir

    print("\n==== DONE ====")
    for tag, out in outputs.items():
        print(f"  {tag:10s} -> {out}")
    print("\nCompare ch00/.. TIFs between the crop_sub_* folders by eye.")


if __name__ == "__main__":
    main()
