"""
batch_compute_pos_shifts_260908_whitelist.py
---------------------------------------------
Re-run compute_pos_shifts.py for 260908, OFFLINE, restricted to the whitelist
channels only -- this is the fix for the cross-channel mass co-fluctuation
bug found in the online production average (r=+0.260 between unrelated
channels of the same Pos, 2026-10-13).

Scope, both by explicit user decision (2026-10-13):
  - Pos NOT present in channel_whitelist_260908.json are skipped entirely
    (no clean reference channel exists there -> do not re-analyse, rather
    than silently falling back to the old whole-channel average).
  - within a covered Pos, ONLY the whitelisted channels are run through ECC.
    The Pos-level correction (shift_x_avg / shift_y_avg) is a single value
    applied to every channel's crop regardless of which channels produced
    it, so channels never used for averaging don't need their own ECC here.
  - 0-100h analysis window only: frame index 0-1202 (t=(frame-2)/12, so this
    covers the 2-1202 range the lineage review uses, plus 2 harmless warmup
    frames before t=0). MAX_FRAMES=1203 truncates to this on purpose -- see
    project_compute_pos_shifts_params.md before assuming this should be None.

Result: 138 channels across 58 Pos instead of 1153 across 98 (12%), on top of
restricting to <=100h instead of the full 1260-frame run.

Reads the FULL-FOV (uncropped) phase reconstruction at ph_zstack_2, which is
unaffected by whatever bug is in the online per-channel crop_sub -- this is a
genuine from-scratch realignment, not a re-average of the old shifts.

Output: a new "channels" subfolder next to the existing ph_zstack_2 phase tifs
(compute_pos_shifts.py's slope-correction mode requires CHANNELS_DIR's parent
to BE the directory holding img_*_ph_000_phase.tif -- mirrors how
batch_compute_pos_shifts_260517.py is laid out). Nothing existing is touched,
this subfolder does not exist today:
  D:/AquisitionData/Kitagishi/260908/ph_zstack_2/PosN/z000/output_phase/channels/
      channel_rois.json          (copied, unchanged, from the online tree)
      pos_shifts_cal_whitelist.json
"""
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))

from ecc_utils import ECC_MIN_CORR  # single source (0.99)

# ============================================================
TL_PHASE_ROOT   = Path(r"D:\AquisitionData\Kitagishi\260908\ph_zstack_2")        # full-FOV phase, uncropped
ONLINE_CH_ROOT  = Path(r"D:\AquisitionData\Kitagishi\260908\online_crop_sub_zstack_2")  # source of channel_rois.json
WHITELIST_JSON  = Path(r"D:\AquisitionData\Kitagishi\260908\_qc\channel_whitelist_260908.json")

GRID_DIR      = Path(r"E:\260908\ye_grid_0p05_2")
GRID_Z_INDEX  = 5          # pos_shifts_cal_online.json: grid_z_index=5 (this dataset's plane)
POS_SPLIT     = 52         # tilt-fit direction boundary (fit_right = pos>=POS_SPLIT);
                            # matches datasets/260908_outside_quad.yaml pos_split and
                            # _qc_outsidech_quadfit_260908.py's own default. NOT the same
                            # POS_SPLIT as the raw-hologram crop_before/after boundary
                            # (watch_grid_then_recon_260908.py uses 51 for THAT, a
                            # different concept -- commit 6318908's "51 across all
                            # scripts" bumped compute_pos_shifts.py's unrelated module
                            # default, not this dataset's tilt-fit value). Only Pos51
                            # itself differs between 51/52 (fit_right=pos>=split); it is
                            # outside this run's 58-Pos whitelist coverage either way.
TILT_CROP_H   = 270        # pos_shifts_cal_online.json: tilt_crop_h_raw=270
ECC_CROP_H    = 80         # project_compute_pos_shifts_params.md
OUTLIER_MAD_THRESH = 5.0   # project_compute_pos_shifts_params.md (2026-04-09: 2.5->5.0)
OUTLIER_TS_THRESH  = 0.0   # project_compute_pos_shifts_params.md (always off)
VMIN, VMAX    = -5.0, 2.0
N_WORKERS_ECC = 24
MAX_FRAMES    = 1203       # frame index 0-1202 = the 0-100h analysis window (see module docstring)
SHIFT_SIGN_X, SHIFT_SIGN_Y = 1, 1   # pos_shifts_cal_online.json
OUTPUT_JSON   = "pos_shifts_cal_whitelist.json"


def load_whitelist():
    d = json.loads(WHITELIST_JSON.read_text(encoding="utf-8"))
    return d["whitelist"]  # {"PosN": [ch, ...]}


def run_pos(pos_label, whitelist_chs):
    pos_num = int(pos_label[3:])
    tl_dir = TL_PHASE_ROOT / pos_label / "z000" / "output_phase"
    if not tl_dir.is_dir():
        print(f"  [SKIP] {pos_label}: no phase dir at {tl_dir}")
        return "skip_missing"

    n_frames_avail = len(list(tl_dir.glob("img_*_ph_000_phase.tif")))
    out_dir = tl_dir / "channels"
    out_json = out_dir / OUTPUT_JSON
    n_target = min(MAX_FRAMES, n_frames_avail)
    if out_json.exists():
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            if d.get("n_frames", 0) >= n_target:
                print(f"  [SKIP] {pos_label}: {OUTPUT_JSON} already covers "
                      f"{d['n_frames']} frames")
                return "skip_done"
        except Exception:
            pass

    src_rois = ONLINE_CH_ROOT / pos_label / "output_phase" / "channels" / "channel_rois.json"
    if not src_rois.exists():
        print(f"  [ERROR] {pos_label}: channel_rois.json missing: {src_rois}")
        return "err_no_rois"
    cal_json = GRID_DIR / f"grid_calibration_{pos_label}.json"
    if not cal_json.exists():
        print(f"  [ERROR] {pos_label}: grid_calibration missing: {cal_json}")
        return "err_no_cal"

    out_dir.mkdir(parents=True, exist_ok=True)
    dst_rois = out_dir / "channel_rois.json"
    if not dst_rois.exists():
        shutil.copy2(src_rois, dst_rois)

    import compute_pos_shifts as cps
    cps.CHANNELS_DIR = str(out_dir)
    cps.CHANNEL_ROIS_JSON = str(dst_rois)
    cps.GRID_DIR = str(GRID_DIR)
    cps.GRID_BASE_LABEL = pos_label
    cps.POS_SPLIT = POS_SPLIT
    cps.GRID_Z_INDEX = GRID_Z_INDEX
    cps.GRID_CALIBRATION_JSON = str(cal_json)
    cps.ECC_CROP_H = ECC_CROP_H
    cps.TILT_CROP_H = TILT_CROP_H
    cps.USE_SLOPE_CORRECTION = True
    cps.USE_GRID_REFERENCE = True
    cps.USE_INCREMENTAL_TRACKING = True
    cps.USE_SECOND_PASS_ECC = True
    cps.USE_THIRD_PASS_ECC = True
    cps.OUTLIER_MAD_THRESH = OUTLIER_MAD_THRESH
    cps.OUTLIER_TIMESERIES_THRESH = OUTLIER_TS_THRESH
    cps.ECC_MIN_CORR = ECC_MIN_CORR
    cps.VMIN = VMIN
    cps.VMAX = VMAX
    cps.MAX_FRAMES = MAX_FRAMES
    cps.N_WORKERS = N_WORKERS_ECC
    cps.TL_Z_INDEX = 0
    cps.SHIFT_SIGN_X = SHIFT_SIGN_X
    cps.SHIFT_SIGN_Y = SHIFT_SIGN_Y
    cps.OUTPUT_JSON = OUTPUT_JSON
    cps.SAVE_CORR_DATA = True
    cps.SAVE_SHIFT_FIGURES = True
    cps.SHIFT_FIGURES_FINE_ONLY = True
    cps.APPLY_BACKSUB_TO_GRID_REF = True
    cps.TILT_FIT_RIGHT = pos_num >= POS_SPLIT
    cps.CHANNEL_WHITELIST = list(whitelist_chs)

    print(f"  {pos_label}: compute_pos_shifts, whitelist ch={sorted(whitelist_chs)}, "
          f"{n_target} frames (0-100h window)...", flush=True)
    cps.main()
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=str, default=None,
                    help="single-Pos worker mode (fresh process; releases memory on exit)")
    args = ap.parse_args()

    whitelist = load_whitelist()

    if args.pos is not None:
        try:
            r = run_pos(args.pos, whitelist[args.pos])
        except Exception as e:
            r = f"EXC:{e}"
            print(f"  [EXC] {args.pos}: {e}", flush=True)
        print(f"[single] {args.pos} -> {r}", flush=True)
        sys.exit(0 if str(r).startswith(("ok", "skip")) else 1)

    pos_labels = sorted(whitelist.keys(), key=lambda p: int(p[3:]))
    print(f"compute_pos_shifts batch (260908, whitelist-only, 0-100h): "
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
