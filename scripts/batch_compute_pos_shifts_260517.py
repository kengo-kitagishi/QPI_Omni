"""
batch_compute_pos_shifts_260517.py
----------------------------------
Run compute_pos_shifts.py for ALL Pos of the 260517 main timelapse, OFFLINE,
using the existing pre-reconstructed phase (NO re-reconstruction; raw holograms
are already cleaned up).

Mirrors batch_pipeline_all_pos.py step2 params (float-ECC pipeline), but:
  - skips reconstruction (uses existing PosN/z000/output_phase/*_phase.tif)
  - handles the z000/ nesting of this z-stack online dataset
  - GRID_Z_INDEX = 8 (this dataset's plane = +1.2 um), with the freshly
    recomputed (float-ECC, z=8) grid_calibration_PosN.json
  - copies channel_rois.json from the grid into the timelapse channels dir
  - resumable: skips a Pos whose pos_shifts_cal.json already covers all frames

Output: <TL_ROOT>/PosN/z000/output_phase/channels/pos_shifts_cal.json
"""
import sys
import json
import shutil
import re
import argparse
import subprocess
from pathlib import Path

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))

from ecc_utils import ECC_MIN_CORR  # single source (0.99)

# ============================================================
TL_ROOT       = Path(r"E:\260517\2per_0055per_0per_2per")   # timelapse (z000 nesting)
GRID_2PER_DIR = Path(r"E:\260517\grid_2pergluc_2")
POS_START     = 1
POS_END       = 104

GRID_Z_INDEX  = 8       # this dataset's plane (+1.2 um); matches new calibration
POS_SPLIT     = 53
ECC_CROP_H    = 80
TILT_CROP_H   = 270
# ECC_MIN_CORR imported from ecc_utils above (single source = 0.99)
OUTLIER_MAD_THRESH = 5.0
OUTLIER_TS_THRESH  = 0.0
VMIN, VMAX    = -5.0, 2.0
N_WORKERS_ECC = 24
# ============================================================


def run_pos(pos_num):
    pos_dir = TL_ROOT / f"Pos{pos_num}" / "z000"
    channels_dir = pos_dir / "output_phase" / "channels"
    op_dir = pos_dir / "output_phase"
    if not op_dir.is_dir():
        print(f"  [SKIP] Pos{pos_num}: no output_phase at {op_dir}")
        return "skip_missing"

    n_frames = len(list(op_dir.glob("img_*_ph_000_phase.tif")))
    out_json = channels_dir / "pos_shifts_cal.json"
    if out_json.exists():
        try:
            d = json.loads(out_json.read_text(encoding="utf-8"))
            if d.get("n_frames", 0) >= n_frames:
                print(f"  [SKIP] Pos{pos_num}: pos_shifts_cal.json covers "
                      f"{d['n_frames']} frames")
                return "skip_done"
        except Exception:
            pass

    cal_json = GRID_2PER_DIR / f"grid_calibration_Pos{pos_num}.json"
    if not cal_json.exists():
        print(f"  [ERROR] Pos{pos_num}: grid_calibration missing: {cal_json}")
        return "err_no_cal"

    # Copy per-Pos channel_rois.json from the grid center point (batch step1 equiv)
    channels_dir.mkdir(parents=True, exist_ok=True)
    grid_rois = (GRID_2PER_DIR / f"Pos{pos_num}_x+0_y+0"
                 / "output_phase" / "channels" / "channel_rois.json")
    if not grid_rois.exists():
        print(f"  [ERROR] Pos{pos_num}: grid channel_rois missing: {grid_rois}")
        return "err_no_rois"
    shutil.copy2(grid_rois, channels_dir / "channel_rois.json")

    import compute_pos_shifts as cps
    cps.CHANNELS_DIR = str(channels_dir)
    cps.CHANNEL_ROIS_JSON = str(channels_dir / "channel_rois.json")
    cps.GRID_DIR = str(GRID_2PER_DIR)
    cps.GRID_BASE_LABEL = f"Pos{pos_num}"
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
    cps.MAX_FRAMES = None
    cps.N_WORKERS = N_WORKERS_ECC
    cps.TL_Z_INDEX = 0
    cps.SHIFT_SIGN_X = 1
    cps.SHIFT_SIGN_Y = 1
    cps.OUTPUT_JSON = "pos_shifts_cal.json"
    cps.SAVE_CORR_DATA = True
    cps.SAVE_SHIFT_FIGURES = True       # save only the fine_ecc figure per Pos...
    cps.SHIFT_FIGURES_FINE_ONLY = True  # ...skip the redundant pass1/pass2/timeseries
    cps.APPLY_BACKSUB_TO_GRID_REF = True
    cps.TILT_FIT_RIGHT = pos_num >= POS_SPLIT

    print(f"  Pos{pos_num}: compute_pos_shifts ({n_frames} frames, z=8)...", flush=True)
    cps.main()
    return "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=None,
                    help="single-Pos worker mode (fresh process; releases memory on exit)")
    args = ap.parse_args()

    # --- Worker mode: run exactly one Pos in this fresh process, then exit ---
    if args.pos is not None:
        try:
            r = run_pos(args.pos)
        except Exception as e:
            r = f"EXC:{e}"
            print(f"  [EXC] Pos{args.pos}: {e}", flush=True)
        print(f"[single] Pos{args.pos} -> {r}", flush=True)
        # non-zero exit signals failure to the orchestrator
        sys.exit(0 if str(r).startswith(("ok", "skip")) else 1)

    # --- Orchestrator: one subprocess per Pos so memory is fully released ---
    print(f"compute_pos_shifts batch (offline, z=8): Pos{POS_START}-{POS_END} "
          f"[subprocess-per-Pos]", flush=True)
    counts = {}
    for n in range(POS_START, POS_END + 1):
        print(f"\n=== Pos{n}/{POS_END} (subprocess) ===", flush=True)
        rc = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                             "--pos", str(n)]).returncode
        key = "ok" if rc == 0 else f"fail(rc={rc})"
        counts[key] = counts.get(key, 0) + 1
        print(f"[{n}/{POS_END}] subprocess rc={rc}", flush=True)
    print("\nSummary:", counts, flush=True)


if __name__ == "__main__":
    main()
