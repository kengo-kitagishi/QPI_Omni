"""
batch_grid_subtract_260517.py
-----------------------------
Run grid_subtract.py (raw-raw) for ALL Pos of the 260517 main timelapse,
OFFLINE, using the freshly recomputed (float-ECC, z=8) pos_shifts_cal.json and
grid_calibration_PosN.json. NO re-reconstruction (uses existing
PosN/z000/output_phase_raw/*_phase.tif and the grid's output_phase_raw).

Mirrors batch_pipeline_all_pos.py step3 params, but:
  - z000/ nesting: timelapse inputs live under PosN/z000/
  - output goes to the separate _crop_sub tree (matches downstream correct /
    apply_oob_mask configs and the F: copy target), under
    .../crop_sub_rawraw/z000/chNN
  - GRID_Z_INDEX / RAW_GRID_Z_INDEX = 8
  - per-Pos: wipe an existing crop_sub_rawraw/z000 first (clean regen; the new
    per-Pos channel count can differ from the old online output)

grid_subtract zeros OOB internally (valid_out > 0.999 + VALID_ERODE_PX=1).
"""
import sys
import json
import shutil
import re
import time
import argparse
import subprocess
from pathlib import Path

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))

# ============================================================
TL_ROOT       = Path(r"E:\260517\2per_0055per_0per_2per")            # timelapse (z000)
CROP_SUB_ROOT = Path(r"E:\260517\2per_0055per_0per_2per_crop_sub")   # output tree
GRID_2PER_DIR = Path(r"E:\260517\grid_2pergluc_2")
POS_START     = 1
POS_END       = 104

# Pos-level parallelism: grid_subtract is I/O/GIL-bound (CPU ~14% solo), so run
# several Pos concurrently (each its own process) to overlap save-I/O + CPU.
K_CONCURRENT    = 6
THREADS_PER_POS = 6
LOG_DIR = _SD.parent / "drift_session" / "grid_subtract_logs"

GRID_Z_INDEX     = 8
RAW_GRID_Z_INDEX = 8
RAW_TL_Z_INDEX   = 0
TILT_CROP_H_RAW  = 270
POS_SPLIT        = 53
SHIFT_SIGN_X     = -1
SHIFT_SIGN_Y     = -1
CROP_BEFORE = (0, 2048, 400, 2448)   # Pos < POS_SPLIT
CROP_AFTER  = (0, 2048, 0, 2048)     # Pos >= POS_SPLIT
# ============================================================


def run_pos(pos_num):
    pos_dir = TL_ROOT / f"Pos{pos_num}" / "z000"
    channels_tl = pos_dir / "output_phase" / "channels"
    raw_phase_dir = pos_dir / "output_phase_raw"
    shifts_json = channels_tl / "pos_shifts_cal.json"
    rois_json = channels_tl / "channel_rois.json"
    cal_json = GRID_2PER_DIR / f"grid_calibration_Pos{pos_num}.json"
    out_dir = (CROP_SUB_ROOT / f"Pos{pos_num}" / "output_phase"
               / "channels" / "crop_sub_rawraw" / "z000")

    for p, nm in [(raw_phase_dir, "output_phase_raw"), (shifts_json, "pos_shifts_cal"),
                  (rois_json, "channel_rois"), (cal_json, "grid_calibration")]:
        if not p.exists():
            print(f"  [ERROR] Pos{pos_num}: {nm} missing: {p}")
            return "err_missing"

    n_frames = len(list(raw_phase_dir.glob("img_*_ph_000_phase.tif")))

    # Resumable: skip only if THIS regen already finished (marker), NOT if stale
    # online crop_sub frames happen to exist. Old online data has no marker ->
    # it gets wiped + regenerated with the new float-ECC / z=8 shifts.
    marker = out_dir / ".regen_z8_floatecc_done"
    if marker.exists():
        return "skip_done"

    # Clean regen of this Pos's crop_sub_rawraw/z000 (overwrites stale online data)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = CROP_BEFORE if pos_num < POS_SPLIT else CROP_AFTER

    import grid_subtract as gs
    gs.N_PARALLEL_FRAMES = THREADS_PER_POS   # per-Pos threads (K Pos run concurrently)
    gs.TIMELAPSE_DIR = str(pos_dir)
    gs.SHIFTS_JSON = str(shifts_json)
    gs.CHANNEL_ROIS_JSON = str(rois_json)
    gs.GRID_DIR = str(GRID_2PER_DIR)
    gs.BASE_LABEL = f"Pos{pos_num}"
    gs.GRID_CALIBRATION_JSON = str(cal_json)
    gs.OUTPUT_DIR = str(out_dir)

    gs.TL_Z_INDEX = 0
    gs.GRID_Z_INDEX = GRID_Z_INDEX
    gs.MAX_FRAMES = None
    gs.PICK_FRAMES = None
    gs.APPLY_SUBPIXEL_CORRECTION = True
    gs.APPLY_INVERSE_SHIFT = False
    gs.OUTPUT_CROP_H = TILT_CROP_H_RAW
    gs.OUTPUT_SAVE_FULL_FRAME = False

    gs.USE_RAW_PHASE = True
    gs.TL_PHASE_DIR = str(raw_phase_dir)
    gs.RAW_CROP = crop
    gs.RAW_TL_Z_INDEX = RAW_TL_Z_INDEX
    gs.RAW_GRID_Z_INDEX = RAW_GRID_Z_INDEX
    gs.TILT_CROP_H_RAW = TILT_CROP_H_RAW
    gs.SHIFT_SIGN_X = SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y = SHIFT_SIGN_Y
    gs.X_STEP = 0.1
    gs.Y_STEP = 0.1
    gs.POS_SPLIT = POS_SPLIT

    print(f"  Pos{pos_num}: grid_subtract ({n_frames} frames, crop={crop})...", flush=True)
    gs.main()
    (out_dir / ".regen_z8_floatecc_done").write_text("done")
    return "ok"


def _needs_run(n):
    out_dir = (CROP_SUB_ROOT / f"Pos{n}" / "output_phase"
               / "channels" / "crop_sub_rawraw" / "z000")
    raw = TL_ROOT / f"Pos{n}" / "z000" / "output_phase_raw"
    if not raw.is_dir():
        return False
    return not (out_dir / ".regen_z8_floatecc_done").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=None)
    args = ap.parse_args()

    if args.pos is not None:
        try:
            r = run_pos(args.pos)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = f"EXC:{e}"
        print(f"[single] Pos{args.pos} -> {r}", flush=True)
        sys.exit(0 if str(r).startswith(("ok", "skip")) else 1)

    # Orchestrator: K Pos concurrent (subprocess each), marker-skip, 3-pass retry
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def run_batch(todo):
        running = {}
        idx = 0
        ok = fail = 0
        while idx < len(todo) or running:
            while idx < len(todo) and len(running) < K_CONCURRENT:
                n = todo[idx]; idx += 1
                lf = open(LOG_DIR / f"Pos{n}.log", "w")
                p = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--pos", str(n)],
                    stdout=lf, stderr=subprocess.STDOUT)
                running[p] = (n, time.time(), lf)
            time.sleep(3)
            for p in list(running):
                rc = p.poll()
                if rc is None:
                    continue
                n, st, lf = running.pop(p); lf.close()
                ok, fail = (ok + 1, fail) if rc == 0 else (ok, fail + 1)
                print(f"  Pos{n} rc={rc} ({(time.time()-st)/60:.1f} min) "
                      f"[ok{ok} fail{fail} run{len(running)}]", flush=True)
        return ok, fail

    for pass_i in range(1, 4):
        todo = [n for n in range(POS_START, POS_END + 1) if _needs_run(n)]
        if not todo:
            print("All Pos complete.", flush=True); break
        print(f"\n=== PASS {pass_i}: {len(todo)} Pos (K={K_CONCURRENT}, "
              f"threads/Pos={THREADS_PER_POS}) ===\n{todo}", flush=True)
        run_batch(todo)

    remaining = [n for n in range(POS_START, POS_END + 1) if _needs_run(n)]
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min. "
          f"incomplete={remaining if remaining else 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
