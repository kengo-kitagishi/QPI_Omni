"""
batch_correct_0pergluc_260508.py
---------------------------------
Run correct_0pergluc for all Pos in 260508 online_crop_sub_zstack,
using per-Pos delta_timelapse as the 0% reference.

Optionally waits for a specified PID to finish before starting
(use --wait-pid to chain after regenerate_grid_subtract_260508.py).

Usage:
    python scripts/batch_correct_0pergluc_260508.py
    python scripts/batch_correct_0pergluc_260508.py --wait-pid 3520
    python scripts/batch_correct_0pergluc_260508.py --pos 1 5 10
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ============================================================
# Paths
# ============================================================
SESSION_ROOT   = Path(r"D:\AquisitionData\Kitagishi\260508\online_crop_sub_zstack")
GRID_2PER_DIR  = Path(r"E:\260504\grid_2pergluc_1")

POS_START = 1
POS_END   = 104

# ============================================================
# Parameters (from previous correct_0pergluc_log.json)
# ============================================================
GLUCOSE_0_START  = 870
GLUCOSE_0_END    = 1445
GRID_Z_INDEX     = 5
POS_SPLIT        = 53
CROP_BEFORE      = (0, 2048, 400, 2448)
CROP_AFTER       = (0, 2048, 0, 2048)
TILT_CROP_H_RAW  = 270
ECC_CROP_H       = 80
DELTA_TIFS_SUBDIR = "delta_timelapse"


def get_crop(pos_num):
    return CROP_BEFORE if pos_num < POS_SPLIT else CROP_AFTER


def wait_for_pid(pid, poll_interval=30):
    """Wait until a process with given PID exits."""
    import psutil
    print(f"Waiting for PID {pid} to finish...")
    while True:
        if not psutil.pid_exists(pid):
            print(f"PID {pid} has exited.")
            return
        try:
            p = psutil.Process(pid)
            if p.status() == psutil.STATUS_ZOMBIE:
                print(f"PID {pid} is zombie, treating as finished.")
                return
        except psutil.NoSuchProcess:
            print(f"PID {pid} has exited.")
            return
        time.sleep(poll_interval)


def run_correct_for_pos(pos_num: int):
    """Run correct_0pergluc for a single Pos."""
    import correct_0pergluc as c0
    importlib.reload(c0)

    pos_label = f"Pos{pos_num}"
    channels_dir = SESSION_ROOT / pos_label / "output_phase" / "channels"
    out_dir = channels_dir / "crop_sub_rawraw"
    log_file = out_dir / "correct_0pergluc_log.json"

    if log_file.exists():
        print(f"  [SKIP] already corrected")
        return True

    if not out_dir.exists():
        print(f"  [ERROR] crop_sub_rawraw not found: {out_dir}")
        return False

    glog = channels_dir / "grid_subtract_log.json"
    crois = channels_dir / "channel_rois.json"
    gcal = GRID_2PER_DIR / f"grid_calibration_{pos_label}.json"

    delta_dir = channels_dir / DELTA_TIFS_SUBDIR
    if not delta_dir.is_dir():
        print(f"  [ERROR] delta_timelapse not found: {delta_dir}")
        return False

    for p, name in [(glog, "grid_subtract_log"), (crois, "channel_rois"), (gcal, "grid_calibration")]:
        if not p.exists():
            print(f"  [ERROR] {name} not found: {p}")
            return False

    crop = get_crop(pos_num)

    c0.OUTPUT_DIR = str(out_dir)
    c0.GRID_SUB_LOG = str(glog)
    c0.CHANNEL_ROIS_JSON = str(crois)
    c0.GRID_2PER_DIR = str(GRID_2PER_DIR)
    c0.GRID_0PER_DIR = str(delta_dir)
    c0.BASE_LABEL = pos_label
    c0.GRID_CALIBRATION_JSON = str(gcal)
    c0.GRID_Z_INDEX = GRID_Z_INDEX
    c0.GLUCOSE_0_START = GLUCOSE_0_START
    c0.GLUCOSE_0_END = GLUCOSE_0_END
    c0.RAW_CROP = crop
    c0.TILT_CROP_H_RAW = TILT_CROP_H_RAW
    c0.ECC_CROP_H = ECC_CROP_H
    c0.OUTPUT_CROP_H = None
    c0.POS_NUMBERS_TO_RUN = []
    c0.POS_SPLIT = POS_SPLIT
    c0.FIT_RIGHT = None
    c0.DELTA_TIFS_DIR = str(delta_dir)
    c0.DELTA_TIFS_SUBDIR = None

    print(f"  Running correct_0pergluc (frames {GLUCOSE_0_START}-{GLUCOSE_0_END - 1})...")
    try:
        c0.run_correct_0pergluc(
            Path(c0.OUTPUT_DIR),
            Path(c0.GRID_SUB_LOG),
            Path(c0.CHANNEL_ROIS_JSON),
            c0.BASE_LABEL,
            Path(c0.GRID_CALIBRATION_JSON),
        )
        return True
    except Exception as e:
        print(f"  [ERROR] correct_0pergluc: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    ap = argparse.ArgumentParser(
        description="Run correct_0pergluc for 260508 using delta_timelapse"
    )
    ap.add_argument(
        "--pos", type=int, nargs="+", default=None,
        help="Specific Pos numbers (default: all 1-104)",
    )
    ap.add_argument(
        "--wait-pid", type=int, default=None,
        help="Wait for this PID to finish before starting",
    )
    args = ap.parse_args()

    if args.wait_pid:
        wait_for_pid(args.wait_pid)

    pos_list = args.pos if args.pos else list(range(POS_START, POS_END + 1))
    total = len(pos_list)

    print("=" * 60)
    print(f"correct_0pergluc batch: {total} Pos")
    print(f"  session:    {SESSION_ROOT}")
    print(f"  grid 2per:  {GRID_2PER_DIR}")
    print(f"  delta src:  {DELTA_TIFS_SUBDIR}")
    print(f"  0% range:   [{GLUCOSE_0_START}, {GLUCOSE_0_END})")
    print(f"  POS_SPLIT:  {POS_SPLIT}")
    print("=" * 60)

    t_start = time.time()
    success = 0
    fail = 0

    for i, pos_num in enumerate(pos_list):
        elapsed = time.time() - t_start
        print(f"\n{'=' * 60}")
        print(f"[Pos{pos_num}] ({i + 1}/{total})  elapsed={elapsed / 60:.0f}min")
        print(f"{'=' * 60}")

        if run_correct_for_pos(pos_num):
            success += 1
        else:
            fail += 1

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done. success={success}  fail={fail}  elapsed={elapsed / 60:.1f}min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
