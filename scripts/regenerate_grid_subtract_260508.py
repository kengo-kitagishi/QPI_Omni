"""
regenerate_grid_subtract_260508.py
-----------------------------------
Re-run grid_subtract.py for all Pos in 260508 online_crop_sub_zstack
to restore crop_sub_rawraw to the state BEFORE correct_0pergluc was applied.

Uses:
  - Raw phase from: D:\AquisitionData\Kitagishi\260508\ph_single\PosN\output_phase_raw
  - Shifts from:    online_crop_sub_zstack\PosN\...\pos_shifts_cal_online.json
  - Channel ROIs:   online_crop_sub_zstack\PosN\...\channel_rois.json
  - Grid data:      E:\260504\grid_2pergluc_1
  - Output to:      online_crop_sub_zstack\PosN\...\crop_sub_rawraw  (overwrite)

Usage:
    python scripts/regenerate_grid_subtract_260508.py
    python scripts/regenerate_grid_subtract_260508.py --pos 1 5 10   # specific Pos only
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ============================================================
# Paths
# ============================================================
PH_SINGLE_ROOT = Path(r"D:\AquisitionData\Kitagishi\260508\ph_single")
ONLINE_ROOT    = Path(r"D:\AquisitionData\Kitagishi\260508\online_crop_sub_zstack")
GRID_2PER_DIR  = Path(r"E:\260504\grid_2pergluc_1")

POS_START = 1
POS_END   = 104

# ============================================================
# Parameters (from pos_shifts_cal_online.json / memory)
# ============================================================
POS_SPLIT        = 53
CROP_BEFORE      = (0, 2048, 400, 2448)
CROP_AFTER       = (0, 2048, 0, 2048)

RAW_GRID_Z_INDEX = 5     # z=0 is index 5 for 260504 grid (11-slice, -2.0 to +2.0 um)
RAW_TL_Z_INDEX   = 0     # ph_single is single-z
TILT_CROP_H_RAW  = 270
SHIFT_SIGN_X     = -1
SHIFT_SIGN_Y     = -1


def get_crop(pos_num):
    return CROP_BEFORE if pos_num < POS_SPLIT else CROP_AFTER


def clean_0per_artifacts(crop_sub_dir: Path):
    """Remove correct_0pergluc outputs from crop_sub_rawraw."""
    removed = []
    for name in ["correct_0pergluc_log.json", "delta_full.tif"]:
        p = crop_sub_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
    delta_dir = crop_sub_dir / "delta_per_ch"
    if delta_dir.exists():
        shutil.rmtree(delta_dir)
        removed.append("delta_per_ch/")
    if removed:
        print(f"  [clean] removed: {', '.join(removed)}")


def run_grid_subtract_for_pos(pos_num: int):
    """Run grid_subtract for a single Pos, overwriting crop_sub_rawraw."""
    import importlib
    import grid_subtract as gs
    importlib.reload(gs)

    pos_label = f"Pos{pos_num}"
    ph_single_pos = PH_SINGLE_ROOT / pos_label
    online_pos = ONLINE_ROOT / pos_label
    channels_dir = online_pos / "output_phase" / "channels"
    crop_sub_dir = channels_dir / "crop_sub_rawraw"

    # Validate inputs
    raw_phase_dir = ph_single_pos / "output_phase_raw"
    if not raw_phase_dir.exists():
        print(f"  [ERROR] output_phase_raw not found: {raw_phase_dir}")
        return False

    shifts_json = channels_dir / "pos_shifts_cal_online.json"
    rois_json = channels_dir / "channel_rois.json"
    for p, name in [(shifts_json, "shifts"), (rois_json, "channel_rois")]:
        if not p.exists():
            print(f"  [ERROR] {name} not found: {p}")
            return False

    grid_cal_json = GRID_2PER_DIR / f"grid_calibration_{pos_label}.json"
    if not grid_cal_json.exists():
        print(f"  [ERROR] grid_calibration not found: {grid_cal_json}")
        return False

    crop = get_crop(pos_num)

    # Clean correct_0pergluc artifacts AND old TIFs
    if crop_sub_dir.exists():
        clean_0per_artifacts(crop_sub_dir)
        # Delete all existing TIFs in ch dirs (old naming convention)
        n_rois = json.loads(rois_json.read_text(encoding="utf-8"))
        for ch in range(len(n_rois)):
            ch_dir = crop_sub_dir / f"ch{ch:02d}"
            if ch_dir.exists():
                old_tifs = list(ch_dir.glob("*.tif"))
                for f in old_tifs:
                    f.unlink()
                if old_tifs:
                    print(f"  [clean] ch{ch:02d}: deleted {len(old_tifs)} old TIFs")

    # Configure grid_subtract
    gs.TIMELAPSE_DIR = str(ph_single_pos)
    gs.SHIFTS_JSON = str(shifts_json)
    gs.CHANNEL_ROIS_JSON = str(rois_json)
    gs.GRID_DIR = str(GRID_2PER_DIR)
    gs.BASE_LABEL = pos_label
    gs.GRID_CALIBRATION_JSON = str(grid_cal_json)
    gs.OUTPUT_DIR = str(crop_sub_dir)

    gs.TL_Z_INDEX = 0
    gs.GRID_Z_INDEX = RAW_GRID_Z_INDEX
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
    gs.POS_SPLIT = POS_SPLIT

    gs.SHIFT_SIGN_X = SHIFT_SIGN_X
    gs.SHIFT_SIGN_Y = SHIFT_SIGN_Y
    gs.X_STEP = 0.1
    gs.Y_STEP = 0.1

    print(f"  Running grid_subtract (raw-raw, crop={crop})...")
    try:
        gs.main()
    except Exception as e:
        print(f"  [ERROR] grid_subtract: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Rename _phase.tif -> .tif to match online pipeline naming convention
    n_renamed = 0
    for ch_dir in sorted(crop_sub_dir.glob("ch*")):
        if not ch_dir.is_dir():
            continue
        for f in ch_dir.glob("*_phase.tif"):
            new_name = f.name.replace("_phase.tif", ".tif")
            f.rename(ch_dir / new_name)
            n_renamed += 1
    if n_renamed > 0:
        print(f"  [rename] {n_renamed} files: *_phase.tif -> *.tif")
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Re-run grid_subtract for 260508 to undo correct_0pergluc"
    )
    ap.add_argument(
        "--pos", type=int, nargs="+", default=None,
        help="Specific Pos numbers to process (default: all 1-104)",
    )
    args = ap.parse_args()

    pos_list = args.pos if args.pos else list(range(POS_START, POS_END + 1))
    total = len(pos_list)

    print("=" * 60)
    print(f"Regenerate grid_subtract: {total} Pos")
    print(f"  ph_single:  {PH_SINGLE_ROOT}")
    print(f"  online:     {ONLINE_ROOT}")
    print(f"  grid 2per:  {GRID_2PER_DIR}")
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

        if run_grid_subtract_for_pos(pos_num):
            success += 1
        else:
            fail += 1

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Done. success={success}  fail={fail}  elapsed={elapsed / 60:.1f}min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
