"""
batch_grid_calibration.py
Run calibrate_grid_positions.py for all Pos labels in a grid directory.
"""
import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import calibrate_grid_positions as cgp

# ============================================================
GRID_DIR      = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
ROIS_JSON     = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
GRID_Z_INDEX  = 5
POS_SPLIT     = 53
# ============================================================

def main():
    grid_path = Path(GRID_DIR)
    labels = set()
    for d in grid_path.iterdir():
        if d.is_dir():
            m = re.match(r"^(Pos\d+)_x\+0_y\+0$", d.name)
            if m and m.group(1) != "Pos0":
                labels.add(m.group(1))

    labels = sorted(labels, key=lambda x: int(re.search(r"\d+", x).group()))
    print(f"Calibrating {len(labels)} positions", flush=True)

    ok = 0
    skip = 0
    errors = []
    for i, label in enumerate(labels):
        out_path = grid_path / f"grid_calibration_{label}.json"
        if out_path.exists():
            skip += 1
            print(f"[SKIP] {label} ({i+1}/{len(labels)})", flush=True)
            continue
        print(f"[{i+1}/{len(labels)}] {label}...", end="", flush=True)
        cgp.GRID_DIR = GRID_DIR
        cgp.BASE_LABEL = label
        cgp.GRID_Z_INDEX = GRID_Z_INDEX
        cgp.CHANNEL_ROIS_JSON = ROIS_JSON
        cgp.OUTPUT_JSON = str(out_path)
        cgp.POS_SPLIT = POS_SPLIT
        try:
            cgp.main()
            ok += 1
            print(f" OK", flush=True)
        except Exception as e:
            errors.append((label, str(e)))
            print(f" ERROR: {e}", flush=True)

    print(f"\nDone: {ok} OK, {skip} skipped, {len(errors)} errors", flush=True)
    for label, err in errors:
        print(f"  ERROR {label}: {err}", flush=True)

if __name__ == "__main__":
    main()
