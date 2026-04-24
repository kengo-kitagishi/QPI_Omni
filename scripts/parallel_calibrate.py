"""Parallel grid calibration: runs calibrate_grid_positions for each Pos in parallel."""
import sys
import re
import time
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

GRID_DIR = Path(r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1")
Z_INDEX = 5
POS_SPLIT = 52
N_WORKERS = 4
N_GRID_THREADS = 7   # per-process grid-point threads (N_WORKERS * N_GRID_THREADS ≈ cpu_count)

SCRIPT_DIR = Path(__file__).resolve().parent


def calibrate_one(args):
    label, rois_path, out_path = args
    cv2.setNumThreads(1)
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import calibrate_grid_positions as cgp
    cgp.GRID_DIR = str(GRID_DIR)
    cgp.BASE_LABEL = label
    cgp.GRID_Z_INDEX = Z_INDEX
    cgp.CHANNEL_ROIS_JSON = rois_path
    cgp.OUTPUT_JSON = out_path
    cgp.POS_SPLIT = POS_SPLIT
    cgp.N_GRID_THREADS = N_GRID_THREADS
    try:
        cgp.main()
        return label, True, None
    except Exception as e:
        return label, False, str(e)


def main():
    pattern_re = re.compile(r"^(Pos\d+)_x\+0_y\+0$")
    labels = sorted(
        [pattern_re.match(d.name).group(1)
         for d in GRID_DIR.iterdir()
         if d.is_dir() and pattern_re.match(d.name) and not d.name.startswith("Pos0_")],
        key=lambda x: int(re.search(r"\d+", x).group()),
    )

    todo = []
    skip = 0
    for label in labels:
        out_path = GRID_DIR / f"grid_calibration_{label}.json"
        if out_path.exists():
            skip += 1
            continue
        rois_path = (GRID_DIR / f"{label}_x+0_y+0" / "output_phase"
                     / "channels" / "channel_rois.json")
        if not rois_path.exists():
            print(f"WARN: {label} no channel_rois.json, skipping")
            continue
        todo.append((label, str(rois_path), str(out_path)))

    print(f"{len(labels)} total, {skip} already done, {len(todo)} to calibrate, {N_WORKERS} workers")

    t0 = time.perf_counter()
    ok = 0
    errors = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(calibrate_one, args): args[0] for args in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            label, success, err = fut.result()
            if success:
                ok += 1
                print(f"[{i}/{len(todo)}] {label} OK", flush=True)
            else:
                errors.append((label, err))
                print(f"[{i}/{len(todo)}] {label} ERROR: {err}", flush=True)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.0f}s: {ok} OK, {skip} skipped, {len(errors)} errors")
    for label, err in errors:
        print(f"  ERROR {label}: {err}")


if __name__ == "__main__":
    main()
