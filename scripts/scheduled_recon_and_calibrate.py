"""
scheduled_recon_and_calibrate.py
1. batch_reconstruction_grid.py
2. channel_crop --detect for every PosN_x+0_y+0 (parallel)
3. calibrate_grid_positions per Pos (parallel)
Designed for scheduled execution (schtasks).
"""
import subprocess
import sys
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PYTHON = sys.executable
SCRIPT_DIR = Path(__file__).resolve().parent
GRID_DIR = Path(r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1")
Z_INDEX = 5
POS_SPLIT = 52
N_WORKERS_DETECT = 12
N_WORKERS_CALIBRATE = 12

pattern_re = re.compile(r"^(Pos\d+)_x\+0_y\+0$")


# --- Step 1: Reconstruction ---
print(f"\n{'='*60}")
print("Step 1: batch_reconstruction_grid")
print(f"{'='*60}", flush=True)
result = subprocess.run(
    [PYTHON, str(SCRIPT_DIR / "batch_reconstruction_grid.py")],
    cwd=str(SCRIPT_DIR),
)
if result.returncode != 0:
    print(f"ERROR: batch_reconstruction_grid exited with code {result.returncode}", flush=True)
    sys.exit(result.returncode)


# --- Step 2: channel_crop --detect (parallel) ---
print(f"\n{'='*60}")
print(f"Step 2: channel_crop --detect (parallel, {N_WORKERS_DETECT} workers)")
print(f"{'='*60}", flush=True)

pos_dirs = sorted(
    [d for d in GRID_DIR.iterdir()
     if d.is_dir() and pattern_re.match(d.name) and not d.name.startswith("Pos0_")],
    key=lambda d: int(re.search(r"\d+", d.name).group()),
)

detect_todo = []
detect_skip = 0
for d in pos_dirs:
    label = pattern_re.match(d.name).group(1)
    output_phase = d / "output_phase"
    rois_path = output_phase / "channels" / "channel_rois.json"
    if rois_path.exists():
        detect_skip += 1
        continue
    detect_todo.append((label, str(output_phase), str(rois_path)))


def _detect_one(args):
    label, output_phase, rois_path = args
    r = subprocess.run(
        [PYTHON, str(SCRIPT_DIR / "channel_crop.py"),
         "--dir", output_phase,
         "--detect",
         "--pattern", f"*_ph_{Z_INDEX:03d}_phase.tif"],
        cwd=str(SCRIPT_DIR),
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    return label, r.returncode == 0 and Path(rois_path).exists()


detect_ok = detect_skip
detect_err = 0
if detect_todo:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS_DETECT) as ex:
        futs = {ex.submit(_detect_one, args): args[0] for args in detect_todo}
        for i, fut in enumerate(as_completed(futs), 1):
            label, success = fut.result()
            if success:
                detect_ok += 1
                print(f"  [{i}/{len(detect_todo)}] {label} OK", flush=True)
            else:
                detect_err += 1
                print(f"  [{i}/{len(detect_todo)}] {label} ERROR", flush=True)
    print(f"channel_crop: {detect_ok} OK, {detect_skip} skipped, {detect_err} errors "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
else:
    print(f"channel_crop: {detect_ok} OK (all skipped)", flush=True)

if detect_err > 0 and detect_ok == 0:
    print("ERROR: No channel_rois.json created, aborting.", flush=True)
    sys.exit(1)


# --- Step 3: calibrate_grid_positions (parallel) ---
print(f"\n{'='*60}")
print(f"Step 3: calibrate_grid_positions (parallel, {N_WORKERS_CALIBRATE} workers)")
print(f"{'='*60}", flush=True)

labels = sorted(
    [pattern_re.match(d.name).group(1) for d in pos_dirs],
    key=lambda x: int(re.search(r"\d+", x).group()),
)

cal_todo = []
cal_skip = 0
for label in labels:
    out_path = GRID_DIR / f"grid_calibration_{label}.json"
    if out_path.exists():
        cal_skip += 1
        continue
    rois_path = (GRID_DIR / f"{label}_x+0_y+0" / "output_phase"
                 / "channels" / "channel_rois.json")
    if not rois_path.exists():
        print(f"WARN: {label} no channel_rois.json, skipping", flush=True)
        continue
    cal_todo.append((label, str(rois_path), str(out_path)))


def _calibrate_one(args):
    label, rois_path, out_path = args
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    import calibrate_grid_positions as cgp
    cgp.GRID_DIR = str(GRID_DIR)
    cgp.BASE_LABEL = label
    cgp.GRID_Z_INDEX = Z_INDEX
    cgp.CHANNEL_ROIS_JSON = rois_path
    cgp.OUTPUT_JSON = out_path
    cgp.POS_SPLIT = POS_SPLIT
    try:
        cgp.main()
        return label, True, None
    except Exception as e:
        return label, False, str(e)


cal_ok = 0
cal_errors = []
if cal_todo:
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS_CALIBRATE) as ex:
        futs = {ex.submit(_calibrate_one, args): args[0] for args in cal_todo}
        for i, fut in enumerate(as_completed(futs), 1):
            label, success, err = fut.result()
            if success:
                cal_ok += 1
                print(f"[{i}/{len(cal_todo)}] {label} OK", flush=True)
            else:
                cal_errors.append((label, err))
                print(f"[{i}/{len(cal_todo)}] {label} ERROR: {err}", flush=True)
    elapsed = time.perf_counter() - t0
    print(f"\nCalibration: {cal_ok} OK, {cal_skip} skipped, {len(cal_errors)} errors "
          f"({elapsed:.0f}s)", flush=True)
    for label, err in cal_errors:
        print(f"  ERROR {label}: {err}", flush=True)
else:
    print(f"Calibration: {cal_skip} OK (all skipped)", flush=True)

print(f"\n{'='*60}")
print("All done.")
print(f"{'='*60}")
