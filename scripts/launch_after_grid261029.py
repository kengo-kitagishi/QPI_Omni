"""Wait for the 261029 grid acquisition to stop, then run recon + calib (+ raw deletion), then the 260928 restore.

The first watcher (run_recon_calib_261029_2per.py) died at 21:27 when the grid was restarted. While the grid
is being acquired this only lists the top folder (no per-point folder opens), so it does not compete with
the acquisition on E:.
Done = point count >= N_POINTS and no new point folder for IDLE_MIN minutes.
"""
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GRID_DIR = Path(r"E:\261029\grid_2per_1")
N_POINTS = 119 * 121          # Pos0-118, 11 x 11 points
IDLE_MIN = 15
POLL_S = 120


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def main():
    last_n, last_change = -1, time.time()
    while True:
        n = sum(1 for e in os.scandir(GRID_DIR) if e.is_dir())
        if n != last_n:
            last_n, last_change = n, time.time()
        idle = (time.time() - last_change) / 60
        log(f"{n}/{N_POINTS} points, idle {idle:.0f} min")
        if n >= N_POINTS and idle >= IDLE_MIN:
            break
        time.sleep(POLL_S)
    log("acquisition done -> run_recon_calib_261029_2per.py")
    rc = subprocess.run([sys.executable, str(SCRIPT_DIR / "run_recon_calib_261029_2per.py")]).returncode
    log(f"recon + calib exited rc={rc} -> chain_grid261029_then_restore260928.py")
    rc = subprocess.run([sys.executable, str(SCRIPT_DIR / "chain_grid261029_then_restore260928.py")]).returncode
    log(f"chain exited rc={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
