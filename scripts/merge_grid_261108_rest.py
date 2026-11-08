"""Merge the resumed 261108 grid into grid_YE_2 once it is complete.

The first run (grid_YE_2) stalled in Pos58 when Micro-Manager ran out of memory (2026-11-08 18:13); after an MM
restart Pos58-118 were re-acquired into grid_YE_3. When grid_YE_3 holds all 61 x 121 points and has been quiet
for IDLE_MIN minutes:
  1. the partial Pos58 points in grid_YE_2 are moved aside to grid_YE_2_Pos58_partial (kept, not deleted)
  2. every point folder of grid_YE_3 is moved into grid_YE_2 (same volume: a rename)
run_recon_calib_261108_2.py, already waiting on grid_YE_2, then sees 14399 points and goes on.
"""
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

MAIN = Path(r"E:\261108\grid_YE_2")
REST = Path(r"E:\261108\grid_YE_3")
ASIDE = Path(r"E:\261108\grid_YE_2_Pos58_partial")
FIRST_REDONE, LAST = 58, 118
N_REST = (LAST - FIRST_REDONE + 1) * 121
IDLE_MIN = 1               # user: no long waits; one quiet minute so the last point is fully written
POINT_RE = re.compile(r"^Pos(\d+)_x[+-]\d+_y[+-]\d+$")


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def main():
    last_n, last_change = -1, time.time()
    while True:
        n = sum(1 for e in os.scandir(REST) if e.is_dir())
        if n != last_n:
            last_n, last_change = n, time.time()
        if n >= N_REST and (time.time() - last_change) / 60 >= IDLE_MIN:
            break
        time.sleep(60)
    log(f"{REST.name} complete: {n} points")
    ASIDE.mkdir(exist_ok=True)
    moved_aside = 0
    for d in MAIN.iterdir():
        m = POINT_RE.match(d.name)
        if m and int(m.group(1)) >= FIRST_REDONE:
            shutil.move(str(d), str(ASIDE / d.name))
            moved_aside += 1
    log(f"moved {moved_aside} partial Pos{FIRST_REDONE} points aside to {ASIDE}")
    moved = 0
    for d in REST.iterdir():
        if POINT_RE.match(d.name):
            if (MAIN / d.name).exists():
                raise RuntimeError(f"{d.name} already in {MAIN}")
            os.rename(d, MAIN / d.name)
            moved += 1
    total = sum(1 for e in os.scandir(MAIN) if e.is_dir() and POINT_RE.match(e.name))
    log(f"moved {moved} points {REST.name} -> {MAIN.name}; {MAIN.name} now has {total} points")
    return 0 if total == 119 * 121 else 1


if __name__ == "__main__":
    sys.exit(main())
