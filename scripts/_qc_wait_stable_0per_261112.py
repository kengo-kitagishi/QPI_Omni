"""_qc_wait_stable_0per_261112.py

Waits until the 0% period of 261112 run 3 has settled: follows the median phase inside the
channels (crop_sub tiles, every 6th Pos, end traps excluded) frame by frame, fits the slope
over the last hour (12 frames) and exits when |slope| < THRESH rad/h for two consecutive
hourly windows.  Prints one line per new frame.

    python scripts/_qc_wait_stable_0per_261112.py
"""
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs

ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_3")
GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
POS = list(range(2, 125, 6))
START = 70
THRESH = 0.0015          # rad per hour
FRAMES_PER_H = 12
MAX_WAIT_S = 6 * 3600


def main():
    masks = {}
    for pos in POS:
        rois = json.load(open(GRID / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json", encoding="utf-8"))
        mimg = tifffile.imread(str(GRID / f"Pos{pos}_x+0_y+0" / "output_phase" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        for ch in range(1, len(rois) - 1):
            r = rois[ch]
            m = gs.extract_rect_roi(mimg, int(r["cy"]), int(r["cx"]), r["crop_w"], 240) < -1.0
            inner = ndimage.binary_erosion(m, iterations=4)
            if inner.any():
                masks[(pos, ch)] = inner
    series = {}
    t = START
    t0 = time.time()
    while time.time() - t0 < MAX_WAIT_S:
        vals = []
        complete = True
        for (pos, ch), inner in masks.items():
            f = ROOT / f"Pos{pos}" / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / f"ch{ch:02d}" / f"img_{t:09d}_ph_000.tif"
            if not f.exists():
                complete = False
                break
            try:
                a = tifffile.imread(str(f)).astype(np.float64)
            except Exception:
                complete = False
                break
            if a.shape == inner.shape:
                vals.append(np.median(a[inner]))
        if not complete:
            time.sleep(60)
            continue
        series[t] = float(np.median(vals))
        ts = sorted(series)
        line = f"t={t} {time.strftime('%H:%M')} inside {series[t]:+.4f} rad"
        if len(ts) >= 2 * FRAMES_PER_H:
            def slope(w):
                x = np.array(w, float)
                y = np.array([series[k] for k in w])
                return np.polyfit(x / FRAMES_PER_H, y, 1)[0]
            s1 = slope(ts[-FRAMES_PER_H:])
            s0 = slope(ts[-2 * FRAMES_PER_H:-FRAMES_PER_H])
            line += f" | slope last hour {s1:+.4f} rad/h, hour before {s0:+.4f} rad/h"
            print(line, flush=True)
            if abs(s1) < THRESH and abs(s0) < THRESH:
                print(f"STABLE at t={t} ({time.strftime('%H:%M')}): |slope| < {THRESH} rad/h for two hours", flush=True)
                return
        else:
            print(line, flush=True)
        t += 1
    print("gave up after the maximum wait", flush=True)


if __name__ == "__main__":
    main()
