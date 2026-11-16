"""_qc_wait_stable_cells1_261112.py

Waits until the channel phase of 261112 timelapse 1 (cells) has settled after the pressure
was touched at T=9: follows the median phase inside the cell-free channels of the drift
whitelist (channel_whitelist.json, end traps excluded), frame by frame, for all Pos and for
the inlet group (Pos41-65, the largest and slowest responders).  Fits the slope over the last
hour (12 frames) and exits when |slope| < THRESH rad/h for two consecutive hourly windows in
both groups.  Same criterion as _qc_wait_stable_0per_261112.py.

    python scripts/_qc_wait_stable_cells1_261112.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import grid_subtract as gs

ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_cells1_5min_31d")
GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
WHITELIST = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112_cells1\channel_whitelist.json")
INLET = range(41, 66)
START = 9                # first frame after the pressure was touched
THRESH = 0.0005          # rad per hour
FRAMES_PER_H = 12
MAX_WAIT_S = 16 * 3600


def main():
    wl = json.load(open(WHITELIST, encoding="utf-8"))["whitelist"]
    masks = {}
    for lab, chs in wl.items():
        pos = int(lab[3:])
        gdir = GRID / f"{lab}_x+0_y+0" / "output_phase"
        rois = json.load(open(gdir / "channels" / "channel_rois.json", encoding="utf-8"))
        mimg = tifffile.imread(str(gdir / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        for ch in chs:
            if ch == 0 or ch >= len(rois) - 1:
                continue
            r = rois[ch]
            m = gs.extract_rect_roi(mimg, int(r["cy"]), int(r["cx"]), r["crop_w"], 240) < -1.0
            inner = ndimage.binary_erosion(m, iterations=4)
            if inner.any():
                masks[(pos, ch)] = inner
    print(f"{len(masks)} cell-free channels in {len({p for p, _ in masks})} Pos "
          f"({sum(1 for p, _ in masks if p in INLET)} in the inlet group)", flush=True)
    series = {"all": {}, "inlet": {}}
    t = START
    t0 = time.time()
    stable_reported = False
    while time.time() - t0 < MAX_WAIT_S:
        vals, vals_in = [], []
        complete = True
        for (pos, ch), inner in masks.items():
            f = (ROOT / f"Pos{pos}" / "output_phase" / "channels" / "crop_sub_rawraw" / "z000"
                 / f"ch{ch:02d}" / f"img_{t:09d}_ph_000.tif")
            if not f.exists():
                complete = False
                break
            try:
                a = tifffile.imread(str(f)).astype(np.float64)
            except Exception:
                complete = False
                break
            if a.shape == inner.shape:
                v = float(np.median(a[inner]))
                vals.append(v)
                if pos in INLET:
                    vals_in.append(v)
        if not complete:
            time.sleep(60)
            continue
        series["all"][t] = float(np.median(vals))
        series["inlet"][t] = float(np.median(vals_in))
        line = (f"t={t} {time.strftime('%H:%M')} inside all {series['all'][t]:+.4f} "
                f"inlet {series['inlet'][t]:+.4f} rad")
        ts = sorted(series["all"])
        if len(ts) >= 2 * FRAMES_PER_H:
            ok = True
            for g in ("all", "inlet"):
                def slope(w):
                    x = np.array(w, float)
                    y = np.array([series[g][k] for k in w])
                    return np.polyfit(x / FRAMES_PER_H, y, 1)[0]
                s1 = slope(ts[-FRAMES_PER_H:])
                s0 = slope(ts[-2 * FRAMES_PER_H:-FRAMES_PER_H])
                line += f" | {g}: last hour {s1:+.4f}, hour before {s0:+.4f} rad/h"
                ok = ok and abs(s1) < THRESH and abs(s0) < THRESH
            print(line, flush=True)
            if ok and not stable_reported:
                print(f"STABLE at t={t} ({time.strftime('%H:%M')}): |slope| < {THRESH} rad/h "
                      f"for two hours in all Pos and in the inlet group", flush=True)
                return
        else:
            print(line, flush=True)
        t += 1
    print("gave up after the maximum wait", flush=True)


if __name__ == "__main__":
    main()
