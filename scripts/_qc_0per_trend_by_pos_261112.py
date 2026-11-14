"""_qc_0per_trend_by_pos_261112.py

Per-Pos time course of the median phase inside the channels (crop_sub tiles, end traps
excluded) during the 0% period of 261112 run 3, to see whether some Pos (e.g. near the
inlet) react more to pressure and are still drifting.  Prints group summaries and writes a
per-Pos csv; the figure shows every Pos as a trace, coloured by Pos number.

    python scripts/_qc_0per_trend_by_pos_261112.py
"""
import collections
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
import figure_logger  # noqa: F401
from figure_logger import save_figure
import matplotlib.pyplot as plt
import grid_subtract as gs

ROOT = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_3")
GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25")
SAVE = Path(r"D:\AquisitionData\Kitagishi\261112\ph_2per_5min_31d_3")
OUT = Path(r"C:\QPI_analysis\261112\_qc")
STEP = 2           # every 2nd frame


def main():
    cnt = collections.Counter()
    for p in ROOT.glob("Pos*"):
        d = p / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / "ch01"
        for f in d.glob("*.tif"):
            cnt[int(re.search(r"img_0*(\d+)_", f.name).group(1))] += 1
    last = max(t for t, n in cnt.items() if n >= 124)
    frames = list(range(14, last + 1, STEP))
    pos_list = list(range(1, 125))
    V = np.full((len(pos_list), len(frames)), np.nan)
    t0 = time.perf_counter()
    for i, pos in enumerate(pos_list):
        rois = json.load(open(GRID / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json", encoding="utf-8"))
        mimg = tifffile.imread(str(GRID / f"Pos{pos}_x+0_y+0" / "output_phase" / "img_000000000_ph_000_phase.tif")).astype(np.float64)
        inners = {}
        for ch in range(1, len(rois) - 1):
            r = rois[ch]
            m = gs.extract_rect_roi(mimg, int(r["cy"]), int(r["cx"]), r["crop_w"], 240) < -1.0
            inner = ndimage.binary_erosion(m, iterations=4)
            if inner.any():
                inners[ch] = inner
        for j, t in enumerate(frames):
            vals = []
            for ch, inner in inners.items():
                f = ROOT / f"Pos{pos}" / "output_phase" / "channels" / "crop_sub_rawraw" / "z000" / f"ch{ch:02d}" / f"img_{t:09d}_ph_000.tif"
                if f.exists():
                    a = tifffile.imread(str(f))
                    if a.shape == inner.shape:
                        vals.append(float(np.median(a[inner])))
            if vals:
                V[i, j] = np.median(vals)
        if pos % 20 == 0:
            print(f"  Pos{pos} ({time.perf_counter()-t0:.0f}s)", flush=True)
    fr = np.array(frames, float)
    hours = (fr - 22) / 12.0                      # 0 = medium switch at frame 22
    stamps = {}
    for t in (22, 62, 64, 100, frames[-1]):
        f = SAVE / "Pos1" / "z000" / "output_phase_raw" / f"img_{t:09d}_ph_000_phase.tif"
        if f.exists():
            stamps[t] = time.strftime("%H:%M", time.localtime(f.stat().st_mtime))
    print(f"\nframes {frames[0]}..{frames[-1]} every {STEP}; switch to 0% at t22 ({stamps.get(22)}), syringe ~t62 ({stamps.get(62)}), latest t{frames[-1]} ({stamps.get(frames[-1])})")

    def at(t):
        return np.nanmean(V[:, (fr >= t - 2) & (fr <= t + 2)], axis=1)
    step = at(30) - at(18)                          # size of the 0% step
    since_syr = at(frames[-2]) - at(70)             # drift after the syringe push
    last2h = fr >= fr[-1] - 24
    slope = np.array([np.polyfit(hours[last2h], V[i, last2h], 1)[0] if np.isfinite(V[i, last2h]).sum() > 5 else np.nan for i in range(len(pos_list))])
    groups = [(1, 39), (40, 80), (81, 124)]
    print(f"\n{'Pos':>9s} {'0% step':>9s} {'drift since t70':>16s} {'slope last 2 h':>15s}")
    for a, b in groups:
        m = (np.array(pos_list) >= a) & (np.array(pos_list) <= b)
        print(f"{a:3d}-{b:<5d} {np.nanmedian(step[m]):+9.4f} {np.nanmedian(since_syr[m]):+16.4f} {np.nanmedian(slope[m]):+12.4f} rad/h  (p90 |slope| {np.nanpercentile(np.abs(slope[m]),90):.4f})")
    for a, b in [(40, 49), (50, 59), (60, 69), (70, 80)]:
        m = (np.array(pos_list) >= a) & (np.array(pos_list) <= b)
        print(f"  {a}-{b}: step {np.nanmedian(step[m]):+.4f}, drift since t70 {np.nanmedian(since_syr[m]):+.4f}, slope {np.nanmedian(slope[m]):+.4f} rad/h")
    order = np.argsort(slope)
    print("\nsteepest 8 Pos (last 2 h):", ", ".join(f"Pos{pos_list[k]} {slope[k]:+.4f}" for k in order[:8]))
    np.savetxt(OUT / "0per_trend_by_pos.csv", np.column_stack([pos_list, step, since_syr, slope]), delimiter=",", fmt="%.6g",
               header="pos,step_0per_rad,drift_since_t70_rad,slope_last2h_rad_per_h")

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 70 / 25.4))
    cmap = plt.get_cmap("cividis")
    for i, pos in enumerate(pos_list):
        axes[0].plot(hours, V[i] - np.nanmean(V[i, (fr >= 14) & (fr <= 20)]), color=cmap(i / len(pos_list)), lw=0.4, alpha=0.6)
    axes[0].axvline(0, color="0.5", lw=0.6)
    axes[0].axvline((62 - 22) / 12, color="0.5", lw=0.6, ls="--")
    axes[0].set_xlabel("time since the switch to 0% [h]")
    axes[0].set_ylabel("channel interior phase, change from 2% [rad]")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, 124))
    fig.colorbar(sm, ax=axes[0], pad=0.01).set_label("Pos")
    axes[1].scatter(pos_list, slope, s=6, c="0.2", linewidths=0)
    axes[1].axhline(0, color="0.6", lw=0.5)
    axes[1].axvspan(40, 80, color="0.9", lw=0)
    axes[1].set_xlabel("Pos")
    axes[1].set_ylabel("slope over the last 2 h [rad/h]")
    fig.tight_layout()
    save_figure(fig, params={"frames": [frames[0], frames[-1]], "step": STEP, "dataset": "261112 run 3"},
                description="Per-Pos median phase inside the channels during the 0% period of 261112 run 3 (change from the 2% level), and the per-Pos slope over the last two hours.",
                data={"pos": np.array(pos_list), "frames": fr, "hours": hours, "inside": V, "step": step, "drift_since_t70": since_syr, "slope_last2h": slope},
                fmt="png")


if __name__ == "__main__":
    main()
