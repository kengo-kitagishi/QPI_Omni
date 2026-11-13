"""_qc_shift_history.py

Over a whole run: how far the frames sat from grid00 (the online shift), how far from the
nearest calibrated grid point (the residual warp), and how that evolved with time.  Reads the
pos_shifts_cal_online.json the online crop_sub writes per Pos, plus grid_calibration_PosN.json
for the convex hull of the measured grid.

    python scripts/_qc_shift_history.py --crop-root D:/.../261108/online_crop_sub_YE_5min_31d_3 \
        --grid-dir E:/261108/recon_YE_grid_2 --tag 261108_run3 --interval-min 5
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import figure_logger  # noqa: F401
from figure_logger import save_figure
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop-root", required=True)
    ap.add_argument("--grid-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--interval-min", type=float, default=5.0)
    ap.add_argument("--first-frame", type=int, default=3, help="frames before this are skipped (controller not engaged)")
    args = ap.parse_args()
    crop = Path(args.crop_root); grid = Path(args.grid_dir)

    rows = []       # pos, t, sx, sy, res, inside_hull
    for sp in sorted(crop.glob("Pos*/output_phase/channels/pos_shifts_cal_online.json"),
                     key=lambda p: int(re.search(r"Pos(\d+)", str(p)).group(1))):
        pos = int(re.search(r"Pos(\d+)", str(sp)).group(1))
        calp = grid / f"grid_calibration_Pos{pos}.json"
        if not calp.exists():
            continue
        pts = json.load(open(calp, encoding="utf-8"))["positions"]
        ax = np.array([-e["actual_dx_px"] for e in pts]); ay = np.array([-e["actual_dy_px"] for e in pts])
        tree = cKDTree(np.column_stack([ax, ay])); hull = Delaunay(np.column_stack([ax, ay]))
        fr = json.load(open(sp, encoding="utf-8"))["frame_results"]
        s = np.array([[f["frame_index"], f["shift_x_avg"], f["shift_y_avg"]] for f in fr if f is not None and f.get("shift_x_avg") is not None and f["frame_index"] >= args.first_frame])
        if len(s) == 0:
            continue
        d, _ = tree.query(s[:, 1:3]); inside = hull.find_simplex(s[:, 1:3]) >= 0
        for (t, sx, sy), dd, ins in zip(s, d, inside):
            rows.append((pos, int(t), sx, sy, dd, bool(ins)))
    a = np.array(rows, dtype=float)
    pos, t, sx, sy, res, ins = a.T
    r = np.hypot(sx, sy)
    n_pos = len(np.unique(pos)); frames = np.unique(t).astype(int)
    print(f"{args.tag}: {n_pos} Pos, frames {frames.min()}..{frames.max()} ({len(frames)} frames, {frames.max()*args.interval_min/60:.1f} h), {len(a)} records")
    print(f"shift vs grid00 [px]: radius p50 {np.median(r):.2f}  p90 {np.percentile(r,90):.2f}  p99 {np.percentile(r,99):.2f}  max {r.max():.2f}"
          f"   |x| p90 {np.percentile(np.abs(sx),90):.2f}  |y| p90 {np.percentile(np.abs(sy),90):.2f}")
    print(f"residual to nearest calibrated point [px]: p50 {np.median(res):.3f}  p90 {np.percentile(res,90):.3f}  p99 {np.percentile(res,99):.3f}"
          f"   >0.1 px {np.mean(res>0.1)*100:.1f}%   >0.3 px {np.mean(res>0.3)*100:.1f}%   outside hull {np.mean(~ins.astype(bool))*100:.1f}%")
    # time course in blocks
    nb = 8
    edges = np.linspace(frames.min(), frames.max() + 1, nb + 1)
    print(f"\n{'frames':>12s} {'hours':>10s} {'radius p50':>10s} {'p90':>6s} {'p99':>6s} {'resid p50':>10s} {'p90':>6s} {'>0.3px':>7s} {'outside':>8s}")
    tc = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        if not m.any():
            continue
        tc.append((lo, hi, np.median(r[m]), np.percentile(r[m], 90), np.percentile(r[m], 99), np.median(res[m]),
                   np.percentile(res[m], 90), np.mean(res[m] > 0.3) * 100, np.mean(~ins[m].astype(bool)) * 100))
        print(f"{int(lo):5d}-{int(hi)-1:<6d} {lo*args.interval_min/60:4.1f}-{hi*args.interval_min/60:<4.1f} {tc[-1][2]:10.2f} {tc[-1][3]:6.2f} {tc[-1][4]:6.2f} {tc[-1][5]:10.3f} {tc[-1][6]:6.3f} {tc[-1][7]:6.1f}% {tc[-1][8]:7.1f}%")
    # the slow part: per-Pos mean shift per block (drift of the sample relative to the grid)
    print("\nslow drift relative to grid00 (median over Pos of the per-Pos block mean) [px]:")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (t >= lo) & (t < hi)
        if not m.any():
            continue
        mx = np.array([sx[m & (pos == p)].mean() for p in np.unique(pos[m])]); my = np.array([sy[m & (pos == p)].mean() for p in np.unique(pos[m])])
        print(f"  frames {int(lo):4d}-{int(hi)-1:<4d}: x {np.median(mx):+.2f} (Pos spread {mx.std():.2f})   y {np.median(my):+.2f} (Pos spread {my.std():.2f})")
    # fast part: within-Pos frame-to-frame jitter
    jit = []
    for p in np.unique(pos):
        m = pos == p
        o = np.argsort(t[m]); dx = np.diff(sx[m][o]); dy = np.diff(sy[m][o])
        jit.append((dx.std(), dy.std()))
    jit = np.array(jit)
    print(f"\nframe-to-frame jitter within a Pos (std of successive differences) [px]: x {np.median(jit[:,0]):.3f} ({np.median(jit[:,0])*0.3457*1e3:.0f} nm), y {np.median(jit[:,1]):.3f} ({np.median(jit[:,1])*0.3457*1e3:.0f} nm)")

    # figure: percentiles vs time for radius and residual
    hours = frames * args.interval_min / 60
    def pct(v):
        return np.array([[np.percentile(v[t == f], q) for q in (50, 90, 99)] for f in frames])
    pr, pres = pct(r), pct(res)
    fig, axes = plt.subplots(2, 1, figsize=(89 / 25.4, 80 / 25.4), sharex=True)
    for ax, P, lab in ((axes[0], pr, "shift vs grid00 [px]"), (axes[1], pres, "residual to nearest grid point [px]")):
        ax.fill_between(hours, P[:, 0], P[:, 2], color="0.85", lw=0, label="p50-p99")
        ax.plot(hours, P[:, 1], color="0.4", lw=0.7, label="p90")
        ax.plot(hours, P[:, 0], color="k", lw=0.8, label="p50")
        ax.set_ylabel(lab)
    axes[1].set_xlabel("time since start [h]")
    axes[0].legend(frameon=False, fontsize=6)
    axes[0].set_title(f"{args.tag}: {n_pos} Pos, per-frame percentiles over Pos", fontsize=7)
    for ax, lab in zip(axes, "ab"):
        ax.text(-0.18, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout()
    save_figure(fig, params={"tag": args.tag, "crop_root": str(crop), "grid_dir": str(grid), "first_frame": args.first_frame,
                             "interval_min": args.interval_min, "n_pos": int(n_pos), "n_frames": int(len(frames))},
                description=(f"{args.tag}: per-frame percentiles over Pos of (a) the online frame shift against grid00 and "
                             f"(b) the residual to the nearest calibrated grid point, both from pos_shifts_cal_online.json."),
                data={"frames": frames, "hours": hours, "radius_p50_p90_p99": pr, "residual_p50_p90_p99": pres,
                      "pos": pos, "t": t, "sx": sx, "sy": sy, "res": res, "inside_hull": ins},
                fmt="png")


if __name__ == "__main__":
    main()
