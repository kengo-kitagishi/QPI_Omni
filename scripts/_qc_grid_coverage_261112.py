"""_qc_grid_coverage_261112.py

How large a grid is needed, judged by the COVERING SPACE of the measured points: a frame is
covered when a measured grid point lies within RHO px of its shift (that is the residual
warp the subtraction will have to absorb).  Frames are the real shift cloud of 261108 run 3
(frames >= 3, all Pos); spike frames (> 3 MAD from the Pos median in x or y) are reported
separately so the answer is "excluding bad frames".

For each calibrated Pos given, nested sub-grids |xi|,|yi| <= h (h = 5, 6, ..., HALF) are cut
out of the measured point cloud, so the curve "covered fraction vs grid size" comes from real
stage behaviour.  The setpoint is the centroid of the sub-grid (what the controller now targets).

    python scripts/_qc_grid_coverage_261112.py --grid D:/.../recon_0p03um_25x25 --pos 1-20 --half 12
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

SHIFTS = Path(r"D:\AquisitionData\Kitagishi\261108\online_crop_sub_YE_5min_31d_3")
RHOS = (0.05, 0.1, 0.15, 0.2, 0.3)


def frame_cloud():
    per = {}
    for sp in SHIFTS.glob("Pos*/output_phase/channels/pos_shifts_cal_online.json"):
        v = [(f["shift_x_avg"], f["shift_y_avg"]) for f in json.load(open(sp, encoding="utf-8"))["frame_results"]
             if f and f.get("shift_x_avg") is not None and f["frame_index"] >= 3]
        if v:
            per[sp.parts[-4]] = np.array(v)
    S, spike = [], []
    for v in per.values():
        med = np.median(v, axis=0)
        mad = 1.4826 * np.median(np.abs(v - med), axis=0) + 1e-9
        sp_ = np.any(np.abs(v - med) > 3 * mad, axis=1)
        S.append(v)
        spike.append(sp_)
    return np.vstack(S), np.concatenate(spike)


def parse_pos(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--pos", required=True)
    ap.add_argument("--half", type=int, required=True, help="half-size of the acquired grid (12 for 25x25)")
    ap.add_argument("--step-um", type=float, default=0.03)
    args = ap.parse_args()
    S, spike = frame_cloud()
    good = ~spike
    print(f"frame cloud: {len(S)} frames, spikes {spike.mean()*100:.1f}% (excluded below); radius p90 {np.percentile(np.hypot(*S[good].T), 90):.2f} p99 {np.percentile(np.hypot(*S[good].T), 99):.2f} px")
    clouds = {}
    for p in parse_pos(args.pos):
        cal = Path(args.grid) / f"grid_calibration_Pos{p}.json"
        if not cal.exists():
            continue
        pts = json.load(open(cal, encoding="utf-8"))["positions"]
        clouds[p] = (np.array([e["xi"] for e in pts]), np.array([e["yi"] for e in pts]),
                     np.column_stack([[-e["actual_dx_px"] for e in pts], [-e["actual_dy_px"] for e in pts]]))
    print(f"calibrated Pos used: {sorted(clouds)}  (median over Pos of each figure)\n")
    hdr = f"{'sub-grid':>9s} {'half [um]':>9s} {'nominal px':>10s}" + "".join(f"  covered<{r}px" % () if False else f"  <{r:.2f}px" for r in RHOS) + "   resid p50   p90"
    print(hdr)
    for h in range(5, args.half + 1):
        rows = []
        for p, (xi, yi, P) in clouds.items():
            m = (np.abs(xi) <= h) & (np.abs(yi) <= h)
            Q = P[m]
            c = Q.mean(axis=0)                       # controller setpoint = centroid of this sub-grid
            d, _ = cKDTree(Q).query(S[good] + c)     # frames land around the centroid
            rows.append([np.mean(d < r) * 100 for r in RHOS] + [np.median(d), np.percentile(d, 90)])
        rows = np.median(np.array(rows), axis=0)
        n = 2 * h + 1
        print(f"{n:3d}x{n:<5d} {h*args.step_um:9.2f} {h*args.step_um/0.34567514677103717:10.2f}" + "".join(f"  {v:7.1f}%" for v in rows[:len(RHOS)]) + f"   {rows[-2]:8.3f} {rows[-1]:5.3f}")


if __name__ == "__main__":
    main()
