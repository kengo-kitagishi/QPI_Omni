"""_qc_stride_residual_sim_261112.py

What residual warp would a timelapse see against the measured point cloud of Pos82 acquired
(a) in the unidirectional row order with 30 nm steps and (b) in the 2D stride order?
Uses the real frame-shift cloud of 261108 run 3 (frames >= 3, all Pos, 90k frames) as the
positions the frames land on, and the nearest measured grid point as the subtraction target.

    python scripts/_qc_stride_residual_sim_261112.py
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree

CALS = {"sequential Pos81 (30 nm)": Path(r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_19x19\grid_calibration_Pos81.json"),
        "2D stride": Path(r"D:\AquisitionData\Kitagishi\261112\recon_stridetest_1\grid_calibration_Pos82.json")}
SHIFTS = Path(r"D:\AquisitionData\Kitagishi\261108\online_crop_sub_YE_5min_31d_3")


def main():
    S = []
    for sp in SHIFTS.glob("Pos*/output_phase/channels/pos_shifts_cal_online.json"):
        for f in json.load(open(sp, encoding="utf-8"))["frame_results"]:
            if f and f.get("shift_x_avg") is not None and f["frame_index"] >= 3:
                S.append((f["shift_x_avg"], f["shift_y_avg"]))
    S = np.array(S)
    print(f"frame shift cloud: {len(S)} frames (261108 run 3), radius p50 {np.median(np.hypot(*S.T)):.2f} p90 {np.percentile(np.hypot(*S.T), 90):.2f} px")
    print(f"{'grid of Pos82':28s} {'resid p50':>9s} {'p90':>6s} {'p99':>6s} {'>0.1px':>7s} {'>0.3px':>7s} {'outside hull':>13s}")
    for tag, cal in CALS.items():
        pts = json.load(open(cal, encoding="utf-8"))["positions"]
        P = np.column_stack([[-e["actual_dx_px"] for e in pts], [-e["actual_dy_px"] for e in pts]])
        d, _ = cKDTree(P).query(S)
        inside = Delaunay(P).find_simplex(S) >= 0
        print(f"{tag:28s} {np.median(d):9.3f} {np.percentile(d,90):6.3f} {np.percentile(d,99):6.3f} {np.mean(d>0.1)*100:6.1f}% {np.mean(d>0.3)*100:6.1f}% {np.mean(~inside)*100:12.1f}%")


if __name__ == "__main__":
    main()
