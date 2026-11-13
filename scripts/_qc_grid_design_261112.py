"""_qc_grid_design_261112.py

How should the grid be laid out so the residual (frame shift minus nearest calibrated grid
point) is small for almost every frame?  Uses the 261112 run itself:

  shifts   every (sx, sy) the online run measured against grid00, frames >= 3 (controller on)
  scatter  how far each calibrated grid point sits from where an ideal lattice would put it
           (actual - affine fit of nominal xi, yi), i.e. the stage's per-step error

Then simulates grids (n x n, nominal step s) with the SAME stage scatter and reports the
residual distribution the live run would have seen against the empirical shift cloud:
  - current 11 x 11 @ 0.05 um (measured, per Pos)
  - 11/15/21 x 11/15/21 @ 0.05 / 0.03 / 0.025 um (simulated)
  - and, for the current grid, "4-point interpolation" = distance to the nearest point
    inside the convex hull counts as 0 (what a reference interpolated between the actual
    points would give)
"""
import json
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree, Delaunay

GRID = Path(r"D:\AquisitionData\Kitagishi\261112\recon_2per_grid_1")
CROP = Path(r"D:\AquisitionData\Kitagishi\261112\online_crop_sub_2per_5min_31d_1")
PX_UM = 0.34567514677103717
rng = np.random.default_rng(0)

# ---- empirical shift cloud and current-grid residuals ----
shifts, resid_now, resid_hull = [], [], []
scat_x, scat_y, slope_x, slope_y = [], [], [], []
for cal_p in sorted(GRID.glob("grid_calibration_Pos*.json"), key=lambda p: int(p.stem.split("Pos")[1])):
    label = "Pos" + cal_p.stem.split("Pos")[1]
    pts = json.load(open(cal_p, encoding="utf-8"))["positions"]
    xi = np.array([e["xi"] for e in pts], float); yi = np.array([e["yi"] for e in pts], float)
    ax = np.array([-e["actual_dx_px"] for e in pts]); ay = np.array([-e["actual_dy_px"] for e in pts])
    # affine lattice fit: actual = A [xi, yi, 1]
    M = np.column_stack([xi, yi, np.ones_like(xi)])
    cx, *_ = np.linalg.lstsq(M, ax, rcond=None); cy, *_ = np.linalg.lstsq(M, ay, rcond=None)
    scat_x.append(ax - M @ cx); scat_y.append(ay - M @ cy)
    slope_x.append(cx[1]); slope_y.append(cy[0])      # image x <- stage yi, image y <- stage xi
    sp = CROP / label / "output_phase" / "channels" / "pos_shifts_cal_online.json"
    if not sp.exists():
        continue
    fr = [f for f in json.load(open(sp, encoding="utf-8"))["frame_results"] if f["frame_index"] >= 3]
    s = np.array([[f["shift_x_avg"], f["shift_y_avg"]] for f in fr])
    shifts.append(s)
    tree = cKDTree(np.column_stack([ax, ay]))
    d, _ = tree.query(s)
    resid_now.append(d)
    hull = Delaunay(np.column_stack([ax, ay]))
    inside = hull.find_simplex(s) >= 0
    resid_hull.append(np.where(inside, 0.0, d))

S = np.vstack(shifts); Rn = np.concatenate(resid_now); Rh = np.concatenate(resid_hull)
sx_scat = np.concatenate(scat_x); sy_scat = np.concatenate(scat_y)
print(f"frames x Pos used: {len(S)}  (frames >= 3)")
print(f"shift vs grid00 [px]: |x| p50 {np.median(np.abs(S[:,0])):.2f} p90 {np.percentile(np.abs(S[:,0]),90):.2f} p99 {np.percentile(np.abs(S[:,0]),99):.2f}"
      f" | |y| p50 {np.median(np.abs(S[:,1])):.2f} p90 {np.percentile(np.abs(S[:,1]),90):.2f} p99 {np.percentile(np.abs(S[:,1]),99):.2f}"
      f" | radius p90 {np.percentile(np.hypot(S[:,0],S[:,1]),90):.2f} p99 {np.percentile(np.hypot(S[:,0],S[:,1]),99):.2f}")
print(f"stage scatter of the calibrated points around an ideal lattice [px]: x std {sx_scat.std():.3f} (={sx_scat.std()*PX_UM*1e3:.0f} nm), "
      f"y std {sy_scat.std():.3f} (={sy_scat.std()*PX_UM*1e3:.0f} nm)")
print(f"actual px per nominal 0.05 um step: x median {np.median(np.abs(slope_x)):.3f} [p10 {np.percentile(np.abs(slope_x),10):.3f}, p90 {np.percentile(np.abs(slope_x),90):.3f}],"
      f" y median {np.median(np.abs(slope_y)):.3f} [p10 {np.percentile(np.abs(slope_y),10):.3f}, p90 {np.percentile(np.abs(slope_y),90):.3f}]  (nominal {0.05/PX_UM:.3f})")


def stats(r):
    return f"p50 {np.median(r):.3f}  p90 {np.percentile(r,90):.3f}  p99 {np.percentile(r,99):.3f}  >0.1px {np.mean(r>0.1)*100:4.1f}%  >0.3px {np.mean(r>0.3)*100:4.1f}%"


print(f"\nresidual [px] against the measured frame shifts:")
print(f"  current 11x11 @0.05um, nearest point      : {stats(Rn)}")
print(f"  current grid, 4-point interpolation (hull) : {stats(Rh)}")

# ---- simulated grids with the same stage behaviour ----
# per-Pos slope and scatter are drawn from the measured distributions; shift cloud = measured
n_pos_sim = 60
print(f"\nsimulated grids (stage scatter and per-step gain drawn from the 261112 calibrations; {n_pos_sim} Pos x all frames):")
print(f"  {'grid':>22s}  {'points':>6s}  {'nominal half-range':>18s}  nearest-point residual")
for n, step_um in [(11, 0.05), (15, 0.05), (21, 0.05), (11, 0.03), (15, 0.03), (21, 0.03), (21, 0.025), (31, 0.025)]:
    half = (n - 1) // 2
    res = []
    for _ in range(n_pos_sim):
        gx = rng.choice(np.abs(slope_x)) * (step_um / 0.05); gy = rng.choice(np.abs(slope_y)) * (step_um / 0.05)
        xi, yi = np.meshgrid(np.arange(-half, half + 1), np.arange(-half, half + 1))
        px = gx * yi.ravel() + rng.normal(0, sx_scat.std(), xi.size)
        py = gy * xi.ravel() + rng.normal(0, sy_scat.std(), xi.size)
        d, _ = cKDTree(np.column_stack([px, py])).query(S)
        res.append(d)
    res = np.concatenate(res)
    print(f"  {n:2d}x{n:<2d} @ {step_um:.3f} um  {n*n:6d}  {half*step_um/PX_UM:9.2f} px      {stats(res)}")
print("\nacquisition cost: 0.30 s/point and 25 MB/point (single z, 261112) -> per Pos:"
      + ", ".join(f"{n*n} pts = {n*n*0.30/60:.1f} min / {n*n*25/1e3:.1f} GB" for n in (11, 15, 21, 31)))
