"""Periodic cubic smoothing spline contour smoothing + volume comparison.

Method: scipy.interpolate.splprep(per=1) periodic cubic spline.
Sweep s in {0.5,1,2,4}*N; pick smallest s that removes pixel staircase while
keeping area_ratio in 1+-0.03 and max_dev_px <= ~1px. Resample ~400 points.
"""
import sys
import os
import numpy as np

sys.path.insert(0, "scripts")
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
from mask_volume_schematic import medial_axis_geometry


# ---------------------------------------------------------------- load crop
d = np.load(r"results/260517/_crop_cache/Pos27_ch06_f1800.npz")
binary = d["binary"].astype(bool)
px = float(d["px"])

cs = find_contours(binary.astype(float), 0.5)
raw = max(cs, key=len)[:, ::-1]  # (N,2) x=col, y=row
# close raw polyline
raw_closed = np.vstack([raw, raw[0]])


# ---------------------------------------------------------------- geometry / cross sections
geo = medial_axis_geometry(binary, pixel_size_um=px)
centers = geo.medial_xy                                     # (M,2)
perp = geo.slice_p1_xy - geo.medial_xy
perp = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
ds = np.linalg.norm(np.gradient(centers, axis=0), axis=1)   # px arc step per section


# ---------------------------------------------------------------- helpers
def shoelace_area(C):
    """Polygon area (px^2) for closed or open ring; uses unique vertices."""
    x = C[:, 0]; y = C[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def perimeter(C):
    """Perimeter of a polyline (assumed already closed by repeating first pt,
    or open -- we close it)."""
    P = C
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


def _seg_intersections_t(P, dvec, C):
    """Signed parameters t where infinite line P + t*dvec crosses each edge of
    closed polygon C. Returns array of t values."""
    A = C[:-1]            # edge starts (K,2)  (C is closed: last==first)
    B = C[1:]             # edge ends
    e = B - A             # edge vectors
    # Solve P + t*d = A + u*e  ->  [d, -e] [t,u]^T = A - P
    # 2x2 system per edge.  d fixed, e varies.
    dx, dy = dvec
    ex = e[:, 0]; ey = e[:, 1]
    det = (-dx * ey + dy * ex)                # det of [[dx,-ex],[dy,-ey]]
    rhs = A - P                               # (K,2)
    rx = rhs[:, 0]; ry = rhs[:, 1]
    good = np.abs(det) > 1e-12
    t = np.full(len(e), np.nan)
    u = np.full(len(e), np.nan)
    # t = (rx*(-ey) - (-ex)*ry)/det ; u = (dx*ry - dy*rx)/det
    t[good] = (rx[good] * (-ey[good]) - (-ex[good]) * ry[good]) / det[good]
    u[good] = (dx * ry[good] - dy * rx[good]) / det[good]
    on_edge = good & (u >= -1e-9) & (u <= 1 + 1e-9)
    return t[on_edge]


def section_width_px(P, dvec, C):
    """Width = |t_pos - t_neg| using nearest positive and nearest negative
    straddling intersections of the infinite line with closed polygon C.
    Returns None if no straddling pair."""
    t = _seg_intersections_t(P, dvec, C)
    if t.size == 0:
        return None
    pos = t[t > 1e-9]
    neg = t[t < -1e-9]
    if pos.size == 0 or neg.size == 0:
        return None
    t_pos = pos.min()
    t_neg = neg.max()   # nearest negative = largest (closest to 0)
    return abs(t_pos - t_neg)


def volume_um3(C_closed):
    """V = sum( pi*(w/2)^2 * ds ) * px^3 over sections; skip non-straddling."""
    V_px = 0.0
    used = 0
    for i in range(len(centers)):
        w = section_width_px(centers[i], perp[i], C_closed)
        if w is None:
            continue
        V_px += np.pi * (w / 2.0) ** 2 * ds[i]
        used += 1
    return V_px * px ** 3, used


def dist_point_to_polyline(pt, poly):
    """Min distance from pt to polyline poly (closed, (K,2))."""
    A = poly[:-1]; B = poly[1:]
    AB = B - A
    AP = pt - A
    denom = np.einsum('ij,ij->i', AB, AB)
    tt = np.where(denom > 1e-12, np.einsum('ij,ij->i', AP, AB) / np.maximum(denom, 1e-12), 0.0)
    tt = np.clip(tt, 0.0, 1.0)
    proj = A + tt[:, None] * AB
    dd = np.linalg.norm(pt - proj, axis=1)
    return dd.min()


# ---------------------------------------------------------------- raw volume
V_raw, used_raw = volume_um3(raw_closed)
binary_area = binary.sum()
raw_perim = perimeter(raw)


# ---------------------------------------------------------------- spline sweep
# splprep wants list of coord arrays; periodic -> first==last not required but
# the input should describe one loop. Use raw (open) points; per=1 closes it.
N = len(raw)
x = raw[:, 0]; y = raw[:, 1]

s_factors = [0.5, 1.0, 2.0, 4.0]
candidates = []
for fac in s_factors:
    s_val = fac * N
    try:
        tck, u = splprep([x, y], s=s_val, per=1, k=3)
    except Exception as ex:
        candidates.append({"fac": fac, "s": s_val, "error": str(ex)})
        continue
    un = np.linspace(0.0, 1.0, 401)
    xs, ys = splev(un, tck)
    C = np.column_stack([xs, ys])           # (401,2), last~=first (periodic)
    # closed ring for metrics
    C_closed = C.copy()
    if not np.allclose(C_closed[0], C_closed[-1]):
        C_closed = np.vstack([C_closed, C_closed[0]])
    area_ratio = shoelace_area(C) / binary_area
    perim_ratio = perimeter(C) / raw_perim
    # use unique vertices (drop duplicate closing pt) for max dev
    Cu = C[:-1] if np.allclose(C[0], C[-1]) else C
    max_dev = max(dist_point_to_polyline(p, raw_closed) for p in Cu)
    candidates.append({
        "fac": fac, "s": s_val, "tck": tck, "C": C, "C_closed": C_closed,
        "area_ratio": float(area_ratio), "perim_ratio": float(perim_ratio),
        "max_dev_px": float(max_dev),
    })

# choose smallest s meeting: |area_ratio-1|<=0.03 and max_dev_px<=1.0
# (smaller s = less smoothing = follows staircase more; larger s = smoother.
#  We want smallest s that removes staircase -> i.e. perim_ratio drops enough
#  while staying within area/dev bounds. We pick smallest s passing constraints;
#  if perim_ratio still near 1 -> staircase not removed. So additionally require
#  perim_ratio <= 0.97 as "visibly removes staircase".)
valid = [c for c in candidates if "error" not in c
         and abs(c["area_ratio"] - 1.0) <= 0.03
         and c["max_dev_px"] <= 1.05]
chosen = None
# prefer those that also remove staircase (perim_ratio <= 0.97), smallest s
removed = [c for c in valid if c["perim_ratio"] <= 0.97]
pool = removed if removed else valid
if pool:
    chosen = min(pool, key=lambda c: c["s"])
elif [c for c in candidates if "error" not in c]:
    # fallback: smallest s overall among runnable
    chosen = min([c for c in candidates if "error" not in c], key=lambda c: c["s"])

print("=== candidate sweep ===")
for c in candidates:
    if "error" in c:
        print(f"  fac={c['fac']} s={c['s']:.1f} ERROR {c['error']}")
    else:
        print(f"  fac={c['fac']} s={c['s']:.1f} area_ratio={c['area_ratio']:.4f} "
              f"perim_ratio={c['perim_ratio']:.4f} max_dev_px={c['max_dev_px']:.3f}")
print(f"chosen fac={chosen['fac']} s={chosen['s']:.1f}")

C = chosen["C"]
C_closed = chosen["C_closed"]
area_ratio = chosen["area_ratio"]
perim_ratio = chosen["perim_ratio"]
max_dev = chosen["max_dev_px"]

V_smooth, used_sm = volume_um3(C_closed)
vol_delta_pct = 100.0 * (V_smooth - V_raw) / V_raw

print(f"V_raw    = {V_raw:.4f} um^3  (sections used {used_raw}/{len(centers)})")
print(f"V_smooth = {V_smooth:.4f} um^3  (sections used {used_sm}/{len(centers)})")
print(f"vol_delta_pct = {vol_delta_pct:.4f}")
print(f"area_ratio={area_ratio:.4f} perim_ratio={perim_ratio:.4f} max_dev_px={max_dev:.4f}")


# ---------------------------------------------------------------- overlay PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(6, 6))
# section chords (gray)
p0 = geo.slice_p0_xy
p1 = geo.slice_p1_xy
for i in range(len(centers)):
    ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]],
            color="0.7", lw=0.5, zorder=1)
# raw contour cyan
ax.plot(raw_closed[:, 0], raw_closed[:, 1], color="cyan", lw=1.4,
        label="raw mask contour", zorder=2)
# smoothed orange
ax.plot(C[:, 0], C[:, 1], color="orange", lw=1.6,
        label=f"cubic spline (s={chosen['fac']:.1f}N)", zorder=3)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend(loc="best", fontsize=8)
ax.set_title(f"Periodic cubic smoothing spline  "
             f"area={area_ratio:.3f} dev={max_dev:.2f}px dV={vol_delta_pct:+.2f}%")
out_png = os.path.abspath("results/260517/_smooth_cubic_spline_periodic.png")
os.makedirs(os.path.dirname(out_png), exist_ok=True)
fig.savefig(out_png, dpi=130, bbox_inches="tight")
plt.close(fig)
print("PNG:", out_png)

# ---------------------------------------------------------------- emit machine line
import json
print("RESULT_JSON=" + json.dumps({
    "method": "periodic cubic smoothing spline (scipy splprep per=1, k=3, ~400 pts)",
    "params": f"s={chosen['fac']:.1f}*N (N={N}, s={chosen['s']:.1f}); k=3; per=1; resample=400",
    "ran_ok": True,
    "area_ratio": area_ratio,
    "perim_ratio": perim_ratio,
    "max_dev_px": max_dev,
    "volume_um3": V_smooth,
    "volume_delta_pct": vol_delta_pct,
    "png_path": out_png,
}))
