"""Baseline (RAW contour, NO smoothing) volume + metrics for the smoothing comparison.

The "smoothed" contour IS the raw find_contours polyline, so by definition:
  area_ratio   = shoelace(raw)/binary.sum()
  perim_ratio  = 1.0
  max_dev_px   = 0.0
  volume_delta = 0.0
We still compute volume_um3 on the raw contour using the SHARED cross-sections.
"""
import sys
import numpy as np

sys.path.insert(0, "scripts")
from skimage.measure import find_contours
from mask_volume_schematic import medial_axis_geometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- shared loading ----------
d = np.load(r"results/260517/_crop_cache/Pos27_ch06_f1800.npz")
binary = d["binary"].astype(bool)
px = float(d["px"])

cs = find_contours(binary.astype(float), 0.5)
raw = max(cs, key=len)[:, ::-1]  # (N,2) x=col,y=row


def close_poly(C):
    C = np.asarray(C, float)
    if not np.allclose(C[0], C[-1]):
        C = np.vstack([C, C[0]])
    return C


raw_closed = close_poly(raw)

# ---------- shared cross-sections ----------
geo = medial_axis_geometry(binary, pixel_size_um=px)
centers = geo.medial_xy  # (M,2)
perp = geo.slice_p1_xy - geo.medial_xy
perp = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
ds = np.linalg.norm(np.gradient(centers, axis=0), axis=1)


# ---------- geometry helpers ----------
def shoelace_area(C):
    C = np.asarray(C, float)
    x, y = C[:, 0], C[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def perimeter(C):
    C = np.asarray(C, float)
    return float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))


def seg_param_intersections(P, dvec, C):
    """Return signed t values where the infinite line P+t*dvec crosses edges of
    closed polygon C. dvec is a unit vector."""
    C = np.asarray(C, float)
    A = C[:-1]
    B = C[1:]
    n = np.array([-dvec[1], dvec[0]])  # normal to the line direction
    # signed distance of each vertex from the line
    fA = (A - P) @ n
    fB = (B - P) @ n
    ts = []
    straddle = (fA * fB < 0) | (np.isclose(fA, 0.0) & ~np.isclose(fB, 0.0))
    idx = np.where(straddle)[0]
    for i in idx:
        a, b = fA[i], fB[i]
        denom = (a - b)
        if abs(denom) < 1e-12:
            continue
        u = a / denom  # fraction along edge A->B where line crosses (f=0)
        if u < 0.0 or u > 1.0:
            continue
        X = A[i] + u * (B[i] - A[i])
        t = (X - P) @ dvec
        ts.append(t)
    return np.array(ts)


def volume_on_contour(C):
    C = close_poly(C)
    V = 0.0
    for i in range(len(centers)):
        P = centers[i]
        dvec = perp[i]
        ts = seg_param_intersections(P, dvec, C)
        if ts.size == 0:
            continue
        pos = ts[ts > 0]
        neg = ts[ts < 0]
        if pos.size == 0 or neg.size == 0:
            continue
        t_pos = pos.min()
        t_neg = neg.max()  # nearest negative (closest to 0)
        width = abs(t_pos - t_neg)
        V += np.pi * (width / 2.0) ** 2 * ds[i]
    return V * px ** 3


def max_dev_to_polyline(verts, poly):
    """Max distance from each vertex in verts to the polyline poly."""
    poly = np.asarray(poly, float)
    A = poly[:-1]
    B = poly[1:]
    AB = B - A
    denom = np.sum(AB * AB, axis=1)
    denom[denom == 0] = 1e-12
    md = 0.0
    for v in verts:
        AP = v - A
        u = np.clip(np.sum(AP * AB, axis=1) / denom, 0.0, 1.0)
        proj = A + u[:, None] * AB
        dist = np.min(np.linalg.norm(proj - v, axis=1))
        if dist > md:
            md = dist
    return float(md)


# ---------- baseline: smoothed == raw ----------
C_smooth = raw_closed

V_raw = volume_on_contour(raw_closed)
V_smooth = volume_on_contour(C_smooth)

area_ratio = shoelace_area(C_smooth) / binary.sum()
perim_ratio = perimeter(C_smooth) / perimeter(raw_closed)
max_dev_px = max_dev_to_polyline(C_smooth, raw_closed)
volume_um3 = V_smooth
volume_delta_pct = 100.0 * (V_smooth - V_raw) / V_raw

# ---------- overlay PNG ----------
png_path = "results/260517/_smooth_raw.png"
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(binary, cmap="gray", alpha=0.25)
# section chords (gray)
for i in range(0, len(centers)):
    P = centers[i]
    dvec = perp[i]
    ts = seg_param_intersections(P, dvec, raw_closed)
    pos = ts[ts > 0]
    neg = ts[ts < 0]
    if pos.size == 0 or neg.size == 0:
        continue
    a = P + pos.min() * dvec
    b = P + neg.max() * dvec
    ax.plot([a[0], b[0]], [a[1], b[1]], color="0.6", lw=0.5, zorder=1)
ax.plot(raw_closed[:, 0], raw_closed[:, 1], color="cyan", lw=1.5, label="raw mask contour", zorder=2)
ax.plot(C_smooth[:, 0], C_smooth[:, 1], color="orange", lw=1.0, ls="--",
        label="smoothed (=raw, baseline)", zorder=3)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.legend(loc="upper right", fontsize=8)
ax.set_title("RAW baseline (no smoothing)")
fig.tight_layout()
fig.savefig(png_path, dpi=130)
plt.close(fig)

print("RESULT_AREA_RATIO", area_ratio)
print("RESULT_PERIM_RATIO", perim_ratio)
print("RESULT_MAX_DEV_PX", max_dev_px)
print("RESULT_VOLUME_UM3", volume_um3)
print("RESULT_VOLUME_DELTA_PCT", volume_delta_pct)
print("RESULT_V_RAW", V_raw)
print("RESULT_PNG", png_path)
print("RESULT_NCENTERS", len(centers))
print("RESULT_BINARY_SUM", int(binary.sum()))
print("RESULT_PX", px)
