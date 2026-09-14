"""Contour smoothing comparison: Chaikin corner-cutting vs Savitzky-Golay/moving-average.

Loads a cached single-cell crop, builds the medial-axis cross-sections ONCE, then
computes the solid-of-revolution volume on (a) the raw mask contour and (b) a smoothed
contour. The cross-sections are identical for every method; only the contour changes.

Method A: Chaikin corner-cutting (closed-curve variant), 2-4 iterations.
Method B: Savitzky-Golay (and plain moving-average) filter on the contour x,y coords.
Reports whichever is "better" (closest area_ratio to 1 with small max deviation).
"""
import sys
import numpy as np

sys.path.insert(0, "scripts")
from skimage.measure import find_contours
from scipy.signal import savgol_filter
from mask_volume_schematic import medial_axis_geometry

CROP = r"results/260517/_crop_cache/Pos27_ch06_f1800.npz"


# ---------------------------------------------------------------- geometry utils
def shoelace_area(C):
    x, y = C[:, 0], C[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def perimeter(C):
    # C may be open or closed; treat as a polyline that is closed (loop)
    d = np.diff(np.vstack([C, C[:1]]), axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def close_poly(C):
    if not np.allclose(C[0], C[-1]):
        return np.vstack([C, C[:1]])
    return C


def open_poly(C):
    # drop duplicated closing vertex if present
    if len(C) > 1 and np.allclose(C[0], C[-1]):
        return C[:-1]
    return C


def point_to_segments_dist(p, C):
    """Min distance from point p to the polyline edges of closed contour C."""
    A = C[:-1]
    B = C[1:]
    AB = B - A
    AP = p - A
    denom = np.maximum((AB ** 2).sum(axis=1), 1e-12)
    t = np.clip((AP * AB).sum(axis=1) / denom, 0.0, 1.0)
    proj = A + t[:, None] * AB
    return np.sqrt(((p - proj) ** 2).sum(axis=1)).min()


def max_deviation(C_smooth, raw_closed):
    return max(point_to_segments_dist(v, raw_closed) for v in C_smooth)


# ------------------------------------------------------------- volume on contour
def _ray_segment_t(P, d, A, B):
    """Signed parameter t where line P + t*d crosses segment A-B, or None.

    Solve P + t d = A + s (B - A), 0<=s<=1. Returns t (can be negative)."""
    e = B - A
    # [d, -e] [t, s]^T = A - P
    M = np.array([[d[0], -e[0]], [d[1], -e[1]]])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        return None
    rhs = A - P
    t = (rhs[0] * M[1, 1] - rhs[1] * M[0, 1]) / det
    s = (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) / det
    if -1e-9 <= s <= 1.0 + 1e-9:
        return t
    return None


def width_along_perp(P, d, C_closed):
    """Width of closed contour C along the infinite line through P direction d.

    width = nearest positive t intersection - nearest negative t intersection."""
    A = C_closed[:-1]
    B = C_closed[1:]
    t_pos = None
    t_neg = None
    for a, b in zip(A, B):
        t = _ray_segment_t(P, d, a, b)
        if t is None:
            continue
        if t >= 0:
            if t_pos is None or t < t_pos:
                t_pos = t
        else:
            if t_neg is None or t > t_neg:
                t_neg = t
    if t_pos is None or t_neg is None:
        return None
    return abs(t_pos - t_neg)


def volume_on_contour(C_closed, centers, perp, ds, px):
    V = 0.0
    used = 0
    for i in range(len(centers)):
        w = width_along_perp(centers[i], perp[i], C_closed)
        if w is None:
            continue
        V += np.pi * (w / 2.0) ** 2 * ds[i]
        used += 1
    return V * px ** 3, used


# ----------------------------------------------------------------- smoothers
def chaikin(C_open, iterations):
    """Closed-curve Chaikin corner-cutting. C_open: (N,2) without closing dup."""
    P = C_open.copy()
    for _ in range(iterations):
        Pc = np.vstack([P, P[:1]])  # close for wrap-around
        Q = 0.75 * Pc[:-1] + 0.25 * Pc[1:]
        R = 0.25 * Pc[:-1] + 0.75 * Pc[1:]
        # interleave Q_i, R_i for each edge
        new = np.empty((2 * len(P), 2))
        new[0::2] = Q
        new[1::2] = R
        P = new
    return P


def savgol_closed(C_open, window, poly):
    if window % 2 == 0:
        window += 1
    n = len(C_open)
    if window >= n:
        window = n - 1 if (n - 1) % 2 == 1 else n - 2
    window = max(window, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    x = savgol_filter(C_open[:, 0], window, poly, mode="wrap")
    y = savgol_filter(C_open[:, 1], window, poly, mode="wrap")
    return np.column_stack([x, y])


def movavg_closed(C_open, window):
    if window % 2 == 0:
        window += 1
    n = len(C_open)
    pad = window // 2
    xs = np.concatenate([C_open[-pad:, 0], C_open[:, 0], C_open[:pad, 0]])
    ys = np.concatenate([C_open[-pad:, 1], C_open[:, 1], C_open[:pad, 1]])
    k = np.ones(window) / window
    xf = np.convolve(xs, k, mode="valid")
    yf = np.convolve(ys, k, mode="valid")
    return np.column_stack([xf, yf])


# ----------------------------------------------------------------------- main
def main():
    d = np.load(CROP)
    binary = d["binary"].astype(bool)
    px = float(d["px"])

    cs = find_contours(binary.astype(float), 0.5)
    raw = max(cs, key=len)[:, ::-1]          # (N,2) x=col,y=row
    raw_open = open_poly(raw)
    raw_closed = close_poly(raw)

    geo = medial_axis_geometry(binary, pixel_size_um=px)
    centers = geo.medial_xy
    perp = geo.slice_p1_xy - geo.medial_xy
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
    ds = np.linalg.norm(np.gradient(centers, axis=0), axis=1)

    V_raw, used_raw = volume_on_contour(raw_closed, centers, perp, ds, px)
    area_mask = float(binary.sum())
    perim_raw = perimeter(raw_open)

    def evaluate(C_open, label):
        Csm = close_poly(C_open)
        Csm_open = open_poly(Csm)
        V, used = volume_on_contour(Csm, centers, perp, ds, px)
        ar = shoelace_area(Csm_open) / area_mask
        pr = perimeter(Csm_open) / perim_raw
        md = max_deviation(Csm_open, raw_closed)
        vd = 100.0 * (V - V_raw) / V_raw
        return dict(label=label, C=Csm, V=V, area_ratio=ar, perim_ratio=pr,
                    max_dev=md, vol_delta=vd, used=used)

    results = []
    for it in (2, 3, 4):
        results.append(evaluate(chaikin(raw_open, it), f"chaikin_it{it}"))
    n = len(raw_open)
    for win, poly in ((max(5, n // 8) | 1, 2), (max(7, n // 5) | 1, 2), (max(9, n // 4) | 1, 3)):
        results.append(evaluate(savgol_closed(raw_open, win, poly), f"savgol_w{win}_p{poly}"))
    for win in (max(5, n // 8) | 1, max(7, n // 5) | 1):
        results.append(evaluate(movavg_closed(raw_open, win), f"movavg_w{win}"))

    print(f"raw: area_px={area_mask:.0f} perim={perim_raw:.2f} V_raw={V_raw:.4f} um3 "
          f"sections={len(centers)} used_raw={used_raw}")
    for r in results:
        print(f"  {r['label']:22s} area_ratio={r['area_ratio']:.4f} "
              f"perim_ratio={r['perim_ratio']:.4f} max_dev={r['max_dev']:.3f}px "
              f"V={r['V']:.4f} dV%={r['vol_delta']:+.3f} used={r['used']}")

    # "Better" = smoothing (lower perim_ratio = noise removed) while staying faithful:
    # rank by combined faithfulness: prefer area_ratio near 1 AND max_dev small AND
    # perim_ratio < 1 (actually smoothed). Score = |area-1| + 0.5*|dV%|/100 + max_dev/5
    # but require perim_ratio < 0.999 so it is genuinely a smoother.
    def score(r):
        smooth_bonus = 0.0 if r["perim_ratio"] < 0.999 else 1.0
        return (abs(r["area_ratio"] - 1.0) + 0.3 * abs(r["vol_delta"]) / 100.0
                + r["max_dev"] / 10.0 + smooth_bonus)

    best = min(results, key=score)
    print(f"\nBEST: {best['label']}  score-ranked")

    # ---- overlay PNG ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.imshow(binary, cmap="gray", alpha=0.25, origin="upper")
    # section chords in gray
    p0 = geo.slice_p0_xy
    p1 = geo.slice_p1_xy
    for i in range(len(centers)):
        ax.plot([p0[i, 0], p1[i, 0]], [p0[i, 1], p1[i, 1]], color="0.6", lw=0.5, zorder=1)
    ax.plot(raw_closed[:, 0], raw_closed[:, 1], "-", color="cyan", lw=1.6,
            label="raw mask contour", zorder=2)
    bc = best["C"]
    ax.plot(bc[:, 0], bc[:, 1], "-", color="orange", lw=1.6,
            label=f"smoothed ({best['label']})", zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Chaikin/SavGol smoothing  area_ratio={best['area_ratio']:.3f} "
                 f"dV%={best['vol_delta']:+.2f} max_dev={best['max_dev']:.2f}px")
    out = r"results/260517/_smooth_chaikin.png"
    fig.tight_layout()
    fig.savefig(out)
    print(f"\nPNG: {out}")

    import json
    print("RESULT_JSON " + json.dumps(dict(
        method=best["label"], area_ratio=best["area_ratio"],
        perim_ratio=best["perim_ratio"], max_dev_px=best["max_dev"],
        volume_um3=best["V"], volume_delta_pct=best["vol_delta"],
        png=out)))


if __name__ == "__main__":
    main()
