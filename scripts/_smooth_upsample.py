"""Contour-smoothing benchmark: IMAGE UPSAMPLE then contour.

Method: upsample the binary mask by factor f, threshold at 0.5, find_contours
at 0.5, divide coords back by f. The fixed medial-axis cross-sections (from
mask_volume_schematic.medial_axis_geometry) are reused for every method so only
the contour changes when computing volume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.measure import find_contours
from skimage.transform import rescale

sys.path.insert(0, "scripts")
from mask_volume_schematic import medial_axis_geometry  # noqa: E402


# ----------------------------------------------------------------------------
# shared helpers
# ----------------------------------------------------------------------------
def close_poly(C):
    """Append first vertex if the polyline is not already closed."""
    C = np.asarray(C, float)
    if not np.allclose(C[0], C[-1]):
        C = np.vstack([C, C[0]])
    return C


def shoelace_area(C):
    C = close_poly(C)
    x, y = C[:, 0], C[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def perimeter(C):
    C = close_poly(C)
    return float(np.sum(np.linalg.norm(np.diff(C, axis=0), axis=1)))


def _seg_point_dist(px, py, ax, ay, bx, by):
    """distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    t = 0.0 if vv == 0 else (wx * vx + wy * vy) / vv
    t = min(1.0, max(0.0, t))
    cx, cy = ax + t * vx, ay + t * vy
    return np.hypot(px - cx, py - cy)


def max_dev_to_polyline(C_smooth, raw):
    """max over C_smooth vertices of distance to the raw contour polyline."""
    raw = close_poly(raw)
    a = raw[:-1]
    b = raw[1:]
    md = 0.0
    for px, py in C_smooth:
        d = min(_seg_point_dist(px, py, ax, ay, bx, by)
                for (ax, ay), (bx, by) in zip(a, b))
        if d > md:
            md = d
    return float(md)


def line_poly_width(P, d, C):
    """Width of contour C along the infinite line P + t*d.

    Find intersections of the line with the edges of C straddling t=0 (nearest
    positive t and nearest negative t); width = |t_pos - t_neg|. Returns None if
    no straddling pair.
    """
    C = close_poly(C)
    ax, ay = d[0], d[1]            # line direction
    px, py = P[0], P[1]
    t_pos = None
    t_neg = None
    A = C[:-1]
    B = C[1:]
    for (x1, y1), (x2, y2) in zip(A, B):
        # edge: Q = E1 + s*(E2-E1), s in [0,1]
        ex, ey = x2 - x1, y2 - y1
        # solve P + t*d = E1 + s*e  ->  t*d - s*e = E1 - P
        det = ax * (-ey) - ay * (-ex)
        if abs(det) < 1e-12:
            continue
        rhs_x = x1 - px
        rhs_y = y1 - py
        # [ d  -e ] [t s]^T = rhs  ; Cramer
        t = (rhs_x * (-ey) - rhs_y * (-ex)) / det
        s = (ax * rhs_y - ay * rhs_x) / det
        if -1e-9 <= s <= 1 + 1e-9:
            if t >= 0:
                if t_pos is None or t < t_pos:
                    t_pos = t
            if t <= 0:
                if t_neg is None or t > t_neg:
                    t_neg = t
    if t_pos is None or t_neg is None:
        return None
    return abs(t_pos - t_neg)


def volume_on_contour(C, centers, perp, ds, px):
    """V = sum( pi*(w/2)^2 * ds ) * px^3 using the SAME cross-sections."""
    C = close_poly(C)
    V = 0.0
    for P, d, dstep in zip(centers, perp, ds):
        w = line_poly_width(P, d, C)
        if w is None:
            continue
        V += np.pi * (w / 2.0) ** 2 * dstep
    return float(V * px ** 3)


# ----------------------------------------------------------------------------
# upsample-then-contour smoothing
# ----------------------------------------------------------------------------
def upsample_contour(binary, f, order=1):
    up = rescale(binary.astype(float), f, order=order, anti_aliasing=True,
                 preserve_range=True)
    cs = find_contours(up, 0.5)
    c = max(cs, key=len)[:, ::-1]      # (N,2) x=col,y=row in upsampled coords
    return c / float(f)                # back to original px units


def main():
    d = np.load(r"results/260517/_crop_cache/Pos27_ch06_f1800.npz")
    binary = d["binary"].astype(bool)
    px = float(d["px"])

    # raw contour
    cs = find_contours(binary.astype(float), 0.5)
    raw = max(cs, key=len)[:, ::-1]
    raw = close_poly(raw)

    # fixed cross-sections (shared across all methods)
    geo = medial_axis_geometry(binary, pixel_size_um=px)
    centers = geo.medial_xy
    perp = geo.slice_p1_xy - geo.medial_xy
    perp = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
    ds = np.linalg.norm(np.gradient(centers, axis=0), axis=1)

    V_raw = volume_on_contour(raw, centers, perp, ds, px)

    # evaluate candidate factors / orders
    candidates = []
    for order in (1, 3):
        for f in (2, 4):
            C = close_poly(upsample_contour(binary, f, order=order))
            ar = shoelace_area(C) / binary.sum()
            pr = perimeter(C) / perimeter(raw)
            md = max_dev_to_polyline(C, raw)
            V = volume_on_contour(C, centers, perp, ds, px)
            candidates.append(dict(order=order, f=f, C=C, area_ratio=ar,
                                   perim_ratio=pr, max_dev_px=md,
                                   volume_um3=V,
                                   volume_delta_pct=100 * (V - V_raw) / V_raw))
            print(f"order={order} f={f}: area_ratio={ar:.4f} "
                  f"perim_ratio={pr:.4f} max_dev_px={md:.4f} "
                  f"V={V:.4f} dV%={100*(V-V_raw)/V_raw:+.3f}")

    # Pick f that best balances smoothness vs area preservation.
    # "Smoothness" = the contour should be no rougher than raw: reward
    # perim_ratio <= 1 (true polygon smoothing reduces perimeter); penalize
    # perim_ratio > 1 (anti-aliased staircase ringing at high f). "Area
    # preservation" = area_ratio near 1 and small max deviation from raw.
    def score(c):
        rough = max(c["perim_ratio"] - 1.0, 0.0)            # only penalize roughening
        smooth_gain = max(1.0 - c["perim_ratio"], 0.0)      # reward genuine smoothing
        return (abs(c["area_ratio"] - 1.0)
                + 2.0 * rough
                + 0.05 * c["max_dev_px"]
                - 0.3 * smooth_gain)
    best = min(candidates, key=score)
    print(f"\nSELECTED order={best['order']} f={best['f']} (score={score(best):.4f})")

    C = best["C"]

    # overlay PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for P, d, dstep in zip(centers, perp, ds):
        w = line_poly_width(P, d, C)
        if w is None:
            continue
        a = P - d * (w / 2.0)
        b = P + d * (w / 2.0)
        ax.plot([a[0], b[0]], [a[1], b[1]], color="0.6", lw=0.6, zorder=1)
    ax.plot(raw[:, 0], raw[:, 1], color="cyan", lw=1.8, zorder=2,
            label="raw mask contour")
    ax.plot(C[:, 0], C[:, 1], color="orange", lw=1.6, zorder=3,
            label=f"upsample f={best['f']} order={best['order']}")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(f"upsample-then-contour  V={best['volume_um3']:.3f} um^3 "
                 f"(dV {best['volume_delta_pct']:+.2f}%)", fontsize=9)
    out = Path("results/260517/_smooth_upsample.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    print(f"PNG: {out.resolve()}")

    print("\nRESULT_JSON", {
        "method": f"image_upsample_then_contour (rescale order={best['order']}, "
                  f"anti_aliasing, threshold 0.5)",
        "params": f"f={best['f']}, order={best['order']}, anti_aliasing=True",
        "area_ratio": best["area_ratio"],
        "perim_ratio": best["perim_ratio"],
        "max_dev_px": best["max_dev_px"],
        "volume_um3": best["volume_um3"],
        "volume_delta_pct": best["volume_delta_pct"],
        "V_raw": V_raw,
        "png": str(out.resolve()),
    })


if __name__ == "__main__":
    main()
