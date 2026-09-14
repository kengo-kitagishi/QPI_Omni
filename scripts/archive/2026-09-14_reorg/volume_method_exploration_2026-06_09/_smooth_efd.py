"""Elliptic Fourier Descriptors (EFD) low-pass smoothing of a single-cell mask
contour, compared against the raw mask contour using the SAME medial-axis
cross-sections for volume.

Method: compute EFD of the closed contour, reconstruct keeping K harmonics
(K in {6,10,15}). Pick the K that removes the staircase while keeping
area_ratio within 1 +/- 0.03. Report K.

Run with PYTHONIOENCODING=utf-8.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from skimage.measure import find_contours

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from mask_volume_schematic import medial_axis_geometry  # noqa: E402


# ---------------------------------------------------------------------------
# Elliptic Fourier Descriptors (standard Kuhl & Giardina 1982 formulation)
# ---------------------------------------------------------------------------

def efd_coeffs(contour: np.ndarray, n_harmonics: int):
    """Compute EFD coefficients for a closed contour (K,2) of (x,y) points.

    Returns (a, b, c, d) each length n_harmonics, plus the DC offsets (A0, C0).
    """
    xy = np.asarray(contour, dtype=float)
    # differences between consecutive points
    dxy = np.diff(xy, axis=0)
    dt = np.sqrt((dxy ** 2).sum(axis=1))
    dt[dt == 0] = 1e-9
    t = np.concatenate([[0.0], np.cumsum(dt)])
    T = t[-1]
    phi = 2.0 * np.pi * t / T

    a = np.zeros(n_harmonics)
    b = np.zeros(n_harmonics)
    c = np.zeros(n_harmonics)
    d = np.zeros(n_harmonics)
    dx = dxy[:, 0]
    dy = dxy[:, 1]
    for n in range(1, n_harmonics + 1):
        const = T / (2.0 * n * n * np.pi * np.pi)
        cos_p = np.cos(n * phi[1:]) - np.cos(n * phi[:-1])
        sin_p = np.sin(n * phi[1:]) - np.sin(n * phi[:-1])
        a[n - 1] = const * np.sum(dx / dt * cos_p)
        b[n - 1] = const * np.sum(dx / dt * sin_p)
        c[n - 1] = const * np.sum(dy / dt * cos_p)
        d[n - 1] = const * np.sum(dy / dt * sin_p)

    # DC components (centroid offset of the reconstruction)
    xi = np.cumsum(dx) - (dx / dt) * t[1:]
    A0 = (1.0 / T) * np.sum((dx / (2 * dt)) * (t[1:] ** 2 - t[:-1] ** 2)
                            + xi * (t[1:] - t[:-1]))
    delta = np.cumsum(dy) - (dy / dt) * t[1:]
    C0 = (1.0 / T) * np.sum((dy / (2 * dt)) * (t[1:] ** 2 - t[:-1] ** 2)
                            + delta * (t[1:] - t[:-1]))
    A0 += xy[0, 0]
    C0 += xy[0, 1]
    return a, b, c, d, A0, C0


def efd_reconstruct(a, b, c, d, A0, C0, n_points=400):
    """Reconstruct a closed contour from EFD coefficients."""
    K = len(a)
    phi = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=True)
    x = np.full(n_points, A0)
    y = np.full(n_points, C0)
    for n in range(1, K + 1):
        x += a[n - 1] * np.cos(n * phi) + b[n - 1] * np.sin(n * phi)
        y += c[n - 1] * np.cos(n * phi) + d[n - 1] * np.sin(n * phi)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def shoelace_area(C: np.ndarray) -> float:
    x = C[:, 0]
    y = C[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def perimeter(C: np.ndarray) -> float:
    d = np.diff(C, axis=0)
    return float(np.sqrt((d ** 2).sum(axis=1)).sum())


def point_to_polyline_dist(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Min distance from each point in pts to the polyline poly (closed)."""
    a = poly[:-1]
    b = poly[1:]
    ab = b - a
    ab2 = (ab ** 2).sum(axis=1)
    ab2[ab2 == 0] = 1e-12
    out = np.empty(len(pts))
    for i, p in enumerate(pts):
        ap = p - a
        tparam = np.clip((ap * ab).sum(axis=1) / ab2, 0.0, 1.0)
        proj = a + tparam[:, None] * ab
        out[i] = np.sqrt(((p - proj) ** 2).sum(axis=1)).min()
    return out


# ---------------------------------------------------------------------------
# Volume on an arbitrary contour using the SAME medial-axis cross-sections
# ---------------------------------------------------------------------------

def seg_intersections_t(P, dvec, A, B):
    """For segment A->B, return parameter t along line P+t*dvec at the
    intersection (or None). Solve P + t*d = A + s*(B-A), 0<=s<=1."""
    e = B - A
    M = np.array([[dvec[0], -e[0]], [dvec[1], -e[1]]])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-12:
        return None
    rhs = A - P
    t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / det
    s = (M[0, 0] * rhs[1] - M[1, 0] * rhs[0]) / det
    if -1e-9 <= s <= 1 + 1e-9:
        return t
    return None


def volume_on_contour(C, centers, perp, ds, px):
    """Volume via solid-of-revolution using fixed medial cross-sections.

    For each (P=center, d=perp) find line P+t*d intersections with edges of C
    straddling t=0 (nearest positive t, nearest negative t); width=|tpos-tneg|.
    V = sum pi*(width/2)^2 * ds[i] * px^3.
    """
    Cc = C if np.allclose(C[0], C[-1]) else np.vstack([C, C[0]])
    A = Cc[:-1]
    B = Cc[1:]
    V = 0.0
    for i in range(len(centers)):
        P = centers[i]
        dvec = perp[i]
        tpos = np.inf
        tneg = -np.inf
        for j in range(len(A)):
            t = seg_intersections_t(P, dvec, A[j], B[j])
            if t is None:
                continue
            if t > 1e-9 and t < tpos:
                tpos = t
            if t < -1e-9 and t > tneg:
                tneg = t
        if not np.isfinite(tpos) or not np.isfinite(tneg):
            continue
        width = abs(tpos - tneg)
        V += np.pi * (width / 2.0) ** 2 * ds[i]
    return V * px ** 3


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    d = np.load(str(REPO / "results/260517/_crop_cache/Pos27_ch06_f1800.npz"))
    binary = d["binary"].astype(bool)
    px = float(d["px"])

    cs = find_contours(binary.astype(float), 0.5)
    raw = max(cs, key=len)[:, ::-1]  # (N,2) x=col,y=row
    raw_closed = np.vstack([raw, raw[0]])

    # fixed medial cross-sections (same for all methods)
    geo = medial_axis_geometry(binary, pixel_size_um=px)
    centers = geo.medial_xy
    perp = geo.slice_p1_xy - geo.medial_xy
    perp /= np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
    ds = np.linalg.norm(np.gradient(centers, axis=0), axis=1)

    V_raw = volume_on_contour(raw_closed, centers, perp, ds, px)

    mask_area = binary.sum()
    perim_raw = perimeter(raw_closed)

    results = {}
    for K in (6, 10, 15):
        a, b, c_, dd, A0, C0 = efd_coeffs(raw_closed, K)
        rec = efd_reconstruct(a, b, c_, dd, A0, C0, n_points=400)
        rec_closed = np.vstack([rec, rec[0]])
        area_ratio = shoelace_area(rec_closed) / mask_area
        perim_ratio = perimeter(rec_closed) / perim_raw
        max_dev = float(point_to_polyline_dist(rec, raw_closed).max())
        V_s = volume_on_contour(rec_closed, centers, perp, ds, px)
        vdelta = 100.0 * (V_s - V_raw) / V_raw
        results[K] = dict(area_ratio=area_ratio, perim_ratio=perim_ratio,
                          max_dev=max_dev, V_s=V_s, vdelta=vdelta, rec=rec)
        print(f"K={K:2d}  area_ratio={area_ratio:.4f}  perim_ratio={perim_ratio:.4f}  "
              f"max_dev={max_dev:.3f}px  V={V_s:.3f}um3  dV={vdelta:+.2f}%")

    # Pick K: smallest K that removes staircase (lowest perim_ratio, i.e. most
    # smoothing) while keeping area within 1 +/- 0.03. Among admissible K,
    # prefer the smallest (lowest harmonics = strongest staircase removal).
    admissible = [K for K in (6, 10, 15) if abs(results[K]["area_ratio"] - 1.0) <= 0.03]
    if admissible:
        chosen = min(admissible)
    else:
        # none in band: pick the one closest to area_ratio 1
        chosen = min((6, 10, 15), key=lambda K: abs(results[K]["area_ratio"] - 1.0))
    print(f"\nadmissible (|area_ratio-1|<=0.03): {admissible}  -> chosen K={chosen}")

    r = results[chosen]
    rec = r["rec"]

    # ---- overlay PNG ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    # gray section chords
    for i in range(len(centers)):
        p0 = geo.slice_p0_xy[i]
        p1 = geo.slice_p1_xy[i]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="0.7", lw=0.6, zorder=1)
    ax.plot(raw_closed[:, 0], raw_closed[:, 1], color="cyan", lw=2.0,
            label="raw mask contour", zorder=2)
    rec_closed = np.vstack([rec, rec[0]])
    ax.plot(rec_closed[:, 0], rec_closed[:, 1], color="orange", lw=1.8,
            label=f"EFD K={chosen}", zorder=3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.axis("off")
    ax.set_title(f"EFD low-pass K={chosen}  "
                 f"area_ratio={r['area_ratio']:.3f}  V={r['V_s']:.3f}um3 "
                 f"(dV={r['vdelta']:+.2f}%)", fontsize=9)
    ax.legend(loc="upper right", fontsize=7)
    out_png = REPO / "results/260517/_smooth_efd.png"
    fig.tight_layout()
    fig.savefig(str(out_png), dpi=150, bbox_inches="tight")
    print(f"\npng: {out_png}")

    # final structured-ish summary line
    print("\nRESULT_JSON " + repr(dict(
        method=f"EFD low-pass K={chosen}",
        params=f"K_candidates=[6,10,15], chosen={chosen}, n_points=400, "
               f"area_band=1+/-0.03",
        area_ratio=r["area_ratio"],
        perim_ratio=r["perim_ratio"],
        max_dev_px=r["max_dev"],
        volume_um3=r["V_s"],
        volume_delta_pct=r["vdelta"],
        png_path=str(out_png),
        V_raw=V_raw,
    )))


if __name__ == "__main__":
    main()
