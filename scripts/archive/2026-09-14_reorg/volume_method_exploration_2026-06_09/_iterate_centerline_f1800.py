"""Iterate the centerline update (full Odermatt-style) on frame 1800.

Seed = stable cos-theta centerline. Each iteration: draw lines perpendicular to
the CURRENT centerline's local tangent, intersect the contour, set the new
centerline to the chord midpoints and the radius to half the chord. Repeat.
Report volume + mean centerline shift per iteration (convergence), on the
EFD-smoothed contour and (for contrast) the raw staircase contour.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_volume_schematic import load_mother_crop, medial_axis_geometry


def close(c):
    return c if np.allclose(c[0], c[-1]) else np.vstack([c, c[:1]])


def fourier_smooth(contour_xy, K=6, M=512):
    c = close(contour_xy)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    tu = np.linspace(0, t[-1], M, endpoint=False)
    Z = np.fft.fft(np.interp(tu, t, c[:, 0]) + 1j * np.interp(tu, t, c[:, 1]))
    keep = np.zeros(M, bool); keep[0] = True; keep[1:K + 1] = True; keep[-K:] = True
    Z[~keep] = 0
    zs = np.fft.ifft(Z)
    return close(np.column_stack([zs.real, zs.imag]))


def chord_to_contour(contour, P, d):
    a = contour[:-1]; b = contour[1:]
    e = b - a; r = a - P
    det = -d[0] * e[:, 1] + e[:, 0] * d[1]
    ok = np.abs(det) > 1e-9
    t = np.full(len(e), np.nan); s = np.full(len(e), np.nan)
    t[ok] = (-r[ok, 0] * e[ok, 1] + e[ok, 0] * r[ok, 1]) / det[ok]
    s[ok] = (d[0] * r[ok, 1] - d[1] * r[ok, 0]) / det[ok]
    val = ok & (s >= 0) & (s <= 1)
    tv = t[val]
    pos = tv[tv > 1e-6]; neg = tv[tv < -1e-6]
    if pos.size == 0 or neg.size == 0:
        return None
    return P + neg.max() * d, P + pos.min() * d


def perp_from_tangent(centers):
    tan = np.gradient(centers, axis=0)
    tan = tan / np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    return np.column_stack([-tan[:, 1], tan[:, 0]])     # rotate +90 deg


def iterate(contour, centers0, perp0, px, n_iter=4):
    centers = centers0.copy()
    perp = perp0.copy()
    history = []
    for k in range(n_iter + 1):
        new_c = centers.copy()
        w = np.zeros(len(centers))
        used = 0
        for i, (P, d) in enumerate(zip(centers, perp)):
            c = chord_to_contour(contour, P, d)
            if c is None:
                continue
            a, b = c
            new_c[i] = (a + b) / 2
            w[i] = np.linalg.norm(b - a)
            used += 1
        ds = np.linalg.norm(np.gradient(new_c, axis=0), axis=1)
        V = float(np.sum(np.pi * (w / 2) ** 2 * ds)) * px ** 3
        shift = float(np.mean(np.linalg.norm(new_c - centers, axis=1)))
        history.append(dict(iter=k, centers=new_c.copy(), w=w.copy(), V=V,
                            used=used, shift=shift))
        centers = new_c
        perp = perp_from_tangent(centers)
    return history


def main():
    binary, frame, px = load_mother_crop("Pos27", "ch06", 1800)
    H, W = binary.shape
    geo = medial_axis_geometry(binary, pixel_size_um=px)
    centers0 = geo.medial_xy
    perp0 = geo.slice_p1_xy - geo.medial_xy
    perp0 = perp0 / np.maximum(np.linalg.norm(perp0, axis=1, keepdims=True), 1e-9)

    raw = close(max(geo.contour_xy, key=len))
    efd = fourier_smooth(raw, K=6)

    print(f"frame {frame}  cos-theta ref V = {geo.volume_um3:.3f} um3\n")
    for name, contour in [("RAW", raw), ("EFD K=6", efd)]:
        hist = iterate(contour, centers0, perp0, px, n_iter=4)
        print(f"=== {name} contour ===")
        print(f"{'iter':>4} {'V(um3)':>9} {'used':>6} {'mean_shift(px)':>14}")
        for h in hist:
            print(f"{h['iter']:>4} {h['V']:9.3f} {h['used']:>6} {h['shift']:14.3f}")
        print()
        if name == "EFD K=6":
            efd_hist = hist

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(efd[:, 0], efd[:, 1], color="deepskyblue", lw=2.2, zorder=2)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(efd_hist)))
    for h, col in zip(efd_hist, cmap):
        ax.plot(h["centers"][:, 0], h["centers"][:, 1], color=col, lw=1.5,
                label=f"iter {h['iter']}  V={h['V']:.2f}", zorder=3)
    # final chords
    last = efd_hist[-1]
    perp = perp_from_tangent(last["centers"])
    for i, (P, d) in enumerate(zip(last["centers"], perp)):
        c = chord_to_contour(efd, P, d)
        if c is not None:
            ax.plot([c[0][0], c[1][0]], [c[0][1], c[1][1]], color="0.7", lw=0.7, zorder=1)
    ax.set_aspect("equal"); ax.set(xlim=(-2, W + 2), ylim=(H + 2, -2)); ax.axis("off")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title(f"Pos27_ch06 frame {frame}: iterative centerline update on EFD contour")
    fig.tight_layout()
    out = Path("results/260517/_iterate_centerline_f1800.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
