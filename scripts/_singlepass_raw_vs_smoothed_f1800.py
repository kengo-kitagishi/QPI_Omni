"""Single-pass contour-intersection (one midpoint update) on frame 1800, using
RAW vs EFD-smoothed contour — and what happens to the MEASURED volume if we
smooth the contour for the measurement too (not just the figure).

Left  : single-pass contour-intersection on the RAW (staircase) contour
Right : same, but on a Fourier-low-pass (EFD K=6) smoothed contour -> the volume
        here is the "smoothed-for-measurement" answer.
Both panels: cyan contour, red updated centerline (= midpoints of the chords),
gray section chords. cos-theta profile volume is printed as a reference.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_volume_schematic import load_mother_crop, medial_axis_geometry


def close(c):
    return c if np.allclose(c[0], c[-1]) else np.vstack([c, c[:1]])


def fourier_smooth(contour_xy, K=6, M=512):
    """Fourier-descriptor low-pass of a closed contour (EFD-equivalent).
    Resample to uniform arc length, FFT the complex boundary, keep DC + ±K
    harmonics, inverse FFT. Robust, no self-intersection at low K."""
    c = close(contour_xy)
    seg = np.linalg.norm(np.diff(c, axis=0), axis=1)
    t = np.concatenate([[0], np.cumsum(seg)])
    L = t[-1]
    tu = np.linspace(0, L, M, endpoint=False)
    x = np.interp(tu, t, c[:, 0])
    y = np.interp(tu, t, c[:, 1])
    Z = np.fft.fft(x + 1j * y)
    keep = np.zeros(M, bool)
    keep[0] = True
    keep[1:K + 1] = True
    keep[-K:] = True
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


def shoelace(c):
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))


def singlepass(contour, centers, perp, ds, px):
    p0s, p1s, mids, w = [], [], [], []
    used = 0
    for P, d in zip(centers, perp):
        c = chord_to_contour(contour, P, d)
        if c is None:
            p0s.append(P); p1s.append(P); mids.append(P); w.append(0.0)
            continue
        a, b = c
        p0s.append(a); p1s.append(b); mids.append((a + b) / 2)
        w.append(float(np.linalg.norm(b - a))); used += 1
    p0s = np.array(p0s); p1s = np.array(p1s); mids = np.array(mids); w = np.array(w)
    V = float(np.sum(np.pi * (w / 2) ** 2 * ds)) * px ** 3
    return dict(p0=p0s, p1=p1s, mid=mids, w=w, V=V, used=used)


def main():
    binary, frame, px = load_mother_crop("Pos27", "ch06", 1800)
    H, W = binary.shape
    geo = medial_axis_geometry(binary, pixel_size_um=px)
    centers = geo.medial_xy
    perp = geo.slice_p1_xy - geo.medial_xy
    perp = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)
    ds = np.linalg.norm(np.gradient(centers, axis=0), axis=1)

    raw = close(max(geo.contour_xy, key=len))
    efd = fourier_smooth(raw, K=6)

    r_raw = singlepass(raw, centers, perp, ds, px)
    r_efd = singlepass(efd, centers, perp, ds, px)

    area_raw = binary.sum()
    print(f"frame {frame}  px={px*1000:.1f}nm  area_px={area_raw}")
    print(f"cos-theta profile (reference)      V = {geo.volume_um3:7.3f} um3")
    print(f"single-pass isec, RAW contour      V = {r_raw['V']:7.3f} um3  "
          f"(used {r_raw['used']}/{len(centers)})")
    print(f"single-pass isec, EFD K=6 contour  V = {r_efd['V']:7.3f} um3  "
          f"(used {r_efd['used']}/{len(centers)})  "
          f"dV vs raw = {100*(r_efd['V']-r_raw['V'])/r_raw['V']:+.2f}%")
    print(f"EFD contour area / raw mask area   = {shoelace(efd)/area_raw:.4f}")

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    lim = dict(xlim=(-2, W + 2), ylim=(H + 2, -2))
    for ax, contour, res, name in [
        (axes[0], raw, r_raw, "RAW contour"),
        (axes[1], efd, r_efd, "EFD K=6 smoothed contour (measured)"),
    ]:
        for i in range(len(res["mid"])):
            ax.plot([res["p0"][i, 0], res["p1"][i, 0]],
                    [res["p0"][i, 1], res["p1"][i, 1]], color="0.55", lw=0.8)
        ax.plot(contour[:, 0], contour[:, 1], color="deepskyblue", lw=2.2)
        ax.plot(res["mid"][:, 0], res["mid"][:, 1], color="red", lw=2)
        ax.set_aspect("equal"); ax.set(**lim); ax.axis("off")
        ax.set_title(f"single-pass midpoint isec — {name}\nV = {res['V']:.2f} µm³",
                     fontsize=10)
    fig.suptitle(f"Pos27_ch06 frame {frame}: measure on raw vs smoothed contour   "
                 f"(cos θ ref = {geo.volume_um3:.2f} µm³)", fontsize=11)
    fig.tight_layout()
    out = Path("results/260517/_singlepass_raw_vs_smoothed_f1800.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
