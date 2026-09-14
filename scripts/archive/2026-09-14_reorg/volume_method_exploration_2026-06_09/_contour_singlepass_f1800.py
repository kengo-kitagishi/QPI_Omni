"""The "+1 step" the user described: start from the stable cos-theta medial
centerline, then do ONE contour-intersection pass — for each section take the
perpendicular line's two contour-intersection points; their midpoint becomes the
updated long axis (centerline) and their half-distance the short axis (radius).
Single pass (no iteration) -> no pole blow-up, and the chords sit on the contour
by construction (no overshoot). Compare volume to the current cos-theta profile.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_volume_schematic import load_mother_crop, medial_axis_geometry


def chord_to_contour(contour, P, d):
    """Two contour-intersection points of the infinite line P + t*d straddling P.
    contour: closed (N,2) xy polyline. Returns (p_neg, p_pos) or None."""
    a = contour[:-1]
    b = contour[1:]
    e = b - a
    r = a - P
    det = -d[0] * e[:, 1] + e[:, 0] * d[1]
    ok = np.abs(det) > 1e-9
    t = np.full(len(e), np.nan)
    s = np.full(len(e), np.nan)
    t[ok] = (-r[ok, 0] * e[ok, 1] + e[ok, 0] * r[ok, 1]) / det[ok]
    s[ok] = (d[0] * r[ok, 1] - d[1] * r[ok, 0]) / det[ok]
    val = ok & (s >= 0) & (s <= 1)
    tv = t[val]
    pos = tv[tv > 1e-6]
    neg = tv[tv < -1e-6]
    if pos.size == 0 or neg.size == 0:
        return None
    return P + neg.max() * d, P + pos.min() * d


def main():
    binary, frame, px = load_mother_crop("Pos27", "ch06", 1800)
    H, W = binary.shape
    geo = medial_axis_geometry(binary, pixel_size_um=px)

    contour = max(geo.contour_xy, key=len)
    contour = np.vstack([contour, contour[:1]])            # close it

    # perpendicular direction at each section (from the cos-theta geometry)
    perp = geo.slice_p1_xy - geo.medial_xy
    perp = perp / np.maximum(np.linalg.norm(perp, axis=1, keepdims=True), 1e-9)

    p0s, p1s, mids, w = [], [], [], []
    for P, d in zip(geo.medial_xy, perp):
        c = chord_to_contour(contour, P, d)
        if c is None:
            p0s.append(P); p1s.append(P); mids.append(P); w.append(0.0)
            continue
        a, b = c
        p0s.append(a); p1s.append(b); mids.append((a + b) / 2)
        w.append(float(np.linalg.norm(b - a)))
    p0s = np.array(p0s); p1s = np.array(p1s); mids = np.array(mids); w = np.array(w)

    ds = np.linalg.norm(np.gradient(mids, axis=0), axis=1)
    vol_um3 = float(np.sum(np.pi * (w / 2) ** 2 * ds)) * px ** 3
    ratio = vol_um3 / geo.volume_um3

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    lim = dict(xlim=(-2, W + 2), ylim=(H + 2, -2))

    ax = axes[0]
    for i in range(len(geo.medial_xy)):
        ax.plot([geo.slice_p0_xy[i, 0], geo.slice_p1_xy[i, 0]],
                [geo.slice_p0_xy[i, 1], geo.slice_p1_xy[i, 1]], color="0.55", lw=0.8)
    for c in geo.contour_xy:
        ax.plot(c[:, 0], c[:, 1], color="deepskyblue", lw=2.2)
    ax.plot(geo.medial_xy[:, 0], geo.medial_xy[:, 1], color="red", lw=2)
    ax.set_aspect("equal"); ax.set(**lim); ax.axis("off")
    ax.set_title(f"current cos θ profile  (width = h_vert·cos θ)\n"
                 f"V = {geo.volume_um3:.2f} µm³", fontsize=10)

    ax = axes[1]
    for i in range(len(mids)):
        ax.plot([p0s[i, 0], p1s[i, 0]], [p0s[i, 1], p1s[i, 1]], color="0.55", lw=0.8)
    for c in geo.contour_xy:
        ax.plot(c[:, 0], c[:, 1], color="deepskyblue", lw=2.2)
    ax.plot(mids[:, 0], mids[:, 1], color="red", lw=2)
    ax.set_aspect("equal"); ax.set(**lim); ax.axis("off")
    ax.set_title(f"+1 step: width & centerline from contour intersection (single pass)\n"
                 f"V = {vol_um3:.2f} µm³", fontsize=10)

    fig.suptitle(f"Pos27_ch06 frame {frame} — add the contour-intersection step "
                 f"(ratio = {ratio:.3f})", fontsize=11)
    fig.tight_layout()
    out = Path("results/260517/_contour_singlepass_f1800.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"cos_theta      = {geo.volume_um3:.3f} um3")
    print(f"contour 1-pass = {vol_um3:.3f} um3   ratio = {ratio:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
