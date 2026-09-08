"""Sweep centerline-smoothing strength on Pos27_ch06 frame 1800 (EFD method).
Render the schematic (cyan contour, red centerline, gray perpendicular sections)
at smoothing_window_frac in {0.15, 0.30, 0.50} so the section-straightening and
any volume change are visible side by side."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_volume_schematic import efd_section_geometry

d = np.load("results/260517/_crop_cache/Pos27_ch06_f1800.npz")
b = d["binary"].astype(bool)
px = float(d["px"])

import matplotlib.pyplot as plt
fracs = [0.15, 0.30, 0.50]
fig, axes = plt.subplots(1, len(fracs), figsize=(15, 4.6))
H, W = b.shape
lim = dict(xlim=(-2, W + 2), ylim=(H + 2, -2))

for ax, frac in zip(axes, fracs):
    geo = efd_section_geometry(b, pixel_size_um=px, smoothing_window_frac=frac)
    n_col = len(geo.medial_xy)
    win = max(int(frac * n_col), 3)
    # mean section tilt from vertical
    v = geo.slice_p1_xy - geo.slice_p0_xy
    ang = np.degrees(np.arctan2(np.abs(v[:, 0]), np.abs(v[:, 1])))
    for i in range(n_col):
        ax.plot([geo.slice_p0_xy[i, 0], geo.slice_p1_xy[i, 0]],
                [geo.slice_p0_xy[i, 1], geo.slice_p1_xy[i, 1]],
                color="0.55", lw=0.8, zorder=1)
    for c in geo.contour_xy:
        ax.plot(c[:, 0], c[:, 1], color="deepskyblue", lw=2.2, zorder=3)
    ax.plot(geo.medial_xy[:, 0], geo.medial_xy[:, 1], color="red", lw=2, zorder=4)
    ax.set_aspect("equal"); ax.set(**lim); ax.axis("off")
    ax.set_title(f"smooth_frac={frac}  (window={win}px = {win*px:.2f} µm)\n"
                 f"V = {geo.volume_um3:.2f} µm³   section tilt mean {ang.mean():.1f}° "
                 f"max {ang.max():.1f}°", fontsize=9)

fig.suptitle("Pos27_ch06 frame 1800 — centerline smoothing sweep (EFD method)",
             fontsize=11)
fig.tight_layout()
out = Path("results/260517/_smooth_sweep_f1800.png")
fig.savefig(out, dpi=170, bbox_inches="tight")
print(f"wrote {out}")
