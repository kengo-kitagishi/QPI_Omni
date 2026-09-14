"""Overlay comparison on the SAME real mask (Pos27 ch06 frame 1800):
  left  = current pipeline  : w_perp = h_vert*cos(theta) slices (mask_morphology)
  right = Odermatt method   : perpendicular sectioning lines intersected with the
          contour (existing 31_roiset_rotational_volume.RotationalSymmetryROIAnalyzer)

Shows the "pre-revolution planar sections" vs the actual mask for both, and the
resulting volume from each. Uses the cached crop so it runs instantly.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from mask_volume_schematic import load_mother_crop, medial_axis_geometry


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    binary, frame, px = load_mother_crop("Pos27", "ch06", 1800)
    H, W = binary.shape

    # --- current pipeline: cos(theta) projection ---
    geo = medial_axis_geometry(binary, pixel_size_um=px)

    # --- Odermatt contour-intersection (existing implementation, no ROI zip) ---
    rot = _load_by_path("rot31",
                        str(Path(__file__).parent / "31_roiset_rotational_volume.py"))
    A = object.__new__(rot.RotationalSymmetryROIAnalyzer)
    A.pixel_size_um = px
    A.section_interval_um = 0.25            # 250 nm, as in the paper
    A.section_interval_px = 0.25 / px
    A.image_width, A.image_height = W, H
    A.max_iterations = 3
    A.convergence_tolerance = 0.5
    res = A.compute_volume_rotational(binary.astype(bool),
                                      return_visualization_data=True,
                                      return_thickness_map=False)
    if res is None:
        print("contour-intersection returned None", file=sys.stderr)
        sys.exit(1)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    lim = dict(xlim=(-2, W + 2), ylim=(H + 2, -2))   # zoom to the crop, image y down

    # panel A: cos(theta) projection
    ax = axes[0]
    for i in range(len(geo.medial_xy)):
        ax.plot([geo.slice_p0_xy[i, 0], geo.slice_p1_xy[i, 0]],
                [geo.slice_p0_xy[i, 1], geo.slice_p1_xy[i, 1]],
                color="0.55", lw=0.8, zorder=1)
    for c in geo.contour_xy:
        ax.plot(c[:, 0], c[:, 1], color="deepskyblue", lw=2.2, zorder=3)
    ax.plot(geo.medial_xy[:, 0], geo.medial_xy[:, 1], color="red", lw=2, zorder=4)
    ax.set_aspect("equal"); ax.set(**lim); ax.axis("off")
    ax.set_title(f"current: w_perp = h_vert·cos θ\n"
                 f"V = {geo.volume_um3:.2f} µm³  ({len(geo.medial_xy)} disks @ 1 px)",
                 fontsize=10)

    # panel B: contour-intersection sections (Odermatt)
    ax = axes[1]
    for (p1, p2) in res["section_lines"]:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="0.55", lw=0.8, zorder=1)
    ct = res["contour"]
    ax.plot(ct[:, 0], ct[:, 1], color="deepskyblue", lw=2.2, zorder=3)
    cl = res["centerline_points"]
    if cl is not None and len(cl):
        ax.plot(cl[:, 0], cl[:, 1], color="red", lw=2, zorder=4)
        ax.plot(cl[:, 0], cl[:, 1], color="red", marker="o", ms=3, lw=2, zorder=5)
    ax.set_aspect("equal"); ax.set(**lim); ax.axis("off")
    ax.set_title(f"contour-intersection (Odermatt)\n"
                 f"V = {res['volume_um3']:.2f} µm³  "
                 f"({res['n_sections']} sections @ 250 nm)", fontsize=10)

    ratio = res["volume_um3"] / geo.volume_um3 if geo.volume_um3 else float("nan")
    fig.suptitle(f"Pos27_ch06 frame {frame} — volume method comparison   "
                 f"(contour/projection = {ratio:.3f})", fontsize=11)
    fig.tight_layout()
    out = Path("results/260517/_compare_volmethods_f1800.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"V_projection   = {geo.volume_um3:.3f} um3")
    print(f"V_contour_isec = {res['volume_um3']:.3f} um3  "
          f"(n={res['n_sections']}, removed={res.get('n_sections_removed')})")
    print(f"ratio contour/projection = {ratio:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
