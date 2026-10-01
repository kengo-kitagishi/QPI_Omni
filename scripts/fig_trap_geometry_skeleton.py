"""Whole trap crop in one frame: phase (top) and the measured geometry alone (bottom).

Top: reconstructed phase of the full trap crop (inferno, fixed range).
Bottom: same pixel frame, black background, for every cell the tracker measured in
this frame (rows of lineage_data3D.csv): EFD K=6 contour, long axis (centerline after
the midpoint update) and every short-axis chord. Geometry comes from
``fig_poster_mask_geometry_strip.cell_geometry`` (the tracker's own call), so it is
the geometry behind long_axis_um / volume_um3_efd. Mask labels without a lineage row
(e.g. cells touching the crop border) are drawn as a grey mask edge only.

Usage:
    python scripts/fig_trap_geometry_skeleton.py --pos Pos18 --ch ch02 --frame 651
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).parent))
import fig_poster_mask_geometry_strip as S  # noqa: E402  (imports figure_logger -> paper style)
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="Pos18")
    ap.add_argument("--ch", default="ch02")
    ap.add_argument("--frame", type=int, default=651)
    args = ap.parse_args()
    S.POS, S.CH = args.pos, args.ch
    paths = S.master_paths()
    px = float(paths["params"]["pixel_size_um"])
    f = args.frame

    phase = np.squeeze(tifffile.imread(paths["phase_dir"] / f"img_{f:09d}_ph_000_phase.tif")).astype(np.float32)
    mask = np.squeeze(tifffile.imread(paths["mask_dir"] / f"img_{f:09d}_ph_000_phase_masks.tif"))
    if phase.shape != mask.shape:
        raise SystemExit(f"phase {phase.shape} != mask {mask.shape}")
    lin = pd.read_csv(paths["lineage"])
    rows = lin[lin.frame == f].sort_values("centroid_x_px")
    H, W = phase.shape

    geos, recs = [], []
    for _, r in rows.iterrows():
        g = S.cell_geometry(mask, int(r.mask_label), paths["offset"])
        if g is None:
            continue
        L = float(g["arc"].sum() * px)
        V = float(g["volume_px3"] * px ** 3)
        assert abs(L / r.long_axis_um - 1) < 1e-9 and abs(V / r.volume_um3_efd - 1) < 1e-9
        geos.append(g)
        recs.append({"cell_id": int(r.cell_id), "rank": int(r["rank"]), "mask_label": int(r.mask_label),
                     "long_axis_um": L, "short_axis_um": float(r.short_axis_um), "volume_um3_efd": V,
                     "n_chords": int((g["w"] > 0).sum())})
    unmeasured = sorted(set(np.unique(mask)) - {0} - set(rows.mask_label.astype(int)))
    man = pd.DataFrame(recs)
    print(man.to_string(index=False))
    print("mask labels without lineage row:", unmeasured)

    fig_w = 183 / 25.4
    ax_h = fig_w * H / W
    gap = 0.25 / 25.4 * 10
    fig = plt.figure(figsize=(fig_w, 2 * ax_h + gap))
    fh = 2 * ax_h + gap
    ax0 = fig.add_axes([0, (ax_h + gap) / fh, 1, ax_h / fh])
    ax1 = fig.add_axes([0, 0, 1, ax_h / fh])
    ax0.imshow(phase, cmap=S.CMAP, vmin=S.VMIN, vmax=S.VMAX, interpolation="nearest")
    ax1.imshow(np.zeros_like(phase), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for g in geos:
        for i in np.flatnonzero(g["w"] > 0):
            ax1.plot([g["p0"][i, 0], g["p1"][i, 0]], [g["p0"][i, 1], g["p1"][i, 1]],
                     color=S.CHORD_C, lw=0.35)
        ax1.plot(g["medial"][:, 0], g["medial"][:, 1], color=S.LONG_C, lw=1.0)
        ax1.plot(g["contour"][:, 0], g["contour"][:, 1], color=S.CONTOUR_C, lw=0.9)
    for lb in unmeasured:
        ax1.contour((mask == lb).astype(float), [0.5], colors="0.5", linewidths=0.5)
    for ax in (ax0, ax1):
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        ax.set_axis_off()
    sb = 5.0 / px
    ax0.plot([2, 2 + sb], [H - 3, H - 3], color="white", lw=2.0, solid_capstyle="butt")
    ax0.text(2 + sb / 2, H - 4.5, "5 µm", color="white", fontsize=7, ha="center", va="bottom")

    out_dir = Path(__file__).resolve().parents[1] / "results" / "poster_mask_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"trap_{args.pos}_{args.ch}_img{f}"
    man_path = out_dir / f"manifest_{stem}.csv"
    man.to_csv(man_path, index=False)
    fig.savefig(out_dir / f"{stem}.png", dpi=600)

    data = {"phase": phase, "mask": mask, "pixel_size_um": px, "vmin": S.VMIN, "vmax": S.VMAX}
    for j, (g, rec) in enumerate(zip(geos, recs)):
        data[f"cell{rec['cell_id']}_contour_xy"] = g["contour"]
        data[f"cell{rec['cell_id']}_long_axis_xy"] = g["medial"]
        data[f"cell{rec['cell_id']}_chord_p0_xy"] = g["p0"][g["w"] > 0]
        data[f"cell{rec['cell_id']}_chord_p1_xy"] = g["p1"][g["w"] > 0]
    caption = (
        f"Top: reconstructed phase [rad] of the whole trap crop ({H}x{W} px, pixel {px:.4f} µm), "
        f"S. pombe 260517 {args.pos} {args.ch} img {f}, inferno fixed {S.VMIN}-{S.VMAX} rad; scale bar 5 µm. "
        "Bottom: same pixel frame, geometry only (not overlaid on the phase), for each of the "
        f"{len(geos)} cells the tracker measured in this frame. Cyan: Omnipose mask boundary low-pass "
        "filtered to the lowest 6 Fourier harmonics (EFD K=6), no offset. Red: long axis = centerline "
        "after one midpoint update; long_axis_um is its arc length. White: every chord perpendicular "
        "to the centerline ending on the smoothed boundary; volume_um3_efd = sum pi (w/2)^2 ds over them. "
        + (f"Grey: mask labels without a lineage row {unmeasured}. " if unmeasured else "")
        + "All lines computed from the masks, not drawn by hand.")
    save_figure(fig, params={"dataset": "260517", "master": paths["master"].name, "pos": args.pos,
                             "ch": args.ch, "frame": f, "n_cells": len(geos), "cmap": S.CMAP,
                             "vmin": S.VMIN, "vmax": S.VMAX, "pixel_size_um": px, "efd_k": 6,
                             "contour_offset_px": paths["offset"], "long_axis_color": S.LONG_C},
                description=f"Trap {args.pos} {args.ch} img {f}: phase and geometry-only panel",
                caption=caption, data=data,
                data_source={"phase_dir": str(paths["phase_dir"]), "mask_dir": str(paths["mask_dir"]),
                             "lineage_csv": str(paths["lineage"])},
                copy_files=[str(man_path)], dpi=600, fmt="png")
    plt.close(fig)
    print(out_dir / f"{stem}.png")


if __name__ == "__main__":
    main()
