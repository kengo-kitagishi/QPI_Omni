"""_fig_volume_method_comparison_260517.py - one-off comparison of the volume methods.

The master carries only the yellow-contour volumes (rod and efd). This figure keeps the
side-by-side record of the methods that were used before, computed on the same masks:

    medial rod       capsule from the medial-axis long/short axes           (master until 2026-09-14)
    medial profile   solid of revolution of the cos-theta corrected columns  (master until 2026-09-14)
    EFD 0 px         solid of revolution of chords on the EFD-smoothed contour without the inward shrink
    yellow rod       capsule from the yellow-contour axes                    (master since 2026-09-14)
    yellow efd       solid of revolution of the yellow-contour chords        (adopted, master since 2026-09-14)

Cells: every mask not touching the crop border in the sampled frames of a few clean
Pos <= 52 channels (mothers and daughters pooled). Saved via figure_logger with the
per-cell values as source data.

Usage:
    python scripts/_fig_volume_method_comparison_260517.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from figure_logger import save_figure  # noqa: E402  (applies the paper style on import)
import matplotlib.pyplot as plt  # noqa: E402
from mask_morphology import measure_all_modes  # noqa: E402
from mask_volume_schematic import efd_section_geometry  # noqa: E402
from central_cell_lineage_tracker import _yellow_axes, calc_rod_volume_um3  # noqa: E402
import _retrack_260517_newmodel as chain  # noqa: E402

PX = float(chain.PIXEL_UM)
CHANNELS = [("Pos1", "ch05"), ("Pos4", "ch02"), ("Pos6", "ch04")]
COLORS = {"medial rod": "#999999", "medial profile": "#bbbbbb", "EFD 0 px": "#56B4E9",
          "yellow rod": "#E69F00", "yellow efd": "#D55E00"}


def measure(mask: np.ndarray, label: int) -> dict | None:
    b = np.pad(mask == label, 6)
    a = measure_all_modes(b)
    g = efd_section_geometry(b, pixel_size_um=1.0)
    g0 = efd_section_geometry(b, pixel_size_um=1.0, contour_offset_px=0.0)
    if a is None or g is None or g0 is None:
        return None
    L, w, v_efd, n = _yellow_axes(g)
    if not np.isfinite(v_efd):
        return None
    return {
        "medial rod": calc_rod_volume_um3(a["medial_long_px"], a["medial_short_px"], PX),
        "medial profile": a["medial_profile_px3"] * PX ** 3,
        "EFD 0 px": g0.volume_px3 * PX ** 3,
        "yellow rod": calc_rod_volume_um3(L, w, PX),
        "yellow efd": v_efd * PX ** 3,
        "short_medial_um": a["medial_short_px"] * PX,
        "short_yellow_um": w * PX,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", default="200:1800:40", help="start:stop:step (absolute img)")
    args = ap.parse_args()
    f0, f1, st = (int(x) for x in args.frames.split(":"))
    rows = []
    for pos, ch in CHANNELS:
        inf = chain.MASK_ROOT / pos / chain.REL / ch / "inference_out"
        for f in range(f0, f1 + 1, st):
            p = inf / f"img_{f:09d}_ph_000_phase_masks.tif"
            if not p.exists():
                continue
            m = tifffile.imread(str(p))
            h, w = m.shape
            for lab in range(1, int(m.max()) + 1):
                sel = m == lab
                if sel.sum() < 20:
                    continue
                ys, xs = np.where(sel)
                if ys.min() <= 0 or xs.min() <= 0 or ys.max() >= h - 1 or xs.max() >= w - 1:
                    continue  # touches border
                r = measure(m, lab)
                if r is None:
                    continue
                r.update(pos=pos, ch=ch, frame=f, label=lab)
                rows.append(r)
    df = pd.DataFrame(rows)
    methods = list(COLORS)
    n = len(df)
    med = {k: float(df[k].median()) for k in methods}
    ratio = {k: float((df[k] / df["medial profile"]).median()) for k in methods}
    print(f"{n} cell-frames from {len(CHANNELS)} channels, frames {f0}..{f1} step {st}")
    for k in methods:
        print(f"  {k:15s} median {med[k]:6.1f} um3  ratio to medial profile {ratio[k]:.3f}")
    print(f"  short axis: medial {df.short_medial_um.median():.2f} um, yellow {df.short_yellow_um.median():.2f} um")

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 72 / 25.4))
    ax = axes[0]
    data = [df[k].to_numpy() for k in methods]
    bp = ax.boxplot(data, widths=0.6, showfliers=False, patch_artist=True,
                    medianprops=dict(color="black", lw=0.8))
    for patch, k in zip(bp["boxes"], methods):
        patch.set_facecolor(COLORS[k])
        patch.set_alpha(0.8)
        patch.set_linewidth(0.6)
    rng = np.random.default_rng(0)
    for i, k in enumerate(methods, 1):
        ax.plot(i + rng.uniform(-0.18, 0.18, n), df[k], ".", ms=1.2, color="black", alpha=0.15, rasterized=True)
    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels(methods, rotation=35, ha="right")
    ax.set_ylabel("cell volume [um$^3$]")
    ax.set_ylim(0, None)
    ax.set_title(f"n = {n} cell-frames", fontsize=7)

    ax = axes[1]
    x = df["medial profile"].to_numpy()
    for k in ("yellow efd", "yellow rod"):
        ax.scatter(x, df[k], s=4, color=COLORS[k], alpha=0.35, linewidths=0, rasterized=True,
                   label=f"{k}: median ratio {ratio[k]:.2f}")
    lim = float(np.nanpercentile(x, 99.5)) * 1.05
    ax.plot([0, lim], [0, lim], color="black", lw=0.6, ls="--")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("medial profile volume [um$^3$]")
    ax.set_ylabel("yellow volume [um$^3$]")
    ax.legend(frameon=False, fontsize=6, loc="upper left")

    ax = axes[2]
    d2 = [df["short_medial_um"].to_numpy(), df["short_yellow_um"].to_numpy()]
    bp = ax.boxplot(d2, widths=0.55, showfliers=False, patch_artist=True, medianprops=dict(color="black", lw=0.8))
    for patch, c in zip(bp["boxes"], (COLORS["medial profile"], COLORS["yellow efd"])):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
        patch.set_linewidth(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["medial", "yellow"])
    ax.set_ylabel("short axis [um]")
    ax.set_ylim(0, None)
    fig.tight_layout()

    caption = (
        f"Volume methods compared on the same Omnipose masks: {n} cell-frames (mothers and daughters not touching "
        f"the crop border) from channels {', '.join(f'{p} {c}' for p, c in CHANNELS)}, absolute frames {f0}-{f1} every "
        f"{st} frames (2% glucose, 2026-05-17 experiment, 0.346 um/px). Left: per-cell-frame volumes; boxes = median and "
        f"quartiles, whiskers = 1.5 IQR, dots = individual cell-frames. medial rod = capsule (4/3 pi r^3 + pi r^2 (L-2r)) "
        f"from the medial-axis long/short axes; medial profile = sum pi (w/2)^2 ds of the cos(theta)-corrected column chords; "
        f"EFD 0 px = sum pi (w/2)^2 ds of chords perpendicular to the centerline bounded by the elliptic-Fourier (K=6) "
        f"smoothed contour without shrink; yellow rod / yellow efd = the same two quantities from the contour shrunk 0.5 px "
        f"inward (adopted 2026-09-07). Middle: per-cell-frame yellow volumes against the medial profile volume, dashed = "
        f"identity; median ratios yellow efd {ratio['yellow efd']:.2f}, yellow rod {ratio['yellow rod']:.2f}. Right: short "
        f"axis, medial (mean cos(theta)-corrected column width over the body plateau) versus yellow (mean chord width over the "
        f"body plateau); medians {df.short_medial_um.median():.2f} and {df.short_yellow_um.median():.2f} um. No statistical test."
    )
    out = save_figure(
        fig,
        params={"channels": CHANNELS, "frames": [f0, f1, st], "n_cell_frames": n, "medians_um3": med,
                "ratio_to_medial_profile": ratio, "efd_k": 6, "contour_offset_px": 0.5},
        description="Side-by-side record of cell-volume methods (medial rod/profile, EFD 0 px, yellow rod/efd) on identical masks",
        caption=caption,
        data={k.replace(" ", "_"): df[k].to_numpy() for k in methods}
        | {"short_medial_um": df.short_medial_um.to_numpy(), "short_yellow_um": df.short_yellow_um.to_numpy(),
           "pos": df.pos.to_numpy().astype(str), "ch": df.ch.to_numpy().astype(str),
           "frame": df.frame.to_numpy(), "label": df.label.to_numpy()},
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
