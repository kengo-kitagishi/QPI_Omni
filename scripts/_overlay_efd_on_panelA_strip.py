"""Overlay the adopted EFD geometry on the Pos18 ch02 633-670 panel-A strip.

Rebuilds the exact cell-cycle strip canvas produced by
``_fig_panelA_cellcycle.py --single Pos18:ch02:633-670 --gap 0 --crop-margin 1
--n-tiles 13 --vmin 0 --vmax 1.8`` (deterministic frame sub-sampling + the same
PCA vertical alignment), recording each tile's (cx, r0) placement and rotated
mask. For every tile it then computes the ADOPTED EFD geometry
(``mask_volume_schematic.efd_section_geometry``: EFD K=6 smoothed contour + one
midpoint update) in the tile's own frame, shifts it into canvas coordinates and
draws it on top of the identical strip image.

Two figures are written to the inbox:
  1. strip + EFD-smoothed contour (boundary line only)
  2. strip + EFD contour + long axis (medial) + short-axis cross-sections

Usage:
    python scripts/_overlay_efd_on_panelA_strip.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _fig_panelA_cellcycle import (  # noqa: E402
    Z, align_vertical, crop_mother, data_root, load_pixel_size,
    _drop_glitch_frames,
)
from mask_volume_schematic import efd_section_geometry  # noqa: E402
from figure_logger import save_figure  # noqa: E402

# colours (Okabe-Ito; same palette family as the fig1A geometry overlay)
CONTOUR = "#56B4E9"   # EFD-smoothed boundary
LONG = "#D55E00"      # long axis (medial centreline)
SHORT = "#FFFFFF"     # short-axis cross-sections

POS, CH = "Pos18", "ch02"
F0, F1 = 633, 670
N_TILES, GAP, CROP_MARGIN = 13, 0, 1
VMIN, VMAX, CMAP = 0.0, 1.8, "inferno"
N_CROSS = 9            # short-axis chords drawn per cell (display only)


def build_strip_with_geometry(ch_dir: Path, seg: pd.DataFrame, px: float):
    """Reproduce build_strip_canvas, also returning per-tile placement + EFD geo.

    Returns dict with: canvas, fill, frames, times, and a list of per-tile geo
    dicts {contour, medial, p0, p1, cx, r0, w, h, volume_um3, long_um}.
    """
    # --- deterministic frame sub-sampling (identical to build_strip_canvas) ---
    if N_TILES is None or N_TILES >= len(seg):
        rows = seg
    else:
        idx = np.unique(np.r_[0, np.linspace(0, len(seg) - 1, N_TILES).round(),
                              len(seg) - 1]).astype(int)
        rows = seg.iloc[idx]

    # --- shared background fill = median of per-tile 1st percentiles ---
    fill_probe = []
    for _, r in rows.iterrows():
        crop = crop_mother(ch_dir, int(r["frame"]), r["centroid_y_px"],
                           r["centroid_x_px"])
        if crop is not None:
            fill_probe.append(np.percentile(crop[0], 1))
    fill = float(np.median(fill_probe)) if fill_probe else 0.0

    tiles, masks, frames, times = [], [], [], []
    for _, r in rows.iterrows():
        crop = crop_mother(ch_dir, int(r["frame"]), r["centroid_y_px"],
                           r["centroid_x_px"])
        if crop is None:
            continue
        phase, binary = crop
        ph_v, bw_v = align_vertical(phase, binary, fill, margin=CROP_MARGIN)
        tiles.append(ph_v)
        masks.append(bw_v)
        frames.append(int(r["frame"]))
        times.append(float(r["time_h"]))
    if not tiles:
        raise RuntimeError("no tiles for strip")

    Hs = max(t.shape[0] for t in tiles)
    total_w = sum(t.shape[1] for t in tiles) + GAP * (len(tiles) - 1)
    canvas = np.full((Hs, total_w), fill, np.float32)

    geos, cx = [], 0
    for j, t in enumerate(tiles):
        h, w = t.shape
        r0 = (Hs - h) // 2
        canvas[r0:r0 + h, cx:cx + w] = t

        g = efd_section_geometry(masks[j], pixel_size_um=px)
        entry = {"cx": cx, "r0": r0, "w": w, "h": h, "frame": frames[j],
                 "time_h": times[j], "contour": None, "medial": None,
                 "p0": None, "p1": None, "volume_um3": np.nan, "long_um": np.nan}
        if g is not None and g.contour_xy:
            shift = np.array([cx, r0], dtype=float)
            entry["contour"] = np.asarray(g.contour_xy[0], float) + shift
            entry["medial"] = np.asarray(g.medial_xy, float) + shift
            entry["p0"] = np.asarray(g.slice_p0_xy, float) + shift
            entry["p1"] = np.asarray(g.slice_p1_xy, float) + shift
            entry["volume_um3"] = float(g.volume_um3)
            entry["long_um"] = float(g.long_axis_px * px)
        geos.append(entry)
        cx += w + GAP

    return {"canvas": canvas, "fill": fill, "frames": np.array(frames),
            "times": np.array(times), "geos": geos}


def _body_slice_indices(n: int, n_cross: int, cap_frac: float = 0.12):
    """Evenly spaced interior column indices (drop the two rounded end caps)."""
    cap = max(int(cap_frac * n), 1)
    body = np.arange(cap, n - cap)
    if body.size == 0:
        body = np.arange(n)
    return body[np.linspace(0, body.size - 1, min(n_cross, body.size)).astype(int)]


def render(strip: dict, px: float, with_axes: bool):
    """One strip figure; overlay EFD contour (+ axes if with_axes)."""
    canvas = strip["canvas"]
    cH, cW = canvas.shape
    panel_w = 7.0
    fig, ax = plt.subplots(figsize=(panel_w, panel_w * cH / cW))
    ax.imshow(canvas, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
    ax.set_axis_off()

    for g in strip["geos"]:
        if g["contour"] is None:
            continue
        if with_axes:
            sel = _body_slice_indices(len(g["medial"]), N_CROSS)
            for i in sel:
                ax.plot([g["p0"][i, 0], g["p1"][i, 0]],
                        [g["p0"][i, 1], g["p1"][i, 1]],
                        color=SHORT, lw=0.5, alpha=0.85, zorder=3)
            ax.plot(g["medial"][:, 0], g["medial"][:, 1],
                    color=LONG, lw=0.9, zorder=4)
        ax.plot(g["contour"][:, 0], g["contour"][:, 1],
                color=CONTOUR, lw=0.8, zorder=5)

    # 5 um scale bar, bottom-left (matches the original strip)
    sb = 5.0 / px
    ax.plot([3, 3 + sb], [cH - 3, cH - 3], color="white", lw=3, zorder=6)
    ax.text(3 + sb / 2, cH - 5, "5 µm", color="white", fontsize=8,
            ha="center", va="bottom", zorder=6)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def main():
    ch_dir = data_root(POS) / CH
    px = load_pixel_size(ch_dir)
    dd = pd.read_csv(ch_dir / "inference_out" / "lineage_out" /
                     "lineage_data3D.csv")
    mm = dd[dd["rank"] == 1].sort_values("frame").reset_index(drop=True)
    seg = _drop_glitch_frames(
        mm[(mm["frame"] >= F0) & (mm["frame"] <= F1)].reset_index(drop=True))

    strip = build_strip_with_geometry(ch_dir, seg, px)
    geos = strip["geos"]
    ok = [g for g in geos if g["contour"] is not None]
    print(f"tiles: {len(geos)}  with EFD geometry: {len(ok)}")
    for g in geos:
        print(f"  frame {g['frame']:4d}  t={g['time_h']:5.1f}h  "
              f"L={g['long_um']:.2f}um  V={g['volume_um3']:.2f}um3"
              if g["contour"] is not None else
              f"  frame {g['frame']:4d}: EFD geometry failed")

    # shared plot data for restyling from npz (every tile array)
    data = {"canvas": strip["canvas"], "frames": strip["frames"],
            "times_h": strip["times"], "vmin": float(VMIN), "vmax": float(VMAX),
            "pixel_size_um": float(px)}
    for j, g in enumerate(geos):
        data[f"tile{j:02d}_placement"] = np.array([g["cx"], g["r0"], g["w"], g["h"]])
        if g["contour"] is not None:
            data[f"tile{j:02d}_contour_xy"] = g["contour"]
            data[f"tile{j:02d}_medial_xy"] = g["medial"]
            data[f"tile{j:02d}_slice_p0_xy"] = g["p0"]
            data[f"tile{j:02d}_slice_p1_xy"] = g["p1"]

    base_params = {
        "dataset": "260517", "pos": POS, "z": Z, "ch": CH,
        "frame_min": F0, "frame_max": F1, "cmap": CMAP,
        "vmin": VMIN, "vmax": VMAX, "gap_px": GAP,
        "crop_margin_px": CROP_MARGIN, "pixel_size_um": px,
        "n_tiles": len(geos), "efd_k": 6,
        "geometry": "efd_section_geometry (EFD K=6 + 1 midpoint update)",
    }

    # --- figure 1: EFD contour only ---
    fig1 = render(strip, px, with_axes=False)
    save_figure(
        fig1,
        params={**base_params, "overlay": "EFD contour",
                "style": "panelA strip + EFD smoothed contour"},
        description=(
            f"Panel-A cell-cycle strip {POS} {CH} {F0}-{F1} with the adopted "
            f"EFD-smoothed mask boundary (K=6) overlaid per tile in cyan. Same "
            f"deterministic tiling / vertical PCA alignment and fixed "
            f"{VMIN}-{VMAX} rad inferno as the original strip; the contour is the "
            f"low-pass Fourier reconstruction of the raw mask boundary used by "
            f"the corrected-volume pipeline."),
        data=data, dpi=400, fmt="png",
    )
    plt.close(fig1)

    # --- figure 2: EFD contour + long axis + short axis ---
    fig2 = render(strip, px, with_axes=True)
    save_figure(
        fig2,
        params={**base_params, "overlay": "EFD contour + long axis + short axis",
                "n_cross_display": N_CROSS,
                "style": "panelA strip + EFD contour + medial long axis + short-axis sections"},
        description=(
            f"Panel-A cell-cycle strip {POS} {CH} {F0}-{F1} with the adopted EFD "
            f"geometry overlaid per tile: cyan = EFD-smoothed boundary, red = "
            f"medial long axis (centerline), white = short-axis cross-sections "
            f"(perpendicular chords, {N_CROSS}/cell display sub-sample; the volume "
            f"integral uses every column). Same tiling / vertical alignment and "
            f"fixed {VMIN}-{VMAX} rad inferno as the original strip."),
        data=data, dpi=400, fmt="png",
    )
    plt.close(fig2)
    print("done")


if __name__ == "__main__":
    main()
