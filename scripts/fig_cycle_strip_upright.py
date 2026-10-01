"""One mother cell cycle as a horizontal strip of upright cells (phase only, every frame).

Each frame: the mother's label mask (master lineage row, mask_label) gives its centroid
and principal axis (second moments of the mask pixels). The phase image is resampled
(bilinear, scipy map_coordinates order=1) on a fixed-size tile whose vertical axis is
that principal axis, centred on the centroid, trap end set by --top-end. Every frame of
the cycle is shown, no sub-sampling. No contour or axes are drawn.

Usage:
    python scripts/fig_cycle_strip_upright.py --pos Pos18 --ch ch02 --f0 633 --f1 670
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
import fig_poster_mask_geometry_strip as S  # noqa: E402  (imports figure_logger -> paper style)
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

TILE_W = 20      # px across the cell (6.9 um)
GAP = 1          # px between tiles
MARGIN = 4       # px above / below the longest cell
BAR_BAND = 14    # px of black band under the tiles for the scale bar


def principal_axis(bw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(bw)
    c = np.array([xs.mean(), ys.mean()])
    evals, evecs = np.linalg.eigh(np.cov(np.vstack([xs - c[0], ys - c[1]])))
    return c, evecs[:, np.argmax(evals)]          # (x, y) centroid, unit (vx, vy)


def upright_tile(phase, c, v, height: int, top_high_x: bool) -> np.ndarray:
    """Sample phase on a TILE_W x height grid: tile up = +v (v points to the top end)."""
    if (v[0] > 0) != top_high_x:
        v = -v
    u = np.array([-v[1], v[0]])                   # tile +x; [u, -v] is a proper rotation
    ty, tx = np.mgrid[0:height, 0:TILE_W].astype(float)
    dx, dy = tx - (TILE_W - 1) / 2, ty - (height - 1) / 2
    x = c[0] + dx * u[0] - dy * v[0]
    y = c[1] + dx * u[1] - dy * v[1]
    return ndimage.map_coordinates(phase, [y, x], order=1, cval=np.nan)


def export_figma(out: Path, tiles, frames, t_min, px: float, scale: int):
    """Separate assets for layout in Figma, all at `scale` screen px per data px.

    tiles/  one PNG per frame (nearest-neighbour upscaling, NaN -> black)
    scale_bar_5um.svg / color_bar.svg  (text kept as editable SVG text)
    tiles.csv  frame, time since birth, file name
    """
    import matplotlib as mpl
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiles").mkdir(exist_ok=True)
    cmap = plt.get_cmap(S.CMAP).copy()
    cmap.set_bad("black")
    norm = mpl.colors.Normalize(S.VMIN, S.VMAX)
    rows = []
    for t, f, tm in zip(tiles, frames, t_min):
        rgb = (cmap(norm(np.ma.masked_invalid(t)))[..., :3] * 255).round().astype(np.uint8)
        rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
        name = f"tile_{len(rows):02d}_img{f:04d}_t{tm:03d}min.png"
        plt.imsave(out / "tiles" / name, rgb)
        rows.append({"index": len(rows), "frame": int(f), "time_from_birth_min": int(tm), "file": name,
                     "width_px": rgb.shape[1], "height_px": rgb.shape[0]})
    pd.DataFrame(rows).to_csv(out / "tiles.csv", index=False)

    bar_len = 5.0 / px * scale
    bar_h = 1.5 * scale                        # thickness 1.5 data px; only the length is the scale
    w = bar_len + 2
    (out / "scale_bar_5um.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.3f}" height="{bar_h + 2:.3f}" '
        f'viewBox="0 0 {w:.3f} {bar_h + 2:.3f}">'
        f'<rect x="1" y="1" width="{bar_len:.3f}" height="{bar_h:.3f}" fill="#FFFFFF"/></svg>\n',
        encoding="utf-8")

    with mpl.rc_context({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica"], "font.size": 7}):
        fig = plt.figure(figsize=(0.35, 1.6))
        cax = fig.add_axes([0.05, 0.05, 0.3, 0.9])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=S.CMAP), cax=cax)
        cb.set_ticks([0.0, 0.6, 1.2, 1.8])
        cb.set_label("Phase (rad)")
        cb.outline.set_linewidth(0.5)
        fig.savefig(out / "color_bar.svg", transparent=True, bbox_inches="tight")
        fig.savefig(out / "color_bar.pdf", transparent=True, bbox_inches="tight")
        plt.close(fig)
    (out / "README.txt").write_text(
        f"Scale: {scale} screen px per data px; data px = {px:.6f} um, so 1 um = {scale / px:.3f} px.\n"
        f"scale_bar_5um.svg: white bar {bar_len:.3f} px = 5 um (no text; add the label in Figma).\n"
        "Place tiles and the scale bar without resizing them, or resize them by the same factor.\n"
        f"color_bar: {S.CMAP}, {S.VMIN}-{S.VMAX} rad, same mapping as the tiles.\n",
        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="Pos18")
    ap.add_argument("--ch", default="ch02")
    ap.add_argument("--f0", type=int, default=633)
    ap.add_argument("--f1", type=int, default=670)
    ap.add_argument("--figma", type=int, default=10, metavar="SCALE",
                    help="also export separate tiles / scale bar / color bar at SCALE px per data px (0: off)")
    args = ap.parse_args()
    S.POS, S.CH = args.pos, args.ch
    paths = S.master_paths()
    px = float(paths["params"]["pixel_size_um"])
    lin = pd.read_csv(paths["lineage"])
    seg = (lin[(lin["rank"] == 1) & lin.frame.between(args.f0, args.f1)]
           .sort_values("frame").reset_index(drop=True))
    if len(seg) != args.f1 - args.f0 + 1 or seg.cell_id.nunique() != 1:
        raise SystemExit(f"cycle not gap-free / single cell: {len(seg)} rows, ids {seg.cell_id.unique()}")
    top_high_x = None
    height = int(np.ceil(seg.long_axis_um.max() / px)) + 2 * MARGIN

    tiles, angles = [], []
    for _, r in seg.iterrows():
        f = int(r.frame)
        phase = np.squeeze(tifffile.imread(paths["phase_dir"] / f"img_{f:09d}_ph_000_phase.tif")).astype(np.float32)
        mask = np.squeeze(tifffile.imread(paths["mask_dir"] / f"img_{f:09d}_ph_000_phase_masks.tif"))
        bw = mask == int(r.mask_label)
        if int(bw.sum()) != int(r.area_px):
            raise SystemExit(f"frame {f}: mask_label {r.mask_label} area {bw.sum()} != {r.area_px}")
        if top_high_x is None:
            top_high_x = float(seg.centroid_x_px.mean()) > phase.shape[1] / 2
        c, v = principal_axis(bw)
        tiles.append(upright_tile(phase, c, v, height, top_high_x))
        angles.append(float(np.degrees(np.arctan2(v[1], v[0]))))

    n = len(tiles)
    canvas = np.full((height + BAR_BAND, n * TILE_W + (n - 1) * GAP), np.nan, np.float32)
    for j, t in enumerate(tiles):
        canvas[:height, j * (TILE_W + GAP): j * (TILE_W + GAP) + TILE_W] = t
    cH, cW = canvas.shape
    t_min = np.round((seg.time_h.to_numpy() - seg.time_h.iloc[0]) * 60).astype(int)

    fig_w = 183 / 25.4
    top = 3.2 / 25.4
    fig_h = fig_w * cH / cW + top
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_axes([0, 0, 1, (fig_h - top) / fig_h])
    cmap = plt.get_cmap(S.CMAP).copy()
    cmap.set_bad("black")
    ax.imshow(np.ma.masked_invalid(canvas), cmap=cmap, vmin=S.VMIN, vmax=S.VMAX, interpolation="nearest")
    ax.set_xlim(-0.5, cW - 0.5)
    ax.set_ylim(cH - 0.5, -0.5)
    ax.set_axis_off()
    for j, tm in enumerate(t_min):
        if tm % 30 == 0:   # tick labels every 30 min; every frame is still shown
            ax.text(j * (TILE_W + GAP) - 0.5 if j == 0 else j * (TILE_W + GAP) + (TILE_W - 1) / 2, -1.0,
                    f"{tm} min" if j == 0 else f"{tm}", ha="left" if j == 0 else "center", va="bottom", fontsize=7, clip_on=False)
    sb = 5.0 / px
    ax.plot([0, sb], [cH - 7.0, cH - 7.0], color="white", lw=2.0, solid_capstyle="butt")
    ax.text(sb + 2, cH - 7.0, "5 µm", color="white", fontsize=7, ha="left", va="center")

    out_dir = Path(__file__).resolve().parents[1] / "results" / "poster_mask_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"upright_{args.pos}_{args.ch}_{args.f0}-{args.f1}"
    fig.savefig(out_dir / f"{stem}.png", dpi=600)

    caption = (
        f"One mother cell cycle of S. pombe (260517 {args.pos} {args.ch}, 2% glucose, 5 min/frame), "
        f"every frame from birth (img {args.f0}) to the frame before division (img {args.f1}), "
        f"n = {n} frames, left to right. Each tile: reconstructed phase [rad], inferno, fixed "
        f"{S.VMIN}-{S.VMAX} rad, resampled bilinearly on a {TILE_W} x {height} px grid (pixel {px:.4f} µm) "
        "rotated so the principal axis of the cell's Omnipose mask (second moments of the mask "
        "pixels) is vertical and centred on the mask centroid; neighbouring cells in the trap "
        "remain visible; black band below the tiles holds the scale bar. Numbers: time since birth [min], labelled every 30 min. Scale bar 5 µm.")
    save_figure(fig, params={"dataset": "260517", "master": paths["master"].name, "pos": args.pos,
                             "ch": args.ch, "frame_min": args.f0, "frame_max": args.f1, "n_frames": n,
                             "cmap": S.CMAP, "vmin": S.VMIN, "vmax": S.VMAX, "tile_w_px": TILE_W,
                             "tile_h_px": height, "resampling": "bilinear (map_coordinates order=1)",
                             "orientation": "mask principal axis vertical", "pixel_size_um": px},
                description=f"Upright cell-cycle strip {args.pos} {args.ch} {args.f0}-{args.f1}, phase only, every frame",
                caption=caption,
                data={"canvas": canvas, "frames": seg.frame.to_numpy(), "time_from_birth_min": t_min,
                      "axis_angle_deg": np.array(angles), "pixel_size_um": px},
                data_source={"phase_dir": str(paths["phase_dir"]), "mask_dir": str(paths["mask_dir"]),
                             "lineage_csv": str(paths["lineage"])},
                dpi=600, fmt="png")
    plt.close(fig)
    print(out_dir / f"{stem}.png")
    if args.figma:
        export_figma(out_dir / f"figma_{stem}", tiles, seg.frame.to_numpy(), t_min, px, args.figma)
        print(out_dir / f"figma_{stem}")


if __name__ == "__main__":
    main()
