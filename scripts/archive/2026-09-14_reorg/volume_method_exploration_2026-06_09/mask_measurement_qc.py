"""mask_measurement_qc.py — visual QC of the rotate-and-project width measurement.

The width methods assume each column of the rotated cell crosses ONE contiguous
mask segment. When a column crosses several segments — a bent cell, a division
neck, or a neighbour/debris blob caught in the crop — the column sum over-counts
the width and the medial axis jumps. This script makes that failure mode visible.

For a channel it:
  1. Re-extracts the mother (rank=1) mask per phase1 frame (same centroid lookup
     as recompute_axes_from_masks).
  2. Counts cross-sections per column (vertical connected runs) and the fraction
     of body columns with >1 run.
  3. Renders a montage of the worst (most multi-section) frames plus evenly
     sampled typical frames, each overlaid with the smoothed medial axis and the
     body window used for the short-axis average.
  4. Saves a per-frame timeseries of the multi-section fraction so globally bad
     frames are obvious.

Usage:
    python scripts/mask_measurement_qc.py --pos Pos27 --ch ch06
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import measure, transform

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import find_lineage_csv, mask_dir_for_channel, find_run_params  # noqa: E402
from recompute_axes_from_masks import (  # noqa: E402
    build_mask_index, _load_label_image, _label_at_centroid,
)
from mask_morphology import (  # noqa: E402
    measure_single_cell_medial, _moving_average,
    rotate_mask_horizontal, column_run_counts,
)
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018


def frame_diagnostics(rotated: np.ndarray):
    """Return dict of per-frame diagnostics from a rotated mother mask."""
    col_sums = rotated.sum(axis=0).astype(float)
    nz = np.where(col_sums > 0)[0]
    if len(nz) < 2:
        return None
    c0, c1 = nz[0], nz[-1]
    n_col = c1 - c0 + 1
    xsec = column_run_counts(rotated, c0, c1)
    # body window (exclude caps ~ short/2 px each end), same idea as the methods
    h_vert = col_sums[c0:c1 + 1]
    trim0 = int(0.10 * n_col)
    rough = float(np.mean(h_vert[trim0:n_col - trim0])) if n_col > 2 * trim0 + 1 else float(np.mean(h_vert))
    cap = int(np.clip(rough / 2.0, 1, max(n_col // 3, 1)))
    body = np.zeros(n_col, dtype=bool)
    if n_col > 2 * cap + 1:
        body[cap:n_col - cap] = True
    else:
        body[:] = True
    multi_frac = float(np.mean(xsec[body] > 1)) if body.any() else 0.0
    # medial axis (column centroid, smoothed)
    rows_idx = np.arange(rotated.shape[0], dtype=float)[:, None]
    col_slice = rotated[:, c0:c1 + 1].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        y_c = (rows_idx * col_slice).sum(axis=0) / np.maximum(col_slice.sum(axis=0), 1e-9)
    y_smooth = _moving_average(y_c, max(int(0.15 * n_col), 3))
    return {
        "c0": c0, "c1": c1, "n_col": n_col, "cap": cap, "body": body,
        "xsec": xsec, "max_xsec": int(xsec.max()),
        "multi_frac": multi_frac, "y_smooth": y_smooth,
    }


def collect(pos: str, ch: str, frame_max: int = PHASE1_END_FRAME):
    csv_path = find_lineage_csv(pos, ch)
    inf_dir = mask_dir_for_channel(csv_path)
    df = pd.read_csv(csv_path)
    m = df[(df["rank"] == 1) & (df["frame"] <= frame_max)].sort_values("frame")
    idx = build_mask_index(inf_dir)
    recs = []
    for _, r in m.iterrows():
        frame = int(r["frame"])
        mpath = idx.get(frame)
        if mpath is None:
            continue
        mask = _load_label_image(mpath)
        lbl = _label_at_centroid(mask, r["centroid_y_px"], r["centroid_x_px"])
        if lbl == 0:
            continue
        binary = mask == lbl
        rot, _ = rotate_mask_horizontal(binary)
        if rot is None or not rot.any():
            continue
        diag = frame_diagnostics(rot)
        if diag is None:
            continue
        diag["frame"] = frame
        diag["rot"] = rot
        diag["short_um_medial"] = measure_single_cell_medial(binary, pixel_size_um=1.0).short_axis_px
        recs.append(diag)
    return recs


def plot_montage(pos, ch, recs):
    # 6 worst (highest multi_frac) + 6 evenly-sampled typical
    by_bad = sorted(recs, key=lambda d: d["multi_frac"], reverse=True)
    worst = by_bad[:6]
    worst_frames = {d["frame"] for d in worst}
    typical_pool = [d for d in recs if d["frame"] not in worst_frames]
    if typical_pool:
        sel = np.linspace(0, len(typical_pool) - 1, 6).astype(int)
        typical = [typical_pool[i] for i in sel]
    else:
        typical = []
    panels = worst + typical
    labels = ["WORST"] * len(worst) + ["typical"] * len(typical)

    n = len(panels)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 3.2, nrow * 2.0))
    axes = np.atleast_1d(axes).ravel()
    for ax, d, tag in zip(axes, panels, labels):
        rot = d["rot"]
        ax.imshow(rot, cmap="gray_r", interpolation="nearest", aspect="equal")
        c0, c1 = d["c0"], d["c1"]
        xs = np.arange(c0, c1 + 1)
        ax.plot(xs, d["y_smooth"], color="#0072B2", lw=0.8)
        # mark multi-cross-section columns in red along the medial axis
        multi = d["xsec"] > 1
        if multi.any():
            ax.scatter(xs[multi], d["y_smooth"][multi], s=6, color="#D55E00", zorder=5)
        # body window vertical guides
        body = d["body"]
        if body.any():
            bx = xs[body]
            ax.axvline(bx[0], color="#009E73", lw=0.5, ls="--")
            ax.axvline(bx[-1], color="#009E73", lw=0.5, ls="--")
        # zoom the view to the cell bbox (+pad) so the contour is inspectable
        ys_nz, xs_nz = np.where(rot)
        pad = 6
        ax.set_xlim(xs_nz.min() - pad, xs_nz.max() + pad)
        ax.set_ylim(ys_nz.max() + pad, ys_nz.min() - pad)  # inverted (image coords)
        color = "#D55E00" if tag == "WORST" else "#333333"
        ax.set_title(f"[{tag}] f{d['frame']}  2r={d['short_um_medial']:.1f}px\n"
                     f"max_xsec={d['max_xsec']}  multi={100*d['multi_frac']:.0f}%",
                     fontsize=6.5, color=color)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{pos}_{ch} mother mask QC — rotated mask + medial axis "
                 f"(red = column with >1 cross-section)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(
        fig,
        params={"pos": pos, "ch": ch, "n_frames_qc": len(recs),
                "panels": n, "phase1_end_frame": PHASE1_END_FRAME},
        description=f"{pos}_{ch} mask measurement QC montage (cross-sections)",
    )
    plt.close(fig)


def plot_timeseries(pos, ch, recs):
    frames = np.array([d["frame"] for d in recs])
    multi = np.array([100 * d["multi_frac"] for d in recs])
    maxx = np.array([d["max_xsec"] for d in recs])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    ax1.plot(frames, multi, color="#D55E00", lw=0.6)
    ax1.set_ylabel("body cols with\n>1 cross-section [%]", fontsize=7)
    ax1.axhline(10, color="gray", lw=0.5, ls=":")
    ax1.tick_params(labelsize=6)
    ax2.plot(frames, maxx, color="#0072B2", lw=0.6)
    ax2.set_ylabel("max cross-sections\nper column", fontsize=7)
    ax2.set_xlabel("frame", fontsize=7)
    ax2.tick_params(labelsize=6)
    n_bad = int(np.sum(multi > 10))
    fig.suptitle(f"{pos}_{ch} cross-section QC over phase1  "
                 f"({n_bad}/{len(frames)} frames >10% multi-section)", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(
        fig,
        params={"pos": pos, "ch": ch, "n_frames": len(frames),
                "n_frames_multi_gt10pct": n_bad},
        description=f"{pos}_{ch} cross-section fraction over phase1",
        data={"frame": frames, "multi_frac_pct": multi, "max_xsec": maxx},
    )
    plt.close(fig)
    return n_bad, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", required=True)
    ap.add_argument("--ch", required=True)
    ap.add_argument("--frame-max", type=int, default=PHASE1_END_FRAME)
    args = ap.parse_args()
    recs = collect(args.pos, args.ch, args.frame_max)
    print(f"{args.pos}_{args.ch}: {len(recs)} frames diagnosed")
    if not recs:
        sys.exit(1)
    n_bad, n_tot = plot_timeseries(args.pos, args.ch, recs)
    plot_montage(args.pos, args.ch, recs)
    worst = max(recs, key=lambda d: d["multi_frac"])
    print(f"worst frame f{worst['frame']}: multi={100*worst['multi_frac']:.1f}%  "
          f"max_xsec={worst['max_xsec']}")
    print(f"frames with >10% multi-section columns: {n_bad}/{n_tot}")


if __name__ == "__main__":
    main()
