"""Demonstrate systematic per-channel registration bias in cell-containing channels.

For a single position (default Pos1) of the z-stack drift log, the per-frame
phase-cross-correlation shift estimated independently in each channel is
compared between channels that contain cells and channels that are empty.

Cell-containing channels carry stationary, high-contrast intracellular
structure that biases the cross-correlation peak, so their per-frame x-shift
estimate is systematically offset from the true stage drift. Empty (cell-free)
channels agree with each other near zero. Averaging *all* channels therefore
injects a spurious, accumulating drift; averaging only the cell-free channels
does not. This script produces the three-panel figure that makes that case.

Panels
------
A  Per-channel distribution of the per-frame x-shift (tx). Cell channels sit
   systematically below zero; cell-free channels are centred on zero.
B  Per-channel distribution of the per-frame y-shift (ty). Control panel: no
   systematic group difference, so the bias is x-specific.
C  Cumulative drift over the time course for the naive all-channel mean vs the
   cell-free-only mean, showing the two estimates diverging by ~10^3 px.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figure_logger import save_figure  # noqa: E402

# Channels that contain cells in this dataset (Pos1). Everything else (except
# the always-failing tilt channel) is treated as cell-free.
CELL_CHANNELS = {0, 3, 5, 9, 10}
# Channel whose registration never converges (status tilt_bounds_ng); excluded.
DEAD_CHANNELS = {11}

CELL_COLOR = "#c0392b"      # crimson
FREE_COLOR = "#2980b9"      # blue


def channel_shift(ch_detail):
    """Refined (pass-2) shift when present, else the pass-1 estimate."""
    tx = ch_detail.get("tx2")
    ty = ch_detail.get("ty2")
    if tx is None:
        tx = ch_detail.get("tx1")
        ty = ch_detail.get("ty1")
    return tx, ty


def extract_position(records, pos_label):
    """Return tx[ch]->list, ty[ch]->list and per-frame all/free mean arrays."""
    tx = {}
    ty = {}
    all_mean_tx, all_mean_ty = [], []
    free_mean_tx, free_mean_ty = [], []

    for rec in records:
        pos = next((p for p in rec["positions"] if p["pos_label"] == pos_label), None)
        if pos is None:
            continue

        frame_tx, frame_ty = {}, {}
        for cd in pos["channel_details"]:
            ch = cd["ch"]
            if ch in DEAD_CHANNELS:
                continue
            sx, sy = channel_shift(cd)
            if sx is None or sy is None:
                continue
            tx.setdefault(ch, []).append(sx)
            ty.setdefault(ch, []).append(sy)
            frame_tx[ch] = sx
            frame_ty[ch] = sy

        if not frame_tx:
            continue
        free_tx = [v for c, v in frame_tx.items() if c not in CELL_CHANNELS]
        free_ty = [v for c, v in frame_ty.items() if c not in CELL_CHANNELS]
        if not free_tx:
            continue
        all_mean_tx.append(np.mean(list(frame_tx.values())))
        all_mean_ty.append(np.mean(list(frame_ty.values())))
        free_mean_tx.append(np.mean(free_tx))
        free_mean_ty.append(np.mean(free_ty))

    return (
        tx, ty,
        np.asarray(all_mean_tx), np.asarray(all_mean_ty),
        np.asarray(free_mean_tx), np.asarray(free_mean_ty),
    )


def _box_panel(ax, data_by_ch, channels, title, ylabel):
    positions = list(range(len(channels)))
    box = ax.boxplot(
        [data_by_ch[c] for c in channels],
        positions=positions,
        widths=0.62,
        showfliers=False,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="0.4"),
        capprops=dict(color="0.4"),
    )
    for patch, ch in zip(box["boxes"], channels):
        is_cell = ch in CELL_CHANNELS
        patch.set_facecolor(CELL_COLOR if is_cell else FREE_COLOR)
        patch.set_alpha(0.55)
        patch.set_edgecolor("0.25")

    ax.axhline(0.0, color="0.3", linestyle="--", linewidth=0.9, zorder=0)
    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c}" for c in channels])
    ax.set_xlabel("Channel")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figure(extracted, pos_label, px_um):
    tx, ty, all_tx, all_ty, free_tx, free_ty = extracted
    channels = sorted(tx.keys())

    fig = plt.figure(figsize=(11, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.42, wspace=0.22)
    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, :])

    # Panel A: per-channel x-shift -------------------------------------------
    _box_panel(axA, tx, channels, "A  Per-frame x-shift by channel",
               "x-shift  $t_x$  (px / frame)")
    cell_mean = np.mean(np.concatenate([tx[c] for c in channels if c in CELL_CHANNELS]))
    free_mean = np.mean(np.concatenate([tx[c] for c in channels if c not in CELL_CHANNELS]))
    axA.text(0.02, 0.04,
             f"cell mean = {cell_mean:+.2f} px\ncell-free mean = {free_mean:+.2f} px",
             transform=axA.transAxes, fontsize=8.5, va="bottom",
             bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # Panel B: per-channel y-shift (control) ---------------------------------
    _box_panel(axB, ty, channels, "B  Per-frame y-shift by channel (control)",
               "y-shift  $t_y$  (px / frame)")

    # Panel C: cumulative drift over time ------------------------------------
    frames = np.arange(len(all_tx))
    cum_all = np.cumsum(all_tx)
    cum_free = np.cumsum(free_tx)
    axC.plot(frames, cum_all, color="0.25", lw=1.6,
             label="all-channel mean (cell + cell-free)")
    axC.plot(frames, cum_free, color=FREE_COLOR, lw=1.6,
             label="cell-free channels only")
    axC.axhline(0.0, color="0.6", linestyle="--", linewidth=0.8, zorder=0)
    axC.set_xlabel("Frame")
    axC.set_ylabel("Cumulative x-drift  (px)")
    axC.set_title("C  Accumulated drift: averaging all channels vs cell-free only",
                  loc="left", fontweight="bold")
    axC.spines["top"].set_visible(False)
    axC.legend(frameon=False, loc="lower left", fontsize=9)

    gap = cum_all[-1] - cum_free[-1]
    axC.annotate(
        f"end gap = {gap:+.0f} px ({gap * px_um:+.1f} um)",
        xy=(frames[-1], cum_all[-1]),
        xytext=(0.62, 0.18), textcoords="axes fraction",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="0.4", lw=1.0),
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    # secondary micrometre axis on Panel C
    secax = axC.secondary_yaxis("right", functions=(lambda v: v * px_um,
                                                    lambda v: v / px_um))
    secax.set_ylabel("Cumulative x-drift  (um)")

    legend_handles = [
        Patch(facecolor=CELL_COLOR, alpha=0.55, edgecolor="0.25",
              label=f"cell channels {sorted(CELL_CHANNELS)}"),
        Patch(facecolor=FREE_COLOR, alpha=0.55, edgecolor="0.25",
              label="cell-free channels"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    fig.suptitle(
        f"Cell-containing channels bias the registration shift ({pos_label})",
        y=1.04, fontsize=13, fontweight="bold")
    return fig, dict(cell_mean_tx=cell_mean, free_mean_tx=free_mean,
                     end_gap_px=float(gap))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drift-log", default=r"f:\drift_log_zstack.json")
    ap.add_argument("--pos", default="Pos1")
    ap.add_argument("--px-um", type=float, default=0.34568,
                    help="pixel size in um (260517 default)")
    args = ap.parse_args()

    records = json.load(open(args.drift_log, encoding="utf-8"))
    extracted = extract_position(records, args.pos)
    if not extracted[0]:
        raise SystemExit(f"No data found for {args.pos}")

    fig, stats = make_figure(extracted, args.pos, args.px_um)

    tx, ty, all_tx, all_ty, free_tx, free_ty = extracted
    channels = sorted(tx.keys())
    n_frames = len(all_tx)

    # Re-plottable data sidecar: everything needed to redraw / restyle the
    # figure later without touching the raw drift log. Per-channel shift
    # samples are stored individually because they differ slightly in length.
    data = {
        "channels": np.asarray(channels),
        "cell_channels": np.asarray(sorted(CELL_CHANNELS)),
        "px_um": np.asarray(args.px_um),
        "frame": np.arange(n_frames),
        "all_mean_tx": all_tx,
        "all_mean_ty": all_ty,
        "free_mean_tx": free_tx,
        "free_mean_ty": free_ty,
        "cum_all_tx": np.cumsum(all_tx),
        "cum_free_tx": np.cumsum(free_tx),
    }
    for c in channels:
        data[f"tx_ch{c}"] = np.asarray(tx[c])
        data[f"ty_ch{c}"] = np.asarray(ty[c])
    out = save_figure(
        fig,
        params={
            "pos": args.pos,
            "cell_channels": sorted(CELL_CHANNELS),
            "dead_channels": sorted(DEAD_CHANNELS),
            "px_um": args.px_um,
            "n_frames": n_frames,
            **stats,
        },
        description=(
            f"Systematic registration bias in cell-containing channels ({args.pos}). "
            f"Cell channels {sorted(CELL_CHANNELS)} show a mean per-frame x-shift of "
            f"{stats['cell_mean_tx']:+.2f} px vs {stats['free_mean_tx']:+.2f} px for "
            f"cell-free channels; averaging all channels accumulates a spurious "
            f"{stats['end_gap_px']:+.0f} px x-drift over {n_frames} frames relative "
            f"to averaging cell-free channels only. Justifies cell-free-only averaging."
        ),
        data_source={"drift_log": str(args.drift_log)},
        data=data,
        dpi=200,
    )
    print(f"saved: {out}")
    print(f"cell mean tx = {stats['cell_mean_tx']:+.3f} px ; "
          f"cell-free mean tx = {stats['free_mean_tx']:+.3f} px ; "
          f"end gap = {stats['end_gap_px']:+.1f} px")


if __name__ == "__main__":
    main()
