"""One-off: minor-axis (short_axis_um) vs time, all completed channels of
Pos1/z001 in a single multi-panel figure, publication-quality styling.

Reads each `<ch>/inference_out/lineage_out/lineage_data3D.csv` from the
260426 dataset (constant 2% glucose, frame_min=137, frame_max=1000), then
plots the mother cell (cell_id=0) trajectory in saturated vermilion plus
every other tracked cell as thin gray lines, one panel per channel.

y-axis is locked at 0..6 um as requested.

Uses figure_logger.save_figure so the PDF and a CSV sidecar land in the
figure-hub inbox automatically.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure_logger import save_figure

OI = {
    "vermilion": "#D55E00",   # longest-lived cell = de-facto mother
    "blue":      "#0072B2",   # cell_id == 0 (rank-1 at frame 0, often a wash-out)
    "gray":      "#999999",
    "gray_light":"#CCCCCC",
}

POS_Z = Path(
    r"H:/260426/online_crop_sub_zstack/Pos1/output_phase/channels/crop_sub_rawraw/z001"
)
CHANNELS = [f"ch{i:02d}" for i in range(11)]
YLIM = (0.0, 6.0)


def load_one(ch_dir: Path) -> pd.DataFrame | None:
    csv = ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    if df.empty or "short_axis_um" not in df.columns:
        return None
    return df


def main() -> None:
    panels: list[tuple[str, pd.DataFrame]] = []
    for ch in CHANNELS:
        df = load_one(POS_Z / ch)
        if df is not None and len(df) > 1:
            panels.append((ch, df))

    if not panels:
        print("No data found; nothing to plot.")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })

    n = len(panels)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.4 * ncols, 1.7 * nrows),
        sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_flat = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (ch, df) in zip(axes_flat, panels):
        # De-facto mother = longest-lived cell (the tracker's cell_id=0 is
        # often a transient wash-out cell that dies after ~100 frames; see
        # diagnostics above). cell_id=0 still gets a blue overlay so the
        # discrepancy stays visible.
        lifespans = df.groupby("cell_id")["time_h"].agg(lambda s: s.max() - s.min())
        true_mother_cid = int(lifespans.idxmax())
        for cid, sub in df.groupby("cell_id"):
            sub = sub.sort_values("time_h")
            cid_int = int(cid)
            if cid_int == true_mother_cid:
                color, lw, alpha, z = OI["vermilion"], 1.2, 1.0, 4
            elif cid_int == 0:
                color, lw, alpha, z = OI["blue"], 1.0, 0.9, 3
            else:
                color, lw, alpha, z = OI["gray_light"], 0.5, 0.7, 1
            ax.plot(sub["time_h"], sub["short_axis_um"],
                    color=color, linewidth=lw, alpha=alpha, zorder=z)
        ax.set_title(f"Pos1 {ch}  (true mother cell_id={true_mother_cid})", fontsize=7.5)
        ax.set_ylim(*YLIM)
        ax.set_yticks([0, 1, 2, 3, 4, 5, 6])
        ax.tick_params(labelsize=7)

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    for i, ax in enumerate(axes_flat[:len(panels)]):
        row, col = divmod(i, ncols)
        if row == nrows - 1 or i + ncols >= len(panels):
            ax.set_xlabel("time [h]  (img_137 = 0)", fontsize=8)
        if col == 0:
            ax.set_ylabel("minor axis [um]", fontsize=8)

    fig.suptitle(
        "Pos1 / z001 — minor axis vs time   "
        "[vermilion: longest-lived cell (de-facto mother)   "
        "blue: tracker cell_id=0 (early death)   gray: others]",
        fontsize=8,
    )

    save_figure(
        fig,
        params={
            "pos": "Pos1",
            "z": "z001",
            "channels_plotted": [c for c, _ in panels],
            "y_axis_um": list(YLIM),
            "x_axis": "time_h (frame_min=137 mapped to 0)",
            "ri_calibration": "H:/260423/grid_2pergluc_1/ri_calibration_results.json",
            "media_schedule": "0:wo_2",
            "n_milliq": 1.33,
        },
        description=(
            "Minor-axis time series for all completed channels of Pos1/z001 "
            "in the 260426 dataset (constant 2% glucose). Mother cell "
            "(cell_id=0) in vermilion, all other tracked cells in light gray. "
            "y-axis locked 0..6 um. img_137 maps to time 0."
        ),
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
