"""One-off: mother-cell axis time series for Pos1/z001/ch09 of the 260426
dataset. Two stacked panels — minor axis (short_axis_um, 0..6 um) and
major axis (long_axis_um, 0..15 um) — x-limit 0..24 h, default matplotlib
colours, publication-quality styling.

"Mother" here is the longest-lived cell in the channel because the
tracker's cell_id=0 dies at frame ~103 (rank-based initialisation
artefact on this dataset).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from figure_logger import save_figure

CSV = Path(
    r"H:/260426/online_crop_sub_zstack/Pos1/output_phase/channels/crop_sub_rawraw/z001"
    r"/ch09/inference_out/lineage_out/lineage_data3D.csv"
)
XLIM = (0.0, 24.0)
YLIM_SHORT = (0.0, 6.0)
YLIM_LONG = (0.0, 15.0)


def main() -> None:
    df = pd.read_csv(CSV)
    df = df.dropna(subset=["short_axis_um", "long_axis_um", "time_h"])

    # The tracker's cell_id=0 dies at frame ~103 in this dataset because the
    # rank-based initialisation picks an unstable rank-1 cell at img_137.
    # The biologically real mother is the longest-lived cell, which we
    # select here for a single clean trajectory.
    longest_cid = int(
        df.groupby("cell_id")["time_h"]
          .agg(lambda s: s.max() - s.min())
          .idxmax()
    )
    m = df[df["cell_id"] == longest_cid].sort_values("time_h")
    m = m[m["time_h"] <= XLIM[1] + 1e-9]
    if m.empty:
        print("No mother data in the requested window.")
        return
    print(f"mother cell_id = {longest_cid}  rows in window = {len(m)}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
    })

    fig, (ax_short, ax_long) = plt.subplots(
        2, 1,
        figsize=(4.2, 4.2),
        sharex=True,
        constrained_layout=True,
    )

    ax_short.plot(m["time_h"], m["short_axis_um"], linewidth=1.1)
    ax_short.set_ylim(*YLIM_SHORT)
    ax_short.set_yticks([0, 1, 2, 3, 4, 5, 6])
    ax_short.set_ylabel("minor axis [um]")
    ax_short.set_title(
        f"Pos1 / z001 / ch09 — mother (cell_id={longest_cid}, longest-lived)"
    )

    ax_long.plot(m["time_h"], m["long_axis_um"], linewidth=1.1, color="C1")
    ax_long.set_xlim(*XLIM)
    ax_long.set_ylim(*YLIM_LONG)
    ax_long.set_xticks([0, 4, 8, 12, 16, 20, 24])
    ax_long.set_yticks([0, 3, 6, 9, 12, 15])
    ax_long.set_xlabel("time [h]  (img_137 = 0)")
    ax_long.set_ylabel("major axis [um]")

    save_figure(
        fig,
        params={
            "pos": "Pos1", "z": "z001", "channel": "ch09",
            "mother_cell_id_used": longest_cid,
            "mother_selection_rule": "longest-lived cell (tracker cid=0 dies at frame ~103)",
            "x_axis_h": list(XLIM),
            "y_axis_short_um": list(YLIM_SHORT),
            "y_axis_long_um": list(YLIM_LONG),
            "x_axis": "time_h (img_137 mapped to 0)",
        },
        description=(
            "Mother cell axis time series for Pos1/z001/ch09 of the 260426 "
            "dataset (constant 2% glucose). Two panels: minor axis "
            "(short_axis_um, 0..6 um) and major axis (long_axis_um, 0..15 um). "
            "x-limit 0..24 h. Mother = longest-lived cell because the tracker's "
            "cell_id=0 dies at frame ~103 (rank-based init artefact)."
        ),
    )

    plt.close(fig)


if __name__ == "__main__":
    main()
