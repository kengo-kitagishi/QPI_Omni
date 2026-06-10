"""Focused 3-metric overlay for elongation-cascade death channels.

The YAML classifies only two phase1-dead channels as catastrophic elongation
cascade: Pos20 ch06 (cascade start frame 1250, lysis ~1561) and Pos30 ch04
(elongation start frame 900, lysis ~1300). This script plots their mother
volume / mean RI / dry mass against the gold-standard envelope so the
physical signature of elongation death can be read off.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    select_gold_standard, load_mother_phase1, _pick_metric, PHASE1_T_MAX,
)
from figure_logger import save_figure  # noqa: E402

ELONGATION_PAIR = [
    ("Pos20", "ch06", 1250, "cascade frame 1250 → lysis 1561"),
    ("Pos30", "ch04",  900, "elongation frame 900 → lysis 1300"),
]
TIME_INTERVAL_MIN = 5.0


def plot_one_metric(gold, metric: str, ylabel: str, ylim, out_label: str):
    fig, ax = plt.subplots(figsize=(7.20, 3.60))
    # gold-standard background (thin gray lines)
    for pos, ch in gold:
        traj = load_mother_phase1(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        ax.plot(t, y, color="#4a4a4a", linewidth=0.15, alpha=0.18,
                zorder=1, solid_capstyle="butt")
    palette = ["#56B4E9", "#D55E00"]  # blue, orange-red
    for i, (pos, ch, death_frame, note) in enumerate(ELONGATION_PAIR):
        traj = load_mother_phase1(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        color = palette[i]
        ax.plot(t, y, color=color, linewidth=0.8, alpha=0.9,
                zorder=10 + i, solid_capstyle="butt",
                label=f"{pos}_{ch} ({note})")
        # vertical line at YAML-reported death frame
        death_h = death_frame * TIME_INTERVAL_MIN / 60.0
        ax.axvline(death_h, color=color, linewidth=0.5,
                   linestyle=":", alpha=0.6, zorder=5)
    ax.set_xlim(0, PHASE1_T_MAX)
    ax.set_ylim(*ylim)
    ax.set_xlabel("time [h]", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    metric_pretty = {
        "mean_RI": "refractive index",
        "volume": "volume",
        "mass": "dry mass",
    }.get(metric, metric)
    ax.legend(loc="upper right", fontsize=7, frameon=False,
              title=f"elongation-cascade death (surviving n={len(gold)})",
              title_fontsize=7)
    ax.set_title(
        f"Mother cell {metric_pretty}: two lineages undergoing "
        f"elongation-cascade death vs surviving lineages",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(
        fig,
        params={
            "metric": metric,
            "pair": "Pos20_ch06 + Pos30_ch04",
            "n_gold": len(gold),
            "phase1_t_max_h": PHASE1_T_MAX,
        },
        description=f"elongation pair overlay {metric} {out_label}",
    )
    plt.close(fig)


def main():
    gold = select_gold_standard()
    print(f"Gold-standard: {len(gold)}")
    print(f"Elongation pair: {[(p, c) for p, c, _, _ in ELONGATION_PAIR]}")
    plot_one_metric(gold, "volume", r"mother volume [$\mu m^3$]",
                    (0, 300), "volume")
    plot_one_metric(gold, "mean_RI", "mother mean RI", (1.345, 1.385),
                    "mean_RI")
    plot_one_metric(gold, "mass", "mother dry mass [pg]", (0, 80),
                    "mass")


if __name__ == "__main__":
    main()
