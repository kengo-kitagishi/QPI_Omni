"""Gold-standard phase1 cycle-aligned short_axis_um (mask minor axis).

Sanity check: is the mother's minor axis essentially constant across the
cell cycle (i.e. growth happens along long axis, S. pombe rod geometry)?
Compared side-by-side with long axis, volume, and mean RI for context.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    find_lineage_csv, select_gold_standard,
)
from gold_standard import rank1_division_frames  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018
N_REL = 100
LOW_MASS_PG = 10.0


def collect_cycles(channels):
    rel = np.linspace(0, 1, N_REL)
    rows = {"short_axis_um": [], "long_axis_um": [],
            "volume_um3_rod": [], "mean_ri": []}
    for pos, ch in channels:
        path = find_lineage_csv(pos, ch)
        if path is None:
            continue
        df = pd.read_csv(path)
        try:
            clist = pd.read_csv(path.parent / "clist.csv")
        except Exception:
            clist = None
        divs = rank1_division_frames(df, clist)
        m = df[df["rank"] == 1].sort_values("frame").copy()
        bad = (m["is_outlier"].astype(bool)
               | m["touches_border"].astype(bool)
               | (m["mass_pg"] < LOW_MASS_PG))
        m_valid = m[~bad]
        for i in range(len(divs) - 1):
            f0, f1 = divs[i], divs[i + 1]
            if f1 > PHASE1_END_FRAME:
                continue
            sub = m_valid[(m_valid["frame"] >= f0) & (m_valid["frame"] < f1)]
            if len(sub) < 4:
                continue
            t = sub["frame"].to_numpy().astype(float)
            t_rel = (t - f0) / (f1 - f0)
            for k in rows:
                rows[k].append(
                    np.interp(rel, t_rel, sub[k].to_numpy())
                )
    return rel, {k: np.vstack(v) for k, v in rows.items() if v}


def main():
    gold = select_gold_standard()
    print(f"gold-standard mothers: {len(gold)}")
    rel, M = collect_cycles(gold)
    n_cycles = M["short_axis_um"].shape[0]
    print(f"n_cycles pooled: {n_cycles}")

    panels = [
        ("short_axis_um", "short axis [μm]", "#0072B2", None),
        ("long_axis_um",  "long axis [μm]",  "#E69F00", None),
        ("volume_um3_rod", r"volume [μm³]",  "#009E73", None),
        ("mean_ri",       "mean RI",         "#CC79A7", (1.355, 1.375)),
    ]
    fig, axes = plt.subplots(len(panels), 1,
                             figsize=(89 / 25.4, 160 / 25.4),
                             sharex=True, constrained_layout=True)
    for ax, (key, ylabel, color, ylim) in zip(axes, panels):
        data = M[key]
        mean = np.nanmean(data, axis=0)
        sd = np.nanstd(data, axis=0)
        ax.fill_between(rel, mean - sd, mean + sd, color=color, alpha=0.25,
                        linewidth=0)
        ax.plot(rel, mean, color=color, linewidth=1.2,
                label=f"mean ± SD (n_cycles={n_cycles})")
        ax.set_ylabel(ylabel, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper left", frameon=False, fontsize=6)
        # report dynamic range
        dr = float(mean.max() - mean.min())
        rel_dr = dr / float(mean.mean())
        ax.set_title(
            f"{ylabel}: Δ(mean) = {dr:.3g}  "
            f"({100 * rel_dr:.1f} % of mean)",
            fontsize=7,
        )
    axes[-1].set_xlabel("relative cell cycle progression", fontsize=8)
    fig.suptitle(
        "Gold-standard phase1 cycle-aligned mother metrics",
        fontsize=9,
    )
    save_figure(
        fig,
        params={
            "selection": "gold-standard, phase1 only",
            "n_gold_mothers": len(gold),
            "n_cycles_pooled": int(n_cycles),
            "phase1_end_frame": PHASE1_END_FRAME,
        },
        description="gold-standard phase1 cycle-aligned short axis, long axis, volume, RI",
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
