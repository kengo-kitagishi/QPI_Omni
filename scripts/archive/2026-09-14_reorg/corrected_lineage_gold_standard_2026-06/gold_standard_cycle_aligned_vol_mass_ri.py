"""Gold-standard phase1 cycle-aligned mother volume / dry mass / mean RI.

A clean, publication-style replacement for the old "cycle-aligned mother
trajectories" figure: each metric is pooled over all gold-standard phase1 cell
cycles, interpolated onto relative cycle progression (0..1), and shown as a
mean +/- SD band (no per-cycle spaghetti). Mirrors the styling of
gold_standard_minor_axis_cycle.py.

Corrected-volume aware: reads the mask-direct corrected lineage when
QPI_USE_CORRECTED=1 (via qpi_paths.resolve_lineage_csv, used by find_lineage_csv).
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
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    find_lineage_csv, select_gold_standard,
)
from gold_standard import rank1_division_frames  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018
N_REL = 100
LOW_MASS_PG = 10.0


def collect_cycles(channels):
    """Pool every gold-standard phase1 cycle onto a 0..1 relative-progress grid."""
    rel = np.linspace(0, 1, N_REL)
    rows = {"volume_um3_rod": [], "mass_pg": [], "mean_ri": [], "conc_mg_ml": []}
    n_mothers = 0
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
        # dry-mass concentration [mg/mL] = 1000 * mass[pg] / volume[um^3]
        # (pg/um^3 == g/mL, so x1000 -> mg/mL). Consistent with how mass_pg was
        # derived from RI, i.e. the QPI dry-mass density.
        m["conc_mg_ml"] = 1000.0 * m["mass_pg"] / m["volume_um3_rod"]
        bad = (m["is_outlier"].astype(bool)
               | m["touches_border"].astype(bool)
               | (m["mass_pg"] < LOW_MASS_PG))
        m_valid = m[~bad]
        used = False
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
                rows[k].append(np.interp(rel, t_rel, sub[k].to_numpy()))
            used = True
        n_mothers += int(used)
    return rel, {k: np.vstack(v) for k, v in rows.items() if v}, n_mothers


def main():
    gold = select_gold_standard()
    rel, M, n_mothers = collect_cycles(gold)
    n_cycles = M["volume_um3_rod"].shape[0]
    print(f"gold mothers={len(gold)} used={n_mothers} cycles_pooled={n_cycles}")

    # Cell-to-cell dry-mass CV at the two cycle endpoints. The interp grid maps
    # rel=0 -> birth and rel=1 -> the last frame before division (div_frame-1,
    # since np.interp clamps to the last sampled value). One row per pooled
    # cycle, so the denominator (n) matches n_cycles in the legend.
    mass = M["mass_pg"]
    birth, last = mass[:, 0], mass[:, -1]
    cv_birth = float(np.nanstd(birth, ddof=1) / np.nanmean(birth) * 100.0)
    cv_div = float(np.nanstd(last, ddof=1) / np.nanmean(last) * 100.0)
    print(f"dry-mass CV: birth={cv_birth:.2f}% div={cv_div:.2f}% "
          f"(n_cycles={n_cycles})")

    panels = [
        ("volume_um3_rod", r"volume [μm³]",          "#0072B2", None),
        ("mass_pg",        "dry mass [pg]",           "#D55E00", None),
        ("mean_ri",        "mean RI",                 "#009E73", (1.355, 1.375)),
        ("conc_mg_ml",     "dry-mass conc. [mg/mL]",  "#CC79A7", None),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(89 / 25.4, 168 / 25.4),
                             sharex=True, constrained_layout=True)
    # Complete plot data for inbox: the relative-progress grid, every panel's
    # full per-cycle matrix (n_cycles x N_REL), and the plotted mean/SD curves.
    # The matrices are sufficient to recompute mean/SD, the birth/division CVs
    # (columns 0 and -1) and to restyle the figure entirely from the npz.
    plot_data = {"rel": rel, "n_cycles": np.array(n_cycles)}
    for ax, (key, ylabel, color, ylim) in zip(axes, panels):
        data = M[key]
        mean = np.nanmean(data, axis=0)
        sd = np.nanstd(data, axis=0)
        ax.fill_between(rel, mean - sd, mean + sd, color=color, alpha=0.25,
                        linewidth=0)
        ax.plot(rel, mean, color=color, linewidth=1.4,
                label=f"mean ± SD (n_cycles={n_cycles})")
        ax.set_ylabel(ylabel, fontsize=8)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=7)
        ax.legend(loc="upper left", frameon=False, fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        plot_data[f"{key}_matrix"] = data   # (n_cycles, N_REL)
        plot_data[f"{key}_mean"] = mean
        plot_data[f"{key}_sd"] = sd
    axes[-1].set_xlabel("relative cell-cycle progression", fontsize=8)
    save_figure(
        fig,
        params={"selection": "gold-standard, phase1 only",
                "n_gold_mothers": len(gold), "n_mothers_used": int(n_mothers),
                "n_cycles_pooled": int(n_cycles),
                "phase1_end_frame": PHASE1_END_FRAME},
        description="gold-standard phase1 cycle-aligned volume / dry mass / mean RI "
                    "/ dry-mass concentration mg/mL (corrected volume, mean +/- SD band)",
        data=plot_data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
