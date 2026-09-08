"""_fig_meanri_divlines_swelling.py — mean RI(t) with division markers, swelling.

For every swelling-death lineage, the EFD cycle-mean-RI trace mean_ri(t) up to its
curated death window, with a vertical dotted line at each division (cycle end). Lets
you read the within-cycle RI pattern relative to the cell cycle (the ~3 h pre-death
RI dip the user is interested in). One panel per lineage; --ncols controls layout.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_meanri_divlines_swelling.py [--ncols N]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gold_standard import list_phase1_dead, MANUAL_EXCLUDE  # noqa: E402
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_growthrate_dead_vs_alive import per_cycle_rates  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

SWELL_C = "#b25450"
DIV_C = "#888888"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


PRE_H = 20.0   # hours to show before the last division
POST_H = 12.0  # hours to show after the last division (terminal / swelling phase)


def draw_panel(ax, ch_key, data):
    pos, ch = ch_key.split("_", 1)
    f1 = DEATH_WINDOW[ch_key][1]                    # curated last division (panelA)
    m = load_mother(pos, ch)
    rates = per_cycle_rates(m, f1)                  # cycle ends -> division markers
    pre = m[m["frame"] <= f1]
    if pre.empty:
        ax.axis("off")
        return
    t_ld = float(pre["time_h"].max())               # x origin (0) = curated last division
    v = m[~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)]
    t = v["time_h"].to_numpy(float) - t_ld
    ri = v["mean_ri"].to_numpy(float)
    win = (t >= -PRE_H) & (t <= POST_H)             # raw window; NO lysis truncation
    t, ri = t[win], ri[win]
    ax.axvspan(0, POST_H, color=SWELL_C, alpha=0.07, lw=0, zorder=0)  # post-division
    div_t = [r["t1"] - t_ld for r in rates if -PRE_H <= r["t1"] - t_ld <= POST_H]
    for dt_ in div_t:
        ax.axvline(dt_, color=DIV_C, lw=0.6, ls=":", alpha=0.8, zorder=1)
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7, zorder=2)   # last division (f1)
    ax.plot(t, ri, "-", color=SWELL_C, lw=0.8, marker="o", ms=1.6, mew=0, zorder=3)
    ax.set_xlim(-PRE_H, POST_H)
    ax.set_title(f"{ch_key}  g{len(rates)}", fontsize=6)
    ax.tick_params(length=2)
    data[f"{ch_key}_t_from_lastdiv_h"] = t
    data[f"{ch_key}_mean_ri"] = ri
    data[f"{ch_key}_division_t"] = np.array(div_t, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncols", type=int, default=5)
    a = ap.parse_args()
    ncols = max(1, a.ncols)

    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    swelling = [pc for pc in dead if f"{pc[0]}_{pc[1]}" not in ELONGATION]
    # keep only lineages with a fittable trace, sorted by lifespan (n cycles)
    keep = []
    for pos, ch in swelling:
        k = f"{pos}_{ch}"
        m = load_mother(pos, ch)
        n = len(per_cycle_rates(m, DEATH_WINDOW[k][1]))
        if n >= 1:
            keep.append((k, n))
    keep.sort(key=lambda kv: -kv[1])
    n = len(keep)
    print(f"swelling lineages: {n}")

    nrows = (n + ncols - 1) // ncols
    width_mm = 110 if ncols == 1 else 183
    rowh_mm = 28 if ncols == 1 else 36
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_mm / 25.4, (rowh_mm * nrows + 8) / 25.4),
                             sharey=True, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    data: dict = {}
    for ax, (ch_key, _n) in zip(axes, keep):
        draw_panel(ax, ch_key, data)
    for ax in axes[n:]:
        ax.axis("off")
    for ax in axes[:n]:
        ax.set_xlabel("time from last division [h]")
        ax.set_ylabel("mean RI")
    fig.suptitle(
        f"mean RI vs time — {PRE_H:g} h before to {POST_H:g} h after the last "
        f"division; swelling-death lineages (EFD, n={n}). grey dotted = division, "
        "black dashed = last division, shaded = post-division (terminal) phase. "
        "Shared y-axis.",
        fontsize=6.5)

    caption = (
        "Cycle-mean refractive index of each swelling-death lineage from "
        f"{PRE_H:g} h before to {POST_H:g} h after its last division (the terminal / "
        "swelling phase), with division timings marked, so the within-cell-cycle RI "
        "pattern and any post-division change can be read against the cell cycle. Each "
        "panel is one swelling-death lineage. x = time from last division [h] (last "
        f"division at 0; window -{PRE_H:g}..+{POST_H:g} h, common to all panels); "
        "y = mean RI (shared across panels). Operational "
        "definitions: mean RI = EFD-based mean_ri of the rank-1 mother per frame "
        "(valid frames only: not is_outlier/touches_border, mass_pg >= 10); grey "
        "dotted vertical line = a division, taken as the end (last frame, time t1) of "
        "each division-bounded cell cycle from enumerate_all_cycles; the last division "
        "(x = 0) is the curated death point f1 = DEATH_WINDOW (fig_panelA 2nd-to-last "
        "row), drawn as the black dashed line; shaded span = the "
        f"{POST_H:g} h after it, shown raw with no truncation (the trench may begin to "
        "repopulate the mother position after the curated last division). swelling "
        "death = phase1-dead lineages excluding the two "
        "elongation cascades (Pos20_ch06, Pos30_ch04). last division = end of the last "
        "completed cell cycle (the curated death window from the fig_panelA "
        f"2nd-to-last row). n = {n} swelling lineages, one panel each, sorted by "
        "lifespan (cycle count). Statistics: descriptive (individual lineages; no "
        "test). "
        "Conditions: Schizosaccharomyces pombe, 260517 Mother-Machine dataset, EMM + "
        "2% glucose, phase1 (frame <= 2018), QPI EFD volume variant [strain/genotype "
        "and temperature: confirm from acquisition metadata]. Abbreviations: RI, "
        "refractive index; EFD, elliptic Fourier descriptor. Data availability: "
        "per-lineage RI trace and division times in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "ncols": ncols, "n_swelling": n},
        description="cycle-mean RI(t) per swelling-death lineage with division dotted "
                    "lines (EFD), one panel each",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
