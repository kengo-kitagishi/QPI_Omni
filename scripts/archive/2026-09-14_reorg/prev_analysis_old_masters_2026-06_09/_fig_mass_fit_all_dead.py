"""_fig_mass_fit_all_dead.py — mother dry mass(t) + per-cycle exp fit, ALL dead.

The dead-vs-alive script (_fig_growthrate_dead_vs_alive.figure_mass_fit) only
draws ONE representative dead lineage. This reuses the same parts
(collect_group / per_cycle_rates / _rep_mother / DEATH_WINDOW) to draw EVERY
phase1-dead lineage in a grid: dry mass(t) up to its curated death window, with
the per-cycle exponential fit M0*exp(k t) overlaid (the same fits whose slopes
are d ln M/dt). Swelling deaths in muted red, the 2 elongation cascades in green;
each panel's LAST cycle (terminal cell cycle before death) drawn thicker, with
its terminal d ln M/dt annotated. EFD volume variant.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_mass_fit_all_dead.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gold_standard import list_phase1_dead, MANUAL_EXCLUDE  # noqa: E402
from _fig_growthrate_dead_vs_alive import collect_group, _rep_mother  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

SWELL_C = "#b25450"   # muted red (swelling death)
ELONG_C = "#009E73"   # green (elongation cascade)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def draw_panel(ax, ch_key, rates, col, data):
    """dry mass(t) up to the death window + per-cycle exp fit overlay."""
    mx = DEATH_WINDOW[ch_key][1]
    m = _rep_mother(ch_key, mx)
    t = m["time_h"].to_numpy(float)
    mass = m["mass_pg"].to_numpy(float)
    ax.plot(t, mass, "o", color=col, ms=1.6, alpha=0.55, mew=0)
    for j, r in enumerate(rates):           # per-cycle exponential fit overlay
        tt = np.linspace(r["t0"], r["t1"], 20)
        last = (j == len(rates) - 1)
        ax.plot(tt, np.exp(r["ic"] + r["rate"] * tt),
                color="#222" if not last else col,
                lw=0.8 if not last else 1.8, ls="--" if not last else "-",
                zorder=5 if not last else 6)
    term = rates[-1]["rate"]
    ax.set_title(f"{ch_key}  g{len(rates)}  k$_t$={term:.2f}", fontsize=6)
    ax.tick_params(length=2)
    data[f"{ch_key}_time_h"] = t
    data[f"{ch_key}_mass_pg"] = mass
    data[f"{ch_key}_fit_rate"] = np.array([r["rate"] for r in rates])
    data[f"{ch_key}_fit_ic"] = np.array([r["ic"] for r in rates])
    data[f"{ch_key}_fit_t0"] = np.array([r["t0"] for r in rates])
    data[f"{ch_key}_fit_t1"] = np.array([r["t1"] for r in rates])
    data[f"{ch_key}_death_mode"] = np.array("elongation" if col == ELONG_C
                                            else "swelling")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ncols", type=int, default=5,
                    help="panels per row (use 1 for a single tall column)")
    a = ap.parse_args()
    ncols = max(1, a.ncols)

    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    dead_data = collect_group(dead, lambda pos, ch: DEATH_WINDOW[f"{pos}_{ch}"][1])
    print(f"dead lineages kept (>=3 cycles): {len(dead_data)}")

    # swelling first (sorted by lifespan, descending), elongation cascades last
    def is_elong(k):
        return k in ELONGATION
    items = sorted(dead_data.items(),
                   key=lambda kv: (is_elong(kv[0]), -len(kv[1])))
    n = len(items)
    nrows = (n + ncols - 1) // ncols
    width_mm = 110 if ncols == 1 else 183
    rowh_mm = 30 if ncols == 1 else 38
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(width_mm / 25.4, (rowh_mm * nrows + 8) / 25.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    data: dict = {}
    n_swell = n_elong = 0
    for ax, (ch_key, rates) in zip(axes, items):
        col = ELONG_C if is_elong(ch_key) else SWELL_C
        if col == ELONG_C:
            n_elong += 1
        else:
            n_swell += 1
        draw_panel(ax, ch_key, rates, col, data)
    for ax in axes[n:]:
        ax.axis("off")
    # shared axis labels
    for ax in axes[:n]:
        ax.set_xlabel("time [h]")
        ax.set_ylabel("dry mass [pg]")
    fig.suptitle(
        f"Mother dry mass(t) + per-cycle exp fit M$_0$e$^{{kt}}$ — all phase1-dead "
        f"lineages (EFD). Red: swelling (n={n_swell}); green: elongation cascade "
        f"(n={n_elong}). Solid coloured = terminal (last) cell cycle; k$_t$ = its "
        f"d ln M/dt [h$^{{-1}}$]. Each trace runs to its curated death window.",
        fontsize=6.5)

    caption = (
        "Every phase1-dead lineage's mother dry mass grows exponentially cycle by "
        "cycle right up to its terminal cell cycle, with no obvious slow-down in the "
        "last fitted cycle for most lineages — consistent with death being a short "
        "terminal event rather than a gradual decline visible in the mass trajectory. "
        "Each panel is one dying lineage. x = time [h]; y = dry mass [pg]. Operational "
        "definitions: dry mass = EFD-based mass_pg of the rank-1 mother; points = "
        "per-frame mass over the lineage up to its curated death window (valid frames "
        "only: not is_outlier/touches_border, mass_pg >= 10). Dashed black line = "
        "per-cycle exponential fit M(t)=exp(ic + k t) where (k, ic) = OLS fit of "
        "ln(mass_pg) vs time_h over each division-bounded cycle [birth_frame, "
        "div_frame-1] (>=4 valid frames); k of each cycle is that cycle's d ln M/dt. "
        "The solid coloured line = the LAST (terminal) cycle's fit; the panel title "
        "gives generation count gN and terminal slope k_t [1/h]. Swelling death = "
        "phase1-dead lineages excluding the two elongation cascades (red); elongation "
        "death = Pos20_ch06, Pos30_ch04 (green). death window = last completed cell "
        "cycle (last division before death / before elongation onset), curated from "
        "the fig_panelA 2nd-to-last row; the contaminated post-lysis trace beyond it "
        "is excluded by design, so the dramatic terminal elongation of the 2 cascades "
        f"is NOT shown. n: swelling = {n_swell} lineages, elongation = {n_elong} "
        "lineages (lineages with < 3 fitted cycles are dropped). Statistics: "
        "descriptive (individual lineages; no test). Conditions: Schizosaccharomyces "
        "pombe, 260517 Mother-Machine dataset, EMM + 2% glucose, phase1 (frame <= "
        "2018), QPI EFD volume variant [strain/genotype and temperature: confirm from "
        "acquisition metadata]. Abbreviations: EFD, elliptic Fourier descriptor; OLS, "
        "ordinary least squares. Data availability: source data (per-lineage mass "
        "trace + per-cycle fit coefficients) in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "ncols": ncols,
                "n_dead": n, "n_swelling": n_swell, "n_elongation": n_elong},
        description="mother dry mass(t) with per-cycle exponential fit M0 exp(kt) "
                    "overlaid, ALL phase1-dead lineages in a grid (swelling + 2 "
                    "elongation cascades), terminal cycle highlighted; EFD",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
