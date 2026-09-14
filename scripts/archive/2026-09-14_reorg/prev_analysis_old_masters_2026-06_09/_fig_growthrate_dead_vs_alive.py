"""_fig_growthrate_dead_vs_alive.py — per-generation dry-mass growth rate,
phase1-dead vs phase1-survived lineages (EFD), mother-machine B/C style.

Two figures:
  1. Per-generation d ln(M)/dt (one fit per cell cycle), dead vs alive:
       - individual lineage trajectories (B-style)
       - mean +/- SD per generation, START-aligned (generation 1,2,... ; the
         reference panel-C style) and DEATH/END-aligned (last generation back).
       The end-aligned panel is where a pre-death dead/alive difference would
       show up (alive lineages are aligned to the phase1 end, frame 2018).
  2. Mother dry mass(t) with the per-cycle exponential fit M0*exp(k t) overlaid,
     for a representative dead and a representative alive lineage.

Cohorts (EFD, QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd):
  - dead  = phase1-dead lineages, cycles up to the curated death window (reliable;
            after death the trench repopulates and the trace is contaminated).
  - alive = phase1 survivors that have an EFD corrected lineage (unbiased — NOT
            gold-standard, which is selected on division interval).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gold_standard import (  # noqa: E402
    list_phase1_survivors, list_phase1_dead, MANUAL_EXCLUDE,
)
from qpi_paths import resolve_lineage_csv, find_corrected_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import enumerate_all_cycles  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END = 2018
MIN_PTS = 4          # min valid frames to fit a cycle
MIN_CYC = 3          # min fitted cycles to keep a lineage
MIN_N = 3            # min lineages in a generation bin for mean +/- SD
ALIVE_C = "#2e8b8b"  # teal (survived)
DEAD_C = "#b25450"   # muted red (extinct/dead)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def per_cycle_rates(m: pd.DataFrame, max_frame: int) -> list[dict]:
    """One exponential fit (d ln M/dt) per division-bounded cycle, in order."""
    cyc, _ = enumerate_all_cycles(m, set(), max_frame=max_frame, min_kept=6)
    out = []
    for c in cyc:
        v = c[~(c["is_outlier"] | c["touches_border"]) & (c["mass_pg"] >= 10.0)]
        if len(v) < MIN_PTS:
            continue
        t = v["time_h"].to_numpy(float)
        mass = v["mass_pg"].to_numpy(float)
        if np.any(mass <= 0):
            continue
        sl, ic = np.polyfit(t, np.log(mass), 1)
        out.append({"rate": float(sl), "ic": float(ic),
                    "t0": float(t.min()), "t1": float(t.max()),
                    "f0": int(c["frame"].min()), "f1": int(c["frame"].max())})
    # generation index assigned by order (1-based)
    for g, r in enumerate(out, start=1):
        r["gen"] = g
    return out


def collect_group(pairs, max_frame_fn) -> dict[str, list[dict]]:
    data = {}
    for i, (pos, ch) in enumerate(pairs):
        p = resolve_lineage_csv(pos, ch)
        if p is None:
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        m = df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)
        rates = per_cycle_rates(m, max_frame_fn(pos, ch))
        if len(rates) >= MIN_CYC:
            data[f"{pos}_{ch}"] = rates
    return data


def _bin_start(group: dict) -> dict:
    g2 = defaultdict(list)
    for rates in group.values():
        for r in rates:
            g2[r["gen"]].append(r["rate"])
    return g2


def _bin_end(group: dict) -> dict:
    g2 = defaultdict(list)
    for rates in group.values():
        n = len(rates)
        for i, r in enumerate(rates):
            g2[-(n - i)].append(r["rate"])   # last cycle = -1
    return g2


def _mean_sd(bins: dict):
    xs = sorted(g for g, v in bins.items() if len(v) >= MIN_N)
    mean = np.array([np.mean(bins[g]) for g in xs])
    sd = np.array([np.std(bins[g], ddof=1) for g in xs])
    n = np.array([len(bins[g]) for g in xs])
    return np.array(xs, float), mean, sd, n


# ---------------------------------------------------------------------------
def figure_per_generation(alive: dict, dead: dict) -> plt.Figure:
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4),
                             constrained_layout=True)

    # Panel 0 — individual lineage trajectories (B-style), start-aligned
    ax = axes[0]
    for grp, col in ((alive, ALIVE_C), (dead, DEAD_C)):
        for rates in grp.values():
            ax.plot([r["gen"] for r in rates], [r["rate"] for r in rates],
                    color=col, lw=0.4, alpha=0.12, solid_capstyle="butt")
    # group means on top
    for grp, col, lab in ((alive, ALIVE_C, "survived"), (dead, DEAD_C, "extinct")):
        x, mean, _, _ = _mean_sd(_bin_start(grp))
        ax.plot(x, mean, color=col, lw=1.6, label=f"{lab} (n={len(grp)})", zorder=5)
    ax.set_xlabel("generation"); ax.set_ylabel(r"$d\ln M/dt$ [h$^{-1}$]")
    ax.set_xlim(0, 60); ax.legend(loc="upper right", frameon=False)
    ax.set_title("per-lineage (start-aligned)", fontsize=7)

    # Panel 1 — mean +/- SD per generation (C-style), start-aligned
    ax = axes[1]
    for grp, col, off in ((alive, ALIVE_C, -0.15), (dead, DEAD_C, 0.15)):
        x, mean, sd, _ = _mean_sd(_bin_start(grp))
        ax.errorbar(x + off, mean, yerr=sd, fmt="o", ms=2.5, color=col,
                    ecolor=col, elinewidth=0.6, capsize=0, alpha=0.85, lw=0)
    ax.set_xlabel("generation (from start)")
    ax.set_ylabel(r"$d\ln M/dt$ [h$^{-1}$]")
    ax.set_xlim(0, 60)
    ax.set_title("mean ± SD (start-aligned)", fontsize=7)

    # Panel 2 — mean +/- SD per generation-before-end (DEATH/END-aligned)
    ax = axes[2]
    for grp, col in ((alive, ALIVE_C), (dead, DEAD_C)):
        x, mean, sd, _ = _mean_sd(_bin_end(grp))
        ax.errorbar(x, mean, yerr=sd, fmt="o-", ms=2.5, color=col, ecolor=col,
                    elinewidth=0.6, capsize=0, alpha=0.85, lw=0.8)
    ax.axvline(-1, color="#888", lw=0.5, ls=":")
    ax.set_xlabel("generation before death / phase1-end")
    ax.set_ylabel(r"$d\ln M/dt$ [h$^{-1}$]")
    ax.set_xlim(-12, 0.5)
    ax.set_title("mean ± SD (end-aligned)", fontsize=7)
    return fig


def _rep_mother(ch_key: str, max_frame: int):
    pos, ch = ch_key.split("_", 1)
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    m = df[df["rank"] == 1].sort_values("frame")
    m = m[(m["frame"] <= max_frame) & ~(m["is_outlier"] | m["touches_border"])
          & (m["mass_pg"] >= 10.0)]
    return m


def figure_mass_fit(alive: dict, dead: dict) -> tuple[plt.Figure, dict]:
    dead_rep = max(dead, key=lambda k: len(dead[k]))
    alive_rep = max(alive, key=lambda k: len(alive[k]))
    reps = [(dead_rep, dead[dead_rep], DEAD_C, DEATH_WINDOW[dead_rep][1], "extinct"),
            (alive_rep, alive[alive_rep], ALIVE_C, PHASE1_END, "survived")]
    fig, axes = plt.subplots(2, 1, figsize=(183 / 25.4, 95 / 25.4),
                             constrained_layout=True)
    data: dict = {}
    tags = ["dead", "alive"]
    for ax, (ch_key, rates, col, mx, lab), tag in zip(axes, reps, tags):
        m = _rep_mother(ch_key, mx)
        ax.plot(m["time_h"], m["mass_pg"], color=col, lw=0.5, alpha=0.7,
                label="mother dry mass(t)")
        for r in rates:           # per-cycle exponential fit overlay
            tt = np.linspace(r["t0"], r["t1"], 20)
            ax.plot(tt, np.exp(r["ic"] + r["rate"] * tt), color="#222",
                    lw=0.9, ls="--", zorder=5)
        ax.plot([], [], color="#222", lw=0.9, ls="--",
                label=r"per-cycle exp fit  $M_0e^{kt}$")
        ax.set_ylabel("dry mass [pg]")
        ax.legend(loc="upper left", frameon=False, fontsize=6)
        ax.set_title(f"{lab}: {ch_key}  ({len(rates)} generations, EFD)", fontsize=7)
        data[f"{tag}_rep"] = np.array(ch_key)
        data[f"{tag}_time_h"] = m["time_h"].to_numpy(float)
        data[f"{tag}_mass_pg"] = m["mass_pg"].to_numpy(float)
        data[f"{tag}_fit_rate"] = np.array([r["rate"] for r in rates])
        data[f"{tag}_fit_ic"] = np.array([r["ic"] for r in rates])
        data[f"{tag}_fit_t0"] = np.array([r["t0"] for r in rates])
        data[f"{tag}_fit_t1"] = np.array([r["t1"] for r in rates])
    axes[1].set_xlabel("time [h]")
    return fig, data


def main():
    surv = [pc for pc in list_phase1_survivors() if pc not in MANUAL_EXCLUDE
            and find_corrected_lineage_csv(*pc) is not None]
    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    print(f"alive (EFD survivors)={len(surv)}  dead={len(dead)}")

    alive_data = collect_group(surv, lambda pos, ch: PHASE1_END)
    dead_data = collect_group(dead, lambda pos, ch: DEATH_WINDOW[f"{pos}_{ch}"][1])
    print(f"kept (>= {MIN_CYC} cycles): alive={len(alive_data)}  dead={len(dead_data)}")

    # quick end-aligned summary (last generation rate, dead vs alive)
    for lab, grp in (("alive", alive_data), ("dead", dead_data)):
        last = [rates[-1]["rate"] for rates in grp.values()]
        allr = [r["rate"] for rates in grp.values() for r in rates]
        print(f"  {lab}: last-gen rate median={np.median(last):.4f}  "
              f"all-gen median={np.median(allr):.4f}  n_lineages={len(grp)}")

    fig1 = figure_per_generation(alive_data, dead_data)
    # long-form data for restyle
    a_ch, a_g, a_r = [], [], []
    for ch, rates in alive_data.items():
        for r in rates:
            a_ch.append(ch); a_g.append(r["gen"]); a_r.append(r["rate"])
    d_ch, d_g, d_r = [], [], []
    for ch, rates in dead_data.items():
        for r in rates:
            d_ch.append(ch); d_g.append(r["gen"]); d_r.append(r["rate"])
    save_figure(
        fig1,
        params={"volume_variant": "efd", "phase1_end_frame": PHASE1_END,
                "n_alive": len(alive_data), "n_dead": len(dead_data),
                "min_cycles": MIN_CYC, "min_bin_n": MIN_N,
                "alive_cohort": "phase1 survivors with EFD (not gold-standard)"},
        description="per-generation dry-mass growth rate d ln M/dt, phase1 survived "
                    "vs extinct lineages (EFD): individual trajectories + mean±SD "
                    "start-aligned and death/phase1-end-aligned",
        data={"alive_ch": np.array(a_ch), "alive_gen": np.array(a_g),
              "alive_rate": np.array(a_r), "dead_ch": np.array(d_ch),
              "dead_gen": np.array(d_g), "dead_rate": np.array(d_r)},
    )
    plt.close(fig1)

    fig2, data2 = figure_mass_fit(alive_data, dead_data)
    save_figure(
        fig2,
        params={"volume_variant": "efd",
                "dead_rep": max(dead_data, key=lambda k: len(dead_data[k])),
                "alive_rep": max(alive_data, key=lambda k: len(alive_data[k]))},
        description="mother dry mass(t) with per-cycle exponential fit M0 exp(kt) "
                    "overlaid, representative extinct and survived lineages (EFD)",
        data=data2,
    )
    plt.close(fig2)


if __name__ == "__main__":
    main()
