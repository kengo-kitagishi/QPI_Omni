"""_fig_growthrate_swelling_individual.py — individual swelling-death lineages.

Per-cycle dry-mass growth rate vs generation (replicative age) for EACH
swelling-death lineage shown individually (overlaying 23 per generation is
unreadable, so 6 lineages per panel, 2×2). Each lineage's LAST cell cycle = its
last division before death (= the curated death window) is marked, so you can see
at which generation it died and whether its terminal growth rate dips. The
survivor median ± IQR band is drawn in every panel as the reference. EFD.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_growthrate_swelling_individual.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from gold_standard import (  # noqa: E402
    list_phase1_survivors, list_phase1_dead, MANUAL_EXCLUDE,
)
from qpi_paths import find_corrected_lineage_csv  # noqa: E402
from _fig_growthrate_dead_vs_alive import per_cycle_rates, collect_group  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END = 2018
MIN_N = 6
N_PER_PANEL = 6
SURV_C = "#888888"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def gen_pool(group_data: dict) -> dict:
    g2 = defaultdict(list)
    for rates in group_data.values():
        for r in rates:
            g2[r["gen"]].append(r["rate"])
    return g2


def med_iqr(g2: dict, max_gen: int):
    gens = sorted(g for g, v in g2.items() if len(v) >= MIN_N and g <= max_gen)
    med = np.array([np.median(g2[g]) for g in gens])
    q25 = np.array([np.percentile(g2[g], 25) for g in gens])
    q75 = np.array([np.percentile(g2[g], 75) for g in gens])
    return np.array(gens, float), med, q25, q75


def draw_panel(ax, panel_items, surv_ref, title, data):
    """One panel: survivor median±IQR band + individual dying-lineage trajectories
    with the last cell cycle (death) marked."""
    gsu, msu, q25su, q75su = surv_ref
    pmax = max(r["gen"] for _ch, rates in panel_items for r in rates)
    sel = gsu <= pmax
    ax.fill_between(gsu[sel], q25su[sel], q75su[sel], color=SURV_C, alpha=0.18,
                    lw=0, zorder=1)
    ax.plot(gsu[sel], msu[sel], color=SURV_C, lw=1.3, zorder=2)
    for (ch, rates), col in zip(panel_items, PALETTE):
        x = [r["gen"] for r in rates]
        y = [r["rate"] for r in rates]
        ax.plot(x, y, "-", color=col, lw=0.8, alpha=0.85, marker="o", ms=2.4,
                zorder=4, label=f"{ch} g{x[-1]}")
        ax.plot(x[-1], y[-1], "o", color=col, ms=7, mec="black", mew=0.9, zorder=6)
        data[f"{ch}_gen"] = np.array(x, float)
        data[f"{ch}_rate"] = np.array(y, float)
        data[f"{ch}_death_gen"] = np.array(x[-1])
    ax.set_xlabel("generation (replicative age)")
    ax.set_ylabel(r"$d\ln M/dt$ [h$^{-1}$]")
    ax.set_ylim(0.0, 0.45)
    ax.set_title(title, fontsize=6.5)
    ax.legend(loc="upper left", frameon=False, fontsize=4.8, ncol=2,
              handlelength=1.0, columnspacing=0.8)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    surv = [pc for pc in list_phase1_survivors() if pc not in MANUAL_EXCLUDE
            and find_corrected_lineage_csv(*pc) is not None]
    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    swelling = [pc for pc in dead if f"{pc[0]}_{pc[1]}" not in ELONGATION]
    elong = [pc for pc in dead if f"{pc[0]}_{pc[1]}" in ELONGATION]

    surv_data = collect_group(surv, lambda pos, ch: PHASE1_END)
    swell_data = collect_group(swelling, lambda pos, ch: DEATH_WINDOW[f"{pos}_{ch}"][1])
    elong_data = collect_group(elong, lambda pos, ch: DEATH_WINDOW[f"{pos}_{ch}"][1])
    print(f"survivor={len(surv_data)}  swelling={len(swell_data)}  elong={len(elong_data)}")

    # survivor reference band (up to the longest dying-lineage lifespan)
    maxgen = max(r["gen"] for d in (swell_data, elong_data)
                 for rates in d.values() for r in rates)
    gsu, msu, q25su, q75su = med_iqr(gen_pool(surv_data), max_gen=maxgen)
    surv_ref = (gsu, msu, q25su, q75su)

    # swelling lineages: one (ch, rates) each, sorted by lifespan, 6 per panel
    items = sorted(swell_data.items(), key=lambda kv: -len(kv[1]))
    panels = [items[i:i + N_PER_PANEL] for i in range(0, len(items), N_PER_PANEL)]

    fig, axes = plt.subplots(3, 2, figsize=(183 / 25.4, 182 / 25.4),
                             sharey=True, constrained_layout=True)
    data = {"surv_gen": gsu, "surv_median": msu, "surv_q25": q25su, "surv_q75": q75su}
    for pi, panel in enumerate(panels):
        draw_panel(axes.flat[pi], panel, surv_ref,
                   f"swelling lineages {pi*N_PER_PANEL+1}–{pi*N_PER_PANEL+len(panel)} "
                   f"of {len(items)}", data)
    # 5th panel = elongation cascades (n=2); 6th cell off
    draw_panel(axes.flat[len(panels)], list(elong_data.items()), surv_ref,
               f"elongation cascades (n={len(elong_data)})", data)
    for ax in axes.flat[len(panels) + 1:]:
        ax.axis("off")
    fig.suptitle("Individual dying lineages vs survivor band — grey: survivor "
                 "median±IQR (n=131); large outlined marker: last cell cycle (last "
                 "division before death). Panels 1–4: swelling deaths (6/6/6/5); "
                 "panel 5: elongation cascades (n=2). y clipped 0–0.45.", fontsize=6.5)

    caption = (
        "Individual dying lineages grow at the survivor rate generation by generation, "
        "up to and including their final cell cycle, whether they die by swelling or "
        "elongation. Panels 1–4 show the 23 swelling-death lineages 6 at a time "
        "(overlaying all 23 at each generation is illegible); panel 5 shows the 2 "
        "elongation-cascade lineages. "
        "x = generation (replicative age = ordinal cell cycle of the lineage, 1,2,…); "
        "y = per-cycle dry-mass specific growth rate. Operational definitions: "
        "d ln M/dt = OLS slope of ln(mass_pg) vs time_h over the cycle [birth_frame, "
        "div_frame−1] (>=4 valid frames; mass_pg = EFD dry mass). generation = ordinal "
        "cell cycle. swelling death = phase1-dead lineages excluding the two elongation "
        "cascades; elongation death = the two elongation-cascade lineages (Pos20_ch06, "
        "Pos30_ch04). For every dying lineage, cycles run to its curated death window "
        "(the last completed cell cycle = last division before death / before elongation "
        "onset, from the fig_panelA 2nd-to-last row). survivor = phase1 survivors with an "
        "EFD corrected lineage. Visual elements: each coloured line+markers = one dying "
        "lineage's trajectory (legend: ch g<death generation>); the large black-outlined "
        "marker = that lineage's LAST cell cycle (last division before death); grey line "
        "+ band = survivor median ± IQR (25th–75th percentile across lineages) as the "
        "reference in every panel. y clipped to 0–0.45 (a few single-cycle fit outliers "
        f"fall outside). n: swelling = {len(swell_data)} lineages (6/6/6/5 across panels "
        f"1–4, sorted by lifespan), elongation = {len(elong_data)} lineages (panel 5); "
        f"survivor reference = {len(surv_data)} lineages. Statistics: descriptive "
        "(individual lineages; no test). Conditions: Schizosaccharomyces pombe, 260517 "
        "Mother-Machine dataset, EMM + 2% glucose, phase1 (frame ≤ 2018), QPI EFD volume "
        "variant [strain/genotype and temperature: confirm from acquisition metadata]. "
        "Abbreviations: IQR, interquartile range; EFD, elliptic Fourier descriptor; OLS, "
        "ordinary least squares. Colours are Okabe-Ito. Data availability: source data "
        "in this run's *_data.npz. Interpretation: both swelling and elongation lineages "
        "track the survivor band across replicative age, and most do NOT show an obvious "
        "drop at their last cell cycle in this per-cycle view — consistent with the "
        "dying signature being a short terminal event, not a replicative-age decline. "
        "Lifespan (death generation) varies widely between lineages."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "phase1_end_frame": PHASE1_END,
                "min_bin_n": MIN_N, "n_per_panel": N_PER_PANEL,
                "n_swelling": len(swell_data), "n_elongation": len(elong_data),
                "n_survivor_ref": len(surv_data), "max_generation": int(maxgen)},
        description="individual dying lineages: per-cycle d ln M/dt vs generation, "
                    "6 lineages/panel (3×2: panels 1–4 swelling 6/6/6/5, panel 5 "
                    "elongation n=2), survivor median±IQR reference, last cell cycle "
                    "(death) marked; EFD",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
