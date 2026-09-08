"""_fig_growthrate_by_generation_3groups.py — per-cycle dry-mass growth rate vs
generation (replicative age), survivor / swelling-death / elongation-death, EFD.

Style after Nakaoka & Wakamoto 2017 PLoS Biol Fig 2B/C: the groups are OVERLAID on
the same generation axis (not dodged). Two boxplots at the same x would be
illegible, so (as the reference Fig 2C and FIGURE_SPEC legibility) the survivor
and swelling groups are drawn as median ± IQR markers with error bars; the n=2
elongation lineages are individual points.

Per lineage, per cell cycle: d ln M/dt = OLS slope of ln(mass_pg) vs time_h over
[birth_frame, div_frame-1] (EFD mass, >=4 fit points). x = generation = the cycle
index within the lineage (replicative age 1,2,3,...). Dying groups lose lineages
at later generations (they die), so n decreases with generation.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_growthrate_by_generation_3groups.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from gold_standard import (  # noqa: E402
    list_phase1_survivors, list_phase1_dead, MANUAL_EXCLUDE,
)
from qpi_paths import find_corrected_lineage_csv  # noqa: E402
from _fig_growthrate_dead_vs_alive import per_cycle_rates, collect_group  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END = 2018
MIN_N = 6                 # min lineages in a generation bin for median/IQR
                          # (also caps the plotted generation range to where the
                          # dying group keeps >=MIN_N lineages; the sparse, noisy
                          # tail beyond it showed the same flat pattern)
SURV_C = "#0072B2"        # Okabe-Ito blue   (survivor)
SWELL_C = "#D55E00"       # Okabe-Ito vermilion (swelling death)
ELONG_C = "#009E73"       # Okabe-Ito green  (elongation death)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def gen_pool(group_data: dict) -> dict:
    """gen -> list of per-cycle d ln M/dt across all lineages in the group."""
    g2 = defaultdict(list)
    for rates in group_data.values():
        for r in rates:
            g2[r["gen"]].append(r["rate"])
    return g2


def med_iqr(g2: dict, max_gen: int | None = None):
    gens = sorted(g for g, v in g2.items() if len(v) >= MIN_N
                  and (max_gen is None or g <= max_gen))
    med = np.array([np.median(g2[g]) for g in gens])
    q25 = np.array([np.percentile(g2[g], 25) for g in gens])
    q75 = np.array([np.percentile(g2[g], 75) for g in gens])
    n = np.array([len(g2[g]) for g in gens])
    return np.array(gens, float), med, q25, q75, n


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

    g_surv, g_swell, g_elong = gen_pool(surv_data), gen_pool(swell_data), gen_pool(elong_data)

    # generation range = where the swelling group still has >=MIN_N lineages
    gsw, msw, q25sw, q75sw, nsw = med_iqr(g_swell)
    MAXGEN = int(gsw.max()) if len(gsw) else 10
    gsu, msu, q25su, q75su, nsu = med_iqr(g_surv, max_gen=MAXGEN)
    print(f"MAXGEN (swelling n>={MIN_N}) = {MAXGEN}; "
          f"swelling n per gen: {nsw[0]}..{nsw[-1]}; survivor n per gen: {nsu[0]}..{nsu[-1]}")

    fig, ax = plt.subplots(figsize=(130 / 25.4, 78 / 25.4), constrained_layout=True)
    ax.errorbar(gsu, msu, yerr=[msu - q25su, q75su - msu], fmt="o", ms=4,
                color=SURV_C, ecolor=SURV_C, elinewidth=1.0, capsize=2, alpha=0.85,
                lw=0, zorder=4, label=f"survivor (n={len(surv_data)})")
    ax.errorbar(gsw, msw, yerr=[msw - q25sw, q75sw - msw], fmt="s", ms=4,
                color=SWELL_C, ecolor=SWELL_C, elinewidth=1.0, capsize=2, alpha=0.85,
                lw=0, zorder=5, label=f"swelling death (n={len(swell_data)})")
    # elongation: individual lineage points (n=2), overlaid on the same x
    e_g, e_r, e_ch = [], [], []
    for ch, rates in elong_data.items():
        for r in rates:
            if r["gen"] <= MAXGEN:
                e_g.append(r["gen"]); e_r.append(r["rate"]); e_ch.append(ch)
    ax.plot(e_g, e_r, "^", ms=5, color=ELONG_C, mec="black", mew=0.4, alpha=0.95,
            lw=0, zorder=6, label=f"elongation death (n={len(elong_data)} lineages)")

    ax.set_xlabel("generation (replicative age)")
    ax.set_ylabel(r"$d\ln M/dt$ [h$^{-1}$]")
    ax.set_xlim(0.5, MAXGEN + 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower left", frameon=False, fontsize=6)

    # overall survivor vs swelling comparison (pooled over plotted generations)
    surv_pool = [v for g in g_surv for v in g_surv[g] if g <= MAXGEN]
    swell_pool = [v for g in g_swell for v in g_swell[g] if g <= MAXGEN]
    p_sw = float(mannwhitneyu(surv_pool, swell_pool, alternative="two-sided").pvalue)
    print(f"overall survivor vs swelling: surv med={np.median(surv_pool):.3f} "
          f"swell med={np.median(swell_pool):.3f}  MWU p={p_sw:.2g}")

    caption = (
        "Per-cycle dry-mass growth rate is independent of replicative age and "
        "indistinguishable between surviving and dying lineages across generations. "
        "x = generation (replicative age = the cell-cycle index within a lineage, "
        "1,2,3,...); y = per-cycle dry-mass specific growth rate. Operational "
        "definitions: d ln M/dt = OLS slope of ln(mass_pg) vs time_h over the cycle "
        "[birth_frame, div_frame−1] (>=4 valid frames; mass_pg = EFD dry mass = "
        "(λ/2πα)·∬φ dA). generation = ordinal cell cycle of that lineage. Group "
        "definitions: survivor = phase1 survivors (YAML phase1.outcome=alive, EFD "
        "corrected lineage present, manual excludes removed), cycles to frame 2018; "
        "swelling death = phase1-dead lineages excluding the two elongation cascades, "
        "cycles to the curated last-completed-cycle frame (death window); elongation "
        "death = Pos20_ch06, Pos30_ch04 (elongation-cascade phase1 deaths). Visual "
        f"elements: survivor = blue circles, swelling = vermilion squares, both "
        "median ± IQR (error bars = 25th–75th percentile across lineages) overlaid on "
        "the same generation (not dodged); elongation = green triangles, individual "
        "per-lineage per-generation points (n=2 lineages). Boxplots overlaid at the "
        "same x would be illegible, so per Fig 2C / FIGURE_SPEC legibility, median ± "
        f"IQR markers are used. n (lineages): survivor = {len(surv_data)}, swelling = "
        f"{len(swell_data)}, elongation = {len(elong_data)}; n DECREASES with "
        f"generation as dying lineages are lost — at generation 1 swelling n={nsw[0]}, "
        f"declining to n={nsw[-1]} at generation {int(gsw[-1])} (the MIN_N={MIN_N} "
        f"cutoff sets the plotted range, generations 1–{MAXGEN}). Statistics: survivor "
        f"vs swelling pooled over generations 1–{MAXGEN}, two-sided Mann–Whitney U "
        f"p={p_sw:.2g} (medians {np.median(surv_pool):.3f} vs "
        f"{np.median(swell_pool):.3f} h⁻¹); per-generation comparison and the n=2 "
        "elongation group are descriptive only. Conditions: Schizosaccharomyces "
        "pombe, 260517 Mother-Machine dataset, EMM + 2% glucose, phase1 (frame ≤ "
        "2018), QPI EFD volume variant [strain/genotype and temperature: confirm from "
        "acquisition metadata]. Abbreviations: IQR, interquartile range; EFD, elliptic "
        "Fourier descriptor; OLS, ordinary least squares; MWU, Mann–Whitney U. Colours "
        "are Okabe-Ito (qpi_colors/paper.mplstyle infra not yet present in repo). Data "
        "availability: source data in this run's *_data.npz / *_data.csv. Interpretation: "
        "growth rate stays ~flat across replicative age in survivors (no growth-rate "
        "aging signature), and the swelling-death group tracks the survivors generation "
        "by generation — consistent with Nakaoka & Wakamoto (no replicative-age "
        "dependence); any dying signature is terminal (last generation), not a function "
        "of replicative age. The two elongation lineages fall within the same band."
    )

    data = {
        "surv_gen": gsu, "surv_median": msu, "surv_q25": q25su, "surv_q75": q75su, "surv_n": nsu,
        "swell_gen": gsw, "swell_median": msw, "swell_q25": q25sw, "swell_q75": q75sw, "swell_n": nsw,
        "elong_gen": np.array(e_g, float), "elong_rate": np.array(e_r, float),
        "elong_ch": np.array(e_ch),
    }
    save_figure(
        fig,
        params={"volume_variant": "efd", "phase1_end_frame": PHASE1_END,
                "min_bin_n": MIN_N, "max_generation_plotted": MAXGEN,
                "n_survivor": len(surv_data), "n_swelling": len(swell_data),
                "n_elongation": len(elong_data),
                "survivor_vs_swelling_mwu_p": p_sw},
        description="per-cycle dry-mass growth rate d ln M/dt vs generation "
                    "(replicative age): survivor / swelling-death / elongation-death "
                    "overlaid (median±IQR markers + n=2 elongation points), EFD; "
                    "style after Nakaoka & Wakamoto 2017 Fig 2B/C",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
