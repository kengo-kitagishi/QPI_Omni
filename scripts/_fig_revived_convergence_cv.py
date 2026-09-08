"""_fig_revived_convergence_cv.py — A5: recovery convergence vs generation.

For the 260517 revived mothers, how many generations after recovery (return to 2%
glucose at frame 2885) does the cell-to-cell spread of mother size and density
return to the phase1 steady-state spread? Per post-recovery generation we take, for
each lineage, the cycle-mean of mean RI, volume and dry mass, and plot the
cross-lineage coefficient of variation (CV) vs generation-after-recovery, with the
phase1 steady-state CV as a reference line for each quantity. EFD.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_revived_convergence_cv.py
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
from overlay_mean_sd_band_full_timecourse import list_revived_mothers  # noqa: E402
from qpi_paths import resolve_lineage_csv, find_corrected_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import enumerate_all_cycles  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END = 2018
RECOVERY_FRAME = 2885
MINN = 10          # min lineages contributing to a generation's CV
QTS = [("mass_pg", "dry mass", "#0072B2"),
       ("volume_um3_rod", "volume", "#009E73"),
       ("mean_ri", "mean RI", "#E69F00")]

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def cycle_means(c):
    """Cycle-mean of each quantity over valid frames; None if too few."""
    v = c[~(c["is_outlier"] | c["touches_border"]) & (c["mass_pg"] >= 10.0)
          & (c["volume_um3_rod"] > 0)]
    if len(v) < 4:
        return None
    return {"mass_pg": float(v["mass_pg"].mean()),
            "volume_um3_rod": float(v["volume_um3_rod"].mean()),
            "mean_ri": float(v["mean_ri"].mean()),
            "f0": int(c["frame"].min()), "f1": int(c["frame"].max())}


def collect():
    revived = [pc for pc in list_revived_mothers()
               if find_corrected_lineage_csv(*pc) is not None]
    # both phase1 and post-recovery binned by generation index (single-cycle values),
    # so the phase1 reference CV is computed the SAME way as each post-recovery gen CV
    p1_gen = {q: defaultdict(list) for q, _, _ in QTS}
    gen_vals = {q: defaultdict(list) for q, _, _ in QTS}
    n_lin = 0
    for pos, ch in revived:
        m = load_mother(pos, ch)
        cyc, _ = enumerate_all_cycles(m, set(), max_frame=3747, min_kept=6)
        cm = [x for x in (cycle_means(c) for c in cyc) if x is not None]
        p1 = sorted([x for x in cm if x["f1"] <= PHASE1_END], key=lambda x: x["f0"])
        rec = sorted([x for x in cm if x["f0"] >= RECOVERY_FRAME], key=lambda x: x["f0"])
        if not p1 or not rec:
            continue
        n_lin += 1
        for i, x in enumerate(p1, 1):
            for q, _, _ in QTS:
                p1_gen[q][i].append(x[q])
        for g, x in enumerate(rec, 1):
            for q, _, _ in QTS:
                gen_vals[q][g].append(x[q])
    return p1_gen, gen_vals, n_lin


def cv(vals):
    a = np.asarray(vals, float)
    return float(np.std(a, ddof=1) / np.mean(a) * 100.0)


def main():
    p1_gen, gen_vals, n_lin = collect()
    print(f"revived lineages used: {n_lin}")

    fig, ax = plt.subplots(figsize=(120 / 25.4, 80 / 25.4), constrained_layout=True)
    data = {}
    summary = {}
    for q, lab, col in QTS:
        # phase1 reference = mean over phase1 generation indices (>= MINN lineages) of
        # the cross-lineage single-cycle CV — matched to the post-recovery computation
        p1_idx_cv = [cv(p1_gen[q][i]) for i in sorted(p1_gen[q])
                     if len(p1_gen[q][i]) >= MINN]
        p1cv = float(np.mean(p1_idx_cv))
        gens = sorted(g for g, v in gen_vals[q].items() if len(v) >= MINN)
        cvs = np.array([cv(gen_vals[q][g]) for g in gens])
        ns = np.array([len(gen_vals[q][g]) for g in gens])
        gens = np.array(gens, float)
        ax.plot(gens, cvs, "-o", color=col, ms=3, lw=1.4, label=f"{lab} (phase1 "
                f"CV={p1cv:.1f}%)")
        ax.axhline(p1cv, color=col, lw=0.8, ls=":")
        # convergence generation = first gen whose CV <= phase1 CV
        conv = next((int(g) for g, c in zip(gens, cvs) if c <= p1cv), None)
        summary[q] = {"phase1_cv": p1cv, "converge_gen": conv}
        data[f"{q}_gen"] = gens
        data[f"{q}_cv"] = cvs
        data[f"{q}_n"] = ns
        data[f"{q}_phase1_cv"] = np.array(p1cv)
        print(f"  {lab}: phase1 CV={p1cv:.1f}%  gen1 CV={cvs[0]:.1f}%  "
              f"converge@gen={conv}")
    ax.set_xlabel("generation after recovery")
    ax.set_ylabel("cross-lineage CV [%]")
    ax.set_title(f"A5 — revived: cell-to-cell CV vs generation after recovery; "
                 f"dotted = phase1 steady-state CV (EFD, n={n_lin})", fontsize=7)
    ax.legend(loc="upper right", frameon=False)
    ax.set_xlim(0.5, max(g for q, _, _ in QTS for g in gen_vals[q]
                         if len(gen_vals[q][g]) >= MINN) + 0.5)

    conv_txt = "; ".join(f"{lab}: gen {summary[q]['converge_gen']}"
                         for q, lab, _ in QTS)
    caption = (
        "After recovery to 2% glucose, the cell-to-cell variability of revived mother "
        "size and density returns toward the phase1 steady-state spread over a few "
        "generations; the figure shows how many. x = generation after recovery (1 = "
        "first completed cell cycle after resuming growth). y = cross-lineage "
        "coefficient of variation (CV = SD/mean x 100). Operational definitions: for "
        "each post-recovery cell cycle of each lineage we take the cycle-mean of "
        "mass_pg, volume_um3_rod and mean_ri over valid frames (not is_outlier/"
        "touches_border, mass_pg >= 10, volume > 0, >= 4 frames); at generation g the "
        "CV is computed across all lineages that have a g-th post-recovery cycle. "
        "Cycles are enumerated with enumerate_all_cycles (the long starvation-arrest "
        "cycle is dropped automatically); recovery frame = 2885 (time_h = frame/12). "
        "phase1 steady-state CV (dotted reference line per quantity) = the mean over "
        "phase1 generation indices (>= MINN lineages) of the cross-lineage single-cycle "
        "CV, i.e. computed exactly like each post-recovery generation's CV but within "
        "phase1 (frame <= 2018) — so the reference and the post-recovery points are "
        "apples-to-apples (both single-cycle cross-lineage spreads). "
        "Solid lines with markers = post-recovery CV per generation (blue dry mass, "
        "green volume, orange mean RI); dotted lines = the matching phase1 CV. A "
        "generation is shown only where >= "
        f"{MINN} lineages contribute (later generations have fewer lineages). "
        f"Convergence (first generation whose CV <= phase1 CV): {conv_txt}. n = "
        f"{n_lin} revived lineages with an EFD corrected lineage. Statistics: "
        "descriptive (CV; no test). Conditions: Schizosaccharomyces pombe, 260517 "
        "Mother-Machine starvation dataset (2% -> 0.0055% -> 0% -> 2%), QPI EFD volume "
        "variant [strain/genotype and temperature: confirm from acquisition metadata]. "
        "Abbreviations: CV, coefficient of variation; SD, standard deviation; RI, "
        "refractive index; EFD, elliptic Fourier descriptor. Data availability: "
        "per-generation CV, n and phase1 references in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "recovery_frame": RECOVERY_FRAME,
                "phase1_end": PHASE1_END, "min_n": MINN, "n_revived": n_lin,
                "converge_gen_mass": summary["mass_pg"]["converge_gen"],
                "converge_gen_volume": summary["volume_um3_rod"]["converge_gen"],
                "converge_gen_ri": summary["mean_ri"]["converge_gen"]},
        description="A5 revived: cross-lineage CV of cycle-mean mass/volume/mean RI vs "
                    "generation after recovery, phase1 steady-state CV reference (EFD)",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
