"""_fig_revival_return_curves.py — how revived cells return to steady state.

Reads the shared per-(lineage, cycle) table and plots, vs generation after recovery,
how generation time, birth dry mass, birth volume, birth mean RI, birth density and
the per-cycle mass/volume decoupling (k_mass - k_vol) return to their phase1
steady-state values. Aggregates are mean +/- SD; the phase1 reference is the pooled
phase1 mean +/- SD band; the first generation that enters and stays within the
phase1 +/-1SD band is marked as 'returned'. EFD (the table is EFD-derived).

Run: python scripts/_fig_revival_return_curves.py
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
from figure_logger import save_figure  # noqa: E402

TABLE = Path("results/260517/_revival_pergen.csv")
MINN = 10
MAXGEN = 14
METRICS = [("interval_h", "generation time [h]"),
           ("birth_mass", "birth dry mass [pg]"),
           ("birth_vol", "birth volume [um$^3$]"),
           ("birth_ri", "birth mean RI"),
           ("birth_dens", "birth density [mg/mL]"),
           ("decouple_k", r"decoupling $k_M-k_V$ [h$^{-1}$]")]
C = "#0072B2"
P1C = "#888888"

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    df = pd.read_csv(TABLE)
    post = df[df["epoch"] == "post"]
    p1 = df[df["epoch"] == "phase1"]

    fig, axes = plt.subplots(2, 3, figsize=(183 / 25.4, 110 / 25.4),
                             constrained_layout=True)
    axes = axes.ravel()
    data = {}
    summary = {}
    for ax, (col, lab) in zip(axes, METRICS):
        p1_mu = float(p1[col].mean()); p1_sd = float(p1[col].std(ddof=1))
        gens, mus, sds, ns = [], [], [], []
        for g in range(1, MAXGEN + 1):
            sub = post[post["gen_rec"] == g][col]
            if len(sub) >= MINN:
                gens.append(g); mus.append(float(sub.mean()))
                sds.append(float(sub.std(ddof=1))); ns.append(len(sub))
        gens = np.array(gens); mus = np.array(mus); sds = np.array(sds)
        # returned = first gen entering & staying within phase1 +/-1SD
        inside = np.abs(mus - p1_mu) <= p1_sd
        ret = None
        for i in range(len(gens)):
            if inside[i] and inside[i:].all():
                ret = int(gens[i]); break
        ax.axhspan(p1_mu - p1_sd, p1_mu + p1_sd, color=P1C, alpha=0.18, lw=0)
        ax.axhline(p1_mu, color=P1C, lw=0.8, ls=":")
        ax.errorbar(gens, mus, yerr=sds, color=C, lw=1.4, marker="o", ms=3,
                    capsize=2, zorder=4)
        if ret is not None:
            ax.axvline(ret, color="#D55E00", lw=1.0, ls="--", zorder=3)
        ax.set_xlabel("generation after recovery"); ax.set_ylabel(lab)
        ax.set_title(f"{lab.split(' [')[0]}  (returns @ gen {ret})", fontsize=7)
        ax.set_xlim(0.5, MAXGEN + 0.5)
        data[f"{col}_gen"] = gens; data[f"{col}_mean"] = mus
        data[f"{col}_sd"] = sds; data[f"{col}_n"] = np.array(ns)
        data[f"{col}_p1_mean"] = np.array(p1_mu); data[f"{col}_p1_sd"] = np.array(p1_sd)
        summary[col] = ret
    fig.suptitle("Return to steady-state growth after recovery — revived lineages "
                 "(EFD, n=73); grey band = phase1 mean±SD, orange dashed = return "
                 "generation", fontsize=8)

    ret_txt = "; ".join(f"{c.split('_')[0] if c!='interval_h' else 'gen-time'}: gen "
                        f"{summary[c]}" for c, _ in METRICS)
    caption = (
        "After recovery to 2% glucose, revived mother lineages return their cell-cycle "
        "time, birth size, birth refractive index/density and per-cycle mass/volume "
        "growth balance to the phase1 steady state within a few generations; this "
        "figure shows the trajectory and the return generation for each. x = "
        "generation after recovery (1 = first post-recovery completed cycle). Panels: "
        "generation time (interval_h), birth dry mass, birth volume, birth mean RI, "
        "birth density (1000*birth_mass/birth_vol), and the per-cycle mass/volume "
        "decoupling k_M - k_V (k = OLS slope of ln(quantity) vs time_h over the cycle; "
        ">0 = mass outgrows volume = densifying). Operational definitions: all "
        "quantities are per division-bounded cell cycle from the shared per-generation "
        "table (EFD; valid frames only: not is_outlier/touches_border, mass_pg >= 10, "
        "volume > 0); birth_* = value at the first valid frame of the cycle. Markers "
        "with error bars = mean +/- SD across lineages per generation (>= "
        f"{MINN} lineages per point); grey band = phase1 pooled mean +/- SD; dotted "
        "grey line = phase1 mean; orange dashed vertical line = the first generation "
        "that enters and stays within the phase1 +/-1 SD band ('returned'). Return "
        f"generations: {ret_txt}. Aggregates are mean +/- SD (survivor convention). "
        "n = 73 revived lineages (260517). Statistics: descriptive; error bars = SD "
        "across lineages. Conditions: Schizosaccharomyces pombe, 260517 Mother-Machine "
        "starvation dataset (2% -> 0.0055% -> 0% -> 2%), QPI EFD volume variant "
        "[strain/genotype and temperature: confirm from acquisition metadata]. "
        "Abbreviations: SD, standard deviation; RI, refractive index; EFD, elliptic "
        "Fourier descriptor; OLS, ordinary least squares. Data availability: "
        "per-generation means/SDs and phase1 references in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "n_revived": 73, "min_n": MINN,
                "return_gen_interval": summary["interval_h"],
                "return_gen_birth_mass": summary["birth_mass"],
                "return_gen_birth_vol": summary["birth_vol"],
                "return_gen_birth_ri": summary["birth_ri"],
                "return_gen_birth_dens": summary["birth_dens"],
                "return_gen_decouple": summary["decouple_k"]},
        description="revival return curves: generation time, birth mass/volume/RI/"
                    "density and mass-volume decoupling vs generation after recovery, "
                    "mean±SD, phase1 band, return generation marked (EFD)",
        caption=caption,
        data=data,
    )
    plt.close(fig)
    print("return generations:", summary)


if __name__ == "__main__":
    main()
