"""_fig_panelG_survivors.py — reference-style panels B/C/D + G for surviving
(normally dividing) mothers, cycle-aligned, EFD.

Replicates the density-paper layout for the gold-standard phase1 cohort (clean
normal dividers that do NOT die):
  B: volume vs relative cell-cycle progression (mean ± SD)
  C: dry-mass concentration [mg/mL] vs progression (mean ± SD)  -- the density
     dip is the volume/mass decoupling signature
  D: dry mass vs progression (mean ± SD)
  G: mean dry mass vs progression with a LINEAR (blue) and an EXPONENTIAL (red)
     fit, and the residuals of each below -- shows mass grows non-linearly
     (exponential), i.e. decouples from a constant-rate increase.

Cohort = gold-standard phase1 survivors (n_cycles pooled). EFD volume variant.
Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_panelG_survivors.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import select_gold_standard  # noqa: E402
from gold_standard_cycle_aligned_vol_mass_ri import collect_cycles  # noqa: E402
from figure_logger import save_figure  # noqa: E402

BAND = "#2c9fb3"     # teal band
LIN_C = "#0072B2"    # linear fit (blue)
EXP_C = "#D55E00"    # exponential fit (red/vermilion)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _band(ax, rel, mat, color, ylabel):
    mean = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    ax.fill_between(rel, mean - sd, mean + sd, color=color, alpha=0.30, linewidth=0)
    ax.plot(rel, mean, color=color, lw=1.6)
    ax.set_xlabel("relative cell cycle progression")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    return mean


def main():
    gold = select_gold_standard()
    rel, M, n_mothers = collect_cycles(gold)
    n_cycles = M["mass_pg"].shape[0]
    print(f"gold survivors n_mothers={n_mothers} n_cycles={n_cycles}")

    fig = plt.figure(figsize=(183 / 25.4, 150 / 25.4), constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.5])
    axB = fig.add_subplot(gs[0, 0])
    axC = fig.add_subplot(gs[0, 1])
    axD = fig.add_subplot(gs[0, 2])
    axG = fig.add_subplot(gs[1, :])
    axR = fig.add_subplot(gs[2, :], sharex=axG)

    vol_mean = _band(axB, rel, M["volume_um3_rod"], BAND, r"volume [µm$^3$]")
    conc_mean = _band(axC, rel, M["conc_mg_ml"], BAND, "density [mg/mL]")
    mass_mean = _band(axD, rel, M["mass_pg"], BAND, "dry mass [pg]")
    axB.set_title(f"B  volume (n_cycles={n_cycles})", fontsize=7)
    axC.set_title("C  density (mass/volume)", fontsize=7)
    axD.set_title("D  dry mass", fontsize=7)

    # Panel G: mean mass vs progression with linear + exponential fits + residuals
    zl = np.polyfit(rel, mass_mean, 1)
    lin = np.polyval(zl, rel)
    ze = np.polyfit(rel, np.log(mass_mean), 1)
    expf = np.exp(np.polyval(ze, rel))
    axG.plot(rel, mass_mean, "o", ms=2.5, color="#222", label="mean dry mass", zorder=5)
    axG.plot(rel, lin, color=LIN_C, lw=1.4, label=f"linear fit (slope={zl[0]:.1f})")
    axG.plot(rel, expf, color=EXP_C, lw=1.4, ls="--",
             label=f"exp fit (d ln M/dτ={ze[0]:.2f})")
    axG.set_ylabel("dry mass [pg]")
    axG.legend(loc="upper left", frameon=False, fontsize=6)
    axG.set_title("G  mean mass: linear vs exponential fit + residuals", fontsize=7)
    axG.spines[["top", "right"]].set_visible(False)
    axG.tick_params(labelbottom=False)

    w = (rel[1] - rel[0]) * 0.4
    axR.bar(rel - w / 2, mass_mean - lin, width=w, color=LIN_C, label="linear")
    axR.bar(rel + w / 2, mass_mean - expf, width=w, color=EXP_C, label="exponential")
    axR.axhline(0, color="#000", lw=0.5)
    axR.set_xlabel("relative cell cycle progression")
    axR.set_ylabel("residual [pg]")
    axR.legend(loc="upper center", frameon=False, fontsize=6, ncol=2)
    axR.spines[["top", "right"]].set_visible(False)

    rmse_lin = float(np.sqrt(np.mean((mass_mean - lin) ** 2)))
    rmse_exp = float(np.sqrt(np.mean((mass_mean - expf) ** 2)))
    print(f"mass fit RMSE: linear={rmse_lin:.3f} pg  exponential={rmse_exp:.3f} pg")
    print(f"density (conc) min={conc_mean.min():.1f} at rel="
          f"{rel[int(np.argmin(conc_mean))]:.2f}, end={conc_mean[-1]:.1f}, "
          f"start={conc_mean[0]:.1f} mg/mL")

    save_figure(
        fig,
        params={"selection": "gold-standard phase1 survivors (normal dividers)",
                "volume_variant": "efd", "n_mothers": int(n_mothers),
                "n_cycles": int(n_cycles),
                "mass_fit_rmse_linear_pg": rmse_lin,
                "mass_fit_rmse_exp_pg": rmse_exp},
        description="reference-style B/C/D/G for surviving normally-dividing mothers, "
                    "cycle-aligned EFD: volume / density(conc) / mass mean±SD bands + "
                    "mean mass with linear vs exponential fit and residuals",
        data={"rel": rel, "volume_mean": vol_mean, "conc_mean": conc_mean,
              "mass_mean": mass_mean, "mass_lin_fit": lin, "mass_exp_fit": expf,
              "volume_matrix": M["volume_um3_rod"], "mass_matrix": M["mass_pg"],
              "conc_matrix": M["conc_mg_ml"], "mean_ri_matrix": M["mean_ri"]},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
