"""_fig_revival_regrowth_rate_lag.py — F1: revival regrowth rate & lag.

For the 260517 revived mothers, the dry-mass regrowth right after recovery to 2%
glucose (frame 2885). For each lineage we fit d ln(mass)/dt over the regrowth
window [recovery, first division] and ask:
  (1) is the regrowth rate higher than the same lineage's phase1 steady-state rate?
  (2) is there a lag? lag = (time at which the regrowth fit crosses the mass value
      at the first recovery frame) - (time of the first recovery frame); ~0 means
      growth resumes immediately.
Aggregates are mean +/- SD. EFD.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_revival_regrowth_rate_lag.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mean_sd_band_full_timecourse import list_revived_mothers  # noqa: E402
from qpi_paths import resolve_lineage_csv, find_corrected_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import enumerate_all_cycles  # noqa: E402
from _fig_growthrate_dead_vs_alive import per_cycle_rates  # noqa: E402
from figure_logger import save_figure  # noqa: E402

RECOVERY_FRAME = 2885
FPH = 12.0
P1_C = "#888888"           # phase1 steady
RE_C = "#D55E00"           # regrowth (vermilion)

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def collect():
    revived = [pc for pc in list_revived_mothers()
               if find_corrected_lineage_csv(*pc) is not None]
    rows = []
    for pos, ch in revived:
        m = load_mother(pos, ch)
        cyc, _ = enumerate_all_cycles(m, set(), max_frame=3747, min_kept=6)
        post = sorted([c for c in cyc if int(c["frame"].min()) >= RECOVERY_FRAME],
                      key=lambda c: int(c["frame"].min()))
        if not post:
            continue
        d_first = int(post[0]["frame"].min())          # first division after recovery
        # regrowth window [recovery, first division]
        v = m[(m["frame"] >= RECOVERY_FRAME) & (m["frame"] <= d_first)
              & ~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)]
        if len(v) < 4:
            continue
        t = v["time_h"].to_numpy(float); mass = v["mass_pg"].to_numpy(float)
        if np.any(mass <= 0):
            continue
        k_re, ic = np.polyfit(t, np.log(mass), 1)
        t_r = float(t[0]); m_r = float(mass[0])         # first recovery-frame value
        # lag = where the fit crosses M_R, minus t_R
        lag = (np.log(m_r) - ic) / k_re - t_r if k_re != 0 else np.nan
        lag_div = (d_first - RECOVERY_FRAME) / FPH      # recovery -> first division
        p1 = per_cycle_rates(m, 2018)
        if not p1:
            continue
        p1_rate = float(np.median([r["rate"] for r in p1]))
        rows.append(dict(ch=f"{pos}_{ch}", k_re=float(k_re), p1=p1_rate,
                         lag=float(lag), lag_div=float(lag_div),
                         n_re=len(v), d_first=d_first))
    return rows


def main():
    rows = collect()
    n = len(rows)
    k_re = np.array([r["k_re"] for r in rows])
    p1 = np.array([r["p1"] for r in rows])
    lag = np.array([r["lag"] for r in rows])
    lag_div = np.array([r["lag_div"] for r in rows])
    print(f"revived lineages used: {n}")
    print(f"k_regrow mean±SD = {k_re.mean():.3f}±{k_re.std(ddof=1):.3f}  "
          f"phase1 = {p1.mean():.3f}±{p1.std(ddof=1):.3f}  "
          f"frac(regrow>phase1) = {np.mean(k_re > p1):.2f}")
    try:
        _, pval = wilcoxon(k_re, p1)
    except ValueError:
        pval = np.nan
    print(f"Wilcoxon(k_re vs phase1) p={pval:.3g}")
    print(f"lag (crossing): median={np.median(lag):.2f}h  mean±SD="
          f"{lag.mean():.2f}±{lag.std(ddof=1):.2f}")
    print(f"lag_div (recovery->1st division): median={np.median(lag_div):.2f}h")

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 62 / 25.4),
                             constrained_layout=True)
    # Panel A — paired k_regrow vs phase1
    ax = axes[0]
    rng = np.random.default_rng  # not used (no randomness); jitter is deterministic
    xj = 0.04 * np.cos(np.arange(n))
    for i in range(n):
        ax.plot([0 + xj[i], 1 + xj[i]], [p1[i], k_re[i]], "-", color="#cccccc",
                lw=0.4, zorder=1)
    ax.plot(0 + xj, p1, "o", color=P1_C, ms=2.5, alpha=0.7, zorder=2)
    ax.plot(1 + xj, k_re, "o", color=RE_C, ms=2.5, alpha=0.7, zorder=2)
    for x, a, c in ((0, p1, P1_C), (1, k_re, RE_C)):
        ax.errorbar(x + 0.22, a.mean(), yerr=a.std(ddof=1), fmt="s", ms=4,
                    color=c, ecolor=c, elinewidth=1.2, capsize=3, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["phase1\nsteady", "regrowth"])
    ax.set_xlim(-0.4, 1.5)
    ax.set_ylabel(r"$d\ln M/dt$ [h$^{-1}$]")
    ax.set_title(f"regrowth vs phase1\n(paired, Wilcoxon p={pval:.2g})", fontsize=7)
    # Panel B — lag distribution
    ax = axes[1]
    ax.axvline(0, color="#888", lw=0.9, ls="--")
    ax.hist(lag, bins=np.linspace(min(-2, lag.min()), max(4, lag.max()), 24),
            color=RE_C, alpha=0.8)
    ax.axvline(np.median(lag), color="k", lw=1.0)
    ax.set_xlabel("lag [h]  (fit crosses M$_R$ − recovery)")
    ax.set_ylabel("lineages")
    ax.set_title(f"lag (median={np.median(lag):.2f} h)", fontsize=7)
    # Panel C — time to first division
    ax = axes[2]
    ax.hist(lag_div, bins=20, color="#0072B2", alpha=0.8)
    ax.axvline(np.median(lag_div), color="k", lw=1.0)
    ax.set_xlabel("recovery → 1st division [h]")
    ax.set_ylabel("lineages")
    ax.set_title(f"time to 1st division\n(median={np.median(lag_div):.1f} h)",
                 fontsize=7)
    fig.suptitle(f"F1 — revival regrowth rate & lag, revived lineages (EFD, n={n})",
                 fontsize=8)

    caption = (
        "Right after recovery to 2% glucose, revived mother lineages regrow their dry "
        "mass essentially without a lag, and at a rate comparable to (panel A reports "
        "whether higher/lower than) their own phase1 steady-state rate. Left panel: "
        "per-lineage paired d ln M/dt — phase1 steady state (grey) vs the post-recovery "
        "regrowth (vermilion); thin lines connect the same lineage; square markers = "
        "mean +/- SD; p = Wilcoxon signed-rank. Middle panel: lag distribution. Right "
        "panel: time from recovery to the first division. Operational definitions: "
        "recovery = frame 2885 (medium -> 2%; time_h = frame/12). regrowth window = "
        "[recovery, first division], where first division = the birth frame of the "
        "first division-bounded cycle starting at/after 2885 (enumerate_all_cycles; the "
        "long starvation-arrest cycle is dropped automatically). k_regrow = OLS slope "
        "of ln(mass_pg) vs time_h over the regrowth window (>= 4 valid frames; not "
        "is_outlier/touches_border; mass_pg >= 10). lag = (time at which the regrowth "
        "fit reaches M_R) - (time of the first recovery frame), where M_R = mass at the "
        "first valid frame >= 2885; lag ~ 0 means growth resumes immediately from the "
        "recovery-frame mass. phase1 steady state = the lineage's median per-cycle "
        "d ln M/dt over cycles ending at frame <= 2018. Aggregates are mean +/- SD "
        "(survivor convention). n = "
        f"{n} revived lineages with an EFD corrected lineage. Statistics: Wilcoxon "
        "signed-rank (paired, k_regrow vs phase1). Conditions: Schizosaccharomyces "
        "pombe, 260517 Mother-Machine starvation dataset (2% -> 0.0055% -> 0% -> 2%), "
        "QPI EFD volume variant [strain/genotype and temperature: confirm from "
        "acquisition metadata]. Abbreviations: SD, standard deviation; EFD, elliptic "
        "Fourier descriptor; OLS, ordinary least squares. Caveat: the lag uses a single "
        "exponential fit over the whole regrowth window; a pronounced initial flat/dip "
        "would bias it. Data availability: per-lineage k_regrow, phase1 rate, lag and "
        "time-to-first-division in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "recovery_frame": RECOVERY_FRAME,
                "n_revived": n, "k_regrow_mean": float(k_re.mean()),
                "phase1_mean": float(p1.mean()),
                "frac_regrow_gt_phase1": float(np.mean(k_re > p1)),
                "wilcoxon_p": float(pval) if np.isfinite(pval) else None,
                "lag_median_h": float(np.median(lag)),
                "lag_div_median_h": float(np.median(lag_div))},
        description="F1 revival: regrowth d ln M/dt vs phase1 steady (paired) + lag + "
                    "time-to-first-division, revived lineages, mean±SD (EFD)",
        caption=caption,
        data={"ch": np.array([r["ch"] for r in rows]), "k_regrow": k_re,
              "phase1_rate": p1, "lag_h": lag, "lag_div_h": lag_div,
              "n_regrow_frames": np.array([r["n_re"] for r in rows])},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
