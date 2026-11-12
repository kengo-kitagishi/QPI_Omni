"""261108 YE run 3 autocorrelation: dataset mean subtracted vs each lineage's own mean subtracted.

Same data, selection and estimator as fig_autocorr_261108.py (imported), except where the mean comes from:
  dataset mean : x - mu, mu over all mothers (between-mother differences count as correlation)
  own mean     : x - mean of that mother's own series (only memory within a lineage remains)
Per mother C_i(k) = mean product of the deviations at lag k / that mother's mean squared deviation;
points = mean over mothers, error bars (h: band) = SEM over mothers.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fig_autocorr_261108 as s  # noqa: E402  (sets the 261108 window, EDGE, MIN_MOTHERS)
import fig_autocorr_261024_vs_260517 as ac  # noqa: E402
import fig_density_returnmap_261024_vs_260517 as base  # noqa: E402
from figure_logger import save_figure  # noqa: E402

COL = {"dataset mean": "#D55E00", "own mean": "#0072B2"}


def sums_own_mean(series_list, max_lag):
    """As ac.per_mother_sums, but each series is centred on its own mean."""
    sxy = np.zeros((len(series_list), max_lag + 1))
    nxy = np.zeros((len(series_list), max_lag + 1))
    for i, x in enumerate(series_list):
        d = x - np.nanmean(x)
        ok = np.isfinite(d)
        d0 = np.where(ok, d, 0.0)
        for k in range(min(max_lag, len(x) - 1) + 1):
            m = ok[: len(x) - k] & ok[k:]
            sxy[i, k] = (d0[: len(x) - k] * d0[k:])[m].sum()
            nxy[i, k] = m.sum()
    return sxy, nxy


def main():
    base.MIN_MOTHER_FRAMES = s.MIN_MOTHER_FRAMES_261030
    f, dv = s.load()
    m = base.mothers(f, None, s.DS)
    m, excluded = s.drop_excluded(m)
    cyc = base.cycles(m, dv, s.DS)[0]
    m_alive, _ = s.living_only(m, dv)
    specs = [("rho", "cycle-mean concentration"), ("rhob", "birth concentration"), ("Mb", "birth mass"),
             ("Vb", "birth volume"), ("Vmean", "cycle-mean volume"), ("dMdt", "mass production rate"),
             ("T_h", "interdivision time")]
    fig, axes2 = plt.subplots(2, 4, figsize=(183 / 25.4, 92 / 25.4), constrained_layout=True)
    axes = list(axes2.ravel())
    arrays, c1 = {}, {}
    k = np.arange(s.MAX_GEN + 1)
    for ax, (key, title), letter in zip(axes, specs, "abcdefg"):
        series = ac.generation_series(cyc, key)
        lows = [-0.3]
        for lab, fn in (("dataset mean", ac.per_mother_sums), ("own mean", sums_own_mean)):
            c, sem, n = ac.acf(*fn(series, s.MAX_GEN))
            ax.errorbar(k, c, yerr=sem, fmt="o-", ms=1.2, lw=0.7, elinewidth=0.5, capsize=0, color=COL[lab], label=lab)
            tag = lab.split()[0]
            arrays.update({f"{tag}_{key}_acf": c, f"{tag}_{key}_sem": sem, f"{tag}_{key}_n_mothers": n})
            c1[f"{tag}_{key}_C1"] = round(float(c[1]), 3)
            lows.append(float(np.nanmin(c - sem)) - 0.05)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel("generation lag k")
        ax.set_ylabel("autocorrelation C(k)")
        ax.set_xticks(range(0, s.MAX_GEN + 1, 4))
        ax.set_ylim(min(lows), 1.02)
        ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    axes[0].legend(fontsize=5.5, loc="upper right")
    t_ax = axes[7]
    max_lag_fr = int(round(50.0 / ac.DT_H))
    tau = np.arange(max_lag_fr + 1) * ac.DT_H
    ts = ac.time_series(m_alive, s.DS)
    for lab, fn in (("dataset mean", ac.per_mother_sums), ("own mean", sums_own_mean)):
        c, sem, n = ac.acf(*fn(ts, max_lag_fr))
        t_ax.plot(tau, c, lw=0.8, color=COL[lab])
        t_ax.fill_between(tau, c - sem, c + sem, color=COL[lab], alpha=0.25, lw=0)
        arrays.update({f"{lab.split()[0]}_conc_time_acf": c, f"{lab.split()[0]}_conc_time_sem": sem})
    arrays["tau_h"] = tau
    t_ax.axhline(0, color="0.5", lw=0.5)
    t_ax.set_xlabel("time lag τ [h]")
    t_ax.set_ylabel("autocorrelation C(τ)")
    t_ax.set_title("mother concentration (time, living)", fontsize=7)
    t_ax.set_xlim(0, 50)
    t_ax.set_ylim(-0.6, 1.02)
    t_ax.text(-0.3, 1.04, "h", transform=t_ax.transAxes, fontweight="bold", fontsize=8)
    print(c1)
    save_figure(
        fig, fmt="png", dpi=300,
        params={"max_gen": s.MAX_GEN, "min_mothers": ac.MIN_MOTHERS, "C1": c1},
        description="261108 YE run 3 autocorrelation: dataset mean vs each lineage's own mean subtracted.",
        data=arrays,
        caption=("Autocorrelation of mother-cell quantities, 261108 YE run 3 (21 traps, frames 2-769), selection and "
                 "panels as fig_autocorr_261108; excluded traps: " + ", ".join(f"{p} {c}" for p, c in excluded) + ". Vermillion: deviations from the dataset mean (between-mother differences "
                 "count as correlation). Blue: deviations from each mother's own mean (only memory within a lineage). "
                 "Per mother C_i(k) = mean product of deviations at lag k / that mother's mean squared deviation; points = "
                 f"mean over mothers, error bars (h: band) = SEM over mothers; lags with < {ac.MIN_MOTHERS} mothers not "
                 "drawn. Subtracting each lineage's own mean biases C(k) slightly negative for short series. "
                 "No statistical test."),
        related=[{"note": "dataset-mean version: fig_autocorr_261108"}],
    )


if __name__ == "__main__":
    main()
