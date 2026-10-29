"""Autocorrelation of mother-cell quantities: 261024 vs 260517 (2% EMM).

Same data and selection as fig_density_returnmap_261024_vs_260517.py (imported from there).

a-c  generation autocorrelation C(k), k = 0..MAX_GEN, of the cycle-mean concentration rho,
     birth mass M_b and interdivision time T, over contiguous cycles of the same mother.
d    time autocorrelation C(tau), tau = 0..MAX_LAG_H, of the mother concentration time series
     (valid frames only, 5-min grid).

Estimator (pooled over mothers, deviations from the dataset mean mu):
    C(k) = [sum_pairs (x_i - mu)(x_{i+k} - mu) / N_k] / [sum_i (x_i - mu)^2 / N_0]
95% CI: bootstrap over mothers (N_BOOT resamples of the mother list).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fig_density_returnmap_261024_vs_260517 as base  # noqa: E402  (also sets rcParams)
from figure_logger import save_figure  # noqa: E402

MAX_GEN = 10
MAX_LAG_H = 24.0
DT_H = 5.0 / 60.0
N_BOOT = 300
RNG = np.random.default_rng(0)


def per_mother_sums(series_list, max_lag):
    """series_list: 1-D arrays on a regular grid with NaN gaps. Returns per-mother
    (sum_xy[lag], n_pairs[lag]) after subtracting the pooled mean."""
    allv = np.concatenate([s[np.isfinite(s)] for s in series_list])
    mu = allv.mean()
    sxy = np.zeros((len(series_list), max_lag + 1))
    nxy = np.zeros((len(series_list), max_lag + 1))
    for i, s in enumerate(series_list):
        d = s - mu
        ok = np.isfinite(d)
        d0 = np.where(ok, d, 0.0)
        for k in range(min(max_lag, len(s) - 1) + 1):
            a, b = d0[: len(s) - k], d0[k:]
            m = ok[: len(s) - k] & ok[k:]
            sxy[i, k] = (a * b)[m].sum()
            nxy[i, k] = m.sum()
    return sxy, nxy


def acf_with_ci(sxy, nxy):
    def est(idx):
        s, n = sxy[idx].sum(0), nxy[idx].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (s / n) / (s[0] / n[0])
        return c
    allidx = np.arange(len(sxy))
    c = est(allidx)
    boots = np.array([est(RNG.integers(0, len(sxy), len(sxy))) for _ in range(N_BOOT)])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
    return c, lo, hi, nxy.sum(0)


def generation_series(cyc, key):
    out = []
    for _, g in cyc.groupby(["pos", "ch"]):
        g = g.sort_values("k")
        s = np.full(int(g["k"].max()) + 1, np.nan)
        s[g["k"].astype(int).values] = g[key].astype(float).values
        out.append(s)
    return out


def time_series(m, ds):
    lo, hi = base.FRAME_WINDOWS[ds]
    out = []
    for _, g in m[m["valid"]].groupby(["pos", "ch"]):
        s = np.full(hi - lo + 1, np.nan)
        s[g["frame"].astype(int).values - lo] = g["conc"].values
        out.append(s)
    return out


def main():
    data = {}
    f, dv, _ = base.load_261024()
    m = base.mothers(f, None, "261024")
    data["261024"] = (m, base.cycles(m, dv, "261024")[0])
    f, dv, (keep, tag) = base.load_260517()
    m = base.mothers(f, keep, "260517")
    data["260517"] = (m, base.cycles(m, dv, "260517")[0])

    fig, axes = plt.subplots(1, 4, figsize=(183 / 25.4, 48 / 25.4), constrained_layout=True)
    arrays, summary = {}, {}
    gen_specs = [("rho", "cycle-mean concentration"), ("Mb", "birth mass"), ("T_h", "interdivision time")]
    max_lag_fr = int(round(MAX_LAG_H / DT_H))
    for ds in ("260517", "261024"):
        m, cyc = data[ds]
        for ax, (key, title) in zip(axes[:3], gen_specs):
            c, lo, hi, n = acf_with_ci(*per_mother_sums(generation_series(cyc, key), MAX_GEN))
            k = np.arange(MAX_GEN + 1)
            ax.fill_between(k, lo, hi, color=base.COLORS[ds], alpha=0.2, lw=0)
            ax.plot(k, c, "o-", ms=2, lw=0.8, color=base.COLORS[ds], label=ds)
            arrays.update({f"{ds}_{key}_acf": c, f"{ds}_{key}_lo": lo, f"{ds}_{key}_hi": hi,
                           f"{ds}_{key}_npairs": n})
            summary[f"{ds}_{key}_C1"] = round(float(c[1]), 3)
            ax.set_title(title, fontsize=7)
        c, lo, hi, n = acf_with_ci(*per_mother_sums(time_series(m, ds), max_lag_fr))
        tau = np.arange(max_lag_fr + 1) * DT_H
        axes[3].fill_between(tau, lo, hi, color=base.COLORS[ds], alpha=0.2, lw=0)
        axes[3].plot(tau, c, lw=0.8, color=base.COLORS[ds], label=ds)
        arrays.update({f"{ds}_conc_time_acf": c, f"{ds}_conc_time_lo": lo, f"{ds}_conc_time_hi": hi,
                       f"{ds}_conc_time_npairs": n, "tau_h": tau})
        summary[f"{ds}_n_mothers"] = int(m.groupby(["pos", "ch"]).ngroups)

    for ax, letter in zip(axes[:3], "abc"):
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlabel("generation lag k")
        ax.set_ylabel("autocorrelation C(k)")
        ax.set_xticks(range(0, MAX_GEN + 1, 2))
        ax.set_ylim(-0.3, 1.02)
        ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    axes[0].legend(fontsize=6, loc="upper right")
    ax = axes[3]
    ax.axhline(0, color="0.5", lw=0.5)
    ax.set_xlabel("time lag τ [h]")
    ax.set_ylabel("autocorrelation C(τ)")
    ax.set_title("mother concentration (time)", fontsize=7)
    ax.set_xlim(0, MAX_LAG_H)
    ax.set_ylim(-0.3, 1.02)
    ax.text(-0.3, 1.04, "d", transform=ax.transAxes, fontweight="bold", fontsize=8)

    caption = (
        "Autocorrelation of mother-cell quantities, 261024 (2% EMM, 55 h) vs 260517 (2% EMM, 7 d). "
        "a-c, generation autocorrelation C(k) over contiguous cycles of the same mother (cycle = between two "
        "QC-validated mother divisions) of a cycle-mean dry-mass concentration, b birth mass (median of the first "
        "3 valid frames after division), c interdivision time. d, time autocorrelation of the mother dry-mass "
        "concentration (valid frames, 5-min grid). C = pooled covariance of deviations from the dataset mean at "
        "lag k (or tau) divided by the pooled variance; shaded = 95% bootstrap CI over mothers "
        f"({N_BOOT} resamples). n mothers: 261024 {summary['261024_n_mothers']}, 260517 "
        f"{summary['260517_n_mothers']}. Concentration = efd density x 1000; 261024 n_medium 1.335029 carried over "
        f"from 260517. 260517 master {tag}, img 2-2017, analysis_recommended channels; 261024 img 2-666, mothers "
        f"with >= {base.MIN_MOTHER_FRAMES} valid frames, edge traps excluded."
    )
    out = save_figure(
        fig, fmt="png", dpi=300,
        params={"max_gen": MAX_GEN, "max_lag_h": MAX_LAG_H, "n_boot": N_BOOT, "seed": 0,
                "min_mother_frames": base.MIN_MOTHER_FRAMES, "master_260517": tag, "C1": summary},
        description="Generation and time autocorrelation of mother concentration, birth mass and "
                    "interdivision time: 261024 vs 260517, 2% glucose",
        data=arrays, caption=caption,
        related=[{"note": "same data/selection as fig_density_returnmap_261024_vs_260517 (return maps = lag 1)"}],
    )
    print(summary)
    print("saved", out)


if __name__ == "__main__":
    main()
