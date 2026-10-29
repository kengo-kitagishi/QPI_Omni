"""Why does the generation ACF of cycle-mean concentration plateau while the time ACF keeps falling?

Checks two candidates on the same data as fig_autocorr_261024_vs_260517.py:
  1. a population-wide trend of concentration over the experiment. The ACFs subtract one dataset
     mean, so a trend lowers C at lags comparable to the run length (261024 is only 55 h long).
     -> recompute both ACFs after subtracting the time-dependent population mean (1-h bins;
        for cycles, the bin of the cycle's midpoint).
  2. the time spanned by a generation lag: k is converted to hours with the observed mean
     elapsed time between cycle k0 and k0+k, so both ACFs share one time axis.

a  population mean concentration of the mothers vs time (1-h bins)
b  261024: C_time(tau) and C_gen(k) placed at the mean elapsed time, raw and trend-removed
c  260517: same
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
import fig_autocorr_261024_vs_260517 as acf  # noqa: E402
from figure_logger import save_figure  # noqa: E402

BIN_FR = 12          # 1 h
MAX_GEN = 18


def binned_mean(m, ds):
    lo, hi = base.FRAME_WINDOWS[ds]
    v = m[m["valid"]]
    b = (v["frame"].astype(int) - lo) // BIN_FR
    mean = v.groupby(b)["conc"].mean()
    nb = (hi - lo) // BIN_FR + 1
    out = np.full(nb, np.nan)
    out[mean.index.values] = mean.values
    return out


def time_series_detr(m, ds, trend):
    lo, hi = base.FRAME_WINDOWS[ds]
    out = []
    for _, g in m[m["valid"]].groupby(["pos", "ch"]):
        s = np.full(hi - lo + 1, np.nan)
        fr = g["frame"].astype(int).values - lo
        s[fr] = g["conc"].values - trend[fr // BIN_FR]
        out.append(s)
    return out


def time_series_in_cycle_span(m, cyc, ds):
    """Mother concentration restricted to [first cycle start, last cycle end) of that mother,
    i.e. the same span the generation ACF draws from."""
    lo, hi = base.FRAME_WINDOWS[ds]
    span = cyc.groupby(["pos", "ch"]).agg(a=("start", "min"), b=("end", "max"))
    out = []
    for (pos, ch), g in m[m["valid"]].groupby(["pos", "ch"]):
        if (pos, ch) not in span.index:
            continue
        a, b = span.loc[(pos, ch)]
        g = g[(g["frame"] >= a) & (g["frame"] < b)]
        s = np.full(hi - lo + 1, np.nan)
        s[g["frame"].astype(int).values - lo] = g["conc"].values
        out.append(s)
    return out


def gen_series(cyc, ds, trend, detrend):
    lo, _ = base.FRAME_WINDOWS[ds]
    vals, times = [], []
    for _, g in cyc.groupby(["pos", "ch"]):
        g = g.sort_values("k")
        n = int(g["k"].max()) + 1
        s = np.full(n, np.nan)
        t = np.full(n, np.nan)
        k = g["k"].astype(int).values
        mid = ((g["start"] + g["end"]) / 2).astype(int).values - lo
        s[k] = g["rho"].values - (trend[np.clip(mid // BIN_FR, 0, len(trend) - 1)] if detrend else 0.0)
        t[k] = mid * 5.0 / 60.0
        vals.append(s)
        times.append(t)
    return vals, times


def mean_elapsed(times, max_k):
    """Mean elapsed time [h] between cycle i and i+k over all mothers."""
    out = np.full(max_k + 1, np.nan)
    for k in range(max_k + 1):
        d = [t[k:] - t[: len(t) - k] for t in times if len(t) > k]
        d = np.concatenate(d) if d else np.array([])
        d = d[np.isfinite(d)]
        if len(d):
            out[k] = d.mean()
    return out


def main():
    data = {}
    f, dv, _ = base.load_261024()
    m = base.mothers(f, None, "261024")
    data["261024"] = (m, base.cycles(m, dv, "261024")[0])
    f, dv, (keep, tag) = base.load_260517()
    m = base.mothers(f, keep, "260517")
    data["260517"] = (m, base.cycles(m, dv, "260517")[0])

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 58 / 25.4), constrained_layout=True)
    arrays, summary = {}, {}
    max_lag_fr = int(round(50.0 / acf.DT_H))
    for ds in ("260517", "261024"):
        m, cyc = data[ds]
        trend = binned_mean(m, ds)
        th = np.arange(len(trend)) + 0.5
        axes[0].plot(th, trend, lw=0.9, color=base.COLORS[ds], label=ds)
        arrays[f"{ds}_trend_mg_ml"] = trend
        arrays[f"{ds}_trend_t_h"] = th
        ok = np.isfinite(trend)
        summary[f"{ds}_trend_first5h"] = round(float(np.nanmean(trend[:5])), 1)
        summary[f"{ds}_trend_last5h"] = round(float(np.nanmean(trend[ok][-5:])), 1)

        ax = axes[1] if ds == "261024" else axes[2]
        tau = np.arange(max_lag_fr + 1) * acf.DT_H
        for detr, ls, lab in ((False, "-", "raw"), (True, "--", "trend removed")):
            series = time_series_detr(m, ds, trend if detr else np.zeros_like(trend))
            c_t = acf.acf_with_ci(*acf.per_mother_sums(series, max_lag_fr))[0]
            vals, times = gen_series(cyc, ds, trend, detr)
            c_g = acf.acf_with_ci(*acf.per_mother_sums(vals, MAX_GEN))[0]
            tk = mean_elapsed(times, MAX_GEN)
            ax.plot(tau, c_t, ls=ls, lw=0.9, color=base.COLORS[ds], label=f"time ACF, {lab}")
            ax.plot(tk, c_g, ls=ls, lw=0.7, marker="o", ms=2, color="0.25",
                    label=f"generation ACF (cycle-mean), {lab}")
            key = "detr" if detr else "raw"
            arrays.update({f"{ds}_time_acf_{key}": c_t, f"{ds}_gen_acf_{key}": c_g,
                           f"{ds}_gen_elapsed_h": tk, "tau_h": tau})
            i50 = int(round(48 / acf.DT_H))
            summary[f"{ds}_time_C48h_{key}"] = round(float(c_t[i50]), 3)
            kk = int(np.nanargmin(np.abs(tk - 48))) if np.isfinite(tk).any() else 0
            summary[f"{ds}_gen_C_at_{tk[kk]:.0f}h_k{kk}_{key}"] = round(float(c_g[kk]), 3)
        c_s = acf.acf_with_ci(*acf.per_mother_sums(time_series_in_cycle_span(m, cyc, ds), max_lag_fr))[0]
        ax.plot(tau, c_s, ls=":", lw=1.1, color=base.COLORS[ds],
                label="time ACF, only frames between first and last division")
        arrays[f"{ds}_time_acf_cycle_span"] = c_s
        summary[f"{ds}_time_C48h_cycle_span"] = round(float(c_s[int(round(48 / acf.DT_H))]), 3)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlim(0, 50); ax.set_ylim(-0.75, 1.02)          # room for the legend below the curves
        ax.set_xlabel("lag [h]")
        ax.set_ylabel("autocorrelation")
        ax.set_title(ds, fontsize=7)
        ax.legend(fontsize=5, loc="lower left")

    axes[0].set_xlabel("time from img 2 [h]")
    axes[0].set_ylabel("mother mean concentration [mg/mL]")
    axes[0].set_title("population trend (1-h bins)", fontsize=7)
    axes[0].set_xlim(0, 60)
    axes[0].legend(fontsize=6, loc="best")
    for ax, letter in zip(axes, "abc"):
        ax.text(-0.25, 1.03, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)

    caption = (
        "Check of why the generation ACF of cycle-mean concentration plateaus while the time ACF of the mother "
        "concentration keeps falling. a, population mean concentration of the analysed mothers in 1-h bins (valid "
        "frames). b, c, time ACF of the mother concentration (coloured) and generation ACF of the cycle-mean "
        "concentration (grey, each lag k placed at the mean elapsed time between cycle midpoints k apart), computed "
        "with the dataset mean subtracted (solid, as in fig_autocorr) and with the time-dependent population mean "
        "subtracted (dashed; cycles use the bin of their midpoint); dotted, time ACF using only the frames between "
        "each mother's first and last QC-validated division (the span the generation ACF draws from). Same data and "
        "selection as fig_autocorr_261024_"
        f"vs_260517; lags with < {acf.MIN_PAIRS} pairs are not drawn. 260517 master {tag}."
        f" Obvious outliers removed: valid mother frames with |conc - median| > {base.OUTLIER_Z} x 1.4826 MAD "
        f"of the dataset ({base.OUTLIER_LOG})."
    )
    path = save_figure(
        fig, fmt="png", dpi=300,
        params={"outliers": base.OUTLIER_LOG, "bin_frames": BIN_FR, "max_gen": MAX_GEN, "summary": summary},
        description="Trend check for the generation vs time autocorrelation of mother concentration "
                    "(261024 vs 260517)",
        data=arrays, caption=caption,
        related=[{"note": "explains fig_autocorr_261024_vs_260517 a (generation) vs h (time)"}],
    )
    print(summary)
    print(base.OUTLIER_LOG)
    print("saved", path)


if __name__ == "__main__":
    main()
