"""_fig_dead_massvol_decoupling.py — A4 (preliminary): dead pre-death decoupling.

Companion to the revived A4 analysis. For the phase1-dead lineages (the 2% EMM
'~7 day' phase1 death series; swelling deaths), per completed cell cycle up to the
curated death window we fit BOTH d ln(dry mass)/dt and d ln(volume)/dt and
end-align by generation-before-death, to see preliminarily whether mass growth and
volume growth DECOUPLE in the terminal cell cycles. Aggregates are mean +/- SD. EFD.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_dead_massvol_decoupling.py
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
from gold_standard import list_phase1_dead, MANUAL_EXCLUDE  # noqa: E402
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import enumerate_all_cycles  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

MASS_C = "#0072B2"
VOL_C = "#009E73"
MINN = 5

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def _fit_log(t, y):
    sl, ic = np.polyfit(t, np.log(y), 1)
    return float(sl)


def cycle_rates_mass_vol(m, max_frame):
    cyc, _ = enumerate_all_cycles(m, set(), max_frame=max_frame, min_kept=6)
    out = []
    for c in cyc:
        v = c[~(c["is_outlier"] | c["touches_border"])]
        vm = v[v["mass_pg"] >= 10.0]
        vv = v[v["volume_um3_rod"] > 0]
        if len(vm) < 4 or len(vv) < 4:
            continue
        if np.any(vm["mass_pg"] <= 0) or np.any(vv["volume_um3_rod"] <= 0):
            continue
        out.append({"m_rate": _fit_log(vm["time_h"].to_numpy(float),
                                       vm["mass_pg"].to_numpy(float)),
                    "v_rate": _fit_log(vv["time_h"].to_numpy(float),
                                       vv["volume_um3_rod"].to_numpy(float)),
                    "f1": int(c["frame"].max())})
    return out


def main():
    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    swelling = [pc for pc in dead if f"{pc[0]}_{pc[1]}" not in ELONGATION]

    gm = defaultdict(list); gv = defaultdict(list); gd = defaultdict(list)
    per_lineage = {}
    for pos, ch in swelling:
        k = f"{pos}_{ch}"
        f1 = DEATH_WINDOW[k][1]
        rates = cycle_rates_mass_vol(load_mother(pos, ch), f1)
        if len(rates) < 2:
            continue
        n = len(rates)
        per_lineage[k] = rates
        for i, r in enumerate(rates):
            ge = -(n - i)                      # last completed cycle = -1
            gm[ge].append(r["m_rate"]); gv[ge].append(r["v_rate"])
            gd[ge].append(r["m_rate"] - r["v_rate"])
    n_lin = len(per_lineage)
    print(f"swelling dead lineages: {n_lin}")

    def agg(d):
        xs = sorted(g for g, v in d.items() if len(v) >= MINN)
        mu = np.array([np.mean(d[g]) for g in xs])
        sd = np.array([np.std(d[g], ddof=1) for g in xs])
        nn = np.array([len(d[g]) for g in xs])
        return np.array(xs, float), mu, sd, nn

    xm, mm, ms, mn = agg(gm)
    xv, vm, vs, _ = agg(gv)
    xd, dm, ds, dn = agg(gd)
    for g, m_, v_ in zip(xm, mm, vm):
        print(f"  gen{int(g):+d}: mass={m_:.3f}  vol={v_:.3f}  diff={m_-v_:+.3f}")

    fig, axes = plt.subplots(2, 1, figsize=(110 / 25.4, 120 / 25.4),
                             sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.errorbar(xm, mm, yerr=ms, color=MASS_C, lw=1.6, marker="o", ms=3, capsize=2,
                label="d ln M/dt (mass)")
    ax.errorbar(xv + 0.05, vm, yerr=vs, color=VOL_C, lw=1.6, marker="s", ms=3,
                capsize=2, label="d ln V/dt (volume)")
    ax.set_ylabel(r"specific growth rate [h$^{-1}$]")
    ax.legend(loc="lower left", frameon=False)
    ax.set_title(f"swelling deaths: per-cycle mass vs volume rate, end-aligned "
                 f"(mean±SD, n={n_lin})", fontsize=7)
    ax = axes[1]
    ax.axhline(0, color="#bbb", lw=0.7)
    ax.errorbar(xd, dm, yerr=ds, color="#222", lw=1.6, marker="D", ms=3, capsize=2)
    ax.set_xlabel("generation before death (-1 = last completed cycle)")
    ax.set_ylabel(r"$d\ln M/dt - d\ln V/dt$ [h$^{-1}$]")
    ax.set_title("decoupling (>0 = mass outgrows volume = densifying)", fontsize=7)
    fig.suptitle("A4 preliminary — pre-death mass/volume decoupling, phase1-dead "
                 "swelling lineages (EFD)", fontsize=8)

    caption = (
        "For the phase1-dead swelling lineages (the 2% EMM phase1 death series), the "
        "per-cell-cycle dry-mass and volume specific growth rates in the cycles leading "
        "up to death, end-aligned by generation before death, to check preliminarily "
        "whether mass and volume growth decouple terminally. x = generation before "
        "death (-1 = last completed cell cycle = the curated death window; more "
        "negative = earlier). Operational definitions: per division-bounded cycle up to "
        "the curated death window DEATH_WINDOW (panelA 2nd-to-last row), d ln M/dt and "
        "d ln V/dt = OLS slope of ln(mass_pg) resp. ln(volume_um3_rod) vs time_h (>= 4 "
        "valid frames; not is_outlier/touches_border; mass_pg >= 10; volume > 0). Top "
        "panel: mass rate (blue) and volume rate (green), mean +/- SD across lineages "
        "per generation-before-death. Bottom panel: per-cycle (mass rate - volume rate), "
        "mean +/- SD (>0 = mass outgrows volume = densifying; <0 = volume outgrows mass "
        "= diluting). A generation is shown only where >= "
        f"{MINN} lineages contribute. swelling death = phase1-dead lineages excluding "
        "the two elongation cascades (Pos20_ch06, Pos30_ch04; n=2, omitted as too few "
        f"for an aggregate). n = {n_lin} swelling-death lineages. Note: the terminal "
        "swelling itself happens AFTER the last completed cycle (it is not a "
        "division-bounded cycle), so this per-cycle view captures decoupling during the "
        "last NORMAL cycles, not the final balloon. Statistics: descriptive; error bars "
        "= SD across lineages. Conditions: Schizosaccharomyces pombe, 260517 "
        "Mother-Machine dataset, EMM + 2% glucose phase1 (frame <= 2018), QPI EFD "
        "volume variant [strain/genotype and temperature: confirm from acquisition "
        "metadata]. Abbreviations: SD, standard deviation; EFD, elliptic Fourier "
        "descriptor; OLS, ordinary least squares. Data availability: per-generation "
        "rates and per-lineage values in this run's *_data.npz."
    )
    data = {"gen_mass": xm, "mean_mass": mm, "sd_mass": ms, "n_mass": mn,
            "gen_vol": xv, "mean_vol": vm, "sd_vol": vs,
            "gen_diff": xd, "mean_diff": dm, "sd_diff": ds, "n_diff": dn}
    for k, rates in per_lineage.items():
        data[f"{k}_m_rate"] = np.array([r["m_rate"] for r in rates])
        data[f"{k}_v_rate"] = np.array([r["v_rate"] for r in rates])
    save_figure(
        fig,
        params={"volume_variant": "efd", "n_swelling": n_lin, "min_n": MINN},
        description="A4 preliminary: pre-death per-cycle mass vs volume growth rate "
                    "(d ln M/dt, d ln V/dt) end-aligned by generation-before-death, "
                    "phase1-dead swelling lineages, mean±SD (EFD)",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
