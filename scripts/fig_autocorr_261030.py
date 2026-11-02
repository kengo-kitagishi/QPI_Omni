"""Autocorrelation of mother-cell quantities, 261030 YE (frames 2-430), same 8 panels and estimator as
fig_autocorr_261024_vs_260517.py (imported): per-mother C_i(k) about the dataset mean, mean +- SEM over mothers.

Selection follows fig_density_returnmap_261024_vs_260517.mothers / cycles (imported, incl. the obvious-outlier rule)
with two 261030 settings:
  - frames 2-430 (contamination after 430)
  - mothers with >= MIN_MOTHER_FRAMES_261030 valid frames (429 frames in total; same 300 as the lineage gallery)
Panel h (time) uses living frames only: up to the mother's last QC-validated division, or to frame 430 if that
division is within ALIVE_GAP_H of it (fig_living_density_261030.py). Panels a-g use cycles between validated
divisions, which never include the non-dividing period before death.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import fig_density_returnmap_261024_vs_260517 as base  # noqa: E402  (also sets rcParams)
import fig_autocorr_261024_vs_260517 as ac  # noqa: E402
from figure_logger import save_figure  # noqa: E402

DS = "261030"
CONS = Path(r"C:\QPI_analysis\261030\seg\_lineage_consolidated")
MIN_MOTHER_FRAMES_261030 = 300
ALIVE_GAP_H = 6.0
MAX_GEN = 13               # ~35 h / 2.7 h
base.FRAME_WINDOWS[DS] = (2, 430)
base.COLORS[DS] = "#D55E00"


def load():
    f = pd.read_csv(CONS / "all_cells_lineage_data3D.csv.gz",
                    usecols=["pos", "ch", "cell_id", "frame", "time_h", "mass_pg", "volume_um3_efd",
                             "density_pg_um3", "is_outlier", "touches_border"])
    f = f.rename(columns={"density_pg_um3": "dens", "volume_um3_efd": "vol"})
    dv = pd.read_csv(CONS / "all_cells_divisions_qc.csv.gz")
    return f, dv


def living_only(m, dv):
    """Mark frames after death (after the last validated division, unless censored) as not valid."""
    lo, hi = base.FRAME_WINDOWS[DS]
    d = dv[dv["is_mother_division"].astype(bool) & dv["validated"].astype(bool) & dv["frame"].between(lo, hi)]
    last = d.groupby(["pos", "ch"])["frame"].max()
    end = last.where((hi - last) * 5.0 / 60.0 > ALIVE_GAP_H, hi)
    e = pd.Series([end.get((p, c), -1) for p, c in zip(m["pos"], m["ch"])], index=m.index)
    m = m.copy()
    m["valid"] = m["valid"] & (m["frame"] <= e)
    return m, int((end < hi).sum())


def main():
    base.MIN_MOTHER_FRAMES = MIN_MOTHER_FRAMES_261030
    f, dv = load()
    m = base.mothers(f, None, DS)
    cyc = base.cycles(m, dv, DS)[0]
    m_alive, n_dead = living_only(m, dv)

    fig, axes2 = plt.subplots(2, 4, figsize=(183 / 25.4, 92 / 25.4), constrained_layout=True)
    axes = list(axes2.ravel())
    gen_specs = [("rho", "cycle-mean concentration"), ("rhob", "birth concentration"),
                 ("Mb", "birth mass"), ("Vb", "birth volume"),
                 ("Vmean", "cycle-mean volume"), ("dMdt", "mass production rate"),
                 ("T_h", "interdivision time")]
    arrays, summary = {}, {}
    col = base.COLORS[DS]
    for ax, (key, title), letter in zip(axes, gen_specs, "abcdefg"):
        c, sem, n = ac.acf(*ac.per_mother_sums(ac.generation_series(cyc, key), MAX_GEN))
        k = np.arange(MAX_GEN + 1)
        ax.errorbar(k, c, yerr=sem, fmt="o-", ms=1.2, lw=0.7, elinewidth=0.5, capsize=0, color=col, label=DS)
        arrays.update({f"{key}_acf": c, f"{key}_sem": sem, f"{key}_n_mothers": n})
        summary[f"{key}_C1"] = round(float(c[1]), 3)
        ax.set_title(title, fontsize=7)
        ax.axhline(0, color="0.5", lw=0.5)
        ax.set_xlabel("generation lag k")
        ax.set_ylabel("autocorrelation C(k)")
        ax.set_xticks(range(0, MAX_GEN + 1, 3))
        ax.set_xlim(-0.5, MAX_GEN + 0.5)
        ax.set_ylim(min(-0.3, float(np.nanmin(c - sem)) - 0.05), 1.02)   # f dips below -0.3 in 261030
        ax.text(-0.3, 1.04, letter, transform=ax.transAxes, fontweight="bold", fontsize=8)
    t_ax = axes[7]
    max_lag_fr = int(round(30.0 / ac.DT_H))
    c, sem, n = ac.acf(*ac.per_mother_sums(ac.time_series(m_alive, DS), max_lag_fr))
    tau = np.arange(max_lag_fr + 1) * ac.DT_H
    t_ax.plot(tau, c, lw=0.8, color=col)
    t_ax.fill_between(tau, c - sem, c + sem, color=col, alpha=0.25, lw=0)
    arrays.update({"conc_time_acf": c, "conc_time_sem": sem, "conc_time_n_mothers": n, "tau_h": tau})
    t_ax.axhline(0, color="0.5", lw=0.5)
    t_ax.set_xlabel("time lag τ [h]")
    t_ax.set_ylabel("autocorrelation C(τ)")
    t_ax.set_title("mother concentration (time, living)", fontsize=7)
    t_ax.set_xlim(0, 30)
    t_ax.set_ylim(-0.3, 1.02)
    t_ax.text(-0.3, 1.04, "h", transform=t_ax.transAxes, fontweight="bold", fontsize=8)
    summary["n_mothers"] = int(m.groupby(["pos", "ch"]).ngroups)
    summary["n_mothers_stopped_dividing"] = n_dead
    summary["n_cycles"] = int(len(cyc))
    print(summary)
    print(base.OUTLIER_LOG)

    save_figure(
        fig, fmt="png", dpi=300,
        params={"outliers": base.OUTLIER_LOG, "max_gen": MAX_GEN, "frames": base.FRAME_WINDOWS[DS],
                "min_mother_frames": MIN_MOTHER_FRAMES_261030, "alive_gap_h": ALIVE_GAP_H,
                "min_pairs_per_mother": ac.MIN_PAIRS_PER_MOTHER, "min_mothers": ac.MIN_MOTHERS, "C1": summary},
        description="Generation and time autocorrelation of mother-cell quantities, 261030 YE (frames 2-430).",
        data=arrays,
        caption=(
            "Autocorrelation of mother-cell quantities, 261030 YE, frames 2-430 (contamination after 430). "
            "a-g, generation autocorrelation over contiguous cycles of the same mother (cycle = between two "
            "QC-validated mother divisions): a cycle-mean dry-mass concentration, b birth concentration, c birth mass, "
            "d birth volume (b-d: median of the first 3 valid frames after division), e cycle-mean volume (efd), "
            "f mass production rate (linear fit of dry mass vs time over the cycle, pg/h), g interdivision time. "
            "h, time autocorrelation of the mother concentration over living frames only (to the last validated "
            f"division, or to frame 430 if that division is within {ALIVE_GAP_H:.0f} h of it; "
            f"{n_dead} mothers stopped dividing earlier). For each mother C_i(k) = mean product of deviations from "
            "the dataset mean at lag k (or tau) / that mother's mean squared deviation; points = mean over mothers, "
            f"error bars (h: band) = SEM over mothers with >= {ac.MIN_PAIRS_PER_MOTHER} pairs at that lag; lags with "
            f"< {ac.MIN_MOTHERS} mothers are not drawn. n mothers {summary['n_mothers']} (>= "
            f"{MIN_MOTHER_FRAMES_261030} valid frames, edge traps excluded), {summary['n_cycles']} cycles. Obvious "
            f"outliers |conc - median| > {base.OUTLIER_Z} x 1.4826 MAD ({base.OUTLIER_LOG.get(DS)}). "
            "Concentration does not depend on the medium RI. No statistical test."),
        related=[{"note": "same estimator as fig_autocorr_261024_vs_260517 (261024 / 260517, 2% EMM)"}],
    )


if __name__ == "__main__":
    main()
