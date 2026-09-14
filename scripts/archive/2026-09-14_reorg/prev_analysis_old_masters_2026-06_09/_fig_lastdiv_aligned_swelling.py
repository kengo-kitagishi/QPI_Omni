"""_fig_lastdiv_aligned_swelling.py — last-division-aligned population means.

Swelling-death lineages, phase-aligned at their LAST division (x = 0), window
-20 h to +3 h. Population mean +/- SEM of mean RI, dry-mass concentration
(density = 1000*mass_pg/volume_um3_rod), and roundness (minor/major). Aligning at
the last division phase-locks the final cell cycle, so a reproducible pre-division
RI dip (if any) survives averaging in the last ~2 h, while the +0..+3 h shows the
post-division terminal / swelling phase. Far-left bins have fewer lineages (short
lineages die young); mean +/- SEM is drawn only where >= MINN lineages contribute.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_lastdiv_aligned_swelling.py
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
from gold_standard import list_phase1_dead, MANUAL_EXCLUDE  # noqa: E402
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_growthrate_dead_vs_alive import per_cycle_rates  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

RI_C = "#0072B2"
DENS_C = "#009E73"
SHAPE_C = "#D55E00"
GRID = np.arange(-20.0, 3.01, 0.25)
MINN = 3
PRE_H = 20.0
POST_H = 3.0

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def lysis_time(m, f1):
    v = m[~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)]
    pre = v[v["frame"] <= f1]
    if pre.empty:
        return None
    tf1 = float(pre["time_h"].max())
    runmax = float(pre["mass_pg"].max())
    fwd = v[(v["time_h"] > tf1) & (v["time_h"] <= tf1 + 12.0)].sort_values("time_h")
    for _, row in fwd.iterrows():
        runmax = max(runmax, float(row["mass_pg"]))
        if float(row["mass_pg"]) < 0.5 * runmax:
            return float(row["time_h"])
    return None


def interp_grid(t, y):
    g = np.interp(GRID, t, y, left=np.nan, right=np.nan)
    g[(GRID < t.min()) | (GRID > t.max())] = np.nan
    return g


def aligned_trace(pos, ch):
    """Per-lineage signals on the common grid, x = time - last_division."""
    k = f"{pos}_{ch}"
    f1 = DEATH_WINDOW[k][1]
    m = load_mother(pos, ch)
    rates = per_cycle_rates(m, f1)
    if not rates:
        return None
    t_ld = rates[-1]["t1"]
    lys = lysis_time(m, f1)
    right_abs = t_ld + POST_H if lys is None else min(t_ld + POST_H, lys)
    v = m[~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)
          & (m["volume_um3_rod"] > 0)].copy()
    v = v[(v["time_h"] >= t_ld - PRE_H) & (v["time_h"] <= right_abs)]
    if len(v) < 5:
        return None
    t = v["time_h"].to_numpy(float) - t_ld
    ri = v["mean_ri"].to_numpy(float)
    dens = 1000.0 * v["mass_pg"].to_numpy(float) / v["volume_um3_rod"].to_numpy(float)
    mm = v["short_axis_um"].to_numpy(float) / v["long_axis_um"].to_numpy(float)
    return (interp_grid(t, ri), interp_grid(t, dens), interp_grid(t, mm))


def msem(a):
    mu = np.where(np.sum(np.isfinite(a), 0) > 0, np.nanmean(a, 0), np.nan)
    cnt = np.sum(np.isfinite(a), 0)
    sd = np.where(cnt > 1, np.nanstd(a, 0, ddof=1), np.nan)
    sem = np.where(cnt > 1, sd / np.sqrt(np.maximum(cnt, 1)), np.nan)
    return mu, sem, cnt


def main():
    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    swelling = [pc for pc in dead if f"{pc[0]}_{pc[1]}" not in ELONGATION]

    ri_a, dn_a, mm_a, chs = [], [], [], []
    for pos, ch in swelling:
        res = aligned_trace(pos, ch)
        if res is None:
            continue
        gri, gdn, gmm = res
        ri_a.append(gri); dn_a.append(gdn); mm_a.append(gmm)
        chs.append(f"{pos}_{ch}")
    ri_a = np.array(ri_a); dn_a = np.array(dn_a); mm_a = np.array(mm_a)
    n = len(chs)
    ri_mu, ri_se, cnt = msem(ri_a)
    dn_mu, dn_se, _ = msem(dn_a)
    mm_mu, mm_se, _ = msem(mm_a)
    sel = cnt >= MINN
    print(f"swelling lineages: {n}")
    for tt in (-20, -12, -8, -4, -2, 0, 2):
        i = int(np.argmin(np.abs(GRID - tt)))
        print(f"  t={tt:+3d}h  n={cnt[i]:2d}  meanRI={ri_mu[i]:.4f}  "
              f"density={dn_mu[i]:.1f}  minor/major={mm_mu[i]:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(120 / 25.4, 145 / 25.4),
                             sharex=True, constrained_layout=True)
    specs = [(axes[0], ri_mu, ri_se, ri_a, RI_C, "mean RI"),
             (axes[1], dn_mu, dn_se, dn_a, DENS_C, "density [mg/mL]"),
             (axes[2], mm_mu, mm_se, mm_a, SHAPE_C, "minor/major")]
    for ax, mu, se, arr, col, lab in specs:
        ax.axvspan(0, POST_H, color="#999", alpha=0.08, lw=0, zorder=0)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7, zorder=2)
        for row in arr:                       # faint individual lineages
            ax.plot(GRID, row, color=col, lw=0.4, alpha=0.10, zorder=1)
        ax.fill_between(GRID[sel], (mu - se)[sel], (mu + se)[sel], color=col,
                        alpha=0.25, lw=0, zorder=3)
        ax.plot(GRID[sel], mu[sel], color=col, lw=1.8, zorder=4)
        ax.set_ylabel(lab, color=col)
        ax.tick_params(axis="y", labelcolor=col)
    axes[2].set_xlabel("time from last division [h]")
    axes[0].set_title(f"swelling death, aligned at last division (x=0); shaded = "
                      f"post-division; mean±SEM where n≥{MINN} (EFD, n={n})",
                      fontsize=7)
    fig.suptitle("Last-division-aligned population means — swelling-death lineages",
                 fontsize=8)

    caption = (
        "Aligned at their last division, swelling-death lineages show their cycle-mean "
        "RI, dry-mass concentration and roundness as population means over a 23 h "
        "window spanning the final cell cycles and the terminal post-division phase, so "
        "any reproducible pre-division RI dip and the post-division decoupling of mass "
        "and volume can be read directly. x = time from last division [h] (last "
        f"division at 0; window -{PRE_H:g}..+{POST_H:g} h). Panels (top to bottom): "
        "mean RI; density = dry-mass concentration = 1000 * mass_pg / volume_um3_rod "
        "[mg/mL]; roundness = minor/major = short_axis_um / long_axis_um. Operational "
        "definitions: all quantities are EFD-based, rank-1 mother, valid frames only "
        "(not is_outlier/touches_border, mass_pg >= 10, volume > 0). last division = "
        "end (time t1) of the last completed cell cycle (curated death window, "
        "fig_panelA 2nd-to-last row). The post-division segment (shaded) is truncated "
        "at lysis (first post-division frame whose mass falls below half the running "
        f"maximum) when that occurs within +{POST_H:g} h. Visual elements: thick line "
        "= population mean across lineages, band = +/- SEM, drawn only where >= "
        f"{MINN} lineages contribute (far-left bins have fewer lineages because short "
        "lineages die young); faint thin lines = individual lineages; black dashed "
        "vertical line = the last division; grey shaded span = the post-division "
        f"terminal phase. swelling death = phase1-dead lineages excluding the two "
        f"elongation cascades (Pos20_ch06, Pos30_ch04). n = {n} swelling lineages. "
        "Statistics: descriptive; error bands are SEM across lineages (per time bin, n "
        "varies with lineage availability). Conditions: Schizosaccharomyces pombe, "
        "260517 Mother-Machine dataset, EMM + 2% glucose, phase1 (frame <= 2018), QPI "
        "EFD volume variant [strain/genotype and temperature: confirm from acquisition "
        "metadata]. Abbreviations: RI, refractive index; SEM, standard error of the "
        "mean; EFD, elliptic Fourier descriptor. Data availability: gridded per-lineage "
        "and population traces in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "n_swelling": n, "pre_h": PRE_H,
                "post_h": POST_H, "min_n": MINN},
        description="last-division-aligned (x=0) population mean±SEM of mean RI, "
                    "density and minor/major for swelling-death lineages, -20..+3 h "
                    "(EFD); faint individual lineages overlaid",
        caption=caption,
        data={"grid_t": GRID, "n_contrib": cnt,
              "ri_mean": ri_mu, "ri_sem": ri_se,
              "density_mean": dn_mu, "density_sem": dn_se,
              "minmaj_mean": mm_mu, "minmaj_sem": mm_se,
              "ch": np.array(chs), "ri_lineages": ri_a,
              "density_lineages": dn_a, "minmaj_lineages": mm_a},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
