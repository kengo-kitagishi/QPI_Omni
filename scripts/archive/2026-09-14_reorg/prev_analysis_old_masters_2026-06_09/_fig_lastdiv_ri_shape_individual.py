"""_fig_lastdiv_ri_shape_individual.py — RI vs roundness per swelling lineage.

Replaces the averaged, mass-collapse-anchored death-aligned figure. Each
swelling-death lineage is shown individually (1xN), aligned at the CURATED last
division (x = 0; f1 = DEATH_WINDOW from the panelA 2nd-to-last row — NOT a
re-derived lysis point). Per panel, twin axes overlay cycle-mean RI (mean_ri) and
roundness (minor/major = short_axis_um / long_axis_um) so you can read, per
lineage, which one departs first and how RI behaves in the post-division swelling
phase. Window -20..+12 h, raw (no truncation; the post-f1 region is not curated
and may include trench repopulation). EFD.

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_lastdiv_ri_shape_individual.py
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

RI_C = "#0072B2"       # blue — mean RI
SHAPE_C = "#D55E00"    # vermilion — roundness (minor/major)
DIV_C = "#888888"
PRE_H = 20.0
POST_H = 12.0

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.labelsize": 7,
    "axes.titlesize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "legend.fontsize": 6, "axes.spines.top": False, "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def collect(ch_key):
    pos, ch = ch_key.split("_", 1)
    f1 = DEATH_WINDOW[ch_key][1]                    # curated last division (panelA)
    m = load_mother(pos, ch)
    rates = per_cycle_rates(m, f1)
    pre = m[m["frame"] <= f1]
    if pre.empty:
        return None
    t_ld = float(pre["time_h"].max())              # x origin (0) = curated last division
    v = m[~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)
          & (m["long_axis_um"] > 0)]
    t = v["time_h"].to_numpy(float) - t_ld
    win = (t >= -PRE_H) & (t <= POST_H)
    t = t[win]
    ri = v["mean_ri"].to_numpy(float)[win]
    mm = (v["short_axis_um"] / v["long_axis_um"]).to_numpy(float)[win]
    div_t = [r["t1"] - t_ld for r in rates if -PRE_H <= r["t1"] - t_ld <= POST_H]
    return dict(ch=ch_key, t=t, ri=ri, mm=mm, div_t=np.array(div_t, float),
                ncyc=len(rates))


def main():
    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    swelling = [f"{pos}_{ch}" for pos, ch in dead
                if f"{pos}_{ch}" not in ELONGATION]
    items = [c for c in (collect(k) for k in swelling) if c is not None]
    items.sort(key=lambda d: -d["ncyc"])
    n = len(items)
    print(f"swelling lineages: {n}")

    # minor/major range common across panels; RI is zoomed per panel (below) so its
    # change is visible — RI baselines differ between lineages, so a common RI range
    # would hide the within-lineage variation.
    mm_all = np.concatenate([d["mm"] for d in items])
    mm_lo, mm_hi = np.nanpercentile(mm_all, [1, 99])
    mm_lo = max(0.0, mm_lo - 0.03); mm_hi = min(1.0, mm_hi + 0.03)

    fig, axes = plt.subplots(n, 1, figsize=(120 / 25.4, (24 * n + 10) / 25.4),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    data: dict = {}
    for i, (ax, d) in enumerate(zip(axes, items)):
        ax2 = ax.twinx()
        ax2.spines["top"].set_visible(False)
        ax.axvspan(0, POST_H, color="#999", alpha=0.07, lw=0, zorder=0)
        for dt_ in d["div_t"]:
            ax.axvline(dt_, color=DIV_C, lw=0.5, ls=":", alpha=0.6, zorder=1)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.7, zorder=2)
        ln1 = ax.plot(d["t"], d["ri"], color=RI_C, lw=0.8, marker="o", ms=1.3,
                      mew=0, zorder=4, label="mean RI")
        ln2 = ax2.plot(d["t"], d["mm"], color=SHAPE_C, lw=0.8, marker="s", ms=1.3,
                       mew=0, zorder=3, label="minor/major")
        ax.set_xlim(-PRE_H, POST_H)
        rlo, rhi = np.nanpercentile(d["ri"], [2, 98])   # per-panel RI zoom (robust)
        pad = 0.12 * (rhi - rlo) + 1e-3
        ax.set_ylim(rlo - pad, rhi + pad)
        ax2.set_ylim(mm_lo, mm_hi)
        ax.set_ylabel("mean RI", color=RI_C); ax.tick_params(axis="y", labelcolor=RI_C)
        ax2.set_ylabel("minor/major", color=SHAPE_C)
        ax2.tick_params(axis="y", labelcolor=SHAPE_C)
        ax.tick_params(length=2); ax2.tick_params(length=2)
        ax.set_title(f"{d['ch']}  g{d['ncyc']}", fontsize=6)
        if i == 0:
            ax.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2],
                      loc="upper left", frameon=False, fontsize=5.5, ncol=2)
        data[f"{d['ch']}_t"] = d["t"]
        data[f"{d['ch']}_mean_ri"] = d["ri"]
        data[f"{d['ch']}_minmaj"] = d["mm"]
        data[f"{d['ch']}_division_t"] = d["div_t"]
    axes[-1].set_xlabel("time from last division [h]")
    fig.suptitle(
        f"mean RI (blue) vs roundness minor/major (vermilion) per swelling-death "
        f"lineage, aligned at the curated last division (x=0; panelA f1); "
        f"-{PRE_H:g}..+{POST_H:g} h, raw. black dashed = last division, grey dotted = "
        f"division, shaded = post-division (EFD, n={n}).", fontsize=6.5)

    caption = (
        "Per swelling-death lineage, cycle-mean RI and cell roundness over a window "
        "spanning the last cell cycles and the post-division swelling phase, aligned "
        "at the curated last division, so one can read per lineage whether RI rises in "
        "the swelling phase and whether RI or shape change departs first (no averaging "
        "across lineages). Each panel is one swelling-death lineage (twin axes). x = "
        "time from last division [h]; left axis (blue) = mean RI; right axis "
        "(vermilion) = roundness. Operational definitions: mean RI = EFD mean_ri of the "
        "rank-1 mother per frame; roundness = minor/major = short_axis_um / "
        "long_axis_um (rises from ~0.3 for a rod toward 1 as the cell balloons); valid "
        "frames only (not is_outlier/touches_border, mass_pg >= 10, long axis > 0). "
        "x = 0 is the CURATED last division f1 = DEATH_WINDOW (panelA 2nd-to-last row), "
        "drawn as the black dashed line; grey dotted lines = earlier divisions (cycle "
        "ends from enumerate_all_cycles); shaded span = the post-division phase, shown "
        f"raw with NO truncation (the post-f1 region is not curated and may include "
        "trench repopulation, so a late signal could come from a new cell). The "
        "minor/major axis range is common across panels; the RI axis is zoomed per "
        "panel (each lineage's robust 2-98th percentile) so the within-lineage RI "
        "change is visible (RI baselines differ between lineages). "
        f"window -{PRE_H:g}..+{POST_H:g} h. swelling death = phase1-dead lineages "
        f"excluding the two elongation cascades (Pos20_ch06, Pos30_ch04). n = {n} "
        "swelling lineages, one panel each, sorted by lifespan. Statistics: "
        "descriptive (individual lineages; no test). Conditions: Schizosaccharomyces "
        "pombe, 260517 Mother-Machine dataset, EMM + 2% glucose, phase1 (frame <= "
        "2018), QPI EFD volume variant [strain/genotype and temperature: confirm from "
        "acquisition metadata]. Abbreviations: RI, refractive index; EFD, elliptic "
        "Fourier descriptor. Data availability: per-lineage RI, roundness and division "
        "times in this run's *_data.npz. Interpretation: across most swelling lineages "
        "mean RI rises GRADUALLY through the post-division swelling phase (RI up = "
        "density up — not the drop one might expect from a ballooning cell), and mean "
        "RI and roundness rise together, so which of the two departs from baseline "
        "first is ambiguous — there is no consistent RI-vs-shape lead in this "
        "individual-lineage view. Caveat: the post-division region is not curated and "
        "may include trench repopulation, so part of a late RI rise could come from a "
        "new cell rather than the swelling mother."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "n_swelling": n, "pre_h": PRE_H,
                "post_h": POST_H, "anchor": "curated_last_division_f1"},
        description="per-lineage (1xN) mean RI vs roundness(minor/major), aligned at "
                    "the curated last division (panelA f1), -20..+12 h, twin axes; EFD",
        caption=caption,
        data=data,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
