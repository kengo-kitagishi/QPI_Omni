"""_fig_death_aligned_ri_vs_shape.py — does RI drop or shape change lead death?

Swelling-death lineages, aligned at the death (lysis) frame. For each lineage we
overlay two terminal time series — cycle-mean RI (mean_ri) and cell roundness
(minor/major = short_axis_um / long_axis_um, which rises as a rod balloons into a
sphere) — then ask which one departs from its baseline FIRST. Death = the first
post-(last-division) mass collapse (the swollen mother lyses and a small new cell
repopulates the trench); t_rel = time - t_death (death at 0, pre-death negative).

Panel A: population mean +/- SEM of each signal in baseline-SD units (z), so both
sit on one comparable axis (RI goes negative, roundness goes positive near death);
the signal whose curve leaves 0 earlier leads. Panel B: raw population means on
twin axes. We also compute, per lineage, the onset time at which each signal first
crosses 2 baseline SD, and test the paired difference (Wilcoxon).

Run: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_death_aligned_ri_vs_shape.py
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
from gold_standard import list_phase1_dead, MANUAL_EXCLUDE  # noqa: E402
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from figure_logger import save_figure  # noqa: E402

RI_C = "#0072B2"       # blue  — mean RI
SHAPE_C = "#D55E00"    # vermilion — roundness (minor/major)
FWD_H = 12.0           # search this many h past last division for the lysis collapse
COLLAPSE = 0.5         # lysis = first frame whose mass < COLLAPSE * running-max
BASE_H = 2.0           # baseline = earliest BASE_H h of each terminal segment
GRID = np.arange(-8.0, 0.01, 0.25)
Z_ONSET = 2.0          # onset = first crossing of this many baseline SD

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def load_mother(pos, ch):
    df = pd.read_csv(resolve_lineage_csv(pos, ch))
    return df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)


def terminal_segment(pos, ch):
    """Valid trace up to the lysis collapse, t_rel = time - t_death. Returns
    (seg_df, t_death, fallback_bool) or None."""
    k = f"{pos}_{ch}"
    f1 = DEATH_WINDOW[k][1]
    m = load_mother(pos, ch)
    v = m[~(m["is_outlier"] | m["touches_border"]) & (m["mass_pg"] >= 10.0)].copy()
    if v.empty:
        return None
    pre = v[v["frame"] <= f1]
    if pre.empty:
        return None
    tf1 = float(pre["time_h"].max())
    runmax = float(pre["mass_pg"].max())
    fwd = v[(v["time_h"] > tf1) & (v["time_h"] <= tf1 + FWD_H)].sort_values("time_h")
    t_death, fallback = None, False
    for _, row in fwd.iterrows():
        runmax = max(runmax, float(row["mass_pg"]))
        if float(row["mass_pg"]) < COLLAPSE * runmax:
            t_death = float(row["time_h"])
            break
    if t_death is None:                       # no collapse seen -> anchor at window end
        t_death = float(fwd["time_h"].max()) if len(fwd) else tf1
        fallback = True
    seg = v[v["time_h"] < t_death].copy()
    if len(seg) < 5:
        return None
    seg["t_rel"] = seg["time_h"] - t_death
    seg["minmaj"] = seg["short_axis_um"] / seg["long_axis_um"]
    return seg, t_death, fallback


def baseline_z(seg, col):
    """Deviation from the earliest-BASE_H-h baseline, in baseline-SD units."""
    t = seg["t_rel"].to_numpy(float)
    y = seg[col].to_numpy(float)
    base = y[t <= t.min() + BASE_H]
    mu = np.mean(base)
    sd = np.std(base, ddof=1) if len(base) > 1 else np.nan
    if not np.isfinite(sd) or sd == 0:
        return None
    return (y - mu) / sd


def interp_grid(t, y):
    g = np.interp(GRID, t, y, left=np.nan, right=np.nan)
    g[(GRID < t.min()) | (GRID > t.max())] = np.nan
    return g


def smooth(y, w=5):
    """nan-aware centred moving average on the grid."""
    out = np.full(len(y), np.nan)
    half = w // 2
    for i in range(len(y)):
        seg = y[max(0, i - half):i + half + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def half_max_onset(y_grid):
    """Time at which a signal first reaches half-way from its baseline (grid in
    [-8,-6] h) to its terminal value (grid in [-1,0] h). Sign-agnostic, so it works
    whether the signal rises or falls. Returns (onset_time, terminal_change)."""
    ys = smooth(y_grid)
    bmask = (GRID >= -8) & (GRID <= -6)
    tmask = (GRID >= -1) & (GRID <= 0)
    if not np.isfinite(ys[bmask]).any() or not np.isfinite(ys[tmask]).any():
        return np.nan, np.nan
    base = np.nanmean(ys[bmask])
    term = np.nanmean(ys[tmask])
    d = term - base
    if abs(d) < 1e-9:
        return np.nan, d
    target = base + 0.5 * d
    for tt, val in zip(GRID, ys):
        if np.isfinite(val) and ((d > 0 and val >= target) or
                                 (d < 0 and val <= target)):
            return float(tt), d
    return np.nan, d


def main():
    dead = [pc for pc in list_phase1_dead() if pc not in MANUAL_EXCLUDE]
    swelling = [pc for pc in dead if f"{pc[0]}_{pc[1]}" not in ELONGATION]

    ri_z, sh_z = [], []        # per-lineage z on the common grid
    ri_raw, sh_raw = [], []    # per-lineage raw on the common grid
    chs, fbs = [], []
    for pos, ch in swelling:
        res = terminal_segment(pos, ch)
        if res is None:
            continue
        seg, t_death, fb = res
        zri = baseline_z(seg, "mean_ri")
        zsh = baseline_z(seg, "minmaj")
        if zri is None or zsh is None:
            continue
        t = seg["t_rel"].to_numpy(float)
        ri_z.append(interp_grid(t, zri)); sh_z.append(interp_grid(t, zsh))
        ri_raw.append(interp_grid(t, seg["mean_ri"].to_numpy(float)))
        sh_raw.append(interp_grid(t, seg["minmaj"].to_numpy(float)))
        chs.append(f"{pos}_{ch}"); fbs.append(fb)

    ri_z = np.array(ri_z); sh_z = np.array(sh_z)
    ri_raw = np.array(ri_raw); sh_raw = np.array(sh_raw)
    n = len(chs)
    print(f"swelling lineages used: {n}  (fallback-anchored: {sum(fbs)})")

    # per-lineage half-max onset (sign-agnostic) on the raw gridded signals
    ri_on, sh_on, d_ri, d_sh = [], [], [], []
    for i in range(n):
        o_ri, dri = half_max_onset(ri_raw[i])
        o_sh, dsh = half_max_onset(sh_raw[i])
        ri_on.append(o_ri); sh_on.append(o_sh); d_ri.append(dri); d_sh.append(dsh)
    ri_on = np.array(ri_on); sh_on = np.array(sh_on)

    def msem(a):
        mu = np.nanmean(a, axis=0)
        cnt = np.sum(np.isfinite(a), axis=0)
        sd = np.nanstd(a, axis=0, ddof=1)
        sem = np.where(cnt > 1, sd / np.sqrt(np.maximum(cnt, 1)), np.nan)
        return mu, sem, cnt

    ri_mu, ri_se, cnt = msem(ri_z)
    sh_mu, sh_se, _ = msem(sh_z)
    ri_rmu, _, _ = msem(ri_raw)
    sh_rmu, _, _ = msem(sh_raw)

    # paired half-max-onset comparison (only where both onsets are defined)
    both = np.isfinite(ri_on) & np.isfinite(sh_on)
    diff = ri_on[both] - sh_on[both]      # >0: RI onset later => shape change leads
    med_ri = np.nanmedian(ri_on); med_sh = np.nanmedian(sh_on)
    if both.sum() >= 3 and np.any(diff != 0):
        try:
            _, p = wilcoxon(ri_on[both], sh_on[both])
        except ValueError:
            p = np.nan
    else:
        p = np.nan
    mdiff = np.nanmedian(diff)
    lead = ("roundness (shape) change leads" if mdiff > 0 else
            "RI change leads" if mdiff < 0 else "simultaneous")
    print(f"half-max onset: RI median={med_ri:.2f}h  roundness median={med_sh:.2f}h"
          f"  median(RI-shape)={mdiff:+.2f}h  Wilcoxon p={p:.3g}  -> {lead}  "
          f"(n_paired={both.sum()})")
    print(f"terminal change direction: RI median Δ={np.nanmedian(d_ri):+.4f}  "
          f"roundness median Δ={np.nanmedian(d_sh):+.4f}")

    # -------- figure --------
    fig, axes = plt.subplots(2, 1, figsize=(120 / 25.4, 120 / 25.4),
                             sharex=True, constrained_layout=True)
    ax = axes[0]
    sel = cnt >= 3
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ax.fill_between(GRID[sel], (ri_mu - ri_se)[sel], (ri_mu + ri_se)[sel],
                    color=RI_C, alpha=0.18, lw=0)
    ax.plot(GRID[sel], ri_mu[sel], color=RI_C, lw=1.6, label="mean RI (z)")
    ax.fill_between(GRID[sel], (sh_mu - sh_se)[sel], (sh_mu + sh_se)[sel],
                    color=SHAPE_C, alpha=0.18, lw=0)
    ax.plot(GRID[sel], sh_mu[sel], color=SHAPE_C, lw=1.6,
            label="roundness minor/major (z)")
    ax.set_ylabel("deviation from baseline [SD]")
    ax.set_title(f"death-aligned: which departs first?  ({lead}, "
                 f"$\\Delta$median={np.nanmedian(diff):+.2f} h, p={p:.2g}, n={n})",
                 fontsize=7.5)
    ax.legend(loc="lower left", frameon=False)

    ax = axes[1]
    ax.axvline(0, color="#888", lw=0.8, ls="--")
    ln1 = ax.plot(GRID[sel], ri_rmu[sel], color=RI_C, lw=1.6, label="mean RI")
    ax.set_ylabel("mean RI", color=RI_C)
    ax.tick_params(axis="y", labelcolor=RI_C)
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ln2 = ax2.plot(GRID[sel], sh_rmu[sel], color=SHAPE_C, lw=1.6,
                   label="minor/major")
    ax2.set_ylabel("minor/major", color=SHAPE_C)
    ax2.tick_params(axis="y", labelcolor=SHAPE_C)
    ax.set_xlabel("time from death (lysis) [h]")
    ax.legend(ln1 + ln2, [l.get_label() for l in ln1 + ln2],
              loc="lower left", frameon=False)
    fig.suptitle("Swelling death: mean RI vs cell rounding, aligned at lysis "
                 f"(EFD, n={n})", fontsize=8)

    caption = (
        "Aligned at the moment of death, swelling-death lineages show their "
        "cycle-mean refractive index and their cell roundness both change in the "
        "terminal hours; the panel-A onset comparison reports which one departs from "
        "baseline first. x = time from death (lysis) [h], death at 0, pre-death "
        "negative. Operational definitions: death (lysis) = the first frame after the "
        "last completed cell cycle whose dry mass falls below "
        f"{COLLAPSE:g}x the running maximum (the swollen mother lyses and a small new "
        "cell repopulates the trench); each lineage's terminal segment runs up to that "
        "frame. mean RI = EFD mean_ri of the rank-1 mother per frame; roundness = "
        "minor/major = short_axis_um / long_axis_um (rises from ~0.3 for a rod toward "
        "1 as the cell balloons). Panel A: each signal is expressed as deviation from "
        f"its own baseline (earliest {BASE_H:g} h of the terminal segment) in baseline "
        "standard-deviation units (z), so RI and roundness (which move in opposite "
        "directions) share one axis; lines = population mean across lineages, bands = "
        "+/- SEM, shown where >= 3 lineages contribute. Panel B: raw population-mean "
        "mean RI (left axis) and minor/major (right axis). Lead test: per lineage and "
        "signal we take the half-max onset = the time at which the smoothed signal "
        "first reaches half-way from its baseline (grid -8..-6 h) to its terminal "
        "value (grid -1..0 h); sign-agnostic (works whether a signal rises or falls) "
        "and scale-free. We compare the paired per-lineage onset times (RI vs "
        "roundness) with a Wilcoxon signed-rank test; median(RI - roundness) > 0 means "
        "roundness reaches its half-max earlier, i.e. shape change leads. "
        "swelling death = phase1-dead lineages excluding the two elongation cascades. "
        f"n = {n} swelling lineages (lineages with too short a baseline or no usable "
        "terminal segment are dropped; fallback-anchored when no mass collapse was "
        "found within "
        f"{FWD_H:g} h of the last division). Statistics: Wilcoxon signed-rank on paired "
        "per-lineage onset times; error bands are SEM across lineages. Conditions: "
        "Schizosaccharomyces pombe, 260517 Mother-Machine dataset, EMM + 2% glucose, "
        "phase1 (frame <= 2018), QPI EFD volume variant [strain/genotype and "
        "temperature: confirm from acquisition metadata]. Abbreviations: RI, "
        "refractive index; SEM, standard error of the mean; SD, standard deviation; "
        "EFD, elliptic Fourier descriptor. Data availability: per-lineage gridded "
        "z and raw traces, and per-lineage onset times, in this run's *_data.npz."
    )
    save_figure(
        fig,
        params={"volume_variant": "efd", "n_swelling": n,
                "collapse_frac": COLLAPSE, "fwd_search_h": FWD_H,
                "baseline_h": BASE_H, "onset_method": "half-max (sign-agnostic)",
                "onset_ri_median_h": float(med_ri),
                "onset_shape_median_h": float(med_sh),
                "onset_diff_median_h": float(np.nanmedian(diff)),
                "wilcoxon_p": float(p) if np.isfinite(p) else None,
                "lead": lead, "n_fallback": int(sum(fbs))},
        description="death(lysis)-aligned mean RI vs roundness(minor/major) for "
                    "swelling-death lineages: z-deviation mean±SEM + raw twin-axis, "
                    "paired onset lead/lag test (EFD)",
        caption=caption,
        data={"grid_t": GRID, "ri_z_mean": ri_mu, "ri_z_sem": ri_se,
              "shape_z_mean": sh_mu, "shape_z_sem": sh_se, "n_contrib": cnt,
              "ri_raw_mean": ri_rmu, "shape_raw_mean": sh_rmu,
              "ch": np.array(chs), "ri_onset_h": ri_on, "shape_onset_h": sh_on,
              "fallback": np.array(fbs)},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
