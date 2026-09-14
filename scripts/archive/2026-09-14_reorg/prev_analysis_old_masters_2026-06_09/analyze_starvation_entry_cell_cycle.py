"""Where in the cell cycle is each mother at phase1 end (frame 2018)?

For each mother, compute:
  elapsed_h = (TARGET_FRAME - last_div_before_target) / FRAMES_PER_HOUR
  rel       = elapsed_h / COHORT_CYCLE_H

where COHORT_CYCLE_H is the median phase1 division interval of the
gold-standard cohort (~3.25 h, fixed for all mothers).

If rel > 1, the mother has been past her "expected" cycle length without
dividing again (extended/aborted cycle). We plot rel anyway — the long
tail is biologically meaningful (mothers that already stopped dividing).

Y-axis is mean RI at the target frame. Background shows gold-standard
cycle-aligned mean RI ± SD (0 ≤ rel ≤ 1 region) so each scatter point
can be read against the typical density at that cycle phase.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    find_lineage_csv, select_gold_standard,
)
from overlay_mean_sd_band_full_timecourse import (  # noqa: E402
    list_revived_mothers, list_never_revived_mothers,
)
from figure_logger import save_figure  # noqa: E402


def rank1_division_frames(df: pd.DataFrame,
                          clist: pd.DataFrame | None = None) -> list[int]:
    """Mother-position divisions, covering both chamber types:
      (A) tracker promotes a new cell_id when rank=1 cell divides
          → cell_id of rank=1 switches at the division frame
      (B) tracker keeps the same cell_id at rank=1 (mother stays anchored)
          → division is recorded only as a daughter birth in clist

    We union both sources of division frames across all cell_ids that
    ever appeared at rank=1.
    """
    divs: set[int] = set()
    m = df[df["rank"] == 1].sort_values("frame")
    if m.empty:
        return []
    cells = m["cell_id"].to_numpy()
    frames = m["frame"].to_numpy()
    rank1_ids = set(int(c) for c in np.unique(cells))
    # (A) rank=1 cell_id switches
    for i in range(1, len(cells)):
        if cells[i] != cells[i - 1]:
            divs.add(int(frames[i]))
    # (B) daughter births of any cell that ever held rank=1
    if clist is not None:
        born = clist[clist["mother_id"].isin(list(rank1_ids))]
        for bf in born["birth_frame"].astype(int):
            divs.add(int(bf))
    return sorted(divs)

TIME_INTERVAL_MIN = 5.0
FRAMES_PER_HOUR = 60.0 / TIME_INTERVAL_MIN
TARGET_FRAME = 2018  # last phase1 frame
COHORT_CYCLE_H = 3.25  # gold-standard median phase1 division interval
N_REL = 100
LOW_MASS_PG = 10.0
RI_WINDOW_FRAMES = 6  # ±30 min for stable RI estimate


def measure_entry(pos: str, ch: str):
    path = find_lineage_csv(pos, ch)
    if path is None:
        return None
    df = pd.read_csv(path)
    try:
        try:
            clist = pd.read_csv(path.parent / "clist.csv")
        except Exception:
            clist = None
        divs = rank1_division_frames(df, clist)
    except Exception:
        return None
    prior = [d for d in divs if d <= TARGET_FRAME]
    if not prior:
        return None
    last_div = max(prior)
    elapsed_h = (TARGET_FRAME - last_div) / FRAMES_PER_HOUR
    rel = elapsed_h / COHORT_CYCLE_H

    m = df[df["rank"] == 1].sort_values("frame").copy()
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (m["mass_pg"] < LOW_MASS_PG))
    m_valid = m[~bad]
    near = m_valid[(m_valid["frame"] >= TARGET_FRAME - RI_WINDOW_FRAMES)
                   & (m_valid["frame"] <= TARGET_FRAME + RI_WINDOW_FRAMES)]
    if near.empty:
        return None
    ri = float(near["mean_ri"].mean())
    return {"pos": pos, "ch": ch, "rel": rel, "elapsed_h": elapsed_h,
            "ri": ri, "last_div_frame": last_div}


def gold_cycle_aligned_ri(gold_channels):
    rel_grid = np.linspace(0, 1, N_REL)
    rows = []
    for pos, ch in gold_channels:
        path = find_lineage_csv(pos, ch)
        if path is None:
            continue
        df = pd.read_csv(path)
        try:
            clist = pd.read_csv(path.parent / "clist.csv")
            divs = mother_division_frames(clist, lineage=df,
                                          exclude_bad_frames=True)
        except Exception:
            continue
        m = df[df["rank"] == 1].sort_values("frame")
        for i in range(len(divs) - 1):
            f0, f1 = divs[i], divs[i + 1]
            if f1 > TARGET_FRAME + 1:
                continue
            sub = m[(m["frame"] >= f0) & (m["frame"] < f1)]
            sub = sub[~(sub["is_outlier"].astype(bool)
                        | sub["touches_border"].astype(bool))]
            sub = sub[sub["mass_pg"] >= LOW_MASS_PG]
            if len(sub) < 4:
                continue
            t = sub["frame"].to_numpy().astype(float)
            t_rel = (t - f0) / (f1 - f0)
            rows.append(np.interp(rel_grid, t_rel, sub["mean_ri"].to_numpy()))
    if not rows:
        return rel_grid, np.full_like(rel_grid, np.nan), np.full_like(rel_grid, np.nan)
    M = np.vstack(rows)
    return rel_grid, np.nanmean(M, axis=0), np.nanstd(M, axis=0)


def collect(channels):
    out = []
    for pos, ch in channels:
        rec = measure_entry(pos, ch)
        if rec is not None:
            out.append(rec)
    return out


MEDIA_SWITCH_FRAMES = [
    (2018, "2 % → 0.0055 %  (frame 2018)"),
    (2306, "0.0055 % → 0 %  (frame 2306)"),
    (2884, "0 % → 2 % recovery  (frame 2884)"),
]

MEDIUM_BOUNDARIES = [0, 2019, 2307, 2885, 3748]  # half-open [start, end)
SMOOTH_WINDOW = 5  # frames, per-medium rolling mean


def _medium_id(frame: int) -> int:
    for i in range(len(MEDIUM_BOUNDARIES) - 1):
        if MEDIUM_BOUNDARIES[i] <= frame < MEDIUM_BOUNDARIES[i + 1]:
            return i
    return -1


def ri_at_frame(pos: str, ch: str, frame: int) -> float | None:
    path = find_lineage_csv(pos, ch)
    if path is None:
        return None
    df = pd.read_csv(path)
    m = df[df["rank"] == 1].sort_values("frame").copy()
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (m["mass_pg"] < LOW_MASS_PG))
    m_valid = m[~bad]
    near = m_valid[(m_valid["frame"] >= frame - RI_WINDOW_FRAMES)
                   & (m_valid["frame"] <= frame + RI_WINDOW_FRAMES)]
    if near.empty:
        return None
    return float(near["mean_ri"].mean())


def plot_ri_at_media_switches(revived_ch, never_revived_ch):
    fig, axes = plt.subplots(len(MEDIA_SWITCH_FRAMES), 1,
                             figsize=(120 / 25.4, 130 / 25.4),
                             sharex=True, sharey=False,
                             constrained_layout=True)
    bins = np.linspace(1.345, 1.385, 29)  # bin width ≈ 0.00143

    for row, (frame, label) in enumerate(MEDIA_SWITCH_FRAMES):
        r_vals = [ri_at_frame(p, c, frame) for p, c in revived_ch]
        r_vals = np.array([v for v in r_vals if v is not None])
        nr_vals = [ri_at_frame(p, c, frame) for p, c in never_revived_ch]
        nr_vals = np.array([v for v in nr_vals if v is not None])

        try:
            from scipy.stats import ks_2samp
            p_ks = float(ks_2samp(r_vals, nr_vals).pvalue)
        except Exception:
            p_ks = float("nan")

        ax = axes[row]
        ax.hist(r_vals, bins=bins, color="#1a1a1a", alpha=0.55,
                edgecolor="white", linewidth=0.4,
                label=f"revived (n={len(r_vals)}, mean={np.mean(r_vals):.4f})")
        ax.hist(nr_vals, bins=bins, color="#d62728", alpha=0.55,
                edgecolor="white", linewidth=0.4,
                label=f"never_revived (n={len(nr_vals)}, mean={np.mean(nr_vals):.4f})")
        if len(r_vals):
            ax.axvline(np.mean(r_vals), color="#1a1a1a", linestyle="-",
                       linewidth=0.8, alpha=0.9)
        if len(nr_vals):
            ax.axvline(np.mean(nr_vals), color="#8c0000", linestyle="-",
                       linewidth=0.8, alpha=0.9)
        ax.set_title(
            f"mean RI at frame {frame}   {label}   "
            f"KS p = {p_ks:.3g}",
            fontsize=7,
        )
        ax.set_ylabel("count", fontsize=8)
        ax.legend(loc="upper right", frameon=False, fontsize=6)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("mean RI", fontsize=8)
    fig.suptitle("Mean RI at medium-switch frames", fontsize=9)

    save_figure(
        fig,
        params={
            "media_switch_frames": [f for f, _ in MEDIA_SWITCH_FRAMES],
            "ri_window_frames": RI_WINDOW_FRAMES,
            "low_mass_pg": LOW_MASS_PG,
        },
        description="mean RI distributions just before each medium switch",
    )
    plt.close(fig)


def phase1_mean_ri(pos: str, ch: str) -> float | None:
    """Mean RI over phase1 (frame 0–2018) for the rank=1 mother."""
    path = find_lineage_csv(pos, ch)
    if path is None:
        return None
    df = pd.read_csv(path)
    m = df[df["rank"] == 1].sort_values("frame").copy()
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (m["mass_pg"] < LOW_MASS_PG))
    m_valid = m[~bad]
    p1 = m_valid[m_valid["frame"] <= TARGET_FRAME]
    if p1.empty:
        return None
    val = float(p1["mean_ri"].mean())
    return val if np.isfinite(val) else None


def plot_phase1_mean_ri(revived_ch, never_revived_ch):
    r_vals = np.array([v for v in (phase1_mean_ri(p, c) for p, c in revived_ch)
                       if v is not None])
    nr_vals = np.array([v for v in (phase1_mean_ri(p, c) for p, c in never_revived_ch)
                        if v is not None])
    try:
        from scipy.stats import ks_2samp, mannwhitneyu
        p_ks = float(ks_2samp(r_vals, nr_vals).pvalue)
        p_mw = float(mannwhitneyu(r_vals, nr_vals, alternative="two-sided").pvalue)
    except Exception:
        p_ks = p_mw = float("nan")

    bins = np.linspace(1.350, 1.380, 29)
    fig, ax = plt.subplots(figsize=(120 / 25.4, 65 / 25.4),
                           constrained_layout=True)
    ax.hist(r_vals, bins=bins, color="#1a1a1a", alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label=f"revived (n={len(r_vals)}, mean={np.mean(r_vals):.4f})")
    ax.hist(nr_vals, bins=bins, color="#d62728", alpha=0.55,
            edgecolor="white", linewidth=0.4,
            label=f"never_revived (n={len(nr_vals)}, mean={np.mean(nr_vals):.4f})")
    ax.axvline(np.mean(r_vals), color="#1a1a1a", linestyle="-",
               linewidth=0.8, alpha=0.9)
    ax.axvline(np.mean(nr_vals), color="#8c0000", linestyle="-",
               linewidth=0.8, alpha=0.9)
    ax.set_xlabel("phase1 mean RI (per mother, frame 0–2018)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.set_title(
        f"phase1 mean RI distribution   "
        f"KS p = {p_ks:.3g},  MW p = {p_mw:.3g}",
        fontsize=8,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.tick_params(labelsize=7)

    save_figure(
        fig,
        params={
            "phase1_end_frame": TARGET_FRAME,
            "n_revived": len(r_vals),
            "n_never_revived": len(nr_vals),
            "mean_revived": float(np.mean(r_vals)),
            "mean_never_revived": float(np.mean(nr_vals)),
            "sd_revived": float(np.std(r_vals)),
            "sd_never_revived": float(np.std(nr_vals)),
            "ks_pvalue": p_ks,
            "mw_pvalue": p_mw,
        },
        description="phase1 per-mother mean RI distribution revived vs never_revived",
    )
    plt.close(fig)


def plot_elapsed_histograms(rev, nr):
    r_x = np.array([r["elapsed_h"] for r in rev])
    nr_x = np.array([r["elapsed_h"] for r in nr])
    xmax_h = max(COHORT_CYCLE_H + 1.0,
                 float(np.percentile(np.concatenate([r_x, nr_x]), 95)) + 0.5)
    bins = np.linspace(0, xmax_h, 25)

    try:
        from scipy.stats import ks_2samp
        p_ks = float(ks_2samp(r_x, nr_x).pvalue)
    except Exception:
        p_ks = float("nan")

    fig, axes = plt.subplots(2, 1, figsize=(120 / 25.4, 90 / 25.4),
                             sharex=True, sharey=False,
                             constrained_layout=True)
    for ax, x, color, label in [
        (axes[0], r_x, "#4a4a4a",
         f"revived (n={len(r_x)})"),
        (axes[1], nr_x, "#d62728",
         f"never_revived (n={len(nr_x)})"),
    ]:
        ax.hist(x, bins=bins, color=color, edgecolor="white",
                linewidth=0.4, alpha=0.85)
        ax.axvline(np.median(x), color="k", linestyle="-", linewidth=0.8,
                   alpha=0.7, label=f"median = {np.median(x):.2f} h")
        ax.axvline(COHORT_CYCLE_H, color="k", linestyle="--", linewidth=0.5,
                   alpha=0.5, label=f"cohort cycle = {COHORT_CYCLE_H:.2f} h")
        ax.set_ylabel(f"count\n{label}", fontsize=8)
        ax.legend(loc="upper right", frameon=False, fontsize=6)
        ax.tick_params(labelsize=7)

    axes[1].set_xlabel("elapsed time since last division at frame 2018 [h]",
                       fontsize=8)
    axes[0].set_title(
        f"Distribution of elapsed time since last division  (KS p = {p_ks:.3g})",
        fontsize=8,
    )

    save_figure(
        fig,
        params={
            "target_frame": TARGET_FRAME,
            "cohort_cycle_h": COHORT_CYCLE_H,
            "n_revived": len(r_x),
            "n_never_revived": len(nr_x),
            "median_elapsed_h_revived": float(np.median(r_x)),
            "median_elapsed_h_never_revived": float(np.median(nr_x)),
            "ks_pvalue": p_ks,
        },
        description="elapsed time histograms revived vs never_revived",
    )
    plt.close(fig)


def plot_results(rel_grid, ref_mean, ref_sd, rev, nr, xmax_h: float):
    fig, ax = plt.subplots(figsize=(120 / 25.4, 75 / 25.4),
                           constrained_layout=True)
    # x for gold reference: rel * cohort cycle so we are on time axis [h]
    ref_x = rel_grid * COHORT_CYCLE_H
    ax.fill_between(ref_x, ref_mean - ref_sd, ref_mean + ref_sd,
                    color="#888888", alpha=0.25, linewidth=0,
                    label=f"gold-standard ± SD (0–{COHORT_CYCLE_H:.2f} h)")
    ax.plot(ref_x, ref_mean, color="#1a1a1a", linewidth=1.2,
            label="gold-standard mean", zorder=3)
    ax.axvline(COHORT_CYCLE_H, color="k", linestyle="--", linewidth=0.5,
               alpha=0.5, label=f"cohort cycle = {COHORT_CYCLE_H:.2f} h")

    r_x = np.array([r["elapsed_h"] for r in rev])
    r_ri = np.array([r["ri"] for r in rev])
    nr_x = np.array([r["elapsed_h"] for r in nr])
    nr_ri = np.array([r["ri"] for r in nr])
    ax.scatter(r_x, r_ri, s=22, color="#4a4a4a", alpha=0.75,
               edgecolor="white", linewidth=0.4,
               label=f"revived (n={len(rev)})", zorder=5)
    ax.scatter(nr_x, nr_ri, s=22, color="#d62728", alpha=0.75,
               edgecolor="white", linewidth=0.4,
               label=f"never_revived (n={len(nr)})", zorder=6)

    ax.set_xlim(-0.2, xmax_h)
    ax.set_ylim(1.345, 1.385)
    ax.set_xlabel("elapsed time since last division at frame 2018 [h]")
    ax.set_ylabel("mean RI at frame 2018")
    ax.set_title(
        "Density at phase1 end vs time since last division  "
        f"(cohort cycle = {COHORT_CYCLE_H:.2f} h reference)",
        fontsize=8,
    )
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    save_figure(
        fig,
        params={
            "target_frame": TARGET_FRAME,
            "cohort_cycle_h": COHORT_CYCLE_H,
            "n_revived": len(rev),
            "n_never_revived": len(nr),
            "median_elapsed_h_revived": float(np.median(r_x)),
            "median_elapsed_h_never_revived": float(np.median(nr_x)),
            "median_ri_revived": float(np.median(r_ri)),
            "median_ri_never_revived": float(np.median(nr_ri)),
            "frac_within_cohort_cycle_revived": float((r_x <= COHORT_CYCLE_H).mean()),
            "frac_within_cohort_cycle_never_revived": float((nr_x <= COHORT_CYCLE_H).mean()),
            "ri_window_frames": RI_WINDOW_FRAMES,
            "low_mass_pg": LOW_MASS_PG,
        },
        description="elapsed time since last division at frame 2018 vs RI",
    )
    plt.close(fig)


def main():
    revived = list_revived_mothers()
    never_revived = list_never_revived_mothers()
    gold = select_gold_standard()
    print(f"revived={len(revived)}, never_revived={len(never_revived)}, "
          f"gold={len(gold)}")

    rel_grid, ref_mean, ref_sd = gold_cycle_aligned_ri(gold)

    rev = collect(revived)
    nr = collect(never_revived)
    print(f"revived analyzed:       {len(rev)}")
    print(f"never_revived analyzed: {len(nr)}")

    if not rev or not nr:
        return
    r_rel = np.array([r["rel"] for r in rev])
    r_ri = np.array([r["ri"] for r in rev])
    nr_rel = np.array([r["rel"] for r in nr])
    nr_ri = np.array([r["ri"] for r in nr])
    print()
    print(f"revived       rel: median={np.median(r_rel):.3f}, "
          f"mean={np.mean(r_rel):.3f}, max={np.max(r_rel):.2f}")
    print(f"never_revived rel: median={np.median(nr_rel):.3f}, "
          f"mean={np.mean(nr_rel):.3f}, max={np.max(nr_rel):.2f}")
    print(f"revived       ri:  median={np.median(r_ri):.4f}")
    print(f"never_revived ri:  median={np.median(nr_ri):.4f}")
    print(f"frac rel ≤ 1, revived:       {(r_rel<=1).mean()*100:.1f}%")
    print(f"frac rel ≤ 1, never_revived: {(nr_rel<=1).mean()*100:.1f}%")
    try:
        from scipy.stats import ks_2samp
        print(f"KS p (rel): {float(ks_2samp(r_rel, nr_rel).pvalue):.4f}")
        print(f"KS p (ri):  {float(ks_2samp(r_ri,  nr_ri).pvalue):.4f}")
    except Exception:
        pass

    # x-axis: extend to capture 95th percentile of pooled elapsed time
    pooled_h = np.concatenate([
        np.array([r["elapsed_h"] for r in rev]),
        np.array([r["elapsed_h"] for r in nr]),
    ])
    xmax_h = float(np.percentile(pooled_h, 95)) + 1.0
    xmax_h = max(xmax_h, COHORT_CYCLE_H + 1.0)
    plot_results(rel_grid, ref_mean, ref_sd, rev, nr, xmax_h=xmax_h)
    plot_elapsed_histograms(rev, nr)
    plot_ri_at_media_switches(revived, never_revived)
    plot_phase1_mean_ri(revived, never_revived)


if __name__ == "__main__":
    main()
