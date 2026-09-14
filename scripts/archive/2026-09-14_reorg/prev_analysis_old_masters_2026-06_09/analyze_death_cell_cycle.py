"""Where in the cell cycle do never_revived mothers die?

For each never_revived mother, we identify:
  1. division frames (from rank=1 trajectory, outlier-aware)
  2. death frame: last frame before the cell's mass drops < 10 pg or vanishes
  3. relative cell cycle position at death:
       rel = (death_frame - last_div_before_death) / cycle_length_h
     where cycle_length_h is the median interval of that mother's
     own phase1 cycles. If the mother has no phase1 cycles, we fall
     back to the cohort median (~3.25 h).

We then build:
  - histogram of relative position at death (>1 means death after one
    full would-be cycle period; >2 means after two)
  - RI vs reference curve: gold-standard mother cycle-aligned mean RI
    is the reference. Each death point is plotted as (rel_position mod 1,
    death_RI) against the reference 0-1 curve.

This shows (a) whether deaths cluster early/late in the cycle and (b)
whether dying mothers have higher or lower cytoplasmic density than
healthy mothers at the same cycle position.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    find_lineage_csv, select_gold_standard, MANUAL_EXCLUDE,
)
from overlay_mean_sd_band_full_timecourse import (  # noqa: E402
    list_never_revived_mothers, load_mother_full,
)
from per_channel_figures import mother_division_frames  # noqa: E402
from figure_logger import save_figure  # noqa: E402

TIME_INTERVAL_MIN = 5.0
FRAMES_PER_HOUR = 60.0 / TIME_INTERVAL_MIN  # 12
PHASE1_END_FRAME = 2019
N_REL_INTERP = 100
COHORT_FALLBACK_CYCLE_H = 3.25
LOW_MASS_PG = 10.0


def detect_death_frame(lineage_df: pd.DataFrame) -> int | None:
    """First frame where mother (rank=1) loses validity:
    mass < LOW_MASS_PG OR row missing (vanished from tracker).

    Returns frame number (int) or None if mother is alive to the end.
    """
    m = lineage_df[lineage_df["rank"] == 1].sort_values("frame").copy()
    if m.empty:
        return None
    # First frame where mass is invalid or below threshold, AFTER at least
    # one valid frame existed
    mass = m["mass_pg"].to_numpy(dtype=float)
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (mass < LOW_MASS_PG) | np.isnan(mass))
    frames = m["frame"].to_numpy()
    # find longest valid prefix end; first bad frame after a valid one
    if bad.all():
        return None
    if not bad.any():
        return None  # alive to the end
    # death = first transition from valid to bad
    valid_seen = False
    for i, b in enumerate(bad):
        if not b:
            valid_seen = True
            continue
        if valid_seen:
            return int(frames[i])
    return None


def cycle_length_h_for_mother(div_frames: list[int]) -> float:
    """Use this mother's phase1 cycles to estimate cycle_length_h."""
    phase1_divs = [d for d in div_frames if d < PHASE1_END_FRAME]
    if len(phase1_divs) >= 2:
        intervals = np.diff(phase1_divs) / FRAMES_PER_HOUR
        return float(np.median(intervals))
    return COHORT_FALLBACK_CYCLE_H


def relative_position_at_death(death_frame: int, div_frames: list[int],
                               cycle_h: float) -> float | None:
    """rel = (death_frame - last_div_before_death) / cycle_length_frames.

    Can exceed 1 if mother died long after the last division (typical for
    starvation deaths).
    """
    cycle_frames = cycle_h * FRAMES_PER_HOUR
    if not div_frames:
        return None
    prior = [d for d in div_frames if d <= death_frame]
    if not prior:
        return None
    last_div = max(prior)
    return float((death_frame - last_div) / cycle_frames)


def reference_cycle_ri_curve(gold_channels) -> tuple[np.ndarray, np.ndarray]:
    """Gold-standard mother cycle-aligned mean RI curve over relative cycle 0-1.

    Uses the mother's own phase1 cycles. Returns (rel, mean_ri).
    """
    rel = np.linspace(0, 1, N_REL_INTERP)
    rows = []
    for pos, ch in gold_channels:
        path = find_lineage_csv(pos, ch)
        if path is None:
            continue
        df = pd.read_csv(path)
        m = df[df["rank"] == 1].sort_values("frame").copy()
        if m.empty:
            continue
        clist_path = path.parent / "clist.csv"
        # use rank=1 to find divisions (cell_id continuity)
        try:
            clist = pd.read_csv(clist_path)
            divs = mother_division_frames(clist, lineage=df,
                                          exclude_bad_frames=True)
        except Exception:
            continue
        divs = [d for d in divs if d < PHASE1_END_FRAME]
        for i in range(len(divs) - 1):
            f0, f1 = divs[i], divs[i + 1] - 1
            if f1 <= f0:
                continue
            sub = m[(m["frame"] >= f0) & (m["frame"] <= f1)]
            sub = sub[~(sub["is_outlier"].astype(bool)
                        | sub["touches_border"].astype(bool))]
            sub = sub[sub["mass_pg"] >= LOW_MASS_PG]
            if len(sub) < 4:
                continue
            t = sub["frame"].to_numpy().astype(float)
            t_rel = (t - f0) / (f1 - f0)
            rows.append(np.interp(rel, t_rel, sub["mean_ri"].to_numpy()))
    if not rows:
        return rel, np.full_like(rel, np.nan, dtype=float)
    M = np.vstack(rows)
    return rel, np.nanmean(M, axis=0)


def death_position_and_ri(pos: str, ch: str
                          ) -> tuple[float, float, float] | None:
    """Return (rel_position, death_RI, cycle_length_h) or None."""
    path = find_lineage_csv(pos, ch)
    if path is None:
        return None
    df = pd.read_csv(path)
    try:
        clist = pd.read_csv(path.parent / "clist.csv")
        divs = mother_division_frames(clist, lineage=df,
                                      exclude_bad_frames=True)
    except Exception:
        return None
    death_f = detect_death_frame(df)
    if death_f is None:
        return None
    cycle_h = cycle_length_h_for_mother(divs)
    rel = relative_position_at_death(death_f, divs, cycle_h)
    if rel is None:
        return None
    # RI at the last valid frame before death
    m = df[df["rank"] == 1].sort_values("frame").copy()
    bad = (m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
           | (m["mass_pg"] < LOW_MASS_PG))
    m_valid = m[~bad]
    pre_death = m_valid[m_valid["frame"] < death_f]
    if pre_death.empty:
        return None
    death_ri = float(pre_death["mean_ri"].iloc[-1])
    return rel, death_ri, cycle_h


def plot_results(records, ref_rel, ref_ri):
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 75 / 25.4),
                             constrained_layout=True)

    # Left: histogram of rel_position
    ax = axes[0]
    rel_vals = np.array([r[0] for r in records])
    bins = np.arange(0, max(rel_vals.max() + 0.25, 1.25), 0.25)
    ax.hist(rel_vals, bins=bins, color="#888888", edgecolor="white",
            linewidth=0.5)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=0.5, alpha=0.6,
               label="1 cycle elapsed")
    ax.set_xlabel("relative cell cycle position at death  "
                  "(0=last division, 1=one full cycle)")
    ax.set_ylabel("count (mothers)")
    ax.set_title(f"Death timing wrt last division (n={len(records)})")
    ax.legend(frameon=False, fontsize=7)

    # Right: death RI vs reference cycle-aligned RI
    ax = axes[1]
    ax.plot(ref_rel, ref_ri, color="#1a1a1a", linewidth=1.5,
            label="gold-standard mean RI", zorder=3)
    # death points: (rel mod 1, death_RI). For rel > 1 we wrap mod 1.
    rel_mod = np.mod(rel_vals, 1.0)
    death_ri_vals = np.array([r[1] for r in records])
    # color by whether rel <= 1 (within first cycle) or > 1 (extended cycle)
    within = rel_vals <= 1.0
    ax.scatter(rel_mod[within], death_ri_vals[within], s=22,
               color="#d62728", alpha=0.75, edgecolor="white", linewidth=0.4,
               label=f"died within 1 cycle (n={int(within.sum())})",
               zorder=4)
    ax.scatter(rel_mod[~within], death_ri_vals[~within], s=22,
               color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.4,
               label=f"died after >1 cycle (n={int((~within).sum())})",
               zorder=4)
    ax.set_xlabel("relative cycle position at death  (mod 1)")
    ax.set_ylabel("mean RI at death")
    ax.set_title("Death-time RI vs healthy reference")
    ax.legend(frameon=False, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(1.345, 1.395)

    save_figure(
        fig,
        params={
            "n_never_revived_analyzed": len(records),
            "rel_position_median": float(np.median(rel_vals)),
            "rel_position_mean": float(np.mean(rel_vals)),
            "fraction_within_one_cycle": float(within.mean()),
            "low_mass_threshold_pg": LOW_MASS_PG,
            "phase1_end_frame": PHASE1_END_FRAME,
            "frames_per_hour": FRAMES_PER_HOUR,
            "cohort_fallback_cycle_h": COHORT_FALLBACK_CYCLE_H,
        },
        description="never_revived death timing vs cell cycle position",
    )
    plt.close(fig)


def main():
    nr = list_never_revived_mothers()
    print(f"never_revived candidates: {len(nr)}")
    gold = select_gold_standard()
    print(f"gold-standard for reference RI: {len(gold)}")

    records = []
    skipped = 0
    for pos, ch in nr:
        rec = death_position_and_ri(pos, ch)
        if rec is None:
            skipped += 1
            continue
        records.append(rec)
    print(f"records collected: {len(records)}, skipped: {skipped}")
    if not records:
        return

    ref_rel, ref_ri = reference_cycle_ri_curve(gold)
    plot_results(records, ref_rel, ref_ri)

    rel = np.array([r[0] for r in records])
    print()
    print(f"rel position at death: median={np.median(rel):.2f}, "
          f"mean={np.mean(rel):.2f}, max={np.max(rel):.2f}")
    print(f"fraction within 1 cycle: {(rel <= 1).mean()*100:.1f}%")
    print(f"fraction within 2 cycles: {(rel <= 2).mean()*100:.1f}%")


if __name__ == "__main__":
    main()
