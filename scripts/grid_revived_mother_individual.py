"""Per-channel grid of revived mother trajectories.

Same layout as grid_phase2_dead_mother_individual.py but for the revived
cohort (phase1=alive AND mother revived in phase2). Useful for visually
sanity-checking each lineage in the population the mean ± SD band was
computed from.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    _pick_metric, select_gold_standard, MANUAL_EXCLUDE,
)
from overlay_mean_sd_band_full_timecourse import (  # noqa: E402
    list_revived_mothers, load_mother_full,
)
from figure_logger import save_figure  # noqa: E402

TIME_INTERVAL_MIN = 5.0
TIME_ZERO_H = 120.0  # original 120 h becomes 0 h on plot axis
T_END_H_RAW = 3748 * TIME_INTERVAL_MIN / 60.0
PHASE1_END_H_RAW = 2019 * TIME_INTERVAL_MIN / 60.0
WO0_START_H_RAW = 2307 * TIME_INTERVAL_MIN / 60.0
RECOVERY_H_RAW = 2885 * TIME_INTERVAL_MIN / 60.0
T_END_H = T_END_H_RAW - TIME_ZERO_H
PHASE1_END_H = PHASE1_END_H_RAW - TIME_ZERO_H
WO0_START_H = WO0_START_H_RAW - TIME_ZERO_H
RECOVERY_H = RECOVERY_H_RAW - TIME_ZERO_H


def gold_envelope(gold, metric: str, t_grid_raw: np.ndarray):
    rows = []
    for pos, ch in gold:
        traj = load_mother_full(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        rows.append(np.interp(t_grid_raw, t, y, left=np.nan, right=np.nan))
    if not rows:
        return None
    M = np.vstack(rows)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    return mean - sd, mean + sd


def plot_grid(channels, metric: str, ylabel: str, ylim,
              gold_lo, gold_hi, t_grid_raw, out_label: str):
    t_grid_shift = t_grid_raw - TIME_ZERO_H
    n = len(channels)
    n_cols = 10
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 1.6, n_rows * 1.2),
                             sharex=True, sharey=True)
    axes = axes.flatten()
    for i, (pos, ch) in enumerate(channels):
        ax = axes[i]
        ax.fill_between(t_grid_shift, gold_lo, gold_hi, color="#888888",
                        alpha=0.25, linewidth=0)
        ax.axvline(PHASE1_END_H, color="k", linestyle=":", linewidth=0.4,
                   alpha=0.5)
        ax.axvline(WO0_START_H, color="k", linestyle=":", linewidth=0.4,
                   alpha=0.5)
        ax.axvline(RECOVERY_H, color="k", linestyle=":", linewidth=0.4,
                   alpha=0.5)
        traj = load_mother_full(pos, ch)
        if traj is not None:
            t, y = _pick_metric(traj, metric)
            ax.plot(t - TIME_ZERO_H, y, color="#1a1a1a", linewidth=0.4,
                    alpha=0.85, solid_capstyle="butt")
        ax.set_xlim(0, T_END_H + 5)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=4)
        ax.set_title(f"{pos}_{ch}", fontsize=5, pad=1)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"Mother cell {ylabel} — {n} lineages where mother revived after "
        f"starvation (gold-standard mean ± SD band in gray)",
        fontsize=8, y=0.995,
    )
    for ax in axes[-n_cols:]:
        ax.set_xlabel("time [h]  (original 120h = 0h)", fontsize=5)
    for ax in axes[::n_cols]:
        ax.set_ylabel(ylabel, fontsize=5)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure(
        fig,
        params={
            "metric": metric,
            "n_channels": n,
            "selection": "phase1=alive AND mother revived in phase2",
        },
        description=f"per-channel grid {metric} {out_label}",
    )
    plt.close(fig)


def main():
    channels = list_revived_mothers()
    print(f"Revived mother channels: {len(channels)}")
    gold = select_gold_standard()
    print(f"Gold-standard channels: {len(gold)}")
    t_grid_raw = np.arange(0, T_END_H_RAW + 0.5, 0.5)
    for metric, ylabel, ylim in [
        ("mean_RI", "mean RI", (1.345, 1.395)),
        ("volume", "volume [μm³]", (0, 300)),
        ("mass", "dry mass [pg]", (0, 80)),
    ]:
        gold_band = gold_envelope(gold, metric, t_grid_raw)
        if gold_band is None:
            continue
        gold_lo, gold_hi = gold_band
        plot_grid(channels, metric, ylabel, ylim, gold_lo, gold_hi,
                  t_grid_raw, f"revived_individual_{metric}")


if __name__ == "__main__":
    main()
