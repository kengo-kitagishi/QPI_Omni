"""Per-channel grid of phase2-dead mother trajectories.

For each of the 49 channels where mother survived phase1 but didn't revive
in phase2, plot a small individual panel showing the full mother trace
(time 0..end of timecourse) for one metric. All 49 panels share a common
y-axis range so they can be visually scanned and compared.

A faint gold-standard band can be overlaid in each panel as a sanity
reference. Three figures are produced — one per metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    _pick_metric, select_gold_standard, MANUAL_EXCLUDE, find_lineage_csv,
)
from check_phase2_dead_tracking_coverage import (  # noqa: E402
    mother_didnt_revive_in_phase2,
)
from figure_logger import save_figure  # noqa: E402
from qpi_paths import yaml_path  # noqa: E402

import yaml

YAML_PATH = yaml_path()
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


def load_mother_full(pos: str, ch: str):
    """Mother trace over the entire timecourse; outliers/border/low-mass → NaN."""
    lineage_path = find_lineage_csv(pos, ch)
    if lineage_path is None:
        return None
    df = pd.read_csv(lineage_path)
    m = df[df["rank"] == 1].sort_values("frame").copy()
    if m.empty:
        return None
    bad = m["is_outlier"].astype(bool) | m["touches_border"].astype(bool)
    vol = m["volume_um3_rod"].to_numpy(dtype=float)
    ri = m["mean_ri"].to_numpy(dtype=float)
    mass = m["mass_pg"].to_numpy(dtype=float)
    low_mass = mass < 10.0
    bad = bad | low_mass
    vol[bad] = np.nan
    ri[bad] = np.nan
    mass[bad] = np.nan
    return m["time_h"].to_numpy(), vol, ri, mass


def list_phase2_dead_channels() -> list[tuple[str, str]]:
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    out = []
    for pos_name, channels in (data.get("positions") or {}).items():
        if not channels:
            continue
        for ch_name, fields in channels.items():
            if not fields or fields.get("status") != "cells":
                continue
            if (pos_name, ch_name) in MANUAL_EXCLUDE:
                continue
            if mother_didnt_revive_in_phase2(fields):
                out.append((pos_name, ch_name))
    return out


def gold_envelope(gold: list[tuple[str, str]], metric: str,
                  t_grid_raw: np.ndarray):
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


def plot_grid(channels: list[tuple[str, str]], metric: str, ylabel: str,
              ylim: tuple[float, float],
              gold_lo: np.ndarray, gold_hi: np.ndarray,
              t_grid_raw: np.ndarray, out_label: str):
    t_grid_shift = t_grid_raw - TIME_ZERO_H
    n = len(channels)
    n_cols = 7
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.0, n_rows * 1.4),
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
            ax.plot(t - TIME_ZERO_H, y, color="#d62728", linewidth=0.5,
                    alpha=0.85, solid_capstyle="butt")
        ax.set_xlim(0, T_END_H + 5)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=5)
        ax.set_title(f"{pos}_{ch}", fontsize=6, pad=1)
    # blank unused
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle(
        f"Mother cell {ylabel} — {n} lineages where mother did not "
        f"revive in starvation/recovery (gold-standard mean ± SD band in gray)",
        fontsize=8, y=0.995,
    )
    for ax in axes[-n_cols:]:
        ax.set_xlabel("time [h]  (original 120h = 0h)", fontsize=6)
    for ax in axes[::n_cols]:
        ax.set_ylabel(ylabel, fontsize=6)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    save_figure(
        fig,
        params={
            "metric": metric,
            "n_channels": n,
            "selection": "phase1=alive AND mother dead in phase2",
            "phase1_end_h": PHASE1_END_H,
            "recovery_start_h": RECOVERY_H,
        },
        description=f"per-channel grid {metric} {out_label}",
    )
    plt.close(fig)


def main():
    channels = list_phase2_dead_channels()
    print(f"Phase2-dead mother channels: {len(channels)}")
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
            print(f"  no gold envelope for {metric}, skipping")
            continue
        gold_lo, gold_hi = gold_band
        plot_grid(channels, metric, ylabel, ylim, gold_lo, gold_hi,
                  t_grid_raw, f"phase2_dead_individual_{metric}")


if __name__ == "__main__":
    main()
