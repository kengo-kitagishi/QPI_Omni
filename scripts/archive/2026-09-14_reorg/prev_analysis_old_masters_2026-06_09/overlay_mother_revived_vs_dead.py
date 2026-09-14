"""Overlay mother cell trajectories: revived vs never_revived groups.

For each (Pos, ch) in two user-defined groups:
  - find the latest per_channel_figures inbox run
  - load lineage_data3D.csv and identify the mother lineage (rank=1, in_tree=True)
  - extract volume and mean_RI time series for the mother

Plot:
  - revived group: gray, thin, faint (background)
  - never_revived group: colored, thin (foreground)
  - vertical lines at media-schedule frames (2019, 2307, 2885)

Outputs PNG via figure_logger (publication-ready style).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure  # noqa: E402


INBOX_DIRS = [
    Path("/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-06-08/per_channel_figures"),
    Path("/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-06-09/per_channel_figures"),
]

# user-supplied groups
GROUP_DEAD = [
    ("Pos1", "ch10"),
    ("Pos3", "ch01"),
    ("Pos3", "ch09"),
    ("Pos6", "ch03"),
    ("Pos15", "ch02"),
    ("Pos15", "ch06"),
    ("Pos17", "ch06"),
    ("Pos22", "ch02"),
    ("Pos23", "ch02"),
    ("Pos25", "ch01"),
    ("Pos26", "ch01"),
    ("Pos26", "ch10"),
    ("Pos27", "ch06"),
    ("Pos28", "ch09"),
]
GROUP_REVIVED = [
    ("Pos1", "ch05"),
    ("Pos3", "ch06"),
    ("Pos4", "ch02"),
    ("Pos4", "ch10"),
    ("Pos6", "ch04"),
    ("Pos7", "ch00"),
    ("Pos8", "ch08"),
    ("Pos9", "ch06"),
    ("Pos9", "ch10"),
    ("Pos10", "ch00"),
    ("Pos11", "ch07"),
    ("Pos12", "ch07"),
    ("Pos12", "ch09"),
    ("Pos14", "ch01"),
    ("Pos14", "ch04"),
    ("Pos14", "ch05"),
    ("Pos16", "ch02"),
    ("Pos19", "ch08"),
    ("Pos20", "ch05"),
    ("Pos21", "ch05"),
    ("Pos22", "ch07"),
    ("Pos22", "ch09"),
    ("Pos23", "ch04"),
    ("Pos24", "ch05"),
    ("Pos25", "ch07"),
    ("Pos26", "ch02"),
    ("Pos27", "ch04"),
    ("Pos29", "ch00"),
    ("Pos29", "ch04"),
    ("Pos30", "ch08"),
]

# 縦点線は媒体交換「前」の最終フレームの位置（media_schedule の表記より 1 フレーム前）
MEDIA_FRAMES = {
    "wo_2 → wo_0.0055%": 2018,  # 2019 で 0.0055% 開始 → 前フレーム=2018
    "wo_0.0055% → wo_0%": 2306,  # 2307 で 0% 開始 → 前フレーム=2306
    "wo_0% → wo_2%": 2884,        # 2885 で 2% 戻し → 前フレーム=2884
}
TIME_INTERVAL_MIN = 5.0  # min/frame
TIME_ZERO_H = 120.0  # subtract this from time_h so plot's 0h = original 120h


def find_latest_run_for_channel(pos: str, ch: str):
    """Return (run_dir, n_mother_divisions) for the latest run on (pos, ch)."""
    best = None
    best_time = ""
    for inbox in INBOX_DIRS:
        if not inbox.exists():
            continue
        for run_dir in inbox.iterdir():
            if not run_dir.is_dir():
                continue
            # check f001 json for channel label match
            for json_path in run_dir.glob("*f001.json"):
                try:
                    meta = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                params = meta.get("params", {})
                label = params.get("channel_label", "")
                if label != f"{pos}_{ch}":
                    continue
                created = meta.get("created_at_utc", "")
                if created > best_time:
                    best_time = created
                    best = (run_dir, params.get("n_mother_divisions", 0))
                break
    return best


def load_mother_trajectory(run_dir: Path):
    """Read lineage_data3D.csv from run_dir, return (time_h, volume, mean_ri) for mother (rank=1)."""
    csv_path = run_dir / "lineage_data3D.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    # mother = cell at rank 1 per frame
    mother_df = df[df["rank"] == 1].sort_values("frame")
    if mother_df.empty:
        return None
    return mother_df["time_h"].values, mother_df["volume_um3_rod"].values, mother_df["mean_ri"].values


def compute_mean_band(yvals_list, t_grid):
    """Interpolate each (t, y) onto t_grid, return (mean, sd) arrays."""
    interp = []
    for entry in yvals_list:
        if len(entry) == 4:
            _, _, t, y = entry
        else:
            t, y = entry
        if t is None or len(t) == 0:
            continue
        # use linear interp; nan outside the cell's time range
        y_interp = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
        interp.append(y_interp)
    if not interp:
        return None, None
    M = np.vstack(interp)
    mean = np.nanmean(M, axis=0)
    sd = np.nanstd(M, axis=0)
    return mean, sd


def plot_overlay(ax, group_dead, group_revived, ycol_label, yvals_dead,
                 yvals_revived, swap_colors=False, multicolor_subset=False):
    """Plot revived + dead groups on one axes with overlap-density alpha.

    swap_colors=False: revived=gray, dead=red
    swap_colors=True:  revived=red, dead=gray
    multicolor_subset=True: foreground group uses distinct colors per trace
                            (for 6v6 subset to identify individual cells).
    """
    if swap_colors:
        revived_color, revived_alpha, revived_lw = "#d62728", 0.40, 0.45
        dead_color, dead_alpha, dead_lw = "#4a4a4a", 0.22, 0.40
        foreground_is_revived = True
    else:
        revived_color, revived_alpha, revived_lw = "#4a4a4a", 0.22, 0.40
        dead_color, dead_alpha, dead_lw = "#d62728", 0.40, 0.45
        foreground_is_revived = False
    # palette for multicolor subset (Okabe-Ito + extras, colorblind-safe ordering)
    palette = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2", "#D55E00",
               "#F0E442", "#000000"]

    # background group first
    for i, (pos, ch, t, y) in enumerate(yvals_revived):
        if t is None or len(t) == 0:
            continue
        if multicolor_subset and foreground_is_revived:
            color = palette[i % len(palette)]
            ax.plot(t - TIME_ZERO_H, y, color=color, linewidth=0.5,
                    alpha=0.75, zorder=2, solid_capstyle="butt",
                    label=f"{pos}_{ch}")
        else:
            ax.plot(t - TIME_ZERO_H, y, color=revived_color, linewidth=revived_lw,
                    alpha=revived_alpha, zorder=1, solid_capstyle="butt")
    for i, (pos, ch, t, y) in enumerate(yvals_dead):
        if t is None or len(t) == 0:
            continue
        if multicolor_subset and not foreground_is_revived:
            color = palette[i % len(palette)]
            ax.plot(t - TIME_ZERO_H, y, color=color, linewidth=0.5,
                    alpha=0.75, zorder=2, solid_capstyle="butt",
                    label=f"{pos}_{ch}")
        else:
            ax.plot(t - TIME_ZERO_H, y, color=dead_color, linewidth=dead_lw,
                    alpha=dead_alpha, zorder=2, solid_capstyle="butt")
    # Fix y-axis early so phase labels position is reproducible
    if "volume" in ycol_label:
        ax.set_ylim(0, 300)
    elif "RI" in ycol_label:
        ax.set_ylim(1.345, 1.385)
    # media-switch vertical lines (shifted)
    for label, frame in MEDIA_FRAMES.items():
        t_h = frame * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
        ax.axvline(t_h, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
    # annotate phase labels at top
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)
    t1 = MEDIA_FRAMES["wo_2 → wo_0.0055%"] * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
    t2 = MEDIA_FRAMES["wo_0.0055% → wo_0%"] * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
    t3 = MEDIA_FRAMES["wo_0% → wo_2%"] * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
    t_end = 3748 * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
    t_start = -TIME_ZERO_H
    ax.text((t_start + t1) / 2, y_text, "2% (phase1)", ha="center", fontsize=6, color="k", alpha=0.6)
    ax.text((t1 + t2) / 2, y_text, "0.0055%", ha="center", fontsize=6, color="k", alpha=0.6)
    ax.text((t2 + t3) / 2, y_text, "0%", ha="center", fontsize=6, color="k", alpha=0.6)
    ax.text((t3 + t_end) / 2, y_text, "2% recovery", ha="center", fontsize=6, color="k", alpha=0.6)
    ax.set_xlabel("time [h]  (original 120h = 0h)", fontsize=8)
    ax.set_ylabel(ycol_label, fontsize=8)
    ax.tick_params(labelsize=7)
    # clip x-axis to start at 0h (= original 120h)
    t_end_h = 3748 * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
    ax.set_xlim(0, t_end_h + 5)
    # fixed y-axis range for cross-figure comparability
    if "volume" in ycol_label:
        ax.set_ylim(0, 300)
    elif "RI" in ycol_label:
        ax.set_ylim(1.345, 1.385)


def main():
    # gather trajectories for both groups
    dead_traj = []
    print(f"\n=== Group A: never_revived mothers ({len(GROUP_DEAD)} channels) ===")
    for pos, ch in GROUP_DEAD:
        result = find_latest_run_for_channel(pos, ch)
        if result is None:
            print(f"  [skip] {pos}_{ch}: no run found")
            dead_traj.append((pos, ch, None, None, None))
            continue
        run_dir, ndiv = result
        traj = load_mother_trajectory(run_dir)
        if traj is None:
            print(f"  [skip] {pos}_{ch}: empty lineage csv")
            dead_traj.append((pos, ch, None, None, None))
            continue
        t, vol, ri = traj
        print(f"  [ok]   {pos}_{ch}: {len(t)} frames, n_div={ndiv}")
        dead_traj.append((pos, ch, t, vol, ri))

    revived_traj = []
    print(f"\n=== Group B: revived mothers ({len(GROUP_REVIVED)} channels) ===")
    for pos, ch in GROUP_REVIVED:
        result = find_latest_run_for_channel(pos, ch)
        if result is None:
            print(f"  [skip] {pos}_{ch}: no run found")
            revived_traj.append((pos, ch, None, None, None))
            continue
        run_dir, ndiv = result
        traj = load_mother_trajectory(run_dir)
        if traj is None:
            print(f"  [skip] {pos}_{ch}: empty lineage csv")
            revived_traj.append((pos, ch, None, None, None))
            continue
        t, vol, ri = traj
        print(f"  [ok]   {pos}_{ch}: {len(t)} frames, n_div={ndiv}")
        revived_traj.append((pos, ch, t, vol, ri))

    # Build 3 plot variants × 2 metrics (volume, mean RI) = 6 figures
    from matplotlib.lines import Line2D

    # representative 6+6 subsets: first 6 and next 6
    n_subset = 6
    dead_subset_a = dead_traj[:n_subset]
    revived_subset_a = revived_traj[:n_subset]
    dead_subset_b = dead_traj[n_subset:2 * n_subset]
    revived_subset_b = revived_traj[n_subset:2 * n_subset]

    variants = [
        ("all_revived_gray_dead_red", dead_traj, revived_traj, False,
         f"all (revived gray, dead red) — n_dead={len(GROUP_DEAD)}, n_revived={len(GROUP_REVIVED)}"),
        ("all_revived_red_dead_gray", dead_traj, revived_traj, True,
         f"all (revived red, dead gray) — n_dead={len(GROUP_DEAD)}, n_revived={len(GROUP_REVIVED)}"),
        ("subsetA_6vs6_revived_gray_dead_red", dead_subset_a, revived_subset_a,
         False, f"subset A (first 6 vs 6, revived gray, dead red)"),
        ("subsetA_6vs6_revived_red_dead_gray", dead_subset_a, revived_subset_a,
         True, f"subset A (first 6 vs 6, revived red, dead gray)"),
        ("subsetB_6vs6_revived_gray_dead_red", dead_subset_b, revived_subset_b,
         False, f"subset B (next 6 vs 6, revived gray, dead red)"),
        ("subsetB_6vs6_revived_red_dead_gray", dead_subset_b, revived_subset_b,
         True, f"subset B (next 6 vs 6, revived red, dead gray)"),
    ]

    # === mean ± SD band figure ===
    t_grid = np.arange(-TIME_ZERO_H, 3748 * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H + 0.5, 0.5)
    t_grid_unshifted = t_grid + TIME_ZERO_H

    for metric, ylabel, idx in [
        ("volume", r"mother volume [$\mathrm{\mu m^3}$]", 3),
        ("mean_RI", "mother mean RI", 4),
    ]:
        fig, ax = plt.subplots(figsize=(7.20, 3.60))
        # extract (t, y) per cell
        dead_yvals = [(p, c, t, (v if metric == "volume" else r))
                      for (p, c, t, v, r) in dead_traj]
        revived_yvals = [(p, c, t, (v if metric == "volume" else r))
                         for (p, c, t, v, r) in revived_traj]
        mean_d, sd_d = compute_mean_band(dead_yvals, t_grid_unshifted)
        mean_r, sd_r = compute_mean_band(revived_yvals, t_grid_unshifted)

        # revived: gray
        ax.fill_between(t_grid, mean_r - sd_r, mean_r + sd_r,
                        color="#4a4a4a", alpha=0.20, zorder=1, linewidth=0)
        ax.plot(t_grid, mean_r, color="#1a1a1a", linewidth=1.2, alpha=0.9,
                zorder=3, label=f"revived (n={len(GROUP_REVIVED)})")
        # dead: red
        ax.fill_between(t_grid, mean_d - sd_d, mean_d + sd_d,
                        color="#d62728", alpha=0.20, zorder=2, linewidth=0)
        ax.plot(t_grid, mean_d, color="#8c0000", linewidth=1.2, alpha=0.9,
                zorder=4, label=f"never_revived (n={len(GROUP_DEAD)})")

        # media-switch vlines
        for label, frame in MEDIA_FRAMES.items():
            t_h = frame * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
            ax.axvline(t_h, color="k", linestyle="--", linewidth=0.5, alpha=0.5)
        # fixed y-axis
        if metric == "volume":
            ax.set_ylim(0, 300)
        else:
            ax.set_ylim(1.345, 1.385)
        # phase labels
        t1 = MEDIA_FRAMES["wo_2 → wo_0.0055%"] * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
        t2 = MEDIA_FRAMES["wo_0.0055% → wo_0%"] * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
        t3 = MEDIA_FRAMES["wo_0% → wo_2%"] * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
        t_end = 3748 * TIME_INTERVAL_MIN / 60.0 - TIME_ZERO_H
        ymin, ymax = ax.get_ylim()
        y_text = ymax - 0.02 * (ymax - ymin)
        ax.text(t1 / 2 if t1 > 0 else -TIME_ZERO_H / 2, y_text,
                "2% (phase1)", ha="center", fontsize=6, color="k", alpha=0.6)
        ax.text((t1 + t2) / 2, y_text, "0.0055%", ha="center", fontsize=6, color="k", alpha=0.6)
        ax.text((t2 + t3) / 2, y_text, "0%", ha="center", fontsize=6, color="k", alpha=0.6)
        ax.text((t3 + t_end) / 2, y_text, "2% recovery", ha="center", fontsize=6, color="k", alpha=0.6)

        ax.set_xlabel("time [h]  (original 120h = 0h)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, t_end + 5)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.set_title(f"Mother {metric.replace('_', ' ')} — mean ± SD "
                     f"(revived gray vs never_revived red)", fontsize=8)
        fig.tight_layout()
        save_figure(
            fig,
            params={
                "variant": f"mean_sd_band_{metric}",
                "groups": "revived_vs_never_revived",
                "n_dead": len(GROUP_DEAD),
                "n_revived": len(GROUP_REVIVED),
                "y_metric": "volume_um3_rod" if metric == "volume" else "mean_ri",
                "interp_grid_dt_h": 0.5,
                "time_zero_h": TIME_ZERO_H,
            },
            description=f"mean ± SD band {metric} revived vs never_revived",
        )
        plt.close(fig)

    for variant_id, dead_data, revived_data, swap, subtitle in variants:
        is_subset = "subset" in variant_id
        for metric, ylabel, idx in [
            ("volume", r"mother volume [$\mathrm{\mu m^3}$]", 3),
            ("mean_RI", "mother mean RI", 4),
        ]:
            fig, ax = plt.subplots(figsize=(7.20, 3.60))
            yvals_dead = [(p, c, t, (v if metric == "volume" else r))
                          for (p, c, t, v, r) in dead_data]
            yvals_revived = [(p, c, t, (v if metric == "volume" else r))
                             for (p, c, t, v, r) in revived_data]
            plot_overlay(ax, GROUP_DEAD, GROUP_REVIVED, ylabel,
                         yvals_dead, yvals_revived, swap_colors=swap,
                         multicolor_subset=is_subset)
            if is_subset:
                # legend automatically built from per-trace labels (foreground only)
                ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=1)
            else:
                if swap:
                    revived_legend_color, dead_legend_color = "#d62728", "#4a4a4a"
                else:
                    revived_legend_color, dead_legend_color = "#4a4a4a", "#d62728"
                legend_handles = [
                    Line2D([0], [0], color=revived_legend_color, lw=1.2, alpha=0.7,
                           label=f"revived (n={len(revived_data)})"),
                    Line2D([0], [0], color=dead_legend_color, lw=1.2, alpha=0.85,
                           label=f"never_revived (n={len(dead_data)})"),
                ]
                ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
                          frameon=False)
            ax.set_title(f"Mother {metric.replace('_', ' ')} vs time — {subtitle}",
                         fontsize=7)
            fig.tight_layout()
            save_figure(
                fig,
                params={
                    "variant": variant_id,
                    "groups": "revived_vs_never_revived",
                    "n_dead": len(dead_data),
                    "n_revived": len(revived_data),
                    "y_metric": "volume_um3_rod" if metric == "volume" else "mean_ri",
                    "swap_colors": swap,
                    "time_zero_h": TIME_ZERO_H,
                    "media_schedule": "0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2",
                    "time_interval_min": TIME_INTERVAL_MIN,
                },
                description=f"overlay {metric} {variant_id}",
            )
            plt.close(fig)


if __name__ == "__main__":
    main()
