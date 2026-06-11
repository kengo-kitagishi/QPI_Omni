"""Overlay all well-tracked mother trajectories in gray.

Pool:
  - Gold-standard phase1 survivors (zero abnormal intervals)
  - Phase1-dead channels that look well tracked up to death
    (median_interval_h in [2.8, 4.0], n_div >= 5, abnormal_before_death=False)

All traces plotted in gray (no group distinction) on one mean RI vs time figure
and one volume vs time figure. Phase1 time window only (0..168 h).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mother_revived_vs_dead import find_latest_run_for_channel  # noqa: E402
from figure_logger import save_figure  # noqa: E402
from qpi_paths import resolve_lineage_csv  # noqa: E402

from qpi_paths import results_dir  # noqa: E402
QUALITY_REVIVED_CSV = results_dir() / "quality_check_all_mother_revived.csv"
QUALITY_PHASE1_DEAD_CSV = results_dir() / "phase1_2per_7days" / "quality_check_division_intervals.csv"
PHASE1_T_MAX = 168.25
MEDIA_FRAMES_MIN = 5.0
TIME_INTERVAL_MIN_GLOBAL = 5.0  # min/frame

# batch_figures inbox uses flat per-channel file naming (Pos40+ lineage lives here)
BATCH_FIGURES_INBOX_DIRS = [
    Path("/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-06-10/batch_figures"),
]


def find_lineage_csv(pos: str, ch: str) -> Path | None:
    """Latest lineage_data3D.csv for (pos, ch) — corrected when enabled.

    Delegates to qpi_paths.resolve_lineage_csv, which returns the mask-direct
    corrected CSV when QPI_USE_CORRECTED=1 (variant via QPI_VOLUME_VARIANT) and
    otherwise the latest inbox CSV (both per_channel_figures and batch_figures
    layouts), resolved cross-platform.
    """
    return resolve_lineage_csv(pos, ch)

# Channels the user has visually confirmed as bad and wants to drop entirely
# Edge channels (ch00, ch11) tend to be affected by OOB / channel boundary
# artefacts, so the user flags individual ones as they appear.
MANUAL_EXCLUDE = {
    ("Pos37", "ch11"),  # segmentation clearly failed
    ("Pos17", "ch11"),  # 端 ch, OOB 影響
    ("Pos22", "ch11"),  # 端 ch, OOB 影響
    ("Pos24", "ch00"),  # 端 ch, OOB 影響
    ("Pos36", "ch11"),  # 端 ch, segmentation 怪しい
    # batch 2 — user-flagged after viewing revived grid 2026-06-10
    ("Pos6", "ch06"),
    ("Pos10", "ch03"),
    ("Pos10", "ch00"),
    ("Pos13", "ch11"),
    ("Pos15", "ch01"),
    ("Pos18", "ch01"),
    ("Pos20", "ch11"),
    ("Pos24", "ch03"),
    ("Pos26", "ch05"),
    ("Pos26", "ch11"),
    ("Pos31", "ch11"),
    ("Pos14", "ch07"),
    ("Pos36", "ch05"),
    ("Pos38", "ch11"),
    ("Pos39", "ch11"),
    ("Pos40", "ch11"),
    ("Pos41", "ch11"),
    ("Pos43", "ch00"),
    ("Pos44", "ch02"),
    ("Pos44", "ch09"),
    ("Pos46", "ch00"),
    ("Pos39", "ch00"),
    ("Pos40", "ch03"),  # phase1 死亡の可能性あり
    ("Pos38", "ch00"),
    ("Pos42", "ch00"),
    ("Pos23", "ch01"),  # seg 質が悪い
    # elongation しながら revive する mother（典型的 revive と区別したい）
    ("Pos32", "ch02"),
    ("Pos30", "ch07"),
    ("Pos7", "ch06"),
    ("Pos32", "ch03"),  # osmotic lysis at recovery (典型例だが平均から外す)
}


def parse_channel(label: str) -> tuple[str, str]:
    pos, ch = label.split("_", 1)
    return pos, ch


def select_gold_standard() -> list[tuple[str, str]]:
    """Gold-standard list: every inter-division interval in [2.5, 5.0] h.

    No additional sanity filtering on volume or RI — the user opted to rely
    on the division-interval criterion alone. MANUAL_EXCLUDE entries are
    still dropped (these are channels flagged visually as wrong-blob or
    edge-OOB problems).

    Uses the precomputed quality CSV when present (macOS), otherwise computes
    the same criterion directly from the lineage data (portable, Windows).
    """
    if QUALITY_REVIVED_CSV.exists():
        df = pd.read_csv(QUALITY_REVIVED_CSV)
        gold = df[(df["status"] == "ok") & (df["is_gold_standard"] == True)]  # noqa: E712
        out = []
        for c in gold["channel"].tolist():
            pair = parse_channel(c)
            if pair in MANUAL_EXCLUDE:
                continue
            out.append(pair)
        return out
    from gold_standard import select_gold_standard as _portable_gold
    return _portable_gold()


def select_all_phase1_dead() -> list[tuple[str, str]]:
    """All phase1-dead channels with lineage data — minus manual excludes.

    Uses the quality CSV when present (macOS), otherwise the YAML phase1=dead
    set (portable, Windows)."""
    if QUALITY_PHASE1_DEAD_CSV.exists():
        df = pd.read_csv(QUALITY_PHASE1_DEAD_CSV)
        sub = df[df["group"] == "phase1_dead"]
        out = []
        for c in sub["channel"].tolist():
            pair = parse_channel(c)
            if pair in MANUAL_EXCLUDE:
                continue
            out.append(pair)
        return out
    from gold_standard import phase1_dead_sorted_by_death_proxy
    return [(p, c) for p, c, _ in phase1_dead_sorted_by_death_proxy()]


def select_phase1_dead_sorted_by_death() -> list[tuple[str, str, int]]:
    """Phase1-dead channels sorted by death frame, manual excludes removed.

    Uses the quality CSV's death_frame when present (macOS), otherwise a
    lineage-derived death-frame proxy (portable, Windows)."""
    if QUALITY_PHASE1_DEAD_CSV.exists():
        df = pd.read_csv(QUALITY_PHASE1_DEAD_CSV)
        sub = df[df["group"] == "phase1_dead"].dropna(subset=["death_frame"])
        sub = sub.sort_values("death_frame")
        out = []
        for c, df_row in zip(sub["channel"], sub.to_dict("records")):
            pair = parse_channel(c)
            if pair in MANUAL_EXCLUDE:
                continue
            out.append((pair[0], pair[1], int(df_row["death_frame"])))
        return out
    from gold_standard import phase1_dead_sorted_by_death_proxy
    return phase1_dead_sorted_by_death_proxy()


def load_mother_phase1(pos: str, ch: str):
    """Return (time_h, volume, mean_ri, mass_pg) for the mother in phase1.

    Outlier and border-touching frames are mapped to NaN so the line breaks
    visually at those frames (we don't want to plot bad-frame values).
    """
    lineage_path = find_lineage_csv(pos, ch)
    if lineage_path is None:
        return None
    df = pd.read_csv(lineage_path)
    m = df[df["rank"] == 1].sort_values("frame").copy()
    m = m[m["time_h"] <= PHASE1_T_MAX]
    if m.empty:
        return None
    bad = m["is_outlier"].to_numpy(dtype=bool) | m["touches_border"].to_numpy(dtype=bool)
    vol = m["volume_um3_rod"].to_numpy(dtype=float)
    ri = m["mean_ri"].to_numpy(dtype=float)
    mass = m["mass_pg"].to_numpy(dtype=float)
    # physically implausible mother mass (healthy mothers are ~15-40 pg);
    # mass < 10 pg here means the tracker latched onto a fragment / wrong blob.
    low_mass = mass < 10.0
    bad = bad | low_mass
    vol[bad] = np.nan
    ri[bad] = np.nan
    mass[bad] = np.nan
    return m["time_h"].to_numpy(), vol, ri, mass


def _gold_percentile_band(gold: list[tuple[str, str]], metric: str,
                          t_grid: np.ndarray):
    """Interpolate every gold-standard trajectory onto a common grid; return
    (p10, p25, p50, p75, p90)."""
    rows = []
    for pos, ch in gold:
        traj = load_mother_phase1(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        # interpolate; outside range -> NaN
        y_interp = np.interp(t_grid, t, y, left=np.nan, right=np.nan)
        rows.append(y_interp)
    if not rows:
        return None
    M = np.vstack(rows)
    return (np.nanpercentile(M, 10, axis=0),
            np.nanpercentile(M, 25, axis=0),
            np.nanpercentile(M, 50, axis=0),
            np.nanpercentile(M, 75, axis=0),
            np.nanpercentile(M, 90, axis=0))


def _pick_metric(traj, metric: str):
    t, vol, ri, mass = traj
    if metric == "volume":
        return t, vol
    if metric == "mass":
        return t, mass
    return t, ri


def plot_overlay_two_groups(
    gold: list[tuple[str, str]],
    dead: list[tuple[str, str]],
    metric: str,
    ylabel: str,
    ylim: tuple[float, float],
    out_label: str,
):
    fig, ax = plt.subplots(figsize=(7.20, 3.60))
    n_gold = 0
    # thinner gold-standard lines (user request)
    for pos, ch in gold:
        traj = load_mother_phase1(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        ax.plot(t, y, color="#4a4a4a", linewidth=0.15, alpha=0.18,
                zorder=1, solid_capstyle="butt")
        n_gold += 1
    palette = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#0072B2",
               "#D55E00", "#F0E442", "#000000", "#9467bd", "#8c564b"]
    n_dead = 0
    for i, (pos, ch) in enumerate(dead):
        traj = load_mother_phase1(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        ax.plot(t, y, color=palette[i % len(palette)], linewidth=0.6,
                alpha=0.85, zorder=2 + i, solid_capstyle="butt",
                label=f"{pos}_{ch}")
        n_dead += 1
    ax.set_xlim(0, PHASE1_T_MAX)
    ax.set_ylim(*ylim)
    ax.set_xlabel("time [h]", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    metric_pretty = {
        "mean_RI": "refractive index",
        "volume": "volume",
        "mass": "dry mass",
    }.get(metric, metric)
    ax.legend(loc="upper right", fontsize=6, frameon=False, ncol=1,
              title=f"dying lineages (surviving n={n_gold})",
              title_fontsize=7)
    ax.set_title(
        f"Mother cell {metric_pretty} during 2% glucose growth — "
        f"surviving (gray) vs dying lineages",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(
        fig,
        params={
            "metric": metric,
            "n_gold": n_gold,
            "n_phase1_dead": n_dead,
            "selection": "gold thin lines + colored phase1-dead",
            "phase1_t_max_h": PHASE1_T_MAX,
        },
        description=f"overlay {metric} {out_label}",
    )
    plt.close(fig)


def main():
    gold = select_gold_standard()
    dead_sorted = select_phase1_dead_sorted_by_death()
    print(f"Gold-standard channels: {len(gold)}")
    print(f"Phase1-dead channels sorted by death_frame: {len(dead_sorted)}")
    # split into panels of 3 channels each
    panels_per_set = 3
    n_total = len(dead_sorted)
    panel_sizes = [panels_per_set] * (n_total // panels_per_set)
    if n_total % panels_per_set:
        panel_sizes.append(n_total % panels_per_set)
    idx = 0
    n_panels = len(panel_sizes)
    for panel_n, size in enumerate(panel_sizes, 1):
        chunk = dead_sorted[idx: idx + size]
        idx += size
        if not chunk:
            continue
        dead_pairs = [(p, c) for p, c, _df in chunk]
        df_range = f"{chunk[0][2]}-{chunk[-1][2]}"
        names = [f"{p}_{c}(death={df_})" for p, c, df_ in chunk]
        print(f"\nPanel {panel_n}/{n_panels} (death_frame {df_range}): {names}")
        plot_overlay_two_groups(
            gold, dead_pairs, "mean_RI", "mother mean RI",
            (1.36, 1.41),
            f"panel{panel_n}of{n_panels}_death_{df_range}_mean_RI",
        )
        plot_overlay_two_groups(
            gold, dead_pairs, "volume",
            r"mother volume [$\mu m^3$]",
            (0, 300),
            f"panel{panel_n}of{n_panels}_death_{df_range}_volume",
        )
        plot_overlay_two_groups(
            gold, dead_pairs, "mass",
            "mother dry mass [pg]",
            (0, 60),
            f"panel{panel_n}of{n_panels}_death_{df_range}_mass",
        )


if __name__ == "__main__":
    main()
