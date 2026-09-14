"""Mean ± SD band overlay over the full timecourse.

Two groups, both restricted to lineages that pass the current quality filter
(MANUAL_EXCLUDE applied, low_mass/border/outlier frames masked):
  - revived mothers (gold-standard ~3.25 h cadence + survived phase2)
  - never_revived mothers (mother died in starvation / failed to revive)

For each group, every mother trajectory is interpolated onto a common time
grid and the per-time-bin mean ± SD is plotted as a band.

Time axis is shifted so the original 120 h becomes 0 h (matches the
reference figure in 2026-06-09/overlay_mother_revived_vs_dead/
20260609T073043Z_0eeee0/).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    _pick_metric, select_gold_standard, MANUAL_EXCLUDE, find_lineage_csv,
)
from figure_logger import save_figure  # noqa: E402
from qpi_paths import yaml_path as _yaml_path  # noqa: E402

YAML_PATH = _yaml_path()

TIME_ZERO_H = 120.0
TIME_INTERVAL_MIN = 5.0
T_END_H_RAW = 3748 * TIME_INTERVAL_MIN / 60.0
PHASE1_END_RAW = 2019 * TIME_INTERVAL_MIN / 60.0          # 168 h
WO0_START_RAW = 2307 * TIME_INTERVAL_MIN / 60.0           # 192 h
RECOVERY_RAW = 2885 * TIME_INTERVAL_MIN / 60.0            # 240 h


def list_revived_mothers() -> list[tuple[str, str]]:
    """phase1=alive + mother revived in phase2."""
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
            p1 = (fields.get("phase1") or {}).get("outcome")
            if p1 != "alive":
                continue
            p2 = fields.get("phase2") or {}
            p2_out = p2.get("outcome")
            notes = (p2.get("notes") or "").lstrip()
            if p2_out == "revived":
                out.append((pos_name, ch_name))
                continue
            if p2_out == "mixed":
                first = notes.split("\n")[0].strip()
                m = re.match(r"^mother[^:]*:\s*(revived|never_revived|dead)",
                             first)
                if m and m.group(1) == "revived":
                    out.append((pos_name, ch_name))
    return out


def list_never_revived_mothers() -> list[tuple[str, str]]:
    """phase1=alive + mother did NOT revive in phase2."""
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
            p1 = (fields.get("phase1") or {}).get("outcome")
            if p1 != "alive":
                continue
            p2 = fields.get("phase2") or {}
            p2_out = p2.get("outcome")
            notes = (p2.get("notes") or "").lstrip()
            if p2_out in ("died_starvation", "never_revived"):
                out.append((pos_name, ch_name))
                continue
            if p2_out == "mixed":
                first = notes.split("\n")[0].strip()
                m = re.match(r"^mother[^:]*:\s*(revived|never_revived|dead)",
                             first)
                if m and m.group(1) != "revived":
                    out.append((pos_name, ch_name))
    return out


def load_mother_full(pos: str, ch: str):
    """Return (time_h, vol, ri, mass) over full timecourse, outliers→NaN."""
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
    bad = bad | (mass < 10.0)
    vol[bad] = np.nan
    ri[bad] = np.nan
    mass[bad] = np.nan
    return m["time_h"].to_numpy(), vol, ri, mass


def interpolate(channels, metric: str, t_grid_raw: np.ndarray):
    rows = []
    for pos, ch in channels:
        traj = load_mother_full(pos, ch)
        if traj is None:
            continue
        t, y = _pick_metric(traj, metric)
        rows.append(np.interp(t_grid_raw, t, y, left=np.nan, right=np.nan))
    if not rows:
        return None
    return np.vstack(rows)


def plot_band(metric: str, ylabel: str, ylim, revived, never_revived,
              t_grid_raw: np.ndarray):
    fig, ax = plt.subplots(figsize=(7.20, 3.60))
    M_r = interpolate(revived, metric, t_grid_raw)
    M_d = interpolate(never_revived, metric, t_grid_raw)
    t_shifted = t_grid_raw - TIME_ZERO_H

    n_r = M_r.shape[0] if M_r is not None else 0
    n_d = M_d.shape[0] if M_d is not None else 0

    if M_r is not None:
        mean = np.nanmean(M_r, axis=0)
        sd = np.nanstd(M_r, axis=0)
        ax.fill_between(t_shifted, mean - sd, mean + sd, color="#4a4a4a",
                        alpha=0.20, linewidth=0, zorder=1)
        ax.plot(t_shifted, mean, color="#1a1a1a", linewidth=1.2, alpha=0.9,
                zorder=3, label=f"revived (n={n_r})")
    if M_d is not None:
        mean = np.nanmean(M_d, axis=0)
        sd = np.nanstd(M_d, axis=0)
        ax.fill_between(t_shifted, mean - sd, mean + sd, color="#d62728",
                        alpha=0.20, linewidth=0, zorder=2)
        ax.plot(t_shifted, mean, color="#8c0000", linewidth=1.2, alpha=0.9,
                zorder=4, label=f"never_revived (n={n_d})")

    # phase markers
    for raw_h, label in [
        (PHASE1_END_RAW, "0.0055%"),
        (WO0_START_RAW, "0%"),
        (RECOVERY_RAW, "2% recovery"),
    ]:
        x = raw_h - TIME_ZERO_H
        ax.axvline(x, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

    # phase labels
    ax.set_xlim(0, T_END_H_RAW - TIME_ZERO_H + 5)
    ax.set_ylim(*ylim)
    t_end_shift = T_END_H_RAW - TIME_ZERO_H
    ymin, ymax = ax.get_ylim()
    y_text = ymax - 0.02 * (ymax - ymin)
    t1 = PHASE1_END_RAW - TIME_ZERO_H
    t2 = WO0_START_RAW - TIME_ZERO_H
    t3 = RECOVERY_RAW - TIME_ZERO_H
    ax.text(t1 / 2, y_text, "2% (phase1)", ha="center", fontsize=6,
            color="k", alpha=0.6)
    ax.text((t1 + t2) / 2, y_text, "0.0055%", ha="center", fontsize=6,
            color="k", alpha=0.6)
    ax.text((t2 + t3) / 2, y_text, "0%", ha="center", fontsize=6,
            color="k", alpha=0.6)
    ax.text((t3 + t_end_shift) / 2, y_text, "2% recovery",
            ha="center", fontsize=6, color="k", alpha=0.6)
    ax.set_xlabel("time [h]  (original 120h = 0h)", fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    metric_pretty = {"mean_RI": "refractive index",
                     "volume": "volume",
                     "mass": "dry mass"}.get(metric, metric)
    ax.set_title(
        f"Mother cell {metric_pretty} — mean ± SD "
        f"(revived gray vs never_revived red)",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(
        fig,
        params={
            "metric": metric,
            "n_revived": n_r,
            "n_never_revived": n_d,
            "time_zero_h": TIME_ZERO_H,
            "interp_grid_dt_h": 0.5,
        },
        description=f"mean SD band {metric}",
    )
    plt.close(fig)


def main():
    revived = list_revived_mothers()
    never_revived = list_never_revived_mothers()
    print(f"revived (mother revived in phase2): {len(revived)}")
    print(f"never_revived (mother died/no recovery): {len(never_revived)}")
    t_grid_raw = np.arange(0, T_END_H_RAW + 0.5, 0.5)
    for metric, ylabel, ylim in [
        ("mean_RI", "mother mean RI", (1.345, 1.385)),
        ("volume", r"mother volume [$\mu m^3$]", (0, 300)),
        ("mass", "mother dry mass [pg]", (0, 80)),
    ]:
        plot_band(metric, ylabel, ylim, revived, never_revived, t_grid_raw)


if __name__ == "__main__":
    main()
