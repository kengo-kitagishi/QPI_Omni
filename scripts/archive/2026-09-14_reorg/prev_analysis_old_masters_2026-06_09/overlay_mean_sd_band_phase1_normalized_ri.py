"""Mean ± SD band for mean RI with phase1-normalized per-channel shift.

For each lineage:
  shift = TARGET_PHASE1_MEAN - nanmean(RI during phase1, i.e. time_h < 168)
  RI_shifted = RI + shift

Then per-time-bin mean ± SD across lineages, plotted separately for the
revived and never_revived groups. This removes between-channel absolute
offsets (gain drift, density, etc.) so the *shape* of the post-starvation
divergence becomes visible.

Time axis is shifted so the original 120 h becomes 0 h.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from overlay_mean_sd_band_full_timecourse import (  # noqa: E402
    list_revived_mothers, list_never_revived_mothers, load_mother_full,
    TIME_ZERO_H, T_END_H_RAW, PHASE1_END_RAW, WO0_START_RAW, RECOVERY_RAW,
)
from figure_logger import save_figure  # noqa: E402

TARGET_PHASE1_MEAN = 1.36
PHASE1_END_H = PHASE1_END_RAW  # 168.25 h on the raw axis


def load_mother_phase1_normalized_ri(pos: str, ch: str):
    """Return (time_h, ri_shifted, shift) with per-channel phase1 normalization.

    shift is NaN (and ri_shifted is all-NaN) if the lineage has no valid RI
    samples in phase1.
    """
    traj = load_mother_full(pos, ch)
    if traj is None:
        return None
    t, _vol, ri, _mass = traj
    in_phase1 = t < PHASE1_END_H
    p1_mean = np.nanmean(ri[in_phase1]) if in_phase1.any() else np.nan
    if not np.isfinite(p1_mean):
        return None
    shift = TARGET_PHASE1_MEAN - p1_mean
    return t, ri + shift, shift


def interpolate_normalized(channels, t_grid_raw):
    rows = []
    shifts = []
    for pos, ch in channels:
        res = load_mother_phase1_normalized_ri(pos, ch)
        if res is None:
            continue
        t, y, shift = res
        rows.append(np.interp(t_grid_raw, t, y, left=np.nan, right=np.nan))
        shifts.append(shift)
    if not rows:
        return None, []
    return np.vstack(rows), shifts


def plot_band(revived, never_revived, t_grid_raw):
    fig, ax = plt.subplots(figsize=(7.20, 3.60))
    M_r, sh_r = interpolate_normalized(revived, t_grid_raw)
    M_d, sh_d = interpolate_normalized(never_revived, t_grid_raw)
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

    for raw_h in (PHASE1_END_RAW, WO0_START_RAW, RECOVERY_RAW):
        x = raw_h - TIME_ZERO_H
        ax.axvline(x, color="k", linestyle="--", linewidth=0.5, alpha=0.5)

    ax.axhline(TARGET_PHASE1_MEAN, color="k", linestyle=":",
               linewidth=0.5, alpha=0.4)

    ax.set_xlim(0, T_END_H_RAW - TIME_ZERO_H + 5)
    ax.set_ylim(1.345, 1.385)
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
    ax.set_ylabel("mother mean RI (phase1-normalized to 1.360)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.set_title(
        "Mother cell mean RI — phase1 mean shifted to 1.360 per channel "
        "(revived gray vs never_revived red)",
        fontsize=8,
    )
    fig.tight_layout()
    save_figure(
        fig,
        params={
            "metric": "mean_RI",
            "normalization": "per-channel phase1 mean -> 1.360",
            "n_revived": n_r,
            "n_never_revived": n_d,
            "shift_revived_mean": float(np.mean(sh_r)) if sh_r else None,
            "shift_revived_sd": float(np.std(sh_r)) if sh_r else None,
            "shift_never_revived_mean": float(np.mean(sh_d)) if sh_d else None,
            "shift_never_revived_sd": float(np.std(sh_d)) if sh_d else None,
            "time_zero_h": TIME_ZERO_H,
            "interp_grid_dt_h": 0.5,
            "phase1_end_h": PHASE1_END_H,
        },
        description="mean SD band mean_RI phase1-normalized",
    )
    plt.close(fig)
    return n_r, n_d, sh_r, sh_d


def main():
    revived = list_revived_mothers()
    never_revived = list_never_revived_mothers()
    print(f"revived: {len(revived)}")
    print(f"never_revived: {len(never_revived)}")
    t_grid_raw = np.arange(0, T_END_H_RAW + 0.5, 0.5)
    n_r, n_d, sh_r, sh_d = plot_band(revived, never_revived, t_grid_raw)
    if sh_r:
        print(f"revived shift: mean={np.mean(sh_r):+.5f}, "
              f"sd={np.std(sh_r):.5f}, n={n_r}")
    if sh_d:
        print(f"never_revived shift: mean={np.mean(sh_d):+.5f}, "
              f"sd={np.std(sh_d):.5f}, n={n_d}")


if __name__ == "__main__":
    main()
