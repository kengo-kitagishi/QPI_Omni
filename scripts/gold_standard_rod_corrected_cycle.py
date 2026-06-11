"""Gold-standard cycle-aligned: raw skimage axes vs rod-model-corrected axes.

Side-by-side comparison of:
  - skimage second-moment ellipse fit (what's in lineage_data3D.csv now)
  - true rod L and 2r recovered via rod_axis_correction (cylinder + caps model)

For each cycle of each gold-standard mother we interpolate the per-frame
axes onto a 0–1 relative-progression grid, then average across cycles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import resolve_lineage_csv as find_lineage_csv  # noqa: E402
from gold_standard import select_gold_standard, rank1_division_frames  # noqa: E402
from rod_axis_correction import correction_factors  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018
N_REL = 100
LOW_MASS_PG = 10.0


def collect_cycles(channels):
    rel = np.linspace(0, 1, N_REL)
    rows = {"short_raw": [], "long_raw": [],
            "short_corr": [], "long_corr": [],
            "volume_raw": [], "volume_corr": []}
    for pos, ch in channels:
        path = find_lineage_csv(pos, ch)
        if path is None:
            continue
        df = pd.read_csv(path)
        try:
            clist = pd.read_csv(path.parent / "clist.csv")
        except Exception:
            clist = None
        divs = rank1_division_frames(df, clist)
        m = df[df["rank"] == 1].sort_values("frame").copy()
        bad = (m["is_outlier"].astype(bool)
               | m["touches_border"].astype(bool)
               | (m["mass_pg"] < LOW_MASS_PG))
        m_valid = m[~bad].copy()
        # corrected axes per frame
        L_corr, w_corr = correction_factors(
            m_valid["long_axis_um"].to_numpy(),
            m_valid["short_axis_um"].to_numpy(),
        )
        m_valid["long_corr"] = L_corr
        m_valid["short_corr"] = w_corr
        # corrected rod volume
        r_corr = w_corr / 2.0
        h_corr = np.maximum(L_corr - 2.0 * r_corr, 0.0)
        m_valid["volume_corr"] = (4.0 / 3.0) * np.pi * r_corr ** 3 + np.pi * r_corr ** 2 * h_corr

        for i in range(len(divs) - 1):
            f0, f1 = divs[i], divs[i + 1]
            if f1 > PHASE1_END_FRAME:
                continue
            sub = m_valid[(m_valid["frame"] >= f0) & (m_valid["frame"] < f1)]
            if len(sub) < 4:
                continue
            t = sub["frame"].to_numpy().astype(float)
            t_rel = (t - f0) / (f1 - f0)
            rows["short_raw"].append(np.interp(rel, t_rel, sub["short_axis_um"].to_numpy()))
            rows["long_raw"].append(np.interp(rel, t_rel, sub["long_axis_um"].to_numpy()))
            rows["short_corr"].append(np.interp(rel, t_rel, sub["short_corr"].to_numpy()))
            rows["long_corr"].append(np.interp(rel, t_rel, sub["long_corr"].to_numpy()))
            rows["volume_raw"].append(np.interp(rel, t_rel, sub["volume_um3_rod"].to_numpy()))
            rows["volume_corr"].append(np.interp(rel, t_rel, sub["volume_corr"].to_numpy()))
    return rel, {k: np.vstack(v) for k, v in rows.items() if v}


def plot(rel, M):
    panels = [
        ("short_raw", "short_corr", "short axis [μm]", "#0072B2"),
        ("long_raw",  "long_corr",  "long axis [μm]",  "#E69F00"),
        ("volume_raw", "volume_corr", r"rod volume [μm³]", "#009E73"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(120 / 25.4, 160 / 25.4),
                             sharex=True, constrained_layout=True)
    for ax, (raw_k, corr_k, ylabel, color) in zip(axes, panels):
        raw = M[raw_k]
        corr = M[corr_k]
        raw_mean = np.nanmean(raw, axis=0)
        raw_sd = np.nanstd(raw, axis=0)
        corr_mean = np.nanmean(corr, axis=0)
        corr_sd = np.nanstd(corr, axis=0)
        ax.fill_between(rel, raw_mean - raw_sd, raw_mean + raw_sd,
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(rel, raw_mean, color=color, linewidth=1.2, linestyle="--",
                label=f"raw (skimage)  Δ={raw_mean.max() - raw_mean.min():.3g} "
                      f"({100*(raw_mean.max()-raw_mean.min())/raw_mean.mean():.1f} %)")
        ax.fill_between(rel, corr_mean - corr_sd, corr_mean + corr_sd,
                        color="#444444", alpha=0.18, linewidth=0)
        ax.plot(rel, corr_mean, color="#1a1a1a", linewidth=1.2,
                label=f"rod-corrected   Δ={corr_mean.max() - corr_mean.min():.3g} "
                      f"({100*(corr_mean.max()-corr_mean.min())/corr_mean.mean():.1f} %)")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="lower right", frameon=False, fontsize=6)
    axes[-1].set_xlabel("relative cell cycle progression", fontsize=8)
    fig.suptitle(
        "Gold-standard phase1 cycle-aligned: raw vs rod-model-corrected axes",
        fontsize=9,
    )
    n_cycles = M["short_raw"].shape[0]
    save_figure(
        fig,
        params={
            "n_cycles_pooled": int(n_cycles),
            "phase1_end_frame": PHASE1_END_FRAME,
            "correction_model": "rod = cylinder + 2 semicircular caps",
        },
        description="gold-standard phase1 cycle-aligned raw vs corrected axes/volume",
    )
    plt.close(fig)


def main():
    gold = select_gold_standard()
    print(f"gold-standard mothers: {len(gold)}")
    rel, M = collect_cycles(gold)
    n_cycles = M["short_raw"].shape[0]
    print(f"n_cycles pooled: {n_cycles}")
    print()
    for key in M:
        mean = np.nanmean(M[key], axis=0)
        print(f"{key:14s}: rel=0 {mean[0]:.4f}  rel=1 {mean[-1]:.4f}  "
              f"Δ {mean.max() - mean.min():.4f}  "
              f"({100 * (mean.max() - mean.min()) / mean.mean():.2f} % of mean)")
    plot(rel, M)


if __name__ == "__main__":
    main()
