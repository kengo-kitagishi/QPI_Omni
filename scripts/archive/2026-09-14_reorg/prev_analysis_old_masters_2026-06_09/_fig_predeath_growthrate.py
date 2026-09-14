"""_fig_predeath_growthrate.py — pre-death dry-mass growth rate by death mode.

For each phase1-dead mother lineage, fit d ln(M)/dt over the LAST COMPLETED cell
cycle (the reliable pre-death window) using EFD dry mass, and compare between
death modes (swelling vs elongation cascade) as a box + individual-points figure.

Hypothesis: the dry-mass growth rate decreases just before death in normal
(swelling) deaths, but does NOT decrease in elongation-cascade deaths.

Death-window curation: after a phase1 mother lyses the trench repopulates and the
rank=1 tracker follows the new cell, so the lineage death_frame (=3747),
death_frame_proxy (=max frame) and auto cycle-enumeration are all corrupted
(e.g. Pos2_ch03 really dies ~frame 300 but the trace "divides" to 1978). The
reliable death point is the last completed cell cycle from the user-curated
2026-06-17 fig_panelA_cellcycle sheets (its 2nd-to-last row). Those windows are
hardcoded below and printed at runtime for verification.

EFD-aware: QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd routes resolve_lineage_csv
to the corrected_lineage_efd CSV whose mass_pg column holds the EFD value.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import resolve_lineage_csv  # noqa: E402
from _fig_panelA_cellcycle import (  # noqa: E402
    enumerate_all_cycles, data_root, load_bad_frames,
)
from figure_logger import save_figure  # noqa: E402

# Last completed cell cycle [f0, f1] per phase1-dead channel = 2nd-to-last row of
# the user-curated 2026-06-17 fig_panelA_cellcycle sheets (death point = f1).
# Pos37_ch11 excluded (MANUAL_EXCLUDE: segmentation failed).
DEATH_WINDOW = {
    "Pos1_ch09": (1815, 1851), "Pos2_ch03": (300, 327), "Pos2_ch08": (591, 629),
    "Pos5_ch08": (1668, 1713), "Pos6_ch09": (473, 502), "Pos9_ch04": (1259, 1293),
    "Pos9_ch08": (1160, 1201), "Pos10_ch07": (752, 796), "Pos11_ch06": (956, 1006),
    "Pos14_ch08": (310, 352), "Pos17_ch02": (137, 171), "Pos17_ch09": (529, 586),
    "Pos18_ch02": (1823, 1855), "Pos20_ch06": (1178, 1226), "Pos26_ch03": (723, 791),
    "Pos26_ch08": (606, 654), "Pos30_ch04": (983, 1011), "Pos31_ch06": (1108, 1138),
    "Pos32_ch05": (382, 418), "Pos35_ch05": (307, 342), "Pos37_ch08": (1122, 1180),
    "Pos39_ch05": (529, 563), "Pos42_ch01": (708, 747), "Pos45_ch00": (662, 719),
    "Pos45_ch02": (106, 173),
}
ELONGATION = {"Pos20_ch06", "Pos30_ch04"}

OI = {"orange": "#E69F00", "skyblue": "#56B4E9", "green": "#009E73",
      "blue": "#0072B2", "vermilion": "#D55E00", "purple": "#CC79A7"}
SWELL_COLOR = OI["skyblue"]
ELONG_COLOR = OI["vermilion"]
MIN_PTS = 4

plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _fit_dlnM_dt(sub: pd.DataFrame) -> float | None:
    """OLS slope of ln(mass_pg) vs time_h over valid frames [1/h], else None."""
    v = sub[~(sub["is_outlier"] | sub["touches_border"]) & (sub["mass_pg"] >= 10.0)]
    if len(v) < MIN_PTS:
        return None
    t = v["time_h"].to_numpy(float)
    m = v["mass_pg"].to_numpy(float)
    if np.any(m <= 0):
        return None
    return float(np.polyfit(t, np.log(m), 1)[0])


def collect() -> pd.DataFrame:
    rows = []
    for ch_key, (f0, f1) in DEATH_WINDOW.items():
        pos, ch = ch_key.split("_", 1)
        p = resolve_lineage_csv(pos, ch)
        if p is None:
            print(f"  [skip] {ch_key}: no lineage CSV")
            continue
        df = pd.read_csv(p)
        m = df[df["rank"] == 1].sort_values("frame").reset_index(drop=True)

        # final (pre-death) rate over the last completed cycle [f0, f1]
        win = m[(m["frame"] >= f0) & (m["frame"] <= f1)]
        final = _fit_dlnM_dt(win)
        if final is None:
            print(f"  [skip] {ch_key}: <{MIN_PTS} valid EFD frames in [{f0},{f1}]")
            continue

        # baseline = median d ln M/dt over earlier completed cycles (frame < f0);
        # the <f0 cutoff excludes the post-death repopulation contamination.
        try:
            bad = load_bad_frames(data_root(pos) / ch, pos)
        except Exception:
            bad = set()
        cyc, _ = enumerate_all_cycles(m, bad, max_frame=f0 - 1, min_kept=6)
        base_rates = [r for r in (_fit_dlnM_dt(c) for c in cyc) if r is not None]
        baseline = float(np.median(base_rates)) if base_rates else np.nan
        ratio = (final / baseline) if (baseline and np.isfinite(baseline)) else np.nan

        rows.append({
            "ch": ch_key, "pos": pos, "chn": ch,
            "mode": "elongation" if ch_key in ELONGATION else "swelling",
            "f0": f0, "f1": f1, "n_pts": int(len(win)),
            "final_rate": final, "baseline": baseline, "ratio": ratio,
            "n_base_cyc": len(base_rates),
        })
    return pd.DataFrame(rows)


def _panel(ax, df: pd.DataFrame, col: str, ylabel: str, hline: float | None = None):
    """Swelling as a box + jittered points; elongation as highlighted diamonds."""
    rng = np.random.default_rng(0)
    sw = df[df["mode"] == "swelling"][col].dropna().to_numpy()
    el = df[df["mode"] == "elongation"][col].dropna().to_numpy()

    bp = ax.boxplot([sw], positions=[0], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops=dict(color="black", lw=1.2),
                    whiskerprops=dict(color="0.4", lw=0.8),
                    capprops=dict(color="0.4", lw=0.8), boxprops=dict(lw=0.8))
    bp["boxes"][0].set_facecolor(SWELL_COLOR)
    bp["boxes"][0].set_alpha(0.45)
    bp["boxes"][0].set_edgecolor("0.25")
    ax.scatter(rng.uniform(-0.13, 0.13, len(sw)), sw, s=14, color=SWELL_COLOR,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=5)
    ax.scatter(np.linspace(-0.06, 0.06, len(el)) + 1, el, s=46, color=ELONG_COLOR,
               edgecolor="black", linewidth=0.6, alpha=0.95, zorder=6, marker="D")
    if hline is not None:
        ax.axhline(hline, color="#000", lw=0.5, ls=":", zorder=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"swelling\n(n={len(sw)})", f"elongation\n(n={len(el)})"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    if len(sw) and len(el):
        try:
            pv = float(mannwhitneyu(sw, el, alternative="two-sided").pvalue)
            ax.set_title(f"MWU p={pv:.2g}  (n=2 elong → descriptive)", fontsize=6.5)
        except ValueError:
            pass


def main():
    df = collect()
    n_sw = int((df["mode"] == "swelling").sum())
    n_el = int((df["mode"] == "elongation").sum())
    print(f"\nn_lineages={len(df)}  swelling={n_sw}  elongation={n_el}")
    print(df[["ch", "mode", "f0", "f1", "n_pts", "final_rate",
              "baseline", "ratio", "n_base_cyc"]].to_string(index=False))
    for g in ("swelling", "elongation"):
        s = df[df["mode"] == g]
        print(f"  {g}: final d lnM/dt median={s['final_rate'].median():.4f} "
              f"[{s['final_rate'].min():.4f},{s['final_rate'].max():.4f}]  "
              f"ratio median={s['ratio'].median():.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(120 / 25.4, 65 / 25.4),
                             constrained_layout=True)
    _panel(axes[0], df, "final_rate", r"pre-death $d\ln M/dt$ [h$^{-1}$]")
    _panel(axes[1], df, "ratio", "final / baseline growth rate", hline=1.0)

    save_figure(
        fig,
        params={"selection": "phase1-dead mothers (curated death windows)",
                "volume_variant": "efd", "max_frame": 2018,
                "window": "last completed cycle (2nd-to-last fig_panelA row)",
                "death_modes": "elongation={Pos20_ch06,Pos30_ch04}, else swelling",
                "n_lineages": int(len(df)), "n_swelling": n_sw,
                "n_elongation": n_el, "min_fit_pts": MIN_PTS},
        description="pre-death dry-mass growth rate d ln M/dt (last completed cycle, "
                    "EFD) by death mode: swelling box+points vs 2 elongation-cascade "
                    "highlighted; panel 2 = final/baseline rate ratio",
        data={"ch": df["ch"].to_numpy(), "mode": df["mode"].to_numpy(),
              "f0": df["f0"].to_numpy(), "f1": df["f1"].to_numpy(),
              "n_pts": df["n_pts"].to_numpy(),
              "final_rate": df["final_rate"].to_numpy(),
              "baseline": df["baseline"].to_numpy(),
              "ratio": df["ratio"].to_numpy()},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
