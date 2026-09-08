"""_fig_ri_vs_period.py — cycle-mean RI vs cell-cycle period (gold-standard phase1).

Single-panel scatter: x = cell-cycle period (interval_h), y = cycle-mean RI, over
all gold-standard phase1 mother cycles (n_mothers=26, n_cycles=1306), EFD volume
variant. Tests whether a normally dividing mother's mean density relates to how
long its cycle lasts.

Only metric (1) cycle-mean RI is computed here. It uses the per-frame *scalar*
mean_ri (total integrated phase / volume; no per-pixel division), so it needs no
thickness/Z map and carries no edge-thickness pathology. Metrics (2) cycle-max RI
and (3) high-density area both require a per-pixel RI map whose modeled thickness
goes to 0 at the cell rim (RI over-estimated there); how to handle that is not yet
decided, so they are deferred (see the plan / future work).

EFD-aware: with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd the lineage resolves to
results/260517/corrected_lineage_efd/<pos>_<ch>/, whose standard mean_ri column
holds the EFD-variant value (same source as fig2/fig4/fig4b).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import select_gold_standard  # noqa: E402
from gold_standard_phase1_homeostasis import (  # noqa: E402
    load_mother_cycles_csv, PHASE1_END_FRAME,
)
from figure_logger import save_figure  # noqa: E402

RI_COLOR = "#009E73"  # OI green, matching mean-RI panels in fig2/fig4/fig4b


def collect() -> tuple[pd.DataFrame, int]:
    """Per gold-standard phase1 cycle: period (interval_h) + cycle-mean RI.

    cycle-mean RI = mean of per-frame mean_ri over the cycle window
    [birth_frame, div_frame-1] (same window as extract_cycle_traces), restricted
    to valid frames ~(is_outlier | touches_border); >=4 valid frames required so n
    matches fig4/fig4b (1306)."""
    gold = select_gold_standard()
    rows: list[dict] = []
    sources: set[str] = set()
    for pos, ch in gold:
        res = load_mother_cycles_csv(pos, ch, max_frame=PHASE1_END_FRAME)
        if res is None:
            continue
        m_df, cycles = res
        if not cycles:
            continue
        m_df = m_df.sort_values("frame")
        for c in cycles:
            f0, f1 = c["birth_frame"], c["div_frame"]
            win = m_df[(m_df["frame"] >= f0) & (m_df["frame"] <= f1 - 1)]
            win = win[~(win["is_outlier"] | win["touches_border"])]
            if len(win) < 4:
                continue
            sources.add(f"{pos}/{ch}")
            rows.append({
                "source": f"{pos}/{ch}",
                "interval_h": float(c["interval_h"]),
                "cycle_mean_ri": float(win["mean_ri"].mean()),
                "birth_ri": float(c["birth_ri"]),  # RI at the birth frame
                "n_frames": int(len(win)),
                "birth_frame": int(f0),
                "div_frame": int(f1),
            })
    return pd.DataFrame(rows), len(sources)


METRICS = {
    "cycle_mean": ("cycle_mean_ri", "cycle-mean RI",
                   "cycle-mean RI = mean(per-frame mean_ri over valid cycle frames)"),
    "birth_ri":   ("birth_ri", "birth RI",
                   "birth RI = mean_ri at the birth frame of each cycle"),
}


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metric", default="cycle_mean", choices=list(METRICS),
                    help="x-axis RI metric: cycle_mean (default) or birth_ri")
    args = ap.parse_args()
    col, label, metric_desc = METRICS[args.metric]

    df, n_mothers = collect()
    n = len(df)
    # x = chosen RI metric, y = cell-cycle period (interval_h). Pearson r/p are
    # symmetric in x/y; only the regression slope's orientation changes.
    x = df[col].to_numpy()
    y = df["interval_h"].to_numpy()
    r, p = pearsonr(x, y)
    z = np.polyfit(x, y, 1)
    print(f"n_mothers={n_mothers} n_cycles={n}")
    print(f"period vs {label}: slope={z[0]:.4g} r={r:.3f} p={p:.2e}")

    fig, ax = plt.subplots(figsize=(90 / 25.4, 78 / 25.4), constrained_layout=True)
    ax.scatter(x, y, s=10, alpha=0.35, color=RI_COLOR, edgecolor="none",
               rasterized=True)
    xline = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xline, np.polyval(z, xline), color="#333", lw=1.0, ls="--",
            label=f"slope={z[0]:.2g}\nr={r:.2f}, p={p:.1e}\n(n_cycles={n})")
    ax.set_xlabel(label, fontsize=8)
    ax.set_ylabel("cell-cycle period [h]", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="best", frameon=False, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)

    save_figure(
        fig,
        params={"selection": "gold-standard, phase1 only",
                "volume_variant": "efd",
                "metric": metric_desc,
                "valid_frame_rule": "~(is_outlier | touches_border), >=4 frames",
                "phase1_end_frame": PHASE1_END_FRAME,
                "n_mothers": int(n_mothers), "n_cycles": int(n),
                "pearson_r": float(r), "pearson_p": float(p),
                "slope": float(z[0])},
        description=f"cell-cycle period (interval_h, y) vs {label} (x), gold-standard "
                    "phase1 mother cycles, EFD volume variant, Pearson r/p + regression",
        data={"interval_h": y, "cycle_mean_ri": df["cycle_mean_ri"].to_numpy(),
              "birth_ri": df["birth_ri"].to_numpy(),
              "n_frames": df["n_frames"].to_numpy(),
              "birth_frame": df["birth_frame"].to_numpy(),
              "div_frame": df["div_frame"].to_numpy(),
              "source": df["source"].to_numpy()},
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
