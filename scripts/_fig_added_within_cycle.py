"""_fig_added_within_cycle.py — within-cycle added vs birth (volume / dry mass / mean RI).

Companion to the birth-to-birth homeostasis scatter (fig4). The birth-to-birth
"added" is value(next division frame) - value(birth). This figure instead uses
the WITHIN-cycle added:

    added_within = value(last frame before division, i.e. div_frame-1) - value(birth)

The last-frame-before-division value is the cell's peak just prior to splitting,
so this isolates true growth accumulated over the cycle from the post-division
drop. The within-cycle endpoint reproduces extract_cycle_traces' rel=1 sample
(np.interp clamps to the last valid frame in [birth_frame, div_frame-1]).

EFD-volume aware: with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd the lineage
resolves (via find_lineage_csv -> qpi_paths.resolve_lineage_csv) to
results/260517/corrected_lineage_efd/<pos>_<ch>/. There the STANDARD columns
(volume_um3_rod / mean_ri / mass_pg) already hold the EFD-variant values, so no
column renaming is needed -- the same source fig2/fig4 use under those env vars.

Cohort and cycle extraction are identical to fig4
(gold_standard_phase1_homeostasis.load_mother_cycles_csv), so n matches: with
EFD this is n_mothers=26, n_cycles=1306.
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
from gold_standard_phase1_homeostasis import (  # noqa: E402
    collect_added_table, shared_axes, fit_stats, caption_stats, PHASE1_END_FRAME,
)
from figure_logger import save_figure  # noqa: E402

# (key, value column in m_df, xlabel, ylabel, color)
PANELS = [
    ("volume", "volume_um3_rod", r"birth volume [μm³]",
     r"within-cycle added volume [μm³]", "#0072B2"),
    ("mass",   "mass_pg",        "birth dry mass [pg]",
     "within-cycle added dry mass [pg]", "#D55E00"),
    ("ri",     "mean_ri",        "birth mean RI",
     "within-cycle Δ mean RI",            "#009E73"),
]
BIRTH_KEY = {"volume": "birth_volume_um3", "mass": "birth_mass_pg", "ri": "birth_ri"}


def main():
    df, n_mothers = collect_added_table()      # shared cohort with fig4
    n = len(df)
    print(f"n_mothers={n_mothers} n_cycles={n}")
    rng = shared_axes(df)                       # union(fig4 b2b, fig5 wc) + 5%

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 65 / 25.4),
                             constrained_layout=True)
    data_out: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    wstats: dict[str, dict] = {}
    for ax, (key, col, xl, yl, color) in zip(axes, PANELS):
        x = df[BIRTH_KEY[key]].to_numpy()
        y = df[f"wc_{key}"].to_numpy()
        yb = df[f"b2b_{key}"].to_numpy()

        ax.scatter(x, y, s=10, alpha=0.35, color=color, edgecolor="none",
                   rasterized=True)
        sw = fit_stats(x, y)                     # within-cycle, with uncertainty
        wstats[key] = sw
        xline = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xline, sw["slope"] * xline + (y.mean() - sw["slope"] * x.mean()),
                color="#333", lw=1.0, ls="--",
                label=f"slope={sw['slope']:.2g}±{sw['slope_se']:.2g}\n"
                      f"r={sw['r']:.2f}, p={sw['p']:.1e}\n(n_cycles={n})")
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", frameon=False, fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(*rng[key]["xlim"])          # common x with fig4
        ax.set_ylim(*rng[key]["ylim_wc"])       # same y-SPAN as fig4, own centering

        sb = fit_stats(x, yb)                     # birth-to-birth, for comparison
        print(f"{key:6s}  fig4(b2b)     slope={sb['slope']:.4g}±{sb['slope_se']:.2g} "
              f"r={sb['r']:.3f}  | fig5(within) slope={sw['slope']:.4g}±"
              f"{sw['slope_se']:.2g} r={sw['r']:.3f}")
        summary[key] = {"b2b": sb, "within": sw}

        data_out[f"birth_{key}"] = x
        data_out[f"within_added_{key}"] = y
        data_out[f"b2b_added_{key}"] = yb
        data_out[f"xlim_{key}"] = np.array(rng[key]["xlim"])
        data_out[f"ylim_{key}"] = np.array(rng[key]["ylim_wc"])
    caption = (
        f"Within-cycle size addition in normally dividing mother cells "
        f"(n = {n} cell cycles, {n_mothers} mothers; phase1 2% glucose growth, "
        f"gold-standard cohort, EFD-corrected geometry). Each point is one cell "
        f"cycle. y-axis = within-cycle added = value at the last frame before "
        f"division (the peak just before splitting) − value at birth; x-axis = "
        f"the birth size. This is the classic within-cycle adder (growth "
        f"accumulated over one cycle), the companion to the birth-to-birth "
        f"generational return map (fig4). Panels left→right: cell volume [µm³], "
        f"dry mass [pg], mean RI (dimensionless). Points are individual cycles "
        f"(no error bars); dashed line = ordinary-least-squares fit; slope ± "
        f"standard error, r with a 95% CI (Fisher z) and two-sided p from a "
        f"Pearson correlation ({caption_stats(wstats)}). "
        f"The y-axis SPAN (height) is matched to the birth-to-birth figure "
        f"(fig4) — each figure is centred on its own data with an identical "
        f"vertical scale (units per length) — and the x-axis (birth size) is "
        f"common, so the two figures can be compared directly.")

    save_figure(
        fig,
        params={"selection": "gold-standard, phase1 only",
                "added_definition": "within-cycle: value(div_frame-1) - value(birth)",
                "phase1_end_frame": PHASE1_END_FRAME,
                "n_mothers": int(n_mothers), "n_cycles": int(n),
                "volume_variant": "efd",
                "shared_axes_with": "fig4 (gold_standard_phase1_homeostasis)",
                "axis_margin": 0.05, "slopes": summary},
        description="within-cycle added vs birth (volume / dry mass / mean RI), "
                    "added = last frame before division minus birth, EFD-corrected "
                    "volume, alpha scatter + dashed regression; axes shared with fig4",
        caption=caption, data=data_out,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
