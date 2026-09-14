"""_fig_birth_vol_vs_ri.py — birth volume vs birth mean RI (size-density at birth).

One scatter panel over the gold-standard phase1 cohort: each point is one cell
cycle's BIRTH state (the frame just after division) — x = birth volume [µm³],
y = birth mean refractive index. Shows whether cells that are born larger are
also denser/less dense at birth. Same cohort/data source as fig4/fig5
(gold_standard_phase1_homeostasis.collect_added_table), EFD-corrected geometry
(QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd).

Run:
  QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd python scripts/_fig_birth_vol_vs_ri.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from gold_standard_phase1_homeostasis import (  # noqa: E402
    collect_added_table, fit_stats, PHASE1_END_FRAME,
)
from figure_logger import save_figure  # noqa: E402

COLOR = "#CC79A7"   # Okabe-Ito purple (distinct from the vol/mass/RI panels)


def main() -> int:
    df, n_mothers = collect_added_table()
    n = len(df)
    x = df["birth_volume_um3"].to_numpy(float)
    y = df["birth_ri"].to_numpy(float)
    s = fit_stats(x, y)
    print(f"n_mothers={n_mothers} n_cycles={n}  "
          f"slope={s['slope']:.4g}±{s['slope_se']:.2g} r={s['r']:.3f} "
          f"[{s['r_lo']:.3f},{s['r_hi']:.3f}] p={s['p']:.2e}")

    fig, ax = plt.subplots(figsize=(75 / 25.4, 70 / 25.4), constrained_layout=True)
    ax.scatter(x, y, s=10, alpha=0.35, color=COLOR, edgecolor="none",
               rasterized=True)
    xline = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xline, s["slope"] * xline + (y.mean() - s["slope"] * x.mean()),
            color="#333", lw=1.0, ls="--",
            label=f"slope={s['slope']:.2g}±{s['slope_se']:.2g}\n"
                  f"r={s['r']:.2f} [{s['r_lo']:.2f},{s['r_hi']:.2f}]\n"
                  f"p={s['p']:.1e} (n={n})")
    ax.set_xlabel(r"birth volume [μm³]", fontsize=8)
    ax.set_ylabel("birth mean RI", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(loc="best", frameon=False, fontsize=6)
    ax.spines[["top", "right"]].set_visible(False)

    direction = ("denser (higher RI)" if s["slope"] > 0
                 else "less dense (lower RI)")
    caption = (
        f"Birth size–density relationship in normally dividing mother cells "
        f"(n = {n} cell cycles, {n_mothers} mothers; phase1 2% glucose growth, "
        f"gold-standard cohort, EFD-corrected geometry). Each point is one cell "
        f"cycle at its BIRTH state (the first valid frame just after division): "
        f"x = birth cell volume [µm³], y = birth mean refractive index "
        f"(dimensionless, the cell-averaged RI = dry-mass density proxy). Points "
        f"are individual cycles (no error bars); dashed line = ordinary-least-"
        f"squares fit; slope ± standard error, r with a 95% CI (Fisher z) and "
        f"two-sided p from a Pearson correlation: slope={s['slope']:.2g}±"
        f"{s['slope_se']:.2g} (µm⁻³ RI units), r={s['r']:.2f} "
        f"[95% CI {s['r_lo']:.2f}, {s['r_hi']:.2f}], p={s['p']:.1e}. "
        f"The {'positive' if s['slope'] > 0 else 'negative'} correlation means "
        f"cells born larger tend to be {direction} at birth.")

    save_figure(
        fig,
        params={"selection": "gold-standard, phase1 only",
                "x": "birth_volume_um3", "y": "birth_ri",
                "phase1_end_frame": PHASE1_END_FRAME,
                "n_mothers": int(n_mothers), "n_cycles": int(n),
                "volume_variant": "efd",
                "fit": {k: s[k] for k in ("slope", "slope_se", "r",
                                          "r_lo", "r_hi", "p")}},
        description="birth volume vs birth mean RI (size-density at birth), "
                    "gold-standard phase1, EFD-corrected, alpha scatter + "
                    "dashed regression with slope/r uncertainty",
        caption=caption,
        data={"birth_volume_um3": x, "birth_ri": y},
    )
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
