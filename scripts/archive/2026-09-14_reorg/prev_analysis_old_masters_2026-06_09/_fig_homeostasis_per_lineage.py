"""_fig_homeostasis_per_lineage.py — per-lineage fig4/fig5 homeostasis scatters.

For ~10 individual gold-standard mother lineages (not the pooled cohort), emit
the birth-to-birth (fig4) and within-cycle (fig5) added-vs-birth scatters as
SEPARATE figures (so 10 lineages -> 20 figures). All 10 lineages share a COMMON
set of axes per panel (computed from the pooled selected lineages) so slopes and
spreads are directly comparable across lineages; fig4 and fig5 share the same
y-axis SPAN (height), each centred on its own added quantity.

Lineage selection: the gold cohort sorted by cycle count, greedily spread across
positions (at most one channel per Pos) until 10 are chosen.

EFD-corrected geometry: run with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd.

Run:
  QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd \
      python scripts/_fig_homeostasis_per_lineage.py
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
    collect_added_table, shared_axes, fit_stats, _PANELS, _BIRTH, PHASE1_END_FRAME,
)
from figure_logger import save_figure  # noqa: E402

N_LINEAGES = 10
PANEL_COLOR = {"volume": "#0072B2", "mass": "#D55E00", "ri": "#009E73"}
XLAB = {"volume": r"birth volume [μm³]", "mass": "birth dry mass [pg]",
        "ri": "birth mean RI"}
YLAB = {
    "b2b": {"volume": r"birth-to-birth Δvolume [μm³]",
            "mass": "birth-to-birth Δdry mass [pg]", "ri": "birth-to-birth Δ mean RI"},
    "wc": {"volume": r"within-cycle added volume [μm³]",
           "mass": "within-cycle added dry mass [pg]", "ri": "within-cycle Δ mean RI"},
}
KIND_NAME = {"b2b": "birth-to-birth", "wc": "within-cycle"}


def pick_lineages(df, n: int) -> list[str]:
    """Sources sorted by cycle count, greedily spread across positions (<=1 per
    Pos) until n are chosen."""
    counts = df["source"].value_counts()              # source -> n_cycles
    chosen, used_pos = [], set()
    for src in counts.index:
        pos = src.split("/")[0]
        if pos in used_pos:
            continue
        chosen.append(src)
        used_pos.add(pos)
        if len(chosen) >= n:
            break
    return chosen


def panel_fig(df_lin, kind: str, xlims: dict, ylims: dict, label: str):
    """One 3-panel scatter (volume/mass/RI) of birth vs the `kind` added, with a
    regression line + slope±SE / r[95% CI] / p, fixed to the shared axes."""
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 65 / 25.4),
                             constrained_layout=True)
    stats, data = {}, {}
    for ax, k in zip(axes, _PANELS):
        x = df_lin[_BIRTH[k]].to_numpy(float)
        y = df_lin[f"{kind}_{k}"].to_numpy(float)
        s = fit_stats(x, y)
        stats[k] = s
        ax.scatter(x, y, s=12, alpha=0.5, color=PANEL_COLOR[k],
                   edgecolor="none", rasterized=True)
        xl = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xl, s["slope"] * xl + (y.mean() - s["slope"] * x.mean()),
                color="#333", lw=1.0, ls="--",
                label=f"slope={s['slope']:.2g}±{s['slope_se']:.2g}\n"
                      f"r={s['r']:.2f}, p={s['p']:.1e}\n(n_cycles={len(x)})")
        ax.axhline(0, color="#000", lw=0.3, zorder=0)
        ax.set_xlabel(XLAB[k], fontsize=8)
        ax.set_ylabel(YLAB[kind][k], fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", frameon=False, fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(*xlims[k])
        ax.set_ylim(*ylims[k])
        data[f"birth_{k}"] = x
        data[f"{kind}_added_{k}"] = y
        data[f"xlim_{k}"] = np.array(xlims[k])
        data[f"ylim_{k}"] = np.array(ylims[k])
    axes[1].set_title(f"{label}  ({KIND_NAME[kind]})", fontsize=9)
    return fig, stats, data


def caption_for(label, kind, n, stats, n_lineages):
    names = {"volume": "volume", "mass": "dry mass", "ri": "mean RI"}
    parts = "; ".join(
        f"{names[k]}: slope={stats[k]['slope']:.2g}±{stats[k]['slope_se']:.2g}, "
        f"r={stats[k]['r']:.2f} [95% CI {stats[k]['r_lo']:.2f}, "
        f"{stats[k]['r_hi']:.2f}], p={stats[k]['p']:.1e}" for k in _PANELS)
    if kind == "b2b":
        what = ("the change between consecutive post-division (birth) sizes, "
                "birth(N+1) − birth(N), a generational return map")
    else:
        what = ("the within-cycle added size, value at the last frame before "
                "division minus value at birth (growth over one cycle)")
    return (
        f"Single-lineage cell-cycle homeostasis for mother {label} "
        f"(n = {n} cell cycles; phase1 2% glucose growth, gold-standard, "
        f"EFD-corrected geometry). Each point is one cell cycle of this one "
        f"mother. y-axis = {what}; x-axis = birth size birth(N). Panels "
        f"left→right: cell volume [µm³], dry mass [pg], mean RI (dimensionless). "
        f"Points are individual cycles (no error bars); dashed line = "
        f"ordinary-least-squares fit; slope ± standard error, r with a 95% CI "
        f"(Fisher z) and two-sided p from a Pearson correlation ({parts}). "
        f"Axes (x and y span) are common across the {n_lineages} per-lineage "
        f"figures and the y-axis span is shared between the birth-to-birth (fig4) "
        f"and within-cycle (fig5) versions, so lineages and the two definitions "
        f"can be compared directly.")


def main() -> int:
    df_all, _ = collect_added_table()
    sources = pick_lineages(df_all, N_LINEAGES)
    print(f"selected {len(sources)} lineages:")
    for s in sources:
        print(f"  {s}  n_cycles={int((df_all['source'] == s).sum())}")

    pooled = df_all[df_all["source"].isin(sources)]
    rng = shared_axes(pooled)                       # common axes over the 10
    xlims = {k: rng[k]["xlim"] for k in _PANELS}
    yl_b2b = {k: rng[k]["ylim_b2b"] for k in _PANELS}
    yl_wc = {k: rng[k]["ylim_wc"] for k in _PANELS}

    for src in sources:
        df_lin = df_all[df_all["source"] == src]
        n = len(df_lin)
        label = src.replace("/", "_")
        for kind, ylims in (("b2b", yl_b2b), ("wc", yl_wc)):
            fig, stats, data = panel_fig(df_lin, kind, xlims, ylims, src)
            save_figure(
                fig,
                params={"lineage": src, "kind": KIND_NAME[kind],
                        "n_cycles": int(n), "volume_variant": "efd",
                        "phase1_end_frame": PHASE1_END_FRAME,
                        "common_axes_over": sources,
                        "fit": {k: stats[k] for k in _PANELS}},
                description=f"{label} {KIND_NAME[kind]} homeostasis "
                            f"(birth vs added volume/mass/RI), per-lineage, "
                            f"common axes, EFD",
                caption=caption_for(src, kind, n, stats, len(sources)),
                data=data,
            )
            plt.close(fig)
        print(f"  done {src}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
