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
from overlay_gold_standard_and_phase1_dead import select_gold_standard  # noqa: E402
from gold_standard_phase1_homeostasis import (  # noqa: E402
    load_mother_cycles_csv, PHASE1_END_FRAME,
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
B2B_ADDED = {"volume": "added_volume_um3", "mass": "added_mass_pg", "ri": "added_ri"}


def within_cycle_end(m_df_sorted: pd.DataFrame, f_birth: int, f_div: int,
                     col: str) -> float | None:
    """Value at the last valid frame in [f_birth, f_div-1].

    Mirrors extract_cycle_traces: validity = ~(is_outlier | touches_border),
    and rel=1 lands on the last such frame (np.interp clamp). Returns None if
    fewer than 4 valid frames (the same guard the trace builder uses)."""
    win = m_df_sorted[(m_df_sorted["frame"] >= f_birth)
                      & (m_df_sorted["frame"] <= f_div - 1)]
    win = win[~(win["is_outlier"] | win["touches_border"])]
    if len(win) < 4:
        return None
    return float(win.iloc[-1][col])


def collect() -> tuple[pd.DataFrame, int]:
    """Pool gold-standard phase1 cycles; per cycle store birth value, the
    birth-to-birth added (fig4) and the within-cycle added (fig4b)."""
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
            ev = within_cycle_end(m_df, c["birth_frame"], c["div_frame"], "volume_um3_rod")
            em = within_cycle_end(m_df, c["birth_frame"], c["div_frame"], "mass_pg")
            er = within_cycle_end(m_df, c["birth_frame"], c["div_frame"], "mean_ri")
            if ev is None or em is None or er is None:
                continue
            sources.add(f"{pos}/{ch}")
            rows.append({
                "source": f"{pos}/{ch}",
                "birth_volume_um3": c["birth_volume_um3"],
                "birth_mass_pg":    c["birth_mass_pg"],
                "birth_ri":         c["birth_ri"],
                # birth-to-birth (fig4)
                "b2b_volume": c["added_volume_um3"],
                "b2b_mass":   c["added_mass_pg"],
                "b2b_ri":     c["added_ri"],
                # within-cycle (fig4b)
                "wc_volume":  ev - c["birth_volume_um3"],
                "wc_mass":    em - c["birth_mass_pg"],
                "wc_ri":      er - c["birth_ri"],
            })
    return pd.DataFrame(rows), len(sources)


def main():
    df, n_mothers = collect()
    n = len(df)
    print(f"n_mothers={n_mothers} n_cycles={n}")

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 65 / 25.4),
                             constrained_layout=True)
    data_out: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    for ax, (key, col, xl, yl, color) in zip(axes, PANELS):
        x = df[BIRTH_KEY[key]].to_numpy()
        y = df[f"wc_{key}"].to_numpy()
        yb = df[f"b2b_{key}"].to_numpy()

        ax.scatter(x, y, s=10, alpha=0.35, color=color, edgecolor="none",
                   rasterized=True)
        z = np.polyfit(x, y, 1)
        r, p = pearsonr(x, y)
        xline = np.linspace(float(x.min()), float(x.max()), 50)
        ax.plot(xline, np.polyval(z, xline), color="#333", lw=1.0, ls="--",
                label=f"slope={z[0]:.2g}\nr={r:.2f}, p={p:.1e}\n(n_cycles={n})")
        ax.set_xlabel(xl, fontsize=8)
        ax.set_ylabel(yl, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", frameon=False, fontsize=6)
        ax.spines[["top", "right"]].set_visible(False)

        # birth-to-birth (fig4) regression for the side-by-side comparison
        zb = np.polyfit(x, yb, 1)
        rb, pb = pearsonr(x, yb)
        print(f"{key:6s}  fig4(b2b)     slope={zb[0]:.4g} r={rb:.3f} p={pb:.2e}  | "
              f"fig4b(within) slope={z[0]:.4g} r={r:.3f} p={p:.2e}")
        summary[key] = {"b2b": (float(zb[0]), float(rb), float(pb)),
                        "within": (float(z[0]), float(r), float(p))}

        data_out[f"birth_{key}"] = x
        data_out[f"within_added_{key}"] = y
        data_out[f"b2b_added_{key}"] = yb

    save_figure(
        fig,
        params={"selection": "gold-standard, phase1 only",
                "added_definition": "within-cycle: value(div_frame-1) - value(birth)",
                "phase1_end_frame": PHASE1_END_FRAME,
                "n_mothers": int(n_mothers), "n_cycles": int(n),
                "volume_variant": "efd",
                "slopes": summary},
        description="within-cycle added vs birth (volume / dry mass / mean RI), "
                    "added = last frame before division minus birth, EFD-corrected "
                    "volume, alpha scatter + dashed regression",
        data=data_out,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
