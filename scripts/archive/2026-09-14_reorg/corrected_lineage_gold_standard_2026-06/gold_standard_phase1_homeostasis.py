"""Phase1 cell-cycle homeostasis for the gold-standard mother cohort.

Pools all gold-standard mothers (selected via
overlay_gold_standard_and_phase1_dead.select_gold_standard) and restricts
analysis to phase1 (frame < 2019, i.e. 2% glucose growth).

Reuses figure functions from mother_cell_cycle_stats.py:
  - fig_homeostasis            (birth vs added scatter, volume/mass/RI)
  - fig_interval_hist          (division interval histogram)
  - fig_aligned_trajectories   (cycle-aligned volume/mass/RI mean ± SD)
  - fig_ri_distribution        (mother RI histogram + Gaussian fit)
  - fig_density_homeostasis    (birth_ri vs added_ri)
  - fig_growth_rate            (d ln V / dτ and d ln M / dτ over relative cycle)

Mother trace and divisions are loaded directly from the inbox lineage
CSVs (no inference_out path needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from overlay_gold_standard_and_phase1_dead import (  # noqa: E402
    find_lineage_csv, select_gold_standard,
)
import mother_cell_cycle_stats as MCC  # noqa: E402
from figure_logger import save_figure  # noqa: E402

PHASE1_END_FRAME = 2018  # last phase1 frame (2% glucose growth)


def load_mother_cycles_csv(pos: str, ch: str,
                           max_frame: int = PHASE1_END_FRAME):
    """Mirror of mother_cell_cycle_stats.load_mother_cycles but reads
    lineage CSVs directly via find_lineage_csv (works for both
    per_channel_figures and batch_figures inbox layouts).
    Returns (m_df, cycles) restricted to frame <= max_frame.
    """
    path = find_lineage_csv(pos, ch)
    if path is None:
        return None
    clist_path = path.parent / "clist.csv"
    if not clist_path.exists():
        # batch_figures flat layout
        clist_path = path.parent / f"{pos}_{ch}__clist.csv"
        if not clist_path.exists():
            return None
    clist = pd.read_csv(clist_path)
    data3D = pd.read_csv(path)
    src = f"{pos}/{ch}"

    false_ids = MCC._find_false_births(clist, data3D)
    if false_ids:
        clist = clist[~clist["cell_id"].isin(false_ids)].copy()

    mother_rows = clist[clist["mother_id"] == -1]
    if mother_rows.empty:
        return None
    mother_id = int(mother_rows.iloc[0]["cell_id"])

    m_df = data3D[data3D["cell_id"] == mother_id].sort_values("frame").reset_index(drop=True)
    m_df = m_df.assign(source=src)

    daughters = clist[clist["mother_id"] == mother_id].sort_values("birth_frame")
    div_frames = daughters["birth_frame"].astype(int).tolist()

    m_idx = m_df.set_index("frame", drop=False)

    def _valid_row(f: int):
        if f not in m_idx.index:
            return None
        r = m_idx.loc[f]
        if isinstance(r, pd.DataFrame):
            r = r.iloc[0]
        if bool(r["is_outlier"]) or bool(r["touches_border"]):
            return None
        if float(r["mass_pg"]) < 10.0:
            return None
        return r

    cycles = []
    for i in range(len(div_frames) - 1):
        f_birth = div_frames[i]
        f_next = div_frames[i + 1]
        if f_next > max_frame:
            continue
        r_birth = _valid_row(f_birth)
        r_div = _valid_row(f_next)
        if r_birth is None or r_div is None:
            continue
        cycles.append({
            "source": src,
            "cycle_idx": i,
            "birth_frame": int(f_birth),
            "div_frame": int(f_next),
            "birth_time_h": MCC.frame_to_h(f_birth),
            "interval_h": MCC.frame_to_h(f_next - f_birth),
            "birth_epoch": "pre",  # phase1 only
            "birth_volume_um3":  float(r_birth["volume_um3_rod"]),
            "div_volume_um3":    float(r_div["volume_um3_rod"]),
            "added_volume_um3":  float(r_div["volume_um3_rod"] - r_birth["volume_um3_rod"]),
            "birth_mass_pg":     float(r_birth["mass_pg"]),
            "div_mass_pg":       float(r_div["mass_pg"]),
            "added_mass_pg":     float(r_div["mass_pg"] - r_birth["mass_pg"]),
            "birth_ri":          float(r_birth["mean_ri"]),
            "div_ri":            float(r_div["mean_ri"]),
            "added_ri":          float(r_div["mean_ri"] - r_birth["mean_ri"]),
        })

    m_df = m_df[m_df["frame"] <= max_frame].reset_index(drop=True)
    return m_df, cycles


# --- shared birth-to-birth (fig4) vs within-cycle (fig5) axis machinery ------
# Panel key -> (birth column in the cycle dict, b2b added column, m_df value col)
_PANELS = ("volume", "mass", "ri")
_BIRTH = {"volume": "birth_volume_um3", "mass": "birth_mass_pg", "ri": "birth_ri"}
_B2B = {"volume": "added_volume_um3", "mass": "added_mass_pg", "ri": "added_ri"}
_VALCOL = {"volume": "volume_um3_rod", "mass": "mass_pg", "ri": "mean_ri"}


def within_cycle_end(m_df_sorted: pd.DataFrame, f_birth: int, f_div: int,
                     col: str) -> float | None:
    """Value at the last valid frame in [f_birth, f_div-1] (peak before split).

    Mirrors extract_cycle_traces' rel=1 sample: validity = ~(is_outlier |
    touches_border); returns None with < 4 valid frames (same guard)."""
    win = m_df_sorted[(m_df_sorted["frame"] >= f_birth)
                      & (m_df_sorted["frame"] <= f_div - 1)]
    win = win[~(win["is_outlier"] | win["touches_border"])]
    if len(win) < 4:
        return None
    return float(win.iloc[-1][col])


def collect_added_table(gold=None) -> tuple[pd.DataFrame, int]:
    """Pool gold-standard phase1 cycles; per cycle store the birth value, the
    birth-to-birth added (fig4) and the within-cycle added (fig5).

    Single source of truth so fig4 and fig5 share an identical cohort and the
    same data-driven axis ranges."""
    if gold is None:
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
            ends = {k: within_cycle_end(m_df, c["birth_frame"], c["div_frame"],
                                        _VALCOL[k]) for k in _PANELS}
            if any(v is None for v in ends.values()):
                continue
            sources.add(c["source"])
            row = {"source": c["source"]}
            for k in _PANELS:
                row[_BIRTH[k]] = c[_BIRTH[k]]
                row[f"b2b_{k}"] = c[_B2B[k]]
                row[f"wc_{k}"] = ends[k] - c[_BIRTH[k]]
            rows.append(row)
    return pd.DataFrame(rows), len(sources)


def shared_axes(df: pd.DataFrame, margin: float = 0.05) -> dict[str, dict]:
    """Per-panel axes that give fig4 & fig5 the SAME y-axis SPAN (height), each
    figure framing its own data (NOT a shared absolute range / union).

    x = birth value (same quantity & cycles in both) -> common xlim.
    y-span = max(birth-to-birth data span, within-cycle data span) padded by
        ``margin`` on each side; each figure's ylim is that common span centred
        on ITS OWN data midpoint. So the vertical scale (units per length) is
        identical between the two figures while the data fills each panel.

    Returns per panel: xlim (common), ylim_b2b (fig4), ylim_wc (fig5).
    """
    out: dict[str, dict] = {}
    for k in _PANELS:
        x = df[_BIRTH[k]].to_numpy(float)
        yb = df[f"b2b_{k}"].to_numpy(float)
        yw = df[f"wc_{k}"].to_numpy(float)
        xlo, xhi = float(np.nanmin(x)), float(np.nanmax(x))
        b_lo, b_hi = float(np.nanmin(yb)), float(np.nanmax(yb))
        w_lo, w_hi = float(np.nanmin(yw)), float(np.nanmax(yw))
        span = max(b_hi - b_lo, w_hi - w_lo) * (1.0 + 2.0 * margin)

        def centred(lo: float, hi: float) -> tuple[float, float]:
            c = 0.5 * (lo + hi)
            return (c - 0.5 * span, c + 0.5 * span)

        mx = (xhi - xlo) * margin
        out[k] = {"xlim": (xlo - mx, xhi + mx),
                  "ylim_b2b": centred(b_lo, b_hi),
                  "ylim_wc": centred(w_lo, w_hi),
                  "yspan": span}
    return out


def fit_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """OLS slope with its standard error, Pearson r with a 95% CI (Fisher z),
    and p. Uncertainties are baked into the figure captions."""
    from scipy.stats import linregress
    lr = linregress(np.asarray(x, float), np.asarray(y, float))
    n = int(len(x))
    z = np.arctanh(lr.rvalue)
    se_z = 1.0 / np.sqrt(max(n - 3, 1))
    r_lo, r_hi = float(np.tanh(z - 1.96 * se_z)), float(np.tanh(z + 1.96 * se_z))
    return {"slope": float(lr.slope), "slope_se": float(lr.stderr),
            "r": float(lr.rvalue), "r_lo": r_lo, "r_hi": r_hi,
            "p": float(lr.pvalue), "n": n}


def panel_stats(df: pd.DataFrame, kind: str) -> dict[str, dict]:
    """Per-panel fit_stats for kind in {'b2b','wc'} (birth vs that added)."""
    return {k: fit_stats(df[_BIRTH[k]].to_numpy(float),
                         df[f"{kind}_{k}"].to_numpy(float)) for k in _PANELS}


def caption_stats(st: dict[str, dict]) -> str:
    """'volume: slope=A±B, r=C [95% CI D, E], p=F; mass: ...; mean RI: ...'."""
    names = {"volume": "volume", "mass": "dry mass", "ri": "mean RI"}
    parts = []
    for k in _PANELS:
        s = st[k]
        parts.append(
            f"{names[k]}: slope={s['slope']:.2g}±{s['slope_se']:.2g}, "
            f"r={s['r']:.2f} [95% CI {s['r_lo']:.2f}, {s['r_hi']:.2f}], "
            f"p={s['p']:.1e}")
    return "; ".join(parts)


def main():
    gold = select_gold_standard()
    print(f"gold-standard mothers: {len(gold)}")

    m_dfs: list[pd.DataFrame] = []
    all_cycles: list[dict] = []
    all_traces: list[dict] = []
    for pos, ch in gold:
        res = load_mother_cycles_csv(pos, ch, max_frame=PHASE1_END_FRAME)
        if res is None:
            continue
        m_df, cycles = res
        if not cycles:
            continue
        traces = MCC.extract_cycle_traces(m_df, cycles)
        m_dfs.append(m_df)
        all_cycles.extend(cycles)
        all_traces.extend(traces)
        print(f"  {pos}/{ch}: cycles={len(cycles)}, traces={len(traces)}")

    if not all_cycles:
        print("no cycles collected")
        return

    pooled = pd.concat(m_dfs, ignore_index=True)
    n_mothers = pooled["source"].nunique()
    print()
    print(f"pooled: n_mothers={n_mothers}, n_cycles={len(all_cycles)}, "
          f"n_traces={len(all_traces)}")

    params = {
        "selection": "gold-standard (interval [2.5,5.0] h, MANUAL_EXCLUDE applied)",
        "phase1_end_frame": PHASE1_END_FRAME,
        "n_mothers": int(n_mothers),
        "n_cycles": len(all_cycles),
        "n_traces": len(all_traces),
    }

    # ---- fig4: birth-to-birth homeostasis, axes shared with fig5 ----------
    df_ax, n_ax = collect_added_table(gold)
    rng = shared_axes(df_ax)                      # common y-SPAN, own centering
    xlims = [rng[k]["xlim"] for k in _PANELS]
    ylims = [rng[k]["ylim_b2b"] for k in _PANELS]
    st = panel_stats(df_ax, "b2b")
    fig4 = MCC.fig_homeostasis(all_cycles, by_quantity=True,
                               xlims=xlims, ylims=ylims)
    if fig4 is not None:
        caption = (
            f"Birth-size homeostasis across generations in normally dividing "
            f"mother cells (n = {len(df_ax)} cell cycles, {n_ax} mothers; phase1 "
            f"2% glucose growth, gold-standard cohort, EFD-corrected geometry). "
            f"Each point is one cell cycle. y-axis = the change between "
            f"consecutive post-division (birth) sizes, birth(N+1) − birth(N) "
            f"(value at the next division minus value at this birth); x-axis = "
            f"the birth size birth(N). This is a generational return map, NOT "
            f"within-cycle growth (the within-cycle adder, division − birth, is "
            f"fig5). Panels left→right: cell volume [µm³], dry mass [pg], mean RI "
            f"(dimensionless). Points are individual cycles (no error bars); "
            f"dashed line = ordinary-least-squares fit; slope ± standard error, "
            f"r with a 95% CI (Fisher z) and two-sided p from a Pearson "
            f"correlation. All three are significantly negatively correlated "
            f"({caption_stats(st)}): "
            f"larger or denser births tend to be followed by smaller, lighter "
            f"next births — birth size and density regress toward the population "
            f"mean across generations. The y-axis SPAN (height) is matched to "
            f"the within-cycle adder figure (fig5) — each figure is centred on "
            f"its own data with an identical vertical scale (units per length) — "
            f"and the x-axis (birth size, the same quantity) is common, so the "
            f"two figures can be compared directly.")
        data4 = {_BIRTH[k]: df_ax[_BIRTH[k]].to_numpy() for k in _PANELS}
        data4.update({f"b2b_added_{k}": df_ax[f"b2b_{k}"].to_numpy() for k in _PANELS})
        data4.update({f"xlim_{k}": np.array(rng[k]["xlim"]) for k in _PANELS})
        data4.update({f"ylim_{k}": np.array(rng[k]["ylim_b2b"]) for k in _PANELS})
        save_figure(
            fig4,
            params={**params, "added_definition": "birth-to-birth: value(next "
                    "division frame) - value(birth)", "volume_variant": "efd",
                    "shared_axes_with": "fig5 (_fig_added_within_cycle)",
                    "axis_margin": 0.05, "b2b_slopes": {k: st[k] for k in _PANELS}},
            description="birth-to-birth homeostasis (birth vs added volume/mass/RI), "
                        "EFD-corrected, axes shared with the within-cycle fig5",
            caption=caption, data=data4,
        )
        plt.close(fig4)

    plots = [
        (MCC.fig_interval_hist(all_cycles),
         "gold-standard phase1 division interval histogram"),
        (MCC.fig_aligned_trajectories(all_traces),
         "gold-standard phase1 cycle-aligned volume/mass/RI"),
        (MCC.fig_ri_distribution(pooled, max_frame=PHASE1_END_FRAME),
         "gold-standard phase1 RI distribution"),
        (MCC.fig_conc_distribution(pooled, max_frame=PHASE1_END_FRAME),
         "gold-standard phase1 dry-mass concentration (mg/mL) distribution"),
        (MCC.fig_density_homeostasis(all_cycles),
         "gold-standard phase1 density homeostasis"),
        (MCC.fig_growth_rate(all_traces),
         "gold-standard phase1 specific growth rate"),
    ]
    for fig, desc in plots:
        if fig is None:
            print(f"  [skip] {desc}")
            continue
        save_figure(fig, params=params, description=desc)
        plt.close(fig)

    # quick summary
    df_cycles = pd.DataFrame(all_cycles)
    print()
    print("--- phase1 summary ---")
    print(f"interval_h:     median={df_cycles['interval_h'].median():.2f}, "
          f"mean={df_cycles['interval_h'].mean():.2f}, "
          f"sd={df_cycles['interval_h'].std():.2f}")
    print(f"birth_volume:   median={df_cycles['birth_volume_um3'].median():.1f}, "
          f"mean={df_cycles['birth_volume_um3'].mean():.1f}")
    print(f"birth_mass_pg:  median={df_cycles['birth_mass_pg'].median():.2f}, "
          f"mean={df_cycles['birth_mass_pg'].mean():.2f}")
    print(f"birth_ri:       median={df_cycles['birth_ri'].median():.4f}, "
          f"mean={df_cycles['birth_ri'].mean():.4f}")


if __name__ == "__main__":
    main()
