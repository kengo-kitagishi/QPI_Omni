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

    plots = [
        (MCC.fig_homeostasis(all_cycles),
         "gold-standard phase1 homeostasis (birth vs added volume/mass/RI)"),
        (MCC.fig_interval_hist(all_cycles),
         "gold-standard phase1 division interval histogram"),
        (MCC.fig_aligned_trajectories(all_traces),
         "gold-standard phase1 cycle-aligned volume/mass/RI"),
        (MCC.fig_ri_distribution(pooled, max_frame=PHASE1_END_FRAME),
         "gold-standard phase1 RI distribution"),
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
