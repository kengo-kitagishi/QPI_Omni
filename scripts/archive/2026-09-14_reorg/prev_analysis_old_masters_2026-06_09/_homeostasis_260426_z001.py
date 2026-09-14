"""Size + density homeostasis for the 260426 z001 dataset.

Walks every ch under
  H:\\260426\\online_crop_sub_zstack\\Pos*\\output_phase\\channels\\crop_sub_rawraw\\z001\\ch*
and keeps only the channels where the mother lineage (cell_id == 0)
was tracked all the way to the end of the recording (last data3D frame
>= --min-survival-frac of the max frame in that ch). For every retained
mother, extract complete cell cycles (birth -> next division) and pool
the (birth, added) pairs.

Produces two figures via figure_logger:
  * size homeostasis: birth_volume_um3 vs added_volume_um3
  * density homeostasis: birth_ri vs added_ri (= div_ri - birth_ri)
plus Pearson r, slope, intercept summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from figure_logger import save_figure
from mother_cell_cycle_stats import load_mother_cycles, extract_cycle_traces, source_tag

ROOT = Path(r"H:\260426\online_crop_sub_zstack")

OI = {
    "blue": "#0072B2", "green": "#009E73", "orange": "#E69F00",
    "vermilion": "#D55E00", "skyblue": "#56B4E9",
}


def find_z001_channels() -> list[Path]:
    """Every ch dir under Pos*/output_phase/channels/crop_sub_rawraw/z001 that
    has inference_out/lineage_out/lineage_data3D.csv."""
    chs: list[Path] = []
    for pos in sorted(ROOT.glob("Pos*"), key=lambda p: int(p.name.removeprefix("Pos"))):
        z_root = pos / "output_phase" / "channels" / "crop_sub_rawraw" / "z001"
        if not z_root.is_dir():
            continue
        for ch in sorted(z_root.glob("ch*"), key=lambda p: int(p.name.removeprefix("ch"))):
            if (ch / "inference_out" / "lineage_out" / "lineage_data3D.csv").exists():
                chs.append(ch)
    return chs


def mother_survives_to_end(ch_dir: Path, min_survival_frac: float) -> tuple[bool, int, int]:
    """Return (survived, mother_last_frame, max_frame). Mother is cell_id==0."""
    out_dir = ch_dir / "inference_out" / "lineage_out"
    try:
        d3 = pd.read_csv(out_dir / "lineage_data3D.csv")
    except Exception:
        return (False, -1, -1)
    if d3.empty:
        return (False, -1, -1)
    max_frame = int(d3["frame"].max())
    mother_rows = d3[d3["cell_id"] == 0]
    if mother_rows.empty:
        return (False, -1, max_frame)
    mother_last = int(mother_rows["frame"].max())
    survived = mother_last >= int(max_frame * min_survival_frac)
    return (survived, mother_last, max_frame)


def scatter_homeostasis(df: pd.DataFrame, xcol: str, ycol: str,
                        xlabel: str, ylabel: str, color: str,
                        title: str) -> tuple[plt.Figure, dict]:
    fig, ax = plt.subplots(figsize=(89/25.4, 80/25.4), constrained_layout=True)
    sub = df[[xcol, ycol]].dropna()
    if len(sub) < 5:
        ax.text(0.5, 0.5, f"too few cycles (n={len(sub)})",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        return fig, {"n": len(sub)}

    x = sub[xcol].to_numpy(); y = sub[ycol].to_numpy()
    ax.scatter(x, y, s=18, alpha=0.5, color=color, edgecolor="white", linewidth=0.4)

    r, p = pearsonr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 60)
    ax.plot(xline, slope * xline + intercept, color="#333", lw=0.9, ls="--",
            label=f"slope={slope:.3g}\nr={r:.2f}, p={p:.1e}\nn={len(x)}")
    ax.axhline(0, color="#000", lw=0.3, zorder=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="best", frameon=False, fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    stats = {
        "n_cycles": int(len(x)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "slope": float(slope),
        "intercept": float(intercept),
        "x_mean": float(np.mean(x)), "x_std": float(np.std(x, ddof=1)),
        "y_mean": float(np.mean(y)), "y_std": float(np.std(y, ddof=1)),
    }
    return fig, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-survival-frac", type=float, default=0.95,
                    help="keep ch only if mother's last tracked frame is >= "
                         "min-survival-frac * max_frame_in_ch (default 0.95)")
    args = ap.parse_args()

    chs = find_z001_channels()
    print(f"[info] {len(chs)} (Pos, ch) directories with lineage CSV", file=sys.stderr)

    kept: list[Path] = []
    skipped: list[tuple[str, str]] = []
    for ch_dir in chs:
        survived, mother_last, max_frame = mother_survives_to_end(ch_dir, args.min_survival_frac)
        if survived:
            kept.append(ch_dir)
        else:
            skipped.append((source_tag(ch_dir), f"mother_last={mother_last}/{max_frame}"))

    print(f"[info] kept {len(kept)}, skipped {len(skipped)}", file=sys.stderr)
    if skipped[:8]:
        for s in skipped[:8]:
            print(f"  [skip] {s[0]}: {s[1]}", file=sys.stderr)

    all_cycles: list[dict] = []
    all_traces: list[dict] = []
    n_failed_load = 0
    for ch_dir in kept:
        try:
            m_df, _clist, cycles = load_mother_cycles(ch_dir, max_frame=None)
        except Exception as e:
            print(f"  [warn] load failed for {source_tag(ch_dir)}: {e}", file=sys.stderr)
            n_failed_load += 1
            continue
        all_cycles.extend(cycles)
        try:
            traces = extract_cycle_traces(m_df, cycles)
            all_traces.extend(traces)
        except Exception as e:
            print(f"  [warn] traces failed for {source_tag(ch_dir)}: {e}", file=sys.stderr)

    print(f"[info] pooled {len(all_cycles)} cycles / {len(all_traces)} cycle traces "
          f"across {len(kept) - n_failed_load} chs", file=sys.stderr)
    if not all_cycles:
        print("[error] no cycles pooled — nothing to plot", file=sys.stderr)
        return

    df = pd.DataFrame(all_cycles)

    common_params = {
        "dataset": "260426_z001",
        "n_chs_total": len(chs),
        "n_chs_kept": len(kept),
        "n_chs_failed_load": int(n_failed_load),
        "n_cycles_total": int(len(df)),
        "min_survival_frac": args.min_survival_frac,
    }

    data_source = {"raw_files": [
        str(ch / "inference_out" / "lineage_out" / "clist.csv") for ch in kept
    ]}

    # Size homeostasis
    fig_v, stats_v = scatter_homeostasis(
        df, "birth_volume_um3", "added_volume_um3",
        r"birth volume [µm$^3$]", r"added volume [µm$^3$]",
        OI["blue"],
        f"Size homeostasis (260426 z001, n_cycles={len(df)})",
    )
    save_figure(fig_v,
                params={**common_params, **{f"vol_{k}": v for k, v in stats_v.items()}},
                description="260426 z001 size homeostasis: birth vs added volume",
                fmt="pdf", data_source=data_source)
    plt.close(fig_v)

    # Density homeostasis
    fig_d, stats_d = scatter_homeostasis(
        df, "birth_ri", "added_ri",
        "birth mean RI", "added mean RI (= div - birth)",
        OI["green"],
        f"Density homeostasis (260426 z001, n_cycles={len(df)})",
    )
    save_figure(fig_d,
                params={**common_params, **{f"ri_{k}": v for k, v in stats_d.items()}},
                description="260426 z001 density homeostasis: birth vs added mean_RI",
                fmt="pdf", data_source=data_source)
    plt.close(fig_d)

    # Cycle-aligned mean_RI (and volume / mass) trajectories — all cycles
    # warped to relative progression [0, 1], mean +/- SD overlay.
    if all_traces:
        rel = all_traces[0]["rel_progress"]
        ri_stack = np.array([t["ri"] for t in all_traces])
        vol_stack = np.array([t["volume"] for t in all_traces])
        mass_stack = np.array([t["mass"] for t in all_traces])

        def _aligned_panel(ax, stack, ylabel, color):
            mean = np.nanmean(stack, axis=0)
            sd = np.nanstd(stack, axis=0)
            for row in stack:
                ax.plot(rel, row, color=color, alpha=0.05, lw=0.3)
            ax.plot(rel, mean, color=color, lw=1.4, label=f"mean (n={len(stack)})")
            ax.fill_between(rel, mean - sd, mean + sd, color=color, alpha=0.22,
                            label=r"±1 SD")
            ax.set_ylabel(ylabel)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.legend(loc="upper left", frameon=False, fontsize=6)

        # Density-along-cycle as its own focused figure (the user's "cell cycle
        # to density" ask).
        fig_ri, ax_ri = plt.subplots(figsize=(89/25.4, 80/25.4),
                                     constrained_layout=True)
        _aligned_panel(ax_ri, ri_stack, "mean RI", OI["green"])
        ax_ri.set_xlabel("relative cell-cycle progression τ")
        ax_ri.set_title(
            f"Cycle-aligned mean RI (260426 z001, n={len(all_traces)})",
            fontsize=9,
        )
        save_figure(fig_ri,
                    params={**common_params, "n_traces": int(len(all_traces))},
                    description="260426 z001 cycle-aligned mean_RI vs relative cycle progression",
                    fmt="pdf", data_source=data_source)
        plt.close(fig_ri)

        # All three quantities together for context.
        fig_3, axes_3 = plt.subplots(3, 1, figsize=(89/25.4, 150/25.4),
                                     sharex=True, constrained_layout=True)
        _aligned_panel(axes_3[0], vol_stack, r"volume [µm$^3$]", OI["blue"])
        _aligned_panel(axes_3[1], mass_stack, "dry mass [pg]", OI["orange"])
        _aligned_panel(axes_3[2], ri_stack, "mean RI", OI["green"])
        axes_3[2].set_xlabel("relative cell-cycle progression τ")
        axes_3[0].set_title(
            f"Cycle-aligned trajectories (260426 z001, n={len(all_traces)})",
            fontsize=9,
        )
        save_figure(fig_3,
                    params={**common_params, "n_traces": int(len(all_traces))},
                    description="260426 z001 cycle-aligned volume / mass / mean_RI",
                    fmt="pdf", data_source=data_source)
        plt.close(fig_3)

    # Bonus: dry mass homeostasis if data column is available
    if "birth_mass_pg" in df.columns and df["birth_mass_pg"].notna().any():
        fig_m, stats_m = scatter_homeostasis(
            df, "birth_mass_pg", "added_mass_pg",
            "birth dry mass [pg]", "added dry mass [pg]",
            OI["orange"],
            f"Mass homeostasis (260426 z001, n_cycles={len(df)})",
        )
        save_figure(fig_m,
                    params={**common_params, **{f"mass_{k}": v for k, v in stats_m.items()}},
                    description="260426 z001 mass homeostasis: birth vs added dry mass",
                    fmt="pdf", data_source=data_source)
        plt.close(fig_m)

    print("[done] homeostasis figures saved via figure_logger", file=sys.stderr)


if __name__ == "__main__":
    main()
