"""_fig_switch_frame_check_260517.py - where does the first media switch actually bite?

Pools the mother cell (cell_id 0) of every production-tracked, cell-dense channel
around the scheduled 2% -> 0.0055% switch (img_2019) and plots, per frame,
the median mean_ri and median total_phase across channels, plus per-position
deltas, to decide the last clean 2% frame for the phase-1 dataset.

Reads the D: working tree (or the master when published). Outputs one figure via
figure_logger and prints the per-position statistics used in the caption.

Usage:
    python scripts/_fig_switch_frame_check_260517.py [--lo 2000 --hi 2040]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import qpi_paths as qp  # noqa: E402
import _retrack_260517_newmodel as chain  # noqa: E402
from figure_logger import save_figure  # noqa: E402

SWITCH = 2019  # scheduled first switch (absolute img number)


def channel_csvs():
    out = []
    for n in range(1, 105):
        pos = f"Pos{n}"
        z = chain.MASK_ROOT / pos / chain.REL
        if not z.is_dir():
            continue
        for ch in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
            if chain.count_masks(ch / "inference_out") < 1000:
                continue
            if qp.master_active():
                csv = qp.find_master_lineage_csv(pos, ch.name)
            else:
                lo = ch / "inference_out" / "lineage_out"
                csv = lo / "lineage_data3D.csv" if chain.is_production(lo) else None
            if csv is not None:
                out.append((n, ch.name, csv))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lo", type=int, default=2000)
    ap.add_argument("--hi", type=int, default=2040)
    args = ap.parse_args()
    frames = list(range(args.lo, args.hi + 1))

    recs = []
    bad_hits = {}  # frame -> number of positions where it is a drift bad frame
    seen_pos = set()
    for n, ch, csv in channel_csvs():
        df = pd.read_csv(csv, usecols=["cell_id", "frame", "total_phase", "mean_ri", "volume_um3_rod",
                                       "is_outlier", "touches_border"])
        m = df[(df.cell_id == 0) & df.frame.isin(frames) & ~(df.is_outlier | df.touches_border)]
        if len(m) < 0.5 * len(frames):
            continue
        m = m.assign(pos=n, ch=ch)
        recs.append(m)
        if n not in seen_pos:
            seen_pos.add(n)
            bj = csv.parent / "bad_frames_used.json"
            if bj.exists():
                try:
                    j = json.loads(bj.read_text(encoding="utf-8"))
                    tps = j.get(f"Pos{n}", {}).get("bad_timepoints", {})
                    for f in tps:
                        f = int(f)
                        if args.lo <= f <= args.hi:
                            bad_hits[f] = bad_hits.get(f, 0) + 1
                except Exception:
                    pass
    if not recs:
        raise SystemExit("no data")
    allm = pd.concat(recs)
    n_ch, n_pos = allm.groupby(["pos", "ch"]).ngroups, allm.pos.nunique()

    # per (pos, ch): normalise to the pre-switch baseline (median of lo..2012) so channels pool cleanly
    def _norm(g):
        base = g[g.frame <= 2012]
        g = g.copy()
        g["ri_rel"] = g.mean_ri - base.mean_ri.median()
        g["tp_rel"] = g.total_phase / base.total_phase.median() - 1.0
        return g
    allm = allm.groupby(["pos", "ch"], group_keys=False).apply(_norm)
    per_frame = allm.groupby("frame").agg(ri=("ri_rel", "median"), ri_q1=("ri_rel", lambda s: s.quantile(0.25)),
                                          ri_q3=("ri_rel", lambda s: s.quantile(0.75)),
                                          tp=("tp_rel", "median"), tp_q1=("tp_rel", lambda s: s.quantile(0.25)),
                                          tp_q3=("tp_rel", lambda s: s.quantile(0.75)), n=("cell_id", "size"))
    per_frame = per_frame.reindex(frames)

    # per-position: frame (2015..2022) with the largest |delta mean_ri| and whether 2018 is already perturbed
    per_pos = allm.groupby(["pos", "frame"]).agg(ri=("ri_rel", "median"), tp=("tp_rel", "median")).reset_index()
    jump_frame, pert2018 = {}, 0
    for p, g in per_pos.groupby("pos"):
        s = g.set_index("frame").reindex(frames)
        d = s.ri.diff().abs().loc[2015:2022]
        if d.notna().any():
            f = int(d.idxmax()); jump_frame[f] = jump_frame.get(f, 0) + 1
        if pd.notna(s.tp.get(2018)) and s.tp.get(2018) > 0.04:
            pert2018 += 1
    n_pos_eval = per_pos.pos.nunique()

    print(f"channels {n_ch}, positions {n_pos}")
    print(per_frame.loc[2012:2024, ["ri", "tp", "n"]].round(5).to_string())
    print("largest |dRI| frame per Pos (2015..2022):", dict(sorted(jump_frame.items())))
    print(f"positions with total_phase at img_2018 > +4% of baseline: {pert2018}/{n_pos_eval}")
    print("drift bad-frame hits per frame (positions):", dict(sorted(bad_hits.items())))

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.4]})
    x = np.array(frames)
    ax = axes[0]
    ax.fill_between(x, per_frame.ri_q1, per_frame.ri_q3, color="#1f77b4", alpha=0.2, lw=0, label="IQR across channels")
    ax.plot(x, per_frame.ri, "o-", ms=3, color="#1f77b4", label="median")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("mother mean RI\nminus pre-switch baseline")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    ax.fill_between(x, per_frame.tp_q1 * 100, per_frame.tp_q3 * 100, color="#d62728", alpha=0.2, lw=0)
    ax.plot(x, per_frame.tp * 100, "o-", ms=3, color="#d62728")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("mother total phase\nchange vs baseline (%)")
    ax = axes[2]
    bh = [bad_hits.get(f, 0) for f in frames]
    ax.bar(x, bh, color="#7f7f7f", width=0.8)
    ax.set_ylabel("positions with\ndrift bad frame")
    ax.set_xlabel("absolute frame (img_NNN)")
    for a in axes:
        a.axvline(SWITCH - 0.5, color="k", ls="--", lw=0.8)
        a.axvspan(2017.5, 2019.5, color="#ffcc80", alpha=0.35, lw=0)
        a.grid(alpha=0.25)
    axes[0].set_title(f"First media switch (scheduled at img_{SWITCH}): pooled mother cells, {n_ch} channels / {n_pos} positions\n"
                      f"img_2018 is already perturbed (total phase > +4% in {pert2018}/{n_pos_eval} positions); "
                      f"last clean 2% frame = img_2017", fontsize=10)
    fig.tight_layout()

    data = {"frame": x,
            "ri_rel_median": per_frame.ri.to_numpy(), "ri_rel_q1": per_frame.ri_q1.to_numpy(), "ri_rel_q3": per_frame.ri_q3.to_numpy(),
            "tp_rel_median": per_frame.tp.to_numpy(), "tp_rel_q1": per_frame.tp_q1.to_numpy(), "tp_rel_q3": per_frame.tp_q3.to_numpy(),
            "n_channels_per_frame": per_frame.n.to_numpy(), "bad_frame_positions": np.array(bh),
            "per_pos_ri_rel": per_pos.pivot(index="pos", columns="frame", values="ri").reindex(columns=frames).to_numpy(),
            "per_pos_tp_rel": per_pos.pivot(index="pos", columns="frame", values="tp").reindex(columns=frames).to_numpy(),
            "per_pos_index": per_pos.pos.unique()}
    out = save_figure(
        fig,
        params={"frames": [args.lo, args.hi], "scheduled_switch": SWITCH, "n_channels": n_ch, "n_positions": n_pos,
                "positions_perturbed_at_2018": pert2018, "largest_dRI_frame_hist": jump_frame,
                "source": "master" if qp.master_active() else "working-tree"},
        description="Empirical check of the first media-switch frame (2% -> 0.0055%) from pooled mother-cell RI and total phase",
        caption=(f"Mother-cell mean RI (top) and total phase (middle) relative to each channel's pre-switch baseline "
                 f"(median of img_{args.lo}-2012), pooled over {n_ch} cell-dense channels in {n_pos} positions; band = IQR. "
                 f"Bottom: number of positions in which the frame is a drift bad frame. The scheduled switch is img_{SWITCH} "
                 f"(dashed). Total phase already rises at img_2018 and mean RI dips at img_2019 before settling from img_2020, "
                 f"so img_2017 is the last unperturbed 2% frame."),
        data=data,
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
