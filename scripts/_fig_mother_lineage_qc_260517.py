"""_fig_mother_lineage_qc_260517.py - QC figure: is the mother-cell lineage intact?

One row per cell-bearing channel of the selected positions. Each row shows the
mother cell (in-tree root, cell_id 0) rod volume against time with

    black ticks (top)     divisions of the mother (birth of a direct daughter)
    red dots (top)        mother rows hidden by the tracker (is_outlier | touches_border)
    orange ticks (bottom) frames where the mother has no row at all (lost / no cells)
    grey bands            drift bad frames excluded before linking (lineage_bad_frames.csv)
    background            media epochs (2% white, 0.0055% yellow, 0% blue)

and a per-row summary: divisions, hidden rows, coverage, longest gap.

Data source: the active master via qpi_paths (default). While the re-track chain
is still running (no master published yet) it falls back to the D: working tree
and says so in the figure title.

Usage:
    python scripts/_fig_mother_lineage_qc_260517.py --pos 3 4 6
    python scripts/_fig_mother_lineage_qc_260517.py --pos 1 2 3 --min-mother-frames 1000
"""
from __future__ import annotations

import argparse
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

EPOCH_COLORS = {"wo_2": "#ffffff", "wo_0p0055": "#fff3c4", "wo_0": "#dce9f7"}
EPOCH_LABELS = {"wo_2": "2%", "wo_0p0055": "0.0055%", "wo_0": "0%"}


def _epochs(dt_min: float, frame_min: int, last_frame: int):
    """[(t_start_h, t_end_h, medium)] from the chain's media schedule."""
    sched = []
    for tok in chain.MEDIA_SCHEDULE.split(","):
        f, name = tok.split(":")
        sched.append((int(f), name))
    out = []
    for i, (f0, name) in enumerate(sched):
        f1 = sched[i + 1][0] if i + 1 < len(sched) else last_frame + 1
        t0 = (max(f0, frame_min) - frame_min) * dt_min / 60.0
        t1 = (f1 - frame_min) * dt_min / 60.0
        out.append((t0, t1, name))
    return out


def _lineage_csv(pos: str, ch: str) -> tuple[Path | None, str]:
    """Master if one is published, else the D: working tree. Never the figure-hub
    inbox: that holds the June 2026 old-model results and must not be QC'd here."""
    if qp.master_active():
        p = qp.find_master_lineage_csv(pos, ch)
        return (p, "master") if p is not None else (None, "none")
    lo = chain.MASK_ROOT / pos / chain.REL / ch / "inference_out" / "lineage_out"
    if chain.is_production(lo):
        return lo / "lineage_data3D.csv", "working-tree"
    return None, "none"


def channel_rows(pos_list: list[int], min_mother_frames: int):
    rows = []
    sources = set()
    for n in pos_list:
        pos = f"Pos{n}"
        z = chain.MASK_ROOT / pos / chain.REL
        if not z.is_dir():
            continue
        for chd in sorted(p for p in z.iterdir() if p.is_dir() and p.name.startswith("ch")):
            ch = chd.name
            csv, src = _lineage_csv(pos, ch)
            if csv is None:
                continue
            df = pd.read_csv(csv)
            if df.empty:
                continue
            roots = df[(df["parent_id"] == -1) & (df["in_tree"] == True)]["cell_id"].unique()
            if len(roots) == 0:
                continue
            mid = int(roots[0])
            m = df[df["cell_id"] == mid].sort_values("frame")
            if len(m) < min_mother_frames:
                continue
            sources.add(src)
            bad_csv = csv.parent / "lineage_bad_frames.csv"
            bad_frames = np.array([], dtype=int)
            if bad_csv.exists():
                try:
                    b = pd.read_csv(bad_csv)
                    if len(b):
                        bad_frames = np.unique(b["frame"].to_numpy(dtype=int))
                except Exception:
                    pass
            rows.append(dict(pos=pos, ch=ch, mother_id=mid, df=df, m=m, bad_frames=bad_frames, source=src))
    return rows, sources


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=int, nargs="+", default=[3, 4, 6])
    ap.add_argument("--min-mother-frames", type=int, default=500,
                    help="skip channels whose mother has fewer rows (empty / transient traps)")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    dt_min = float(chain.DT_MIN)
    frame_min = chain.FRAME_MIN
    rows, sources = channel_rows(args.pos, args.min_mother_frames)
    if not rows:
        raise SystemExit("no channels to plot")
    src_label = "master" if sources == {"master"} else "+".join(sorted(sources))

    n = len(rows)
    fig_h = 0.95 * n + 1.6
    fig, axes = plt.subplots(n, 1, figsize=(16, fig_h), sharex=True)
    axes = np.atleast_1d(axes)
    data: dict[str, np.ndarray] = {}
    summary = []

    last_frame_global = max(int(r["df"]["frame"].max()) for r in rows)
    epochs = _epochs(dt_min, frame_min, last_frame_global)

    for ax, r in zip(axes, rows):
        m, df = r["m"], r["df"]
        key = f"{r['pos']}_{r['ch']}"
        t = m["time_h"].to_numpy(dtype=float)
        v = m["volume_um3_rod"].to_numpy(dtype=float)
        hidden = (m["is_outlier"].to_numpy(dtype=bool) | m["touches_border"].to_numpy(dtype=bool))
        frames = m["frame"].to_numpy(dtype=int)
        first, last = int(df["frame"].min()), int(df["frame"].max())
        all_frames = np.arange(first, last + 1)
        present = np.isin(all_frames, frames)
        badmask = np.isin(all_frames, r["bad_frames"])
        missing = all_frames[~present & ~badmask]
        t_missing = (missing - frame_min) * dt_min / 60.0
        t_bad = (r["bad_frames"] - frame_min) * dt_min / 60.0
        # divisions of the mother = birth frames of its direct daughters
        d = df[df["parent_id"] == r["mother_id"]].groupby("cell_id")["birth_frame"].first()
        div_frames = np.sort(d.to_numpy(dtype=int))
        t_div = (div_frames - frame_min) * dt_min / 60.0
        # longest gap between consecutive mother rows (h)
        gaps = np.diff(frames) if len(frames) > 1 else np.array([0])
        max_gap_h = float(gaps.max()) * dt_min / 60.0 if len(gaps) else 0.0
        coverage = len(frames) / max(1, len(all_frames))

        ymax = np.nanpercentile(v, 99.5) * 1.15 if np.isfinite(v).any() else 1.0
        for t0, t1, name in epochs:
            ax.axvspan(t0, t1, color=EPOCH_COLORS.get(name, "#ffffff"), lw=0, zorder=0)
        if len(t_bad):
            for tb in t_bad:
                ax.axvspan(tb - dt_min / 120.0, tb + dt_min / 120.0, color="#9e9e9e", alpha=0.45, lw=0, zorder=1)
        ax.plot(t, v, color="#1f4e79", lw=0.6, zorder=3)
        if hidden.any():
            ax.plot(t[hidden], np.full(hidden.sum(), ymax * 0.93), "o", ms=2.2, color="#d62728", zorder=5)
        if len(t_div):
            ax.vlines(t_div, ymax * 0.80, ymax * 0.99, color="black", lw=0.6, zorder=4)
        if len(t_missing):
            ax.vlines(t_missing, 0, ymax * 0.10, color="#ff7f0e", lw=0.6, zorder=4)
        ax.set_ylim(0, ymax)
        ax.set_ylabel(f"{r['pos']} {r['ch']}\nV (um$^3$)", fontsize=8, rotation=0, ha="right", va="center", labelpad=28)
        ax.tick_params(labelsize=7)
        txt = (f"div {len(div_frames)} | hidden {int(hidden.sum())} | missing {len(missing)} | "
               f"bad {len(r['bad_frames'])} | cover {coverage*100:.1f}% | max gap {max_gap_h:.2f} h")
        ax.text(0.995, 0.92, txt, transform=ax.transAxes, ha="right", va="top", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.5), zorder=6)
        for k, arr in (("time_h", t), ("volume_um3_rod", v), ("hidden", hidden.astype(np.uint8)),
                       ("div_time_h", t_div), ("missing_time_h", t_missing), ("bad_time_h", t_bad)):
            data[f"{key}_{k}"] = np.asarray(arr)
        summary.append(dict(pos=r["pos"], ch=r["ch"], mother_id=r["mother_id"], n_div=len(div_frames),
                            n_hidden=int(hidden.sum()), n_missing=len(missing), n_bad=len(r["bad_frames"]),
                            coverage=round(coverage, 4), max_gap_h=round(max_gap_h, 3), source=r["source"]))

    axes[-1].set_xlabel("time (h)  [img_2 = 0 h; 5 min/frame]", fontsize=9)
    axes[-1].set_xlim(0, (last_frame_global - frame_min) * dt_min / 60.0)
    ep_txt = "  ".join(f"{EPOCH_LABELS[nm]}: {t0:.0f}-{t1:.0f} h" for t0, t1, nm in epochs)
    fig.suptitle(f"Mother-cell lineage QC - 260517 new-model tracking ({src_label}) - "
                 f"Pos {', '.join(map(str, args.pos))}, {n} cell-bearing channels\n"
                 f"black tick = mother division, red dot = hidden row (outlier/border), orange tick = mother missing, "
                 f"grey band = drift bad frame; epochs {ep_txt}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    df_sum = pd.DataFrame(summary)
    print(df_sum.to_string(index=False))
    out = save_figure(
        fig,
        params={"pos": args.pos, "min_mother_frames": args.min_mother_frames,
                "media_schedule": chain.MEDIA_SCHEDULE, "frame_min": frame_min,
                "source": src_label, "n_channels": n},
        description="Mother-cell lineage QC per channel (volume trace, divisions, hidden/missing/bad frames) "
                    "for the 260517 new-model tracking",
        caption=("Mother cell (in-tree root) rod volume versus time for every cell-bearing channel of the "
                 f"selected positions ({src_label} data). Black ticks mark mother divisions, red dots rows hidden "
                 "by the tracker (area-rule outlier or border contact), orange ticks frames without a mother row, "
                 "grey bands drift bad frames excluded before linking; background shading gives the glucose epochs."),
        data=data,
        extra_meta={"summary": summary},
        dpi=args.dpi,
    )
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
