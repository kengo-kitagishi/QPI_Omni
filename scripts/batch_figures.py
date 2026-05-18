"""
batch_figures.py — Pooled overlay plots across many channels.

Reads the lineage CSVs from N channel directories and emits three separate
single-quantity PDFs that overlay every channel's mother trajectory:

    pooled_mother_volume.pdf     one faint line per ch, thick population mean
    pooled_mother_mean_ri.pdf    same for mean RI
    pooled_mother_mass.pdf       same for dry mass (skipped if mass is all NaN)

This is the only "batch" output the default pipeline produces. The detail
analyses (homeostasis scatter, cycle-aligned, growth-rate oscillation,
survival curves, fate-colored trees) live in their own scripts and can be
invoked separately:

    mother_cell_cycle_stats.py      cycle homeostasis / aligned trajectories
    lineage_survival_analysis.py    fate classification + survival figures
    qpi_fig_03_lineage_analysis.py  per-cell-CSV cycle analyses
    qpi_fig_04_growth_oscillation.py  Liu 2020 growth-rate oscillations

Example:
    python batch_figures.py --indir /path/Pos9/ch01 --indir /path/Pos9/ch02
    python batch_figures.py --root /path/Pos9 --exclude ch10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_logger import save_figure

# Okabe-Ito-friendly defaults; one color per physical quantity.
COLORS = {
    "volume":  "#0072B2",  # blue
    "mean_ri": "#009E73",  # bluish green
    "mass":    "#E69F00",  # orange
    "mean":    "#222222",  # near-black for the population mean line
    "vline":   "#888888",  # media-switch
}


# =============================================================================
# Channel discovery
# =============================================================================
def channel_label(channel_dir: Path) -> str:
    parts = channel_dir.resolve().parts
    pos = next((p for p in parts if p.startswith("Pos")), None)
    return f"{pos}_{channel_dir.name}" if pos else channel_dir.name


def discover_channels(root: Path, exclude: list[str]) -> list[Path]:
    """Find <root>/**/ch* directories that have a usable lineage_data3D.csv."""
    hits = list(root.glob("**/ch*"))
    keep: list[Path] = []
    for p in sorted(hits):
        if any(ex in str(p) for ex in exclude):
            continue
        if (p / "inference_out" / "lineage_out" / "lineage_data3D.csv").is_file():
            keep.append(p)
    return keep


# =============================================================================
# Data loading
# =============================================================================
def load_channel_data(channel_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (full_data3D_with_source, mother_df_for_plotting, run_meta).

    The full data3D has every cell × every frame, with a `source` column added
    so multiple channels can be concatenated. The mother df is filtered to
    cell_id==0 and has outlier / border-touching frames replaced with NaN for
    plotting; it is a slice of the full data3D, not a separate read.
    """
    out = channel_dir / "inference_out" / "lineage_out"
    data3D = pd.read_csv(out / "lineage_data3D.csv")
    run_meta_path = out / "lineage_run_params.json"
    run_meta = {}
    if run_meta_path.exists():
        try:
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    src = channel_label(channel_dir)
    data3D = data3D.copy()
    data3D["source"] = src

    m = data3D[data3D["cell_id"] == 0].sort_values("frame").copy()
    if not m.empty:
        bad = m["is_outlier"].to_numpy(dtype=bool) | m["touches_border"].to_numpy(dtype=bool)
        for col in ("volume_um3_rod", "mean_ri", "mass_pg"):
            if col in m.columns:
                m.loc[bad, col] = np.nan
    return data3D, m, run_meta


def parse_schedule_str(s: Optional[str]) -> list[tuple[int, str]]:
    if not s:
        return []
    out: list[tuple[int, str]] = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok or ":" not in tok:
            continue
        f, name = tok.split(":", 1)
        try:
            out.append((int(f.strip()), name.strip()))
        except ValueError:
            continue
    out.sort(key=lambda x: x[0])
    return out


def merge_schedules(metas: list[dict]) -> tuple[list[tuple[int, str]], dict[str, float]]:
    """Use the first non-empty media_schedule + media_ri found among the run
    metadata files. All channels in one run should share a schedule; warn if
    they differ."""
    schedule: list[tuple[int, str]] = []
    media_ri: dict[str, float] = {}
    seen_strs: set[str] = set()
    for m in metas:
        sched_str = m.get("media_schedule") or ""
        if sched_str:
            seen_strs.add(sched_str)
        if not schedule and sched_str:
            schedule = parse_schedule_str(sched_str)
        ri = m.get("media_ri") or {}
        if not media_ri and ri:
            media_ri = dict(ri)
    if len(seen_strs) > 1:
        print(f"[warn] channels report different media_schedule strings: {seen_strs}",
              file=sys.stderr)
    return schedule, media_ri


# =============================================================================
# One-quantity overlay figure
# =============================================================================
def _add_media_vlines(
    ax,
    media_schedule: list[tuple[int, str]],
    media_ri: dict[str, float],
    time_interval_min: Optional[float],
) -> None:
    if not media_schedule:
        return
    to_x = (lambda f: f * (time_interval_min / 60.0)) if time_interval_min else float
    for f_switch, name in media_schedule:
        if f_switch <= 0:
            continue
        x = to_x(f_switch)
        ax.axvline(x, color=COLORS["vline"], linestyle="--", linewidth=0.6,
                   alpha=0.7, zorder=1)
        ri = media_ri.get(name, float("nan"))
        label = f"{name} (n={ri:.4f})" if np.isfinite(ri) else name
        ax.annotate(
            label, xy=(x, 1.0), xycoords=("data", "axes fraction"),
            xytext=(2, -2), textcoords="offset points",
            ha="left", va="top", fontsize=6, color="#555555",
        )


def overlay_figure(
    pooled: pd.DataFrame,
    value_col: str,
    ylabel: str,
    color: str,
    media_schedule: list[tuple[int, str]],
    media_ri: dict[str, float],
    time_interval_min: Optional[float],
    title: str,
) -> Optional[plt.Figure]:
    """One thin line per channel + thick pooled mean."""
    sub = pooled.dropna(subset=[value_col]) if value_col in pooled.columns else pd.DataFrame()
    if sub.empty:
        print(f"[warn] no usable {value_col} samples after dropping NaN; skipping figure",
              file=sys.stderr)
        return None

    fig, ax = plt.subplots(figsize=(140/25.4, 70/25.4), constrained_layout=True)
    _add_media_vlines(ax, media_schedule, media_ri, time_interval_min)

    sources = sorted(sub["source"].unique())
    x_key = "time_h" if "time_h" in sub.columns else "frame"
    for src in sources:
        s = sub[sub["source"] == src].sort_values(x_key)
        ax.plot(s[x_key].to_numpy(), s[value_col].to_numpy(),
                color=color, linewidth=0.5, alpha=0.4, zorder=3)

    # Population mean per frame (valid samples only).
    grp = sub.groupby(x_key)[value_col].mean()
    ax.plot(grp.index.to_numpy(), grp.values, color=COLORS["mean"], linewidth=1.2,
            zorder=5, label=f"mean (n={len(sources)} channels)")

    ax.set_xlabel("time [h]" if x_key == "time_h" else "frame")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", frameon=False, fontsize=7)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig


# =============================================================================
# Per-channel sidecar file list — collect the three artifacts for every ch,
# renamed with a "<source>__" prefix so 8 channels of lineage_data3D.csv
# don't overwrite each other in a single inbox directory.
# =============================================================================
def _per_channel_sidecar_files(channel_dirs: list[Path]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for ch in channel_dirs:
        src = channel_label(ch)
        lineage_out = ch / "inference_out" / "lineage_out"
        for name in ("lineage_data3D.csv", "clist.csv", "lineage_run_params.json"):
            p = lineage_out / name
            if p.exists():
                out.append((str(p), f"{src}__{name}"))
    return out


# =============================================================================
# Main
# =============================================================================
def run(channel_dirs: list[Path]) -> None:
    if not channel_dirs:
        raise SystemExit("No channel directories provided.")

    m_dfs: list[pd.DataFrame] = []
    metas: list[dict] = []
    raw_files: list[str] = []
    succeeded_chs: list[Path] = []
    for ch in channel_dirs:
        try:
            _d3_df, m_df, meta = load_channel_data(ch)
        except Exception as e:
            print(f"[warn] skip {ch}: {e}", file=sys.stderr)
            continue
        if m_df.empty:
            print(f"[warn] no mother rows in {ch}", file=sys.stderr)
            continue
        m_dfs.append(m_df)
        metas.append(meta)
        succeeded_chs.append(ch)
        raw_files.append(str(ch / "inference_out" / "lineage_out" / "lineage_data3D.csv"))
        print(f"[info] {channel_label(ch)}: {len(m_df)} mother frames", file=sys.stderr)

    if not m_dfs:
        raise SystemExit("No usable mother trajectories.")

    pooled = pd.concat(m_dfs, ignore_index=True)
    sidecar_files = _per_channel_sidecar_files(succeeded_chs)
    media_schedule, media_ri = merge_schedules(metas)
    time_interval_min = next(
        (m.get("time_interval_min") for m in metas if m.get("time_interval_min")),
        None,
    )

    params = {
        "n_channels": len(m_dfs),
        "sources": sorted(pooled["source"].unique().tolist()),
        "media_schedule": metas[0].get("media_schedule") if metas else None,
        "calibration_id": metas[0].get("calibration_id") if metas else None,
        "time_interval_min": time_interval_min,
    }
    data_source = {"raw_files": raw_files}

    panels = [
        ("volume_um3_rod", r"volume [µm$^3$]",  COLORS["volume"],
         "pooled mother volume", "pooled mother volume overlay"),
        ("mean_ri",        "mean RI",            COLORS["mean_ri"],
         "pooled mother mean RI", "pooled mother mean RI overlay"),
        ("mass_pg",        "dry mass [pg]",      COLORS["mass"],
         "pooled mother dry mass", "pooled mother dry mass overlay"),
    ]
    for col, ylab, color, title, desc in panels:
        fig = overlay_figure(pooled, col, ylab, color,
                             media_schedule, media_ri, time_interval_min,
                             title)
        if fig is None:
            continue
        # Copy every channel's lineage_data3D.csv / clist.csv /
        # lineage_run_params.json into the inbox, prefixed with the channel
        # label so an inbox folder of 8 channels doesn't collapse to one
        # overwriting file.
        save_figure(fig, params=params, description=desc, fmt="pdf",
                    data_source=data_source, copy_files=sidecar_files)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--indir", type=Path, action="append", default=[],
                   help="channel dir with inference_out/lineage_out/ (repeatable)")
    p.add_argument("--root", type=Path, default=None,
                   help="glob all <root>/**/ch* containing lineage_data3D.csv")
    p.add_argument("--exclude", action="append", default=[],
                   help="substring to skip when discovering channels (repeatable)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    channel_dirs = list(args.indir)
    if args.root is not None:
        channel_dirs.extend(discover_channels(args.root, args.exclude))
    # de-duplicate while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for c in channel_dirs:
        rp = c.resolve()
        if rp not in seen:
            seen.add(rp)
            uniq.append(c)
    if not uniq:
        raise SystemExit("No channels found (provide --indir or --root).")
    run(uniq)
    return 0


if __name__ == "__main__":
    sys.exit(main())
