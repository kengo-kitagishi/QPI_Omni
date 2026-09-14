"""
npz_to_lineage_csv.py — figure-hub inbox に保存された *_data.npz を読み込み、
チャネルごとの CSV に展開して lineage 解析 (qpi_fig_03_lineage_analysis.py) で
使える形に変換する。

npz 構造（batch_volume_trace_overlay.series_to_npz_dict 由来）:
    n_series : int
    label_i, frame_index_i, volume_um3_rod_i, mean_ri_i, mass_pg_i  (i=0..n-1)

Usage:
    python npz_to_lineage_csv.py path/to/*_data.npz
    python npz_to_lineage_csv.py path/to/file.npz --out-dir results/lineage_csv/my_run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from batch_volume_trace_overlay import load_series_from_npz


def npz_to_csvs(npz_path: Path, out_dir: Path) -> list[Path]:
    """Convert one npz to per-series CSVs. Returns list of written paths."""
    series, meta = load_series_from_npz(npz_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    time_interval_min = meta.get("time_interval_min")
    written: list[Path] = []

    for label, df in series:
        # Add convenience columns expected by lineage analysis
        out_df = df.copy()
        if time_interval_min is not None:
            out_df["time_h"] = out_df["frame_index"] * (time_interval_min / 60.0)

        # Sanitize label for filename
        safe = label.replace("/", "_").replace("\\", "_")
        csv_path = out_dir / f"{safe}.csv"
        out_df.to_csv(csv_path, index=False)
        written.append(csv_path)
        n_valid = int(np.isfinite(out_df["volume_um3_rod"]).sum())
        print(f"  -> {csv_path.name}  ({len(out_df)} rows, {n_valid} valid volume)")

    if time_interval_min is not None:
        print(f"  time_interval_min = {time_interval_min}")

    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("npz", nargs="+", type=Path, help="*_data.npz file(s) to convert")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path(__file__).resolve().parent / "results" / "lineage_csv",
        help="Output root directory (default: scripts/results/lineage_csv)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Explicit output directory (overrides --out-root).",
    )
    args = ap.parse_args()

    for npz_path in args.npz:
        if not npz_path.exists():
            print(f"ERROR: not found: {npz_path}", file=sys.stderr)
            continue
        if args.out_dir is not None:
            out_dir = args.out_dir
        else:
            # Use the npz stem (sans _data suffix) as the subdir name
            stem = npz_path.stem
            if stem.endswith("_data"):
                stem = stem[:-5]
            out_dir = args.out_root / stem

        print(f"\n[npz_to_lineage_csv] {npz_path.name}")
        print(f"  out: {out_dir}")
        npz_to_csvs(npz_path, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
