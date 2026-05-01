"""
Batch pipeline: segmentation -> lineage tracker -> per-cell CSV split -> qpi_fig_03/04
for every ch* directory under a root.

Usage:
  python batch_all_channels.py
  python batch_all_channels.py --channels ch01 ch02
  python batch_all_channels.py --root "G:/マイドライブ" --run-label 2026-05-01_d20
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

PYTHON = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
SCRIPTS_DIR = Path(__file__).resolve().parent
LINEAGE_CSV_BASE = SCRIPTS_DIR / "results" / "lineage_csv"


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(map(str, cmd))}\n", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def split_lineage_csv(lineage_csv: Path, out_dir: Path) -> int:
    df = pd.read_csv(lineage_csv)
    df = df[df["in_tree"]].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cell_id, g in df.groupby("cell_id"):
        g.to_csv(out_dir / f"cell_{int(cell_id):03d}.csv", index=False)
        n += 1
    return n


def process_channel(ch_dir: Path, run_label: str, skip_seg: bool) -> bool:
    print(f"\n{'='*60}\n=== {ch_dir.name} ===\n{'='*60}", flush=True)

    if not skip_seg:
        rc = run([PYTHON, "-u", str(SCRIPTS_DIR / "07_segmentation.py"),
                  "--indir", str(ch_dir)])
        if rc != 0:
            print(f"!! 07_segmentation.py failed for {ch_dir.name} (rc={rc})", flush=True)
            return False

    rc = run([PYTHON, "-u", str(SCRIPTS_DIR / "central_cell_lineage_tracker.py"),
              "--indir", str(ch_dir),
              "--pixel-size-um", "0.348",
              "--time-interval-min", "5.0",
              "--wavelength-nm", "658",
              "--n-medium", "1.333",
              "--alpha-ri", "0.00018"])
    if rc != 0:
        print(f"!! central_cell_lineage_tracker.py failed for {ch_dir.name} (rc={rc})", flush=True)
        return False

    lineage_csv = ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"
    if not lineage_csv.exists():
        print(f"!! lineage_data3D.csv not found at {lineage_csv}", flush=True)
        return False

    base_dir = LINEAGE_CSV_BASE / f"{ch_dir.name}_{run_label}"
    n_cells = split_lineage_csv(lineage_csv, base_dir)
    print(f"  Split {n_cells} per-cell CSVs at {base_dir}", flush=True)
    if n_cells == 0:
        print(f"!! no cells with in_tree=True in {ch_dir.name}", flush=True)
        return False

    rc = run([PYTHON, "-u", str(SCRIPTS_DIR / "qpi_fig_03_lineage_analysis.py"),
              "--base-dir", str(base_dir), "--mode", "physical"])
    if rc != 0:
        print(f"!! qpi_fig_03 failed for {ch_dir.name} (rc={rc})", flush=True)

    rc = run([PYTHON, "-u", str(SCRIPTS_DIR / "qpi_fig_04_growth_oscillation.py"),
              "--base-dir", str(base_dir)])
    if rc != 0:
        print(f"!! qpi_fig_04 failed for {ch_dir.name} (rc={rc})", flush=True)

    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=r"G:\マイドライブ")
    p.add_argument("--channels", nargs="*", default=None,
                   help="ch directory names. Default: all ch* subdirs.")
    p.add_argument("--run-label", default="2026-05-01_d20")
    p.add_argument("--skip-seg", action="store_true",
                   help="Skip 07_segmentation.py (reuse existing inference_out)")
    args = p.parse_args()

    root = Path(args.root)
    if args.channels:
        chs = [root / c for c in args.channels]
    else:
        chs = sorted(d for d in root.glob("ch*") if d.is_dir())

    print(f"Channels to process ({len(chs)}): {[c.name for c in chs]}", flush=True)

    results = {}
    for ch in chs:
        try:
            results[ch.name] = process_channel(ch, args.run_label, args.skip_seg)
        except Exception as e:
            print(f"!! {ch.name} crashed: {e!r}", flush=True)
            results[ch.name] = False

    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}", flush=True)
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
