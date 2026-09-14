"""
Per-checkpoint volume-trace overlay runner.

For each saved checkpoint (`*_eNNNN`) in --model-dir:
  1. Run 07_segmentation.py with that checkpoint on --ch-dir
  2. Run central_cell_lineage_tracker.py on the same dir
  3. Pack per-cell lineage CSVs into a *_data.npz with the schema that
     batch_volume_trace_overlay.py understands (label_i / frame_index_i /
     volume_um3_rod_i / mean_ri_i / mass_pg_i / n_series).
  4. Call `batch_volume_trace_overlay.py --from-npz <npz>` to get a figure
     where each line is one cell of the lineage at that checkpoint.

Usage:
  python checkpoint_overlay_runner.py \\
    --model-dir "C:/Users/QPI/Desktop/train/omni_model_d20/models" \\
    --ch-dir   "G:/マイドライブ/ch02"
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
SCRIPTS = Path(__file__).resolve().parent
EPOCH_RE = re.compile(r"_e(\d{4})$")


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(map(str, cmd))}\n", flush=True)
    return subprocess.run(cmd).returncode


def find_checkpoints(model_dir: Path) -> list[tuple[int, Path]]:
    out = []
    for p in model_dir.iterdir():
        m = EPOCH_RE.search(p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort(key=lambda x: x[0])
    return out


def split_lineage_to_cells(lineage_csv: Path) -> list[tuple[str, pd.DataFrame]]:
    df = pd.read_csv(lineage_csv)
    if "frame_index" not in df.columns and "frame" in df.columns:
        df = df.rename(columns={"frame": "frame_index"})
    df = df[df["in_tree"]].copy()
    series = []
    for cell_id, g in df.groupby("cell_id"):
        g = g.sort_values("frame_index")
        series.append((f"cell_{int(cell_id):03d}", g))
    return series


def series_to_npz(series: list[tuple[str, pd.DataFrame]], out_path: Path,
                  time_interval_min: float | None = None) -> None:
    out: dict = {"n_series": np.array(len(series), dtype=np.int64)}
    for i, (label, df) in enumerate(series):
        out[f"label_{i}"] = np.array([label], dtype=object)
        out[f"frame_index_{i}"] = df["frame_index"].to_numpy(dtype=np.float64)
        out[f"volume_um3_rod_{i}"] = df["volume_um3_rod"].to_numpy(dtype=np.float64)
        out[f"mean_ri_{i}"] = df["mean_ri"].to_numpy(dtype=np.float64)
        out[f"mass_pg_{i}"] = df["mass_pg"].to_numpy(dtype=np.float64)
    if time_interval_min is not None:
        out["time_interval_min"] = np.array(float(time_interval_min), dtype=np.float64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


def process_checkpoint(epoch: int, ckpt: Path, ch_dir: Path,
                       npz_dir: Path, time_interval_min: float,
                       skip_seg: bool, skip_lineage: bool,
                       overlay_args: list[str]) -> bool:
    print(f"\n{'='*60}\n=== epoch {epoch} ({ckpt.name}) ===\n{'='*60}", flush=True)

    if not skip_seg:
        rc = run([PYTHON, "-u", str(SCRIPTS / "07_segmentation.py"),
                  "--indir", str(ch_dir),
                  "--model-path", str(ckpt)])
        if rc != 0:
            print(f"!! 07_segmentation.py failed (rc={rc})")
            return False

    if not skip_lineage:
        rc = run([PYTHON, "-u", str(SCRIPTS / "central_cell_lineage_tracker.py"),
                  "--indir", str(ch_dir),
                  "--pixel-size-um", "0.348",
                  "--time-interval-min", str(time_interval_min),
                  "--wavelength-nm", "658",
                  "--n-medium", "1.333",
                  "--alpha-ri", "0.00018"])
        if rc != 0:
            print(f"!! central_cell_lineage_tracker.py failed (rc={rc})")
            return False

    lineage_csv = ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"
    if not lineage_csv.exists():
        print(f"!! lineage_data3D.csv missing at {lineage_csv}")
        return False

    series = split_lineage_to_cells(lineage_csv)
    if not series:
        print(f"!! no in_tree cells for epoch {epoch}")
        return False

    npz_path = npz_dir / f"{ch_dir.name}_e{epoch:04d}_data.npz"
    series_to_npz(series, npz_path, time_interval_min=time_interval_min)
    print(f"npz: {npz_path}  ({len(series)} cells)", flush=True)

    rc = run([PYTHON, "-u", str(SCRIPTS / "batch_volume_trace_overlay.py"),
              "--from-npz", str(npz_path),
              *overlay_args])
    if rc != 0:
        print(f"!! batch_volume_trace_overlay failed (rc={rc})")
        return False
    return True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--ch-dir", required=True, type=Path)
    p.add_argument("--npz-dir", type=Path,
                   default=SCRIPTS / "results" / "checkpoint_npz")
    p.add_argument("--time-interval-min", type=float, default=5.0)
    p.add_argument("--epochs", nargs="*", type=int, default=None,
                   help="Only these epochs (e.g. 200 500 1000). Default: all.")
    p.add_argument("--skip-seg", action="store_true",
                   help="Skip segmentation step (reuse existing inference_out)")
    p.add_argument("--skip-lineage", action="store_true",
                   help="Skip lineage tracker (reuse existing lineage_out)")
    # Pass-through to batch_volume_trace_overlay
    p.add_argument("--volume-ylim", nargs=2, type=float, default=None)
    p.add_argument("--mean-ri-ylim", nargs=2, type=float, default=None)
    p.add_argument("--mass-ylim", nargs=2, type=float, default=None)
    args = p.parse_args()

    cps = find_checkpoints(args.model_dir)
    if args.epochs:
        cps = [(e, p) for e, p in cps if e in args.epochs]
    if not cps:
        print("No checkpoints to process.")
        return 1

    print(f"Processing {len(cps)} checkpoints: {[e for e, _ in cps]}", flush=True)

    overlay_args = ["--time-interval-min", str(args.time_interval_min)]
    if args.volume_ylim:
        overlay_args += ["--volume-ylim", str(args.volume_ylim[0]), str(args.volume_ylim[1])]
    if args.mean_ri_ylim:
        overlay_args += ["--mean-ri-ylim", str(args.mean_ri_ylim[0]), str(args.mean_ri_ylim[1])]
    if args.mass_ylim:
        overlay_args += ["--mass-ylim", str(args.mass_ylim[0]), str(args.mass_ylim[1])]

    results = {}
    for epoch, ckpt in cps:
        try:
            results[epoch] = process_checkpoint(
                epoch, ckpt, args.ch_dir, args.npz_dir,
                args.time_interval_min, args.skip_seg, args.skip_lineage,
                overlay_args,
            )
        except Exception as e:
            print(f"!! epoch {epoch} crashed: {e!r}")
            results[epoch] = False

    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}")
    for e, ok in results.items():
        print(f"  epoch {e}: {'OK' if ok else 'FAILED'}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
