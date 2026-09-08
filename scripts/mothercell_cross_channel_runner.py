"""
Per-checkpoint multi-channel runner.

For each requested epoch:
  1. For each channel in --channels:
       - If a snapshot of lineage_data3D.csv already exists under
         results/per_model/eNNNN/chXX/, skip seg+lineage (idempotent).
       - Otherwise run 07_segmentation.py with the requested checkpoint and
         central_cell_lineage_tracker.py, then COPY the resulting
         lineage_data3D.csv into the snapshot dir BEFORE the next epoch can
         overwrite it.
  2. Read each ch's snapshot CSV and split into per-cell CSVs at
     results/per_cell_csv/eNNNN/<ch>_cell_NNN.csv (namespaced by ch).
  3. Extract mother cell (cell_id == 0, in_tree) from each ch's snapshot,
     pack into a single NPZ, and call batch_volume_trace_overlay.py to
     render the 3-panel cross-channel mother-cell figure.
  4. Run qpi_fig_03_lineage_analysis.py with --base-dir
     results/per_cell_csv/eNNNN/ --mode physical (12 figures aggregating all
     channels for that model).

Idempotency: re-running the script will skip any (epoch, ch) whose snapshot
already exists. To force re-segmentation, delete the snapshot directory.

Usage:
  python mothercell_cross_channel_runner.py \
    --model-dir "C:/Users/QPI/Desktop/train/omni_model_d20/models" \
    --ch-root   "G:/マイドライブ" \
    --channels ch01 ch02 ch07 ch08 ch11 \
    --epochs 200 500 800 1100 1400
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
SCRIPTS = Path(__file__).resolve().parent
RESULTS = SCRIPTS / "results"
PER_MODEL_DIR = RESULTS / "per_model"
PER_CELL_CSV_DIR = RESULTS / "per_cell_csv"
MOTHER_NPZ_DIR = RESULTS / "mothercell_cross_ch_npz"

MODEL_PREFIX = (
    "cellpose_residual_on_style_on_concatenation_off_omni_abstract"
    "_nclasses_3_nchan_1_dim_2_omni_model_d20_2026_05_01_19_01_52.321350"
)


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(map(str, cmd))}\n", flush=True)
    return subprocess.run(cmd).returncode


def find_checkpoint(model_dir: Path, epoch: int) -> Path | None:
    p = model_dir / f"{MODEL_PREFIX}_e{epoch:04d}"
    return p if p.exists() else None


def snapshot_lineage(ch_dir: Path, snap_dir: Path) -> bool:
    """Copy inference_out/lineage_out artifacts into snap_dir."""
    src_dir = ch_dir / "inference_out" / "lineage_out"
    src_csv = src_dir / "lineage_data3D.csv"
    if not src_csv.exists():
        print(f"  !! lineage_data3D.csv missing at {src_csv}", flush=True)
        return False
    snap_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_csv, snap_dir / "lineage_data3D.csv")
    for opt in ("clist.csv", "lineage_cells.json"):
        s = src_dir / opt
        if s.exists():
            shutil.copy2(s, snap_dir / opt)
    return True


def ensure_per_ch_lineage(ch_dir: Path, ckpt: Path, snap_dir: Path,
                          time_interval_min: float) -> bool:
    """Idempotent: skip seg+lineage if snapshot already on disk."""
    snap_csv = snap_dir / "lineage_data3D.csv"
    if snap_csv.exists():
        print(f"  {ch_dir.name}: snapshot exists, skip seg+lineage "
              f"({snap_csv.name})", flush=True)
        return True

    print(f"  {ch_dir.name}: seg + lineage (no snapshot)", flush=True)
    rc = run([PYTHON, "-u", str(SCRIPTS / "07_segmentation.py"),
              "--indir", str(ch_dir),
              "--model-path", str(ckpt)])
    if rc != 0:
        print(f"  !! 07_segmentation.py failed (rc={rc})", flush=True)
        return False

    rc = run([PYTHON, "-u", str(SCRIPTS / "central_cell_lineage_tracker.py"),
              "--indir", str(ch_dir),
              "--pixel-size-um", "0.348",
              "--time-interval-min", str(time_interval_min),
              "--wavelength-nm", "658",
              "--n-medium", "1.333",
              "--alpha-ri", "0.00018"])
    if rc != 0:
        print(f"  !! central_cell_lineage_tracker.py failed (rc={rc})", flush=True)
        return False

    return snapshot_lineage(ch_dir, snap_dir)


def split_to_per_cell_csvs(snap_csv: Path, out_dir: Path, ch_label: str) -> int:
    df = pd.read_csv(snap_csv)
    if "frame_index" not in df.columns and "frame" in df.columns:
        df = df.rename(columns={"frame": "frame_index"})
    df = df[df["in_tree"]].copy()
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for cell_id, g in df.groupby("cell_id"):
        g.to_csv(out_dir / f"{ch_label}_cell_{int(cell_id):03d}.csv", index=False)
        n += 1
    return n


def mother_from_snapshot(snap_csv: Path) -> pd.DataFrame | None:
    df = pd.read_csv(snap_csv)
    if "frame_index" not in df.columns and "frame" in df.columns:
        df = df.rename(columns={"frame": "frame_index"})
    df = df[df["in_tree"] & (df["cell_id"] == 0)].copy()
    if df.empty:
        return None
    df = df.sort_values("frame_index")
    return df[["frame_index", "volume_um3_rod", "mean_ri", "mass_pg"]]


def pack_mother_npz(series: list[tuple[str, pd.DataFrame]], out_path: Path,
                    time_interval_min: float) -> None:
    out: dict = {"n_series": np.array(len(series), dtype=np.int64)}
    for i, (label, df) in enumerate(series):
        out[f"label_{i}"] = np.array([label], dtype=object)
        out[f"frame_index_{i}"] = df["frame_index"].to_numpy(dtype=float)
        out[f"volume_um3_rod_{i}"] = df["volume_um3_rod"].to_numpy(dtype=float)
        out[f"mean_ri_{i}"] = df["mean_ri"].to_numpy(dtype=float)
        out[f"mass_pg_{i}"] = df["mass_pg"].to_numpy(dtype=float)
    out["time_interval_min"] = np.array(float(time_interval_min), dtype=float)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True, type=Path)
    p.add_argument("--ch-root", required=True, type=Path,
                   help='e.g. "G:/マイドライブ"')
    p.add_argument("--channels", nargs="+",
                   default=["ch01", "ch02", "ch07", "ch08", "ch11"])
    p.add_argument("--epochs", nargs="+", type=int, required=True)
    p.add_argument("--time-interval-min", type=float, default=5.0)
    p.add_argument("--per-model-dir", type=Path, default=PER_MODEL_DIR)
    p.add_argument("--per-cell-csv-dir", type=Path, default=PER_CELL_CSV_DIR)
    p.add_argument("--mother-npz-dir", type=Path, default=MOTHER_NPZ_DIR)
    p.add_argument("--skip-fig03", action="store_true",
                   help="Skip qpi_fig_03 step (e.g. if it crashes for a model)")
    args = p.parse_args()

    summary: dict[int, str] = {}
    for epoch in args.epochs:
        ckpt = find_checkpoint(args.model_dir, epoch)
        if ckpt is None:
            print(f"!! checkpoint missing for epoch {epoch}", flush=True)
            summary[epoch] = "missing-ckpt"
            continue
        print(f"\n{'='*60}\n=== epoch {epoch} ({ckpt.name}) ===\n{'='*60}",
              flush=True)

        per_cell_dir = args.per_cell_csv_dir / f"e{epoch:04d}"

        series: list[tuple[str, pd.DataFrame]] = []
        ch_status: dict[str, str] = {}
        for ch in args.channels:
            ch_dir = args.ch_root / ch
            snap_dir = args.per_model_dir / f"e{epoch:04d}" / ch
            ok = ensure_per_ch_lineage(ch_dir, ckpt, snap_dir,
                                       args.time_interval_min)
            if not ok:
                ch_status[ch] = "lineage-failed"
                continue

            snap_csv = snap_dir / "lineage_data3D.csv"
            n_cells = split_to_per_cell_csvs(snap_csv, per_cell_dir, ch)
            print(f"  {ch}: split {n_cells} per-cell CSVs -> "
                  f"{per_cell_dir.name}/", flush=True)

            mother = mother_from_snapshot(snap_csv)
            if mother is None or mother.empty:
                print(f"  !! {ch}: no mother cell (cell_id=0, in_tree)",
                      flush=True)
                ch_status[ch] = "no-mother"
                continue
            print(f"  {ch}: mother cell {len(mother)} frames", flush=True)
            series.append((ch, mother))
            ch_status[ch] = f"ok ({n_cells} cells)"

        print(f"  ch status: {ch_status}", flush=True)

        if not series:
            print(f"!! epoch {epoch}: no channels usable", flush=True)
            summary[epoch] = "no-series"
            continue

        out_npz = args.mother_npz_dir / f"mothercell_cross_ch_e{epoch:04d}_data.npz"
        pack_mother_npz(series, out_npz, args.time_interval_min)
        print(f"  npz: {out_npz}  ({len(series)} channels)", flush=True)

        rc_overlay = run([PYTHON, "-u",
                          str(SCRIPTS / "batch_volume_trace_overlay.py"),
                          "--from-npz", str(out_npz),
                          "--time-interval-min", str(args.time_interval_min)])
        if rc_overlay != 0:
            print(f"  !! batch_volume_trace_overlay failed (rc={rc_overlay})",
                  flush=True)

        rc_fig03 = 0
        if not args.skip_fig03:
            rc_fig03 = run([PYTHON, "-u",
                            str(SCRIPTS / "qpi_fig_03_lineage_analysis.py"),
                            "--base-dir", str(per_cell_dir),
                            "--mode", "physical"])
            if rc_fig03 != 0:
                print(f"  !! qpi_fig_03 failed (rc={rc_fig03})", flush=True)

        if rc_overlay == 0 and rc_fig03 == 0:
            summary[epoch] = "OK"
        else:
            summary[epoch] = (
                f"partial (overlay={rc_overlay}, fig03={rc_fig03})")

    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}")
    for e, status in summary.items():
        print(f"  epoch {e}: {status}")
    return 0 if all(s == "OK" for s in summary.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
