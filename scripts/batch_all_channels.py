"""
Batch pipeline: segmentation -> lineage tracker -> per-channel figures,
then a single pooled batch_figures pass over all successful channels.

Default per-channel flow (each ch produces its own PDFs):
    07_segmentation.py                  ->  inference_out/*_masks.tif
    central_cell_lineage_tracker.py     ->  lineage_data3D.csv, clist.csv,
                                            lineage_run_params.json
    per_channel_figures.py              ->  mother_volume / mother_mean_ri /
                                            lineage_tree (3 PDFs per ch)

After the per-channel loop, pooled overlay plots across all channels:
    batch_figures.py                    ->  pooled_mother_volume /
                                            pooled_mother_mean_ri /
                                            pooled_mother_mass (3 PDFs total)

The advanced detail scripts (mother_cell_cycle_stats.py,
lineage_survival_analysis.py, qpi_fig_03_lineage_analysis.py,
qpi_fig_04_growth_oscillation.py) are NOT called by this pipeline; run them
manually when you want those analyses.

Usage:
  python batch_all_channels.py
  python batch_all_channels.py --channels ch01 ch02
  python batch_all_channels.py --root "G:/マイドライブ" \\
      --ri-calibration "G:/マイドライブ/calibration/ri_calibration.json" \\
      --media-schedule "0:wo_2,575:wo_0p01,863:wo_0,1439:wo_2"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYTHON = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
SCRIPTS_DIR = Path(__file__).resolve().parent


def run(cmd: list[str]) -> int:
    print(f"\n>>> {' '.join(map(str, cmd))}\n", flush=True)
    proc = subprocess.run(cmd)
    return proc.returncode


def process_channel(
    ch_dir: Path,
    skip_seg: bool,
    model_path: str | None,
    pixel_size_um: float,
    time_interval_min: float,
    wavelength_nm: float,
    n_medium: float,
    alpha_ri: float,
    ri_calibration: str | None,
    calibration_id: str | None,
    media_schedule: str | None,
    n_milliq: float | None,
    bad_frames: str | None = None,
) -> bool:
    print(f"\n{'='*60}\n=== {ch_dir.name} ===\n{'='*60}", flush=True)

    if not skip_seg:
        cmd = [PYTHON, "-u", str(SCRIPTS_DIR / "07_segmentation.py"),
               "--indir", str(ch_dir)]
        if model_path:
            cmd += ["--model-path", model_path]
        rc = run(cmd)
        if rc != 0:
            print(f"!! 07_segmentation.py failed for {ch_dir.name} (rc={rc})", flush=True)
            return False

    tracker_cmd = [
        PYTHON, "-u", str(SCRIPTS_DIR / "central_cell_lineage_tracker.py"),
        "--indir", str(ch_dir),
        "--pixel-size-um", str(pixel_size_um),
        "--time-interval-min", str(time_interval_min),
        "--wavelength-nm", str(wavelength_nm),
        "--n-medium", str(n_medium),
        "--alpha-ri", str(alpha_ri),
    ]
    if ri_calibration:
        tracker_cmd += ["--ri-calibration", ri_calibration]
    if calibration_id:
        tracker_cmd += ["--calibration-id", calibration_id]
    if media_schedule:
        tracker_cmd += ["--media-schedule", media_schedule]
    if n_milliq is not None:
        tracker_cmd += ["--n-milliq", str(n_milliq)]
    if bad_frames:
        tracker_cmd += ["--bad-frames", bad_frames]
    rc = run(tracker_cmd)
    if rc != 0:
        print(f"!! central_cell_lineage_tracker.py failed for {ch_dir.name} (rc={rc})", flush=True)
        return False

    lineage_csv = ch_dir / "inference_out" / "lineage_out" / "lineage_data3D.csv"
    if not lineage_csv.exists():
        print(f"!! lineage_data3D.csv not found at {lineage_csv}", flush=True)
        return False

    rc = run([PYTHON, "-u", str(SCRIPTS_DIR / "per_channel_figures.py"),
              "--indir", str(ch_dir)])
    if rc != 0:
        print(f"!! per_channel_figures.py failed for {ch_dir.name} (rc={rc})", flush=True)
        return False

    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=r"G:\マイドライブ")
    p.add_argument("--channels", nargs="*", default=None,
                   help="ch directory names. Default: all ch* subdirs.")
    p.add_argument("--skip-seg", action="store_true",
                   help="Skip 07_segmentation.py (reuse existing inference_out)")
    p.add_argument("--model-path", default=None,
                   help="Explicit checkpoint to pass to 07_segmentation.py")

    # Tracker physical params
    p.add_argument("--pixel-size-um", type=float, default=0.348)
    p.add_argument("--time-interval-min", type=float, default=5.0)
    p.add_argument("--wavelength-nm", type=float, default=658.0)
    p.add_argument("--n-medium", type=float, default=1.333,
                   help="Fallback scalar n_medium when no --ri-calibration is given.")
    p.add_argument("--alpha-ri", type=float, default=0.00018)

    # Frame-dependent n_medium passthrough to the tracker
    p.add_argument("--ri-calibration", default=None,
                   help="Path to RI calibration JSON (enables frame-dependent n_medium).")
    p.add_argument("--calibration-id", default=None,
                   help="Pick a non-active calibration entry by id.")
    p.add_argument("--media-schedule", default=None,
                   help="Frame-keyed medium switches, e.g. "
                        "'0:wo_2,575:wo_0p01,863:wo_0,1439:wo_2'.")
    p.add_argument("--n-milliq", type=float, default=None,
                   help="Override protein-density baseline RI.")

    p.add_argument("--bad-frames", default=None,
                   help="Path to bad_frames.json (from extract_bad_frames.py). "
                        "Bad timepoints for each Pos are excluded from tracking.")
    p.add_argument("--skip-batch-figures", action="store_true",
                   help="Skip the final cross-channel pooled overlay step.")
    p.add_argument("--ch-workers", type=int, default=1,
                   help="Number of channels to process in parallel via ProcessPoolExecutor "
                        "(default 1 = sequential). Each worker loads its own cellpose model "
                        "on the GPU; with ~3 GB per model + 24 GB GPU, 4-6 is safe.")
    args = p.parse_args()

    root = Path(args.root)
    if args.channels:
        chs = [root / c for c in args.channels]
    else:
        chs = sorted(d for d in root.glob("ch*") if d.is_dir())

    print(f"Channels to process ({len(chs)}): {[c.name for c in chs]}", flush=True)
    print(f"ch-workers: {args.ch_workers}", flush=True)

    results: dict[str, bool] = {}
    succeeded: list[Path] = []
    kwargs = dict(
        skip_seg=args.skip_seg,
        model_path=args.model_path,
        pixel_size_um=args.pixel_size_um,
        time_interval_min=args.time_interval_min,
        wavelength_nm=args.wavelength_nm,
        n_medium=args.n_medium,
        alpha_ri=args.alpha_ri,
        ri_calibration=args.ri_calibration,
        calibration_id=args.calibration_id,
        media_schedule=args.media_schedule,
        n_milliq=args.n_milliq,
        bad_frames=args.bad_frames,
    )

    if args.ch_workers <= 1:
        # Sequential path (preserves previous behavior + interleaved output)
        for ch in chs:
            try:
                ok = process_channel(ch, **kwargs)
            except Exception as e:
                print(f"!! {ch.name} crashed: {e!r}", flush=True)
                ok = False
            results[ch.name] = ok
            if ok:
                succeeded.append(ch)
    else:
        # Parallel: each ch is a separate subprocess running its own cellpose +
        # tracker + per_channel_figures. ProcessPoolExecutor spawns ch_workers
        # Python processes; each one calls process_channel which itself spawns
        # subprocess.run for 07_segmentation.py etc.
        with ProcessPoolExecutor(max_workers=args.ch_workers) as ex:
            future_to_ch = {ex.submit(process_channel, ch, **kwargs): ch for ch in chs}
            for fut in as_completed(future_to_ch):
                ch = future_to_ch[fut]
                try:
                    ok = fut.result()
                except Exception as e:
                    print(f"!! {ch.name} crashed: {e!r}", flush=True)
                    ok = False
                results[ch.name] = ok
                if ok:
                    succeeded.append(ch)
        # Stable order for the SUMMARY block + batch_figures.
        succeeded.sort(key=lambda p: p.name)

    # Pooled batch plots across all successful channels.
    if not args.skip_batch_figures and succeeded:
        print(f"\n{'='*60}\n=== batch_figures.py over {len(succeeded)} channels ===\n{'='*60}",
              flush=True)
        cmd = [PYTHON, "-u", str(SCRIPTS_DIR / "batch_figures.py")]
        for ch in succeeded:
            cmd += ["--indir", str(ch)]
        rc = run(cmd)
        if rc != 0:
            print(f"!! batch_figures.py failed (rc={rc})", flush=True)

    print(f"\n{'='*60}\n=== SUMMARY ===\n{'='*60}", flush=True)
    for name, ok in results.items():
        print(f"  {name}: {'OK' if ok else 'FAILED'}", flush=True)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
