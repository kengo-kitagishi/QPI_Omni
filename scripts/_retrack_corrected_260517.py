"""Re-track already-segmented 260517 positions with the CORRECTED tracker
(mask-direct / medial-axis volume), regenerating lineage_data3D.csv + clist.csv
ONLY (no figures, no Notion). Single-stream, resumable.

Why tracker-only: per_channel_figures/batch_figures post every figure to the
figure-hub inbox + Notion. Re-tracking 46 positions through batch_all_channels
would emit ~1500 figures and Notion posts. The corrected *data* is the goal;
figures can be regenerated later from the corrected CSVs.

Resumable: a channel is skipped if its lineage_data3D.csv already has the
`volume_um3_profile` column (i.e. already produced by the corrected tracker).

Masks are reused as-is (GPU-made); no segmentation is run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(r"F:\260517\2per_0055per_0per_2per_crop_sub")
SCRIPTS = Path(__file__).resolve().parent
PY = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
TRACKER = SCRIPTS / "central_cell_lineage_tracker.py"
LOG = SCRIPTS / "_retrack_corrected_260517.log"

CAL = r"F:\260517\grid_2pergluc_2\ri_calibration_results.json"
BAD = r"F:\260517\drift_session_20260521T2142\bad_frames.json"
MEDIA_SCHEDULE = "0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2"
PIXEL_UM = "0.34567514677103717"
DT_MIN = "5.0"
FRAME_MIN = "2"
Z = "z000"

CORRECTED_MARKER = "volume_um3_profile"  # column only the corrected tracker writes


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_corrected(lineage_csv: Path) -> bool:
    """True if the CSV already has the corrected-tracker columns."""
    if not lineage_csv.exists():
        return False
    try:
        head = pd.read_csv(lineage_csv, nrows=0)
        return CORRECTED_MARKER in head.columns
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=46)
    ap.add_argument("--force", action="store_true",
                    help="re-track even channels already corrected")
    args = ap.parse_args()

    LOG.write_text(f"=== 260517 corrected re-track START {_stamp()} ===\n", encoding="utf-8")
    _log(f"Pos {args.start}..{args.end}  force={args.force}")

    # Build the channel work-list (only channels that have masks).
    targets: list[Path] = []
    skipped = 0
    for n in range(args.start, args.end + 1):
        z = ROOT / f"Pos{n}" / "output_phase" / "channels" / "crop_sub_rawraw" / Z
        if not z.is_dir():
            continue
        for ch in sorted(z.glob("ch*")):
            if not ch.is_dir():
                continue
            masks = ch / "inference_out"
            if not masks.is_dir() or not any(masks.glob("*_masks.tif")):
                continue
            lineage = masks / "lineage_out" / "lineage_data3D.csv"
            if (not args.force) and is_corrected(lineage):
                skipped += 1
                continue
            targets.append(ch)

    _log(f"queued {len(targets)} channels to re-track ({skipped} already corrected, skipped)")
    if not targets:
        _log("nothing to do")
        return

    n_ok = n_err = 0
    for i, ch in enumerate(targets, 1):
        pos = next((p.name for p in ch.parents if p.name.startswith("Pos")), "?")
        _log(f"--- [{i}/{len(targets)}] {pos}/{ch.name} ---")
        cmd = [
            PY, "-u", str(TRACKER),
            "--indir", str(ch),
            "--pixel-size-um", PIXEL_UM,
            "--time-interval-min", DT_MIN,
            "--wavelength-nm", "658.0",
            "--alpha-ri", "0.00018",
            "--ri-calibration", CAL,
            "--media-schedule", MEDIA_SCHEDULE,
            "--bad-frames", BAD,
            "--frame-min", FRAME_MIN,
        ]
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            _log(f"!! {pos}/{ch.name} tracker rc={rc}")
            n_err += 1
        else:
            n_ok += 1

    _log(f"=== DONE: ok={n_ok} err={n_err} ===")


if __name__ == "__main__":
    main()
