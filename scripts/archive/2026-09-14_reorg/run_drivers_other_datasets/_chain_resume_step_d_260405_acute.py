"""Resume Step D of the 260405_acute_z18_200h analysis: for every Pos
that has crop_sub_rawraw output but no lineage_data3D.csv yet, run
batch_all_channels.py with parallel ch processing. Already-analysed Pos
are skipped automatically based on the presence of any lineage_data3D.csv
under crop_sub_rawraw/ch*/inference_out/lineage_out/.

Supports optional Pos-level concurrency, but defaults to 1 Pos at a time
because each Pos already spawns ch-workers cellpose models on the GPU."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(r"F:\260405_acute_z18_200h")
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable
LOG = SCRIPTS / "_chain_resume_step_d_260405_acute.log"

CAL_PATH = ROOT / "grid_2pergluc_60ms_1" / "ri_calibration_results.json"
MEDIA_SCHEDULE = "0:wo_2,575:wo_0,1440:wo_2"


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pos_already_done(pos_dir: Path) -> bool:
    cs = pos_dir / "output_phase" / "channels" / "crop_sub_rawraw"
    if not cs.is_dir():
        return False
    for ch in cs.iterdir():
        if ch.is_dir() and ch.name.startswith("ch"):
            if (ch / "inference_out" / "lineage_out" / "lineage_data3D.csv").exists():
                return True
    return False


def run_pos_analysis(pos_dir: Path, ch_workers: int) -> tuple[str, int]:
    ch_root = pos_dir / "output_phase" / "channels" / "crop_sub_rawraw"
    cmd = [
        PY,
        str(SCRIPTS / "batch_all_channels.py"),
        "--root", str(ch_root),
        "--ri-calibration", str(CAL_PATH),
        "--media-schedule", MEDIA_SCHEDULE,
        "--ch-workers", str(ch_workers),
    ]
    proc = subprocess.run(cmd)
    return (pos_dir.name, proc.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ch-workers", type=int, default=4,
                    help="ch parallelism inside each Pos (default 4)")
    ap.add_argument("--pos-workers", type=int, default=1,
                    help="how many Pos to run simultaneously (default 1)")
    ap.add_argument("--force", action="store_true",
                    help="re-analyse Pos even if lineage_data3D.csv already exists")
    args = ap.parse_args()

    LOG.write_text(f"=== resume Step D START {_stamp()} ===\n", encoding="utf-8")
    _log(f"ch_workers={args.ch_workers}  pos_workers={args.pos_workers}  force={args.force}")

    all_pos = sorted(
        (p for p in (ROOT / "ph_260405").iterdir()
         if p.is_dir() and p.name.startswith("Pos") and p.name != "Pos0"
         and (p / "output_phase" / "channels" / "crop_sub_rawraw").is_dir()),
        key=lambda p: int(p.name.removeprefix("Pos")),
    )
    if not args.force:
        targets = [p for p in all_pos if not pos_already_done(p)]
        skipped = [p for p in all_pos if pos_already_done(p)]
        if skipped:
            _log(f"skipping {len(skipped)} already-analysed Pos: "
                 f"{', '.join(p.name for p in skipped)}")
    else:
        targets = all_pos
    _log(f"will analyse {len(targets)} Pos: {', '.join(p.name for p in targets)}")

    if args.pos_workers <= 1:
        for pd in targets:
            _log(f"--- analyse {pd.name} ---")
            name, rc = run_pos_analysis(pd, args.ch_workers)
            if rc != 0:
                _log(f"!! {name} batch_all_channels.py rc={rc}")
    else:
        _log(f"running {args.pos_workers} Pos in parallel")
        with ProcessPoolExecutor(max_workers=args.pos_workers) as ex:
            futs = {ex.submit(run_pos_analysis, pd, args.ch_workers): pd for pd in targets}
            for fut in as_completed(futs):
                pd = futs[fut]
                try:
                    name, rc = fut.result()
                    _log(f"--- done {name} (rc={rc}) ---")
                except Exception as e:
                    _log(f"!! {pd.name} crashed: {e!r}")

    _log("=== resume Step D DONE ===")


if __name__ == "__main__":
    main()
