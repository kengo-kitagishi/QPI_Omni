"""Resume the 260405_acute_z18_200h pipeline after the channel_rois.json
fix. Skips the multi-hour recon (output_phase / output_phase_raw already
on disk) and runs only steps 1-4 (channel_rois copy, ECC, grid_subtract,
correct_0pergluc), followed by the per-Pos analysis loop."""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\260405_acute_z18_200h")
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable
LOG = SCRIPTS / "_chain_resume_pipeline_260405_acute.log"

CAL_PATH = ROOT / "grid_2pergluc_60ms_1" / "ri_calibration_results.json"
MEDIA_SCHEDULE = "0:wo_2,575:wo_0,1440:wo_2"


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _run(cmd: list[str], allow_fail: bool = False) -> int:
    _log(">>> " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd)
    rc = proc.returncode
    if rc != 0:
        if allow_fail:
            _log(f"<<< (rc={rc}) allow_fail=True, continuing")
        else:
            _log(f"<<< (rc={rc}) FATAL, aborting")
            sys.exit(rc)
    else:
        _log(f"<<< (rc=0)")
    return rc


def main() -> None:
    LOG.write_text(f"=== resume chain START {_stamp()} ===\n", encoding="utf-8")

    _log("Step B+C (resume): batch_pipeline_all_pos.py --skip-grid-0per --no-reset")
    _run([
        PY,
        str(SCRIPTS / "batch_pipeline_all_pos.py"),
        "--skip-grid-0per",
        "--no-reset",
    ])

    _log("Step D: per-Pos analysis loop")
    pos_dirs = sorted(
        (p for p in (ROOT / "ph_260405").iterdir()
         if p.is_dir() and p.name.startswith("Pos")
         and (p / "output_phase" / "channels" / "crop_sub_rawraw").is_dir()),
        key=lambda p: int(p.name.removeprefix("Pos")),
    )
    _log(f"  found {len(pos_dirs)} Pos dirs with crop_sub_rawraw output")
    for pd in pos_dirs:
        ch_root = pd / "output_phase" / "channels" / "crop_sub_rawraw"
        _log(f"  --- analyse {pd.name} --- root={ch_root}")
        _run([
            PY,
            str(SCRIPTS / "batch_all_channels.py"),
            "--root", str(ch_root),
            "--ri-calibration", str(CAL_PATH),
            "--media-schedule", MEDIA_SCHEDULE,
        ], allow_fail=True)

    _log("=== resume chain DONE ===")


if __name__ == "__main__":
    main()
