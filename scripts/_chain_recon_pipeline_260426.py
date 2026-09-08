"""Chained driver for the 260426 dataset (`online_crop_sub_zstack`).

The reconstruction + crop step has already been done upstream, so this
driver only runs the per-channel analysis pipeline (segmentation ->
central-cell lineage tracker -> per-channel figures -> pooled
batch_figures) on each Pos*/output_phase/channels/crop_sub_rawraw/<Z>/.

Per-channel analysis is restricted to img_NNN in [FRAME_MIN, FRAME_MAX].
The tracker re-indexes the surviving frames starting at 0, so img_NNN ==
FRAME_MIN ends up at time 0 in all downstream plots (lineage_data3D.csv:
time_h = idx * dt / 60).

Streams stdout through, aborts on the first hard failure, but tolerates
per-Pos analysis failures so one bad Pos doesn't sink the rest of the
batch. Run from the omnipose conda env via the absolute python path
hard-coded in batch_all_channels.py.
"""
from __future__ import annotations

import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"H:\260426\online_crop_sub_zstack")
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable
LOG = SCRIPTS / "_chain_recon_pipeline_260426.log"

# img_NNN-based analysis window.  FRAME_MIN -> time 0.
FRAME_MIN = 137
FRAME_MAX = 1000

# Which z-stack layer(s) to analyse under each Pos*/crop_sub_rawraw/.
# 260426 has z000/z001/z002; default is the middle z. Add more entries
# (e.g. ["z000", "z001", "z002"]) to fan out across all focal planes.
Z_LAYERS = ["z001"]

# Medium for the 260426 dataset is constant 2% glucose for the entire run
# (no switches). We re-use the 260423 RI calibration (closest available in
# time; same protocol assumed) so that n_cell = n_medium + measured uses
# the calibrated wo_2 value (~1.3350287) instead of the scalar default of
# 1.333. A single-anchor schedule at frame 0 keeps that medium for the
# whole window.
#
# If a 260426-native calibration gets produced later, just repoint CAL_PATH.
CAL_PATH: Path | None = Path(r"H:\260423\grid_2pergluc_1\ri_calibration_results.json")
MEDIA_SCHEDULE: str | None = "0:wo_2"

# Protein-density baseline: user-fixed at 1.33 so that mass density is
# computed as (n_cell - 1.33) / alpha_ri, independent of the calibration's
# internal n_milliq reference (1.3312 in 260423's JSON).
N_MILLIQ: float = 1.33


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
        _log("<<< (rc=0)")
    return rc


def main() -> None:
    LOG.write_text(f"=== chain START {_stamp()} ===\n", encoding="utf-8")
    _log(f"ROOT={ROOT}  FRAME_MIN={FRAME_MIN}  FRAME_MAX={FRAME_MAX}  Z_LAYERS={Z_LAYERS}")

    pos_dirs = sorted(
        (p for p in ROOT.iterdir()
         if p.is_dir() and p.name.startswith("Pos")
         and (p / "output_phase" / "channels" / "crop_sub_rawraw").is_dir()),
        key=lambda p: int(p.name.removeprefix("Pos")),
    )
    _log(f"  found {len(pos_dirs)} Pos dirs with crop_sub_rawraw")

    for pd in pos_dirs:
        for z in Z_LAYERS:
            ch_root = pd / "output_phase" / "channels" / "crop_sub_rawraw" / z
            if not ch_root.is_dir():
                _log(f"  -- {pd.name}/{z} missing, skip")
                continue
            _log(f"  --- analyse {pd.name}/{z} --- root={ch_root}")
            cmd = [
                PY,
                str(SCRIPTS / "batch_all_channels.py"),
                "--root", str(ch_root),
                "--frame-min", str(FRAME_MIN),
                "--frame-max", str(FRAME_MAX),
            ]
            if CAL_PATH is not None:
                cmd += ["--ri-calibration", str(CAL_PATH)]
            if MEDIA_SCHEDULE is not None:
                cmd += ["--media-schedule", MEDIA_SCHEDULE]
            if N_MILLIQ is not None:
                cmd += ["--n-milliq", str(N_MILLIQ)]
            _run(cmd, allow_fail=True)

    _log("=== chain DONE ===")


if __name__ == "__main__":
    main()
