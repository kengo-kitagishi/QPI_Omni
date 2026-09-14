"""Generate channel_rois.json for every grid_2per/Pos{N}_x+0_y+0 by
invoking channel_crop.py --detect on the z=18 phase tif. Required by
batch_pipeline_all_pos.py step 1, which copies these into each
timelapse Pos."""
from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
GRID_2PER = Path(r"F:\260405_acute_z18_200h\grid_2pergluc_60ms_1")
GRID_Z_INDEX = 18
PATTERN = f"*_ph_{GRID_Z_INDEX:03d}_phase.tif"
PY = sys.executable


def main() -> None:
    centers = sorted(
        [p for p in GRID_2PER.iterdir()
         if p.is_dir() and re.fullmatch(r"Pos\d+_x\+0_y\+0", p.name)],
        key=lambda p: int(re.match(r"Pos(\d+)", p.name).group(1)),
    )
    print(f"[info] {len(centers)} grid Pos centers found")
    n_ok, n_skip, n_fail = 0, 0, 0
    for pos in centers:
        op = pos / "output_phase"
        if not op.is_dir():
            print(f"[skip] {pos.name}: no output_phase")
            n_fail += 1
            continue
        rois = op / "channels" / "channel_rois.json"
        if rois.exists():
            print(f"[skip] {pos.name}: channel_rois.json already exists")
            n_skip += 1
            continue
        cmd = [PY, str(SCRIPTS / "channel_crop.py"),
               "--dir", str(op),
               "--detect",
               "--pattern", PATTERN]
        # Discard the child's stdout/stderr — channel_crop.py emits cp932 bytes
        # on some Windows consoles and the utf-8 decoder explodes. We only
        # need to know whether the json file landed on disk.
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if proc.returncode == 0 and rois.exists():
            print(f"[ok]   {pos.name}: wrote {rois}", flush=True)
            n_ok += 1
        else:
            print(f"[fail] {pos.name}: rc={proc.returncode}  rois_exists={rois.exists()}", flush=True)
            n_fail += 1
    print(f"[summary] ok={n_ok}  skip={n_skip}  fail={n_fail}")
    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
