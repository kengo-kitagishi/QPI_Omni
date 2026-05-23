"""Run seg + lineage tracker + per_channel_figures + batch_figures on
the 260426 zstack dataset (constant 2% glucose, z000/z001/z002 per Pos,
analysis window img_NNN [137,1000]).

For every (Pos, z) directory under H:\\260426\\online_crop_sub_zstack
that has ch*/ subdirs, this calls batch_all_channels.py with:
  --media-schedule "0:wo_2"       (constant 2% glucose)
  --frame-min 137 --frame-max 1000
  --ri-calibration <260405 cal>   (reused — same wo_2 value)
  --ch-workers 4

Already-analysed (Pos, z) — defined as having any ch with lineage_data3D.csv
under inference_out/lineage_out/ — are skipped, so a kill / restart is safe."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"H:\260426\online_crop_sub_zstack")
SCRIPTS = Path(__file__).resolve().parent
PY = sys.executable
LOG = SCRIPTS / "_chain_seg_260426_zstack.log"

# Reuse the 260405 calibration JSON — both datasets share the same wo_2 RI.
CAL_PATH = Path(r"F:\260405_acute_z18_200h\grid_2pergluc_60ms_1\ri_calibration_results.json")
MEDIA_SCHEDULE = "0:wo_2"   # constant 2% glucose, no switches
FRAME_MIN = 137             # img_137 -> local frame 0 (time 0)
FRAME_MAX = 1000


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def already_done(z_root: Path) -> bool:
    """True if any ch under z_root has lineage_data3D.csv."""
    if not z_root.is_dir():
        return False
    for ch in z_root.iterdir():
        if ch.is_dir() and ch.name.startswith("ch"):
            if (ch / "inference_out" / "lineage_out" / "lineage_data3D.csv").exists():
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--z-planes", nargs="+", default=["z000", "z001", "z002"],
                    help="which z planes to analyse (default: all three)")
    ap.add_argument("--ch-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="re-analyse (Pos, z) even if lineage_data3D.csv exists")
    ap.add_argument("--start", type=int, default=1,
                    help="lowest Pos number (default 1)")
    ap.add_argument("--end", type=int, default=101,
                    help="highest Pos number (default 101)")
    args = ap.parse_args()

    LOG.write_text(f"=== 260426 zstack chain START {_stamp()} ===\n", encoding="utf-8")
    _log(f"z_planes={args.z_planes}  ch_workers={args.ch_workers}  "
         f"Pos {args.start}..{args.end}  force={args.force}")

    if not CAL_PATH.exists():
        _log(f"FATAL: calibration JSON not found at {CAL_PATH}")
        sys.exit(1)

    poses = sorted(
        (p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("Pos")),
        key=lambda p: int(p.name.removeprefix("Pos")),
    )
    poses = [p for p in poses
             if args.start <= int(p.name.removeprefix("Pos")) <= args.end]
    _log(f"found {len(poses)} Pos in range")

    targets: list[tuple[Path, str]] = []
    for pos in poses:
        cs = pos / "output_phase" / "channels" / "crop_sub_rawraw"
        for z in args.z_planes:
            z_root = cs / z
            if not z_root.is_dir():
                continue
            if (not args.force) and already_done(z_root):
                continue
            targets.append((z_root, z))

    _log(f"queued {len(targets)} (Pos, z) targets to analyse")
    if not targets:
        _log("nothing to do")
        return

    for i, (z_root, z) in enumerate(targets, 1):
        pos_name = z_root.parent.parent.parent.parent.name  # crop_sub_rawraw/<z>'s .parent chain
        _log(f"--- [{i}/{len(targets)}] {pos_name}/{z}: {z_root} ---")
        cmd = [
            PY,
            str(SCRIPTS / "batch_all_channels.py"),
            "--root", str(z_root),
            "--ri-calibration", str(CAL_PATH),
            "--media-schedule", MEDIA_SCHEDULE,
            "--frame-min", str(FRAME_MIN),
            "--frame-max", str(FRAME_MAX),
            "--ch-workers", str(args.ch_workers),
        ]
        _log(">>> " + " ".join(shlex.quote(c) for c in cmd))
        proc = subprocess.run(cmd)
        rc = proc.returncode
        if rc != 0:
            _log(f"!! {pos_name}/{z} batch_all_channels rc={rc}, continuing")

    _log("=== 260426 zstack chain DONE ===")


if __name__ == "__main__":
    main()
