"""Run seg + lineage tracker + per_channel_figures + batch_figures over every
position of the 260517 2per_0055per_0per_2per dataset (glucose media-switch:
2% -> 0.0055% -> 0% -> 2%, single z plane z000, ~3748 frames/ch).

For every PosN/output_phase/channels/crop_sub_rawraw/z000 that has ch*/ subdirs,
this calls batch_all_channels.py with the parameters validated on Pos1:
  --model-path     <d20 Omnipose checkpoint>            (unchanged)
  --ri-calibration <260517 ri_calibration_results.json> (wo_0p0055 == wo_0)
  --media-schedule "0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2"  (all 104 Pos)
  --bad-frames     <drift_session_20260521T2142/bad_frames.json> (per-Pos)
  --pixel-size-um  0.34567514677103717   (from grid_subtract_log.json)
  --time-interval-min 5.0
  --ch-workers 4
Full frame range (no --frame-min/max) so img_0 -> time 0 and the media schedule
(keyed on absolute img_NNN) aligns exactly; empty traps are auto-skipped by
07_segmentation.py's NO_CELL_BREAK_AFTER.

Already-analysed Pos (any ch with lineage_data3D.csv under inference_out/
lineage_out/) are skipped, so a kill / restart is safe. Pos1 is therefore
skipped automatically."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"F:\260517\2per_0055per_0per_2per_crop_sub")
SCRIPTS = Path(__file__).resolve().parent
PY = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
LOG = SCRIPTS / "_chain_seg_260517.log"

MODEL = (r"C:\Users\QPI\Desktop\train\omni_model_d20\models\\"
         r"cellpose_residual_on_style_on_concatenation_off_omni_abstract_"
         r"nclasses_3_nchan_1_dim_2_omni_model_d20_2026_05_01_19_01_52.321350")
CAL = Path(r"F:\260517\grid_2pergluc_2\ri_calibration_results.json")
BAD = Path(r"F:\260517\drift_session_20260521T2142\bad_frames.json")
MEDIA_SCHEDULE = "0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2"
PIXEL_UM = "0.34567514677103717"
DT_MIN = "5.0"
# Drop low-quality T=0,1 (spurious non-cell masks). With frame-min 2 the tracker
# re-bases time so img_2 is exactly time 0 (time_zero_frame=frame_min). Media
# schedule stays keyed on absolute img_NNN, so 2019/2307/2885 are unaffected.
FRAME_MIN = 2
Z = "z000"


def _stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    line = f"[{_stamp()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def already_done(z_root: Path) -> bool:
    """True only if EVERY image-bearing channel has lineage_data3D.csv.

    Coarse "any channel done" skipping leaves crash-partial positions (some
    channels tracked, others not) permanently skipped. Here a position counts as
    done only when all channels that actually contain phase images have a lineage
    CSV. Channels with no img_*.tif (e.g. Pos47 ch11, an upstream gap) are not
    required, so they don't block completion. Empty traps still produce a
    lineage CSV (0 rows), so they satisfy the check."""
    if not z_root.is_dir():
        return False
    data_chs = [
        ch for ch in z_root.iterdir()
        if ch.is_dir() and ch.name.startswith("ch")
        and any(ch.glob("img_*_ph_*.tif"))
    ]
    if not data_chs:
        return False
    return all(
        (ch / "inference_out" / "lineage_out" / "lineage_data3D.csv").exists()
        for ch in data_chs
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ch-workers", type=int, default=4)
    ap.add_argument("--force", action="store_true",
                    help="re-analyse Pos even if lineage_data3D.csv exists")
    ap.add_argument("--start", type=int, default=1, help="lowest Pos number")
    ap.add_argument("--end", type=int, default=104, help="highest Pos number")
    ap.add_argument("--skip-seg", action="store_true",
                    help="reuse existing masks (no GPU seg); re-track only. "
                         "For re-tracking already-segmented Pos with the corrected tracker.")
    args = ap.parse_args()

    LOG.write_text(f"=== 260517 chain START {_stamp()} ===\n", encoding="utf-8")
    _log(f"ch_workers={args.ch_workers}  Pos {args.start}..{args.end}  force={args.force}")
    for must in (CAL, BAD):
        if not must.exists():
            _log(f"FATAL: required file not found: {must}")
            sys.exit(1)

    poses = sorted(
        (p for p in ROOT.iterdir()
         if p.is_dir() and p.name.startswith("Pos") and p.name[3:].isdigit()),
        key=lambda p: int(p.name[3:]),
    )
    poses = [p for p in poses if args.start <= int(p.name[3:]) <= args.end]
    _log(f"found {len(poses)} Pos in range")

    targets: list[Path] = []
    for pos in poses:
        z_root = pos / "output_phase" / "channels" / "crop_sub_rawraw" / Z
        if not z_root.is_dir():
            _log(f"skip {pos.name}: no {Z} dir")
            continue
        if (not args.force) and already_done(z_root):
            _log(f"skip {pos.name}: already has lineage_data3D.csv")
            continue
        targets.append(z_root)

    _log(f"queued {len(targets)} Pos to analyse")
    if not targets:
        _log("nothing to do")
        return

    for i, z_root in enumerate(targets, 1):
        pos_name = z_root.parent.parent.parent.parent.name
        _log(f"--- [{i}/{len(targets)}] {pos_name}: {z_root} ---")
        cmd = [
            PY, "-u", str(SCRIPTS / "batch_all_channels.py"),
            "--root", str(z_root),
            "--model-path", MODEL,
            "--ri-calibration", str(CAL),
            "--media-schedule", MEDIA_SCHEDULE,
            "--bad-frames", str(BAD),
            "--pixel-size-um", PIXEL_UM,
            "--time-interval-min", DT_MIN,
            "--wavelength-nm", "658.0",
            "--alpha-ri", "0.00018",
            "--frame-min", str(FRAME_MIN),
            "--ch-workers", str(args.ch_workers),
        ]
        if args.skip_seg:
            cmd.append("--skip-seg")
        _log(">>> " + " ".join(shlex.quote(c) for c in cmd))
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            _log(f"!! {pos_name} batch_all_channels rc={rc}, continuing")

    _log("=== 260517 chain DONE ===")


if __name__ == "__main__":
    main()
