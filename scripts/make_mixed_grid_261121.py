"""make_mixed_grid_261121.py

Reference grid folder for the 261121 timelapse after the 11x11 re-shoot:
  Pos1-90   -> D:\\...\\261121\\recon_0p03um_11x11   (11 x 11 @ 0.03 um, taken 2026-11-22 12:18-14:10,
                                                     centred on where the timelapse parks each Pos)
  Pos91-123 -> D:\\...\\261121\\recon_0p03um_25x25   (25 x 25 @ 0.03 um, taken 2026-11-22 04:30-07:06)
  Pos0      -> the 11x11 folder

Nothing is copied or moved: the mixed folder holds directory junctions to the point folders of the two
grids and copies of the small grid_calibration_PosN.json files, so both grids stay intact and the
session can be pointed back at either of them.

    python scripts/make_mixed_grid_261121.py            # build
    python scripts/make_mixed_grid_261121.py --check    # verify only
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import _winapi

ROOT = Path(r"D:\AquisitionData\Kitagishi\261121")
G11 = ROOT / "recon_0p03um_11x11"
G25 = ROOT / "recon_0p03um_25x25"
OUT = ROOT / "recon_mixed_11x11_Pos1-90_25x25_Pos91-123"
NEW_POS = set(range(0, 91))
POINT_RE = re.compile(r"^Pos(\d+)_x[+-]\d+_y[+-]\d+$")


def source_of(pos):
    return G11 if pos in NEW_POS else G25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    n_link = n_cal = 0
    per_pos = {}
    for src in (G11, G25):
        for d in src.iterdir():
            m = POINT_RE.match(d.name)
            if not m or not d.is_dir():
                continue
            pos = int(m.group(1))
            if source_of(pos) != src:
                continue
            per_pos[pos] = per_pos.get(pos, 0) + 1
            dst = OUT / d.name
            if not dst.exists():
                if args.check:
                    raise SystemExit(f"missing junction: {dst}")
                _winapi.CreateJunction(str(d), str(dst))
                n_link += 1
    for pos in sorted(per_pos):
        if pos == 0:
            continue
        cal = source_of(pos) / f"grid_calibration_Pos{pos}.json"
        if not cal.exists():
            raise SystemExit(f"no calibration for Pos{pos}: {cal}")
        dst = OUT / cal.name
        if not dst.exists() or dst.read_bytes() != cal.read_bytes():
            if args.check:
                raise SystemExit(f"calibration differs or missing: {dst}")
            shutil.copyfile(cal, dst)
            n_cal += 1
        n_pts = len(json.load(open(dst, encoding="utf-8"))["positions"])
        if n_pts != per_pos[pos]:
            raise SystemExit(f"Pos{pos}: {per_pos[pos]} point folders but {n_pts} calibrated points")
        rois = OUT / f"Pos{pos}_x+0_y+0" / "output_phase" / "channels" / "channel_rois.json"
        if not rois.exists():
            raise SystemExit(f"no channel_rois.json through the junction: {rois}")
    sizes = sorted(set(per_pos[p] for p in per_pos if p != 0))
    print(f"{OUT}: {len(per_pos) - 1} sample Pos + Pos0; points per Pos {sizes}; "
          f"121-point Pos: {sum(1 for p, n in per_pos.items() if n == 121 and p)}; 625-point Pos: {sum(1 for p, n in per_pos.items() if n == 625)}; "
          f"new junctions {n_link}, calibration files copied {n_cal}")


if __name__ == "__main__":
    main()
