"""Make room on E: during the 261029 grid: move the 261024 grid holograms of Pos100-118 to C:.

E:/261024/2per_grid_1 is the raw of a grid that was reconstructed, calibrated and verified on
2026-10-24 (E:/261024/recon_2per_grid_1); its raw was only kept as a spare. Each point folder is
copied to C:/261024_grid_raw_parked/2per_grid_1/<point>, every file is size-checked, and only then is
the E: folder removed. Nothing is deleted without a verified copy.
"""
import os
import re
import shutil
import sys
from pathlib import Path

SRC = Path(r"E:\261024\2per_grid_1")
DST = Path(r"C:\261024_grid_raw_parked\2per_grid_1")
POS_MIN, POS_MAX = 100, 118
POINT_RE = re.compile(r"^Pos(\d+)_x[+-]\d+_y[+-]\d+$")


def main():
    pts = sorted(d for d in SRC.iterdir() if d.is_dir() and (m := POINT_RE.match(d.name))
                 and POS_MIN <= int(m.group(1)) <= POS_MAX)
    moved = freed = 0
    for d in pts:
        out = DST / d.name
        out.mkdir(parents=True, exist_ok=True)
        ok = True
        for f in d.iterdir():
            t = out / f.name
            shutil.copy2(f, t)
            ok &= t.stat().st_size == f.stat().st_size
        if not ok:
            print(f"size mismatch in {d.name}; E: copy kept", flush=True)
            continue
        size = sum(f.stat().st_size for f in d.iterdir())
        shutil.rmtree(d)
        moved += 1
        freed += size
        if moved % 200 == 0:
            print(f"{moved}/{len(pts)} points moved, {freed / 1e9:.1f} GB freed, "
                  f"E: free {shutil.disk_usage('E:/').free / 1e9:.0f} GB", flush=True)
    print(f"DONE {moved}/{len(pts)} points moved to {DST}, {freed / 1e9:.1f} GB freed on E:", flush=True)
    sys.exit(0 if moved == len(pts) else 1)


if __name__ == "__main__":
    main()
