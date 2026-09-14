"""Parallel deletion of output_phase / output_phase_raw under both grid
roots, for the 260405_acute_z18_200h dataset. Coexists with the slower
PowerShell cleanup that was started earlier — racing rmtree is harmless
since FileNotFoundError is swallowed."""
from __future__ import annotations
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

GRIDS = [
    Path(r"F:\260405_acute_z18_200h\grid_2pergluc_60ms_1"),
    Path(r"F:\260405_acute_z18_200h\grid_0pergluc_60ms_1"),
]


def _kill(target: str) -> tuple[str, bool]:
    p = Path(target)
    if not p.exists():
        return (target, False)
    try:
        shutil.rmtree(p, ignore_errors=True)
        return (target, not p.exists())
    except Exception:
        return (target, False)


def main() -> None:
    targets: list[str] = []
    for g in GRIDS:
        if not g.is_dir():
            print(f"[warn] grid not found: {g}", file=sys.stderr)
            continue
        for pos in g.iterdir():
            if not pos.is_dir():
                continue
            for sub in ("output_phase", "output_phase_raw"):
                p = pos / sub
                if p.exists():
                    targets.append(str(p))
    print(f"[info] {len(targets)} target dirs to remove across {len(GRIDS)} grids")

    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        futs = {ex.submit(_kill, t): t for t in targets}
        for f in as_completed(futs):
            t, ok = f.result()
            done += 1
            if done % 500 == 0:
                rate = done / max(time.time() - t0, 0.01)
                print(f"[info] {done}/{len(targets)} dirs removed  ({rate:.1f}/s)")
    print(f"[done] {done}/{len(targets)} dirs removed in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
