"""Re-run channel detection on the 260917 single-z grid with a shifted search range.

The MDA positions were moved by hand before the grid was acquired: Y +10 um for Pos0-51 and
Y -10 um for Pos52-99. Stage Y maps to the image long axis, so the traps moved along it by
+21 px (first half) and -36 px (second half), measured against the previous grid.

channel_crop.py --detect looks for the channel edge only inside [cx_min, cx_max] (default
100..400) and skips anything outside, so with the field moved the peaks landed on the search
bounds: 765 of 1092 channels sat at cx <= 110 or cx >= 390, and 79 channels were dropped
(1171 -> 1092). Shifting the search range by the measured amount puts the detection back where
it was: Pos5 and Pos60 returned 12 channels each, matching the previous grid, with cy within
2 px.

    python redetect_channels_260917_g4.py            # all Pos
    python redetect_channels_260917_g4.py --pos 5 60
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GRID_DIR = Path(r"D:\AquisitionData\Kitagishi\260917\grid_hologram_0p05_4")
PATTERN = "img_*_ph_000_phase.tif"      # single-z grid: the one plane is index 0
POS_SPLIT = 52
CX_FIRST = (121, 421)      # Pos < POS_SPLIT : default 100..400 shifted by the measured +21 px
CX_SECOND = (64, 364)      # Pos >= POS_SPLIT: shifted by the measured -36 px


def detect_one(pos):
    lo, hi = CX_FIRST if pos < POS_SPLIT else CX_SECOND
    d = GRID_DIR / f"Pos{pos}_x+0_y+0" / "output_phase"
    if not d.is_dir():
        return pos, None, "no output_phase"
    r = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "channel_crop.py"), "--dir", str(d), "--detect",
         "--pattern", PATTERN, "--cx-min", str(lo), "--cx-max", str(hi)],
        cwd=str(SCRIPT_DIR), capture_output=True, text=True, encoding="utf-8", errors="replace")
    rois = d / "channels" / "channel_rois.json"
    if r.returncode != 0 or not rois.exists():
        return pos, None, (r.stderr or r.stdout)[-200:]
    import json
    return pos, len(json.loads(rois.read_text(encoding="utf-8"))), "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, nargs="+", default=list(range(1, 100)))
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    print(f"re-detect {len(a.pos)} Pos: first half {CX_FIRST}, second half {CX_SECOND}", flush=True)
    t0 = time.time()
    counts, failed = {}, []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(detect_one, p): p for p in a.pos}
        for fut in as_completed(futs):
            pos, n, status = fut.result()
            if n is None:
                failed.append((pos, status))
            else:
                counts[pos] = n
    total = sum(counts.values())
    print(f"done in {time.time()-t0:.0f} s: {len(counts)} Pos, {total} channels", flush=True)
    if failed:
        print(f"failed: {failed[:5]} ({len(failed)} total)", flush=True)


if __name__ == "__main__":
    main()
