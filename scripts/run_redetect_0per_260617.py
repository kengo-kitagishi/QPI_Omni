"""
Re-run channel detection for the 0per grid (260617) with a side-restricted cx
search range, so all channels of a Pos converge onto the single channel wall
(~390 for right-side Pos<POS_SPLIT, ~110 for left-side Pos>=POS_SPLIT) instead of
scattering onto background-tilt garbage.

PARAMETER-ONLY fix: only --cx-min/--cx-max change per side. The channel_crop.py
detection logic is unchanged. Each Pos saves its channel_detection_preview.png
(under output_phase/channels/) for visual verification.

Does NOT recalibrate -- run grid calibration afterwards once previews look right.
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

SD = Path(__file__).resolve().parent
PY = sys.executable
GRID = Path(r"E:\260617\0per_grid_0p05um_1")
POS_SPLIT = 53
PATTERN = "*_ph_005_phase.tif"
RIGHT_CX = (350, 420)   # Pos <  POS_SPLIT : wall near ~390
LEFT_CX = (80, 180)     # Pos >= POS_SPLIT : wall near ~110

pat = re.compile(r"^(Pos\d+)_x\+0_y\+0$")


def main():
    poss = sorted(
        [d for d in GRID.iterdir()
         if d.is_dir() and pat.match(d.name) and not d.name.startswith("Pos0_")],
        key=lambda d: int(re.search(r"\d+", d.name).group()),
    )
    print(f"[redetect] {len(poss)} Pos", flush=True)
    ok = err = 0
    for d in poss:
        label = pat.match(d.name).group(1)
        n = int(label[3:])
        op = d / "output_phase"
        rois = op / "channels" / "channel_rois.json"
        # preserve the ORIGINAL rois once
        bak = rois.with_suffix(".json.bak")
        if rois.exists() and not bak.exists():
            shutil.copy2(rois, bak)
        cxmin, cxmax = LEFT_CX if n >= POS_SPLIT else RIGHT_CX
        r = subprocess.run(
            [PY, str(SD / "channel_crop.py"), "--dir", str(op), "--detect",
             "--pattern", PATTERN, "--cx-min", str(cxmin), "--cx-max", str(cxmax)],
            cwd=str(SD), capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        good = r.returncode == 0 and rois.exists()
        if good:
            ok += 1
        else:
            err += 1
        print(f"  {label} (cx[{cxmin},{cxmax}]): {'OK' if good else 'FAIL'}", flush=True)
        if not good:
            print((r.stderr or r.stdout)[-300:], flush=True)
    print(f"\n[redetect] done: {ok} OK, {err} FAIL", flush=True)
    print(f"[redetect] previews: {GRID}\\PosN_x+0_y+0\\output_phase\\channels\\channel_detection_preview.png", flush=True)


if __name__ == "__main__":
    main()
