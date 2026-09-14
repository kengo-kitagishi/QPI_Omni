"""refit_tilt_right_260517.py - redo the 2-pi + linear tilt removal of the 260517
channel crops with the background fitted on the RIGHT third (Pos53..Pos104).

Why: in Pos >= 53 the trap is mirrored (cells at the left, open end at the
right), but batch_grid_subtract_260517.py handed grid_subtract a ``PosN\\z000``
directory, so grid_subtract could not parse the Pos number and fell back to
``fit_right=False`` for every position. The tilt was therefore fitted on the
cell side for Pos >= 53, which pushes the true background (right third) to
-0.1..-1.7 rad and biases total_phase / RI / mass of those positions.

Why this is exact: the stored crops are the full (40, 270) tilt window that
grid_subtract fitted on (OUTPUT_CROP_H = TILT_CROP_H_RAW = 270), and the model
is a plane along the channel axis. Subtracting one plane and then another is
still a single plane subtraction, so re-fitting the stored crop on the right
third gives the same image the pipeline would have produced with
``fit_right=True``. The same ``ecc_utils.apply_2pi_tilt_crop`` is used.

Input : H:\\260517\\2per_0055per_0per_2per_crop_sub\\PosN\\output_phase\\channels\\crop_sub_rawraw\\z000\\chNN\\img_*_ph_000_phase.tif
Output: D:\\260517_tiltfix\\PosN\\output_phase\\channels\\crop_sub_rawraw\\z000\\chNN\\<same file names>  (float32)
        + the per-Pos JSON sidecars copied for provenance, + <chNN>\\_REFIT_DONE marker.
H: is never written to.

Usage:
    python scripts/refit_tilt_right_260517.py --pos 53            # one position
    python scripts/refit_tilt_right_260517.py --start 53 --end 104
    python scripts/refit_tilt_right_260517.py --pos 53 --ch ch02 --max-files 200   # quick test
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from ecc_utils import apply_2pi_tilt_crop  # noqa: E402

SRC_ROOT = Path(r"H:\260517\2per_0055per_0per_2per_crop_sub")
DST_ROOT = Path(r"D:\260517_tiltfix")
REL = Path("output_phase") / "channels" / "crop_sub_rawraw" / "z000"
TILT_CROP_H = 270
FIT_RIGHT = True
LOG = SCRIPTS / "refit_tilt_right_260517.log"
_FRAME_RE = re.compile(r"img_(\d+)_")


def _log(msg: str) -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def refit_image(img: np.ndarray) -> np.ndarray:
    """2-pi offset + linear tilt on the right third, no re-cropping (270 -> 270)."""
    img64 = np.asarray(img, dtype=np.float64)
    if img64.shape[1] != TILT_CROP_H:
        raise ValueError(f"unexpected crop width {img64.shape}; expected (*, {TILT_CROP_H})")
    out = apply_2pi_tilt_crop(img64, TILT_CROP_H, TILT_CROP_H, fit_right=FIT_RIGHT)
    return out.astype(np.float32)


def _one_file(src: Path, dst: Path) -> bool:
    if dst.exists():
        return False
    img = tifffile.imread(str(src))
    tifffile.imwrite(str(dst), refit_image(img))
    return True


def refit_channel(pos: str, ch: str, max_files: int | None = None, workers: int = 4) -> tuple[int, int, float]:
    src = SRC_ROOT / pos / REL / ch
    dst = DST_ROOT / pos / REL / ch
    marker = dst / "_REFIT_DONE"
    if marker.exists() and not max_files:
        return 0, 0, 0.0
    dst.mkdir(parents=True, exist_ok=True)
    files = sorted((p for p in src.glob("img_*_ph_000_phase.tif")),
                   key=lambda p: int(_FRAME_RE.search(p.name).group(1)))
    if max_files:
        files = files[:max_files]
    t0 = time.time()
    n_new = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_one_file, f, dst / f.name) for f in files]
        for fut in as_completed(futs):
            n_new += int(fut.result())
    dt = time.time() - t0
    if not max_files:
        marker.write_text(f"{len(files)} files, fit_right={FIT_RIGHT}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return len(files), n_new, dt


def copy_sidecars(pos: str) -> None:
    """Copy the small JSON sidecars (provenance); never the large TIFFs."""
    src_ch = SRC_ROOT / pos / "output_phase" / "channels"
    dst_ch = DST_ROOT / pos / "output_phase" / "channels"
    dst_ch.mkdir(parents=True, exist_ok=True)
    for name in ("channel_rois.json", "grid_subtract_log.json", "pos_shifts_cal_online.json"):
        p = src_ch / name
        if p.exists() and not (dst_ch / name).exists():
            shutil.copy2(p, dst_ch / name)
    z_src, z_dst = src_ch / "crop_sub_rawraw" / "z000", dst_ch / "crop_sub_rawraw" / "z000"
    z_dst.mkdir(parents=True, exist_ok=True)
    p = z_src / "correct_0pergluc_log.json"
    if p.exists() and not (z_dst / p.name).exists():
        shutil.copy2(p, z_dst / p.name)
    (z_dst / "TILT_REFIT.txt").write_text(
        "Crops re-flattened from H: with ecc_utils.apply_2pi_tilt_crop(fit_right=True) on the right third "
        "(refit_tilt_right_260517.py). Original grid_subtract run used fit_right=False for all Pos.\n",
        encoding="utf-8")


def refit_pos(pos_num: int, max_files: int | None = None, workers: int = 4, only_ch: str | None = None) -> None:
    pos = f"Pos{pos_num}"
    z = SRC_ROOT / pos / REL
    if not z.is_dir():
        _log(f"{pos}: no source dir {z}")
        return
    copy_sidecars(pos)
    chs = sorted(p.name for p in z.iterdir() if p.is_dir() and p.name.startswith("ch"))
    if only_ch:
        chs = [c for c in chs if c == only_ch]
    t0 = time.time()
    tot = 0
    for ch in chs:
        n, n_new, dt = refit_channel(pos, ch, max_files=max_files, workers=workers)
        tot += n
        if n:
            _log(f"{pos}/{ch}: {n} frames ({n_new} written) in {dt:.0f}s = {n / max(dt, 1e-9):.1f} f/s")
        else:
            _log(f"{pos}/{ch}: already done")
    if not max_files and not only_ch:
        (DST_ROOT / pos / "_REFIT_DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    _log(f"{pos} DONE: {tot} frames in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pos", type=int, default=None)
    ap.add_argument("--start", type=int, default=53)
    ap.add_argument("--end", type=int, default=104)
    ap.add_argument("--ch", default=None, help="single channel (test)")
    ap.add_argument("--max-files", type=int, default=None, help="cap per channel (test; no markers written)")
    ap.add_argument("--workers", type=int, default=4, help="concurrent file reads per channel")
    args = ap.parse_args()
    poss = [args.pos] if args.pos is not None else list(range(args.start, args.end + 1))
    _log(f"=== refit start: Pos {poss[0]}..{poss[-1]} fit_right={FIT_RIGHT} workers={args.workers} ===")
    for n in poss:
        refit_pos(n, max_files=args.max_files, workers=args.workers, only_ch=args.ch)
    _log("=== refit end ===")


if __name__ == "__main__":
    main()
