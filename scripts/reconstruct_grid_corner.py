"""
QPI-reconstruct only a single grid point (default: Pos1_x+0_y+0), and optionally
create channel_rois.json.

Prerequisites:
  - Raw data img_*_ph_*.tif for Pos0_* / Pos1_* must exist at the same (xi, yi)
    (same as pipeline_full Step0)
  - Intended to prepare the minimum set for prepare_drift_session / drift without
    waiting for full-grid reconstruction

Examples:
  python reconstruct_grid_corner.py
  python reconstruct_grid_corner.py --grid-dir "E:\\...\\grid_2pergluc_60ms_1" --rois-only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile

_script_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(_script_dir))

from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

POS_SPLIT = 53
CROP_BEFORE = (0, 2048, 400, 2448)
CROP_AFTER = (0, 2048, 0, 2048)

DEFAULT_GRID_DIR = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
BG_LABEL = "Pos0"
TARGET_LABEL = "Pos1"
XI, YI = 0, 0


def _pos_number_from_label(label: str) -> int:
    m = re.match(r"Pos(\d+)", label)
    return int(m.group(1)) if m else 0


def _get_crop(pos_number: int):
    return CROP_BEFORE if pos_number < POS_SPLIT else CROP_AFTER


def _get_z_index(path: Path) -> int:
    m = re.search(r"_ph_(\d+)", path.stem)
    return int(m.group(1)) if m else -1


def _make_qpi_params(img_path: Path, crop):
    from PIL import Image
    from qpi import QPIParameters

    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    cropped = img[rs:re_, cs:ce]
    return QPIParameters(
        wavelength=WAVELENGTH,
        NA=NA,
        img_shape=cropped.shape,
        pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )


def _reconstruct(img_path: Path, qpi_params, crop) -> np.ndarray:
    from PIL import Image
    from qpi import get_field
    from skimage.restoration import unwrap_phase

    img = np.array(Image.open(str(img_path)))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce]
    field = get_field(img, qpi_params)
    return unwrap_phase(np.angle(field))


def reconstruct_one_point(
    grid_dir: Path,
    bg_label: str,
    target_label: str,
    xi: int,
    yi: int,
    skip_if_exists: bool,
) -> Path:
    """BG-subtracted reconstruction of Pos{target}_x{xi}_y{yi}. Returns output_phase."""
    bg_dir = grid_dir / f"{bg_label}_x{xi:+d}_y{yi:+d}"
    tgt_dir = grid_dir / f"{target_label}_x{xi:+d}_y{yi:+d}"
    out_dir = tgt_dir / "output_phase"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not bg_dir.is_dir():
        raise FileNotFoundError(f"BG folder not found: {bg_dir}")
    if not tgt_dir.is_dir():
        raise FileNotFoundError(f"Target folder not found: {tgt_dir}")

    pos_num = _pos_number_from_label(target_label)
    crop = _get_crop(pos_num)

    z_tgt = {_get_z_index(p): p for p in sorted(tgt_dir.glob("img_*_ph_*.tif"))}
    z_bg = {_get_z_index(p): p for p in sorted(bg_dir.glob("img_*_ph_*.tif"))}
    if not z_tgt:
        raise FileNotFoundError(f"Raw images not found: {tgt_dir}/img_*_ph_*.tif")

    sample = next(iter(z_tgt.values()))
    qpi = _make_qpi_params(sample, crop)

    n_ok = n_skip = 0
    for z_idx, tgt_path in sorted(z_tgt.items()):
        out_path = out_dir / (tgt_path.stem + "_phase.tif")
        if skip_if_exists and out_path.exists():
            n_skip += 1
            continue
        # Get BG phase (pre-reconstructed raw first, reconstruct as fallback)
        phase_bg = None
        bg_raw_subdir = "output_phase_raw" if pos_num < POS_SPLIT else "output_phase_raw_crop_after"
        bg_raw_path = bg_dir / bg_raw_subdir / (tgt_path.stem + "_phase.tif")
        if bg_raw_path.exists():
            phase_bg = tifffile.imread(str(bg_raw_path)).astype(np.float64)
        elif z_idx in z_bg:
            phase_bg = _reconstruct(z_bg[z_idx], qpi, crop)
        if phase_bg is None:
            print(f"  [SKIP z={z_idx}] BG hologram and pre-reconstructed raw both missing")
            continue
        phase = _reconstruct(tgt_path, qpi, crop) - phase_bg
        h, w = phase.shape
        if pos_num < POS_SPLIT:
            region = phase[1 : h - 1, 1 : w // 2]
        else:
            region = phase[1 : h - 1, w // 2 : w - 1]
        if region.size > 0:
            phase -= np.mean(region)
        tifffile.imwrite(str(out_path), phase.astype(np.float32))
        n_ok += 1
        print(f"  wrote {out_path.name}")

    print(f"Reconstruction: new {n_ok} frames, skipped {n_skip} frames -> {out_dir}")
    return out_dir


def run_channel_detect(phase_dir: Path, python_exe: str) -> None:
    """Create channel_rois.json (detect only; --detect is sufficient if channel TIFFs are not needed)."""
    cmd = [
        python_exe,
        str(_script_dir / "channel_crop.py"),
        "--dir",
        str(phase_dir),
        "--pattern",
        "*_phase.tif",
        "--detect",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(description="Reconstruct a single grid corner + optionally detect channel_rois")
    ap.add_argument("--grid-dir", default=DEFAULT_GRID_DIR, help="GRID_DIR (parent of Pos0_x+0_y+0 etc.)")
    ap.add_argument("--bg-label", default=BG_LABEL)
    ap.add_argument("--target-label", default=TARGET_LABEL)
    ap.add_argument("--xi", type=int, default=XI)
    ap.add_argument("--yi", type=int, default=YI)
    ap.add_argument("--no-skip", action="store_true", help="Overwrite and recompute even existing *_phase.tif")
    ap.add_argument("--recon-only", action="store_true", help="Reconstruction only (do not create channel_rois)")
    ap.add_argument("--rois-only", action="store_true", help="Only create channel_rois from existing output_phase")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python used to launch channel_crop (default: current interpreter)",
    )
    args = ap.parse_args()
    grid_dir = Path(args.grid_dir)

    phase_dir = grid_dir / f"{args.target_label}_x{args.xi:+d}_y{args.yi:+d}" / "output_phase"

    if args.rois_only:
        if not phase_dir.is_dir() or not any(phase_dir.glob("*_phase.tif")):
            print(f"ERROR: no *_phase.tif found: {phase_dir}")
            print("Run reconstruction first without --rois-only, or complete pipeline_full Step0.")
            sys.exit(1)
    else:
        reconstruct_one_point(
            grid_dir,
            args.bg_label,
            args.target_label,
            args.xi,
            args.yi,
            skip_if_exists=not args.no_skip,
        )

    if not args.recon_only:
        run_channel_detect(phase_dir, args.python)
        print(f"channel_rois.json: {phase_dir / 'channels' / 'channel_rois.json'}")


if __name__ == "__main__":
    main()
