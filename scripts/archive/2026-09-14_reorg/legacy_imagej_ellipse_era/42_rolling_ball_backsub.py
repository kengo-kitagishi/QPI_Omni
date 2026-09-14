"""
42_rolling_ball_backsub.py

Large-scale background gradient subtraction for phase TIF images.

Two methods available (set METHOD below):
  "gaussian"    : subtract a heavily blurred version of the image (recommended for phase).
                  Estimates the mean background surface → corrected background ≈ 0.
  "rolling_ball": skimage rolling_ball.  Finds the LOWER envelope, so corrected values
                  are always ≥ 0 and shifted upward.  Useful for fluorescence-style images
                  but not ideal for signed phase images.

Outputs per frame:
  - 32-bit float TIF  : background-subtracted (for downstream analysis)
  - 8-bit TIF         : normalized with fixed VMIN/VMAX (timelapse-consistent)
"""

import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import gaussian_filter
from tqdm import tqdm

# ===================== PARAMETERS =====================
INPUT_DIR  = Path(r"e:\Acuisition\kitagishi\260301\movetest_8\Pos1\output_phase")
OUTPUT_DIR = INPUT_DIR.parent / "output_phase_bgsub"

# --- Method ---
# "gaussian"    : Gaussian blur background subtraction (recommended for phase images)
# "rolling_ball": rolling ball (lower-envelope; use for non-negative images)
METHOD = "gaussian"

# Gaussian sigma in pixels.  Larger = remove slower/larger-scale gradients only.
# For a 511x511 image, sigma=150 means the ball "radius" is ~150 px.
# Corresponds roughly to a rolling-ball radius of ~2-3× sigma.
GAUSSIAN_SIGMA = 50

# Rolling ball radius (only used when METHOD = "rolling_ball")
BALL_RADIUS = 250

# Fixed normalization range for 8-bit output (rad).
# Keep constant across the timelapse for temporal consistency.
VMIN = -5.0
VMAX =  2.0

SAVE_FLOAT32 = True   # 32-bit float background-subtracted TIF
SAVE_UINT8   = True   # 8-bit TIF with fixed VMIN/VMAX

# Crop this many pixels from each edge before saving (removes edge artifacts).
# Applied to both float32 and 8-bit outputs.  0 = no crop.
EDGE_CROP = 0
# ======================================================


def estimate_background(img: np.ndarray) -> np.ndarray:
    if METHOD == "gaussian":
        # mode='nearest' repeats edge pixels instead of mirroring, which reduces
        # the influence of edge artifacts on the background estimate.
        return gaussian_filter(img, sigma=GAUSSIAN_SIGMA, mode='nearest')
    elif METHOD == "rolling_ball":
        from skimage.restoration import rolling_ball
        return rolling_ball(img, radius=BALL_RADIUS)
    else:
        raise ValueError(f"Unknown METHOD: {METHOD!r}")


def normalize_to_uint8(img: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    clipped = np.clip(img, vmin, vmax)
    scaled  = (clipped - vmin) / (vmax - vmin) * 255.0
    return scaled.astype(np.uint8)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process reconstructed phase images only (exclude raw *_ph_000.tif)
    tif_files = sorted(INPUT_DIR.glob("*_phase.tif"))
    if not tif_files:
        print(f"No TIF files found in {INPUT_DIR}")
        return

    print(f"Found {len(tif_files)} files")
    print(f"Method      : {METHOD}"
          + (f"  (sigma={GAUSSIAN_SIGMA} px)" if METHOD == "gaussian"
             else f"  (radius={BALL_RADIUS} px)"))
    print(f"8-bit range : [{VMIN}, {VMAX}] rad  (fixed for timelapse)")
    print(f"Edge crop   : {EDGE_CROP} px")
    print(f"Output      : {OUTPUT_DIR}\n")

    for path in tqdm(tif_files, unit="frame"):
        img = tifffile.imread(str(path)).astype(np.float32)
        bg  = estimate_background(img)
        corrected = img - bg

        stem = path.stem

        # Apply edge crop
        if EDGE_CROP > 0:
            c = corrected[EDGE_CROP:-EDGE_CROP, EDGE_CROP:-EDGE_CROP]
        else:
            c = corrected

        if SAVE_FLOAT32:
            tifffile.imwrite(str(OUTPUT_DIR / f"{stem}_bgsub.tif"),
                             c.astype(np.float32))

        if SAVE_UINT8:
            tifffile.imwrite(str(OUTPUT_DIR / f"{stem}_bgsub_8bit.tif"),
                             normalize_to_uint8(c, VMIN, VMAX))

    print(f"\nDone. {len(tif_files)} frames → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
