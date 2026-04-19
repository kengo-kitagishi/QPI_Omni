"""
timelapse_plane_bgsub.py
------------------------
For output_phase frames:
  1. Sub-pixel correction using shift values from pos_shifts.json
  2. Select nearest grid position and subtract
  3. 2D linear plane fit on left BG_MASK_FRAC region -> subtract from entire image
  4. Output per-pixel std map of processed frames (quality evaluation)

Output:
  TL_DIR/output_phase_planesub/img_*_ph_000_phase.tif  (float32)
  TL_DIR/output_phase_planesub/plane_bgsub_log.json
  results/figures/ for std map + residual time series (via figure_logger)
"""
import sys
import json
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from grid_subtract import (
    scan_grid_positions,
    find_nearest_grid,
    load_grid_image,
    apply_inverse_shift_warp,
    load_timelapse_frames,
)
from figure_logger import save_figure

# ============================================================
# Configuration parameters
# ============================================================
TL_DIR    = Path(r"F:\timelapse_11day_exp200ms_1pos_EMM2\Pos1")
GRID_DIR  = Path(r"F:\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1")
BASE_LABEL = "Pos1"

TL_Z_INDEX   = 0   # img_*_ph_000_phase.tif
GRID_Z_INDEX = 5   # Grid reference z index

# Optical parameters
SENSOR_PIXEL_SIZE = 3.45e-6  # [m]
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511

# Shift signs (ECC output sign convention: same as grid_subtract.py)
SHIFT_SIGN_X = 1
SHIFT_SIGN_Y = 1

# Grid step [um]
X_STEP = 0.1
Y_STEP = 0.1

# 2D plane fit: use center BG_MASK_FRAC of left half as background region
# Exclude (1-BG_MASK_FRAC)/2 from both edges of left half (to avoid edge artifacts)
BG_MASK_FRAC = 0.80

# Max frames for std map computation (None = all frames)
STD_SUBSAMPLE = 200

# For testing: None for all frames, integer for first N frames only
MAX_FRAMES = None
# ============================================================


def fit_plane_2d(img: np.ndarray, col_start: int, col_end: int):
    """
    Least-squares fit of linear plane z = a*x + b*y + c to img[:, col_start:col_end].
    Returns the plane evaluated over the full image and coefficients (a, b, c).

    Parameters
    ----------
    img : (H, W) float64
    col_start : int  left column index for fitting region (inclusive)
    col_end   : int  right column index for fitting region (exclusive)

    Returns
    -------
    plane : (H, W) float64  plane evaluated over the full image
    coeffs : (a, b, c)
    """
    H, W = img.shape
    # Coordinate grid for fitting region (xs=column direction, ys=row direction)
    ys, xs = np.mgrid[0:H, col_start:col_end]
    z = img[:, col_start:col_end].ravel()
    A = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    # Evaluate over the full image
    ys_f, xs_f = np.mgrid[0:H, 0:W]
    plane = a * xs_f + b * ys_f + c
    return plane, (float(a), float(b), float(c))


def main():
    # ---- Initialization ----
    tl_dir  = TL_DIR
    out_dir = tl_dir / "output_phase_planesub"
    shifts_json = tl_dir / "output_phase" / "channels" / "pos_shifts.json"

    if not tl_dir.exists():
        print(f"ERROR: TL_DIR not found: {tl_dir}")
        sys.exit(1)
    if not shifts_json.exists():
        print(f"ERROR: pos_shifts.json not found: {shifts_json}")
        sys.exit(1)

    # ---- Load shift data ----
    with open(shifts_json, encoding="utf-8") as f:
        shifts_data = json.load(f)
    frame_results = shifts_data.get("frame_results") or shifts_data.get("alignment_results")
    if not frame_results:
        print("ERROR: frame_results not found in pos_shifts.json")
        sys.exit(1)

    n_frames = len(frame_results)
    if MAX_FRAMES is not None and MAX_FRAMES < n_frames:
        frame_results = frame_results[:MAX_FRAMES]
        n_frames = MAX_FRAMES
        print(f"[TEST] Limiting to {n_frames} frames")
    print(f"Number of frames: {n_frames}")

    # ---- Timelapse frame list ----
    tl_frames = load_timelapse_frames(tl_dir, TL_Z_INDEX)
    if not tl_frames:
        print(f"ERROR: TL frames not found")
        sys.exit(1)
    if len(tl_frames) < n_frames:
        print(f"WARNING: TIF files {len(tl_frames)} < pos_shifts frames {n_frames}")
        n_frames = len(tl_frames)
        frame_results = frame_results[:n_frames]
    print(f"TIF files: {len(tl_frames)}")

    # ---- Grid scan ----
    pos_map = scan_grid_positions(GRID_DIR, BASE_LABEL)
    if not pos_map:
        print(f"ERROR: Grid positions not found: {GRID_DIR}/{BASE_LABEL}_x*_y*")
        sys.exit(1)
    print(f"Grid positions: {len(pos_map)}")

    # ---- pixel scale ----
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")

    # ---- Frame shift list ----
    frame_shifts = []
    for r in frame_results:
        sx = r.get("shift_x_avg") or r.get("shift_x") or 0.0
        sy = r.get("shift_y_avg") or r.get("shift_y") or 0.0
        frame_shifts.append((sx, sy))

    # ---- Output directory ----
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ---- Grid image cache ----
    grid_img_cache: dict = {}

    def get_grid_image(xi, yi):
        key = (xi, yi)
        if key not in grid_img_cache:
            pos_dir = pos_map[key]
            grid_img_cache[key] = load_grid_image(pos_dir, GRID_Z_INDEX)
        return grid_img_cache[key]

    # ---- BG mask column range ----
    # Use center BG_MASK_FRAC of left half (exclude edge artifacts)
    sample_img = tifffile.imread(str(tl_frames[0]))
    H, W = sample_img.shape
    left_half = W // 2
    margin = int(left_half * (1 - BG_MASK_FRAC) / 2)
    mask_col_start = margin
    mask_col_end   = left_half - margin
    print(f"Image size: {H}x{W}, BG mask cols: {mask_col_start}~{mask_col_end} "
          f"(center {BG_MASK_FRAC*100:.0f}% of left half)")

    # ---- Main loop ----
    subtract_log = []

    for t in tqdm(range(n_frames), desc="grid subtract + plane bgsub"):
        sx, sy = frame_shifts[t]

        # Nearest grid selection (same as grid_subtract.py L279-293)
        dx_um = SHIFT_SIGN_X * sy * pixel_scale_um
        dy_um = SHIFT_SIGN_Y * sx * pixel_scale_um
        (xi, yi), dist_um = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)

        # Residual sub-pixel shift
        residual_x = SHIFT_SIGN_Y * sx - yi * Y_STEP / pixel_scale_um
        residual_y = SHIFT_SIGN_X * sy - xi * X_STEP / pixel_scale_um

        # Load timelapse frame
        tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)

        # Sub-pixel correction
        if residual_x != 0.0 or residual_y != 0.0:
            tl_warped = apply_inverse_shift_warp(tl_img, residual_x, residual_y)
        else:
            tl_warped = tl_img

        # Nearest grid subtraction
        grid_img = get_grid_image(xi, yi)
        grid_sub = tl_warped - grid_img

        # 2D plane fit & subtraction
        bg_mean_before = float(np.mean(grid_sub[:, mask_col_start:mask_col_end]))
        bg_std_before  = float(np.std(grid_sub[:, mask_col_start:mask_col_end]))
        plane, (a, b, c) = fit_plane_2d(grid_sub, mask_col_start, mask_col_end)
        result = grid_sub - plane
        bg_mean_after = float(np.mean(result[:, mask_col_start:mask_col_end]))
        bg_std_after  = float(np.std(result[:, mask_col_start:mask_col_end]))

        # Save
        tifffile.imwrite(str(out_dir / tl_frames[t].name), result.astype(np.float32))

        subtract_log.append({
            "frame_index": t,
            "shift_x_px": sx,
            "shift_y_px": sy,
            "dx_um": dx_um,
            "dy_um": dy_um,
            "grid_xi": xi,
            "grid_yi": yi,
            "grid_dist_um": dist_um,
            "residual_x_px": residual_x,
            "residual_y_px": residual_y,
            "plane_a": a,
            "plane_b": b,
            "plane_c": c,
            "bg_mean_before": bg_mean_before,
            "bg_std_before": bg_std_before,
            "bg_mean_after": bg_mean_after,
            "bg_std_after": bg_std_after,
        })

    # ---- Save log ----
    log_path = out_dir / "plane_bgsub_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "tl_dir": str(tl_dir),
            "grid_dir": str(GRID_DIR),
            "base_label": BASE_LABEL,
            "tl_z_index": TL_Z_INDEX,
            "grid_z_index": GRID_Z_INDEX,
            "bg_mask_frac": BG_MASK_FRAC,
            "pixel_scale_um": pixel_scale_um,
            "n_frames_processed": n_frames,
            "frame_log": subtract_log,
        }, f, ensure_ascii=False, indent=2)
    print(f"Log saved: {log_path}")

    # ---- Quality evaluation: per-pixel std map ----
    print("Computing std map...")
    processed_files = sorted(out_dir.glob(f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"))
    n_files = len(processed_files)

    if STD_SUBSAMPLE is not None and n_files > STD_SUBSAMPLE:
        indices = np.linspace(0, n_files - 1, STD_SUBSAMPLE, dtype=int)
    else:
        indices = np.arange(n_files)

    stack = np.stack([
        tifffile.imread(str(processed_files[i])).astype(np.float32)
        for i in tqdm(indices, desc="std map loading")
    ], axis=0)  # (N, H, W)

    std_map = np.std(stack, axis=0)  # (H, W)

    # Right half residual time series (independent verification region not used for fitting)
    right_region_mean = stack[:, :, left_half:].mean(axis=(1, 2))  # (N,) entire right half
    frame_times = indices  # use frame index as proxy

    # ---- Visualization ----
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.4])

    # Panel 1: std map
    ax1 = fig.add_subplot(gs[0])
    im = ax1.imshow(std_map, cmap="hot", origin="upper")
    ax1.axvline(mask_col_start, color="cyan", lw=1, ls="--", alpha=0.7, label=f"BG mask start (col {mask_col_start})")
    ax1.axvline(mask_col_end,   color="cyan", lw=1, ls="--", alpha=0.7, label=f"BG mask end (col {mask_col_end})")
    ax1.axvline(left_half, color="lime", lw=1, ls="-", alpha=0.6, label=f"Left/Right half (col {left_half})")
    plt.colorbar(im, ax=ax1, label="std [rad]")
    ax1.set_title(f"Per-pixel std  (N={len(indices)} frames)")
    ax1.legend(fontsize=8)

    # Panel 2: right half residual time series
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(frame_times, right_region_mean, lw=0.8, color="steelblue")
    ax2.axhline(0, color="k", lw=0.5, ls="--")
    ax2.set_xlabel("Frame index")
    ax2.set_ylabel("Mean phase [rad]")
    ax2.set_title("Right half mean (ground truth BG, independent check)")

    fig.suptitle("2D plane bgsub quality check", fontsize=12)
    fig.tight_layout()

    save_figure(
        fig,
        params={
            "bg_mask_frac": BG_MASK_FRAC,
            "n_frames_total": n_files,
            "n_frames_std": int(len(indices)),
            "std_map_median": float(np.median(std_map)),
            "std_map_max": float(np.max(std_map)),
            "right_region_mean_std": float(np.std(right_region_mean)),
        },
        description="per-pixel std map after 2D plane background subtraction (grid-subtracted timelapse)",
    )
    plt.close(fig)

    print(f"\n--- Done ---")
    print(f"Frames processed: {n_frames}")
    print(f"Output: {out_dir}")
    print(f"std map median: {np.median(std_map):.4f} rad")
    print(f"std map max: {np.max(std_map):.4f} rad")
    print(f"Right half residual std: {np.std(right_region_mean):.4f} rad")


if __name__ == "__main__":
    main()
