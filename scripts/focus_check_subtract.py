# %%
"""
focus_check_subtract.py
-----------------------
Focus check script.

Subtracts corresponding z frames between grid/00 (background reference = Pos{N}_x+0_y+0
for each Pos) and focus_test (sample z-stack) to visually verify the focal position.

Alignment is computed once per Pos via ECC and applied to all z frames.

Output: OUTPUT_DIR/Pos{N}/z{z:03d}.tif (float32)
Open Pos{N}/ in ImageJ via File > Import > Image Sequence to browse as a z-stack.
"""

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ===========================================================================
# Parameters (edit here before use)
# ===========================================================================

DO_RECONSTRUCTION = True
GRID_DIR    = r"C:\260416\2per_gridgluc_1"   # Parent directory containing Pos{N}_x+0_y+0
FOCUS_DIR   = r"C:\260416\_for_focus_check_subtract_3"  # Parent directory containing Pos0..PosN
GRID_SUFFIX = "x+0_y+0"           # Suffix for background reference grid position

POS_LABELS  = None     # None=auto-detect, e.g.: ["Pos1", "Pos2"] (Pos0=BG auto-excluded)
ALIGN_Z     = 5                    # z index used for ECC alignment (z=0 = index 5, 260416)

CROP_OUTPUT = False                # True=crop by channel ROI, False=full frame
OUTPUT_DIR  = r"C:\260416\focus_check_subtracted_3"

N_WORKERS   = 4                    # Number of parallel workers (only when DO_RECONSTRUCTION=True)

# Reconstruction parameters (only used when DO_RECONSTRUCTION=True)
# Written following the same convention as batch_reconstruction_grid.py
from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

# Crop selection by Pos number (matching batch_reconstruction_grid.py)
# Pos number < POS_SPLIT -> CROP_BEFORE;  Pos number >= POS_SPLIT -> CROP_AFTER
# OFFAXIS_CENTER is in post-crop coordinates, shared for left/right
# Physical mapping:
#   Low Pos number (< POS_SPLIT) = right channel -> col 400:2448
#   High Pos number (>= POS_SPLIT) = left channel -> col 0:2048
# [!] Left/right may swap depending on dataset. Always verify with actual data.
POS_SPLIT    = 53
CROP_BEFORE  = (0, 2048, 400, 2448)   # pos < POS_SPLIT  -> right channel (col 400-2448)
CROP_AFTER   = (0, 2048,   0, 2048)   # pos >= POS_SPLIT -> left channel (col 0-2048)

FORCE_RECONSTRUCT = False            # True: overwrite existing output_phase with new reconstruction

# ECC normalization range
ECC_VMIN = -5.0
ECC_VMAX =  2.0

# Tilt correction (same settings as compute_pos_shifts.py)
USE_SLOPE_CORRECTION = True   # True: use _tilt_correct (no backsub needed)
TILT_CROP_H = 270             # X width for tilt correction [px]
ECC_CROP_H  = 80              # Crop width for ECC and focus metrics [px]

# ===========================================================================


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def to_uint8(img: np.ndarray, vmin: float = ECC_VMIN, vmax: float = ECC_VMAX) -> np.ndarray:
    # Float ECC input: clip only, no 8-bit quantisation (cv2.findTransformECC
    # accepts float32; quantisation introduced a systematic X bias). Name kept.
    return np.clip(img, vmin, vmax).astype(np.float32)


def compute_ecc_warp(ref_img: np.ndarray, src_img: np.ndarray):
    """
    Compute and return the warp matrix using ECC (MOTION_TRANSLATION).

    Parameters
    ----------
    ref_img : z=ALIGN_Z frame from the grid side (reference)
    src_img : z=ALIGN_Z frame from the focus side (target to align)

    Returns
    -------
    warp_matrix : np.ndarray (2x3)  None on failure
    correlation : float             None on failure
    """
    ref_u8 = to_uint8(ref_img)
    src_u8 = to_uint8(src_img)
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        correlation, warp_matrix = cv2.findTransformECC(
            ref_u8, src_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return warp_matrix, float(correlation)
    except Exception as e:
        print(f"  ECC failed: {e}")
        return None, None


def apply_warp(img: np.ndarray, warp_matrix: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.warpAffine(
        img.astype(np.float32),
        warp_matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
    )


def _z_from_filename(fname: str):
    """Extract z index from filename.

    Examples:
        img_000000000_ph_003_phase.tif  -> 3
        img_000000000_ph_003.tif        -> 3  (unreconstructed fallback)
        img_000000000_Default_003_phase.tif -> 3
        img_000000000_Default_003.tif   -> 3
    """
    m = re.search(r'_(?:ph|Default)_(\d+)_phase\.tif$', fname, re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'_(?:ph|Default)_(\d+)\.tif$', fname, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def _detect_pos_labels(focus_dir: Path):
    """Return Pos1 and above. Pos0 is excluded as it is the background."""
    candidates = [
        item for item in focus_dir.iterdir()
        if item.is_dir() and re.match(r'^Pos\d+$', item.name) and item.name != "Pos0"
    ]
    return [item.name for item in sorted(candidates, key=lambda p: int(re.search(r'\d+', p.name).group()))]


def _load_phase_stack(phase_dir: Path) -> dict:
    """Load all z frames from output_phase/ and return {z_idx: ndarray}."""
    stack = {}
    for f in sorted(phase_dir.glob("*_phase.tif")):
        z = _z_from_filename(f.name)
        if z is None:
            continue
        stack[z] = tifffile.imread(str(f))
    return stack


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_one(raw_path: Path, out_dir: Path, params_dict: dict):
    """Reconstruct a single raw hologram and save to output_phase/. Skip if exists.

    Same processing as batch_reconstruction_dual:
    - If bg_path is specified, reconstruct Pos0 and compute the phase difference
    - Subtract the mean of the center region for offset normalization
    """
    try:
        from PIL import Image
        from skimage.restoration import unwrap_phase
        from qpi import QPIParameters, get_field

        out_path = out_dir / (raw_path.stem + "_phase.tif")
        if out_path.exists() and not params_dict.get("force", False):
            return raw_path.name, "skip"

        pos_number = params_dict.get("pos_number", 0)
        pos_split  = params_dict.get("pos_split", 9999)
        crop = params_dict["crop_before"] if pos_number < pos_split else params_dict["crop_after"]
        y0, y1, x0, x1 = crop

        img = np.array(Image.open(str(raw_path)))
        img = img[y0:y1, x0:x1]

        params = QPIParameters(
            wavelength=params_dict["wavelength"],
            NA=params_dict["NA"],
            img_shape=img.shape,
            pixelsize=params_dict["pixelsize"],
            offaxis_center=params_dict["offaxis_center"],
        )
        field = get_field(img, params)
        angle = unwrap_phase(np.angle(field))

        # Pos0 background subtraction (same processing as batch_reconstruction_dual)
        bg_path = params_dict.get("bg_path")
        if bg_path is not None and Path(bg_path).exists():
            bg_img = np.array(Image.open(str(bg_path)))
            bg_img = bg_img[y0:y1, x0:x1]
            field_bg = get_field(bg_img, params)
            angle_bg = unwrap_phase(np.angle(field_bg))
            angle = angle - angle_bg

            # Subtract center region mean for offset normalization
            h, w = angle.shape
            if pos_number < pos_split:
                center_region = angle[1:h-1, 1:w//2]
            else:
                center_region = angle[1:h-1, w//2:w-1]
            if center_region.size > 0:
                angle -= np.mean(center_region)

        tifffile.imwrite(str(out_path), angle.astype(np.float32))
        return raw_path.name, "ok"
    except Exception as e:
        return raw_path.name, f"err: {e}"


def reconstruct_dir(raw_dir: Path, recon_params: dict, pos_number: int, n_workers: int = 4,
                    bg_dir: Path = None):
    """Reconstruct all raw TIFFs in the directory and save to output_phase/.

    If bg_dir is specified, subtract the same-named file as Pos0 (same as batch_reconstruction_dual).
    """
    out_dir = raw_dir / "output_phase"
    out_dir.mkdir(exist_ok=True)

    # Check if output_phase or output_phase_raw already has reconstructed files
    for existing in [out_dir, raw_dir / "output_phase_raw"]:
        phase_files = sorted(existing.glob("*_phase.tif")) if existing.exists() else []
        if phase_files:
            print(f"  Already reconstructed: {existing} ({len(phase_files)} files)")
            return

    raw_files = [
        f for f in sorted(raw_dir.glob("*.tif"))
        if not f.name.startswith("._") and "output" not in f.name.lower()
    ]
    if not raw_files:
        raise FileNotFoundError(
            f"No raw TIFFs and no existing output_phase found: {raw_dir}"
        )

    tasks = []
    for f in raw_files:
        task_params = {**recon_params, "pos_number": pos_number}
        if bg_dir is not None:
            task_params["bg_path"] = str(bg_dir / f.name)
        tasks.append((f, out_dir, task_params))
    n_ok = n_skip = n_err = 0

    if n_workers == 1:
        results = [_reconstruct_one(*t) for t in tqdm(tasks, desc=f"  recon {raw_dir.name}")]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(_reconstruct_one, *t): t for t in tasks}
            results = []
            for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"  recon {raw_dir.name}"):
                results.append(fut.result())

    for name, status in results:
        if status == "ok":
            n_ok += 1
        elif status == "skip":
            n_skip += 1
        else:
            n_err += 1
            print(f"  [x] {name}: {status}")

    print(f"  done={n_ok}, skipped={n_skip}, errors={n_err}")


# ---------------------------------------------------------------------------
# Pos processing
# ---------------------------------------------------------------------------

def process_pos(pos_label: str,
                focus_dir: Path,
                grid_dir: Path,
                align_z: int,
                crop_output: bool,
                output_dir: Path,
                pos_number: int = 1):
    """ECC alignment + subtraction for a single Pos.

    Returns
    -------
    (z_list, diff_frames) or None (when skipped)
    """
    focus_pos_dir = focus_dir / pos_label
    grid_pos_name = f"{pos_label}_{GRID_SUFFIX}"
    grid_pos_dir  = grid_dir / grid_pos_name

    if not focus_pos_dir.exists():
        print(f"  [!] Skipping: {focus_pos_dir} does not exist")
        return None
    if not grid_pos_dir.exists():
        print(f"  [!] Skipping: {grid_pos_dir} does not exist")
        return None

    focus_phase_dir = focus_pos_dir / "output_phase"
    # Grid: prefer output_phase, fallback to output_phase_raw
    grid_phase_dir  = grid_pos_dir  / "output_phase"
    if not grid_phase_dir.exists() or not list(grid_phase_dir.glob("*_phase.tif")):
        alt = grid_pos_dir / "output_phase_raw"
        if alt.exists() and list(alt.glob("*_phase.tif")):
            grid_phase_dir = alt

    if not focus_phase_dir.exists():
        print(f"  [!] output_phase not found: {focus_phase_dir}")
        return None
    if not grid_phase_dir.exists():
        print(f"  [!] output_phase not found: {grid_phase_dir}")
        return None

    # Load stacks
    print(f"  Loading focus stack: {focus_phase_dir}")
    focus_stack = _load_phase_stack(focus_phase_dir)
    print(f"  Loading grid  stack: {grid_phase_dir}")
    grid_stack  = _load_phase_stack(grid_phase_dir)

    if not focus_stack:
        print(f"  [!] No focus phase files found")
        return None
    if not grid_stack:
        print(f"  [!] No grid phase files found")
        return None

    z_list = sorted(focus_stack.keys())
    print(f"  Number of z frames: {len(z_list)}  (z={z_list[0]}..{z_list[-1]})")

    # ECC alignment computation (ALIGN_Z frame only)
    actual_align_z = align_z if align_z in focus_stack else z_list[0]
    actual_grid_z  = align_z if align_z in grid_stack  else min(grid_stack.keys())

    ref_img = grid_stack[actual_grid_z]
    src_img = focus_stack[actual_align_z]

    # Load channel_rois.json (shared for ECC + output crop)
    # Located in grid's output_phase/channels/ (not on the focus_test side)
    ecc_rois_path = grid_phase_dir / "channels" / "channel_rois.json"
    if ecc_rois_path.exists():
        with open(ecc_rois_path) as fp:
            rois = json.load(fp)
    else:
        print(f"  channel_rois.json not found; falling back to full-frame ECC")
        rois = []

    print(f"  ECC alignment (focus z={actual_align_z}, grid z={actual_grid_z})...")

    warp_matrix = np.eye(2, 3, dtype=np.float32)
    if rois:
        # Per-channel tilt_correct -> ECC -> MAD outlier removal (same pattern as compute_pos_shifts)
        from ecc_utils import tilt_fit_crop, remove_outliers_mad

        fit_right = pos_number >= POS_SPLIT
        tx_list, ty_list, ch_names = [], [], []
        for ch_idx, roi_info in enumerate(rois):
            cy     = roi_info["cy"]
            cx     = roi_info["cx"]
            crop_w = roi_info.get("crop_w", 256)
            ref_crop = tilt_fit_crop(ref_img.astype(np.float64), cy, cx, crop_w, ECC_CROP_H, TILT_CROP_H, fit_right=fit_right)
            src_crop = tilt_fit_crop(src_img.astype(np.float64), cy, cx, crop_w, ECC_CROP_H, TILT_CROP_H, fit_right=fit_right)
            if ref_crop is None or src_crop is None:
                print(f"    ch{ch_idx}: skipped (tilt OOB)")
                continue
            warp_ch, corr_ch = compute_ecc_warp(ref_crop, src_crop)
            if warp_ch is not None:
                tx_list.append(warp_ch[0, 2])
                ty_list.append(warp_ch[1, 2])
                ch_names.append(f"ch{ch_idx}")
                print(f"    ch{ch_idx}: corr={corr_ch:.4f}, tx={warp_ch[0,2]:.2f}, ty={warp_ch[1,2]:.2f}")

        if len(tx_list) == 0:
            print("  [!] ECC failed on all channels. Proceeding without alignment")
            warp_matrix = np.eye(2, 3, dtype=np.float32)
        else:
            valid_mask = np.ones(len(tx_list), dtype=bool)
            if len(tx_list) >= 3:
                out_x = remove_outliers_mad(tx_list, 5.0)
                out_y = remove_outliers_mad(ty_list, 5.0)
                valid_mask = ~(out_x | out_y)
                removed = [ch_names[i] for i, v in enumerate(valid_mask) if not v]
                if removed:
                    print(f"  [!] outlier removed: {removed}")
            valid_tx = [tx_list[i] for i in range(len(tx_list)) if valid_mask[i]]
            valid_ty = [ty_list[i] for i in range(len(ty_list)) if valid_mask[i]]
            tx_mean = float(np.mean(valid_tx))
            ty_mean = float(np.mean(valid_ty))
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            warp_matrix[0, 2] = -tx_mean
            warp_matrix[1, 2] = -ty_mean
            print(f"  ECC done: tx={tx_mean:.2f} px, ty={ty_mean:.2f} px  ({len(valid_tx)}/{len(tx_list)} ch used)")
    else:
        warp_ff, corr_ff = compute_ecc_warp(ref_img, src_img)
        if warp_ff is not None:
            warp_matrix[0, 2] = -warp_ff[0, 2]
            warp_matrix[1, 2] = -warp_ff[1, 2]
            print(f"  ECC (full-frame): corr={corr_ff:.4f}, tx={warp_ff[0,2]:.2f}, ty={warp_ff[1,2]:.2f}")
        else:
            print("  [!] Full-frame ECC failed; using identity")

    rois_for_focus = rois  # Retained for focus evaluation

    # Output directory
    pos_out_dir = output_dir / pos_label
    pos_out_dir.mkdir(parents=True, exist_ok=True)

    # Process all z frames
    diff_frames = []
    for z in tqdm(z_list, desc=f"  {pos_label}"):
        focus_frame = focus_stack[z]
        # Use nearest z from grid if exact match not available
        grid_z = z if z in grid_stack else min(grid_stack.keys(), key=lambda gz: abs(gz - z))
        grid_frame = grid_stack[grid_z]

        warped_focus = apply_warp(focus_frame, warp_matrix)
        diff = warped_focus - grid_frame
        diff_frames.append(diff)

        # Save full frame
        tifffile.imwrite(str(pos_out_dir / f"z{z:03d}.tif"), diff.astype(np.float32))

        # Save tilt-corrected channel ROI crops for ALL detected channels
        if rois_for_focus:
            from channel_crop import extract_rect_roi
            for ch_idx, roi_info in enumerate(rois_for_focus):
                cy     = roi_info["cy"]
                cx     = roi_info["cx"]
                crop_w = roi_info.get("crop_w", 256)
                # Tilt correction (same as ecc_utils.tilt_fit_crop but no OOB rejection)
                big = extract_rect_roi(diff.astype(np.float64), cy, cx,
                                       crop_w, TILT_CROP_H).astype(np.float64)
                xv = np.arange(TILT_CROP_H, dtype=np.float64)
                prof = big.mean(axis=0)
                fit_n = max(1, TILT_CROP_H // 3)
                if fit_right:
                    a, b = np.polyfit(xv[-fit_n:], prof[-fit_n:], 1)
                else:
                    a, b = np.polyfit(xv[:fit_n], prof[:fit_n], 1)
                cropped = big - (a * xv + b)[np.newaxis, :]
                ch_dir = pos_out_dir / f"ch{ch_idx:02d}"
                ch_dir.mkdir(exist_ok=True)
                tifffile.imwrite(str(ch_dir / f"z{z:03d}.tif"), cropped.astype(np.float32))

    return z_list, diff_frames, rois_for_focus


# ---------------------------------------------------------------------------
# Montage
# ---------------------------------------------------------------------------

def make_montage(z_list, diff_frames, pos_label: str):
    try:
        from figure_logger import save_figure
    except ImportError:
        print("  figure_logger not found. Skipping montage")
        return

    n = len(diff_frames)
    fig, axes = plt.subplots(1, n, figsize=(max(3 * n, 6), 3))
    if n == 1:
        axes = [axes]

    all_vals = np.concatenate([f.ravel() for f in diff_frames])
    vmin = float(np.percentile(all_vals, 2))
    vmax = float(np.percentile(all_vals, 98))

    for ax, z, frame in zip(axes, z_list, diff_frames):
        ax.imshow(frame, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"z={z}", fontsize=8)
        ax.axis("off")

    fig.suptitle(f"Focus check subtracted: {pos_label}", fontsize=10)
    fig.tight_layout()

    save_figure(
        fig,
        params={"pos": pos_label, "align_z": ALIGN_Z, "n_z": n,
                "focus_dir": FOCUS_DIR, "grid_dir": GRID_DIR},
        description=f"Focus check subtracted montage: {pos_label}",
    )
    plt.close(fig)
    print(f"  Montage saved")


# ---------------------------------------------------------------------------
# Focus detection
# ---------------------------------------------------------------------------

def _focus_metrics(frames: list) -> tuple:
    """Compute and return (lap_vars, stds) from a list of frames."""
    lap_vars, stds = [], []
    for frame in frames:
        f32 = frame.astype(np.float32)
        lap = cv2.Laplacian(f32, cv2.CV_32F)
        lap_vars.append(float(np.var(lap)))
        stds.append(float(np.std(f32)))
    return lap_vars, stds


def _best_z_from_metrics(z_list, lap_vars, stds):
    lap_rank = np.argsort(np.argsort(lap_vars))
    std_rank  = np.argsort(np.argsort(stds))
    combined  = (lap_rank + std_rank).astype(float)
    return z_list[int(np.argmin(combined))], combined


def find_best_focus_z(z_list: list, diff_frames: list, pos_label: str,
                      rois: list = None, pos_number: int = 1) -> dict:
    """Detect the best focus z per channel from background-subtracted phase frames.

    If rois is specified, evaluation is per channel ROI; otherwise full frame is used.

    Metric: average rank of Laplacian variance (edge sharpness) + standard deviation (phase contrast).

    Returns
    -------
    results : dict   {"ch0": best_z, "ch1": best_z, ...}  or {"full": best_z}
    """
    try:
        from figure_logger import save_figure
    except ImportError:
        save_figure = None

    from ecc_utils import tilt_fit_crop as _tilt_fit

    # Build per-channel frame lists (same pattern as compute_pos_shifts)
    if rois:
        fit_right = pos_number >= POS_SPLIT
        channels = {}
        for ch_idx, roi_info in enumerate(rois):
            cy     = roi_info["cy"]
            cx     = roi_info["cx"]
            crop_w = roi_info.get("crop_w", 256)
            crops = [
                _tilt_fit(f.astype(np.float64), cy, cx, crop_w, ECC_CROP_H, TILT_CROP_H, fit_right=fit_right)
                for f in diff_frames
            ]
            if any(c is None for c in crops):
                continue
            channels[f"ch{ch_idx}"] = crops
    else:
        channels = {"full": diff_frames}

    results = {}
    all_metrics = {}  # For plotting

    print(f"\n  === Focus detection results: {pos_label} ===")
    for ch_name, frames in channels.items():
        lap_vars, stds = _focus_metrics(frames)
        best_z, combined = _best_z_from_metrics(z_list, lap_vars, stds)
        results[ch_name] = best_z
        all_metrics[ch_name] = {"lap_vars": lap_vars, "stds": stds, "combined": combined}

        print(f"\n  [{ch_name}]")
        header = f"{'z':>4}  {'Lap var':>12}  {'std':>10}  {'score':>6}"
        print(f"  {header}")
        best_idx = z_list.index(best_z)
        for i, z in enumerate(z_list):
            marker = " <-- best" if i == best_idx else ""
            print(f"  {z:>4}  {lap_vars[i]:>12.4f}  {stds[i]:>10.4f}  {combined[i]:>6.0f}{marker}")
        print(f"  => best z = {best_z}")

    # Plot focus curves (n_channels x 2 panels)
    if save_figure is not None:
        n_ch = len(channels)
        fig, axes = plt.subplots(n_ch, 2, figsize=(9, 2.2 * n_ch), squeeze=False)

        # Pre-compute common ylim per column
        all_lap = [v for m in all_metrics.values() for v in m["lap_vars"]]
        all_std = [v for m in all_metrics.values() for v in m["stds"]]
        margin = 0.05
        lap_ylim = (min(all_lap) * (1 - margin), max(all_lap) * (1 + margin))
        std_ylim = (min(all_std) * (1 - margin), max(all_std) * (1 + margin))
        col_ylims = [lap_ylim, std_ylim]

        for row, (ch_name, metrics) in enumerate(all_metrics.items()):
            best_z = results[ch_name]
            for col, (vals, ylabel) in enumerate([
                (metrics["lap_vars"], "Laplacian variance"),
                (metrics["stds"],     "Std dev (phase)"),
            ]):
                ax = axes[row][col]
                ax.plot(z_list, vals, marker="o", linewidth=1.5, markersize=5, color="steelblue")
                ax.axvline(best_z, color="tomato", linestyle="--", linewidth=1.5,
                           label=f"best z={best_z}")
                ax.set_xlabel("z index", fontsize=11)
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(ch_name, fontsize=10)
                ax.set_ylim(col_ylims[col])
                ax.tick_params(direction="in", labelsize=10)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.legend(fontsize=9, frameon=False)
        fig.suptitle(f"Focus curve: {pos_label}", fontsize=12)
        fig.tight_layout()
        data_dict = {"z_list": np.array(z_list)}
        for ch_name, metrics in all_metrics.items():
            data_dict[f"{ch_name}_lap_vars"] = np.array(metrics["lap_vars"])
            data_dict[f"{ch_name}_stds"]     = np.array(metrics["stds"])
        save_figure(
            fig,
            params={"pos": pos_label, "best_z_per_ch": results,
                    "focus_dir": FOCUS_DIR, "grid_dir": GRID_DIR},
            description=f"Focus curve per channel: {pos_label}, best_z={results}",
            data=data_dict,
        )
        plt.close(fig)

    return results


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    focus_dir  = Path(FOCUS_DIR)
    grid_dir   = Path(GRID_DIR)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detect Pos labels
    pos_labels = POS_LABELS if POS_LABELS is not None else _detect_pos_labels(focus_dir)
    if not pos_labels:
        print("[x] No Pos directories found")
        return
    print(f"Target Pos: {pos_labels}")

    # Reconstruction phase (when DO_RECONSTRUCTION=True)
    if DO_RECONSTRUCTION:
        print("\n=== Reconstruction phase ===")
        recon_params = {
            "wavelength":     WAVELENGTH,
            "NA":             NA,
            "pixelsize":      PIXELSIZE,
            "offaxis_center": OFFAXIS_CENTER,
            "pos_split":      POS_SPLIT,
            "crop_before":    CROP_BEFORE,
            "crop_after":     CROP_AFTER,
            "force":          FORCE_RECONSTRUCT,
        }
        focus_bg_dir = focus_dir / "Pos0"
        grid_bg_dir  = grid_dir / f"Pos0_{GRID_SUFFIX}"
        if focus_bg_dir.exists():
            print(f"  Pos0 BG (focus): {focus_bg_dir}")
        else:
            print(f"  [!] Pos0 BG not found (no subtraction): {focus_bg_dir}")
            focus_bg_dir = None
        if grid_bg_dir.exists():
            print(f"  Pos0 BG (grid ): {grid_bg_dir}")
        else:
            print(f"  [!] Pos0 BG not found (no subtraction): {grid_bg_dir}")
            grid_bg_dir = None

        for pos_label in pos_labels:
            pos_number = int(re.search(r'\d+', pos_label).group())
            crop_label = "BEFORE" if pos_number < POS_SPLIT else "AFTER"
            print(f"\n[focus] {pos_label}  (pos={pos_number}, crop={crop_label})")
            reconstruct_dir(focus_dir / pos_label, recon_params, pos_number, N_WORKERS,
                            bg_dir=focus_bg_dir)
            grid_pos_name = f"{pos_label}_{GRID_SUFFIX}"
            print(f"\n[grid ] {grid_pos_name}")
            reconstruct_dir(grid_dir / grid_pos_name, recon_params, pos_number, N_WORKERS,
                            bg_dir=grid_bg_dir)

    # Subtraction phase
    print("\n=== Subtraction phase ===")
    for pos_label in pos_labels:
        print(f"\n[{pos_label}]")
        pos_number = int(re.search(r'\d+', pos_label).group())
        result = process_pos(pos_label, focus_dir, grid_dir, ALIGN_Z, CROP_OUTPUT, output_dir,
                             pos_number=pos_number)
        if result is None:
            continue
        z_list, diff_frames, rois = result

        # Focus detection (all Pos, per channel)
        best_z_per_ch = find_best_focus_z(z_list, diff_frames, pos_label, rois=rois,
                                          pos_number=pos_number)

        # Montage for first Pos only
        if pos_label == pos_labels[0]:
            print("  Generating montage...")
            make_montage(z_list, diff_frames, pos_label)

    print(f"\nDone: {output_dir}")


if __name__ == "__main__":
    main()


# %%
