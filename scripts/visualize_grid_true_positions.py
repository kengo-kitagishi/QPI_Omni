# %%
"""
visualize_grid_true_positions.py
---------------------------------
Measure and visualize the "true pixel positions" of each grid scan point using ECC.
Compare two methods: Direct ECC and BFS chain.

[ECC procedure] Fully aligned with compute_pos_shifts.py:
  - Preprocessing: _tilt_correct (TILT_CROP_H=270 -> ECC_CROP_H=80)
  - Normalization: to_uint8 with fixed VMIN/VMAX (-5/2)
  - Channel averaging: MAD outlier removal (OUTLIER_MAD_THRESH=5.0) -> average of remaining channels
  - ECC: MOTION_TRANSLATION, 100000 iter, 1e-8

[Method 1: Direct ECC]
  Direct ECC of each point using center (0,0) as reference. Max shift +/-1.45 px.

[Method 2: BFS chain]
  Same BFS method as calibrate_grid_positions.py. Same preprocessing.

[Output]
  Figure 1: Direct ECC -- Nominal vs Measured + Residual map
  Figure 2: BFS        -- Nominal vs Measured + Residual map
  Units: um
"""
import sys
import re
import json
import numpy as np
import tifffile
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import deque
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ============================================================
# Configuration parameters
# ============================================================
GRID_DIR   = Path(r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1")
POS_PREFIX = "Pos1"
Z_IDX      = 9

# ECC parameters (same values as compute_pos_shifts.py)
VMIN, VMAX       = -5.0, 2.0
# ECC convergence params now in ecc_utils

# Tilt correction parameters (same values as compute_pos_shifts.py)
TILT_CROP_H = 270   # Big crop width in X direction [px]
ECC_CROP_H  = 80    # Center crop width used for ECC [px]

# Channel outlier removal (same values as compute_pos_shifts.py)
OUTLIER_MAD_THRESH = 5.0

# Optical parameters
SENSOR_PIXEL_SIZE = 3.45e-6   # [m]
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511

# Grid step
X_STEP       = 0.05   # [um]
Y_STEP       = 0.05   # [um]
SHIFT_SIGN_X = -1
SHIFT_SIGN_Y = -1

# channel_rois.json path (None -> auto-detect from Pos1_x+0_y+0/output_phase/channels/)
CHANNEL_ROIS_JSON = None
# ============================================================

pixel_scale_um = (SENSOR_PIXEL_SIZE / MAGNIFICATION) * (ORIGINAL_DIM / RECONSTRUCTED_DIM) * 1e6
STEP_PX = X_STEP / pixel_scale_um  # ~0.289 px/step
PX2UM   = pixel_scale_um


# ---- Utility functions ----

from ecc_utils import (
    tilt_fit_crop, extract_rect_roi, to_uint8, ecc_align,
    remove_outliers_mad,
)


def _channel_average(tx_list, ty_list, corr_list):
    """
    Compute the mean after MAD outlier removal from channel lists.
    Same logic as _frame_result_from_per_channel in compute_pos_shifts.py.
    Returns: (actual_dx, actual_dy, mean_corr)
    """
    if not tx_list:
        return None
    tx_arr   = np.array(tx_list,   dtype=np.float64)
    ty_arr   = np.array(ty_list,   dtype=np.float64)
    corr_arr = np.array(corr_list, dtype=np.float64)
    if len(tx_list) >= 3:
        out_x = remove_outliers_mad(tx_list, OUTLIER_MAD_THRESH)
        out_y = remove_outliers_mad(ty_list, OUTLIER_MAD_THRESH)
        mask  = ~(out_x | out_y)
        if mask.any():
            tx_arr   = tx_arr[mask]
            ty_arr   = ty_arr[mask]
            corr_arr = corr_arr[mask]
    return (float(-np.mean(tx_arr)),   # actual_dx = -tx
            float(-np.mean(ty_arr)),   # actual_dy = -ty
            float(np.mean(corr_arr)))


def scan_grid_positions(grid_dir, base_label):
    """Return a map of (xi, yi) -> folder_path."""
    pattern = re.compile(rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
    pos_map = {}
    for d in Path(grid_dir).iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            pos_map[(int(m.group(1)), int(m.group(2)))] = d
    return pos_map


def load_phase_image(pos_dir, z_idx):
    """Load output_phase/img_000000000_ph_{z_idx:03d}_phase.tif as float64."""
    path = Path(pos_dir) / "output_phase" / f"img_000000000_ph_{z_idx:03d}_phase.tif"
    if not path.exists():
        raise FileNotFoundError(f"Phase image not found: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def make_crops_u8(img_f64, rois, fit_right: bool = False):
    """Return tilt-corrected uint8 crops for all channels (None if OOB)."""
    crops = []
    for roi in rois:
        tc = tilt_fit_crop(img_f64, roi["cy"], roi["cx"],
                           roi["crop_w"], ECC_CROP_H, TILT_CROP_H,
                           fit_right=fit_right)
        crops.append(to_uint8(tc, VMIN, VMAX) if tc is not None else None)
    return crops


# ---- Direct ECC method ----

def run_direct_ecc(crops_cache, pos_map, rois, pixel_scale_um):
    """Direct ECC of each grid position using center (0,0) as reference."""
    n_channels = len(rois)
    results = {(0, 0): {
        "actual_dx": 0.0, "actual_dy": 0.0,
        "nominal_dx": 0.0, "nominal_dy": 0.0,
        "corr": 1.0, "failed": False,
    }}

    if (0, 0) not in crops_cache:
        raise RuntimeError("center (0,0) crops not found in cache")
    ref_crops = crops_cache[(0, 0)]

    sorted_keys = sorted(pos_map.keys(), key=lambda k: abs(k[0]) + abs(k[1]))
    pbar = tqdm(total=len(pos_map), desc="Direct ECC")
    pbar.update(1)  # skip (0,0)

    for (xi, yi) in sorted_keys:
        if (xi, yi) == (0, 0):
            continue
        nominal_dx = float(SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um)
        nominal_dy = float(SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um)

        if (xi, yi) not in crops_cache:
            results[(xi, yi)] = {
                "actual_dx": nominal_dx, "actual_dy": nominal_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
            pbar.update(1)
            continue

        cur_crops = crops_cache[(xi, yi)]
        tx_list, ty_list, corr_list = [], [], []
        for ch in range(n_channels):
            res = ecc_align(ref_crops[ch], cur_crops[ch])
            if res is not None:
                tx_list.append(res[0]); ty_list.append(res[1]); corr_list.append(res[2])

        avg = _channel_average(tx_list, ty_list, corr_list)
        if avg is None:
            results[(xi, yi)] = {
                "actual_dx": nominal_dx, "actual_dy": nominal_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
        else:
            results[(xi, yi)] = {
                "actual_dx": avg[0], "actual_dy": avg[1],
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": avg[2], "failed": False,
            }
        pbar.update(1)

    pbar.close()
    return results


# ---- BFS chain method ----

def run_bfs(crops_cache, pos_map, rois, pixel_scale_um):
    """
    BFS chain method: same logic as calibrate_grid_positions.py.
    Preprocessing and channel averaging aligned with compute_pos_shifts.py.
    """
    n_channels = len(rois)
    calibrated = {(0, 0): (0.0, 0.0)}
    results    = {(0, 0): {
        "actual_dx": 0.0, "actual_dy": 0.0,
        "nominal_dx": 0.0, "nominal_dy": 0.0,
        "corr": 1.0, "failed": False,
    }}
    visited = {(0, 0)}
    queue   = deque()

    def enqueue_neighbors(xi, yi):
        for dxi, dyi in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (xi + dxi, yi + dyi)
            if nb in pos_map and nb not in visited:
                queue.append(nb)
                visited.add(nb)

    enqueue_neighbors(0, 0)
    pbar = tqdm(total=len(pos_map), desc="BFS ECC")
    pbar.update(1)

    while queue:
        xi, yi = queue.popleft()
        nominal_dx = float(SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um)
        nominal_dy = float(SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um)
        cal_nb = [(xi + d, yi + e) for d, e in [(1, 0), (-1, 0), (0, 1), (0, -1)]
                  if (xi + d, yi + e) in calibrated]

        if (xi, yi) not in crops_cache or not cal_nb:
            ref_nb = cal_nb[0] if cal_nb else (0, 0)
            ref_dx, ref_dy = calibrated.get(ref_nb, (0.0, 0.0))
            step_dx = SHIFT_SIGN_Y * (yi - ref_nb[1]) * Y_STEP / pixel_scale_um
            step_dy = SHIFT_SIGN_X * (xi - ref_nb[0]) * X_STEP / pixel_scale_um
            act_dx  = ref_dx + step_dx
            act_dy  = ref_dy + step_dy
            calibrated[(xi, yi)] = (act_dx, act_dy)
            results[(xi, yi)] = {
                "actual_dx": act_dx, "actual_dy": act_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
            pbar.update(1)
            enqueue_neighbors(xi, yi)
            continue

        cur_crops = crops_cache[(xi, yi)]
        dx_est, dy_est, corr_est = [], [], []

        for ref_nb in cal_nb:
            if ref_nb not in crops_cache:
                continue
            ref_crops_nb = crops_cache[ref_nb]
            ref_dx, ref_dy = calibrated[ref_nb]
            tx_list, ty_list, corr_list = [], [], []
            for ch in range(n_channels):
                res = ecc_align(ref_crops_nb[ch], cur_crops[ch])
                if res is not None:
                    tx_list.append(res[0]); ty_list.append(res[1]); corr_list.append(res[2])
            avg = _channel_average(tx_list, ty_list, corr_list)
            if avg is not None:
                dx_est.append(ref_dx + avg[0])
                dy_est.append(ref_dy + avg[1])
                corr_est.append(avg[2])

        if not dx_est:
            ref_nb = cal_nb[0]
            ref_dx, ref_dy = calibrated[ref_nb]
            step_dx = SHIFT_SIGN_Y * (yi - ref_nb[1]) * Y_STEP / pixel_scale_um
            step_dy = SHIFT_SIGN_X * (xi - ref_nb[0]) * X_STEP / pixel_scale_um
            act_dx  = ref_dx + step_dx
            act_dy  = ref_dy + step_dy
            calibrated[(xi, yi)] = (act_dx, act_dy)
            results[(xi, yi)] = {
                "actual_dx": act_dx, "actual_dy": act_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": None, "failed": True,
            }
        else:
            act_dx = float(np.mean(dx_est))
            act_dy = float(np.mean(dy_est))
            calibrated[(xi, yi)] = (act_dx, act_dy)
            results[(xi, yi)] = {
                "actual_dx": act_dx, "actual_dy": act_dy,
                "nominal_dx": nominal_dx, "nominal_dy": nominal_dy,
                "corr": float(np.mean(corr_est)), "failed": False,
            }
        pbar.update(1)
        enqueue_neighbors(xi, yi)

    pbar.close()
    return results


# ---- Figure generation and saving ----

def make_figure(results, method_label, pixel_scale_um):
    """Generate a 2-panel figure and return (fig, arrays_dict). Units: um."""
    sorted_keys = sorted(results.keys())
    xi_list  = np.array([k[0] for k in sorted_keys])
    yi_list  = np.array([k[1] for k in sorted_keys])
    nom_dx   = np.array([results[k]["nominal_dx"] for k in sorted_keys]) * PX2UM
    nom_dy   = np.array([results[k]["nominal_dy"] for k in sorted_keys]) * PX2UM
    act_dx   = np.array([results[k]["actual_dx"]  for k in sorted_keys]) * PX2UM
    act_dy   = np.array([results[k]["actual_dy"]  for k in sorted_keys]) * PX2UM
    corr_arr = np.array([results[k]["corr"] if results[k]["corr"] is not None else 0.0
                         for k in sorted_keys])
    res_um   = np.sqrt((act_dx - nom_dx)**2 + (act_dy - nom_dy)**2)

    n_failed = sum(1 for r in results.values() if r["failed"])
    success  = [r for r in results.values() if not r["failed"]
                and (r["nominal_dx"] != 0 or r["nominal_dy"] != 0)]
    if success:
        res_px_vals = np.array([np.sqrt((r["actual_dx"]-r["nominal_dx"])**2 +
                                        (r["actual_dy"]-r["nominal_dy"])**2)
                                for r in success])
        print(f"  [{method_label}] Success: {len(results)-n_failed}/{len(results)}")
        print(f"  Residual [px]: mean={res_px_vals.mean():.4f}  std={res_px_vals.std():.4f}  "
              f"max={res_px_vals.max():.4f}")
        print(f"  Residual [um]: mean={res_px_vals.mean()*pixel_scale_um:.4f}  "
              f"max={res_px_vals.max()*pixel_scale_um:.4f}")
        corrs = [r["corr"] for r in success if r["corr"] is not None]
        if corrs:
            print(f"  ECC correlation: mean={np.mean(corrs):.4f}  min={np.min(corrs):.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        f"Grid true position check: {POS_PREFIX}, z={Z_IDX}  [{method_label}]",
        fontsize=11,
    )

    # Panel 1: Nominal lattice vs measured positions
    ax = axes[0]
    ax.scatter(nom_dx, nom_dy, s=80, facecolors="none", edgecolors="gray",
               linewidths=1.2, zorder=2, label="Nominal")
    sc = ax.scatter(act_dx, act_dy, s=30, c=corr_arr, cmap="RdYlGn",
                    vmin=0.97, vmax=1.0, zorder=3, label="Measured")
    plt.colorbar(sc, ax=ax, label="ECC correlation")
    ax.set_xlabel("dx (um)  [Stage Y / image X direction]")
    ax.set_ylabel("dy (um)  [Stage X / image Y direction]")
    ax.set_title("Nominal lattice vs Measured positions")
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    ax.grid(True, linewidth=0.3, alpha=0.5)

    # Panel 2: Residual error map
    ax2 = axes[1]
    sc2 = ax2.scatter(xi_list, yi_list, c=res_um, s=100,
                      cmap="hot_r", vmin=0, vmax=max(float(res_um.max()), 0.05))
    plt.colorbar(sc2, ax=ax2, label="Position error (um)")
    ax2.set_xlabel("xi  (Stage X direction)")
    ax2.set_ylabel("yi  (Stage Y direction)")
    ax2.set_title(
        f"Residual |actual - nominal| (um)\n"
        f"mean={res_um.mean():.4f} um  max={res_um.max():.4f} um"
    )
    ax2.set_aspect("equal")
    ax2.grid(True, linewidth=0.3, alpha=0.5)
    for xi, yi, r in zip(xi_list, yi_list, res_um):
        if r > 0.05:
            ax2.text(xi, yi, f"{r:.2f}", fontsize=5,
                     ha="center", va="center", color="white", fontweight="bold")

    plt.tight_layout()

    arrays = {
        "nominal_dx_um": nom_dx, "nominal_dy_um": nom_dy,
        "actual_dx_um":  act_dx, "actual_dy_um":  act_dy,
        "xi": xi_list, "yi": yi_list,
        "residual_um": res_um, "corr": corr_arr,
    }
    return fig, arrays, res_um


# ---- Main ----

def main():
    print(f"Pixel scale: {pixel_scale_um:.4f} um/px")
    print(f"Step: {X_STEP} um = {STEP_PX:.4f} px/step  "
          f"(+-5 steps = +-{5*STEP_PX:.3f} px max)")

    # Load ROIs
    if CHANNEL_ROIS_JSON is not None:
        rois_path = Path(CHANNEL_ROIS_JSON)
    else:
        rois_path = (GRID_DIR / f"{POS_PREFIX}_x+0_y+0"
                     / "output_phase" / "channels" / "channel_rois.json")
    if not rois_path.exists():
        print(f"ERROR: channel_rois.json not found: {rois_path}")
        sys.exit(1)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)
    print(f"Number of channels: {n_channels}")

    # Scan grid positions
    pos_map = scan_grid_positions(GRID_DIR, POS_PREFIX)
    if not pos_map:
        print(f"ERROR: Grid positions not found: {GRID_DIR}/{POS_PREFIX}_x*_y*")
        sys.exit(1)
    xi_all = [k[0] for k in pos_map]
    yi_all = [k[1] for k in pos_map]
    print(f"Grid positions: {len(pos_map)}  "
          f"xi: [{min(xi_all)}, {max(xi_all)}]  yi: [{min(yi_all)}, {max(yi_all)}]")

    if (0, 0) not in pos_map:
        print("ERROR: center (0,0) not found")
        sys.exit(1)

    # Load all crops at once (shared by Direct ECC / BFS)
    print("\nLoading all grid images...")
    crops_cache = {}
    n_load_failed = 0
    for (xi, yi), pos_dir in tqdm(sorted(pos_map.items()), desc="Loading crops"):
        try:
            img = load_phase_image(pos_dir, Z_IDX)
            crops_cache[(xi, yi)] = make_crops_u8(img, rois)
        except FileNotFoundError:
            n_load_failed += 1
    print(f"Loading complete: {len(crops_cache)}/{len(pos_map)}  "
          f"(crop shape: {crops_cache[(0,0)][0].shape})")
    if n_load_failed:
        print(f"  Loading failed: {n_load_failed} points (no output_phase)")

    # ---- Method 1: Direct ECC ----
    print("\n=== Method 1: Direct ECC (center reference) ===")
    results_direct = run_direct_ecc(crops_cache, pos_map, rois, pixel_scale_um)

    print("\n=== Method 2: BFS chain ===")
    results_bfs = run_bfs(crops_cache, pos_map, rois, pixel_scale_um)

    # ---- Figure generation and saving ----
    print("\n=== Saving figures ===")

    print("\n[Figure 1] Direct ECC")
    fig1, arr1, res1 = make_figure(results_direct, "Direct ECC", pixel_scale_um)
    save_figure(
        fig1,
        params={
            "pos_prefix": POS_PREFIX, "z_idx": Z_IDX,
            "method": "direct_ecc",
            "step_um": X_STEP, "step_px": float(STEP_PX),
            "tilt_crop_h": TILT_CROP_H, "ecc_crop_h": ECC_CROP_H,
            "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        },
        description=(
            f"Grid true position check: {POS_PREFIX} z={Z_IDX}, "
            f"direct ECC vs nominal. "
            f"mean_residual={res1.mean():.4f} um"
        ),
        data=arr1,
    )
    plt.close(fig1)

    print("\n[Figure 2] BFS chain")
    fig2, arr2, res2 = make_figure(results_bfs, "BFS chain", pixel_scale_um)
    save_figure(
        fig2,
        params={
            "pos_prefix": POS_PREFIX, "z_idx": Z_IDX,
            "method": "bfs_chain",
            "step_um": X_STEP, "step_px": float(STEP_PX),
            "tilt_crop_h": TILT_CROP_H, "ecc_crop_h": ECC_CROP_H,
            "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        },
        description=(
            f"Grid true position check: {POS_PREFIX} z={Z_IDX}, "
            f"BFS chain vs nominal. "
            f"mean_residual={res2.mean():.4f} um"
        ),
        data=arr2,
    )
    plt.close(fig2)

    print(f"\nComparison: Direct ECC mean={res1.mean():.4f} um  |  BFS mean={res2.mean():.4f} um")
    if res2.mean() > res1.mean():
        print("  -> BFS is larger: possible BFS error accumulation")
    else:
        print("  -> BFS and Direct ECC are comparable: error accumulation effect is small")


if __name__ == "__main__":
    main()

# %%
