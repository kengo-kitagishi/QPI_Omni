"""
analyze_ch_xtilt_corrected_ecc.py
----------------------------------
Expand channel ROI to 270px in X direction, linear fit on the left 1/3 (90px) BG, subtract, then ECC.
Compare inter-channel std of ECC X shift before and after correction.

Processing:
  1. For each channel, obtain a wide ROI with crop_h expanded to 270px -> shape (crop_w=40, 270)
  2. X profile = crop.mean(axis=0) -> 270 points (averaged along Y)
  3. Linear fit on the left 1/3 (90pt) with np.polyfit -> slope [rad/px]
  4. Subtract slope * x_local from the full 270px (intercept is preserved)
  5. Extract central crop_h=80px, backsub -> to_uint8 -> ECC
  6. Apply the same correction to the grid ref side
  7. Compare inter-channel std of ECC X shift before and after correction

Visualization:
  Fig 1 - summary: inter-ch std / histogram / mean ECC X / per-channel time series / representative ch profile
  Fig 2 - images: representative 4ch wide crops (raw 40x270 / corrected 40x270) + X profile
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tifffile
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from compute_drift_online import (
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
)
from figure_logger import save_figure

# ============================================================
# Settings
# ============================================================
PHASE_DIR = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON = Path(
    r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)

PIXEL_SCALE_UM   = 0.3462
ECC_VMIN         = -5.0
ECC_VMAX         =  2.0

# Grid reference crops (uncorrected version = conventional reference)
GRID_REF_CROPS_TIF = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\grid_ref_crops.tif")

# Grid(0,0) full phase image (for computing corrected ref crops)
GRID_REF_PHASE = Path(
    r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\img_000000000_ph_009_phase.tif"
)

# X-direction tilt correction parameters
CH_CROP_H_TILT = 270   # expanded size in X direction [px]
FIT_FRAC       = 1/3   # left 1/3 (90pt) = BG region
N_MAX_TP       = 20    # max frames for verification (None for all frames)

# Number of representative channels for image visualization
VIS_N_CH = 4
# ============================================================


def compute_x_slope_and_correct(
    phase_img: np.ndarray,
    cy: int, cx: int,
    crop_w: int, orig_crop_h: int,
) -> tuple[np.ndarray, float]:
    """
    Estimate slope from the left 1/3 of the 270px X-expanded crop and return the corrected orig_crop_h crop.

    Returns:
        corrected_crop: shape (crop_w, orig_crop_h) - central orig_crop_h px extracted
        slope: float [rad/px]
    """
    big_crop = extract_rect_roi(
        phase_img, cy, cx, crop_w, CH_CROP_H_TILT
    ).astype(np.float32)   # shape: (crop_w, CH_CROP_H_TILT) = (40, 270)

    x_profile = big_crop.mean(axis=0).astype(np.float64)   # (270,) = X profile
    n_fit = max(3, int(len(x_profile) * FIT_FRAC))         # 90
    slope, _ = np.polyfit(np.arange(n_fit, dtype=float), x_profile[:n_fit], 1)

    # Correction: subtract slope * x from all 270px (intercept is preserved)
    x_local = np.arange(CH_CROP_H_TILT, dtype=np.float32)
    corrected_big = big_crop - (slope * x_local).astype(np.float32)[np.newaxis, :]

    # Extract central orig_crop_h px
    start = (CH_CROP_H_TILT - orig_crop_h) // 2
    corrected_crop = corrected_big[:, start:start + orig_crop_h]

    return corrected_crop, float(slope)


def main():
    phase_paths = sorted(PHASE_DIR.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: no phase images found: {PHASE_DIR}")
        sys.exit(1)
    if N_MAX_TP is not None:
        phase_paths = phase_paths[:N_MAX_TP + 1]
    rois = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    n_ch = len(rois)
    n_tp = len(phase_paths) - 1
    print(f"Phase: {len(phase_paths)} imgs, Channels: {n_ch}, TPs: {n_tp}")

    # ---- Uncorrected ref (grid_ref_crops.tif as-is) ----
    grid_ref_f = tifffile.imread(str(GRID_REF_CROPS_TIF)).astype(np.float64)
    ref_u8_raw = [to_uint8(grid_ref_f[ch], ECC_VMIN, ECC_VMAX) for ch in range(n_ch)]
    print(f"grid_ref_crops loaded: shape={grid_ref_f.shape}")

    # ---- Corrected ref (grid ph_009: 270px -> correction -> 80px extraction -> backsub -> u8) ----
    grid_phase = tifffile.imread(str(GRID_REF_PHASE)).astype(np.float32)
    cfg_dummy = {}
    ref_u8_corr = []
    grid_ref_slopes = []
    for roi in rois:
        crop_c, sl = compute_x_slope_and_correct(
            grid_phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
        )
        offset = compute_backsub_offset(crop_c, cfg_dummy)
        ref_u8_corr.append(to_uint8(crop_c + offset, ECC_VMIN, ECC_VMAX))
        grid_ref_slopes.append(sl)
    print(f"grid ref slopes: min={min(grid_ref_slopes):.5f} max={max(grid_ref_slopes):.5f}")

    tx_raw  = np.full((n_ch, n_tp), np.nan)
    tx_corr = np.full((n_ch, n_tp), np.nan)
    slopes  = np.full((n_ch, n_tp), np.nan)

    for t_idx, path in enumerate(phase_paths[1:]):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except Exception:
            continue

        for c_idx, roi in enumerate(rois):
            # --- Uncorrected ECC ---
            crop = extract_rect_roi(
                phase_tp, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
            ).astype(np.float32)
            off = compute_backsub_offset(crop, cfg_dummy)
            res_raw = ecc_align(ref_u8_raw[c_idx],
                                to_uint8(crop + off, ECC_VMIN, ECC_VMAX))
            if res_raw is not None:
                tx_raw[c_idx, t_idx] = res_raw[0]   # image-X shift [px]

            # --- Corrected ECC (270px -> fit -> subtract -> 80px extraction) ---
            crop_c, sl = compute_x_slope_and_correct(
                phase_tp, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
            )
            slopes[c_idx, t_idx] = sl
            off_c = compute_backsub_offset(crop_c, cfg_dummy)
            res_corr = ecc_align(ref_u8_corr[c_idx],
                                 to_uint8(crop_c + off_c, ECC_VMIN, ECC_VMAX))
            if res_corr is not None:
                tx_corr[c_idx, t_idx] = res_corr[0]   # image-X shift [px]

        if (t_idx + 1) % 5 == 0:
            print(f"  TP {t_idx+1}/{n_tp}", flush=True)

    tx_raw_um  = tx_raw  * PIXEL_SCALE_UM
    tx_corr_um = tx_corr * PIXEL_SCALE_UM

    std_raw  = np.nanstd(tx_raw_um,  axis=0)
    std_corr = np.nanstd(tx_corr_um, axis=0)
    mean_raw  = np.nanmean(tx_raw_um,  axis=0)
    mean_corr = np.nanmean(tx_corr_um, axis=0)

    print(f"\nInter-channel std  raw={np.nanmean(std_raw):.4f} um  corr={np.nanmean(std_corr):.4f} um")
    print(f"Mean |ECC X|        raw={np.nanmean(np.abs(mean_raw)):.4f} um  corr={np.nanmean(np.abs(mean_corr)):.4f} um")

    # =================================================================
    # Fig 1: Summary statistics (same layout as ytilt version, X direction)
    # =================================================================
    tp_axis = np.arange(n_tp)
    COLORS  = plt.cm.tab20(np.linspace(0, 1, n_ch))
    fig1 = plt.figure(figsize=(16, 10))
    gs1  = gridspec.GridSpec(3, 3, hspace=0.48, wspace=0.38,
                             left=0.07, right=0.97, top=0.92, bottom=0.07)

    # (A) inter-channel std time series
    ax0 = fig1.add_subplot(gs1[0, 0])
    ax0.plot(tp_axis, std_raw,  color="#E53935", lw=0.9, label="Raw")
    ax0.plot(tp_axis, std_corr, color="#1E88E5", lw=0.9, label="X-tilt corr")
    ax0.set_xlabel("TP"); ax0.set_ylabel("Inter-ch std (um)")
    ax0.set_title("Inter-channel std", fontsize=10)
    ax0.legend(fontsize=8)
    ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)

    # (B) std histogram
    ax1 = fig1.add_subplot(gs1[0, 1])
    bins = np.linspace(0, max(np.nanmax(std_raw), np.nanmax(std_corr)) * 1.1, 25)
    ax1.hist(std_raw[~np.isnan(std_raw)],   bins=bins, alpha=0.6,
             color="#E53935", label=f"Raw μ={np.nanmean(std_raw):.4f}")
    ax1.hist(std_corr[~np.isnan(std_corr)], bins=bins, alpha=0.6,
             color="#1E88E5", label=f"Corr μ={np.nanmean(std_corr):.4f}")
    ax1.set_xlabel("Inter-ch std (um)"); ax1.set_ylabel("Count")
    ax1.set_title("Histogram", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # (C) mean ECC X time series
    ax2 = fig1.add_subplot(gs1[0, 2])
    ax2.plot(tp_axis, mean_raw,  color="#E53935", lw=0.9, label="Raw")
    ax2.plot(tp_axis, mean_corr, color="#1E88E5", lw=0.9, label="X-tilt corr")
    ax2.set_xlabel("TP"); ax2.set_ylabel("Mean ECC X (um)")
    ax2.set_title("Mean ECC X shift", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # (D) per-channel time series raw
    ax3 = fig1.add_subplot(gs1[1, 0])
    for c in range(n_ch):
        ax3.plot(tp_axis, tx_raw_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax3.set_xlabel("TP"); ax3.set_ylabel("ECC X (um)")
    ax3.set_title("Per-channel ECC X — Raw", fontsize=10)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # (E) per-channel time series corrected
    ax4 = fig1.add_subplot(gs1[1, 1])
    for c in range(n_ch):
        ax4.plot(tp_axis, tx_corr_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax4.set_xlabel("TP"); ax4.set_ylabel("ECC X (um)")
    ax4.set_title("Per-channel ECC X — X-tilt corr", fontsize=10)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # (F) raw vs corr scatter
    ax5 = fig1.add_subplot(gs1[1, 2])
    valid = ~(np.isnan(tx_raw_um) | np.isnan(tx_corr_um))
    if valid.sum() > 1:
        ax5.scatter(tx_raw_um[valid], tx_corr_um[valid], s=3, alpha=0.3,
                    color="#555", linewidths=0)
        lim = max(np.abs(tx_raw_um[valid]).max(), np.abs(tx_corr_um[valid]).max()) * 1.05
        ax5.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, alpha=0.6)
        r_sc, _ = pearsonr(tx_raw_um[valid], tx_corr_um[valid])
        ax5.set_title(f"Raw vs Corrected\nr={r_sc:.3f}", fontsize=10)
    ax5.set_xlabel("Raw ECC X (um)"); ax5.set_ylabel("Corr ECC X (um)")
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # (G) Representative channel X profile (center ch, TP5)
    _t = min(4, n_tp - 1)
    phase_ex = tifffile.imread(str(phase_paths[_t + 1])).astype(np.float32)
    roi_ex = rois[n_ch // 2]

    ax6 = fig1.add_subplot(gs1[2, 0])
    big_ex = extract_rect_roi(phase_ex, roi_ex["cy"], roi_ex["cx"],
                              roi_ex["crop_w"], CH_CROP_H_TILT).astype(np.float32)
    x_prof_ex = big_ex.mean(axis=0).astype(np.float64)
    n_fit_ex = max(3, int(len(x_prof_ex) * FIT_FRAC))
    sl_ex, ic_ex = np.polyfit(np.arange(n_fit_ex, dtype=float), x_prof_ex[:n_fit_ex], 1)
    x_ax_ex = np.arange(CH_CROP_H_TILT, dtype=float)
    ax6.plot(x_ax_ex, x_prof_ex, color="#E53935", lw=0.9, label="Raw profile")
    ax6.plot(x_ax_ex[:n_fit_ex], sl_ex * x_ax_ex[:n_fit_ex] + ic_ex,
             color="#FF9800", lw=1.2, ls="--", label="Linear fit (left 1/3)")
    ax6.plot(x_ax_ex, x_prof_ex - sl_ex * x_ax_ex, color="#1E88E5", lw=0.9, label="Corrected")
    ax6.axvline(n_fit_ex, color="#999", lw=0.7, ls=":")
    start_ex = (CH_CROP_H_TILT - roi_ex["crop_h"]) // 2
    ax6.axvspan(start_ex, start_ex + roi_ex["crop_h"], alpha=0.1, color="#1E88E5",
                label="ECC crop region")
    ax6.set_xlabel("X [px]"); ax6.set_ylabel("Phase (rad)")
    ax6.set_title(f"X profile: ch{n_ch//2} TP={_t+1}", fontsize=10)
    ax6.legend(fontsize=7)
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    # (H) raw ECC crop imshow (center ch, TP5)
    ax7 = fig1.add_subplot(gs1[2, 1])
    c_raw_ex = extract_rect_roi(phase_ex, roi_ex["cy"], roi_ex["cx"],
                                roi_ex["crop_w"], roi_ex["crop_h"]).astype(np.float32)
    off_r = compute_backsub_offset(c_raw_ex, cfg_dummy)
    ax7.imshow(to_uint8(c_raw_ex + off_r, ECC_VMIN, ECC_VMAX),
               cmap="RdBu_r", aspect="auto", interpolation="nearest")
    ax7.set_title(f"Raw crop (u8)  ch{n_ch//2} TP={_t+1}", fontsize=8)
    ax7.set_xlabel("X [px]"); ax7.set_ylabel("Y [px]")
    ax7.tick_params(labelsize=6)

    # (I) corrected ECC crop imshow (center ch, TP5)
    ax8 = fig1.add_subplot(gs1[2, 2])
    crop_c_ex, _ = compute_x_slope_and_correct(
        phase_ex, roi_ex["cy"], roi_ex["cx"], roi_ex["crop_w"], roi_ex["crop_h"]
    )
    off_c = compute_backsub_offset(crop_c_ex, cfg_dummy)
    ax8.imshow(to_uint8(crop_c_ex + off_c, ECC_VMIN, ECC_VMAX),
               cmap="RdBu_r", aspect="auto", interpolation="nearest")
    ax8.set_title(f"X-tilt corr crop (u8)  ch{n_ch//2} TP={_t+1}", fontsize=8)
    ax8.set_xlabel("X [px]"); ax8.set_ylabel("Y [px]")
    ax8.tick_params(labelsize=6)

    fig1.suptitle(
        f"Ch X-tilt correction (270px → left-1/3 fit → subtract → 80px crop) vs ECC X\n"
        f"Inter-ch std: raw={np.nanmean(std_raw):.4f} um  corr={np.nanmean(std_corr):.4f} um",
        fontsize=10,
    )

    save_figure(
        fig1,
        params={
            "ch_crop_h_tilt": CH_CROP_H_TILT, "fit_frac": FIT_FRAC,
            "n_ch": n_ch, "n_tp": n_tp,
            "mean_std_raw_um": float(np.nanmean(std_raw)),
            "mean_std_corr_um": float(np.nanmean(std_corr)),
        },
        description=(
            f"Ch X-tilt correction (270px left-1/3 polyfit) on ECC X. "
            f"raw std={np.nanmean(std_raw):.4f} um, corr std={np.nanmean(std_corr):.4f} um"
        ),
        data={
            "tx_raw_um": tx_raw_um, "tx_corr_um": tx_corr_um,
            "std_raw": std_raw, "std_corr": std_corr, "slopes": slopes,
        },
    )
    plt.close(fig1)

    # =================================================================
    # Fig 2: Wide crop images for representative 4ch (raw 40x270 / corrected 40x270) + X profile
    # =================================================================
    vis_ch_indices = np.linspace(0, n_ch - 1, VIS_N_CH, dtype=int).tolist()
    n_fit_vis = max(3, int(CH_CROP_H_TILT * FIT_FRAC))  # 90
    ecc_start  = (CH_CROP_H_TILT - rois[0]["crop_h"]) // 2  # 95 (when crop_h=80)

    fig2, axes2 = plt.subplots(3, VIS_N_CH, figsize=(4 * VIS_N_CH, 9))
    fig2.subplots_adjust(hspace=0.50, wspace=0.35,
                         left=0.06, right=0.98, top=0.90, bottom=0.06)

    for col, c_idx in enumerate(vis_ch_indices):
        roi_v = rois[c_idx]
        ecc_s = (CH_CROP_H_TILT - roi_v["crop_h"]) // 2

        # Wide big crop: shape (crop_w=40, 270)
        big_v = extract_rect_roi(
            phase_ex, roi_v["cy"], roi_v["cx"], roi_v["crop_w"], CH_CROP_H_TILT
        ).astype(np.float32)
        x_prof_v = big_v.mean(axis=0).astype(np.float64)
        sl_v, ic_v = np.polyfit(
            np.arange(n_fit_vis, dtype=float), x_prof_v[:n_fit_vis], 1
        )
        x_ax_v = np.arange(CH_CROP_H_TILT, dtype=float)
        corrected_big_v = big_v - (sl_v * np.arange(CH_CROP_H_TILT, dtype=np.float32))[np.newaxis, :]

        # Common vmin/vmax (unified at 2-98 percentile of raw)
        vmin_v, vmax_v = np.nanpercentile(big_v, [2, 98])

        # Row 0: X profile
        ax_p = axes2[0, col]
        ax_p.plot(x_ax_v, x_prof_v, color="#E53935", lw=0.9, label="Raw")
        ax_p.plot(x_ax_v[:n_fit_vis], sl_v * x_ax_v[:n_fit_vis] + ic_v,
                  color="#FF9800", lw=1.2, ls="--", label="Fit")
        ax_p.plot(x_ax_v, x_prof_v - sl_v * x_ax_v, color="#1E88E5", lw=0.9, label="Corr")
        ax_p.axvline(n_fit_vis, color="#999", lw=0.7, ls=":")
        ax_p.axvspan(ecc_s, ecc_s + roi_v["crop_h"], alpha=0.12, color="#1E88E5")
        ax_p.set_xlabel("X [px]"); ax_p.set_ylabel("Phase (rad)")
        ax_p.set_title(f"ch{c_idx}  TP={_t+1}", fontsize=8)
        if col == 0:
            ax_p.legend(fontsize=6, loc="upper right")
        ax_p.spines["top"].set_visible(False); ax_p.spines["right"].set_visible(False)

        # Row 1: raw big crop imshow (40x270)
        ax_r = axes2[1, col]
        ax_r.imshow(big_v, cmap="RdBu_r", aspect="auto", interpolation="nearest",
                    vmin=vmin_v, vmax=vmax_v)
        ax_r.axvline(n_fit_vis, color="yellow", lw=0.8, ls=":")
        ax_r.axvline(ecc_s,                   color="cyan",   lw=0.8, ls="--")
        ax_r.axvline(ecc_s + roi_v["crop_h"], color="cyan",   lw=0.8, ls="--")
        ax_r.set_title(f"ch{c_idx}  Raw (40×270)", fontsize=8)
        ax_r.set_xlabel("X [px]"); ax_r.set_ylabel("Y [px]")
        ax_r.tick_params(labelsize=6)

        # Row 2: corrected big crop imshow (40x270)
        ax_c = axes2[2, col]
        ax_c.imshow(corrected_big_v, cmap="RdBu_r", aspect="auto", interpolation="nearest",
                    vmin=vmin_v, vmax=vmax_v)
        ax_c.axvline(n_fit_vis, color="yellow", lw=0.8, ls=":")
        ax_c.axvline(ecc_s,                   color="cyan",   lw=0.8, ls="--")
        ax_c.axvline(ecc_s + roi_v["crop_h"], color="cyan",   lw=0.8, ls="--")
        ax_c.set_title(f"ch{c_idx}  X-tilt corr (40×270)", fontsize=8)
        ax_c.set_xlabel("X [px]"); ax_c.set_ylabel("Y [px]")
        ax_c.tick_params(labelsize=6)

    fig2.suptitle(
        f"X-tilt correction: raw vs corrected 40×270 crops  |  TP={_t+1}\n"
        f"yellow: BG fit boundary (left 90px)  |  cyan: ECC crop region",
        fontsize=10,
    )

    save_figure(
        fig2,
        params={
            "ch_crop_h_tilt": CH_CROP_H_TILT, "fit_frac": FIT_FRAC,
            "vis_ch_indices": vis_ch_indices,
            "example_tp": int(_t + 1),
        },
        description=(
            f"X-tilt correction image visualization: raw/corrected 40×270 crops "
            f"for {VIS_N_CH} representative channels at TP={_t+1}"
        ),
        data={
            "vis_ch_indices": np.array(vis_ch_indices),
        },
    )
    plt.close(fig2)

    print("\nDone.")


if __name__ == "__main__":
    main()
