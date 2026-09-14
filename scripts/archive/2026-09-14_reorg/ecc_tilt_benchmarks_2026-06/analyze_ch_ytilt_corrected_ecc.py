"""
analyze_ch_ytilt_corrected_ecc.py
----------------------------------
Expand channel ROI to 270px in Y direction, linear fit on the top 1/3 (90px) BG, subtract, then ECC.
Compare inter-channel std of ECC Y shift before and after correction.

Processing:
  1. For each channel, obtain a Y ROI with crop_w expanded to 270px
  2. Polyfit on the top 1/3 (90pt) of Y profile (column mean) -> slope [rad/px]
  3. Subtract slope * y_local from the 270px crop
  4. Extract central crop_w=40px, backsub -> to_uint8 -> ECC
  5. Apply the same correction to the grid ref side (grid ph_009: 270px crop -> correction -> 40px extraction)
  6. Compare with uncorrected (ECC using grid_ref_crops.tif as-is)

Comparison metric:
  per-frame inter-channel std of ECC Y shift [um]
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

# Y-direction tilt correction parameters
CH_CROP_W_TILT = 270   # expanded size in Y direction [px]
FIT_FRAC       = 1/3   # top 1/3 (90pt) = BG region
N_MAX_TP       = 20    # max frames for verification (None for all frames)
# ============================================================


def compute_y_slope_and_correct(phase_img: np.ndarray,
                                cy: int, cx: int,
                                orig_crop_w: int, crop_h: int
                                ) -> tuple[np.ndarray, float]:
    """
    Estimate slope from the top 1/3 of the 270px Y-expanded crop and return the corrected orig_crop_w crop.
    Returns: (corrected_crop [orig_crop_w x crop_h], slope [rad/px])
    """
    big_crop = extract_rect_roi(
        phase_img, cy, cx, CH_CROP_W_TILT, crop_h
    ).astype(np.float32)   # (270, crop_h)

    y_profile = big_crop.mean(axis=1).astype(np.float64)  # (270,)
    n_fit = max(3, int(len(y_profile) * FIT_FRAC))        # 90
    slope, intercept = np.polyfit(np.arange(n_fit, dtype=float), y_profile[:n_fit], 1)

    # Correction: subtract slope * y from all 270px
    y_local = np.arange(CH_CROP_W_TILT, dtype=np.float32)
    corrected_big = big_crop - (slope * y_local).astype(np.float32)[:, np.newaxis]

    # Extract central orig_crop_w px
    start = (CH_CROP_W_TILT - orig_crop_w) // 2
    corrected_crop = corrected_big[start:start + orig_crop_w, :]

    return corrected_crop, float(slope)


def main():
    phase_paths = sorted(PHASE_DIR.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: {PHASE_DIR}")
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

    # ---- Corrected ref (grid ph_009: 270px -> correction -> 40px extraction -> backsub -> u8) ----
    grid_phase = tifffile.imread(str(GRID_REF_PHASE)).astype(np.float32)
    cfg_dummy = {}
    ref_u8_corr = []
    grid_ref_slopes = []
    for roi in rois:
        crop_c, sl = compute_y_slope_and_correct(
            grid_phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
        )
        offset = compute_backsub_offset(crop_c, cfg_dummy)
        ref_u8_corr.append(to_uint8(crop_c + offset, ECC_VMIN, ECC_VMAX))
        grid_ref_slopes.append(sl)
    print(f"grid ref slopes: min={min(grid_ref_slopes):.5f} max={max(grid_ref_slopes):.5f}")

    ty_raw  = np.full((n_ch, n_tp), np.nan)
    ty_corr = np.full((n_ch, n_tp), np.nan)
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
                ty_raw[c_idx, t_idx] = res_raw[1]

            # --- Corrected ECC (270px -> fit -> subtract -> 40px extraction) ---
            crop_c, sl = compute_y_slope_and_correct(
                phase_tp, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
            )
            slopes[c_idx, t_idx] = sl
            off_c = compute_backsub_offset(crop_c, cfg_dummy)
            res_corr = ecc_align(ref_u8_corr[c_idx],
                                 to_uint8(crop_c + off_c, ECC_VMIN, ECC_VMAX))
            if res_corr is not None:
                ty_corr[c_idx, t_idx] = res_corr[1]

        if (t_idx + 1) % 5 == 0:
            print(f"  TP {t_idx+1}/{n_tp}", flush=True)

    ty_raw_um  = ty_raw  * PIXEL_SCALE_UM
    ty_corr_um = ty_corr * PIXEL_SCALE_UM

    std_raw  = np.nanstd(ty_raw_um,  axis=0)
    std_corr = np.nanstd(ty_corr_um, axis=0)
    mean_raw  = np.nanmean(ty_raw_um,  axis=0)
    mean_corr = np.nanmean(ty_corr_um, axis=0)

    print(f"\nInter-channel std  raw={np.nanmean(std_raw):.4f} um  corr={np.nanmean(std_corr):.4f} um")
    print(f"Mean |ECC Y|        raw={np.nanmean(np.abs(mean_raw)):.4f} um  corr={np.nanmean(np.abs(mean_corr)):.4f} um")

    # ---- Figure ----
    tp_axis = np.arange(n_tp)
    COLORS = plt.cm.tab20(np.linspace(0, 1, n_ch))
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, hspace=0.48, wspace=0.38,
                           left=0.07, right=0.97, top=0.92, bottom=0.07)

    # (A) inter-channel std time series
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(tp_axis, std_raw,  color="#E53935", lw=0.9, label="Raw")
    ax0.plot(tp_axis, std_corr, color="#1E88E5", lw=0.9, label="Y-tilt corr")
    ax0.set_xlabel("TP"); ax0.set_ylabel("Inter-ch std (um)")
    ax0.set_title("Inter-channel std", fontsize=10)
    ax0.legend(fontsize=8)
    ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)

    # (B) std histogram
    ax1 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, max(np.nanmax(std_raw), np.nanmax(std_corr)) * 1.1, 25)
    ax1.hist(std_raw[~np.isnan(std_raw)],   bins=bins, alpha=0.6,
             color="#E53935", label=f"Raw μ={np.nanmean(std_raw):.4f}")
    ax1.hist(std_corr[~np.isnan(std_corr)], bins=bins, alpha=0.6,
             color="#1E88E5", label=f"Corr μ={np.nanmean(std_corr):.4f}")
    ax1.set_xlabel("Inter-ch std (um)"); ax1.set_ylabel("Count")
    ax1.set_title("Histogram", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # (C) mean ECC Y time series
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(tp_axis, mean_raw,  color="#E53935", lw=0.9, label="Raw")
    ax2.plot(tp_axis, mean_corr, color="#1E88E5", lw=0.9, label="Y-tilt corr")
    ax2.set_xlabel("TP"); ax2.set_ylabel("Mean ECC Y (um)")
    ax2.set_title("Mean ECC Y shift", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    # (D) per-channel time series raw
    ax3 = fig.add_subplot(gs[1, 0])
    for c in range(n_ch):
        ax3.plot(tp_axis, ty_raw_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax3.set_xlabel("TP"); ax3.set_ylabel("ECC Y (um)")
    ax3.set_title("Per-channel ECC Y — Raw", fontsize=10)
    ax3.spines["top"].set_visible(False); ax3.spines["right"].set_visible(False)

    # (E) per-channel time series corrected
    ax4 = fig.add_subplot(gs[1, 1])
    for c in range(n_ch):
        ax4.plot(tp_axis, ty_corr_um[c], lw=0.5, alpha=0.7, color=COLORS[c])
    ax4.set_xlabel("TP"); ax4.set_ylabel("ECC Y (um)")
    ax4.set_title("Per-channel ECC Y — Y-tilt corr", fontsize=10)
    ax4.spines["top"].set_visible(False); ax4.spines["right"].set_visible(False)

    # (F) raw vs corr scatter
    ax5 = fig.add_subplot(gs[1, 2])
    valid = ~(np.isnan(ty_raw_um) | np.isnan(ty_corr_um))
    ax5.scatter(ty_raw_um[valid], ty_corr_um[valid], s=3, alpha=0.3,
                color="#555", linewidths=0)
    lim = max(np.abs(ty_raw_um[valid]).max(), np.abs(ty_corr_um[valid]).max()) * 1.05
    ax5.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, alpha=0.6)
    r_sc, _ = pearsonr(ty_raw_um[valid], ty_corr_um[valid])
    ax5.set_xlabel("Raw ECC Y (um)"); ax5.set_ylabel("Corr ECC Y (um)")
    ax5.set_title(f"Raw vs Corrected\nr={r_sc:.3f}", fontsize=10)
    ax5.spines["top"].set_visible(False); ax5.spines["right"].set_visible(False)

    # (G) Representative channel Y profile (before/after correction): ch6, TP5
    ax6 = fig.add_subplot(gs[2, 0])
    _t = min(4, n_tp - 1)
    phase_ex = tifffile.imread(str(phase_paths[_t + 1])).astype(np.float32)
    roi_ex = rois[n_ch // 2]
    big = extract_rect_roi(phase_ex, roi_ex["cy"], roi_ex["cx"],
                           CH_CROP_W_TILT, roi_ex["crop_h"]).astype(np.float32)
    y_prof = big.mean(axis=1).astype(np.float64)
    n_fit = max(3, int(len(y_prof) * FIT_FRAC))
    sl_ex, ic_ex = np.polyfit(np.arange(n_fit, dtype=float), y_prof[:n_fit], 1)
    y_ax = np.arange(CH_CROP_W_TILT, dtype=float)
    fit_line = sl_ex * y_ax + ic_ex
    y_corr_prof = y_prof - sl_ex * y_ax
    ax6.plot(y_ax, y_prof,       color="#E53935", lw=0.9, label="Raw profile")
    ax6.plot(y_ax[:n_fit], fit_line[:n_fit], color="#FF9800", lw=1.2, ls="--", label="Linear fit (top 1/3)")
    ax6.plot(y_ax, y_corr_prof,  color="#1E88E5", lw=0.9, label="Corrected")
    ax6.axvline(n_fit, color="#999", lw=0.7, ls=":")
    start = (CH_CROP_W_TILT - roi_ex["crop_w"]) // 2
    ax6.axvspan(start, start + roi_ex["crop_w"], alpha=0.1, color="#1E88E5", label="ECC crop region")
    ax6.set_xlabel("Y [px]"); ax6.set_ylabel("Phase (rad)")
    ax6.set_title(f"Y profile: ch{n_ch//2} TP={_t+1}", fontsize=10)
    ax6.legend(fontsize=7)
    ax6.spines["top"].set_visible(False); ax6.spines["right"].set_visible(False)

    # (H) Corrected crop colormap (raw vs corr)
    for col_i, (label, use_corr) in enumerate([("Raw crop (u8)", False), ("Y-tilt corr crop (u8)", True)]):
        ax_ = fig.add_subplot(gs[2, col_i + 1])
        if use_corr:
            crop_show, _ = compute_y_slope_and_correct(
                phase_ex, roi_ex["cy"], roi_ex["cx"], roi_ex["crop_w"], roi_ex["crop_h"]
            )
            off_ = compute_backsub_offset(crop_show, cfg_dummy)
            u8_show = to_uint8(crop_show + off_, ECC_VMIN, ECC_VMAX)
        else:
            c_raw = extract_rect_roi(phase_ex, roi_ex["cy"], roi_ex["cx"],
                                     roi_ex["crop_w"], roi_ex["crop_h"]).astype(np.float32)
            off_ = compute_backsub_offset(c_raw, cfg_dummy)
            u8_show = to_uint8(c_raw + off_, ECC_VMIN, ECC_VMAX)
        ax_.imshow(u8_show, cmap="RdBu_r", aspect="auto", interpolation="nearest")
        ax_.set_title(f"{label}\nch{n_ch//2} TP={_t+1}", fontsize=8)
        ax_.set_xlabel("X [px]"); ax_.set_ylabel("Y [px]")
        ax_.tick_params(labelsize=6)

    fig.suptitle(
        f"Ch Y-tilt correction (270px→top-1/3 fit→subtract→40px crop) vs ECC Y\n"
        f"Inter-ch std: raw={np.nanmean(std_raw):.4f} um  corr={np.nanmean(std_corr):.4f} um",
        fontsize=10,
    )

    save_figure(
        fig,
        params={
            "ch_crop_w_tilt": CH_CROP_W_TILT, "fit_frac": FIT_FRAC,
            "n_ch": n_ch, "n_tp": n_tp,
            "mean_std_raw_um": float(np.nanmean(std_raw)),
            "mean_std_corr_um": float(np.nanmean(std_corr)),
        },
        description=(
            f"Ch Y-tilt correction (270px top-1/3 polyfit) on ECC Y. "
            f"raw std={np.nanmean(std_raw):.4f} um, corr std={np.nanmean(std_corr):.4f} um"
        ),
        data={
            "ty_raw_um": ty_raw_um, "ty_corr_um": ty_corr_um,
            "std_raw": std_raw, "std_corr": std_corr, "slopes": slopes,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
