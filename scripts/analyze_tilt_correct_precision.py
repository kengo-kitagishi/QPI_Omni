"""
analyze_tilt_correct_precision.py
----------------------------------
ph_3/Pos0 (100フレーム・温度ドリフトなし) を使って
_tilt_correct あり/なしの ECC 精度を比較する。

真のドリフトがないデータでECCを回すと、
  ECC shift の std across 100 frames = ECC精度の下限（精度誤差）

参照 grid  : F:\\grid_2pergluc_60ms_1\\Pos1_x+0_y+0\\output_phase\\img_000000000_ph_009_phase.tif
テスト     : D:\\AquisitionData\\Kitagishi\\basler_image_seq\\ph_3\\Pos0\\output_phase\\img_*_ph_000_phase.tif

Raw ECC        : extract_rect_roi → compute_backsub_offset → to_uint8 → ecc_align
Tilt-corrected : _tilt_correct(tilt_crop_h=270, ecc_crop_h=80, fit_right=False) → to_uint8 → ecc_align
                 （compute_drift_online._tilt_correct と完全に同一の処理）
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
    _tilt_correct,
    compute_backsub_offset,
    extract_rect_roi,
    to_uint8,
    ecc_align,
)
from figure_logger import save_figure

# ============================================================
# 設定
# ============================================================
PHASE_DIR = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")

ROIS_JSON = Path(
    r"F:\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)

GRID_REF_PHASE = Path(
    r"F:\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\img_000000000_ph_009_phase.tif"
)

PIXEL_SCALE_UM = 0.34567514677103717
ECC_VMIN       = -5.0
ECC_VMAX       =  2.0
TILT_CROP_H    = 270
ECC_CROP_H     = 80
FIT_RIGHT      = False   # pos1 < pos_split=33 → 左1/3をBGフィット
# ============================================================


def compute_x_tilt_slope(phase_img: np.ndarray, cy: int, cx: int, crop_w: int) -> float:
    """270px cropの左1/3 (90px) でX方向チルトslopeを推定する [rad/px]。"""
    big = extract_rect_roi(phase_img, cy, cx, crop_w, TILT_CROP_H).astype(np.float64)
    prof = big.mean(axis=0)
    n_fit = max(3, TILT_CROP_H // 3)
    slope, _ = np.polyfit(np.arange(n_fit, dtype=float), prof[:n_fit], 1)
    return float(slope)


def main():
    # ---- Load ROIs ----
    rois = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    n_ch = len(rois)
    print(f"Channels: {n_ch}")

    # ---- Generate reference crops from F: grid z=9 ----
    print(f"Loading grid ref: {GRID_REF_PHASE.name}")
    grid_phase = tifffile.imread(str(GRID_REF_PHASE)).astype(np.float64)

    cfg_dummy = {}
    raw_ref_u8  = []   # list[n_ch] of uint8 (80px crop, backsub)
    corr_ref_u8 = []   # list[n_ch] of uint8 (80px crop, _tilt_correct)
    grid_tilt_slopes = []

    for ch, roi in enumerate(rois):
        # Raw ref: extract_rect_roi + backsub
        crop_raw = extract_rect_roi(
            grid_phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
        ).astype(np.float64)
        offset = compute_backsub_offset(crop_raw, cfg_dummy)
        raw_ref_u8.append(to_uint8(crop_raw + offset, ECC_VMIN, ECC_VMAX))

        # Tilt-corrected ref: _tilt_correct (slope+intercept両方引く)
        crop_corr = _tilt_correct(
            grid_phase, roi["cy"], roi["cx"], roi["crop_w"],
            TILT_CROP_H, ECC_CROP_H, fit_right=FIT_RIGHT
        )
        corr_ref_u8.append(to_uint8(crop_corr, ECC_VMIN, ECC_VMAX))

        slope = compute_x_tilt_slope(grid_phase, roi["cy"], roi["cx"], roi["crop_w"])
        grid_tilt_slopes.append(slope)

    print(f"Grid X-tilt slopes: min={min(grid_tilt_slopes):.5f}  max={max(grid_tilt_slopes):.5f} rad/px")

    # ---- Load test frames ----
    phase_paths = sorted(PHASE_DIR.glob("img_*_ph_000_phase.tif"))
    if not phase_paths:
        print(f"ERROR: phase images not found in {PHASE_DIR}")
        sys.exit(1)
    n_tp = len(phase_paths)
    print(f"Test frames: {n_tp}")

    tx_raw   = np.full((n_ch, n_tp), np.nan)
    tx_corr  = np.full((n_ch, n_tp), np.nan)
    ty_raw   = np.full((n_ch, n_tp), np.nan)
    ty_corr  = np.full((n_ch, n_tp), np.nan)
    slopes   = np.full((n_ch, n_tp), np.nan)   # X-tilt slope per channel per frame

    for t_idx, path in enumerate(phase_paths):
        try:
            phase = tifffile.imread(str(path)).astype(np.float64)
        except Exception as e:
            print(f"  frame {t_idx}: load error: {e}")
            continue

        for ch, roi in enumerate(rois):
            # --- Raw ECC ---
            crop_r = extract_rect_roi(
                phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
            ).astype(np.float64)
            offset = compute_backsub_offset(crop_r, cfg_dummy)
            res_r = ecc_align(raw_ref_u8[ch], to_uint8(crop_r + offset, ECC_VMIN, ECC_VMAX))
            if res_r is not None:
                tx_raw[ch, t_idx]  = res_r[0]
                ty_raw[ch, t_idx]  = res_r[1]

            # --- Tilt-corrected ECC ---
            crop_c = _tilt_correct(
                phase, roi["cy"], roi["cx"], roi["crop_w"],
                TILT_CROP_H, ECC_CROP_H, fit_right=FIT_RIGHT
            )
            res_c = ecc_align(corr_ref_u8[ch], to_uint8(crop_c, ECC_VMIN, ECC_VMAX))
            if res_c is not None:
                tx_corr[ch, t_idx] = res_c[0]
                ty_corr[ch, t_idx] = res_c[1]

            # --- X-tilt slope (for correlation analysis) ---
            slopes[ch, t_idx] = compute_x_tilt_slope(
                phase, roi["cy"], roi["cx"], roi["crop_w"]
            )

        if (t_idx + 1) % 20 == 0 or t_idx == 0:
            print(f"  frame {t_idx+1}/{n_tp}  "
                  f"tx_raw_mean={np.nanmean(tx_raw[:, t_idx]):.4f}px  "
                  f"tx_corr_mean={np.nanmean(tx_corr[:, t_idx]):.4f}px")

    # ---- Statistics ----
    tx_raw_nm   = tx_raw  * PIXEL_SCALE_UM * 1000   # px → nm
    tx_corr_nm  = tx_corr * PIXEL_SCALE_UM * 1000

    # std across frames (= ECC precision error per channel)
    std_raw_per_ch  = np.nanstd(tx_raw_nm,  axis=1)   # (n_ch,)
    std_corr_per_ch = np.nanstd(tx_corr_nm, axis=1)

    mean_std_raw  = float(np.nanmean(std_raw_per_ch))
    mean_std_corr = float(np.nanmean(std_corr_per_ch))

    # mean ECC X across channels per frame
    mean_tx_raw_nm  = np.nanmean(tx_raw_nm,  axis=0)   # (n_tp,)
    mean_tx_corr_nm = np.nanmean(tx_corr_nm, axis=0)
    mean_slope      = np.nanmean(slopes, axis=0)        # mean X-tilt per frame

    # Pearson r: X-tilt vs ECC X (raw)
    valid_m = ~(np.isnan(mean_tx_raw_nm) | np.isnan(mean_slope))
    r_raw, p_raw   = pearsonr(mean_slope[valid_m], mean_tx_raw_nm[valid_m])
    valid_c = ~(np.isnan(mean_tx_corr_nm) | np.isnan(mean_slope))
    r_corr, p_corr = pearsonr(mean_slope[valid_c], mean_tx_corr_nm[valid_c])

    # Per-channel pooled: X-tilt slope vs ECC X
    valid_all_r = ~(np.isnan(slopes) | np.isnan(tx_raw_nm))
    valid_all_c = ~(np.isnan(slopes) | np.isnan(tx_corr_nm))
    r_perch_raw,  p_perch_raw  = pearsonr(slopes[valid_all_r],  tx_raw_nm[valid_all_r])
    r_perch_corr, p_perch_corr = pearsonr(slopes[valid_all_c], tx_corr_nm[valid_all_c])

    print(f"\n{'='*55}")
    print(f"  ECC X precision (std across {n_tp} frames)")
    print(f"  Raw:           mean σ = {mean_std_raw:.1f} nm")
    print(f"  Tilt-corrected: mean σ = {mean_std_corr:.1f} nm")
    print(f"  Improvement:   {mean_std_raw/mean_std_corr:.2f}x")
    print(f"  r(X-tilt, ECC X): raw={r_raw:.3f}  corr={r_corr:.3f}")
    print(f"{'='*55}\n")

    # ============================================================
    # Figure
    # ============================================================
    COLORS = plt.cm.tab20(np.linspace(0, 1, n_ch))
    tp_axis = np.arange(n_tp)
    ch_axis = np.arange(n_ch)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(
        3, 3,
        hspace=0.50, wspace=0.40,
        left=0.07, right=0.97, top=0.91, bottom=0.07
    )

    # (A) Time series: mean ECC X across channels ─────────────
    ax_A = fig.add_subplot(gs[0, :2])
    ax_A.plot(tp_axis, mean_tx_raw_nm,  color="#E53935", lw=0.9, alpha=0.85,
              label=f"Raw  (σ={np.nanstd(mean_tx_raw_nm):.0f} nm)")
    ax_A.plot(tp_axis, mean_tx_corr_nm, color="#1E88E5", lw=0.9, alpha=0.85,
              label=f"Tilt-corrected  (σ={np.nanstd(mean_tx_corr_nm):.0f} nm)")
    ax_A.axhline(0, color="k", lw=0.5, ls="--", alpha=0.4)
    ax_A.set_xlabel("Frame #")
    ax_A.set_ylabel("ECC X shift (nm)")
    ax_A.set_title("ECC X shift — mean across 12 channels  (no real drift)", fontsize=10)
    ax_A.legend(fontsize=9)
    ax_A.spines["top"].set_visible(False)
    ax_A.spines["right"].set_visible(False)

    # (B) Histogram of mean tx values ─────────────────────────
    ax_B = fig.add_subplot(gs[0, 2])
    all_vals = np.concatenate([mean_tx_raw_nm[~np.isnan(mean_tx_raw_nm)],
                                mean_tx_corr_nm[~np.isnan(mean_tx_corr_nm)]])
    lim = np.max(np.abs(all_vals)) * 1.1
    bins = np.linspace(-lim, lim, 28)
    ax_B.hist(mean_tx_raw_nm[~np.isnan(mean_tx_raw_nm)],   bins=bins,
              color="#E53935", alpha=0.65,
              label=f"Raw  σ={np.nanstd(mean_tx_raw_nm):.0f} nm")
    ax_B.hist(mean_tx_corr_nm[~np.isnan(mean_tx_corr_nm)], bins=bins,
              color="#1E88E5", alpha=0.65,
              label=f"Corr σ={np.nanstd(mean_tx_corr_nm):.0f} nm")
    ax_B.set_xlabel("ECC X shift (nm)")
    ax_B.set_ylabel("Count (frames)")
    ax_B.set_title("Distribution of ECC X shift\n(mean across channels)", fontsize=10)
    ax_B.legend(fontsize=8)
    ax_B.spines["top"].set_visible(False)
    ax_B.spines["right"].set_visible(False)

    # (C) Per-channel std comparison ──────────────────────────
    ax_C = fig.add_subplot(gs[1, 0])
    bar_w = 0.38
    ax_C.bar(ch_axis - bar_w/2, std_raw_per_ch,  bar_w,
             color="#E53935", alpha=0.8, label="Raw")
    ax_C.bar(ch_axis + bar_w/2, std_corr_per_ch, bar_w,
             color="#1E88E5", alpha=0.8, label="Tilt-corrected")
    ax_C.axhline(mean_std_raw,  color="#E53935", lw=1.2, ls="--", alpha=0.6)
    ax_C.axhline(mean_std_corr, color="#1E88E5", lw=1.2, ls="--", alpha=0.6)
    ax_C.set_xlabel("Channel #")
    ax_C.set_ylabel("σ of ECC X (nm)")
    ax_C.set_title(f"Per-channel ECC X precision (σ across {n_tp} frames)\n"
                   f"Raw mean={mean_std_raw:.0f} nm  |  Corr mean={mean_std_corr:.0f} nm", fontsize=9)
    ax_C.legend(fontsize=8)
    ax_C.spines["top"].set_visible(False)
    ax_C.spines["right"].set_visible(False)

    # (D) Per-channel scatter: raw σ vs corrected σ ───────────
    ax_D = fig.add_subplot(gs[1, 1])
    ax_D.scatter(std_raw_per_ch, std_corr_per_ch,
                 c=ch_axis, cmap="tab20", s=60, zorder=3)
    for c in range(n_ch):
        ax_D.annotate(f"{c}", (std_raw_per_ch[c], std_corr_per_ch[c]),
                      fontsize=6, ha="center", va="bottom", alpha=0.7)
    lim_d = max(std_raw_per_ch.max(), std_corr_per_ch.max()) * 1.1
    ax_D.plot([0, lim_d], [0, lim_d], "k--", lw=0.8, alpha=0.4, label="y=x (no change)")
    ax_D.set_xlim(0, lim_d)
    ax_D.set_ylim(0, lim_d)
    ax_D.set_xlabel("Raw ECC X σ (nm)")
    ax_D.set_ylabel("Tilt-corrected ECC X σ (nm)")
    ax_D.set_title("Per-channel: Raw σ vs Corrected σ\n(below y=x → improved)", fontsize=9)
    ax_D.legend(fontsize=8)
    ax_D.spines["top"].set_visible(False)
    ax_D.spines["right"].set_visible(False)

    # (E) X-tilt slope vs ECC X (mean across channels) ────────
    ax_E = fig.add_subplot(gs[1, 2])
    ax_E.scatter(mean_slope[valid_m], mean_tx_raw_nm[valid_m],
                 s=14, alpha=0.7, color="#E53935", linewidths=0,
                 label=f"Raw  r={r_raw:.3f}")
    ax_E.scatter(mean_slope[valid_c], mean_tx_corr_nm[valid_c],
                 s=14, alpha=0.7, color="#1E88E5", linewidths=0,
                 label=f"Corr r={r_corr:.3f}")
    if valid_m.sum() > 1:
        c_r = np.polyfit(mean_slope[valid_m], mean_tx_raw_nm[valid_m], 1)
        xr  = np.linspace(mean_slope[valid_m].min(), mean_slope[valid_m].max(), 100)
        ax_E.plot(xr, np.polyval(c_r, xr), color="#E53935", lw=1.2)
    if valid_c.sum() > 1:
        c_c = np.polyfit(mean_slope[valid_c], mean_tx_corr_nm[valid_c], 1)
        xc  = np.linspace(mean_slope[valid_c].min(), mean_slope[valid_c].max(), 100)
        ax_E.plot(xc, np.polyval(c_c, xc), color="#1E88E5", lw=1.2)
    ax_E.set_xlabel("Mean X-tilt slope (rad/px)")
    ax_E.set_ylabel("Mean ECC X shift (nm)")
    ax_E.set_title("X-tilt slope vs ECC X shift\n(mean across channels)", fontsize=9)
    ax_E.legend(fontsize=8)
    ax_E.spines["top"].set_visible(False)
    ax_E.spines["right"].set_visible(False)

    # (F) Per-channel × all frames scatter ────────────────────
    ax_F = fig.add_subplot(gs[2, 0])
    for c in range(n_ch):
        v_r = ~(np.isnan(slopes[c]) | np.isnan(tx_raw_nm[c]))
        v_c = ~(np.isnan(slopes[c]) | np.isnan(tx_corr_nm[c]))
        ax_F.scatter(slopes[c, v_r], tx_raw_nm[c, v_r],
                     s=3, alpha=0.35, color=COLORS[c], linewidths=0)
    if valid_all_r.sum() > 1:
        c_fr = np.polyfit(slopes[valid_all_r], tx_raw_nm[valid_all_r], 1)
        xfr  = np.linspace(slopes[valid_all_r].min(), slopes[valid_all_r].max(), 100)
        ax_F.plot(xfr, np.polyval(c_fr, xfr), "r-", lw=1.5,
                  label=f"r={r_perch_raw:.3f}")
    ax_F.set_xlabel("X-tilt slope (rad/px)")
    ax_F.set_ylabel("ECC X shift (nm)")
    ax_F.set_title(f"Per-ch × all frames: Raw\nr={r_perch_raw:.3f}  p={p_perch_raw:.1e}", fontsize=9)
    ax_F.legend(fontsize=8)
    ax_F.spines["top"].set_visible(False)
    ax_F.spines["right"].set_visible(False)

    ax_G = fig.add_subplot(gs[2, 1])
    for c in range(n_ch):
        v_c = ~(np.isnan(slopes[c]) | np.isnan(tx_corr_nm[c]))
        ax_G.scatter(slopes[c, v_c], tx_corr_nm[c, v_c],
                     s=3, alpha=0.35, color=COLORS[c], linewidths=0)
    if valid_all_c.sum() > 1:
        c_fc = np.polyfit(slopes[valid_all_c], tx_corr_nm[valid_all_c], 1)
        xfc  = np.linspace(slopes[valid_all_c].min(), slopes[valid_all_c].max(), 100)
        ax_G.plot(xfc, np.polyval(c_fc, xfc), "b-", lw=1.5,
                  label=f"r={r_perch_corr:.3f}")
    ax_G.set_xlabel("X-tilt slope (rad/px)")
    ax_G.set_ylabel("ECC X shift (nm)")
    ax_G.set_title(f"Per-ch × all frames: Tilt-corrected\nr={r_perch_corr:.3f}  p={p_perch_corr:.1e}", fontsize=9)
    ax_G.legend(fontsize=8)
    ax_G.spines["top"].set_visible(False)
    ax_G.spines["right"].set_visible(False)

    # (H) X-tilt time series ───────────────────────────────────
    ax_H = fig.add_subplot(gs[2, 2])
    ax_H.plot(tp_axis, mean_slope * 1000, color="#FF6F00", lw=0.9, alpha=0.85)
    ax_H.set_xlabel("Frame #")
    ax_H.set_ylabel("Mean X-tilt slope (mrad/px)")
    ax_H.set_title("X-tilt fluctuation across frames", fontsize=9)
    ax_H.spines["top"].set_visible(False)
    ax_H.spines["right"].set_visible(False)

    fig.suptitle(
        f"ECC X precision: _tilt_correct vs Raw  |  "
        f"ph_3/Pos0 ({n_tp} frames, no thermal drift)  |  "
        f"grid ref: F:/grid Pos1_x+0_y+0 z=9  |  "
        f"Raw σ={mean_std_raw:.0f} nm → Corr σ={mean_std_corr:.0f} nm  "
        f"({mean_std_raw/mean_std_corr:.2f}× improvement)",
        fontsize=10,
    )

    save_figure(
        fig,
        params={
            "n_frames": n_tp,
            "n_channels": n_ch,
            "grid_z_index": 9,
            "fit_right": FIT_RIGHT,
            "tilt_crop_h": TILT_CROP_H,
            "ecc_crop_h": ECC_CROP_H,
            "mean_std_raw_nm": round(mean_std_raw, 1),
            "mean_std_corr_nm": round(mean_std_corr, 1),
            "improvement_factor": round(mean_std_raw / mean_std_corr, 3) if mean_std_corr > 0 else None,
            "pearson_r_raw_mean": round(r_raw, 4),
            "pearson_r_corr_mean": round(r_corr, 4),
            "pearson_r_raw_perch": round(r_perch_raw, 4),
            "pearson_r_corr_perch": round(r_perch_corr, 4),
        },
        description=(
            f"_tilt_correct precision test: {n_tp} frames, no drift. "
            f"ECC X σ: raw={mean_std_raw:.0f} nm → tilt_corr={mean_std_corr:.0f} nm "
            f"({mean_std_raw/mean_std_corr:.2f}x improvement). "
            f"X-tilt vs ECC: r(raw)={r_raw:.3f} r(corr)={r_corr:.3f}"
        ),
        data={
            "tx_raw_nm":      tx_raw_nm,
            "tx_corr_nm":     tx_corr_nm,
            "ty_raw_nm":      ty_raw  * PIXEL_SCALE_UM * 1000,
            "ty_corr_nm":     ty_corr * PIXEL_SCALE_UM * 1000,
            "slopes_rad_px":  slopes,
            "std_raw_nm_per_ch":  std_raw_per_ch,
            "std_corr_nm_per_ch": std_corr_per_ch,
            "mean_tx_raw_nm":  mean_tx_raw_nm,
            "mean_tx_corr_nm": mean_tx_corr_nm,
            "mean_slope":      mean_slope,
        },
    )
    plt.close(fig)
    print("Figure saved via figure_logger.")


if __name__ == "__main__":
    main()
