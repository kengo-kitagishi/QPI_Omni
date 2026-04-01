"""
analyze_tilt_corrected_ecc.py
-----------------------------
BG Y-tilt slope をフレームごとに除去してから ECC を実行し、
補正前後の ECC Y シフトのチャネル間ばらつきを比較する。

補正方法:
  1. 各フレームの BG Y-tilt slope [rad/px] を polyfit で推定
  2. 各チャネル crop から slope * y_local を引いてから ECC
     (ref フレームも同様に slope_0 で補正)
  3. backsub → to_uint8 → ecc_align の順は変えない

比較指標:
  per-frame inter-channel std of ECC Y shift [um]
    → 補正後に小さくなれば、チャネル間のずれが揃っている

TODO: 同様の補正を grid パイプライン (compute_pos_shifts.py) にも適用する
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
# 設定
# ============================================================
PHASE_DIR = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON = Path(
    r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
    r"\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"
)

PIXEL_SCALE_UM = 0.3462
ECC_VMIN = -5.0
ECC_VMAX =  2.0

# BG Y-tilt 推定用 ROI（analyze_bg_ytilt_vs_ecc.py と同じ設定）
BG_CX     = 100
BG_CY     = 255
BG_CROP_W = 240   # Y 方向
BG_CROP_H = 30    # X 方向
N_MAX_TP  = None  # 検証用フレーム上限（None で全フレーム）
# ============================================================


def compute_y_slope(phase_img: np.ndarray) -> float:
    """BG ROI の Y プロファイルを全点 polyfit → slope [rad/px]"""
    crop = extract_rect_roi(phase_img, BG_CY, BG_CX, BG_CROP_W, BG_CROP_H)
    row_mean = crop.mean(axis=1).astype(np.float64)
    x = np.arange(len(row_mean), dtype=float)
    slope, _ = np.polyfit(x, row_mean, 1)
    return float(slope)


def apply_y_tilt_correction(crop: np.ndarray, slope: float) -> np.ndarray:
    """
    crop (shape: crop_w × crop_h) の Y 方向チルトを除去する。
    y_local=0 を基準に slope * y_local を引く。
    """
    crop_w = crop.shape[0]
    y_local = np.arange(crop_w, dtype=np.float32)
    correction = (slope * y_local).astype(np.float32)
    return crop - correction[:, np.newaxis]


def ecc_pair(ref_u8, crop_raw, slope_ref, slope_tp, corrected: bool):
    """補正あり/なしで ECC を実行。ty [px] を返す（失敗時 nan）。"""
    if corrected:
        crop_f = apply_y_tilt_correction(crop_raw.astype(np.float32), slope_tp - slope_ref)
    else:
        crop_f = crop_raw.astype(np.float32)
    cfg_dummy = {}
    offset = compute_backsub_offset(crop_f, cfg_dummy)
    u8 = to_uint8(crop_f + offset, ECC_VMIN, ECC_VMAX)
    result = ecc_align(ref_u8, u8)
    return result[1] if result is not None else np.nan  # image-Y shift [px]


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

    # ---- ref フレーム ----
    phase_ref = tifffile.imread(str(phase_paths[0])).astype(np.float32)
    slope_ref = compute_y_slope(phase_ref)
    print(f"slope_ref = {slope_ref:.6f} rad/px")

    # ref crops（補正なし版 — 補正あり版は slope_tp-slope_ref=0 で同一になる）
    cfg_dummy = {}
    ref_crops_u8 = []
    ref_crops_corr_u8 = []
    for roi in rois:
        crop = extract_rect_roi(
            phase_ref, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
        ).astype(np.float32)
        # 補正なし
        off = compute_backsub_offset(crop, cfg_dummy)
        ref_crops_u8.append(to_uint8(crop + off, ECC_VMIN, ECC_VMAX))
        # 補正あり（ref の slope を引く）
        crop_c = apply_y_tilt_correction(crop, slope_ref)
        off_c = compute_backsub_offset(crop_c, cfg_dummy)
        ref_crops_corr_u8.append(to_uint8(crop_c + off_c, ECC_VMIN, ECC_VMAX))

    ty_raw  = np.full((n_ch, n_tp), np.nan)
    ty_corr = np.full((n_ch, n_tp), np.nan)
    slopes  = np.full(n_tp, np.nan)

    for t_idx, path in enumerate(phase_paths[1:]):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except Exception:
            continue
        slope_tp = compute_y_slope(phase_tp)
        slopes[t_idx] = slope_tp

        for c_idx, roi in enumerate(rois):
            crop = extract_rect_roi(
                phase_tp, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
            ).astype(np.float32)

            # 補正なし
            off = compute_backsub_offset(crop, cfg_dummy)
            res_raw = ecc_align(ref_crops_u8[c_idx],
                                to_uint8(crop + off, ECC_VMIN, ECC_VMAX))
            if res_raw is not None:
                ty_raw[c_idx, t_idx] = res_raw[1]

            # 補正あり（differential slope = slope_tp - slope_ref を除去）
            crop_c = apply_y_tilt_correction(crop, slope_tp - slope_ref)
            off_c = compute_backsub_offset(crop_c, cfg_dummy)
            res_corr = ecc_align(ref_crops_corr_u8[c_idx],
                                 to_uint8(crop_c + off_c, ECC_VMIN, ECC_VMAX))
            if res_corr is not None:
                ty_corr[c_idx, t_idx] = res_corr[1]

        if (t_idx + 1) % 20 == 0:
            print(f"  TP {t_idx+1}/{n_tp}  slope={slope_tp:.6f}", flush=True)

    ty_raw_um  = ty_raw  * PIXEL_SCALE_UM
    ty_corr_um = ty_corr * PIXEL_SCALE_UM

    # ---- 指標計算 ----
    # per-frame inter-channel std
    std_raw  = np.nanstd(ty_raw_um,  axis=0)   # (n_tp,)
    std_corr = np.nanstd(ty_corr_um, axis=0)

    # per-frame mean
    mean_raw  = np.nanmean(ty_raw_um,  axis=0)
    mean_corr = np.nanmean(ty_corr_um, axis=0)

    print(f"\nInter-channel std [um]  raw: {np.nanmean(std_raw):.4f}  corr: {np.nanmean(std_corr):.4f}")
    print(f"Mean ECC Y shift [um]   raw: {np.nanmean(np.abs(mean_raw)):.4f}  corr: {np.nanmean(np.abs(mean_corr)):.4f}")

    # ---- 図 ----
    tp_axis = np.arange(n_tp)
    fig = plt.figure(figsize=(14, 9))
    gs = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38,
                           left=0.08, right=0.97, top=0.91, bottom=0.08)

    # (A) per-frame inter-channel std の時系列
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(tp_axis, std_raw,  color="#E53935", lw=0.9, label="Raw")
    ax0.plot(tp_axis, std_corr, color="#1E88E5", lw=0.9, label="Tilt-corrected")
    ax0.set_xlabel("TP")
    ax0.set_ylabel("Inter-channel std of ECC Y (um)")
    ax0.set_title("Inter-channel std (per frame)", fontsize=10)
    ax0.legend(fontsize=8)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # (B) std の histogram（raw vs corr）
    ax1 = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, max(np.nanmax(std_raw), np.nanmax(std_corr)) * 1.1, 30)
    ax1.hist(std_raw[~np.isnan(std_raw)],   bins=bins, alpha=0.6,
             color="#E53935", label=f"Raw  μ={np.nanmean(std_raw):.4f}")
    ax1.hist(std_corr[~np.isnan(std_corr)], bins=bins, alpha=0.6,
             color="#1E88E5", label=f"Corr μ={np.nanmean(std_corr):.4f}")
    ax1.set_xlabel("Inter-channel std (um)")
    ax1.set_ylabel("Count")
    ax1.set_title("Histogram of inter-channel std", fontsize=10)
    ax1.legend(fontsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # (C) mean ECC Y の時系列（raw vs corr）
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.plot(tp_axis, mean_raw,  color="#E53935", lw=0.9, label="Raw mean")
    ax2.plot(tp_axis, mean_corr, color="#1E88E5", lw=0.9, label="Corr mean")
    ax2.set_xlabel("TP")
    ax2.set_ylabel("Mean ECC Y (um)")
    ax2.set_title("Mean ECC Y shift (per frame)", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # (D) per-channel 時系列（raw） — チャネルごとの線
    COLORS = plt.cm.tab20(np.linspace(0, 1, n_ch))
    ax3 = fig.add_subplot(gs[1, 0])
    for c_idx in range(n_ch):
        ax3.plot(tp_axis, ty_raw_um[c_idx], lw=0.5, alpha=0.7,
                 color=COLORS[c_idx], label=f"ch{c_idx}")
    ax3.set_xlabel("TP")
    ax3.set_ylabel("ECC Y (um)")
    ax3.set_title("Per-channel ECC Y — Raw", fontsize=10)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # (E) per-channel 時系列（corrected）
    ax4 = fig.add_subplot(gs[1, 1])
    for c_idx in range(n_ch):
        ax4.plot(tp_axis, ty_corr_um[c_idx], lw=0.5, alpha=0.7,
                 color=COLORS[c_idx])
    ax4.set_xlabel("TP")
    ax4.set_ylabel("ECC Y (um)")
    ax4.set_title("Per-channel ECC Y — Tilt-corrected", fontsize=10)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # (F) scatter: raw ty vs corr ty (all points)
    ax5 = fig.add_subplot(gs[1, 2])
    valid = ~(np.isnan(ty_raw_um) | np.isnan(ty_corr_um))
    ax5.scatter(ty_raw_um[valid], ty_corr_um[valid],
                s=3, alpha=0.3, color="#555", linewidths=0)
    lim = max(np.abs(ty_raw_um[valid]).max(), np.abs(ty_corr_um[valid]).max()) * 1.05
    ax5.plot([-lim, lim], [-lim, lim], "r--", lw=0.8, alpha=0.6)
    r_sc, p_sc = pearsonr(ty_raw_um[valid], ty_corr_um[valid])
    ax5.set_xlabel("Raw ECC Y (um)")
    ax5.set_ylabel("Tilt-corrected ECC Y (um)")
    ax5.set_title(f"Raw vs Corrected (all ch×TP)\nr={r_sc:.3f}", fontsize=10)
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    fig.suptitle(
        f"Y-tilt correction effect on ECC  [BG slope, differential]\n"
        f"Inter-ch std: raw={np.nanmean(std_raw):.4f} um → corr={np.nanmean(std_corr):.4f} um",
        fontsize=10,
    )

    save_figure(
        fig,
        params={
            "bg_cx": BG_CX, "bg_cy": BG_CY,
            "bg_crop_w": BG_CROP_W, "bg_crop_h": BG_CROP_H,
            "mean_std_raw_um": float(np.nanmean(std_raw)),
            "mean_std_corr_um": float(np.nanmean(std_corr)),
            "n_ch": n_ch, "n_tp": n_tp,
        },
        description=(
            f"Y-tilt correction effect on ECC Y inter-channel std. "
            f"raw std={np.nanmean(std_raw):.4f} um, corr std={np.nanmean(std_corr):.4f} um"
        ),
        data={
            "ty_raw_um": ty_raw_um,
            "ty_corr_um": ty_corr_um,
            "std_raw": std_raw,
            "std_corr": std_corr,
            "slopes": slopes,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
