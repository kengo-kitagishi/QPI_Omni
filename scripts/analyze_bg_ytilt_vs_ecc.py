"""
analyze_bg_ytilt_vs_ecc.py
--------------------------
再構成画像の背景Y方向チルト (wavefront Y-tilt) と
ECC Y方向シフトの per-frame 相関を検証する。

仮説:
  フレームごとの波面線形チルト（Y方向）が全チャネルに共通の
  ECC Y 誤差を生む → それが channel 間相関(r=0.41)の原因か？

処理:
  1. 各フレームで ECC Y シフトを全チャネルで計算 → frame-mean を取る
  2. 同フレームで背景ROI（画像左側 cx=BG_CX, crop_h=BG_CROP_H px 幅）を取り
     行平均 → 240点 Y プロファイルを作る
  3. 全 240点を np.polyfit(..., 1) で線形フィット → slope [rad/px]
  4. scatter plot: slope vs mean ECC Y shift (um) + Pearson r
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
PHASE_DIR  = Path(r"D:\AquisitionData\Kitagishi\basler_image_seq\ph_3\Pos0\output_phase")
ROIS_JSON  = Path(r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json")

# ECC パラメータ
PIXEL_SCALE_UM = 0.3462
ECC_VMIN = -5.0
ECC_VMAX =  2.0

# grid 参照 crops（prepare_drift_session.py で生成した backsub 済み float32）
GRID_REF_CROPS_TIF = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\grid_ref_crops.tif")

# 背景ROI（画像左側の背景領域）
BG_CX       = 100   # 背景 cx（チャネルは cx≈360 なので十分左）
BG_CY       = 255   # 画像中央
BG_CROP_W   = 240   # Y 方向サイズ [px]（端と端のチルトを見る）
BG_CROP_H   = 30    # X 方向サイズ [px]（細い短冊）

# Y プロファイルは純粋な背景領域 → 全点で線形フィット
N_MAX_TP = None  # 検証用フレーム上限（None で全フレーム）

# ============================================================


def compute_y_tilt(phase_img: np.ndarray) -> float:
    """背景 Y プロファイルを全点 np.polyfit → slope [rad/px] を返す。"""
    crop = extract_rect_roi(phase_img, BG_CY, BG_CX, BG_CROP_W, BG_CROP_H)
    row_mean = crop.mean(axis=1).astype(np.float64)   # (BG_CROP_W,) Y プロファイル
    x_fit = np.arange(len(row_mean), dtype=float)
    slope, _ = np.polyfit(x_fit, row_mean, 1)
    return float(slope)  # [rad/px]


def main():
    phase_paths = sorted(PHASE_DIR.glob("img_*_phase.tif"))
    if not phase_paths:
        print(f"ERROR: 位相画像が見つかりません: {PHASE_DIR}")
        sys.exit(1)
    if N_MAX_TP is not None:
        phase_paths = phase_paths[:N_MAX_TP + 1]
    print(f"Phase images: {len(phase_paths)} (N_MAX_TP={N_MAX_TP})")

    rois = json.loads(ROIS_JSON.read_text(encoding="utf-8"))
    n_ch = len(rois)
    n_tp = len(phase_paths) - 1
    print(f"Channels: {n_ch},  TPs: {n_tp}")

    # ---- ECC Y シフト（全チャネル） ----
    # grid_ref_crops.tif = backsub 済み float32 → to_uint8 で ECC reference
    grid_ref_f = tifffile.imread(str(GRID_REF_CROPS_TIF)).astype(np.float64)
    ref_crops_u8 = [to_uint8(grid_ref_f[ch], ECC_VMIN, ECC_VMAX) for ch in range(n_ch)]
    print(f"grid_ref_crops loaded: shape={grid_ref_f.shape}")

    cfg_dummy = {}

    ty_raw = np.full((n_ch, n_tp), np.nan)   # image Y 方向 [px]

    for t_idx, path in enumerate(phase_paths[1:]):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except Exception:
            continue
        for c_idx, roi in enumerate(rois):
            crop = extract_rect_roi(phase_tp, roi["cy"], roi["cx"],
                                    roi["crop_w"], roi["crop_h"]).astype(np.float32)
            offset = compute_backsub_offset(crop, cfg_dummy)
            result = ecc_align(ref_crops_u8[c_idx],
                               to_uint8(crop + offset, ECC_VMIN, ECC_VMAX))
            if result is not None:
                ty_raw[c_idx, t_idx] = result[1]   # image-Y shift [px]

        if (t_idx + 1) % 20 == 0:
            print(f"  ECC TP {t_idx+1}/{n_tp}", flush=True)

    ty_um = ty_raw * PIXEL_SCALE_UM                  # (n_ch, n_tp) [um]
    mean_ty_um = np.nanmean(ty_um, axis=0)           # (n_tp,) per-frame mean

    # ---- 背景 Y チルト ----
    print("Computing Y-tilt per frame ...")
    tilt_rad = np.full(n_tp, np.nan)
    for t_idx, path in enumerate(phase_paths[1:]):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except Exception:
            continue
        tilt_rad[t_idx] = compute_y_tilt(phase_tp)

        if (t_idx + 1) % 20 == 0:
            print(f"  Tilt TP {t_idx+1}/{n_tp}", flush=True)

    # ---- 相関 ----
    valid = ~(np.isnan(mean_ty_um) | np.isnan(tilt_rad))
    r, pval = pearsonr(tilt_rad[valid], mean_ty_um[valid])
    print(f"\nPearson r (Y-tilt vs mean ECC Y): r={r:.3f}  p={pval:.3e}  N={valid.sum()}")

    # ---- 図 ----
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 3, wspace=0.38, left=0.07, right=0.97,
                           top=0.88, bottom=0.13)

    # (A) scatter: tilt vs ECC Y
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(tilt_rad[valid], mean_ty_um[valid],
                s=18, alpha=0.7, color="#2196F3", linewidths=0)
    # 回帰直線
    coeffs = np.polyfit(tilt_rad[valid], mean_ty_um[valid], 1)
    x_line = np.linspace(tilt_rad[valid].min(), tilt_rad[valid].max(), 100)
    ax0.plot(x_line, np.polyval(coeffs, x_line), "r-", lw=1.2)
    ax0.set_xlabel("BG Y-tilt slope (rad/px)")
    ax0.set_ylabel("Mean ECC Y shift (um)")
    ax0.set_title(f"Y-tilt vs ECC Y shift\nr={r:.3f}, p={pval:.2e}", fontsize=10)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # (B) 時系列比較
    ax1 = fig.add_subplot(gs[1])
    tp_axis = np.arange(n_tp)
    ax1.plot(tp_axis, mean_ty_um, color="#2196F3", lw=0.9, label="ECC mean Y shift (um)")
    ax1b = ax1.twinx()
    ax1b.plot(tp_axis, tilt_rad, color="#FF9800", lw=0.9, alpha=0.8, label="BG Y-tilt slope (rad/px)")
    ax1.set_xlabel("TP")
    ax1.set_ylabel("ECC mean Y (um)", color="#2196F3")
    ax1b.set_ylabel("Y-tilt slope (rad/px)", color="#FF9800")
    ax1.set_title("Time series", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1b.tick_params(axis="y", labelcolor="#FF9800")
    ax1.spines["top"].set_visible(False)

    # (C) per-channel scatter（全ch × 全TP のscatter、background tiltとの関係）
    ax2 = fig.add_subplot(gs[2])
    # tilt を各 TP に broadcast → 全 ch と 1対1 対応
    tilt_broadcast = np.tile(tilt_rad, (n_ch, 1))   # (n_ch, n_tp)
    valid2d = ~(np.isnan(ty_um) | np.isnan(tilt_broadcast))
    r2, pval2 = pearsonr(tilt_broadcast[valid2d], ty_um[valid2d])
    ax2.scatter(tilt_broadcast[valid2d], ty_um[valid2d],
                s=4, alpha=0.3, color="#666666", linewidths=0)
    coeffs2 = np.polyfit(tilt_broadcast[valid2d], ty_um[valid2d], 1)
    x2 = np.linspace(tilt_broadcast[valid2d].min(), tilt_broadcast[valid2d].max(), 100)
    ax2.plot(x2, np.polyval(coeffs2, x2), "r-", lw=1.2)
    ax2.set_xlabel("BG Y-tilt slope (rad/px)")
    ax2.set_ylabel("Per-channel ECC Y shift (um)")
    ax2.set_title(f"Per-channel (all ch × all TP)\nr={r2:.3f}, p={pval2:.2e}", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"Wavefront Y-tilt vs ECC Y shift  "
        f"[BG ROI: cx={BG_CX}, crop_w={BG_CROP_W}×crop_h={BG_CROP_H}, polyfit all]\n"
        f"Mean shift r={r:.3f}  Per-channel r={r2:.3f}",
        fontsize=10
    )

    save_figure(
        fig,
        params={
            "bg_cx": BG_CX, "bg_cy": BG_CY,
            "bg_crop_w": BG_CROP_W, "bg_crop_h": BG_CROP_H,
            "pearson_r_mean": float(r), "pearson_pval_mean": float(pval),
            "pearson_r_perch": float(r2), "pearson_pval_perch": float(pval2),
            "n_valid_tp": int(valid.sum()),
        },
        description=(
            f"BG Y-tilt slope (polyfit) vs ECC Y shift correlation. "
            f"r(mean)={r:.3f}, r(per-ch)={r2:.3f}"
        ),
        data={
            "tilt_rad": tilt_rad,
            "mean_ty_um": mean_ty_um,
            "ty_um": ty_um,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
