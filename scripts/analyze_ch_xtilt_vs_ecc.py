"""
analyze_ch_xtilt_vs_ecc.py
--------------------------
各チャネルROI内のX方向チルト (wavefront X-tilt) と
ECC X方向シフトの per-frame, per-channel 相関を検証する。

仮説:
  チャネルごとの波面X方向傾きが ECC X 誤差を生んでいるか？
  前回の全体背景Y-tiltとの比較として、チャネル自身のROI内X-tiltを見る。

処理:
  1. 各チャネルの ROI を crop_h=CH_CROP_H_TILT (270) に拡大して取得
  2. X プロファイル = crop.mean(axis=0) → 270点（Y方向を平均）
  3. 左 1/3（90pt）を np.polyfit(..., 1) で線形フィット → slope [rad/px]
  4. xtilt_rad_per_px = slope  (X方向チルト量)
  5. ECC result[0] = image-X shift (tx) [px] → um に変換
  6. per-channel, per-frame の scatter + Pearson r

※ ECC 計算には JSON の元の crop パラメータを使う（crop_h=80）
   X-tilt 計算のみ crop_h=270 の拡大 ROI を使う
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

# ECC パラメータ
PIXEL_SCALE_UM = 0.3462
ECC_VMIN = -5.0
ECC_VMAX =  2.0

# grid 参照 crops（prepare_drift_session.py で生成した backsub 済み float32）
GRID_REF_CROPS_TIF = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\grid_ref_crops.tif")

# X-tilt 計算用の拡大 crop_h (X方向)
# チャネル中心 cx≈360、crop_h=270 → x1=225, x2=495
# 左 90px (x=225~314) = 背景領域（線形フィット領域）
CH_CROP_H_TILT = 270       # X方向拡大サイズ [px]
FIT_FRAC       = 1 / 3    # 左 1/3 (90pt) を線形フィット領域
N_MAX_TP       = None      # 検証用フレーム上限（None で全フレーム）
# ============================================================


def compute_x_tilt(phase_img: np.ndarray, cy: int, cx: int, crop_w: int) -> float:
    """
    チャネル ROI (crop_h=CH_CROP_H_TILT) の X方向プロファイルから
    左1/3 を np.polyfit(..., 1) で線形フィット → slope [rad/px] を返す。
    """
    crop = extract_rect_roi(
        phase_img, cy, cx, crop_w, CH_CROP_H_TILT
    ).astype(np.float64)  # shape: (crop_w, CH_CROP_H_TILT)

    x_profile = crop.mean(axis=0)  # (CH_CROP_H_TILT,) = X方向プロファイル

    n_fit = max(3, int(len(x_profile) * FIT_FRAC))  # 左 1/3 (≈90pt)
    x_fit = np.arange(n_fit, dtype=float)

    slope, _ = np.polyfit(x_fit, x_profile[:n_fit], 1)
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

    # ---- ECC 参照クロップ: grid_ref_crops.tif（backsub済み float32）----
    grid_ref_f = tifffile.imread(str(GRID_REF_CROPS_TIF)).astype(np.float64)
    ref_crops_u8 = [to_uint8(grid_ref_f[ch], ECC_VMIN, ECC_VMAX) for ch in range(n_ch)]
    print(f"grid_ref_crops loaded: shape={grid_ref_f.shape}")
    cfg_dummy = {}

    # tx_raw: image-X shift [px]  shape: (n_ch, n_tp)
    tx_raw    = np.full((n_ch, n_tp), np.nan)
    xtilt_rad = np.full((n_ch, n_tp), np.nan)

    for t_idx, path in enumerate(phase_paths[1:]):
        try:
            phase_tp = tifffile.imread(str(path)).astype(np.float32)
        except Exception:
            continue

        for c_idx, roi in enumerate(rois):
            # --- ECC X shift ---
            crop_ecc = extract_rect_roi(
                phase_tp, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"]
            ).astype(np.float32)
            offset = compute_backsub_offset(crop_ecc, cfg_dummy)
            result = ecc_align(
                ref_crops_u8[c_idx],
                to_uint8(crop_ecc + offset, ECC_VMIN, ECC_VMAX),
            )
            if result is not None:
                tx_raw[c_idx, t_idx] = result[0]   # image-X shift [px]

            # --- X-tilt (拡大 crop_h=270) ---
            xtilt_rad[c_idx, t_idx] = compute_x_tilt(
                phase_tp, roi["cy"], roi["cx"], roi["crop_w"]
            )

        if (t_idx + 1) % 20 == 0:
            print(f"  TP {t_idx+1}/{n_tp}", flush=True)

    tx_um      = tx_raw * PIXEL_SCALE_UM          # (n_ch, n_tp) [um]
    mean_tx_um = np.nanmean(tx_um, axis=0)        # (n_tp,) per-frame mean ECC X
    mean_xt    = np.nanmean(xtilt_rad, axis=0)    # (n_tp,) per-frame mean X-tilt

    # ---- 相関: per-frame (mean over channels) ----
    valid_m = ~(np.isnan(mean_tx_um) | np.isnan(mean_xt))
    r_mean, p_mean = pearsonr(mean_xt[valid_m], mean_tx_um[valid_m])
    print(f"\nPearson r (mean X-tilt vs mean ECC tx):"
          f" r={r_mean:.3f}  p={p_mean:.3e}  N={valid_m.sum()}")

    # ---- 相関: per-channel × per-frame (all points) ----
    valid_all = ~(np.isnan(tx_um) | np.isnan(xtilt_rad))
    r_all, p_all = pearsonr(xtilt_rad[valid_all], tx_um[valid_all])
    print(f"Pearson r (per-ch × per-frame):"
          f" r={r_all:.3f}  p={p_all:.3e}  N={valid_all.sum()}")

    # ---- 図 ----
    COLORS = plt.cm.tab20(np.linspace(0, 1, n_ch))
    fig = plt.figure(figsize=(14, 5))
    gs  = gridspec.GridSpec(1, 3, wspace=0.40, left=0.07, right=0.97,
                            top=0.86, bottom=0.13)

    # (A) scatter: mean X-tilt vs mean ECC tx
    ax0 = fig.add_subplot(gs[0])
    ax0.scatter(mean_xt[valid_m], mean_tx_um[valid_m],
                s=18, alpha=0.7, color="#2196F3", linewidths=0)
    c0 = np.polyfit(mean_xt[valid_m], mean_tx_um[valid_m], 1)
    xl = np.linspace(mean_xt[valid_m].min(), mean_xt[valid_m].max(), 100)
    ax0.plot(xl, np.polyval(c0, xl), "r-", lw=1.2)
    ax0.set_xlabel("Mean channel X-tilt slope (rad/px, left BG)")
    ax0.set_ylabel("Mean ECC X shift (um)")
    ax0.set_title(f"Mean X-tilt vs ECC X\nr={r_mean:.3f}, p={p_mean:.2e}", fontsize=10)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)

    # (B) 時系列比較
    ax1 = fig.add_subplot(gs[1])
    tp_axis = np.arange(n_tp)
    ax1.plot(tp_axis, mean_tx_um, color="#2196F3", lw=0.9, label="ECC mean X (um)")
    ax1b = ax1.twinx()
    ax1b.plot(tp_axis, mean_xt, color="#FF9800", lw=0.9, alpha=0.8,
              label="Mean X-tilt (rad)")
    ax1.set_xlabel("TP")
    ax1.set_ylabel("ECC mean X (um)", color="#2196F3")
    ax1b.set_ylabel("X-tilt (rad)", color="#FF9800")
    ax1.set_title("Time series (mean)", fontsize=10)
    ax1.tick_params(axis="y", labelcolor="#2196F3")
    ax1b.tick_params(axis="y", labelcolor="#FF9800")
    ax1.spines["top"].set_visible(False)

    # (C) per-channel × per-frame scatter（色 = channel）
    ax2 = fig.add_subplot(gs[2])
    for c_idx in range(n_ch):
        v = ~(np.isnan(tx_um[c_idx]) | np.isnan(xtilt_rad[c_idx]))
        if v.sum() < 2:
            continue
        ax2.scatter(xtilt_rad[c_idx, v], tx_um[c_idx, v],
                    s=4, alpha=0.4, color=COLORS[c_idx], linewidths=0,
                    label=f"ch{c_idx}")
    # 全データの回帰直線
    if valid_all.sum() > 1:
        c2 = np.polyfit(xtilt_rad[valid_all], tx_um[valid_all], 1)
        x2 = np.linspace(xtilt_rad[valid_all].min(), xtilt_rad[valid_all].max(), 100)
        ax2.plot(x2, np.polyval(c2, x2), "r-", lw=1.2)
    ax2.set_xlabel("Channel X-tilt slope (rad/px, left BG)")
    ax2.set_ylabel("Per-channel ECC X shift (um)")
    ax2.set_title(f"Per-channel × all TP\nr={r_all:.3f}, p={p_all:.2e}", fontsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        f"Wavefront X-tilt vs ECC X shift  "
        f"[crop_h_tilt={CH_CROP_H_TILT}px, polyfit left-{int(FIT_FRAC*100)}%]\n"
        f"Mean r={r_mean:.3f}  Per-channel r={r_all:.3f}",
        fontsize=10,
    )

    save_figure(
        fig,
        params={
            "ch_crop_h_tilt": CH_CROP_H_TILT,
            "fit_frac": FIT_FRAC,
            "n_channels": n_ch,
            "n_tp": n_tp,
            "pearson_r_mean": float(r_mean),
            "pearson_pval_mean": float(p_mean),
            "pearson_r_perch": float(r_all),
            "pearson_pval_perch": float(p_all),
            "n_valid_mean": int(valid_m.sum()),
            "n_valid_all": int(valid_all.sum()),
        },
        description=(
            f"Per-channel wavefront X-tilt slope (polyfit) vs ECC X shift correlation. "
            f"r(mean)={r_mean:.3f}, r(per-ch)={r_all:.3f}"
        ),
        data={
            "xtilt_rad": xtilt_rad,
            "tx_raw": tx_raw,
            "tx_um": tx_um,
            "mean_tx_um": mean_tx_um,
            "mean_xt": mean_xt,
        },
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
