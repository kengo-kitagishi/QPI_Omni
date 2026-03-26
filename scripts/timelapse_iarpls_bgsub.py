"""
timelapse_iarpls_bgsub.py
--------------------------
BG ROI クロップ版 pspline_iarpls 背景除去

処理フロー（フレームごと）:
  1. pos_shifts.json からシフト量を読み込む
  2. 最近傍グリッド選択
  3. TL フレーム読み込み（output_phase/img_*_ph_000_phase.tif）
  4. サブピクセル補正（apply_inverse_shift_warp）
  5. 最近傍グリッド画像を引く → grid_sub (511×511)
  6. BG ROI クロップ: extract_rect_roi → (40, 440)
  7. 左/右境界: left_boundary = W_full//2 - (BG_CX - BG_CROP_H//2) = 215
     （全画像 x=255 に対応するクロップ内列番号）
  8. 列平均: col_mean = crop.mean(axis=0) → (440,)
  9. 左側 col 0〜left_boundary に pspline_iarpls フィット → baseline_left (215,)
 10. 末端 N_EXTRAP_PTS 点で np.polyfit(deg=1) → 右側を線形外挿 → baseline_right (225,)
 11. baseline_full = concat([baseline_left, baseline_right]) (440,)
 12. result = crop - baseline_full[np.newaxis, :] → (40, 440)
 13. output_phase_iarpls/img_*_ph_000_phase.tif として保存 (float32)

出力:
  TL_DIR/output_phase_iarpls/img_*_ph_000_phase.tif  (float32, 40×440)
  TL_DIR/output_phase_iarpls/iarpls_bgsub_log.json
  results/figures/ に プロファイル可視化 + std map（figure_logger 経由）
"""
import sys
import json
import numpy as np
import tifffile
from pathlib import Path
from tqdm import tqdm
from pybaselines import Baseline

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
# 設定パラメータ
# ============================================================
TL_DIR     = Path(r"C:\ph\Pos1")
GRID_DIR   = Path(r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1")
BASE_LABEL = "Pos1"

TL_Z_INDEX   = 0   # img_*_ph_000_phase.tif
GRID_Z_INDEX = 0   # グリッド参照の z インデックス

# 光学パラメータ
SENSOR_PIXEL_SIZE = 3.45e-6
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511

# シフト符号（compute_shifts_bgroi.py と同値）
SHIFT_SIGN_X = 1
SHIFT_SIGN_Y = 1

# グリッドステップ [μm]
X_STEP = 0.1
Y_STEP = 0.1

# BG ROI（compute_shifts_bgroi.py と同値）
BG_CY      = 359   # y 中心
BG_CX      = 260   # x 中心
BG_CROP_W  = 40    # y 方向サイズ [px]
BG_CROP_H  = 440   # x 方向サイズ [px]

# iarpls
IARPLS_LAM       = 1e5
IARPLS_NUM_KNOTS = 20

# 外挿に使う末端点数（左側 baseline の右端 N_EXTRAP_PTS 点で線形フィット → 右側外挿）
N_EXTRAP_PTS = 20

# std map 計算に使うフレーム数上限（None = 全フレーム）
STD_SUBSAMPLE = 200

# プロファイル可視化で均等サンプリングするフレーム数
PROFILE_N_FRAMES = 12

# テスト用: None で全フレーム
MAX_FRAMES = None
# ============================================================


def extract_rect_roi(img, cy, cx, crop_w, crop_h):
    """compute_shifts_bgroi.py と同じ ROI crop ロジック"""
    h, w = img.shape
    y1 = cy - crop_w // 2; y2 = y1 + crop_w
    x1 = cx - crop_h // 2; x2 = x1 + crop_h
    pad_y0 = max(0, -y1); y1 = max(0, y1)
    pad_y1 = max(0, y2 - h); y2 = min(h, y2)
    pad_x0 = max(0, -x1); x1 = max(0, x1)
    pad_x1 = max(0, x2 - w); x2 = min(w, x2)
    crop = img[y1:y2, x1:x2]
    if any([pad_y0, pad_y1, pad_x0, pad_x1]):
        crop = np.pad(crop, ((pad_y0, pad_y1), (pad_x0, pad_x1)), mode="constant")
    return crop


def iarpls_bgsub_bgroi(crop: np.ndarray, left_boundary: int) -> tuple:
    """
    BG ROI crop (H_crop, W_crop) に対して:
      - 左側 col 0〜left_boundary の列平均に pspline_iarpls フィット
      - 右側は末端 N_EXTRAP_PTS 点から線形外挿
      - baseline (W_crop,) を全行から引く

    Parameters
    ----------
    crop         : (H_crop, W_crop) float64
    left_boundary: int  全画像中心に対応するクロップ内列番号

    Returns
    -------
    result        : (H_crop, W_crop) float64
    baseline_full : (W_crop,) float64
    col_mean      : (W_crop,) float64  列平均プロファイル（可視化用）
    """
    H, W = crop.shape
    col_mean = crop.mean(axis=0)  # (W,)

    # 左側フィット
    x_left = np.arange(left_boundary)
    bl_fitter = Baseline(x_data=x_left)
    baseline_left, _ = bl_fitter.pspline_iarpls(
        col_mean[:left_boundary], lam=IARPLS_LAM, num_knots=IARPLS_NUM_KNOTS
    )  # (left_boundary,)

    # 右側: 末端 N_EXTRAP_PTS 点から線形外挿
    n_ext = min(N_EXTRAP_PTS, left_boundary)
    x_tail = x_left[-n_ext:]
    y_tail = baseline_left[-n_ext:]
    slope, intercept = np.polyfit(x_tail, y_tail, 1)
    x_right = np.arange(left_boundary, W)
    baseline_right = slope * x_right + intercept  # (W - left_boundary,)

    baseline_full = np.concatenate([baseline_left, baseline_right])  # (W,)
    result = crop - baseline_full[np.newaxis, :]

    return result, baseline_full, col_mean


def main():
    import matplotlib.pyplot as plt

    # ---- 初期化 ----
    tl_dir  = TL_DIR
    out_dir = tl_dir / "output_phase_iarpls"
    shifts_json = tl_dir / "output_phase" / "channels" / "pos_shifts.json"

    if not tl_dir.exists():
        print(f"ERROR: TL_DIR が見つかりません: {tl_dir}")
        sys.exit(1)
    if not shifts_json.exists():
        print(f"ERROR: pos_shifts.json が見つかりません: {shifts_json}")
        sys.exit(1)

    # ---- シフトデータ ----
    with open(shifts_json, encoding="utf-8") as f:
        shifts_data = json.load(f)
    frame_results = shifts_data.get("frame_results") or shifts_data.get("alignment_results")
    if not frame_results:
        print("ERROR: pos_shifts.json に frame_results が見つかりません")
        sys.exit(1)

    n_frames = len(frame_results)
    if MAX_FRAMES is not None and MAX_FRAMES < n_frames:
        frame_results = frame_results[:MAX_FRAMES]
        n_frames = MAX_FRAMES
        print(f"[TEST] フレームを {n_frames} に制限")
    print(f"フレーム数: {n_frames}")

    # ---- TL フレームリスト ----
    tl_frames = load_timelapse_frames(tl_dir, TL_Z_INDEX)
    if not tl_frames:
        print("ERROR: TL フレームが見つかりません")
        sys.exit(1)
    if len(tl_frames) < n_frames:
        print(f"WARNING: TIF {len(tl_frames)} < pos_shifts {n_frames}")
        n_frames = len(tl_frames)
        frame_results = frame_results[:n_frames]
    print(f"TIF ファイル数: {len(tl_frames)}")

    # ---- グリッドスキャン ----
    pos_map = scan_grid_positions(GRID_DIR, BASE_LABEL)
    if not pos_map:
        print(f"ERROR: グリッドPos が見つかりません: {GRID_DIR}/{BASE_LABEL}_x*_y*")
        sys.exit(1)
    print(f"グリッドPos数: {len(pos_map)}")

    # ---- pixel scale ----
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")

    # ---- BG ROI / left_boundary ----
    # 全画像中心 x = RECONSTRUCTED_DIM // 2 = 255
    # クロップ開始 x = BG_CX - BG_CROP_H // 2 = 260 - 220 = 40
    # left_boundary = 255 - 40 = 215
    x_crop_start  = BG_CX - BG_CROP_H // 2
    left_boundary = RECONSTRUCTED_DIM // 2 - x_crop_start
    print(f"BG ROI: cy={BG_CY}, cx={BG_CX}, crop=({BG_CROP_W}×{BG_CROP_H})")
    print(f"クロップ x range: [{x_crop_start}, {x_crop_start + BG_CROP_H})")
    print(f"全画像中心 x={RECONSTRUCTED_DIM // 2} → クロップ内列={left_boundary} "
          f"(左側フィット {left_boundary}px, 右側外挿 {BG_CROP_H - left_boundary}px)")

    # ---- フレームシフトリスト ----
    frame_shifts = []
    for r in frame_results:
        sx = r.get("shift_x_avg") or r.get("shift_x") or 0.0
        sy = r.get("shift_y_avg") or r.get("shift_y") or 0.0
        frame_shifts.append((sx, sy))

    # ---- 出力ディレクトリ ----
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力先: {out_dir}")

    # ---- グリッドキャッシュ ----
    grid_img_cache: dict = {}

    def get_grid_image(xi, yi):
        key = (xi, yi)
        if key not in grid_img_cache:
            grid_img_cache[key] = load_grid_image(pos_map[key], GRID_Z_INDEX)
        return grid_img_cache[key]

    # ---- プロファイル可視化用インデックス ----
    profile_indices = set(
        np.linspace(0, n_frames - 1, min(PROFILE_N_FRAMES, n_frames), dtype=int).tolist()
    )
    profile_data = []  # list of (frame_idx, col_mean, baseline_full, crop)

    # ---- メインループ ----
    subtract_log = []

    for t in tqdm(range(n_frames), desc="grid subtract + iarpls bgroi"):
        sx, sy = frame_shifts[t]

        dx_um = SHIFT_SIGN_X * sy * pixel_scale_um
        dy_um = SHIFT_SIGN_Y * sx * pixel_scale_um
        (xi, yi), dist_um = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)

        residual_x = SHIFT_SIGN_Y * sx - yi * Y_STEP / pixel_scale_um
        residual_y = SHIFT_SIGN_X * sy - xi * X_STEP / pixel_scale_um

        tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)

        if residual_x != 0.0 or residual_y != 0.0:
            tl_warped = apply_inverse_shift_warp(tl_img, residual_x, residual_y)
        else:
            tl_warped = tl_img

        grid_img  = get_grid_image(xi, yi)
        grid_sub  = tl_warped - grid_img  # (511, 511)

        # BG ROI クロップ
        crop = extract_rect_roi(grid_sub, BG_CY, BG_CX, BG_CROP_W, BG_CROP_H)  # (40, 440)

        # iarpls 背景除去
        result, baseline_full, col_mean = iarpls_bgsub_bgroi(crop, left_boundary)

        # プロファイル可視化用データ保存
        if t in profile_indices:
            profile_data.append((t, col_mean.copy(), baseline_full.copy(), crop.copy()))

        # 保存
        tifffile.imwrite(str(out_dir / tl_frames[t].name), result.astype(np.float32))

        subtract_log.append({
            "frame_index":        t,
            "shift_x_px":         sx,
            "shift_y_px":         sy,
            "dx_um":              dx_um,
            "dy_um":              dy_um,
            "grid_xi":            xi,
            "grid_yi":            yi,
            "grid_dist_um":       dist_um,
            "residual_x_px":      residual_x,
            "residual_y_px":      residual_y,
            "baseline_left_start": float(baseline_full[0]),
            "baseline_left_end":   float(baseline_full[left_boundary - 1]),
            "baseline_right_end":  float(baseline_full[-1]),
            "bg_mean_before":     float(np.mean(crop)),
            "bg_std_before":      float(np.std(crop)),
            "bg_mean_after":      float(np.mean(result)),
            "bg_std_after":       float(np.std(result)),
        })

    # ---- ログ保存 ----
    log_path = out_dir / "iarpls_bgsub_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "tl_dir":           str(tl_dir),
            "grid_dir":         str(GRID_DIR),
            "base_label":       BASE_LABEL,
            "tl_z_index":       TL_Z_INDEX,
            "grid_z_index":     GRID_Z_INDEX,
            "bg_roi": {
                "cy":             BG_CY,
                "cx":             BG_CX,
                "crop_w":         BG_CROP_W,
                "crop_h":         BG_CROP_H,
                "x_crop_start":   x_crop_start,
                "left_boundary":  left_boundary,
            },
            "iarpls_lam":       IARPLS_LAM,
            "iarpls_num_knots": IARPLS_NUM_KNOTS,
            "n_extrap_pts":     N_EXTRAP_PTS,
            "pixel_scale_um":   pixel_scale_um,
            "n_frames_processed": n_frames,
            "frame_log":        subtract_log,
        }, f, ensure_ascii=False, indent=2)
    print(f"ログ保存: {log_path}")

    # ====================================================
    # 図1: プロファイル可視化（N×3 パネル）
    # ====================================================
    print("プロファイル可視化を生成中...")
    n_prof = len(profile_data)
    fig1, axes1 = plt.subplots(n_prof, 3, figsize=(14, 2.5 * n_prof))
    if n_prof == 1:
        axes1 = axes1[np.newaxis, :]

    x_cols = np.arange(BG_CROP_H)

    for row, (t_idx, col_mean, baseline_full, crop) in enumerate(profile_data):
        residual_prof = col_mean - baseline_full

        # 左列: 列平均プロファイル + baseline
        ax = axes1[row, 0]
        ax.plot(x_cols, col_mean, lw=0.8, color="steelblue", label="col mean")
        ax.plot(x_cols, baseline_full, lw=1.2, color="red", label="iarpls baseline")
        ax.axvline(left_boundary, color="gray", lw=0.8, ls="--", alpha=0.7,
                   label=f"center (col {left_boundary})")
        if row == 0:
            ax.legend(fontsize=7)
        ax.set_ylabel(f"t={t_idx}\n[rad]", fontsize=8)
        ax.set_xlabel("col [px]", fontsize=7)
        ax.tick_params(labelsize=7)

        # 中列: 残差（profile - baseline）
        ax = axes1[row, 1]
        ax.plot(x_cols, residual_prof, lw=0.8, color="darkorange")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.axvline(left_boundary, color="gray", lw=0.8, ls="--", alpha=0.7)
        ax.set_xlabel("col [px]", fontsize=7)
        ax.set_ylabel("residual [rad]", fontsize=8)
        ax.tick_params(labelsize=7)

        # 右列: クロップ画像
        ax = axes1[row, 2]
        vlo, vhi = np.percentile(crop, [2, 98])
        im = ax.imshow(crop, aspect="auto", cmap="RdBu_r", vmin=vlo, vmax=vhi)
        ax.axvline(left_boundary, color="yellow", lw=0.8, ls="--", alpha=0.8)
        plt.colorbar(im, ax=ax, label="[rad]")
        ax.set_xlabel("col [px]", fontsize=7)
        ax.set_ylabel("row [px]", fontsize=8)
        ax.tick_params(labelsize=7)

    fig1.suptitle(
        f"BG ROI iarpls profile  "
        f"(lam={IARPLS_LAM:.0e}, knots={IARPLS_NUM_KNOTS}, extrap={N_EXTRAP_PTS}pt)  "
        f"| gray dashed = image center col {left_boundary}",
        fontsize=10,
    )
    fig1.tight_layout()

    save_figure(
        fig1,
        params={
            "iarpls_lam":       IARPLS_LAM,
            "iarpls_num_knots": IARPLS_NUM_KNOTS,
            "n_extrap_pts":     N_EXTRAP_PTS,
            "left_boundary":    left_boundary,
            "n_profile_frames": n_prof,
            "bg_cy":            BG_CY,
            "bg_cx":            BG_CX,
            "bg_crop_w":        BG_CROP_W,
            "bg_crop_h":        BG_CROP_H,
        },
        description="BG ROI iarpls profile visualization: col mean + baseline + residual per sampled frame",
    )
    plt.close(fig1)

    # ====================================================
    # 図2: 品質評価（std map + crop 全体 mean 時系列）
    # ====================================================
    print("std map を計算中...")
    processed_files = sorted(out_dir.glob(f"img_*_ph_{TL_Z_INDEX:03d}_phase.tif"))
    n_files = len(processed_files)

    if STD_SUBSAMPLE is not None and n_files > STD_SUBSAMPLE:
        indices = np.linspace(0, n_files - 1, STD_SUBSAMPLE, dtype=int)
    else:
        indices = np.arange(n_files)

    stack = np.stack([
        tifffile.imread(str(processed_files[i])).astype(np.float32)
        for i in tqdm(indices, desc="std map 読み込み")
    ], axis=0)  # (N, 40, 440)

    std_map      = np.std(stack, axis=0)        # (40, 440)
    crop_mean_ts = stack.mean(axis=(1, 2))       # (N,)

    fig2, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4))

    # 左: per-pixel std map（クロップ座標系）
    im = ax_l.imshow(std_map, aspect="auto", cmap="hot", origin="upper")
    ax_l.axvline(left_boundary, color="lime", lw=1, ls="--", alpha=0.8,
                 label=f"image center (col {left_boundary})")
    plt.colorbar(im, ax=ax_l, label="std [rad]")
    ax_l.set_title(f"Per-pixel std  (N={len(indices)} frames)")
    ax_l.set_xlabel("col [px]")
    ax_l.set_ylabel("row [px]")
    ax_l.legend(fontsize=8)

    # 右: クロップ全体 mean 時系列（全体が BG なので 0 に収束するはず）
    ax_r.plot(indices, crop_mean_ts, lw=0.8, color="steelblue")
    ax_r.axhline(0, color="k", lw=0.5, ls="--")
    ax_r.set_xlabel("Frame index")
    ax_r.set_ylabel("Mean phase [rad]")
    ax_r.set_title("BG ROI crop mean (all background, should ≈ 0)")

    fig2.suptitle(
        f"BG ROI pspline_iarpls bgsub quality  "
        f"(lam={IARPLS_LAM:.0e}, knots={IARPLS_NUM_KNOTS})",
        fontsize=11,
    )
    fig2.tight_layout()

    save_figure(
        fig2,
        params={
            "iarpls_lam":          IARPLS_LAM,
            "iarpls_num_knots":    IARPLS_NUM_KNOTS,
            "n_extrap_pts":        N_EXTRAP_PTS,
            "n_frames_total":      n_files,
            "n_frames_std":        int(len(indices)),
            "std_map_median":      float(np.median(std_map)),
            "std_map_max":         float(np.max(std_map)),
            "crop_mean_ts_mean":   float(np.mean(crop_mean_ts)),
            "crop_mean_ts_std":    float(np.std(crop_mean_ts)),
        },
        description="BG ROI per-pixel std map and mean time series after pspline_iarpls bgsub",
    )
    plt.close(fig2)

    print(f"\n--- 完了 ---")
    print(f"処理フレーム数:    {n_frames}")
    print(f"出力先:            {out_dir}")
    print(f"出力サイズ:        ({BG_CROP_W}×{BG_CROP_H})")
    print(f"std map 中央値:    {np.median(std_map):.4f} rad")
    print(f"std map 最大値:    {np.max(std_map):.4f} rad")
    print(f"BG ROI mean:       {np.mean(crop_mean_ts):.4f} rad")
    print(f"BG ROI mean std:   {np.std(crop_mean_ts):.4f} rad")


if __name__ == "__main__":
    main()
