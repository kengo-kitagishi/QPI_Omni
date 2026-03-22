"""
timelapse_plane_bgsub.py
------------------------
output_phase フレームに対して：
  1. pos_shifts.json のシフト量でサブピクセル補正
  2. 最近傍グリッド位置を選択して引き算
  3. 左 BG_MASK_FRAC の領域で 2D 線形平面フィット → 画像全体から引く
  4. 処理後フレームの per-pixel std map を出力（品質評価）

出力:
  TL_DIR/output_phase_planesub/img_*_ph_000_phase.tif  (float32)
  TL_DIR/output_phase_planesub/plane_bgsub_log.json
  results/figures/ に std map + 残差時系列（figure_logger 経由）
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
# 設定パラメータ
# ============================================================
TL_DIR    = Path(r"F:\timelapse_11day_exp200ms_1pos_EMM2\Pos1")
GRID_DIR  = Path(r"F:\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1")
BASE_LABEL = "Pos1"

TL_Z_INDEX   = 0   # img_*_ph_000_phase.tif
GRID_Z_INDEX = 5   # grid 参照の z インデックス

# 光学パラメータ
SENSOR_PIXEL_SIZE = 3.45e-6  # [m]
MAGNIFICATION     = 40
ORIGINAL_DIM      = 2048
RECONSTRUCTED_DIM = 511

# シフト符号（ECC 出力の符号規則: grid_subtract.py と同値）
SHIFT_SIGN_X = 1
SHIFT_SIGN_Y = 1

# グリッドステップ [μm]
X_STEP = 0.1
Y_STEP = 0.1

# 2D 平面フィット: 左半分の中央 BG_MASK_FRAC を背景領域として使用
# 左半分の両端 (1-BG_MASK_FRAC)/2 を除外（画面端アーティファクト対策）
BG_MASK_FRAC = 0.80

# std map 計算に使うフレーム数上限（None = 全フレーム）
STD_SUBSAMPLE = 200

# テスト用: None で全フレーム、整数で先頭 N フレームのみ
MAX_FRAMES = None
# ============================================================


def fit_plane_2d(img: np.ndarray, col_start: int, col_end: int):
    """
    img[:, col_start:col_end] に線形平面 z = a*x + b*y + c を最小二乗フィット。
    画像全体に評価した平面配列と係数 (a, b, c) を返す。

    Parameters
    ----------
    img : (H, W) float64
    col_start : int  フィットに使う列インデックスの左端（inclusive）
    col_end   : int  フィットに使う列インデックスの右端（exclusive）

    Returns
    -------
    plane : (H, W) float64  フル画像に評価した平面
    coeffs : (a, b, c)
    """
    H, W = img.shape
    # フィット領域の座標グリッド（xs=列方向, ys=行方向）
    ys, xs = np.mgrid[0:H, col_start:col_end]
    z = img[:, col_start:col_end].ravel()
    A = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    # フル画像に評価
    ys_f, xs_f = np.mgrid[0:H, 0:W]
    plane = a * xs_f + b * ys_f + c
    return plane, (float(a), float(b), float(c))


def main():
    # ---- 初期化 ----
    tl_dir  = TL_DIR
    out_dir = tl_dir / "output_phase_planesub"
    shifts_json = tl_dir / "output_phase" / "channels" / "pos_shifts.json"

    if not tl_dir.exists():
        print(f"ERROR: TL_DIR が見つかりません: {tl_dir}")
        sys.exit(1)
    if not shifts_json.exists():
        print(f"ERROR: pos_shifts.json が見つかりません: {shifts_json}")
        sys.exit(1)

    # ---- シフトデータ読み込み ----
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

    # ---- タイムラプスフレームリスト ----
    tl_frames = load_timelapse_frames(tl_dir, TL_Z_INDEX)
    if not tl_frames:
        print(f"ERROR: TL フレームが見つかりません")
        sys.exit(1)
    if len(tl_frames) < n_frames:
        print(f"WARNING: TIF ファイル数 {len(tl_frames)} < pos_shifts フレーム数 {n_frames}")
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

    # ---- フレームシフトリスト ----
    frame_shifts = []
    for r in frame_results:
        sx = r.get("shift_x_avg") or r.get("shift_x") or 0.0
        sy = r.get("shift_y_avg") or r.get("shift_y") or 0.0
        frame_shifts.append((sx, sy))

    # ---- 出力ディレクトリ ----
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"出力先: {out_dir}")

    # ---- グリッド画像キャッシュ ----
    grid_img_cache: dict = {}

    def get_grid_image(xi, yi):
        key = (xi, yi)
        if key not in grid_img_cache:
            pos_dir = pos_map[key]
            grid_img_cache[key] = load_grid_image(pos_dir, GRID_Z_INDEX)
        return grid_img_cache[key]

    # ---- BG マスク列幅 ----
    # 左半分の中央 BG_MASK_FRAC を使用（両端のアーティファクトを除外）
    sample_img = tifffile.imread(str(tl_frames[0]))
    H, W = sample_img.shape
    left_half = W // 2
    margin = int(left_half * (1 - BG_MASK_FRAC) / 2)
    mask_col_start = margin
    mask_col_end   = left_half - margin
    print(f"画像サイズ: {H}x{W}, BG マスク列: {mask_col_start}~{mask_col_end} "
          f"(左半分の中央 {BG_MASK_FRAC*100:.0f}%)")

    # ---- メインループ ----
    subtract_log = []

    for t in tqdm(range(n_frames), desc="grid subtract + plane bgsub"):
        sx, sy = frame_shifts[t]

        # 最近傍グリッド選択（grid_subtract.py L279-293 と同じ）
        dx_um = SHIFT_SIGN_X * sy * pixel_scale_um
        dy_um = SHIFT_SIGN_Y * sx * pixel_scale_um
        (xi, yi), dist_um = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)

        # 残差サブピクセルシフト
        residual_x = SHIFT_SIGN_Y * sx - yi * Y_STEP / pixel_scale_um
        residual_y = SHIFT_SIGN_X * sy - xi * X_STEP / pixel_scale_um

        # タイムラプスフレーム読み込み
        tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)

        # サブピクセル補正
        if residual_x != 0.0 or residual_y != 0.0:
            tl_warped = apply_inverse_shift_warp(tl_img, residual_x, residual_y)
        else:
            tl_warped = tl_img

        # 最近傍グリッド減算
        grid_img = get_grid_image(xi, yi)
        grid_sub = tl_warped - grid_img

        # 2D 平面フィット & 減算
        bg_mean_before = float(np.mean(grid_sub[:, mask_col_start:mask_col_end]))
        bg_std_before  = float(np.std(grid_sub[:, mask_col_start:mask_col_end]))
        plane, (a, b, c) = fit_plane_2d(grid_sub, mask_col_start, mask_col_end)
        result = grid_sub - plane
        bg_mean_after = float(np.mean(result[:, mask_col_start:mask_col_end]))
        bg_std_after  = float(np.std(result[:, mask_col_start:mask_col_end]))

        # 保存
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

    # ---- ログ保存 ----
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
    print(f"ログ保存: {log_path}")

    # ---- 品質評価: per-pixel std map ----
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
    ], axis=0)  # (N, H, W)

    std_map = np.std(stack, axis=0)  # (H, W)

    # 右 20% 残差時系列（フィットに使っていない独立検証領域）
    right_region_mean = stack[:, :, left_half:].mean(axis=(1, 2))  # (N,) 右半分全体
    frame_times = indices  # フレームインデックスで代用

    # ---- 可視化 ----
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

    # Panel 2: 右 20% 残差時系列
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

    print(f"\n--- 完了 ---")
    print(f"処理フレーム数: {n_frames}")
    print(f"出力先: {out_dir}")
    print(f"std map 中央値: {np.median(std_map):.4f} rad")
    print(f"std map 最大値: {np.max(std_map):.4f} rad")
    print(f"右20%領域 残差std: {np.std(right_region_mean):.4f} rad")


if __name__ == "__main__":
    main()
