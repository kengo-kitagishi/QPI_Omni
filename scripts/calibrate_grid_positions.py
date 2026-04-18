# %%
"""
calibrate_grid_positions.py
---------------------------
グリッド各点の実際のピクセルオフセットを BFS チェーン方式で実測し、
grid_calibration.json に保存する。

【方式】
  直接 grid(0,0) と ECC するのでなく、BFS 順に「既キャリブレーション済みの
  最近傍 4 隣接点」を基準として ECC → 変位を累積する。
  1 ステップの変位は ~0.29 px なので ECC が最も精度よく収束する。

  accumulated_dx(xi,yi) = ref_actual_dx + (-tx)
  accumulated_dy(xi,yi) = ref_actual_dy + (-ty)

  ここで (tx,ty) は ecc_align(ref_crop, cur_crop) → warp_matrix[0,2], [1,2]。
  ref[col,row] ≈ cur[col-tx, row-ty] より、cur 側の content は ref に対し (-tx,-ty) 移動。

【出力】
  GRID_DIR/grid_calibration.json
  → compute_pos_shifts.py / grid_subtract.py の GRID_CALIBRATION_JSON に指定して使う
"""
import numpy as np
import tifffile
import cv2
import json
import re
import sys
from collections import deque
from pathlib import Path
from tqdm import tqdm

# ============================================================
# 設定パラメータ
# ============================================================
GRID_DIR          = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
BASE_LABEL        = "Pos1"
GRID_Z_INDEX      = 10

CHANNEL_ROIS_JSON = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"

# ECC 正規化範囲（compute_pos_shifts.py の VMIN/VMAX と同値にする）
VMIN = -5.0
VMAX =  2.0

# ECC convergence params now in ecc_utils (100000, 1e-8)

# tilt 補正パラメータ（compute_pos_shifts.py と同値）
TILT_CROP_H = 270   # X 方向の big crop 幅 [px]
ECC_CROP_H  = 80    # ECC に使う中央 crop 幅 [px]

# 光学パラメータ（名目値との比較用のみ。find_nearest には使わない）
SENSOR_PIXEL_SIZE  = 3.45e-6   # [m]
MAGNIFICATION      = 40
ORIGINAL_DIM       = 2048
RECONSTRUCTED_DIM  = 511
X_STEP             = 0.1       # グリッドステップ [μm]
Y_STEP             = 0.1
SHIFT_SIGN_X       = 1
SHIFT_SIGN_Y       = 1
POS_SPLIT          = 33    # Pos < POS_SPLIT: 左1/3 fit, Pos >= POS_SPLIT: 右1/3 fit

# None → GRID_DIR/grid_calibration.json
OUTPUT_JSON = None
# ============================================================


from ecc_utils import (
    tilt_fit_crop, extract_rect_roi, to_uint8, ecc_align,
    remove_outliers_mad,
)


def scan_grid_positions(grid_dir, base_label):
    grid_dir = Path(grid_dir)
    pattern = re.compile(rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$")
    pos_map = {}
    for d in grid_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            pos_map[(int(m.group(1)), int(m.group(2)))] = d
    return pos_map


def load_grid_image(pos_dir, z_index):
    fname = f"img_000000000_ph_{z_index:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"グリッド画像が見つかりません: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def get_crops_u8(img_f64, rois, n_channels, fit_right: bool = False):
    """Return tilt-corrected uint8 crops for all channels (None if OOB)."""
    crops = []
    for ch in range(n_channels):
        tc = tilt_fit_crop(img_f64, rois[ch]["cy"], rois[ch]["cx"],
                           rois[ch]["crop_w"], ECC_CROP_H, TILT_CROP_H,
                           fit_right=fit_right)
        crops.append(to_uint8(tc, VMIN, VMAX) if tc is not None else None)
    return crops


def ecc_relative(ref_crops_u8, cur_crops_u8, n_channels):
    """
    ref_crops_u8 と cur_crops_u8 の間でチャネルごとに ECC を実行し、
    全チャネル平均の (actual_dx_px, actual_dy_px, mean_corr) を返す。
    全チャネル失敗時は None。
    """
    dx_list, dy_list, corr_list = [], [], []
    for ch in range(n_channels):
        res = ecc_align(ref_crops_u8[ch], cur_crops_u8[ch])
        if res is not None:
            tx, ty, corr = res
            dx_list.append(-tx)   # actual_dx = -tx
            dy_list.append(-ty)   # actual_dy = -ty
            corr_list.append(corr)
    if not dx_list:
        return None
    return float(np.mean(dx_list)), float(np.mean(dy_list)), float(np.mean(corr_list))


def main():
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")

    # Pos番号に基づいて fit_right を決定（compute_pos_shifts.py と同一ロジック）
    m_label = re.match(r"Pos(\d+)", BASE_LABEL)
    pos_num = int(m_label.group(1)) if m_label else 1
    fit_right = pos_num >= POS_SPLIT
    print(f"BASE_LABEL: {BASE_LABEL}  pos_num={pos_num}  fit_right={fit_right}")

    # ROI 読み込み
    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        print(f"ERROR: CHANNEL_ROIS_JSON が見つかりません: {rois_path}")
        sys.exit(1)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)
    print(f"チャネル数: {n_channels}")

    # グリッドスキャン
    pos_map = scan_grid_positions(GRID_DIR, BASE_LABEL)
    if not pos_map:
        print(f"ERROR: グリッドPosが見つかりません: {GRID_DIR}/{BASE_LABEL}_x*_y*")
        sys.exit(1)
    xi_vals = [k[0] for k in pos_map]
    yi_vals = [k[1] for k in pos_map]
    print(f"グリッドPos数: {len(pos_map)}")
    print(f"  xi 範囲: [{min(xi_vals)}, {max(xi_vals)}]  yi 範囲: [{min(yi_vals)}, {max(yi_vals)}]")

    # grid(0,0) の存在確認
    if (0, 0) not in pos_map:
        print("ERROR: grid(0,0) が見つかりません")
        sys.exit(1)

    # (0,0) 基準画像の crops を一度読み込む
    ref_img = load_grid_image(pos_map[(0, 0)], GRID_Z_INDEX)
    ref_crops = get_crops_u8(ref_img, rois, n_channels, fit_right=fit_right)
    print(f"基準画像 (0,0) 読み込み完了")

    # ---- 直接比較キャリブレーション ----
    # 各 (xi,yi) を (0,0) と直接 ECC して actual_dx/dy を求める
    results = {}
    n_failed = 0

    # (0,0) は基準なので先に登録
    results[(0, 0)] = {
        "xi": 0, "yi": 0,
        "actual_dx_px": 0.0,
        "actual_dy_px": 0.0,
        "nominal_dx_px": 0.0,
        "nominal_dy_px": 0.0,
        "error_dx_px": 0.0,
        "error_dy_px": 0.0,
        "ref_xi": None,
        "ref_yi": None,
        "n_channels_used": n_channels,
        "mean_correlation": 1.0,
        "failed": False,
    }

    print("\n直接比較キャリブレーション開始（各点を (0,0) と直接 ECC）...")
    other_positions = sorted((k, v) for k, v in pos_map.items() if k != (0, 0))

    for (xi, yi), pos_dir in tqdm(other_positions, desc="計測"):
        nominal_dx = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um
        nominal_dy = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um

        try:
            cur_img = load_grid_image(pos_dir, GRID_Z_INDEX)
            cur_crops = get_crops_u8(cur_img, rois, n_channels, fit_right=fit_right)
        except FileNotFoundError as e:
            print(f"\n  [{xi},{yi}] 画像なし → 名目値で代替: {e}")
            results[(xi, yi)] = {
                "xi": xi, "yi": yi,
                "actual_dx_px": nominal_dx, "actual_dy_px": nominal_dy,
                "nominal_dx_px": nominal_dx, "nominal_dy_px": nominal_dy,
                "error_dx_px": 0.0, "error_dy_px": 0.0,
                "ref_xi": 0, "ref_yi": 0,
                "n_channels_used": 0, "mean_correlation": None,
                "failed": True,
            }
            n_failed += 1
            continue

        res = ecc_relative(ref_crops, cur_crops, n_channels)

        if res is None:
            print(f"\n  [{xi},{yi}] 全チャネル ECC 失敗 → 名目値で代替")
            results[(xi, yi)] = {
                "xi": xi, "yi": yi,
                "actual_dx_px": nominal_dx, "actual_dy_px": nominal_dy,
                "nominal_dx_px": nominal_dx, "nominal_dy_px": nominal_dy,
                "error_dx_px": 0.0, "error_dy_px": 0.0,
                "ref_xi": 0, "ref_yi": 0,
                "n_channels_used": 0, "mean_correlation": None,
                "failed": True,
            }
            n_failed += 1
            continue

        actual_dx, actual_dy, corr = res
        results[(xi, yi)] = {
            "xi": xi, "yi": yi,
            "actual_dx_px": actual_dx,
            "actual_dy_px": actual_dy,
            "nominal_dx_px": nominal_dx,
            "nominal_dy_px": nominal_dy,
            "error_dx_px": actual_dx - nominal_dx,
            "error_dy_px": actual_dy - nominal_dy,
            "ref_xi": 0,
            "ref_yi": 0,
            "n_channels_used": n_channels,
            "mean_correlation": corr,
            "failed": False,
        }

    # ---- 統計 ----
    successful = [r for r in results.values()
                  if not r["failed"] and (r["xi"] != 0 or r["yi"] != 0)]
    print(f"\n計測成功: {len(results) - n_failed}/{len(pos_map)}")
    if successful:
        errors_dx = [r["error_dx_px"] for r in successful]
        errors_dy = [r["error_dy_px"] for r in successful]
        print(f"X方向誤差 (actual - nominal): "
              f"mean={np.mean(errors_dx):+.3f}px  std={np.std(errors_dx):.3f}px  "
              f"max_abs={np.max(np.abs(errors_dx)):.3f}px")
        print(f"Y方向誤差 (actual - nominal): "
              f"mean={np.mean(errors_dy):+.3f}px  std={np.std(errors_dy):.3f}px  "
              f"max_abs={np.max(np.abs(errors_dy)):.3f}px")
        corrs = [r["mean_correlation"] for r in successful if r["mean_correlation"] is not None]
        if corrs:
            print(f"ECC 相関係数: mean={np.mean(corrs):.4f}  min={np.min(corrs):.4f}")

    # ---- 経験的 pixel_scale 推定（yi=0 軸: actual_dy vs xi, xi=0 軸: actual_dx vs yi） ----
    # actual_dy_px[xi, yi=0] の xi への傾き ≈ X_STEP / pixel_scale → pixel_scale = X_STEP / slope
    pts_yi0 = [(r["xi"], r["actual_dy_px"]) for r in results.values()
               if r["yi"] == 0 and not r["failed"] and r["xi"] != 0]
    pts_xi0 = [(r["yi"], r["actual_dx_px"]) for r in results.values()
               if r["xi"] == 0 and not r["failed"] and r["yi"] != 0]
    if len(pts_yi0) >= 3:
        xs, dys = zip(*sorted(pts_yi0))
        slope_x = float(np.polyfit(xs, dys, 1)[0])
        psc_est_x = X_STEP / abs(slope_x) if slope_x != 0 else float("nan")
        print(f"経験的 pixel_scale（xi軸/actual_dy）: {psc_est_x:.4f} μm/px  "
              f"(理論値: {pixel_scale_um:.4f}, 比: {psc_est_x/pixel_scale_um:.3f})")
    if len(pts_xi0) >= 3:
        ys, dxs = zip(*sorted(pts_xi0))
        slope_y = float(np.polyfit(ys, dxs, 1)[0])
        psc_est_y = Y_STEP / abs(slope_y) if slope_y != 0 else float("nan")
        print(f"経験的 pixel_scale（yi軸/actual_dx）: {psc_est_y:.4f} μm/px  "
              f"(理論値: {pixel_scale_um:.4f}, 比: {psc_est_y/pixel_scale_um:.3f})")

    # ---- 保存 ----
    out_path = Path(OUTPUT_JSON) if OUTPUT_JSON else Path(GRID_DIR) / f"grid_calibration_{BASE_LABEL}.json"
    positions_list = [results[k] for k in sorted(results.keys())]
    out_data = {
        "grid_dir": str(GRID_DIR),
        "base_label": BASE_LABEL,
        "grid_z_index": GRID_Z_INDEX,
        "channel_rois_json": str(rois_path),
        "pixel_scale_um": pixel_scale_um,
        "n_channels": n_channels,
        "x_step_um": X_STEP,
        "y_step_um": Y_STEP,
        "shift_sign_x": SHIFT_SIGN_X,
        "shift_sign_y": SHIFT_SIGN_Y,
        "n_positions": len(results),
        "n_failed": n_failed,
        "positions": positions_list,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
    print(f"\n保存完了: {out_path}")


if __name__ == "__main__":
    main()

# %%
