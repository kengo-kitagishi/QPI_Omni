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
GRID_DIR          = r"D:\AquisitionData\Kitagishi\260310\grid_0p5_0p5_0p1_exp200ms_1pos_EMM2_1"
BASE_LABEL        = "Pos1"
GRID_Z_INDEX      = 5

CHANNEL_ROIS_JSON = r"D:\AquisitionData\Kitagishi\260310\timelapse_11day_exp200ms_1pos_EMM2\Pos1\output_phase\channels\channel_rois.json"

# ECC 正規化範囲（compute_pos_shifts.py の VMIN/VMAX と同値にする）
VMIN = -5.0
VMAX =  2.0

# ECC 収束パラメータ
ECC_MAX_ITER = 10000
ECC_EPSILON  = 1e-8

# 光学パラメータ（名目値との比較用のみ。find_nearest には使わない）
SENSOR_PIXEL_SIZE  = 3.45e-6   # [m]
MAGNIFICATION      = 40
ORIGINAL_DIM       = 2048
RECONSTRUCTED_DIM  = 511
X_STEP             = 0.1       # グリッドステップ [μm]
Y_STEP             = 0.1
SHIFT_SIGN_X       = 1
SHIFT_SIGN_Y       = 1

# None → GRID_DIR/grid_calibration.json
OUTPUT_JSON = None
# ============================================================


def to_uint8(img, vmin=VMIN, vmax=VMAX):
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    """
    ECC アライメント。(tx, ty, correlation) を返す。失敗時は None。
    findTransformECC(ref, tl) → warp_matrix s.t. ref ≈ warpAffine(tl, W)
    W = [[1,0,tx],[0,1,ty]] → ref[col,row] ≈ tl[col-tx, row-ty]
    content の変位: tl 内での位置 = ref 内での位置 - (tx, ty)
                    actual_dx = -tx, actual_dy = -ty
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_MAX_ITER, ECC_EPSILON)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


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


def extract_rect_roi(img, cy, cx, crop_w, crop_h):
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


def get_crops_u8(img, rois, n_channels):
    """1枚の画像から全チャネルの ROI crop (uint8) を返す。"""
    return [to_uint8(extract_rect_roi(img, rois[ch]["cy"], rois[ch]["cx"],
                                      rois[ch]["crop_w"], rois[ch]["crop_h"]))
            for ch in range(n_channels)]


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

    # グリッド画像キャッシュ（読み込み済み crops を保持）
    crops_cache = {}   # (xi,yi) → list of uint8 crops

    def get_or_load_crops(xi, yi):
        if (xi, yi) not in crops_cache:
            img = load_grid_image(pos_map[(xi, yi)], GRID_Z_INDEX)
            crops_cache[(xi, yi)] = get_crops_u8(img, rois, n_channels)
        return crops_cache[(xi, yi)]

    # ---- BFS チェーン キャリブレーション ----
    # calibrated: (xi,yi) → (actual_dx_px, actual_dy_px)
    calibrated = {(0, 0): (0.0, 0.0)}

    # BFS: (0,0) から Manhattan 距離の昇順に処理
    # visited = キューに追加済みか
    visited = {(0, 0)}
    queue = deque()

    def enqueue_neighbors(xi, yi):
        for dxi, dyi in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (xi + dxi, yi + dyi)
            if nb in pos_map and nb not in visited:
                queue.append(nb)
                visited.add(nb)

    enqueue_neighbors(0, 0)

    results = {}   # (xi,yi) → result dict（最後に list 化して保存）
    n_failed = 0

    print("\nBFS チェーン キャリブレーション開始...")
    pbar = tqdm(total=len(pos_map), desc="計測")

    # grid(0,0) は基準なので先に登録
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
    pbar.update(1)

    while queue:
        xi, yi = queue.popleft()

        # ---- 最近傍の既キャリブレーション済み 4-隣接点を探す ----
        # BFS 順なので必ず 1 つ以上ある。複数ある場合は全部使って平均する。
        calibrated_neighbors = []
        for dxi, dyi in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nb = (xi + dxi, yi + dyi)
            if nb in calibrated:
                calibrated_neighbors.append(nb)

        if not calibrated_neighbors:
            # BFS 上あり得ないが念のため
            print(f"\n  [{xi},{yi}] キャリブレーション済み隣接点なし → スキップ")
            n_failed += 1
            pbar.update(1)
            enqueue_neighbors(xi, yi)
            continue

        # ---- 対象画像の crops を取得 ----
        try:
            cur_crops = get_or_load_crops(xi, yi)
        except FileNotFoundError as e:
            print(f"\n  [{xi},{yi}] 画像なし → 名目値で代替: {e}")
            nominal_dx = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um
            nominal_dy = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um
            # 隣接点の平均から補間（名目ステップを加算）
            ref_xi, ref_yi = calibrated_neighbors[0]
            ref_dx, ref_dy = calibrated[calibrated_neighbors[0]]
            # 1 step 分の名目変位を加算
            step_dx = SHIFT_SIGN_Y * (yi - ref_yi) * Y_STEP / pixel_scale_um
            step_dy = SHIFT_SIGN_X * (xi - ref_xi) * X_STEP / pixel_scale_um
            actual_dx = ref_dx + step_dx
            actual_dy = ref_dy + step_dy
            results[(xi, yi)] = {
                "xi": xi, "yi": yi,
                "actual_dx_px": actual_dx,
                "actual_dy_px": actual_dy,
                "nominal_dx_px": nominal_dx,
                "nominal_dy_px": nominal_dy,
                "error_dx_px": actual_dx - nominal_dx,
                "error_dy_px": actual_dy - nominal_dy,
                "ref_xi": ref_xi, "ref_yi": ref_yi,
                "n_channels_used": 0,
                "mean_correlation": None,
                "failed": True,
            }
            calibrated[(xi, yi)] = (actual_dx, actual_dy)
            n_failed += 1
            pbar.update(1)
            enqueue_neighbors(xi, yi)
            continue

        # ---- 全キャリブレーション済み隣接点との ECC を実行して平均 ----
        dx_estimates, dy_estimates, corr_estimates = [], [], []
        best_ref = None

        for ref_nb in calibrated_neighbors:
            ref_xi, ref_yi = ref_nb
            ref_actual_dx, ref_actual_dy = calibrated[ref_nb]
            try:
                ref_crops = get_or_load_crops(ref_xi, ref_yi)
            except FileNotFoundError:
                continue

            res = ecc_relative(ref_crops, cur_crops, n_channels)
            if res is not None:
                rel_dx, rel_dy, corr = res
                dx_estimates.append(ref_actual_dx + rel_dx)
                dy_estimates.append(ref_actual_dy + rel_dy)
                corr_estimates.append(corr)
                if best_ref is None:
                    best_ref = ref_nb

        nominal_dx = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um
        nominal_dy = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um

        if not dx_estimates:
            # 全隣接点との ECC が失敗 → 名目値で代替
            print(f"\n  [{xi},{yi}] 全隣接ECC失敗 → 名目値で代替")
            ref_xi, ref_yi = calibrated_neighbors[0]
            ref_dx, ref_dy = calibrated[(ref_xi, ref_yi)]
            step_dx = SHIFT_SIGN_Y * (yi - ref_yi) * Y_STEP / pixel_scale_um
            step_dy = SHIFT_SIGN_X * (xi - ref_xi) * X_STEP / pixel_scale_um
            actual_dx = ref_dx + step_dx
            actual_dy = ref_dy + step_dy
            results[(xi, yi)] = {
                "xi": xi, "yi": yi,
                "actual_dx_px": actual_dx,
                "actual_dy_px": actual_dy,
                "nominal_dx_px": nominal_dx,
                "nominal_dy_px": nominal_dy,
                "error_dx_px": actual_dx - nominal_dx,
                "error_dy_px": actual_dy - nominal_dy,
                "ref_xi": ref_xi, "ref_yi": ref_yi,
                "n_channels_used": 0,
                "mean_correlation": None,
                "failed": True,
            }
            calibrated[(xi, yi)] = (actual_dx, actual_dy)
            n_failed += 1
            pbar.update(1)
            enqueue_neighbors(xi, yi)
            continue

        # 複数の推定値がある場合は平均
        actual_dx = float(np.mean(dx_estimates))
        actual_dy = float(np.mean(dy_estimates))
        calibrated[(xi, yi)] = (actual_dx, actual_dy)

        results[(xi, yi)] = {
            "xi": xi, "yi": yi,
            "actual_dx_px": actual_dx,
            "actual_dy_px": actual_dy,
            "nominal_dx_px": nominal_dx,
            "nominal_dy_px": nominal_dy,
            "error_dx_px": actual_dx - nominal_dx,
            "error_dy_px": actual_dy - nominal_dy,
            "ref_xi": best_ref[0], "ref_yi": best_ref[1],
            "n_calibrated_refs": len(dx_estimates),
            "n_channels_used": n_channels,
            "mean_correlation": float(np.mean(corr_estimates)),
            "failed": False,
        }
        pbar.update(1)
        enqueue_neighbors(xi, yi)

    pbar.close()

    # BFS で到達できなかったグリッド点（孤立島など）を名目値で補完
    for key in pos_map:
        if key not in results:
            xi, yi = key
            nominal_dx = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um
            nominal_dy = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um
            results[key] = {
                "xi": xi, "yi": yi,
                "actual_dx_px": nominal_dx,
                "actual_dy_px": nominal_dy,
                "nominal_dx_px": nominal_dx,
                "nominal_dy_px": nominal_dy,
                "error_dx_px": 0.0,
                "error_dy_px": 0.0,
                "ref_xi": None, "ref_yi": None,
                "n_channels_used": 0,
                "mean_correlation": None,
                "failed": True,
            }
            n_failed += 1

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

    # ---- 保存 ----
    out_path = Path(OUTPUT_JSON) if OUTPUT_JSON else Path(GRID_DIR) / f"grid_calibration_{BASE_LABEL}.json"
    # 保存順を (xi, yi) でソート
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
