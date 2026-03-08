# %%
"""
grid_subtract.py
----------------
pos_shifts.json のフレームごと平均シフト量を使って、
generate_grid_pos.py で取得したグリッド画像の中から
最もシフト差分に近いXYオフセットのものを選び、
チャネルROIでcropしてsubtractする。

出力: CHANNELS_DIR/grid_subtracted/channel_{ch:02d}_grid_sub.tif (T,H,W)
      CHANNELS_DIR/grid_subtract_log.json
"""
import numpy as np
import tifffile
import json
import re
import sys
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from compute_pos_shifts import compute_backsub_offset

# ============================================================
# 設定パラメータ
# ============================================================
CHANNELS_DIR = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels"
CHANNEL_PATTERN = "channel_*.tif"       # backsub済みなら "channel_*_bg_corr.tif"
SHIFTS_JSON = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels\pos_shifts.json"
CHANNEL_ROIS_JSON = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels\channel_rois.json"

# グリッド画像ディレクトリ（generate_grid_pos.py で取得したデータ）
GRID_DIR = r"E:\Acuisition\kitagishi\260301\multipos_test_1"
BASE_LABEL = "Pos4"               # グリッドPosのベースラベル → Pos4_x{xi:+d}_y{yi:+d}
Z_INDEX = 5                       # 使用するz番号 → img_000000000_ph_{Z_INDEX:03d}.tif

# グリッドステップ [μm]（generate_grid_pos.py の X_STEP / Y_STEP と合わせる）
X_STEP = 0.1
Y_STEP = 0.1

# 座標変換（shift_visualize.py と同値）
SENSOR_PIXEL_SIZE = 3.45e-6  # [m]
MAGNIFICATION = 40
ORIGINAL_DIM = 2048
RECONSTRUCTED_DIM = 511

# シフト符号（実データで確認後に変更。1 or -1）
# shift_x_avg が正のとき sample は +x 方向に動いたとみなす場合は 1
SHIFT_SIGN_X = 1
SHIFT_SIGN_Y = 1

# 逆シフト適用（subtracted をシフト補正して元の位置に戻すか）
APPLY_INVERSE_SHIFT = False
# グリッド画像に backsub オフセットを適用するか
APPLY_BACKSUB_TO_GRID = True
# サブピクセル残差 warp をグリッド画像に適用するか
APPLY_SUBPIXEL_CORRECTION = True
# ============================================================


def extract_rect_roi(img, cy, cx, crop_w, crop_h):
    """channel_crop.py と同じROI crop ロジック"""
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


def scan_grid_positions(grid_dir, base_label):
    """
    {grid_dir}/{base_label}_x{xi:+d}_y{yi:+d} フォルダを全列挙し、
    (xi, yi) → folder_path の辞書を返す。
    """
    grid_dir = Path(grid_dir)
    pattern = re.compile(
        rf"^{re.escape(base_label)}_x([+-]?\d+)_y([+-]?\d+)$"
    )
    pos_map = {}
    for d in grid_dir.iterdir():
        if not d.is_dir():
            continue
        m = pattern.match(d.name)
        if m:
            xi = int(m.group(1))
            yi = int(m.group(2))
            pos_map[(xi, yi)] = d
    return pos_map


def find_nearest_grid(pos_map, dx_um, dy_um, x_step, y_step):
    """
    (dx_um, dy_um) に最も近い (xi, yi) を返す。
    距離: sqrt((xi*x_step - dx_um)^2 + (yi*y_step - dy_um)^2)
    """
    best_key = None
    best_dist = float('inf')
    for (xi, yi) in pos_map:
        dist = ((xi * x_step - dx_um) ** 2 + (yi * y_step - dy_um) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = (xi, yi)
    return best_key, best_dist


def load_grid_image(pos_dir, z_index):
    """グリッドPosフォルダから再構成済み位相画像を読み込む。"""
    fname = f"img_000000000_ph_{z_index:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"グリッド画像が見つかりません: {path}")
    return tifffile.imread(str(path)).astype(np.float64)


def apply_inverse_shift_warp(img, shift_x, shift_y):
    """ECC シフト (shift_x, shift_y) の逆変換を適用して元位置に戻す。"""
    import cv2
    h, w = img.shape
    warp_matrix = np.array([
        [1.0, 0.0, -shift_x],
        [0.0, 1.0, -shift_y]
    ], dtype=np.float32)
    return cv2.warpAffine(
        img.astype(np.float32), warp_matrix, (w, h),
        flags=cv2.INTER_LINEAR
    ).astype(np.float64)


def main():
    # --- 入力確認 ---
    channels_dir = Path(CHANNELS_DIR)
    shifts_json = Path(SHIFTS_JSON)
    rois_json = Path(CHANNEL_ROIS_JSON)

    for p, name in [(channels_dir, "CHANNELS_DIR"), (shifts_json, "SHIFTS_JSON"), (rois_json, "CHANNEL_ROIS_JSON")]:
        if not Path(p).exists():
            print(f"ERROR: {name} が見つかりません: {p}")
            sys.exit(1)

    # --- 読み込み ---
    with open(shifts_json, encoding="utf-8") as f:
        shifts_data = json.load(f)
    with open(rois_json, encoding="utf-8") as f:
        rois = json.load(f)

    frame_results = shifts_data.get("frame_results") or shifts_data.get("alignment_results")
    if not frame_results:
        print("ERROR: pos_shifts.json に frame_results が見つかりません")
        sys.exit(1)

    n_frames = len(frame_results)
    print(f"フレーム数: {n_frames}")
    print(f"チャネルROI数: {len(rois)}")

    # pixel → μm スケール
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")

    # --- グリッドPosスキャン ---
    pos_map = scan_grid_positions(GRID_DIR, BASE_LABEL)
    if not pos_map:
        print(f"ERROR: グリッドPosが見つかりません: {GRID_DIR}/{BASE_LABEL}_x*_y*")
        sys.exit(1)
    print(f"グリッドPos数: {len(pos_map)}")
    xi_vals = [k[0] for k in pos_map]
    yi_vals = [k[1] for k in pos_map]
    print(f"  x範囲: [{min(xi_vals)}, {max(xi_vals)}], y範囲: [{min(yi_vals)}, {max(yi_vals)}]")

    # --- チャネルスタック読み込み ---
    stacks_paths = sorted(channels_dir.glob(CHANNEL_PATTERN))
    if not stacks_paths:
        print(f"ERROR: {CHANNEL_PATTERN} に合うファイルが見つかりません")
        sys.exit(1)
    if len(stacks_paths) != len(rois):
        print(f"WARNING: チャネルスタック数({len(stacks_paths)}) != ROI数({len(rois)})")

    stacks = []
    for p in stacks_paths:
        arr = tifffile.imread(str(p))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        stacks.append(arr.astype(np.float64))
    n_channels = len(stacks)
    print(f"チャネル数: {n_channels}")

    # --- 出力ディレクトリ ---
    out_dir = channels_dir / "grid_subtracted"
    out_dir.mkdir(exist_ok=True)

    # --- グリッド画像キャッシュ（同じPos画像を毎回読まないよう）---
    grid_img_cache = {}

    def get_grid_image(xi, yi):
        key = (xi, yi)
        if key not in grid_img_cache:
            pos_dir = pos_map[key]
            grid_img_cache[key] = load_grid_image(pos_dir, Z_INDEX)
        return grid_img_cache[key]

    # --- フレームごとにシフト取得しておく ---
    frame_shifts = []
    for r in frame_results:
        sx = r.get("shift_x_avg") or r.get("shift_x")
        sy = r.get("shift_y_avg") or r.get("shift_y")
        frame_shifts.append((sx, sy))

    # --- チャネルごとにsubtracted stackを作成 ---
    subtract_log = []
    out_stacks = [[] for _ in range(n_channels)]

    for t in tqdm(range(n_frames), desc="フレーム処理"):
        sx, sy = frame_shifts[t]
        if sx is None or sy is None:
            sx, sy = 0.0, 0.0

        # μm変換
        dx_um = SHIFT_SIGN_X * sx * pixel_scale_um
        dy_um = SHIFT_SIGN_Y * sy * pixel_scale_um

        # 最近傍グリッドPos
        (xi, yi), dist_um = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)
        pos_label = f"{BASE_LABEL}_x{xi:+d}_y{yi:+d}"

        log_entry = {
            "frame_index": t,
            "shift_x_avg_px": sx,
            "shift_y_avg_px": sy,
            "dx_um": dx_um,
            "dy_um": dy_um,
            "grid_xi": xi,
            "grid_yi": yi,
            "grid_pos_label": pos_label,
            "grid_nearest_dist_um": dist_um,
            "is_outlier_timeseries": frame_results[t].get("is_outlier_timeseries", False)
        }
        subtract_log.append(log_entry)

        # グリッド画像取得
        try:
            grid_img = get_grid_image(xi, yi)
        except FileNotFoundError as e:
            print(f"\n  [t={t}] {e}  → ゼロ画像を使用")
            grid_img = None

        # 各チャネルに対してsubtract
        for ch in range(n_channels):
            frame_data = stacks[ch][t]
            roi = rois[ch] if ch < len(rois) else rois[-1]
            cy, cx = roi["cy"], roi["cx"]
            crop_w, crop_h = roi["crop_w"], roi["crop_h"]

            if grid_img is not None:
                grid_cropped = extract_rect_roi(grid_img, cy, cx, crop_w, crop_h)

                # backsub オフセット補正
                if APPLY_BACKSUB_TO_GRID:
                    offset = compute_backsub_offset(grid_cropped)
                    grid_cropped = grid_cropped + offset

                # サブピクセル残差 warp
                if APPLY_SUBPIXEL_CORRECTION and (sx != 0.0 or sy != 0.0):
                    pixel_scale_um_local = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
                    residual_x_px = SHIFT_SIGN_X * sx - xi * X_STEP / pixel_scale_um_local
                    residual_y_px = SHIFT_SIGN_Y * sy - yi * Y_STEP / pixel_scale_um_local
                    grid_cropped = apply_inverse_shift_warp(grid_cropped, residual_x_px, residual_y_px)

                # サイズ不一致はリサイズで吸収
                if grid_cropped.shape != frame_data.shape:
                    import cv2
                    grid_cropped = cv2.resize(
                        grid_cropped.astype(np.float32),
                        (frame_data.shape[1], frame_data.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    ).astype(np.float64)
                subtracted = frame_data - grid_cropped
            else:
                subtracted = frame_data.copy()

            if APPLY_INVERSE_SHIFT and (sx != 0.0 or sy != 0.0):
                subtracted = apply_inverse_shift_warp(subtracted, sx, sy)

            out_stacks[ch].append(subtracted.astype(np.float32))

    # --- TIF保存 ---
    for ch in range(n_channels):
        arr = np.array(out_stacks[ch], dtype=np.float32)  # (T, H, W)
        out_path = out_dir / f"channel_{ch:02d}_grid_sub.tif"
        tifffile.imwrite(str(out_path), arr, imagej=True)
        print(f"保存: {out_path}  shape={arr.shape}")

    # --- ログ保存 ---
    log_path = channels_dir / "grid_subtract_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "channels_dir": str(channels_dir),
            "shifts_json": str(shifts_json),
            "grid_dir": str(GRID_DIR),
            "base_label": BASE_LABEL,
            "z_index": Z_INDEX,
            "x_step_um": X_STEP,
            "y_step_um": Y_STEP,
            "pixel_scale_um": pixel_scale_um,
            "shift_sign_x": SHIFT_SIGN_X,
            "shift_sign_y": SHIFT_SIGN_Y,
            "apply_inverse_shift": APPLY_INVERSE_SHIFT,
            "apply_backsub_to_grid": APPLY_BACKSUB_TO_GRID,
            "apply_subpixel_correction": APPLY_SUBPIXEL_CORRECTION,
            "frame_log": subtract_log
        }, f, indent=2, ensure_ascii=False)
    print(f"ログ保存: {log_path}")
    print("\n完了")


if __name__ == "__main__":
    main()

# %%
