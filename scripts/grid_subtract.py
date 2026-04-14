# %%
"""
grid_subtract.py
----------------
pos_shifts.json のフレームごと平均シフト量を使って、
generate_grid_pos.py で取得したグリッド画像の中から
最もシフト差分に近い XY オフセットのものを選び、
フル再構成フレームに -residual warp を適用して grid(m,n) に合わせ、
両方を (m,n)-シフトした crop 位置で crop → bgcorr → 引き算する。

出力: channels_dir/grid_subtracted/channel_{ch:02d}_grid_sub.tif (T,H,W)
      channels_dir/grid_subtract_log.json
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
# タイムラプス Pos ディレクトリ（output_phase/*_phase.tif を含む）
TIMELAPSE_DIR = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1"

# pos_shifts.json と channel_rois.json の場所
SHIFTS_JSON       = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1\output_phase\channels\pos_shifts_cal.json"
CHANNEL_ROIS_JSON = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1\output_phase\channels\channel_rois.json"

# グリッド画像ディレクトリ（generate_grid_pos.py で取得したデータ）
GRID_DIR   = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
BASE_LABEL = "Pos1"               # グリッドPosのベースラベル → Pos1_x{xi:+d}_y{yi:+d}

# タイムラプスの z インデックス（img_*_ph_{TL_Z_INDEX:03d}_phase.tif）
TL_Z_INDEX   = 0
# グリッドの z インデックス
GRID_Z_INDEX = 18

# グリッドステップ [μm]（generate_grid_pos.py の X_STEP / Y_STEP と合わせる）
X_STEP = 0.1
Y_STEP = 0.1

# 座標変換（shift_visualize.py と同値）
SENSOR_PIXEL_SIZE  = 3.45e-6  # [m]
MAGNIFICATION      = 40
ORIGINAL_DIM       = 2048
RECONSTRUCTED_DIM  = 511

# シフト符号（実データで確認済み: stage+X → image -Y, stage+Y → image -X）
SHIFT_SIGN_X = -1
SHIFT_SIGN_Y = -1

# サブピクセル残差 warp をタイムラプス画像に適用するか（デバッグ用フラグ）
APPLY_SUBPIXEL_CORRECTION = True

# subtracted を逆シフトして元の位置に戻すか（通常不要）
APPLY_INVERSE_SHIFT = False
MAX_FRAMES = None  # テストラン用: None で全フレーム、整数で先頭 N フレームのみ
PICK_FRAMES = None  # None → all, list → only these indices (e.g. [0, 192, 385])

# グリッドキャリブレーション（calibrate_grid_positions.py の出力 JSON）
# None → 名目値 (xi*X_STEP/pixel_scale_um) を使用
GRID_CALIBRATION_JSON = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1\grid_calibration_Pos1.json"

# 出力クロップ長（X方向）: None → channel_rois.json の crop_h をそのまま使用
OUTPUT_CROP_H = None

# 出力ディレクトリ: None → channels_dir/grid_subtracted/ に自動設定
OUTPUT_DIR = r"D:\AquisitionData\Kitagishi\260405\ph_260405\Pos1\output_phase\channels\crop_sub_rawraw_z018"

# True → crop前のフルフレーム（subpixel correction適用済み）を full_frame_grid_sub.tif として保存
OUTPUT_SAVE_FULL_FRAME = False

# Pre-reconstructed phase directory override (USE_RAW_PHASE=False only).
# None → default (TIMELAPSE_DIR/output_phase).
# Set to output_phase_raw/ path to use raw phase without BG subtraction.
TL_PHASE_DIR = None

# ============================================================
# raw-raw subtraction モード
# ============================================================
# True → ホログラム原版TIFからPos0参照なしでon-the-fly再構成し、raw同士を直接引き算する。
# その後 2π 補正（global integer offset）と tilt 補正を適用して出力する。
# ECC 計算（compute_pos_shifts.py）は output_phase/ を使い続けるため変更不要。
USE_RAW_PHASE    = True

# 再構成用センサークロップ (r1, r2, c1, c2)  pipeline_full.py の CROP_BEFORE / CROP_AFTER と合わせる
RAW_CROP         = (0, 2048, 400, 2448)

# 生ホログラムの z インデックス（通常 TL_Z_INDEX / GRID_Z_INDEX と同値）
RAW_TL_Z_INDEX   = 0
RAW_GRID_Z_INDEX = 18

# 2π補正 + tilt補正に使う大クロップ高さ（axis=1 方向の px 数）
TILT_CROP_H_RAW  = 270
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


def load_grid_calibration(json_path):
    """
    grid_calibration.json を読み込み、(xi, yi) → (cal_dx_px, cal_dy_px) の dict を返す。

    calibrate_grid_positions.py は actual_dx_px = -tx（コンテンツ変位）で保存するが、
    grid_subtract.py の shift_x_avg = +tx（ECC warp_matrix 生値）。
    符号を揃えるため、ロード時に符号反転する。
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    cal = {}
    for entry in data.get("positions", []):
        cal[(entry["xi"], entry["yi"])] = (
            -entry["actual_dx_px"],   # actual_dx = -tx → cal_dx = +tx (shift_x 規約に合わせる)
            -entry["actual_dy_px"],
        )
    return cal


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


def load_timelapse_frames(tl_dir, z_index):
    """
    output_phase/ から img_*_ph_{z_index:03d}_phase.tif をソート済みリストで返す。
    """
    phase_dir = Path(tl_dir) / "output_phase"
    pattern = f"img_*_ph_{z_index:03d}_phase.tif"
    frames = sorted(phase_dir.glob(pattern))
    return frames


def apply_inverse_shift_warp(img, shift_x, shift_y):
    """(shift_x, shift_y) の逆変換を適用して (-shift_x, -shift_y) 移動する。"""
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


# ============================================================
# raw-raw モード用ヘルパー（USE_RAW_PHASE=True 時のみ使用）
# ============================================================

def _make_qpi_params_raw(holo_path, crop):
    """生ホログラム1枚から QPIParameters を生成する（pipeline_full._make_qpi_params と同等）。"""
    from PIL import Image
    from qpi import QPIParameters
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE
    img = np.array(Image.open(str(holo_path)))
    rs, re_, cs, ce = crop
    cropped = img[rs:re_, cs:ce]
    return QPIParameters(
        wavelength=WAVELENGTH, NA=NA,
        img_shape=cropped.shape, pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )


def _reconstruct_raw(holo_path, qpi_params, crop):
    """Pos0 参照なしで位相（unwrap済み）を返す（pipeline_full._reconstruct と同等）。"""
    from PIL import Image
    from qpi import get_field
    from skimage.restoration import unwrap_phase
    img = np.array(Image.open(str(holo_path)))
    rs, re_, cs, ce = crop
    img = img[rs:re_, cs:ce]
    field = get_field(img, qpi_params)
    return unwrap_phase(np.angle(field))


def load_timelapse_holos(tl_dir, z_index):
    """タイムラプスの生ホログラム（output_phase/ でなく tl_dir 直下）のパスリストを返す。"""
    return sorted(Path(tl_dir).glob(f"img_*_ph_{z_index:03d}.tif"))


def load_grid_holo_path(pos_dir, z_index):
    """グリッドPosフォルダから生ホログラムのパスを返す。"""
    path = pos_dir / f"img_000000000_ph_{z_index:03d}.tif"
    if not path.exists():
        raise FileNotFoundError(f"グリッドホログラムが見つかりません: {path}")
    return path


def _raw_subtract_correct(sub_large, out_crop_h):
    """
    raw-raw 引き算後の大クロップ shape: (crop_w, TILT_CROP_H_RAW) に対して
    2π補正 → tilt補正 → center crop を適用して (crop_w, out_crop_h) を返す。

    axis=1 が horizontal 方向（~270 px）。
    left 1/3（axis=1 の 0..TILT_CROP_H_RAW//3）= 背景領域として利用。

    1. 2π correction : left 1/3 の平均から k = round(mean/2π) を求め k×2π を引く
    2. tilt correction: left 1/3 で 1次フィット → 線形勾配を全体から引く
    3. center crop   : 中央 out_crop_h px を切り出す
    """
    tch = TILT_CROP_H_RAW
    fit_n = max(1, tch // 3)

    # 1. 2π correction
    bg_mean = float(np.mean(sub_large[:, :fit_n]))
    k = int(round(bg_mean / (2.0 * np.pi)))
    if k != 0:
        sub_large = sub_large - k * 2.0 * np.pi

    # 2. tilt correction
    x = np.arange(tch, dtype=np.float64)
    prof = sub_large.mean(axis=0)           # (tch,) profile along axis=1
    a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    sub_large = sub_large - (a * x + b)[np.newaxis, :]

    # 3. center crop
    start = (tch - out_crop_h) // 2
    return sub_large[:, start : start + out_crop_h]


def main():
    # --- 入力確認 ---
    tl_dir    = Path(TIMELAPSE_DIR)
    shifts_json = Path(SHIFTS_JSON)
    rois_json   = Path(CHANNEL_ROIS_JSON)

    for p, name in [(tl_dir, "TIMELAPSE_DIR"), (shifts_json, "SHIFTS_JSON"), (rois_json, "CHANNEL_ROIS_JSON")]:
        if not Path(p).exists():
            print(f"ERROR: {name} が見つかりません: {p}")
            sys.exit(1)

    # channels_dir は rois_json の親ディレクトリ
    channels_dir = rois_json.parent

    # --- 読み込み ---
    with open(shifts_json, encoding="utf-8") as f:
        shifts_data = json.load(f)
    with open(rois_json, encoding="utf-8") as f:
        rois = json.load(f)

    frame_results = shifts_data.get("frame_results") or shifts_data.get("alignment_results")
    if not frame_results:
        print("ERROR: pos_shifts.json に frame_results が見つかりません")
        sys.exit(1)

    n_frames_total = len(frame_results)
    if MAX_FRAMES is not None and MAX_FRAMES < n_frames_total:
        frame_results = frame_results[:MAX_FRAMES]
        n_frames_total = MAX_FRAMES
        print(f"[TEST] MAX_FRAMES={MAX_FRAMES}")

    # PICK_FRAMES: process only selected indices
    if PICK_FRAMES is not None:
        pick_set = [i for i in PICK_FRAMES if i < n_frames_total]
        print(f"[PICK] {len(pick_set)} frames: {pick_set}")
    else:
        pick_set = list(range(n_frames_total))
    n_frames = len(pick_set)
    n_channels = len(rois)
    print(f"フレーム数: {n_frames}")
    print(f"チャネルROI数: {n_channels}")

    # pixel → μm スケール
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    print(f"Pixel scale: {pixel_scale_um:.4f} μm/px")

    # --- タイムラプスフレームリスト ---
    _use_prerecon = (TL_PHASE_DIR is not None)
    _qpi_params_raw = None
    if USE_RAW_PHASE:
        if _use_prerecon:
            # Pre-reconstructed raw phase: read tif instead of reconstructing
            phase_dir = Path(TL_PHASE_DIR)
            tl_frames = sorted(phase_dir.glob("img_*_phase.tif"))
            if not tl_frames:
                print(f"ERROR: pre-recon phase not found: {phase_dir}/img_*_phase.tif")
                sys.exit(1)
            print(f"[pre-recon mode] {len(tl_frames)} frames from {phase_dir}")
        else:
            tl_frames = load_timelapse_holos(tl_dir, RAW_TL_Z_INDEX)
            if not tl_frames:
                print(f"ERROR: 生ホログラムが見つかりません: {tl_dir}/img_*_ph_{RAW_TL_Z_INDEX:03d}.tif")
                sys.exit(1)
            _qpi_params_raw = _make_qpi_params_raw(tl_frames[0], RAW_CROP)
            print(f"[raw mode] QPIParams 作成完了  ホログラム: {tl_frames[0].name}")
    else:
        tl_frames = load_timelapse_frames(tl_dir, TL_Z_INDEX)
        if not tl_frames:
            print(f"ERROR: タイムラプスフレームが見つかりません: {tl_dir}/output_phase/img_*_ph_{TL_Z_INDEX:03d}_phase.tif")
            sys.exit(1)
    if PICK_FRAMES is not None:
        max_needed = max(pick_set) + 1
        if len(tl_frames) < max_needed:
            print(f"WARNING: tif files={len(tl_frames)} < max pick index={max_needed}")
    elif len(tl_frames) != n_frames:
        print(f"WARNING: フレーム数不一致  pos_shifts={n_frames}  tif files={len(tl_frames)}")

    # --- グリッドPosスキャン ---
    pos_map = scan_grid_positions(GRID_DIR, BASE_LABEL)
    if not pos_map:
        print(f"ERROR: グリッドPosが見つかりません: {GRID_DIR}/{BASE_LABEL}_x*_y*")
        sys.exit(1)
    print(f"グリッドPos数: {len(pos_map)}")
    xi_vals = [k[0] for k in pos_map]
    yi_vals = [k[1] for k in pos_map]
    print(f"  x範囲: [{min(xi_vals)}, {max(xi_vals)}], y範囲: [{min(yi_vals)}, {max(yi_vals)}]")

    # pre-recon mode: build _qpi_params_raw from grid hologram (for grid reconstruction)
    if USE_RAW_PHASE and _use_prerecon and _qpi_params_raw is None:
        grid_origin = pos_map.get((0, 0), next(iter(pos_map.values())))
        _grid_holo = load_grid_holo_path(grid_origin, RAW_GRID_Z_INDEX)
        _qpi_params_raw = _make_qpi_params_raw(_grid_holo, RAW_CROP)
        print(f"[pre-recon] QPIParams from grid hologram: {_grid_holo.name}")

    # --- グリッドキャリブレーション読み込み ---
    grid_cal = {}
    if GRID_CALIBRATION_JSON:
        cal_path = Path(GRID_CALIBRATION_JSON)
        if cal_path.exists():
            grid_cal = load_grid_calibration(str(cal_path))
            print(f"[calibration] {len(grid_cal)} 点の実計測オフセットを読み込み: {cal_path}")
        else:
            print(f"[calibration] JSON が見つかりません: {cal_path}  → 名目値を使用")

    # --- 出力ディレクトリ ---
    out_dir = Path(OUTPUT_DIR) if OUTPUT_DIR else channels_dir / "grid_subtracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- グリッド画像キャッシュ ---
    grid_img_cache = {}

    def get_grid_image(xi, yi):
        key = (xi, yi)
        if key not in grid_img_cache:
            pos_dir = pos_map[key]
            if USE_RAW_PHASE:
                holo_path = load_grid_holo_path(pos_dir, RAW_GRID_Z_INDEX)
                grid_img_cache[key] = _reconstruct_raw(holo_path, _qpi_params_raw, RAW_CROP)
            else:
                grid_img_cache[key] = load_grid_image(pos_dir, GRID_Z_INDEX)
        return grid_img_cache[key]

    # --- フレームごとにシフト取得 ---
    frame_shifts = []
    for r in frame_results:
        sx = r.get("shift_x_avg") or r.get("shift_x")
        sy = r.get("shift_y_avg") or r.get("shift_y")
        frame_shifts.append((sx, sy))

    # --- メインループ ---
    subtract_log = []
    out_stacks = [[] for _ in range(n_channels)]
    full_frame_stack = [] if OUTPUT_SAVE_FULL_FRAME else None

    for idx, t in enumerate(tqdm(pick_set, desc="フレーム処理")):
        sx, sy = frame_shifts[t]
        if sx is None or sy is None:
            sx, sy = 0.0, 0.0

        # ---- 最近傍グリッド点を選択 ----
        dx_um = None
        dy_um = None
        if grid_cal:
            # キャリブレーション済み: 実ピクセル変位と (sx, sy) の距離で比較
            best_key, best_dist = None, float('inf')
            for key, (adx, ady) in grid_cal.items():
                if key not in pos_map:
                    continue
                dist = ((adx - sx) ** 2 + (ady - sy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_key = key
            (xi, yi), dist_um = best_key, best_dist
        else:
            # 名目値: ステージ空間で比較
            # shift_y（画像Y） → xi（ステージX）、shift_x（画像X） → yi（ステージY）
            dx_um = SHIFT_SIGN_X * sy * pixel_scale_um  # shift_y → xi
            dy_um = SHIFT_SIGN_Y * sx * pixel_scale_um  # shift_x → yi
            (xi, yi), dist_um = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)
        pos_label = f"{BASE_LABEL}_x{xi:+d}_y{yi:+d}"

        # ---- grid(xi,yi) の content offset と残差 ----
        if grid_cal and (xi, yi) in grid_cal:
            cal_dx, cal_dy = grid_cal[(xi, yi)]
            residual_x = sx - cal_dx   # 画像X方向残差
            residual_y = sy - cal_dy   # 画像Y方向残差
        else:
            cal_dx = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um
            cal_dy = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um
            residual_x = SHIFT_SIGN_Y * sx - yi * Y_STEP / pixel_scale_um
            residual_y = SHIFT_SIGN_X * sy - xi * X_STEP / pixel_scale_um

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
            "residual_x_px": residual_x,
            "residual_y_px": residual_y,
            "is_outlier_timeseries": frame_results[t].get("is_outlier_timeseries", False)
        }
        subtract_log.append(log_entry)

        # タイムラプスフレームをフル読み込み
        if USE_RAW_PHASE:
            if _use_prerecon:
                tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)
            else:
                tl_img = _reconstruct_raw(tl_frames[t], _qpi_params_raw, RAW_CROP)
        else:
            tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)

        # -residual warp でフル画像を grid(m,n) に合わせる
        if APPLY_SUBPIXEL_CORRECTION and (residual_x != 0.0 or residual_y != 0.0):
            tl_warped = apply_inverse_shift_warp(tl_img, residual_x, residual_y)
        else:
            tl_warped = tl_img

        # grid(m,n) フル画像
        try:
            grid_img = get_grid_image(xi, yi)
        except FileNotFoundError as e:
            print(f"\n  [t={t}] {e}  → ゼロ画像を使用")
            grid_img = None

        if full_frame_stack is not None:
            if grid_img is not None:
                full_frame_stack.append((tl_warped - grid_img).astype(np.float32))
            else:
                full_frame_stack.append(tl_warped.astype(np.float32))

        # --- 各チャネル: (m,n)-シフト済み位置で crop → (bgcorr or raw-raw) → 引き算 ---
        for ch in range(n_channels):
            roi = rois[ch] if ch < len(rois) else rois[-1]
            cx, cy   = roi["cx"], roi["cy"]
            crop_w, crop_h = roi["crop_w"], roi["crop_h"]
            out_crop_h = OUTPUT_CROP_H if OUTPUT_CROP_H is not None else crop_h

            # grid(xi,yi) 内での content 位置（cal_dx/cal_dy はキャリブレーション or 名目値）
            crop_cx = int(round(cx + cal_dx))
            crop_cy = int(round(cy + cal_dy))

            if USE_RAW_PHASE:
                # ── raw-raw モード: 大クロップで引き算 → 2π補正 → tilt補正 → center crop ──
                tl_large = extract_rect_roi(tl_warped, crop_cy, crop_cx, crop_w, TILT_CROP_H_RAW)
                if grid_img is not None:
                    grid_large = extract_rect_roi(grid_img, crop_cy, crop_cx, crop_w, TILT_CROP_H_RAW)
                    if grid_large.shape != tl_large.shape:
                        import cv2
                        grid_large = cv2.resize(
                            grid_large.astype(np.float32),
                            (tl_large.shape[1], tl_large.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        ).astype(np.float64)
                    sub_large  = tl_large - grid_large
                    subtracted = _raw_subtract_correct(sub_large, out_crop_h)
                else:
                    subtracted = np.zeros((crop_w, out_crop_h), dtype=np.float64)

            else:
                # ── 既存の backsub モード ──
                tl_crop = extract_rect_roi(tl_warped, crop_cy, crop_cx, crop_w, out_crop_h)
                if grid_img is not None:
                    grid_crop = extract_rect_roi(grid_img, crop_cy, crop_cx, crop_w, out_crop_h)
                    if grid_crop.shape != tl_crop.shape:
                        import cv2
                        grid_crop = cv2.resize(
                            grid_crop.astype(np.float32),
                            (tl_crop.shape[1], tl_crop.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        ).astype(np.float64)
                    tl_bc   = tl_crop   + compute_backsub_offset(tl_crop)
                    grid_bc = grid_crop + compute_backsub_offset(grid_crop)
                    subtracted = tl_bc - grid_bc
                else:
                    subtracted = tl_crop.copy()

            if APPLY_INVERSE_SHIFT and (sx != 0.0 or sy != 0.0):
                subtracted = apply_inverse_shift_warp(subtracted, sx, sy)

            out_stacks[ch].append(subtracted.astype(np.float32))

    # --- TIF 保存 (per-frame in ch{ch:02d}/ subdirectories) ---
    for ch in range(n_channels):
        ch_dir = out_dir / f"ch{ch:02d}"
        ch_dir.mkdir(parents=True, exist_ok=True)
        for idx, t in enumerate(pick_set):
            frame_path = ch_dir / tl_frames[t].name
            tifffile.imwrite(str(frame_path), out_stacks[ch][idx].astype(np.float32))
        print(f"保存: {ch_dir}/  ({n_frames} frames, shape={out_stacks[ch][0].shape})")

    if full_frame_stack is not None:
        full_arr = np.array(full_frame_stack, dtype=np.float32)  # (T, H, W)
        full_path = out_dir / "full_frame_grid_sub.tif"
        tifffile.imwrite(str(full_path), full_arr, imagej=True)
        print(f"保存: {full_path}  shape={full_arr.shape}")

    # --- ログ保存 ---
    log_path = channels_dir / "grid_subtract_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "timelapse_dir": str(tl_dir),
            "tl_z_index": TL_Z_INDEX,
            "shifts_json": str(shifts_json),
            "grid_dir": str(GRID_DIR),
            "base_label": BASE_LABEL,
            "grid_z_index": GRID_Z_INDEX,
            "x_step_um": X_STEP,
            "y_step_um": Y_STEP,
            "pixel_scale_um": pixel_scale_um,
            "shift_sign_x": SHIFT_SIGN_X,
            "shift_sign_y": SHIFT_SIGN_Y,
            "apply_subpixel_correction": APPLY_SUBPIXEL_CORRECTION,
            "apply_inverse_shift": APPLY_INVERSE_SHIFT,
            "use_raw_phase": USE_RAW_PHASE,
            "raw_crop": list(RAW_CROP) if USE_RAW_PHASE else None,
            "tilt_crop_h_raw": TILT_CROP_H_RAW if USE_RAW_PHASE else None,
            "frame_log": subtract_log
        }, f, indent=2, ensure_ascii=False)
    print(f"ログ保存: {log_path}")
    print("\n完了")


if __name__ == "__main__":
    main()

# %%
