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
from concurrent.futures import ThreadPoolExecutor, as_completed

# Number of threads for the per-frame main loop (cv2/tifffile release the GIL).
N_PARALLEL_FRAMES = 8

sys.path.insert(0, str(Path(__file__).parent))
from compute_pos_shifts import compute_backsub_offset
from optical_config import RAW_CROP as _OPTICAL_RAW_CROP
from ecc_utils import apply_2pi_tilt_crop, extract_rect_roi

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

# Pre-reconstructed phase directory override.
# None → auto-detect:
#   USE_RAW_PHASE=True  : TIMELAPSE_DIR/output_phase_raw/ if it exists, else on-the-fly
#   USE_RAW_PHASE=False : TIMELAPSE_DIR/output_phase/
# Set explicitly to override.
TL_PHASE_DIR = None

# ============================================================
# raw-raw subtraction モード
# ============================================================
# True → ホログラム原版TIFからPos0参照なしでon-the-fly再構成し、raw同士を直接引き算する。
# その後 2π 補正（global integer offset）と tilt 補正を適用して出力する。
# ECC 計算（compute_pos_shifts.py）は output_phase/ を使い続けるため変更不要。
USE_RAW_PHASE    = True

# 再構成用センサークロップ (r1, r2, c1, c2)  optical_config.py から import
# 一時的に上書きしたい場合のみここで明示的に値を入れる
RAW_CROP         = _OPTICAL_RAW_CROP

# 生ホログラムの z インデックス（通常 TL_Z_INDEX / GRID_Z_INDEX と同値）
RAW_TL_Z_INDEX   = 0
RAW_GRID_Z_INDEX = 18

# 2π補正 + tilt補正に使う大クロップ高さ（axis=1 方向の px 数）
TILT_CROP_H_RAW  = 270

# 背景 1/3 を左右どちら側にとるかの Pos 分割閾値（compute_pos_shifts.py と同値）。
# Pos番号 < POS_SPLIT → left 1/3 で fit。Pos番号 >= POS_SPLIT → right 1/3 で fit。
POS_SPLIT        = 33
# ============================================================


# extract_rect_roi is imported from ecc_utils


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
# Canonical reconstruction is in batch_reconstruction_grid.py.
# Aliases kept for compute_drift_online.py compatibility (real-time use).
# ============================================================
from batch_reconstruction_grid import (
    make_qpi_params as _make_qpi_params_raw,
    reconstruct_image as _reconstruct_raw,
)


def load_timelapse_holos(tl_dir, z_index):
    """タイムラプスの生ホログラム（output_phase/ でなく tl_dir 直下）のパスリストを返す。"""
    return sorted(Path(tl_dir).glob(f"img_*_ph_{z_index:03d}.tif"))


def load_grid_holo_path(pos_dir, z_index):
    """グリッドPosフォルダから生ホログラムのパスを返す。"""
    path = pos_dir / f"img_000000000_ph_{z_index:03d}.tif"
    if not path.exists():
        raise FileNotFoundError(f"グリッドホログラムが見つかりません: {path}")
    return path


def _raw_subtract_correct(sub_large, out_crop_h, fit_right=False, tilt_crop_h_raw=None):
    """Thin wrapper around tilt_utils.apply_2pi_tilt_crop for raw-raw delta.

    Shared implementation matches correct_0pergluc.py, so the delta used by
    downstream correction scripts sees the same 2pi + tilt + center crop as
    the one baked into grid_subtract output.
    """
    th = tilt_crop_h_raw if tilt_crop_h_raw is not None else TILT_CROP_H_RAW
    return apply_2pi_tilt_crop(sub_large, out_crop_h, th, fit_right=fit_right)


def select_grid(sx, sy, pos_map, grid_cal,
                pixel_scale_um, x_step=X_STEP, y_step=Y_STEP,
                shift_sign_x=SHIFT_SIGN_X, shift_sign_y=SHIFT_SIGN_Y):
    """Given a frame's ECC shift (px), pick the nearest grid (xi, yi) and return
    calibration + residual values used by process_single_frame.

    Returns: (xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, residual_x, residual_y)
    """
    dx_um = None
    dy_um = None
    if grid_cal:
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
        dx_um = shift_sign_x * sy * pixel_scale_um
        dy_um = shift_sign_y * sx * pixel_scale_um
        (xi, yi), dist_um = find_nearest_grid(pos_map, dx_um, dy_um, x_step, y_step)

    if grid_cal and (xi, yi) in grid_cal:
        cal_dx, cal_dy = grid_cal[(xi, yi)]
        residual_x = sx - cal_dx
        residual_y = sy - cal_dy
    else:
        cal_dx = shift_sign_y * yi * y_step / pixel_scale_um
        cal_dy = shift_sign_x * xi * x_step / pixel_scale_um
        residual_x = shift_sign_y * sx - yi * y_step / pixel_scale_um
        residual_y = shift_sign_x * sy - xi * x_step / pixel_scale_um
    return xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, residual_x, residual_y


def process_single_frame(tl_img, sx, sy, rois,
                         cal_dx, cal_dy, residual_x, residual_y,
                         grid_img,
                         output_crop_h_override=None,
                         tilt_crop_h_raw=None,
                         use_raw_phase=True,
                         apply_subpixel_correction=True,
                         fit_right=False,
                         apply_inverse_shift=False):
    """Process one frame end-to-end: residual subpixel warp -> per-channel crop + grid-subtract.

    This is the shared kernel used by both grid_subtract.main() (offline) and
    compute_drift_online.py Phase B (online). Keep the math here byte-identical
    between the two so downstream analysis is source-agnostic.

    Parameters
    ----------
    tl_img : np.ndarray (float64)
        Full reconstructed timelapse frame (raw-raw or phase).
    sx, sy : float
        ECC shift_x_avg / shift_y_avg [px]. Used only when apply_inverse_shift=True
        and kept in the per-frame log by the caller.
    rois : list of dict
        channel_rois.json entries (cy, cx, crop_w, crop_h).
    cal_dx, cal_dy, residual_x, residual_y : float
        Grid calibration offsets and residuals (from ``select_grid``).
    grid_img : np.ndarray or None
        Pre-loaded grid reference image. None -> write tl_crop or zeros.
    output_crop_h_override : int or None
        Override the per-channel output crop_h. None -> use roi["crop_h"].
    tilt_crop_h_raw : int or None
        Height of the wide crop used for tilt fitting. None -> module TILT_CROP_H_RAW.
    use_raw_phase : bool
        True -> raw-raw subtraction + tilt-correction path; False -> phase+backsub path.
    apply_subpixel_correction : bool
        If True and residual_{x,y} != 0, apply inverse-shift warp to tl_img first.
    fit_right : bool
        tilt-correction side (per Pos number vs POS_SPLIT).
    apply_inverse_shift : bool
        Legacy: apply -(sx, sy) to final subtracted crop (normally False).

    Returns
    -------
    per_channel_out : list[np.ndarray (float32)]
        Per-channel cropped, grid-subtracted, corrected images.
    full_frame : np.ndarray (float32) or None
        Full-frame (tl_warped - grid_img) if grid_img present else tl_warped,
        caller may discard.
    """
    n_channels = len(rois)
    _tilt_h = tilt_crop_h_raw if tilt_crop_h_raw is not None else TILT_CROP_H_RAW

    if apply_subpixel_correction and (residual_x != 0.0 or residual_y != 0.0):
        tl_warped = apply_inverse_shift_warp(tl_img, residual_x, residual_y)
    else:
        tl_warped = tl_img

    if grid_img is not None:
        full_frame = (tl_warped - grid_img).astype(np.float32)
    else:
        full_frame = tl_warped.astype(np.float32)

    per_channel_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        cx, cy = roi["cx"], roi["cy"]
        crop_w, crop_h = roi["crop_w"], roi["crop_h"]
        out_crop_h = output_crop_h_override if output_crop_h_override is not None else crop_h

        crop_cx = int(round(cx + cal_dx))
        crop_cy = int(round(cy + cal_dy))

        if use_raw_phase:
            tl_large = extract_rect_roi(tl_warped, crop_cy, crop_cx, crop_w, _tilt_h)
            if grid_img is not None:
                grid_large = extract_rect_roi(grid_img, crop_cy, crop_cx, crop_w, _tilt_h)
                if grid_large.shape != tl_large.shape:
                    import cv2
                    grid_large = cv2.resize(
                        grid_large.astype(np.float32),
                        (tl_large.shape[1], tl_large.shape[0]),
                        interpolation=cv2.INTER_LINEAR
                    ).astype(np.float64)
                sub_large = tl_large - grid_large
                subtracted = _raw_subtract_correct(sub_large, out_crop_h,
                                                    fit_right=fit_right,
                                                    tilt_crop_h_raw=_tilt_h)
            else:
                subtracted = np.zeros((crop_w, out_crop_h), dtype=np.float64)
        else:
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
                tl_bc = tl_crop + compute_backsub_offset(tl_crop)
                grid_bc = grid_crop + compute_backsub_offset(grid_crop)
                subtracted = tl_bc - grid_bc
            else:
                subtracted = tl_crop.copy()

        if apply_inverse_shift and (sx != 0.0 or sy != 0.0):
            subtracted = apply_inverse_shift_warp(subtracted, sx, sy)

        per_channel_out.append(subtracted.astype(np.float32))

    return per_channel_out, full_frame


def main():
    # --- 入力確認 ---
    tl_dir    = Path(TIMELAPSE_DIR)
    shifts_json = Path(SHIFTS_JSON)
    rois_json   = Path(CHANNEL_ROIS_JSON)

    for p, name in [(tl_dir, "TIMELAPSE_DIR"), (shifts_json, "SHIFTS_JSON"), (rois_json, "CHANNEL_ROIS_JSON")]:
        if not Path(p).exists():
            print(f"ERROR: {name} が見つかりません: {p}")
            sys.exit(1)

    # Determine tilt fit side from the timelapse Pos number (same rule as compute_pos_shifts.py).
    _pos_match = re.match(r"Pos(\d+)", tl_dir.name)
    if _pos_match is None:
        print(f"WARNING: TIMELAPSE_DIR name {tl_dir.name!r} does not start with PosN; defaulting fit_right=False")
        _pos_num = 0
    else:
        _pos_num = int(_pos_match.group(1))
    fit_right = _pos_num >= POS_SPLIT
    print(f"[tilt] Pos{_pos_num}  POS_SPLIT={POS_SPLIT}  fit_right={fit_right}")

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
    # TL_PHASE_DIR=None のとき output_phase_raw/ を auto-detect
    _tl_phase_dir = TL_PHASE_DIR
    if _tl_phase_dir is None and USE_RAW_PHASE:
        _candidate = tl_dir / "output_phase_raw"
        if _candidate.exists() and any(_candidate.glob("img_*_phase.tif")):
            _tl_phase_dir = _candidate
            print(f"[auto-detect] using pre-reconstructed raw at {_candidate}")
    _qpi_params_raw = None
    _tl_raw_source = None  # "prerecon"
    if USE_RAW_PHASE:
        if _tl_phase_dir is not None:
            phase_dir = Path(_tl_phase_dir)
            tl_frames = sorted(phase_dir.glob("img_*_phase.tif"))
            if not tl_frames:
                raise FileNotFoundError(
                    f"Pre-reconstructed raw phase not found: {phase_dir}/img_*_phase.tif\n"
                    f"Run batch_reconstruction_grid.py / batch_pipeline_all_pos.py first."
                )
            _tl_raw_source = "prerecon"
            print(f"[pre-recon mode] {len(tl_frames)} frames from {phase_dir}")
        else:
            raise FileNotFoundError(
                f"output_phase_raw/ not found under {tl_dir}\n"
                f"Run batch_pipeline_all_pos.py (Step 0) first to generate output_phase_raw/."
            )
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
    _grid_raw_sources = {}  # key -> "prerecon" | "phase"

    def _grid_prerecon_path(pos_dir, z_index):
        return pos_dir / "output_phase_raw" / f"img_000000000_ph_{z_index:03d}_phase.tif"

    def get_grid_image(xi, yi):
        key = (xi, yi)
        if key not in grid_img_cache:
            pos_dir = pos_map[key]
            if USE_RAW_PHASE:
                prerecon = _grid_prerecon_path(pos_dir, RAW_GRID_Z_INDEX)
                if not prerecon.exists():
                    raise FileNotFoundError(
                        f"Pre-reconstructed raw phase not found: {prerecon}\n"
                        f"Run batch_reconstruction_grid.py first."
                    )
                grid_img_cache[key] = tifffile.imread(str(prerecon)).astype(np.float64)
                _grid_raw_sources[key] = "prerecon"
            else:
                grid_img_cache[key] = load_grid_image(pos_dir, GRID_Z_INDEX)
                _grid_raw_sources[key] = "phase"
        return grid_img_cache[key]

    # --- フレームごとにシフト取得 ---
    frame_shifts = []
    for r in frame_results:
        sx = r.get("shift_x_avg") or r.get("shift_x")
        sy = r.get("shift_y_avg") or r.get("shift_y")
        frame_shifts.append((sx, sy))

    # --- Pre-allocate output buffers (parallel-safe: each idx writes once) ---
    subtract_log = [None] * n_frames
    out_stacks = [[None] * n_frames for _ in range(n_channels)]
    full_frame_stack = [None] * n_frames if OUTPUT_SAVE_FULL_FRAME else None

    def _select_grid(sx, sy):
        """Thin wrapper around module-level select_grid(), bound to this call's
        pos_map / grid_cal / pixel_scale."""
        return select_grid(sx, sy, pos_map, grid_cal, pixel_scale_um,
                           x_step=X_STEP, y_step=Y_STEP,
                           shift_sign_x=SHIFT_SIGN_X, shift_sign_y=SHIFT_SIGN_Y)

    # ---- Pre-populate grid image cache serially (thread-unsafe loader) ----
    _unique_keys = set()
    for idx in range(n_frames):
        t = pick_set[idx]
        sx, sy = frame_shifts[t]
        if sx is None or sy is None:
            sx, sy = 0.0, 0.0
        xi, yi, *_ = _select_grid(sx, sy)
        _unique_keys.add((xi, yi))
    for key in sorted(_unique_keys):
        if key in grid_img_cache:
            continue
        try:
            get_grid_image(*key)
        except FileNotFoundError as e:
            print(f"\n  [prefill] grid{key}: {e}  -> mark missing")
            grid_img_cache[key] = None  # sentinel: worker treats None as "no grid"

    def _process_one(idx):
        t = pick_set[idx]
        sx, sy = frame_shifts[t]
        if sx is None or sy is None:
            sx, sy = 0.0, 0.0

        xi, yi, dist_um, dx_um, dy_um, cal_dx, cal_dy, residual_x, residual_y = _select_grid(sx, sy)
        pos_label = f"{BASE_LABEL}_x{xi:+d}_y{yi:+d}"

        subtract_log[idx] = {
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

        tl_img = tifffile.imread(str(tl_frames[t])).astype(np.float64)

        grid_img = grid_img_cache.get((xi, yi))  # pre-populated; None if missing

        per_channel_out, full_frame = process_single_frame(
            tl_img, sx, sy, rois,
            cal_dx, cal_dy, residual_x, residual_y,
            grid_img,
            output_crop_h_override=OUTPUT_CROP_H,
            tilt_crop_h_raw=TILT_CROP_H_RAW,
            use_raw_phase=USE_RAW_PHASE,
            apply_subpixel_correction=APPLY_SUBPIXEL_CORRECTION,
            fit_right=fit_right,
            apply_inverse_shift=APPLY_INVERSE_SHIFT,
        )

        if full_frame_stack is not None:
            full_frame_stack[idx] = full_frame

        for ch in range(n_channels):
            out_stacks[ch][idx] = per_channel_out[ch]

    # --- Parallel frame processing ---
    with ThreadPoolExecutor(max_workers=N_PARALLEL_FRAMES) as ex:
        futures = [ex.submit(_process_one, i) for i in range(n_frames)]
        for f in tqdm(as_completed(futures), total=n_frames, desc="フレーム処理"):
            f.result()

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
            "tl_raw_source": _tl_raw_source,
            "tl_phase_dir": str(_tl_phase_dir) if _tl_phase_dir else None,
            "grid_raw_sources": {f"{k[0]:+d},{k[1]:+d}": v
                                  for k, v in _grid_raw_sources.items()},
            "frame_log": subtract_log
        }, f, indent=2, ensure_ascii=False)
    print(f"ログ保存: {log_path}")
    print("\n完了")


if __name__ == "__main__":
    main()

# %%
