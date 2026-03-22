# %%
"""
compute_pos_shifts.py
---------------------
1つのPosのチャネルスタック群（channel_XX*.tif）に対して
ECC or phase_correlationでフレームごとのシフト量を計算し、
チャネル間で外れ値除去しながら平均してpos_shifts.jsonに保存する。
"""
import numpy as np
import tifffile
import cv2
import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import median_filter
import concurrent.futures
import threading

# ============================================================
# 設定パラメータ
# ============================================================
CHANNELS_DIR = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels"
CHANNEL_PATTERN = "channel_*.tif"      # backsub済みなら "channel_*_bg_corr.tif"

# --- 基準画像の選択 ---
# USE_GRID_REFERENCE = True  : グリッドの x+0_y+0 画像をcropして各チャネルの基準にする（推奨）
# USE_GRID_REFERENCE = False : タイムラプスの REFERENCE_FRAME 番目を基準にする（従来方式）
USE_GRID_REFERENCE  = True
GRID_DIR            = r"E:\Acuisition\kitagishi\260301\multipos_test_1"
GRID_BASE_LABEL     = "Pos4"           # PosX_x+0_y+0 の PosX 部分
GRID_Z_INDEX        = 2               # img_000000000_ph_{Z_INDEX:03d}.tif
CHANNEL_ROIS_JSON   = r"E:\Acuisition\kitagishi\260301\movetest_8\Pos4\channels\channel_rois.json"

REFERENCE_FRAME = 150                  # USE_GRID_REFERENCE=False の場合のみ使用（1始まり）

ALIGNMENT_METHOD = 'ecc'              # 'ecc' or 'phase_correlation'
VMIN = -5.0
VMAX = 1.0                            # to_uint8の正規化範囲（ECC精度に影響）
OUTLIER_MAD_THRESH = 2.5              # チャネル間外れ値除去のMAD閾値
OUTLIER_TIMESERIES_WINDOW = 11        # 時系列外れ値検出のメジアンフィルタ幅（奇数）
OUTLIER_TIMESERIES_THRESH = 3.0       # 時系列MAD閾値（0で無効）
ECC_MIN_CORR = 0.0                    # ECC スコアがこれ未満のチャネルを除外（0.0 = 無効、pipeline から上書き）
OUTPUT_JSON = "pos_shifts.json"

# --- グリッド基準画像への gaussian_backsub 適用 ---
# True にするとグリッド基準画像にも timelapse と同じ backsub を適用する
APPLY_BACKSUB_TO_GRID_REF = True
BACKSUB_MIN_PHASE   = -1.1
BACKSUB_HIST_MIN    = -1.1
BACKSUB_HIST_MAX    =  1.5
BACKSUB_N_BINS      = 512
BACKSUB_SMOOTH_WINDOW = 20

# --- 逐次追跡モード ---
# True にすると前フレームのシフトから最近傍 grid(xi,yi) を基準に選ぶ
USE_INCREMENTAL_TRACKING   = False   # デフォルト False（後から pipeline で上書き）
X_STEP                     = 0.1    # グリッドステップ [μm]
Y_STEP                     = 0.1    # グリッドステップ [μm]
SHIFT_SIGN_X               = 1      # シフト符号（1 or -1）
SHIFT_SIGN_Y               = 1
JUMP_THRESH_UM             = 1.0   # 前フレームとのシフト差がこれ [μm] を超えたら外れ値（0で無効）
MAX_FRAMES                 = None  # テストラン用: None で全フレーム、整数で先頭 N フレームのみ
# 光学パラメータ（pixel scale 計算用）
SENSOR_PIXEL_SIZE          = 3.45e-6  # [m]
MAGNIFICATION              = 40
ORIGINAL_DIM               = 2048
RECONSTRUCTED_DIM          = 511
# グリッドキャリブレーション（calibrate_grid_positions.py の出力 JSON）
# None → 名目値 (xi*X_STEP/pixel_scale_um) を使用
GRID_CALIBRATION_JSON      = None
# --- 2段階ECC（USE_INCREMENTAL_TRACKING=True 時のみ有効） ---
USE_SECOND_PASS_ECC    = False   # True で2回目ECCを有効化
FIRST_PASS_HALF        = False   # (無効化済み: pass1/2/3 ともに full crop を使用)
SECOND_PASS_HALF       = 'right' # (無効化済み: full crop に統一)
USE_THIRD_PASS_ECC     = False   # True で3回目ECC（pass2結果から最近傍grid再選択 → half ECC）
# corr/shift データを NPZ + CSV に保存（True 推奨: subtract 画像との対応確認用）
SAVE_CORR_DATA         = True
# --- 並列処理 ---
N_WORKERS = None               # None = os.cpu_count(). 1 = 直列（デバッグ用）

# ============================================================
# ★ シフト適用＋引き算＋最終 crop 出力
# ============================================================
APPLY_SHIFT_AND_CROP    = True     # False でこのステップをスキップ

# タイムラプスの z インデックス（img_*_ph_{Z}_phase.tif のZZZ部分）
APPLY_TL_Z_INDEX        = 0        # 通常 0 (ph_000)

# 最終 crop パラメータ（横長: 40行 × 440列）
FINAL_CROP_CY           = 256      # Y中心（ユーザーが設定）
FINAL_CROP_CX           = 256      # X中心（crop_h=440 で cx±220 が画像内に収まるよう設定）
FINAL_CROP_W            = 40       # Y方向行数（高さ）
FINAL_CROP_H            = 440      # X方向列数（幅）

# 出力先（None なら channels_dir/crop_subtracted/ に自動設定）
APPLY_OUT_DIR           = None
# ============================================================


def to_uint8(img, vmin=VMIN, vmax=VMAX):
    clipped = np.clip(img, vmin, vmax)
    normalized = (clipped - vmin) / (vmax - vmin)
    return (normalized * 255).astype(np.uint8)


def compute_backsub_offset(img: np.ndarray) -> float:
    """
    gaussian_backsub と同じ手法で背景ピークをガウスフィットし、
    補正オフセット（= -peak_mean）を返す。ファイル保存なし。
    img: float 配列（任意形状）
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.optimize import curve_fit

    bin_edges = np.linspace(BACKSUB_HIST_MIN, BACKSUB_HIST_MAX, BACKSUB_N_BINS + 1)
    hist_counts, _ = np.histogram(img.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]

    smoothed = uniform_filter1d(hist_counts, size=BACKSUB_SMOOTH_WINDOW, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=BACKSUB_SMOOTH_WINDOW, mode='nearest')

    valid_idx = np.where(bin_centers >= BACKSUB_MIN_PHASE)[0]
    max_search_idx = int(len(bin_centers) * 0.95)
    search_idx = valid_idx[valid_idx < max_search_idx]
    if len(search_idx) == 0:
        return 0.0

    peak_idx = search_idx[np.argmax(smoothed[search_idx])]
    peak_value = bin_centers[peak_idx]

    fit_width = 300
    s = max(0, peak_idx - fit_width)
    e = min(len(bin_centers), peak_idx + fit_width)
    x_data = bin_centers[s:e]
    y_data = smoothed[s:e]

    def gaussian(x, amp, mean, std):
        return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))

    try:
        p0 = [float(np.max(y_data)), peak_value, bin_width * 20]
        popt, _ = curve_fit(gaussian, x_data, y_data, p0=p0, maxfev=5000)
        _, mean_fit, _ = popt
        return float(-mean_fit)
    except Exception as ex:
        print(f"    [backsub] Gaussian fit 失敗 ({ex}), ピーク値で代用")
        return float(-peak_value)


def ecc_align(ref_u8, tl_u8):
    """ECC アライメントで (shift_x, shift_y, correlation) を返す。失敗時は None。"""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-7)
    try:
        correlation, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(correlation)
    except Exception:
        return None


def phase_align(ref_img, tl_img):
    """phase_cross_correlation で (shift_x, shift_y, correlation) を返す。失敗時は None。"""
    from skimage import registration
    try:
        shift, error, _ = registration.phase_cross_correlation(
            ref_img, tl_img, upsample_factor=10
        )
        return float(shift[1]), float(shift[0]), float(1.0 - error)
    except Exception:
        return None


def mad(arr):
    """Median Absolute Deviation"""
    m = np.median(arr)
    return np.median(np.abs(arr - m))


def remove_outliers_mad(values, thresh):
    """外れ値フラグを返す。values: list of float, thresh: MAD閾値倍率。"""
    arr = np.array(values, dtype=np.float64)
    m = np.median(arr)
    md = mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - m) > thresh * md


def detect_timeseries_outliers(shift_avg, window, thresh):
    """
    時系列シフトに対してrolling median basedの外れ値検出。
    window: メジアンフィルタ幅（奇数推奨）
    thresh: MAD倍率（0で全フラグfalse）
    Returns: bool array, shape=(n_frames,)
    """
    if thresh <= 0:
        return np.zeros(len(shift_avg), dtype=bool)
    arr = np.array(shift_avg, dtype=np.float64)
    smoothed = median_filter(arr, size=window, mode='reflect')
    residual = arr - smoothed
    md = mad(residual)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(residual) > thresh * md


def load_grid_refs(channels_dir, n_channels):
    """
    グリッドの x+0_y+0 画像を読み込み、各チャネルのROIでcropして
    per-channel基準画像リストを返す。
    """
    from channel_crop import extract_rect_roi

    # 候補を優先順に試す:
    #   1. output_phase/*_ph_ZZZ_phase.tif  (pipeline_full.py 再構成済み)
    #   2. output_phase/*_ph_ZZZ.tif        (旧命名)
    #   3. *_ph_ZZZ.tif                     (未再構成の生画像 fallback)
    base_dir = Path(GRID_DIR) / f"{GRID_BASE_LABEL}_x+0_y+0"
    z_str = f"ph_{GRID_Z_INDEX:03d}"

    candidates = [
        base_dir / "output_phase" / f"img_000000000_{z_str}_phase.tif",
        base_dir / "output_phase" / f"img_000000000_{z_str}.tif",
        base_dir / f"img_000000000_{z_str}.tif",
    ]
    grid_ref_path = next((p for p in candidates if p.exists()), None)
    if grid_ref_path is None:
        raise FileNotFoundError(
            f"グリッド基準画像が見つかりません: {base_dir}\n"
            f"  試したパス:\n" + "\n".join(f"    {p}" for p in candidates)
        )

    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        raise FileNotFoundError(f"channel_rois.json が見つかりません: {rois_path}")

    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)

    print(f"グリッド基準画像: {grid_ref_path}")
    print(f"  グリッド画像サイズ: {grid_img.shape}")

    refs = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        if APPLY_BACKSUB_TO_GRID_REF:
            offset = compute_backsub_offset(cropped)
            cropped = cropped + offset
            print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}  backsub offset={offset:+.4f} rad")
        else:
            print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}")
        refs.append(cropped)

    return refs, str(grid_ref_path)


def load_grid_calibration(json_path):
    """
    grid_calibration.json を読み込み、(xi, yi) → (actual_dx_px, actual_dy_px) の dict を返す。
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    cal = {}
    for entry in data.get("positions", []):
        cal[(entry["xi"], entry["yi"])] = (
            entry["actual_dx_px"],
            entry["actual_dy_px"],
        )
    return cal


def scan_grid_positions(grid_dir, base_label):
    """(xi, yi) → folder_path のマップを返す。"""
    import re
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


def find_nearest_grid(pos_map, dx_um, dy_um, x_step, y_step):
    """最近傍 (xi, yi) と距離を返す。"""
    best_key, best_dist = None, float('inf')
    for (xi, yi) in pos_map:
        dist = ((xi * x_step - dx_um) ** 2 + (yi * y_step - dy_um) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = (xi, yi)
    return best_key, best_dist


def load_grid_ref_mn(pos_map, xi, yi, rois, n_channels):
    """
    grid(xi, yi) の各チャネル ROI crop を返す。
    固定 (cx, cy) で crop するので pre-cropped stacks と直接比較可能。
    """
    from channel_crop import extract_rect_roi
    pos_dir = pos_map[(xi, yi)]
    fname = f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"グリッド画像が見つかりません: {path}")
    grid_img = tifffile.imread(str(path)).astype(np.float64)
    refs_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        if APPLY_BACKSUB_TO_GRID_REF:
            cropped = cropped + compute_backsub_offset(cropped)
        refs_out.append(cropped)
    return refs_out


def load_grid_ref_mn_half(pos_map, xi, yi, rois, n_channels):
    """grid(xi, yi) の各チャネル ROI full crop を返す（2段階ECC用）。"""
    from channel_crop import extract_rect_roi
    pos_dir = pos_map[(xi, yi)]
    fname = f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"グリッド画像が見つかりません: {path}")
    grid_img = tifffile.imread(str(path)).astype(np.float64)
    refs_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        if APPLY_BACKSUB_TO_GRID_REF:
            offset = compute_backsub_offset(cropped)
            cropped = cropped + offset
            print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}  backsub offset={offset:+.4f} rad")
        else:
            print(f"  ch{ch:02d} ROI crop (full): {cropped.shape}")
        refs_out.append(cropped)
    return refs_out


def _select_nearest_grid(shift_x, shift_y, grid_cal, pos_map, pixel_scale_um):
    """shift_x/y [px] から最近傍 (xi, yi) を返す。"""
    if grid_cal:
        best_key, best_dist = None, float('inf')
        for key, (adx, ady) in grid_cal.items():
            if key not in pos_map:
                continue
            dist = ((adx - shift_x) ** 2 + (ady - shift_y) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key
    else:
        dx_um = SHIFT_SIGN_X * shift_y * pixel_scale_um
        dy_um = SHIFT_SIGN_Y * shift_x * pixel_scale_um
        (xi, yi), _ = find_nearest_grid(pos_map, dx_um, dy_um, X_STEP, Y_STEP)
        return xi, yi


def _get_grid_offset(xi, yi, grid_cal, pixel_scale_um):
    """grid(xi, yi) の content offset [px] (grid(0,0) 基準) を返す。"""
    if grid_cal and (xi, yi) in grid_cal:
        return grid_cal[(xi, yi)]
    return (SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um,
            SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um)


def _save_corr_npz_csv(records, channels_dir):
    """
    corr データを NPZ（全データ）+ CSV（per-frame サマリー）として保存する。
    subtract 画像と corr 値の対応確認用。
    """
    import csv as _csv
    channels_dir = Path(channels_dir)

    def _get(key):
        return np.array(
            [r.get(key) if r.get(key) is not None else np.nan for r in records],
            dtype=np.float32)

    save_dict = {
        "t":       np.array([r["t"]  for r in records], dtype=np.int32),
        "ch":      np.array([r["ch"] for r in records], dtype=np.int32),
        "shift_x": _get("shift_x"),
        "shift_y": _get("shift_y"),
        "corr":    _get("corr"),
        "grid_xi": np.array([r.get("grid_xi", 0) for r in records], dtype=np.int32),
        "grid_yi": np.array([r.get("grid_yi", 0) for r in records], dtype=np.int32),
        "failed":  np.array([r.get("failed", False) for r in records], dtype=bool),
    }
    has_2pass = any("pass1_corr" in r for r in records)
    has_3pass = any("pass3_corr" in r for r in records)
    if has_2pass:
        save_dict.update({
            "pass1_corr":    _get("pass1_corr"),
            "pass1_shift_x": _get("pass1_shift_x"),
            "pass1_shift_y": _get("pass1_shift_y"),
            "pass1_grid_xi": np.array([r.get("pass1_grid_xi", 0) for r in records], dtype=np.int32),
            "pass1_grid_yi": np.array([r.get("pass1_grid_yi", 0) for r in records], dtype=np.int32),
            "pass2_corr":    _get("pass2_corr"),
            "pass2_shift_x": _get("pass2_shift_x"),
            "pass2_shift_y": _get("pass2_shift_y"),
            "pass2_grid_xi": np.array([r.get("pass2_grid_xi", 0) for r in records], dtype=np.int32),
            "pass2_grid_yi": np.array([r.get("pass2_grid_yi", 0) for r in records], dtype=np.int32),
        })
    if has_3pass:
        save_dict.update({
            "pass3_corr":    _get("pass3_corr"),
            "pass3_shift_x": _get("pass3_shift_x"),
            "pass3_shift_y": _get("pass3_shift_y"),
            "pass3_grid_xi": np.array([r.get("pass3_grid_xi", 0) for r in records], dtype=np.int32),
            "pass3_grid_yi": np.array([r.get("pass3_grid_yi", 0) for r in records], dtype=np.int32),
        })

    npz_path = channels_dir / "pos_shifts_corr_data.npz"
    np.savez_compressed(str(npz_path), **save_dict)
    print(f"  [corr_data] NPZ保存: {npz_path}")

    # per-frame summary CSV
    csv_path = channels_dir / "pos_shifts_corr_summary.csv"
    frame_indices = sorted(set(r["t"] for r in records))
    cols = ["frame_index", "n_channels", "corr_mean", "corr_min", "corr_max", "grid_xi", "grid_yi"]
    if has_2pass:
        cols += ["pass1_corr_mean", "pass2_corr_mean"]
    if has_3pass:
        cols += ["pass3_corr_mean"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for t_idx in frame_indices:
            recs_t = [r for r in records
                      if r["t"] == t_idx and not r.get("failed", True) and r.get("corr") is not None]
            if not recs_t:
                continue
            corrs = [r["corr"] for r in recs_t]
            from collections import Counter
            xi_counts = Counter(r.get("grid_xi", 0) for r in recs_t)
            yi_counts = Counter(r.get("grid_yi", 0) for r in recs_t)
            row = {
                "frame_index": t_idx,
                "n_channels":  len(recs_t),
                "corr_mean":   round(float(np.mean(corrs)), 6),
                "corr_min":    round(float(np.min(corrs)), 6),
                "corr_max":    round(float(np.max(corrs)), 6),
                "grid_xi":     xi_counts.most_common(1)[0][0],
                "grid_yi":     yi_counts.most_common(1)[0][0],
            }
            if has_2pass:
                p1 = [r["pass1_corr"] for r in recs_t if r.get("pass1_corr") is not None]
                p2 = [r["pass2_corr"] for r in recs_t if r.get("pass2_corr") is not None]
                row["pass1_corr_mean"] = round(float(np.mean(p1)), 6) if p1 else ""
                row["pass2_corr_mean"] = round(float(np.mean(p2)), 6) if p2 else ""
            if has_3pass:
                p3 = [r["pass3_corr"] for r in recs_t if r.get("pass3_corr") is not None]
                row["pass3_corr_mean"] = round(float(np.mean(p3)), 6) if p3 else ""
            writer.writerow(row)
    print(f"  [corr_data] CSV保存: {csv_path}")


def _save_exclusion_summary_csv(frame_results, channels_dir):
    """
    per-frame の除外チャネル内訳を CSV として保存する。
    columns: frame_index, n_total, n_excl_failed, n_excl_low_ecc, n_excl_mad, n_used
    """
    import csv as _csv
    channels_dir = Path(channels_dir)
    csv_path = channels_dir / "pos_shifts_exclusion_summary.csv"
    cols = ["frame_index", "n_total", "n_excl_failed", "n_excl_low_ecc", "n_excl_mad", "n_used"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in frame_results:
            pc = r.get("per_channel", [])
            n_total = len(pc)
            n_failed  = sum(1 for c in pc if c.get("exclude_reason") == "alignment_failed")
            n_low_ecc = sum(1 for c in pc if c.get("exclude_reason") == "low_ecc_score")
            n_mad     = sum(1 for c in pc if c.get("exclude_reason") == "channel_outlier_mad")
            n_used    = r.get("n_channels_used", 0)
            writer.writerow({
                "frame_index":    r["frame_index"],
                "n_total":        n_total,
                "n_excl_failed":  n_failed,
                "n_excl_low_ecc": n_low_ecc,
                "n_excl_mad":     n_mad,
                "n_used":         n_used,
            })
    print(f"  [exclusion_summary] CSV保存: {csv_path}")


def _frame_result_from_per_channel(t, per_channel):
    """
    per_channel リストから外れ値除去・平均を計算し (frame_result, sx_avg, sy_avg) を返す。
    全チャネル失敗時は sx_avg=sy_avg=None。
    """
    valid = [c for c in per_channel if not c["excluded"]]
    if len(valid) == 0:
        return {
            "frame_index": t,
            "shift_x_avg": None,
            "shift_y_avg": None,
            "n_channels_used": 0,
            "n_channels_excluded_outlier": 0,
            "is_outlier_timeseries": False,
            "per_channel": per_channel
        }, None, None

    xs = np.array([c["shift_x"] for c in valid])
    ys = np.array([c["shift_y"] for c in valid])

    if len(valid) >= 3:
        outlier_x = remove_outliers_mad(xs.tolist(), OUTLIER_MAD_THRESH)
        outlier_y = remove_outliers_mad(ys.tolist(), OUTLIER_MAD_THRESH)
        is_outlier = outlier_x | outlier_y
    else:
        is_outlier = np.zeros(len(valid), dtype=bool)

    for i, c in enumerate(valid):
        if is_outlier[i]:
            c["excluded"] = True
            c["exclude_reason"] = "channel_outlier_mad"

    used = [c for c in valid if not c["excluded"]]
    n_excluded = int(np.sum(is_outlier))

    if len(used) == 0:
        sx_avg = float(np.mean(xs))
        sy_avg = float(np.mean(ys))
        n_used = len(valid)
        n_excluded = 0
        for c in valid:
            c["excluded"] = False
            c["exclude_reason"] = None
    else:
        sx_avg = float(np.mean([c["shift_x"] for c in used]))
        sy_avg = float(np.mean([c["shift_y"] for c in used]))
        n_used = len(used)

    return {
        "frame_index": t,
        "shift_x_avg": sx_avg,
        "shift_y_avg": sy_avg,
        "n_channels_used": n_used,
        "n_channels_excluded_outlier": n_excluded,
        "is_outlier_timeseries": False,
        "per_channel": per_channel
    }, sx_avg, sy_avg


def apply_shifts_and_crop(channels_dir, frame_results, grid_ref_path_str=None):
    """
    pos_shifts.json で求めたシフトを output_phase 全フレームに適用し、
    参照引き算 → 40×440 crop → channels_dir/crop_subtracted/ に保存。
    """
    from channel_crop import extract_rect_roi

    tl_dir = Path(channels_dir).parent          # = output_phase/
    out_dir = Path(APPLY_OUT_DIR) if APPLY_OUT_DIR else Path(channels_dir) / "crop_subtracted"
    out_dir.mkdir(parents=True, exist_ok=True)

    # タイムラプスフレーム一覧
    tl_frames = sorted(tl_dir.glob(f"img_*_ph_{APPLY_TL_Z_INDEX:03d}_phase.tif"))
    if not tl_frames:
        print(f"[APPLY] output_phase フレームが見つかりません: {tl_dir}")
        return

    # 参照画像（フル解像度）
    if grid_ref_path_str and Path(grid_ref_path_str).exists():
        ref_full = tifffile.imread(str(grid_ref_path_str)).astype(np.float64)
        print(f"[APPLY] 参照: グリッド {Path(grid_ref_path_str).name}")
    else:
        ref_full = tifffile.imread(str(tl_frames[0])).astype(np.float64)
        print(f"[APPLY] 参照: タイムラプス 1フレーム目 {tl_frames[0].name}")

    # シフト辞書 {frame_index: (sx, sy)}
    shift_map = {
        r["frame_index"]: (r["shift_x_avg"] or 0.0, r["shift_y_avg"] or 0.0)
        for r in frame_results
    }

    h, w = ref_full.shape
    for i, fp in enumerate(tqdm(tl_frames, desc="apply+crop")):
        sx, sy = shift_map.get(i, (0.0, 0.0))
        frame = tifffile.imread(str(fp)).astype(np.float64)

        # warpAffine でシフト補正
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        warp_matrix[0, 2] = float(sx)
        warp_matrix[1, 2] = float(sy)
        aligned = cv2.warpAffine(
            frame.astype(np.float32), warp_matrix, (w, h),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
        ).astype(np.float64)

        # 参照引き算
        subtracted = aligned - ref_full

        # 40×440 crop
        crop = extract_rect_roi(
            subtracted, FINAL_CROP_CY, FINAL_CROP_CX,
            FINAL_CROP_W, FINAL_CROP_H
        )

        tifffile.imwrite(str(out_dir / fp.name), crop.astype(np.float32))

    print(f"[APPLY] {len(tl_frames)} フレーム保存完了 → {out_dir}")
    print(f"[APPLY] crop shape: ({FINAL_CROP_W}, {FINAL_CROP_H})")


def main():
    channels_dir = Path(CHANNELS_DIR)
    if not channels_dir.exists():
        print(f"ERROR: CHANNELS_DIR が見つかりません: {channels_dir}")
        sys.exit(1)

    # チャネルスタック一覧
    stacks_paths = sorted(channels_dir.glob(CHANNEL_PATTERN))
    if not stacks_paths:
        print(f"ERROR: {CHANNEL_PATTERN} に合うファイルが見つかりません: {channels_dir}")
        sys.exit(1)
    print(f"チャネル数: {len(stacks_paths)}")
    for p in stacks_paths:
        print(f"  {p.name}")

    # スタック読み込み
    stacks = []
    for p in stacks_paths:
        arr = tifffile.imread(str(p))
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]  # 1フレームの場合
        stacks.append(arr.astype(np.float64))
        print(f"  {p.name}: shape={arr.shape}")

    n_frames = stacks[0].shape[0]
    n_channels = len(stacks)
    if MAX_FRAMES is not None and MAX_FRAMES < n_frames:
        stacks = [s[:MAX_FRAMES] for s in stacks]
        n_frames = MAX_FRAMES
        print(f"[TEST] フレームを {n_frames} に制限")
    print(f"\nフレーム数: {n_frames}")
    print(f"アライメント手法: {ALIGNMENT_METHOD}")
    print(f"外れ値MAD閾値: {OUTLIER_MAD_THRESH}")
    if OUTLIER_TIMESERIES_THRESH > 0:
        print(f"時系列外れ値: window={OUTLIER_TIMESERIES_WINDOW}, thresh={OUTLIER_TIMESERIES_THRESH}")
    else:
        print("時系列外れ値検出: 無効")

    # 基準画像の構築
    reference_info = {}
    if USE_GRID_REFERENCE or USE_INCREMENTAL_TRACKING:
        print(f"\n基準: グリッド x+0_y+0  ({GRID_BASE_LABEL})")
        try:
            refs, grid_ref_path_str = load_grid_refs(channels_dir, n_channels)
            reference_info = {
                "reference_type": "grid",
                "grid_dir": GRID_DIR,
                "grid_base_label": GRID_BASE_LABEL,
                "grid_z_index": GRID_Z_INDEX,
                "grid_reference_path": grid_ref_path_str,
            }
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        ref_idx = REFERENCE_FRAME - 1
        if ref_idx < 0 or ref_idx >= n_frames:
            print(f"ERROR: REFERENCE_FRAME={REFERENCE_FRAME} が範囲外 (1~{n_frames})")
            sys.exit(1)
        print(f"\n基準: タイムラプス フレーム {REFERENCE_FRAME} (0-indexed: {ref_idx})")
        refs = [stacks[ch][ref_idx] for ch in range(n_channels)]
        reference_info = {
            "reference_type": "timelapse_frame",
            "reference_frame": REFERENCE_FRAME,
        }

    if ALIGNMENT_METHOD == 'ecc':
        refs_u8 = [to_uint8(r) for r in refs]

    # channel_rois.json 読み込み（逐次追跡モードで使用）
    rois_for_incremental = None
    if USE_INCREMENTAL_TRACKING:
        rois_path = Path(CHANNEL_ROIS_JSON)
        if not rois_path.exists():
            print(f"ERROR: CHANNEL_ROIS_JSON が見つかりません: {rois_path}")
            sys.exit(1)
        with open(rois_path, encoding="utf-8") as f:
            rois_for_incremental = json.load(f)

    # フレームごとにアライメント計算
    frame_results = []
    corr_records  = []   # corr/shift の全記録（NPZ/CSV 保存用）

    if USE_INCREMENTAL_TRACKING:
        pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
        pos_map = scan_grid_positions(GRID_DIR, GRID_BASE_LABEL)
        if not pos_map:
            print("ERROR: グリッド Pos が見つかりません")
            sys.exit(1)
        print(f"[incremental] grid Pos数: {len(pos_map)}")

        # グリッドキャリブレーション読み込み
        grid_cal = {}
        if GRID_CALIBRATION_JSON:
            cal_path = Path(GRID_CALIBRATION_JSON)
            if cal_path.exists():
                grid_cal = load_grid_calibration(str(cal_path))
                print(f"[calibration] {len(grid_cal)} 点の実計測オフセットを読み込み: {cal_path}")
            else:
                print(f"[calibration] JSON が見つかりません: {cal_path}  → 名目値を使用")

        prev_shift_x, prev_shift_y = 0.0, 0.0
        refs_dict      = {(0, 0): refs}           # (xi,yi) → full crop refs (非2段階ECC用)
        refs_u8_dict   = {(0, 0): refs_u8}        # (xi,yi) → full crop refs uint8
        grid_half_cache    = {}   # (xi, yi) → list of float crops (full crop, pass2用)
        grid_half_u8_cache = {}   # (xi, yi) → list of uint8 crops (full crop, pass2用)

        _n_workers = N_WORKERS or os.cpu_count()
        _cache_lock = threading.Lock()
        print(f"並列ワーカー数: {_n_workers} (incremental / チャネル並列)")

        # pass1用: grid(0,0)の half or full crop（USE_SECOND_PASS_ECC 時は常にgrid(0,0)固定）
        if USE_SECOND_PASS_ECC:
            if FIRST_PASS_HALF:
                print(f"[pass1] grid(0,0) HALF crop ({SECOND_PASS_HALF}側) を使用")
                p1_refs    = load_grid_ref_mn_half(pos_map, 0, 0, rois_for_incremental, n_channels)
                p1_refs_u8 = [to_uint8(r) for r in p1_refs] if ALIGNMENT_METHOD == 'ecc' else None
            else:
                print(f"[pass1] grid(0,0) FULL crop を使用（上記の backsub offset が適用済み）")
                p1_refs    = refs
                p1_refs_u8 = refs_u8 if ALIGNMENT_METHOD == 'ecc' else None

        for t in tqdm(range(n_frames), desc="フレーム処理"):
            # ---- pass1: 常に grid(0,0) 基準でECC（インクリメンタルなし） ----
            # grid_offset は (0,0) なので fine1 = shift1
            grid_offset_x, grid_offset_y = 0.0, 0.0

            # USE_SECOND_PASS_ECC=False の場合は従来通り nearest grid 選択
            if not USE_SECOND_PASS_ECC:
                xi, yi = _select_nearest_grid(prev_shift_x, prev_shift_y, grid_cal, pos_map, pixel_scale_um)
                if (xi, yi) not in refs_dict:
                    try:
                        refs_dict[(xi, yi)]    = load_grid_ref_mn(pos_map, xi, yi, rois_for_incremental, n_channels)
                        refs_u8_dict[(xi, yi)] = ([to_uint8(r) for r in refs_dict[(xi, yi)]]
                                                   if ALIGNMENT_METHOD == 'ecc' else None)
                    except FileNotFoundError as e:
                        print(f"\n[t={t}] {e}  → grid(0,0) にフォールバック")
                        xi, yi = 0, 0
                p1_refs    = refs_dict.get((xi, yi), refs)
                p1_refs_u8 = refs_u8_dict.get((xi, yi), refs_u8 if ALIGNMENT_METHOD == 'ecc' else None)
                grid_offset_x, grid_offset_y = _get_grid_offset(xi, yi, grid_cal, pixel_scale_um)
            else:
                xi, yi = 0, 0  # pass1 は常に grid(0,0)

            per_channel = []

            def _compute_ch(ch):
                frame = stacks[ch][t]

                # ---- 1st pass ECC（grid(0,0) 固定 or nearest grid） ----
                frame_p1 = frame

                if ALIGNMENT_METHOD == 'ecc':
                    result1 = ecc_align(p1_refs_u8[ch], to_uint8(frame_p1))
                else:
                    result1 = phase_align(p1_refs[ch], frame_p1)

                if result1 is None:
                    return (
                        {"channel": ch, "shift_x": None, "shift_y": None,
                         "correlation": None, "excluded": True,
                         "exclude_reason": "alignment_failed",
                         "grid_xi": xi, "grid_yi": yi},
                        {"t": t, "ch": ch, "shift_x": None, "shift_y": None,
                         "corr": None, "grid_xi": xi, "grid_yi": yi, "failed": True},
                    )

                fine1_x, fine1_y, corr1 = result1
                shift1_x = fine1_x + grid_offset_x
                shift1_y = fine1_y + grid_offset_y

                if not USE_SECOND_PASS_ECC:
                    low_ecc = ECC_MIN_CORR > 0 and corr1 < ECC_MIN_CORR
                    return (
                        {"channel": ch,
                         "shift_x": shift1_x, "shift_y": shift1_y, "correlation": corr1,
                         "excluded": low_ecc,
                         "exclude_reason": "low_ecc_score" if low_ecc else None,
                         "grid_xi": xi, "grid_yi": yi},
                        {"t": t, "ch": ch,
                         "shift_x": shift1_x, "shift_y": shift1_y, "corr": corr1,
                         "grid_xi": xi, "grid_yi": yi, "failed": False},
                    )

                # ---- 2nd pass ECC (full crop) ----
                xi2, yi2 = _select_nearest_grid(shift1_x, shift1_y, grid_cal, pos_map, pixel_scale_um)
                grid_offset_x2, grid_offset_y2 = _get_grid_offset(xi2, yi2, grid_cal, pixel_scale_um)

                with _cache_lock:
                    if (xi2, yi2) not in grid_half_cache:
                        try:
                            grid_half_cache[(xi2, yi2)] = load_grid_ref_mn_half(
                                pos_map, xi2, yi2, rois_for_incremental, n_channels)
                            if ALIGNMENT_METHOD == 'ecc':
                                grid_half_u8_cache[(xi2, yi2)] = [
                                    to_uint8(r) for r in grid_half_cache[(xi2, yi2)]]
                        except FileNotFoundError as e:
                            print(f"\n[t={t},ch={ch}] full-crop ロード失敗: {e}  → pass1 結果を使用")
                            low_ecc = ECC_MIN_CORR > 0 and corr1 < ECC_MIN_CORR
                            return (
                                {"channel": ch,
                                 "shift_x": shift1_x, "shift_y": shift1_y, "correlation": corr1,
                                 "excluded": low_ecc,
                                 "exclude_reason": "low_ecc_score" if low_ecc else None,
                                 "grid_xi": xi, "grid_yi": yi,
                                 "pass1_shift_x": shift1_x, "pass1_shift_y": shift1_y,
                                 "pass1_fine_x": fine1_x, "pass1_fine_y": fine1_y,
                                 "pass1_grid_offset_x": grid_offset_x, "pass1_grid_offset_y": grid_offset_y,
                                 "pass1_corr": corr1, "pass1_grid_xi": xi, "pass1_grid_yi": yi,
                                 "pass2_shift_x": None, "pass2_shift_y": None, "pass2_corr": None,
                                 "pass2_fine_x": None, "pass2_fine_y": None,
                                 "pass2_grid_offset_x": None, "pass2_grid_offset_y": None,
                                 "pass2_grid_xi": xi2, "pass2_grid_yi": yi2},
                                {"t": t, "ch": ch,
                                 "shift_x": shift1_x, "shift_y": shift1_y, "corr": corr1,
                                 "grid_xi": xi, "grid_yi": yi, "failed": False,
                                 "pass1_corr": corr1, "pass1_grid_xi": xi, "pass1_grid_yi": yi,
                                 "pass2_corr": None, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2},
                            )

                if ALIGNMENT_METHOD == 'ecc':
                    result2 = ecc_align(grid_half_u8_cache[(xi2, yi2)][ch], to_uint8(frame))
                else:
                    result2 = phase_align(grid_half_cache[(xi2, yi2)][ch], frame)

                if result2 is None:
                    # pass2 失敗 → pass1 結果を採用
                    low_ecc = ECC_MIN_CORR > 0 and corr1 < ECC_MIN_CORR
                    return (
                        {"channel": ch,
                         "shift_x": shift1_x, "shift_y": shift1_y, "correlation": corr1,
                         "excluded": low_ecc,
                         "exclude_reason": "low_ecc_score" if low_ecc else None,
                         "grid_xi": xi, "grid_yi": yi,
                         "pass1_shift_x": shift1_x, "pass1_shift_y": shift1_y,
                         "pass1_fine_x": fine1_x, "pass1_fine_y": fine1_y,
                         "pass1_grid_offset_x": grid_offset_x, "pass1_grid_offset_y": grid_offset_y,
                         "pass1_corr": corr1, "pass1_grid_xi": xi, "pass1_grid_yi": yi,
                         "pass2_shift_x": None, "pass2_shift_y": None, "pass2_corr": None,
                         "pass2_fine_x": None, "pass2_fine_y": None,
                         "pass2_grid_offset_x": grid_offset_x2, "pass2_grid_offset_y": grid_offset_y2,
                         "pass2_grid_xi": xi2, "pass2_grid_yi": yi2},
                        {"t": t, "ch": ch,
                         "shift_x": shift1_x, "shift_y": shift1_y, "corr": corr1,
                         "grid_xi": xi, "grid_yi": yi, "failed": False,
                         "pass1_corr": corr1, "pass1_grid_xi": xi, "pass1_grid_yi": yi,
                         "pass2_corr": None, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2},
                    )

                fine2_x, fine2_y, corr2 = result2
                shift2_x = fine2_x + grid_offset_x2
                shift2_y = fine2_y + grid_offset_y2
                final_shift_x, final_shift_y = shift2_x, shift2_y
                final_corr = corr2
                final_xi, final_yi = xi2, yi2

                # ---- 3rd pass ECC (pass2結果から最近傍grid再選択 → full crop ECC) ----
                fine3_x = fine3_y = corr3 = None
                xi3, yi3 = xi2, yi2
                grid_offset_x3, grid_offset_y3 = grid_offset_x2, grid_offset_y2
                if USE_THIRD_PASS_ECC:
                    xi3, yi3 = _select_nearest_grid(shift2_x, shift2_y, grid_cal, pos_map, pixel_scale_um)
                    grid_offset_x3, grid_offset_y3 = _get_grid_offset(xi3, yi3, grid_cal, pixel_scale_um)
                    with _cache_lock:
                        if (xi3, yi3) not in grid_half_cache:
                            try:
                                grid_half_cache[(xi3, yi3)] = load_grid_ref_mn_half(
                                    pos_map, xi3, yi3, rois_for_incremental, n_channels)
                                if ALIGNMENT_METHOD == 'ecc':
                                    grid_half_u8_cache[(xi3, yi3)] = [
                                        to_uint8(r) for r in grid_half_cache[(xi3, yi3)]]
                            except FileNotFoundError as e:
                                print(f"\n[t={t},ch={ch}] pass3 full-crop ロード失敗: {e}  → pass2 結果を使用")
                                xi3, yi3 = xi2, yi2
                    if (xi3, yi3) in grid_half_cache:
                        if ALIGNMENT_METHOD == 'ecc':
                            result3 = ecc_align(grid_half_u8_cache[(xi3, yi3)][ch], to_uint8(frame))
                        else:
                            result3 = phase_align(grid_half_cache[(xi3, yi3)][ch], frame)
                        if result3 is not None:
                            fine3_x, fine3_y, corr3 = result3
                            final_shift_x = fine3_x + grid_offset_x3
                            final_shift_y = fine3_y + grid_offset_y3
                            final_corr = corr3
                            final_xi, final_yi = xi3, yi3

                low_ecc_corrs = [corr1, corr2]
                if USE_THIRD_PASS_ECC and corr3 is not None:
                    low_ecc_corrs.append(corr3)
                low_ecc = ECC_MIN_CORR > 0 and any(c < ECC_MIN_CORR for c in low_ecc_corrs)

                pc_entry = {
                    "channel": ch,
                    "shift_x": final_shift_x, "shift_y": final_shift_y, "correlation": final_corr,
                    "excluded": low_ecc,
                    "exclude_reason": "low_ecc_score" if low_ecc else None,
                    "grid_xi": final_xi, "grid_yi": final_yi,
                    "pass1_shift_x": shift1_x, "pass1_shift_y": shift1_y,
                    "pass1_fine_x": fine1_x, "pass1_fine_y": fine1_y,
                    "pass1_grid_offset_x": grid_offset_x, "pass1_grid_offset_y": grid_offset_y,
                    "pass1_corr": corr1, "pass1_grid_xi": xi, "pass1_grid_yi": yi,
                    "pass2_shift_x": shift2_x, "pass2_shift_y": shift2_y,
                    "pass2_fine_x": fine2_x, "pass2_fine_y": fine2_y,
                    "pass2_grid_offset_x": grid_offset_x2, "pass2_grid_offset_y": grid_offset_y2,
                    "pass2_corr": corr2, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2,
                }
                cr_entry = {
                    "t": t, "ch": ch,
                    "shift_x": final_shift_x, "shift_y": final_shift_y, "corr": final_corr,
                    "grid_xi": final_xi, "grid_yi": final_yi, "failed": False,
                    "pass1_corr": corr1, "pass1_grid_xi": xi, "pass1_grid_yi": yi,
                    "pass2_corr": corr2, "pass2_grid_xi": xi2, "pass2_grid_yi": yi2,
                }
                if USE_THIRD_PASS_ECC:
                    pc_entry.update({
                        "pass3_shift_x": final_shift_x if corr3 is not None else None,
                        "pass3_shift_y": final_shift_y if corr3 is not None else None,
                        "pass3_fine_x": fine3_x, "pass3_fine_y": fine3_y,
                        "pass3_grid_offset_x": grid_offset_x3, "pass3_grid_offset_y": grid_offset_y3,
                        "pass3_corr": corr3, "pass3_grid_xi": xi3, "pass3_grid_yi": yi3,
                    })
                    cr_entry.update({
                        "pass3_corr": corr3,
                        "pass3_grid_xi": xi3, "pass3_grid_yi": yi3,
                    })
                return pc_entry, cr_entry

            with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers) as ex_ch:
                ch_results = list(ex_ch.map(_compute_ch, range(n_channels)))
            for pc_e, cr_e in ch_results:
                per_channel.append(pc_e)
                corr_records.append(cr_e)

            frame_result, sx_avg, sy_avg = _frame_result_from_per_channel(t, per_channel)

            # ジャンプ検出: 前フレームとのシフト差が JUMP_THRESH_UM を超えたら外れ値
            if sx_avg is not None and JUMP_THRESH_UM > 0:
                jump = (((sx_avg - prev_shift_x) * pixel_scale_um) ** 2
                        + ((sy_avg - prev_shift_y) * pixel_scale_um) ** 2) ** 0.5
                if jump > JUMP_THRESH_UM:
                    frame_result["is_outlier_timeseries"] = True
                    sx_avg = sy_avg = None  # prev_shift を更新しない

            frame_results.append(frame_result)
            if sx_avg is not None:
                prev_shift_x, prev_shift_y = sx_avg, sy_avg

        # ---- corr データ保存 ----
        if SAVE_CORR_DATA and corr_records:
            _save_corr_npz_csv(corr_records, channels_dir)

    else:
        # ---- 非インクリメンタル: フレームをスレッドプールで並列処理 ----
        def _run_frame(t):
            pc, cr = [], []
            for ch in range(n_channels):
                frame = stacks[ch][t]
                if ALIGNMENT_METHOD == 'ecc':
                    res = ecc_align(refs_u8[ch], to_uint8(frame))
                else:
                    res = phase_align(refs[ch], frame)
                if res is None:
                    pc.append({
                        "channel": ch, "shift_x": None, "shift_y": None,
                        "correlation": None, "excluded": True,
                        "exclude_reason": "alignment_failed"
                    })
                    cr.append({"t": t, "ch": ch, "shift_x": None, "shift_y": None,
                               "corr": None, "failed": True})
                else:
                    sx, sy, corr = res
                    low_ecc = ECC_MIN_CORR > 0 and corr < ECC_MIN_CORR
                    pc.append({
                        "channel": ch, "shift_x": sx, "shift_y": sy,
                        "correlation": corr, "excluded": low_ecc,
                        "exclude_reason": "low_ecc_score" if low_ecc else None
                    })
                    cr.append({"t": t, "ch": ch, "shift_x": sx, "shift_y": sy,
                               "corr": corr, "failed": False})
            return pc, cr

        _n_workers = N_WORKERS or os.cpu_count()
        print(f"並列ワーカー数: {_n_workers} (non-incremental / フレーム並列)")
        _frame_results_map = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers) as ex:
            fs = {ex.submit(_run_frame, t): t for t in range(n_frames)}
            for fut in tqdm(concurrent.futures.as_completed(fs), total=n_frames, desc="フレーム処理"):
                t = fs[fut]
                _frame_results_map[t] = fut.result()

        for t in range(n_frames):
            pc, cr = _frame_results_map[t]
            frame_result, _, _ = _frame_result_from_per_channel(t, pc)
            frame_results.append(frame_result)
            corr_records.extend(cr)

        # ---- corr データ保存 ----
        if SAVE_CORR_DATA and corr_records:
            _save_corr_npz_csv(corr_records, channels_dir)

    # 時系列外れ値検出
    avg_x = [r["shift_x_avg"] for r in frame_results]
    avg_y = [r["shift_y_avg"] for r in frame_results]
    # None を含む場合は線形補間してから検出
    def fill_none(arr):
        arr = np.array([np.nan if v is None else v for v in arr], dtype=np.float64)
        nans = np.isnan(arr)
        if nans.all():
            return arr
        xp = np.where(~nans)[0]
        arr[nans] = np.interp(np.where(nans)[0], xp, arr[xp])
        return arr

    avg_x_filled = fill_none(avg_x)
    avg_y_filled = fill_none(avg_y)
    ts_outlier_x = detect_timeseries_outliers(avg_x_filled, OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH)
    ts_outlier_y = detect_timeseries_outliers(avg_y_filled, OUTLIER_TIMESERIES_WINDOW, OUTLIER_TIMESERIES_THRESH)
    ts_outlier = ts_outlier_x | ts_outlier_y

    for i, r in enumerate(frame_results):
        r["is_outlier_timeseries"] = bool(ts_outlier[i])

    n_ts_outlier = int(np.sum(ts_outlier))
    print(f"\n時系列外れ値フレーム数: {n_ts_outlier} / {n_frames}")

    # シフト統計
    valid_avg_x = [r["shift_x_avg"] for r in frame_results if r["shift_x_avg"] is not None]
    valid_avg_y = [r["shift_y_avg"] for r in frame_results if r["shift_y_avg"] is not None]
    if valid_avg_x:
        print(f"shift_x: 平均={np.mean(valid_avg_x):.3f}, 範囲=[{np.min(valid_avg_x):.3f}, {np.max(valid_avg_x):.3f}]")
        print(f"shift_y: 平均={np.mean(valid_avg_y):.3f}, 範囲=[{np.min(valid_avg_y):.3f}, {np.max(valid_avg_y):.3f}]")

    # JSON保存
    out = {
        "method": ALIGNMENT_METHOD,
        "n_channels": n_channels,
        "n_frames": n_frames,
        "outlier_mad_thresh": OUTLIER_MAD_THRESH,
        "outlier_timeseries_window": OUTLIER_TIMESERIES_WINDOW,
        "outlier_timeseries_thresh": OUTLIER_TIMESERIES_THRESH,
        "channels_dir": str(channels_dir),
        "channel_pattern": CHANNEL_PATTERN,
        **reference_info,
        # shift_visualize.py互換フィールド（平均シフト量を alignment_results 形式でも持つ）
        "alignment_results": [
            {
                "filename": f"frame_{r['frame_index']:06d}",
                "shift_x": r["shift_x_avg"] if r["shift_x_avg"] is not None else 0.0,
                "shift_y": r["shift_y_avg"] if r["shift_y_avg"] is not None else 0.0,
                "correlation": None,
                "warp_matrix": [[1.0, 0.0, r["shift_x_avg"] or 0.0],
                                 [0.0, 1.0, r["shift_y_avg"] or 0.0]],
                "is_outlier_timeseries": r["is_outlier_timeseries"]
            }
            for r in frame_results
        ],
        "frame_results": frame_results
    }

    out_path = channels_dir / OUTPUT_JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n保存完了: {out_path}")

    # 除外サマリー CSV 保存
    _save_exclusion_summary_csv(frame_results, channels_dir)

    # shift_visualize で可視化
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from shift_visualize import visualize_shifts, visualize_2pass_shifts, visualize_exclusion_summary
        visualize_shifts(str(out_path))
        if USE_SECOND_PASS_ECC:
            visualize_2pass_shifts(str(out_path))
        excl_csv = channels_dir / "pos_shifts_exclusion_summary.csv"
        if excl_csv.exists():
            visualize_exclusion_summary(str(excl_csv), str(out_path))
    except Exception as e:
        print(f"[shift_visualize] スキップ: {e}")

    # ---- シフト適用＋最終 crop 出力 ----
    if APPLY_SHIFT_AND_CROP:
        _grid_ref_path = reference_info.get("grid_reference_path") if USE_GRID_REFERENCE else None
        apply_shifts_and_crop(str(channels_dir), frame_results, _grid_ref_path)


if __name__ == "__main__":
    main()

# %%
