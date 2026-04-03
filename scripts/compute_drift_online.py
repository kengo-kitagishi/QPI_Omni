"""
compute_drift_online_v2.py
--------------------------
compute_drift_online.py の精度向上版。

変更点 (v1 → v2):
  1. ECC 収束基準: max_iter 10,000→100,000, epsilon 1e-6→1e-7→1e-8
  2. 2-pass ECC:
       Pass 1: grid(0,0) full-crop で粗いシフト量を取得
       Pass 2: shift1 から最近傍グリッド (xi,yi) を選択 → full-crop ECC で精密化
       最終出力は常に grid(0,0) 基準の絶対シフト量
  3. per-channel backsub は v1 と同じ（既にヒストグラムベース per-ROI）

引数・出力フォーマットは v1 と完全互換（BeanShell 側変更不要）。

使い方 (Beanshell が呼び出す):
    python compute_drift_online_v2.py \\
        --timepoint 5 \\
        --sample-raw "D:/path/Pos1/img_000000005_ph_000.tif" \\
        --bg-raw     "D:/path/Pos0/img_000000005_ph_000.tif" \\
        --config     "C:/Users/QPI/Documents/QPI_Omni/drift_session/drift_config.json"
"""

import sys
import re
import json
import argparse
import threading
import numpy as np
import tifffile
import cv2
import concurrent.futures
from pathlib import Path
from datetime import datetime
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--timepoint",   type=int,  required=True)
    p.add_argument("--sample-raw",  required=True, help="参照Posの生画像パス")
    p.add_argument("--bg-raw",      default="none", help="BG Posの生画像パス（noneでスキップ）")
    p.add_argument("--config",      required=True, help="drift_config.json のパス")
    return p.parse_args()


def load_config(config_path):
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def write_state(state_path: str, t: int, dx_um: float, dy_um: float,
                cum_dx: float, cum_dy: float, valid: bool, corr: float,
                jump: bool, ema_tx: float = 0.0, ema_ty: float = 0.0):
    """drift_state.txt を書き込む（Beanshell が key=value 形式で読む）"""
    lines = [
        "# drift_state.txt - written by compute_drift_online_v2.py",
        f"STATUS={'correction_ready' if valid else 'correction_skipped'}",
        f"TIMEPOINT={t}",
        f"DX_UM={dx_um:.6f}",
        f"DY_UM={dy_um:.6f}",
        f"CUMULATIVE_DX_UM={cum_dx:.6f}",
        f"CUMULATIVE_DY_UM={cum_dy:.6f}",
        f"CORRECTION_VALID={'true' if valid else 'false'}",
        f"ECC_CORRELATION={corr:.6f}",
        f"JUMP_DETECTED={'true' if jump else 'false'}",
        f"EMA_TX_PX={ema_tx:.6f}",
        f"EMA_TY_PX={ema_ty:.6f}",
        f"TIMESTAMP={datetime.now().isoformat()}",
    ]
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_state(state_path: str) -> dict:
    """前回の状態（累積ドリフト値など）を読む"""
    result = {
        "cumulative_dx_um": 0.0,
        "cumulative_dy_um": 0.0,
        "ema_tx_px": None,  # None = 初回フレーム（EMAをスキップ）
        "ema_ty_px": None,
    }
    try:
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k == "CUMULATIVE_DX_UM":
                    result["cumulative_dx_um"] = float(v)
                elif k == "CUMULATIVE_DY_UM":
                    result["cumulative_dy_um"] = float(v)
                elif k == "EMA_TX_PX":
                    result["ema_tx_px"] = float(v)
                elif k == "EMA_TY_PX":
                    result["ema_ty_px"] = float(v)
    except Exception:
        pass
    return result


def kf_step_1d(z_px: float, pos_est_px: float, P_nm2: float,
               Q_nm2: float, R_nm2: float, px_scale_nm: float):
    """1次元 Kalman filter (random-walk モデル) の predict+update。
    入出力はピクセル単位。内部計算は nm 単位。
    Returns: (pos_new_px, P_new_nm2, K)
    """
    z_nm   = z_px       * px_scale_nm
    pos_nm = pos_est_px * px_scale_nm
    P_pred = P_nm2 + Q_nm2
    K      = P_pred / (P_pred + R_nm2)
    pos_nm_new = pos_nm + K * (z_nm - pos_nm)
    P_new      = (1.0 - K) * P_pred
    return pos_nm_new / px_scale_nm, P_new, float(K)


def kf_step_posonly_nm(z_nm: float, pos_nm: float, P: float,
                       Q: float, R: float):
    """1D pos-only random walk Kalman step (scalar, nm units).
    Model: x_k = x_{k-1} + w_k  (w ~ N(0,Q)),  z_k = x_k + v_k  (v ~ N(0,R))
    Ref: Kalman (1960), also Labbe "Kalman and Bayesian Filters in Python" ch.4
    Returns: (pos_new_nm, P_new, K)
    """
    P_pred = P + Q
    K      = P_pred / (P_pred + R)
    pos_new = pos_nm + K * (z_nm - pos_nm)
    P_new   = (1.0 - K) * P_pred
    return float(pos_new), float(P_new), float(K)


def load_kf_state(path: str, R_nm2: float) -> dict:
    """drift_kf_state.json を読む。なければ pos-only 初期値を返す。"""
    default = {"kf_pos_tx_nm": 0.0, "kf_P_tx": R_nm2,
               "kf_pos_ty_nm": 0.0, "kf_P_ty": R_nm2}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        # 旧 pos+vel フォーマット互換: P が 2x2 リストの場合は [0][0] だけ使う
        for key in ("kf_P_tx", "kf_P_ty"):
            if key in d and isinstance(d[key], list):
                d[key] = float(np.array(d[key])[0, 0])
        return {k: d.get(k, default[k]) for k in default}
    except Exception:
        return default


def save_kf_state(path: str, pos_tx: float, P_tx: float,
                  pos_ty: float, P_ty: float) -> None:
    """drift_kf_state.json を書き込む（pos-only）。"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"kf_pos_tx_nm": pos_tx, "kf_P_tx": float(P_tx),
                   "kf_pos_ty_nm": pos_ty, "kf_P_ty": float(P_ty)},
                  f, indent=2)


def compute_backsub_offset(img, cfg) -> float:
    min_phase    = cfg.get("backsub_min_phase", -1.1)
    hist_min     = cfg.get("backsub_hist_min", -1.1)
    hist_max     = cfg.get("backsub_hist_max",  1.5)
    n_bins       = cfg.get("backsub_n_bins", 512)
    smooth_w     = cfg.get("backsub_smooth_window", 20)

    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(img.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    smoothed = uniform_filter1d(hist_counts.astype(float), size=smooth_w, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=smooth_w, mode='nearest')
    valid_idx = np.where(bin_centers >= min_phase)[0]
    search_idx = valid_idx[valid_idx < int(len(bin_centers) * 0.95)]
    if len(search_idx) == 0:
        return 0.0
    peak_idx = search_idx[np.argmax(smoothed[search_idx])]
    peak_value = bin_centers[peak_idx]
    s = max(0, peak_idx - 300)
    e = min(len(bin_centers), peak_idx + 300)
    try:
        popt, _ = curve_fit(
            lambda x, a, m, s_: a * np.exp(-((x - m)**2) / (2 * s_**2)),
            bin_centers[s:e], smoothed[s:e],
            p0=[float(np.max(smoothed[s:e])), peak_value, bin_width * 20],
            maxfev=5000
        )
        return float(-popt[1])
    except Exception:
        return float(-peak_value)


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


def _tilt_correct(img_f64, cy, cx, crop_w, tilt_crop_h, ecc_crop_h, fit_right: bool = False):
    """
    フル位相画像から tilt_crop_h px 幅の crop を取り、
    背景側1/3 で slope+intercept fit → 補正 → 中央 ecc_crop_h px を返す。
    fit_right=False: 左1/3、True: 右1/3（pos_split で決定）。
    ゼロパディングが発生する場合は tilt 補正をスキップして simple crop を返す。
    """
    h, w  = img_f64.shape
    if (cx - tilt_crop_h // 2) < 0 or (cx + tilt_crop_h // 2) > w:
        return extract_rect_roi(img_f64, cy, cx, crop_w, ecc_crop_h).astype(np.float64)
    big   = extract_rect_roi(img_f64, cy, cx, crop_w, tilt_crop_h).astype(np.float64)
    x     = np.arange(tilt_crop_h, dtype=np.float64)
    prof  = big.mean(axis=0)
    fit_n = max(1, tilt_crop_h // 3)
    if fit_right:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    corrected = big - (a * x + b)[np.newaxis, :]
    start = (tilt_crop_h - ecc_crop_h) // 2
    return corrected[:, start : start + ecc_crop_h]


def to_uint8(img, vmin, vmax):
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    """ECC アライメント。(tx, ty, correlation) を返す。失敗時は None。
    v2: max_iter=100000, epsilon=1e-8（v1 より高精度）
    """
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


def _mad(arr):
    m = np.median(arr)
    return float(np.median(np.abs(arr - m)))


def _remove_outliers_mad(values, thresh=5.0):
    arr = np.array(values, dtype=np.float64)
    md = _mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - np.median(arr)) > thresh * md


# ---- 2-pass ECC 用グリッドユーティリティ ----

def scan_grid_positions(grid_dir, base_label):
    """(xi, yi) → folder_path のマップを返す。"""
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


def _find_nearest_grid(pos_map, dx_um, dy_um, x_step, y_step):
    """(dx_um, dy_um) に最近傍の (xi, yi) を返す。"""
    best_key, best_dist = None, float('inf')
    for (xi, yi) in pos_map:
        dist = ((xi * x_step - dx_um) ** 2 + (yi * y_step - dy_um) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_key = (xi, yi)
    return best_key


def _select_nearest_grid(shift_x, shift_y, pos_map, sx_sign, sy_sign, x_step, y_step, pixel_scale_um,
                         grid_cal=None):
    """shift_x/y [画像px] から最近傍 (xi, yi) を返す。
    grid_cal が指定された場合は実測オフセット（px）で最近傍を選ぶ。
    なければ名目値（ステージ μm → px 換算）にフォールバック。
    """
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
    dx_um = sx_sign * shift_y * pixel_scale_um  # 画像Y → ステージX
    dy_um = sy_sign * shift_x * pixel_scale_um  # 画像X → ステージY
    return _find_nearest_grid(pos_map, dx_um, dy_um, x_step, y_step)


def _get_grid_offset_px(xi, yi, sx_sign, sy_sign, x_step, y_step, pixel_scale_um,
                        grid_cal=None):
    """grid(xi,yi) の grid(0,0) 基準オフセット [画像px] を返す (offset_x, offset_y)。
    grid_cal が指定された場合は実測値を返す。なければ名目値にフォールバック。
    """
    if grid_cal and (xi, yi) in grid_cal:
        return grid_cal[(xi, yi)]   # (cal_dx, cal_dy) = (offset_x, offset_y)
    offset_x = sy_sign * yi * y_step / pixel_scale_um  # ステージY → 画像X
    offset_y = sx_sign * xi * x_step / pixel_scale_um  # ステージX → 画像Y
    return offset_x, offset_y


def _load_grid_ref_full(pos_map, xi, yi, rois, n_channels, z_index, cfg,
                        tilt_crop_h=0, ecc_crop_h=0, fit_right=False):
    """grid(xi,yi) の full-crop 参照を返す（pass 2/3 用）。
    tilt_crop_h > 0 かつ ecc_crop_h > 0 のとき tilt 補正を適用（backsub は不要）。
    それ以外は従来の backsub を適用。
    """
    pos_dir = pos_map[(xi, yi)]
    fname = f"img_000000000_ph_{z_index:03d}_phase.tif"
    path = pos_dir / "output_phase" / fname
    if not path.exists():
        raise FileNotFoundError(f"グリッド画像が見つかりません: {path}")
    grid_img = tifffile.imread(str(path)).astype(np.float64)
    use_tilt = tilt_crop_h > 0 and ecc_crop_h > 0
    refs_out = []
    for ch in range(n_channels):
        roi = rois[ch] if ch < len(rois) else rois[-1]
        if use_tilt:
            cropped = _tilt_correct(grid_img, roi["cy"], roi["cx"], roi["crop_w"],
                                    tilt_crop_h, ecc_crop_h, fit_right=fit_right)
        else:
            cropped = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
            offset = compute_backsub_offset(cropped, cfg)
            cropped = cropped + offset
        refs_out.append(cropped)
    return refs_out


# ---- 位相再構成 ----

def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None) -> np.ndarray:
    """QPI 位相再構成。bg_path があれば差分を返す。"""
    script_dir = Path(cfg["script_dir"])
    sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    ref_idx = cfg["ref_pos_index"]
    pos_split = cfg.get("pos_split", 3)
    if ref_idx < pos_split:
        crop = tuple(cfg["crop_before"])
    else:
        crop = tuple(cfg["crop_after"])
    rs, re_, cs, ce = crop

    def _recon(path):
        img = np.array(Image.open(str(path)))
        img_crop = img[rs:re_, cs:ce]
        qpi_params = QPIParameters(
            wavelength=WAVELENGTH, NA=NA,
            img_shape=img_crop.shape, pixelsize=PIXELSIZE,
            offaxis_center=OFFAXIS_CENTER,
        )
        field = get_field(img_crop, qpi_params)
        return unwrap_phase(np.angle(field))

    phase = _recon(raw_path)
    if bg_path is not None and bg_path.exists():
        phase = phase - _recon(bg_path)
        print(f"  BG subtraction: {bg_path.name}")
    else:
        print("  No BG (backsub only)")

    return phase


# ---- メイン ----

def main():
    args = parse_args()
    cfg = load_config(args.config)

    t          = args.timepoint
    sample_raw = Path(args.sample_raw)
    bg_raw     = Path(args.bg_raw) if args.bg_raw.lower() != "none" else None
    state_path = cfg["state_file"]
    log_path   = cfg["log_file"]
    prev_crops_path = Path(cfg["prev_frame_crops_tif"])
    ref_crops_path  = Path(cfg["grid_ref_crops_tif"])
    rois_path       = Path(cfg["channel_rois_json"])

    vmin           = cfg.get("ecc_vmin", -5.0)
    vmax           = cfg.get("ecc_vmax",  2.0)
    jump_thresh    = cfg.get("jump_thresh_um", 1.0)
    max_total      = cfg.get("max_total_corr_um", 15.0)
    pixel_scale_um = cfg.get("pixel_scale_um", 0.3462)
    sx_sign        = cfg.get("shift_sign_x", 1)
    sy_sign        = cfg.get("shift_sign_y", 1)
    x_step         = cfg.get("x_step_um", 0.1)
    y_step         = cfg.get("y_step_um", 0.1)
    second_pass_half = cfg.get("second_pass_half", "right")
    grid_dir       = cfg.get("grid_dir", None)
    grid_base_label = cfg.get("grid_base_label", None)
    grid_z_index   = cfg.get("grid_z_index", 0)
    enable_third_pass = cfg.get("enable_third_pass", True)
    tilt_crop_h    = cfg.get("tilt_crop_h", 0)
    ecc_crop_h     = cfg.get("ecc_crop_h", 0)
    use_tilt       = tilt_crop_h > 0 and ecc_crop_h > 0
    pos_split      = cfg.get("pos_split", 3)
    ref_pos_index  = cfg.get("ref_pos_index", 0)
    tilt_fit_right = ref_pos_index >= pos_split

    # ---- grid calibration（実測 px オフセット辞書）----
    # {(xi, yi): (cal_dx_px, cal_dy_px)}  cal = +tx 規約（shift_x に直接加算できる）
    grid_cal = {}
    grid_cal_path = cfg.get("grid_calibration_json", None)
    if grid_cal_path and Path(grid_cal_path).exists():
        try:
            with open(grid_cal_path, encoding="utf-8") as f:
                _cal_data = json.load(f)
            for entry in _cal_data.get("positions", []):
                grid_cal[(entry["xi"], entry["yi"])] = (
                    -entry["actual_dx_px"],   # actual_dx = -tx → cal = +tx
                    -entry["actual_dy_px"],
                )
            print(f"  grid_calibration: {len(grid_cal)} positions loaded from {Path(grid_cal_path).name}")
        except Exception as _ex:
            print(f"  [WARNING] grid_calibration_json 読み込み失敗: {_ex} → 名目値にフォールバック")
            grid_cal = {}
    elif grid_cal_path:
        print(f"  [WARNING] grid_calibration_json が見つかりません: {grid_cal_path} → 名目値にフォールバック")
    else:
        print("  [INFO] grid_calibration_json 未設定 → 名目値で grid 選択")

    print(f"[T={t}] compute_drift_online_v2.py start  ({'3' if enable_third_pass else '2'}-pass ECC)")
    print(f"  sample: {sample_raw.name}")
    print(f"  bg:     {bg_raw.name if bg_raw else 'none'}")

    # ---- 入力ファイルの確認 ----
    if not sample_raw.exists():
        print(f"ERROR: sample-raw が見つかりません: {sample_raw}")
        write_state(state_path, t, 0.0, 0.0, 0.0, 0.0, False, 0.0, False)
        sys.exit(1)

    if not ref_crops_path.exists():
        print(f"ERROR: grid_ref_crops.tif が見つかりません: {ref_crops_path}")
        write_state(state_path, t, 0.0, 0.0, 0.0, 0.0, False, 0.0, False)
        sys.exit(1)

    # ---- channel_rois.json ----
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)

    # ---- grid(0,0) 参照 crops（pass 1 用）----
    grid_ref_crops = tifffile.imread(str(ref_crops_path)).astype(np.float64)
    if grid_ref_crops.ndim == 2:
        grid_ref_crops = grid_ref_crops[np.newaxis, ...]
    grid_ref_crops_u8 = [to_uint8(grid_ref_crops[ch], vmin, vmax)
                         for ch in range(min(n_channels, len(grid_ref_crops)))]
    print(f"  Pass 1: grid(0,0) full-crop reference ({n_channels} channels)")

    # ---- グリッドポジションマップ（pass 2 用）----
    use_second_pass = False
    pos_map = {}
    if grid_dir and grid_base_label:
        try:
            pos_map = scan_grid_positions(grid_dir, grid_base_label)
            if pos_map:
                use_second_pass = True
                print(f"  Pass 2: {len(pos_map)} grid positions loaded, full-crop")
            else:
                print("  [WARNING] グリッドPosが見つかりません → 1-pass にフォールバック")
        except Exception as ex:
            print(f"  [WARNING] グリッドスキャン失敗: {ex} → 1-pass にフォールバック")
    else:
        print("  [INFO] grid_dir/grid_base_label 未設定 → 1-pass ECC")

    # ---- QPI 位相再構成 ----
    print("  Phase reconstruction...")
    try:
        phase = reconstruct_phase(sample_raw, cfg, bg_raw)
    except Exception as ex:
        print(f"ERROR: Phase reconstruction failed: {ex}")
        import traceback; traceback.print_exc()
        write_state(state_path, t, 0.0, 0.0, 0.0, 0.0, False, 0.0, False)
        sys.exit(1)

    # 平均除去
    h_p, w_p = phase.shape
    region = phase[1:h_p-1, 1:w_p//2]
    if region.size > 0:
        phase -= np.mean(region)

    # Gaussian 空間グラジェント除去
    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        from scipy.ndimage import gaussian_filter
        bg = gaussian_filter(phase, sigma=gradient_sigma, mode='nearest')
        phase = phase - bg
        print(f"  Gaussian gradient removal: sigma={gradient_sigma}px")

    # 再構成済み位相を保存
    out_phase_dir = sample_raw.parent / "output_phase"
    out_phase_dir.mkdir(exist_ok=True)
    out_phase_path = out_phase_dir / (sample_raw.stem + "_phase.tif")
    tifffile.imwrite(str(out_phase_path), phase.astype(np.float32))
    print(f"  Phase saved: {out_phase_path.name}")

    # ---- 各チャネルの full-crop（tilt 補正 or backsub）----
    current_crops = []
    for ch_idx, roi in enumerate(rois):
        if use_tilt:
            crop = _tilt_correct(phase, roi["cy"], roi["cx"], roi["crop_w"],
                                 tilt_crop_h, ecc_crop_h, fit_right=tilt_fit_right)
        else:
            crop = extract_rect_roi(phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
            offset = compute_backsub_offset(crop, cfg)
            crop = crop + offset
        current_crops.append(crop)

    # ---- 2-pass ECC（チャネル並列）----
    # キャッシュはスレッド間で共有するためロックを使う
    grid_half_cache    = {}   # (xi, yi) → list of float32 full-crops (pass 2)
    grid_half_u8_cache = {}   # (xi, yi) → list of uint8 full-crops (pass 2)
    _cache_lock = threading.Lock()

    def _align_ch(ch_idx):
        ref_u8_p1 = grid_ref_crops_u8[ch_idx] if ch_idx < len(grid_ref_crops_u8) else grid_ref_crops_u8[-1]
        cur_crop   = current_crops[ch_idx]
        roi        = rois[ch_idx] if ch_idx < len(rois) else rois[-1]

        # ---- Pass 1: grid(0,0) full-crop ECC ----
        result1 = ecc_align(ref_u8_p1, to_uint8(cur_crop, vmin, vmax))
        if result1 is None:
            return ch_idx, None, "pass1_failed", {}

        fine1_x, fine1_y, corr1 = result1
        # Pass 1 offset は (0,0) なので shift1 = fine1
        shift1_x, shift1_y = fine1_x, fine1_y

        if not use_second_pass:
            return ch_idx, (shift1_x, shift1_y, corr1), "pass1_only", {
                "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                "xi": 0, "yi": 0,
                "tx2": shift1_x, "ty2": shift1_y, "corr2": corr1,
            }

        # ---- Pass 2: nearest grid half-crop ECC ----
        xi2, yi2 = _select_nearest_grid(
            shift1_x, shift1_y, pos_map,
            sx_sign, sy_sign, x_step, y_step, pixel_scale_um,
            grid_cal=grid_cal
        )
        offset_x2, offset_y2 = _get_grid_offset_px(
            xi2, yi2, sx_sign, sy_sign, x_step, y_step, pixel_scale_um,
            grid_cal=grid_cal
        )

        # full-crop をロード（キャッシュ利用）
        with _cache_lock:
            if (xi2, yi2) not in grid_half_cache:
                try:
                    halves = _load_grid_ref_full(
                        pos_map, xi2, yi2, rois, n_channels,
                        grid_z_index, cfg,
                        tilt_crop_h=tilt_crop_h, ecc_crop_h=ecc_crop_h,
                        fit_right=tilt_fit_right
                    )
                    grid_half_cache[(xi2, yi2)]    = halves
                    grid_half_u8_cache[(xi2, yi2)] = [to_uint8(h, vmin, vmax) for h in halves]
                    print(f"    [grid cache] ({xi2:+d},{yi2:+d}) loaded")
                except FileNotFoundError as ex:
                    print(f"    [WARNING] ch{ch_idx:02d} half-crop load failed: {ex} → pass1 result")
                    return ch_idx, (shift1_x, shift1_y, corr1), "pass2_load_failed", {
                        "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                        "xi": xi2, "yi": yi2,
                        "tx2": shift1_x, "ty2": shift1_y, "corr2": corr1,
                    }

        ref_u8_p2 = grid_half_u8_cache[(xi2, yi2)][ch_idx]
        result2 = ecc_align(ref_u8_p2, to_uint8(cur_crop, vmin, vmax))

        if result2 is None:
            # Pass 2 失敗 → Pass 1 結果を採用
            print(f"    ch{ch_idx:02d}: Pass2 ECC failed → using pass1  "
                  f"shift=({shift1_x:+.3f},{shift1_y:+.3f})px  corr1={corr1:.4f}")
            return ch_idx, (shift1_x, shift1_y, corr1), "pass2_ecc_failed", {
                "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                "xi": xi2, "yi": yi2,
                "tx2": shift1_x, "ty2": shift1_y, "corr2": corr1,
            }

        fine2_x, fine2_y, corr2 = result2
        # grid(xi2,yi2) からのfineシフト + grid(0,0)基準オフセット → 絶対シフト
        shift2_x = fine2_x + offset_x2
        shift2_y = fine2_y + offset_y2

        # ---- Pass 3: shift2 から再グリッド選択 ----
        if enable_third_pass:
            xi3, yi3 = _select_nearest_grid(
                shift2_x, shift2_y, pos_map,
                sx_sign, sy_sign, x_step, y_step, pixel_scale_um,
                grid_cal=grid_cal
            )

            if (xi3, yi3) == (xi2, yi2):
                print(f"    ch{ch_idx:02d}: "
                      f"pass1=({shift1_x:+.3f},{shift1_y:+.3f})px corr1={corr1:.4f}  "
                      f"grid2=({xi2:+d},{yi2:+d})  "
                      f"pass2=({shift2_x:+.3f},{shift2_y:+.3f})px corr2={corr2:.4f}  "
                      f"[pass3=skip]")
                return ch_idx, (shift2_x, shift2_y, corr2), "pass2_ok", {
                    "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                    "xi": xi2, "yi": yi2,
                    "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2,
                    "xi3": xi3, "yi3": yi3,
                    "tx3": shift2_x, "ty3": shift2_y, "corr3": corr2,
                }

            offset_x3, offset_y3 = _get_grid_offset_px(
                xi3, yi3, sx_sign, sy_sign, x_step, y_step, pixel_scale_um,
                grid_cal=grid_cal
            )

            with _cache_lock:
                if (xi3, yi3) not in grid_half_cache:
                    try:
                        halves3 = _load_grid_ref_full(
                            pos_map, xi3, yi3, rois, n_channels,
                            grid_z_index, cfg,
                            tilt_crop_h=tilt_crop_h, ecc_crop_h=ecc_crop_h,
                            fit_right=tilt_fit_right
                        )
                        grid_half_cache[(xi3, yi3)]    = halves3
                        grid_half_u8_cache[(xi3, yi3)] = [to_uint8(h, vmin, vmax) for h in halves3]
                        print(f"    [grid cache] ({xi3:+d},{yi3:+d}) loaded")
                    except FileNotFoundError as ex:
                        print(f"    [WARNING] ch{ch_idx:02d} pass3 load failed: {ex} → pass2 result")
                        print(f"    ch{ch_idx:02d}: "
                              f"pass1=({shift1_x:+.3f},{shift1_y:+.3f})px corr1={corr1:.4f}  "
                              f"grid2=({xi2:+d},{yi2:+d})  "
                              f"pass2=({shift2_x:+.3f},{shift2_y:+.3f})px corr2={corr2:.4f}  "
                              f"[pass3=load_failed]")
                        return ch_idx, (shift2_x, shift2_y, corr2), "pass3_load_failed", {
                            "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                            "xi": xi2, "yi": yi2,
                            "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2,
                            "xi3": xi3, "yi3": yi3,
                            "tx3": shift2_x, "ty3": shift2_y, "corr3": corr2,
                        }

            ref_u8_p3 = grid_half_u8_cache[(xi3, yi3)][ch_idx]
            result3 = ecc_align(ref_u8_p3, to_uint8(cur_crop, vmin, vmax))

            if result3 is None:
                print(f"    ch{ch_idx:02d}: Pass3 ECC failed → using pass2  "
                      f"shift=({shift2_x:+.3f},{shift2_y:+.3f})px  corr2={corr2:.4f}")
                return ch_idx, (shift2_x, shift2_y, corr2), "pass3_ecc_failed", {
                    "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                    "xi": xi2, "yi": yi2,
                    "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2,
                    "xi3": xi3, "yi3": yi3,
                    "tx3": shift2_x, "ty3": shift2_y, "corr3": corr2,
                }

            fine3_x, fine3_y, corr3 = result3
            shift3_x = fine3_x + offset_x3
            shift3_y = fine3_y + offset_y3

            print(f"    ch{ch_idx:02d}: "
                  f"pass1=({shift1_x:+.3f},{shift1_y:+.3f})px corr1={corr1:.4f}  "
                  f"grid2=({xi2:+d},{yi2:+d})  "
                  f"pass2=({shift2_x:+.3f},{shift2_y:+.3f})px corr2={corr2:.4f}  "
                  f"grid3=({xi3:+d},{yi3:+d})  "
                  f"pass3=({shift3_x:+.3f},{shift3_y:+.3f})px corr3={corr3:.4f}")
            return ch_idx, (shift3_x, shift3_y, corr3), "pass3_ok", {
                "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
                "xi": xi2, "yi": yi2,
                "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2,
                "xi3": xi3, "yi3": yi3,
                "tx3": shift3_x, "ty3": shift3_y, "corr3": corr3,
            }

        # enable_third_pass=False → pass2 をそのまま使用
        print(f"    ch{ch_idx:02d}: "
              f"pass1=({shift1_x:+.3f},{shift1_y:+.3f})px corr1={corr1:.4f}  "
              f"grid=({xi2:+d},{yi2:+d})  "
              f"pass2=({shift2_x:+.3f},{shift2_y:+.3f})px corr2={corr2:.4f}")
        return ch_idx, (shift2_x, shift2_y, corr2), "pass2_ok", {
            "tx1": shift1_x, "ty1": shift1_y, "corr1": corr1,
            "xi": xi2, "yi": yi2,
            "tx2": shift2_x, "ty2": shift2_y, "corr2": corr2,
        }

    n_ch = min(n_channels, len(current_crops))
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_ch) as ex:
        ch_results = list(ex.map(_align_ch, range(n_ch)))

    tx_list, ty_list, corr_list = [], [], []
    valid_ch_indices = []  # tx_list の各要素に対応する ch_idx
    ch_detail_map = {}     # ch_idx -> detail dict
    for ch_idx, result, status, detail in ch_results:
        ch_detail_map[ch_idx] = dict(detail, status=status)
        if result is not None:
            tx, ty, corr = result
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)
            valid_ch_indices.append(ch_idx)
        else:
            print(f"    ch{ch_idx:02d}: ECC failed ({status})")

    if not tx_list:
        print("ERROR: ECC failed on all channels")
        prev_state = read_state(state_path)
        write_state(state_path, t, 0.0, 0.0,
                    prev_state["cumulative_dx_um"], prev_state["cumulative_dy_um"],
                    False, 0.0, False)
        sys.exit(0)

    # チャネル平均（MAD 外れ値除去）
    n_ch_raw = len(tx_list)
    if n_ch_raw >= 3:
        out_x = _remove_outliers_mad(tx_list)
        out_y = _remove_outliers_mad(ty_list)
        is_out = out_x | out_y
        used_idx = [i for i, o in enumerate(is_out) if not o]
        if len(used_idx) == 0:
            used_idx = list(range(n_ch_raw))
        excl = [i for i in range(n_ch_raw) if is_out[i]]
        if excl:
            print(f"  [外れ値除去] idx={excl}: {len(excl)}ch除外")
    else:
        used_idx = list(range(n_ch_raw))

    tx_arr   = np.array(tx_list)
    ty_arr   = np.array(ty_list)
    corr_arr = np.array(corr_list)
    tx_avg   = float(np.mean(tx_arr[used_idx]))
    ty_avg   = float(np.mean(ty_arr[used_idx]))
    corr_avg = float(np.mean(corr_arr[used_idx]))

    # ---- EMA フィルタ（correction_ema_alpha < 1.0 で有効）----
    prev_state = read_state(state_path)
    ema_alpha  = cfg.get("correction_ema_alpha", 1.0)
    prev_ema_tx = prev_state["ema_tx_px"]
    prev_ema_ty = prev_state["ema_ty_px"]
    if prev_ema_tx is None:                        # 初回フレーム: フィルタなし
        tx_filt, ty_filt = tx_avg, ty_avg
    else:
        tx_filt = ema_alpha * tx_avg + (1.0 - ema_alpha) * prev_ema_tx
        ty_filt = ema_alpha * ty_avg + (1.0 - ema_alpha) * prev_ema_ty
    if ema_alpha < 1.0:
        print(f"  EMA(α={ema_alpha}): tx={tx_filt:+.4f}px  ty={ty_filt:+.4f}px")

    # ---- Kalman フィルタ（pos-only random walk、use_kalman_filter: true で有効）----
    # 方向別 Q/R 設定対応: kf_R_ty_nm2 (image X ECC) / kf_R_tx_nm2 (image Y ECC)
    # analyze_stage_repeatability 実測値より:
    #   R_ty = (33.1/√12)² ≈  91 nm²  (image X, K_ss ≈ 0.95)
    #   R_tx = (57.3/√12)² ≈ 274 nm²  (image Y, K_ss ≈ 0.87)
    #   Q    = σ_stage²    ≈ 1650 nm²  (stage repositioning noise が主因)
    # Q >> R のため高ゲイン設計が正しい（旧 R=12100 は 130× 過大だった）
    use_kalman          = cfg.get("use_kalman_filter", False)
    kf_K_tx             = kf_K_ty = kf_P_tx = kf_P_ty = None
    kf_innov_tx         = kf_innov_ty = None
    kf_pos_ty_nm_new    = kf_pos_tx_nm_new = None
    if use_kalman:
        kf_Q        = cfg.get("kf_Q_pos_nm2", 548.0)
        kf_R        = cfg.get("kf_R_nm2",     454.0)
        # 方向別 Q/R（未指定時は kf_Q / kf_R にフォールバック）
        # ty state は tx_avg (image X 由来), tx state は ty_avg (image Y 由来)
        kf_Q_ty     = cfg.get("kf_Q_ty_nm2",  kf_Q)
        kf_Q_tx     = cfg.get("kf_Q_tx_nm2",  kf_Q)
        kf_R_ty     = cfg.get("kf_R_ty_nm2",  kf_R)
        kf_R_tx     = cfg.get("kf_R_tx_nm2",  kf_R)
        kf_file     = cfg.get("kf_state_file",
                               str(Path(state_path).parent / "drift_kf_state.json"))
        px_scale_nm = pixel_scale_um * 1000.0
        kf_state    = load_kf_state(kf_file, max(kf_R_ty, kf_R_tx))

        # open-loop 再構成位置 [nm]: 累積補正 + 現フレーム ECC 残差
        z_ty_nm      = tx_avg * px_scale_nm * sy_sign
        z_tx_nm      = ty_avg * px_scale_nm * sx_sign
        ol_pos_ty_nm = prev_state["cumulative_dy_um"] * 1000.0 + z_ty_nm
        ol_pos_tx_nm = prev_state["cumulative_dx_um"] * 1000.0 + z_tx_nm

        # イノベーション（update 前の予測誤差）
        kf_innov_ty = abs(ol_pos_ty_nm - kf_state["kf_pos_ty_nm"])
        kf_innov_tx = abs(ol_pos_tx_nm - kf_state["kf_pos_tx_nm"])

        # pos-only random walk Kalman update（方向別 Q/R）
        pos_ty_new, P_ty_new, K_ty = kf_step_posonly_nm(
            ol_pos_ty_nm, kf_state["kf_pos_ty_nm"], kf_state["kf_P_ty"], kf_Q_ty, kf_R_ty)
        pos_tx_new, P_tx_new, K_tx = kf_step_posonly_nm(
            ol_pos_tx_nm, kf_state["kf_pos_tx_nm"], kf_state["kf_P_tx"], kf_Q_tx, kf_R_tx)

        save_kf_state(kf_file, pos_tx_new, P_tx_new, pos_ty_new, P_ty_new)

        kf_pos_ty_nm_new = pos_ty_new
        kf_pos_tx_nm_new = pos_tx_new
        kf_P_ty          = P_ty_new
        kf_P_tx          = P_tx_new
        kf_K_ty          = K_ty
        kf_K_tx          = K_tx
        print(f"  KF pos-only: "
              f"ol_ty={ol_pos_ty_nm:.0f}nm pos={pos_ty_new:.0f}nm K={K_ty:.3f}(R={kf_R_ty:.0f})  "
              f"ol_tx={ol_pos_tx_nm:.0f}nm pos={pos_tx_new:.0f}nm K={K_tx:.3f}(R={kf_R_tx:.0f})")

    # channel_details: valid_ch_indices の順序で outlier フラグを付与
    is_out_arr = is_out if n_ch_raw >= 3 else np.zeros(n_ch_raw, dtype=bool)
    channel_details = []
    for list_idx, ch_idx in enumerate(valid_ch_indices):
        d = ch_detail_map.get(ch_idx, {})
        channel_details.append({
            "ch": ch_idx,
            "tx1": round(d.get("tx1", 0.0), 6), "ty1": round(d.get("ty1", 0.0), 6),
            "corr1": round(d.get("corr1", 0.0), 6),
            "xi": d.get("xi", 0), "yi": d.get("yi", 0),
            "tx2": round(d.get("tx2", 0.0), 6), "ty2": round(d.get("ty2", 0.0), 6),
            "corr2": round(d.get("corr2", 0.0), 6),
            "outlier": bool(is_out_arr[list_idx]),
            "status": d.get("status", ""),
        })
    # ECC failed チャネルも記録
    for ch_idx in sorted(set(ch_detail_map) - set(valid_ch_indices)):
        d = ch_detail_map[ch_idx]
        channel_details.append({"ch": ch_idx, "outlier": True, "status": d.get("status", "failed")})
    print(f"  ECC平均: tx={tx_avg:+.4f}px  ty={ty_avg:+.4f}px  corr={corr_avg:.4f}  "
          f"(使用{len(used_idx)}/{n_ch_raw}ch)")

    # ---- 符号・スケール変換（pixel → μm、画像軸 → ステージ軸）----
    shift_x = tx_filt  # 画像X方向ずれ [px]（EMAフィルタ済み）
    shift_y = ty_filt  # 画像Y方向ずれ [px]（EMAフィルタ済み）

    correction_stage_x_um = sx_sign * shift_y * pixel_scale_um  # ステージX方向補正 [μm]
    correction_stage_y_um = sy_sign * shift_x * pixel_scale_um  # ステージY方向補正 [μm]

    # pos-only KF: 推定絶対位置から incremental 補正に変換
    if use_kalman and kf_pos_ty_nm_new is not None:
        correction_stage_y_um = kf_pos_ty_nm_new / 1000.0 - prev_state["cumulative_dy_um"]
        correction_stage_x_um = kf_pos_tx_nm_new / 1000.0 - prev_state["cumulative_dx_um"]

    # ---- 累積ドリフト計算 ----
    cum_dx = prev_state["cumulative_dx_um"] + correction_stage_x_um
    cum_dy = prev_state["cumulative_dy_um"] + correction_stage_y_um

    print(f"  Correction: stage_x={correction_stage_x_um:+.4f}um  stage_y={correction_stage_y_um:+.4f}um")
    print(f"  Cumulative: stage_x={cum_dx:+.4f}um  stage_y={cum_dy:+.4f}um")

    # ---- ジャンプ検出 ----
    step_um  = (correction_stage_x_um**2 + correction_stage_y_um**2) ** 0.5
    total_um = (cum_dx**2 + cum_dy**2) ** 0.5
    if jump_thresh is None:
        jump = total_um > max_total
    else:
        jump = (step_um > jump_thresh) or (total_um > max_total)

    if jump:
        thresh_str = "off" if jump_thresh is None else f"{jump_thresh}um"
        print(f"  [JUMP] step={step_um:.3f}um (thresh={thresh_str}), "
              f"total={total_um:.3f}um (max={max_total}um) -> skipped")
        write_state(state_path, t,
                    correction_stage_x_um, correction_stage_y_um,
                    prev_state["cumulative_dx_um"], prev_state["cumulative_dy_um"],
                    False, corr_avg, True,
                    ema_tx=tx_filt, ema_ty=ty_filt)
    else:
        # prev_frame_crops.tif を更新（互換性維持）
        current_crops_arr = np.stack([c.astype(np.float32) for c in current_crops], axis=0)
        tifffile.imwrite(str(prev_crops_path), current_crops_arr)
        write_state(state_path, t,
                    correction_stage_x_um, correction_stage_y_um,
                    cum_dx, cum_dy,
                    True, corr_avg, False,
                    ema_tx=tx_filt, ema_ty=ty_filt)
        print(f"  [OK] Correction valid")

    # ---- ログ追記 ----
    log_entry = {
        "timepoint":             t,
        "timestamp":             datetime.now().isoformat(),
        "sample_raw":            str(sample_raw),
        "bg_raw":                str(bg_raw) if bg_raw else None,
        "used_prev_frame":       False,
        "n_channels_used":       len(used_idx),
        "n_channels_raw":        n_ch_raw,
        "tx_avg_px":             tx_avg,
        "ty_avg_px":             ty_avg,
        "ecc_correlation":       corr_avg,
        "correction_stage_x_um": correction_stage_x_um,
        "correction_stage_y_um": correction_stage_y_um,
        "cumulative_dx_um":      cum_dx,
        "cumulative_dy_um":      cum_dy,
        "jump_detected":         jump,
        "correction_valid":      not jump,
        "two_pass_ecc":          use_second_pass,
        "tx_filt_px":            round(tx_filt, 6),
        "ty_filt_px":            round(ty_filt, 6),
        "kf_active":             use_kalman,
        "kf_K_tx":               round(kf_K_tx, 4) if kf_K_tx is not None else None,
        "kf_K_ty":               round(kf_K_ty, 4) if kf_K_ty is not None else None,
        "kf_P_tx_nm2":           round(kf_P_tx, 2) if kf_P_tx is not None else None,
        "kf_P_ty_nm2":           round(kf_P_ty, 2) if kf_P_ty is not None else None,
        "kf_innovation_tx_nm":   round(kf_innov_tx, 2) if kf_innov_tx is not None else None,
        "kf_innovation_ty_nm":   round(kf_innov_ty, 2) if kf_innov_ty is not None else None,
        "channel_details":       sorted(channel_details, key=lambda x: x["ch"]),
    }
    try:
        with open(log_path, encoding="utf-8") as f:
            log = json.load(f)
    except Exception:
        log = []
    log.append(log_entry)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)

    print(f"[T={t}] done  status: {'corrected' if not jump else 'skipped'}")


if __name__ == "__main__":
    main()
