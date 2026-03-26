"""
prepare_drift_session.py
------------------------
リアルタイムドリフト補正セッションの事前準備スクリプト。
realtime_drift_mda.bsh を実行する前に1回だけ実行する。

何をするか:
  1. .pos ファイル → positions.csv（Beanshell 用の簡易フォーマット）
  2. grid(0,0) の再構成済み位相画像 → チャネルROIでcrop + backsub → grid_ref_crops.tif
  3. drift_config.json（全パラメータを集約）
  4. drift_state.txt 初期化

使い方:
  python prepare_drift_session.py
"""

import sys
import json
import numpy as np
import tifffile
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.optimize import curve_fit

_script_dir = Path(__file__).parent
sys.path.insert(0, str(_script_dir))

# ============================================================
# ★★★ 実験ごとにここを変更 ★★★
# ============================================================

# タイムラプスで使う .pos ファイル（グリッドなし、実際に撮影するポジションリスト）
POSITIONS_FILE   = r"D:\AquisitionData\Kitagishi\260321\focused_timelapse.pos"

# グリッド撮影ディレクトリ（小規模グリッドでよい）
GRID_DIR         = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1"
GRID_BASE_LABEL  = "Pos1"   # ドリフト推定に使う Pos のラベル（タイムラプスと同じ）
GRID_Z_INDEX     = 0        # グリッド画像の z インデックス（単z → 0）

# channel_rois.json（事前に pipeline_full.py などで生成済みのもの）
CHANNEL_ROIS_JSON = r"D:\AquisitionData\Kitagishi\260321\grid_2pergluc_60ms_1\Pos1_x+0_y+0\output_phase\channels\channel_rois.json"

# セッション作業ディレクトリ（ここに設定ファイルと状態ファイルが保存される）
SESSION_DIR      = r"C:\Users\QPI\Documents\QPI_Omni\drift_session"

# タイムラプス画像の保存先（MM1.4 で撮影した画像の保存先）
SAVE_DIR         = r"C:\ph"

# .pos ファイル内の各 Pos のインデックス（0始まり）
REF_POS_INDEX    = 1   # ドリフト推定に使う Pos（サンプルがいる Pos）
BG_POS_INDEX     = 0   # BG Pos（Pos0: 細胞なし、位相補正用）

# MM1.4 撮影パラメータ
N_TIMEPOINTS     = 2304     # 260321実験
INTERVAL_SEC     = 300       # タイムポイント間隔 [秒]（5分）
EXPOSURE_MS      = 60.0      # カメラ露光時間 [ms]
SETTLE_MS        = 150      # ステージ移動後の待機時間 [ms]
PFS_SETTLE_MS    = 200      # PFS ロック待機時間 [ms]（不要なら 0）

# MM1.4 デバイス名
XY_STAGE_DEVICE  = "XYStage"
Z_OFFSET_DEVICE  = "TIPFSOffset"   # Nikon PFS オフセット
CHANNEL_NAME     = "ph"            # MM チャンネルグループ名
PYTHON_EXE       = r"C:\Users\QPI\AppData\Local\Programs\Python\Python311\python.exe"

# ドリフト補正パラメータ
JUMP_THRESH_UM       = 1.0    # これを超えるドリフトは外れ値として無視 [μm]
MAX_TOTAL_CORR_UM    = 15.0   # 累積補正量の上限 [μm]（安全弁）
SHIFT_SIGN_X         = 1      # 符号（実データで確認して変更）
SHIFT_SIGN_Y         = 1

# 光学パラメータ
SENSOR_PIXEL_SIZE    = 3.45e-6   # [m]
MAGNIFICATION        = 40
ORIGINAL_DIM         = 2048
RECONSTRUCTED_DIM    = 511

# 位置クロップ設定（pipeline_full.py と同値）
POS_SPLIT    = 31
CROP_BEFORE  = (0, 2048, 400, 2448)   # pos_number < POS_SPLIT → 右側
CROP_AFTER   = (0, 2048,   0, 2048)   # Pos3 以降

# バックグラウンド減算パラメータ
BACKSUB_MIN_PHASE    = -1.1
BACKSUB_HIST_MIN     = -1.1
BACKSUB_HIST_MAX     =  1.5
BACKSUB_N_BINS       = 512
BACKSUB_SMOOTH_WINDOW = 20

# Gaussian 空間グラジェント除去（channel crop の前に適用）
# alignment 用: sigma >= 150 推奨（channel crop サイズ 40×120 px より十分大きく）
# 0 で無効
GRADIENT_SIGMA = 0

# ECC パラメータ
VMIN = -5.0
VMAX =  2.0

# ============================================================


def compute_backsub_offset(img: np.ndarray, min_phase=BACKSUB_MIN_PHASE,
                            hist_min=BACKSUB_HIST_MIN, hist_max=BACKSUB_HIST_MAX,
                            n_bins=BACKSUB_N_BINS, smooth_window=BACKSUB_SMOOTH_WINDOW) -> float:
    """ガウスフィットによる背景ピーク検出。オフセット = -peak_mean を返す。"""
    bin_edges = np.linspace(hist_min, hist_max, n_bins + 1)
    hist_counts, _ = np.histogram(img.flatten(), bins=bin_edges)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_centers[1] - bin_centers[0]
    smoothed = uniform_filter1d(hist_counts.astype(float), size=smooth_window, mode='nearest')
    smoothed = uniform_filter1d(smoothed, size=smooth_window, mode='nearest')
    valid_idx = np.where(bin_centers >= min_phase)[0]
    max_search_idx = int(len(bin_centers) * 0.95)
    search_idx = valid_idx[valid_idx < max_search_idx]
    if len(search_idx) == 0:
        return 0.0
    peak_idx = search_idx[np.argmax(smoothed[search_idx])]
    peak_value = bin_centers[peak_idx]
    fit_width = 300
    s = max(0, peak_idx - fit_width)
    e = min(len(bin_centers), peak_idx + fit_width)
    x_data, y_data = bin_centers[s:e], smoothed[s:e]
    try:
        popt, _ = curve_fit(
            lambda x, amp, mean, std: amp * np.exp(-((x - mean)**2) / (2 * std**2)),
            x_data, y_data, p0=[float(np.max(y_data)), peak_value, bin_width * 20], maxfev=5000
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


def get_crop_for_pos_index(pos_index):
    return CROP_BEFORE if pos_index < POS_SPLIT else CROP_AFTER


def reconstruct_phase(raw_path: Path, crop_region, bg_path: Path = None) -> np.ndarray:
    """QPI 位相再構成。bg_path があれば差分、なければ生画像の位相を返す。"""
    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    def _load_and_reconstruct(path, params, crop):
        img = np.array(Image.open(str(path)))
        rs, re_, cs, ce = crop
        img = img[rs:re_, cs:ce]
        field = get_field(img, params)
        return unwrap_phase(np.angle(field))

    img = np.array(Image.open(str(raw_path)))
    rs, re_, cs, ce = crop_region
    cropped = img[rs:re_, cs:ce]
    qpi_params = QPIParameters(
        wavelength=WAVELENGTH, NA=NA,
        img_shape=cropped.shape, pixelsize=PIXELSIZE,
        offaxis_center=OFFAXIS_CENTER,
    )
    phase = _load_and_reconstruct(raw_path, qpi_params, crop_region)
    if bg_path is not None and bg_path.exists():
        phase -= _load_and_reconstruct(bg_path, qpi_params, crop_region)
    return phase


def main():
    session_dir = Path(SESSION_DIR)
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"セッションディレクトリ: {session_dir}")

    # ---- 1. .pos ファイルを読んで positions.csv を出力 ----
    pos_file = Path(POSITIONS_FILE)
    if not pos_file.exists():
        print(f"ERROR: .pos ファイルが見つかりません: {pos_file}")
        sys.exit(1)

    with open(pos_file, encoding="utf-8") as f:
        pos_data = json.load(f)

    positions = []
    for i, pos_entry in enumerate(pos_data["POSITIONS"]):
        label = pos_entry["LABEL"]
        x, y, z_offset = 0.0, 0.0, 0.0
        for dev in pos_entry["DEVICES"]:
            if dev["DEVICE"] == XY_STAGE_DEVICE:
                x = float(dev["X"])
                y = float(dev["Y"])
            elif dev["DEVICE"] == Z_OFFSET_DEVICE:
                z_offset = float(dev["X"])
        positions.append({"index": i, "label": label, "x": x, "y": y, "z_offset": z_offset})
        print(f"  Pos[{i}] {label}: x={x:.3f}, y={y:.3f}, z_offset={z_offset:.4f}")

    n_positions = len(positions)
    print(f"ポジション数: {n_positions}")

    if REF_POS_INDEX >= n_positions:
        print(f"ERROR: REF_POS_INDEX={REF_POS_INDEX} が範囲外 (0~{n_positions-1})")
        sys.exit(1)

    # positions.csv を書き出し
    csv_path = session_dir / "positions.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,label,x,y,z_offset\n")
        for p in positions:
            f.write(f"{p['index']},{p['label']},{p['x']:.6f},{p['y']:.6f},{p['z_offset']:.6f}\n")
    print(f"positions.csv 保存: {csv_path}")

    # ---- 2. channel_rois.json 読み込み ----
    rois_path = Path(CHANNEL_ROIS_JSON)
    if not rois_path.exists():
        print(f"ERROR: channel_rois.json が見つかりません: {rois_path}")
        sys.exit(1)
    with open(rois_path, encoding="utf-8") as f:
        rois = json.load(f)
    n_channels = len(rois)
    print(f"チャネル数: {n_channels}")
    if n_channels == 0:
        print(f"ERROR: channel_rois.json にチャネルが登録されていません: {rois_path}")
        print("  calibrate_grid_positions.py --detect で検出し直すか、ROI を手動で設定して --apply を実行してください。")
        sys.exit(1)

    # ---- 3. grid(0,0) の位相画像を読んでリファレンスcropを生成 ----
    grid_ref_path = (
        Path(GRID_DIR)
        / f"{GRID_BASE_LABEL}_x+0_y+0"
        / "output_phase"
        / f"img_000000000_ph_{GRID_Z_INDEX:03d}_phase.tif"
    )
    if not grid_ref_path.exists():
        # フォールバック: _ph_{z:03d} なし（旧命名）
        alt = grid_ref_path.parent / f"img_000000000_ph_{GRID_Z_INDEX:03d}.tif"
        if alt.exists():
            grid_ref_path = alt
        else:
            print(f"ERROR: grid(0,0) 位相画像が見つかりません: {grid_ref_path}")
            print("  先に pipeline_full.py Step0 (grid reconstruction) を実行してください。")
            sys.exit(1)

    print(f"grid(0,0) 参照画像: {grid_ref_path}")
    grid_img = tifffile.imread(str(grid_ref_path)).astype(np.float64)

    # Gaussian 空間グラジェント除去（タイムラプス側と同じ処理を必ず適用）
    if GRADIENT_SIGMA > 0:
        from scipy.ndimage import gaussian_filter
        bg = gaussian_filter(grid_img.astype(np.float32), sigma=GRADIENT_SIGMA, mode='nearest')
        grid_img = grid_img - bg.astype(np.float64)
        print(f"  Gaussian gradient removal: sigma={GRADIENT_SIGMA}px applied to grid ref")

    # チャネルごとに crop + backsub
    ref_crops = []
    for ch_idx, roi in enumerate(rois):
        crop = extract_rect_roi(grid_img, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        offset = compute_backsub_offset(crop)
        crop = crop + offset
        ref_crops.append(crop.astype(np.float32))
        print(f"  ch{ch_idx:02d}: crop shape={crop.shape}, backsub offset={offset:+.4f}")

    # grid_ref_crops.tif として保存（shape: n_channels, H, W）
    ref_crops_arr = np.stack(ref_crops, axis=0)  # (C, H, W)
    grid_ref_tif = session_dir / "grid_ref_crops.tif"
    tifffile.imwrite(str(grid_ref_tif), ref_crops_arr)
    print(f"grid_ref_crops.tif 保存: {grid_ref_tif}  shape={ref_crops_arr.shape}")

    # ---- 4. drift_config.json 生成 ----
    pixel_scale_um = SENSOR_PIXEL_SIZE / MAGNIFICATION * ORIGINAL_DIM / RECONSTRUCTED_DIM * 1e6
    config = {
        "python_exe":         PYTHON_EXE,
        "script_dir":         str(_script_dir),
        "session_dir":        str(session_dir),
        "save_dir":           SAVE_DIR,
        "positions_csv":      str(csv_path),
        "grid_ref_crops_tif": str(grid_ref_tif),
        "state_file":         str(session_dir / "drift_state.txt"),
        "log_file":           str(session_dir / "drift_log.json"),
        "prev_frame_crops_tif": str(session_dir / "prev_frame_crops.tif"),
        "channel_rois_json":  str(rois_path),
        "n_timepoints":       N_TIMEPOINTS,
        "interval_sec":       INTERVAL_SEC,
        "exposure_ms":        EXPOSURE_MS,
        "settle_ms":          SETTLE_MS,
        "pfs_settle_ms":      PFS_SETTLE_MS,
        "ref_pos_index":      REF_POS_INDEX,
        "bg_pos_index":       BG_POS_INDEX,
        "n_channels":         n_channels,
        "xy_stage_device":    XY_STAGE_DEVICE,
        "z_offset_device":    Z_OFFSET_DEVICE,
        "channel_name":       CHANNEL_NAME,
        "jump_thresh_um":     JUMP_THRESH_UM,
        "max_total_corr_um":  MAX_TOTAL_CORR_UM,
        "shift_sign_x":       SHIFT_SIGN_X,
        "shift_sign_y":       SHIFT_SIGN_Y,
        "pixel_scale_um":     pixel_scale_um,
        "pos_split":          POS_SPLIT,
        "crop_before":        list(CROP_BEFORE),
        "crop_after":         list(CROP_AFTER),
        "backsub_min_phase":  BACKSUB_MIN_PHASE,
        "backsub_hist_min":   BACKSUB_HIST_MIN,
        "backsub_hist_max":   BACKSUB_HIST_MAX,
        "backsub_n_bins":     BACKSUB_N_BINS,
        "backsub_smooth_window": BACKSUB_SMOOTH_WINDOW,
        "gradient_sigma":     GRADIENT_SIGMA,
        "ecc_vmin":           VMIN,
        "ecc_vmax":           VMAX,
        "sensor_pixel_size":  SENSOR_PIXEL_SIZE,
        "magnification":      MAGNIFICATION,
        "original_dim":       ORIGINAL_DIM,
        "reconstructed_dim":  RECONSTRUCTED_DIM,
        "grid_dir":           GRID_DIR,
        "grid_base_label":    GRID_BASE_LABEL,
        "grid_z_index":       GRID_Z_INDEX,
    }
    config_path = session_dir / "drift_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"drift_config.json 保存: {config_path}")

    # ---- 5. drift_state.txt 初期化 ----
    state_path = session_dir / "drift_state.txt"
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("# drift_state.txt - realtime_drift_mda.bsh / compute_drift_online.py\n")
        f.write("STATUS=idle\n")
        f.write("TIMEPOINT=-1\n")
        f.write("DX_UM=0.0\n")
        f.write("DY_UM=0.0\n")
        f.write("CUMULATIVE_DX_UM=0.0\n")
        f.write("CUMULATIVE_DY_UM=0.0\n")
        f.write("CORRECTION_VALID=false\n")
        f.write("ECC_CORRELATION=0.0\n")
        f.write("JUMP_DETECTED=false\n")
    print(f"drift_state.txt 初期化: {state_path}")

    # ---- 6. drift_log.json 初期化 ----
    log_path = session_dir / "drift_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump([], f)
    print(f"drift_log.json 初期化: {log_path}")

    # ---- 7. 保存ディレクトリ作成 ----
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)
    for p in positions:
        (save_dir / p["label"]).mkdir(exist_ok=True)
    print(f"保存ディレクトリ作成: {save_dir}")

    # ---- サマリ ----
    print("\n" + "="*60)
    print("セットアップ完了。以下の順で実行してください:")
    print(f"  1. MM1.4 の Script Panel を開く")
    print(f"  2. realtime_drift_mda.bsh を読み込む")
    print(f"  3. スクリプト先頭の CONFIG_FILE を以下に設定:")
    print(f"     {config_path}")
    print(f"  4. Run ボタンを押す")
    print(f"\n保存先: {SAVE_DIR}")
    print(f"タイムポイント数: {N_TIMEPOINTS}  間隔: {INTERVAL_SEC}s")
    print(f"ドリフト参照Pos: [{REF_POS_INDEX}] {positions[REF_POS_INDEX]['label']}")
    print(f"BG Pos:         [{BG_POS_INDEX}] {positions[BG_POS_INDEX]['label']}")
    print(f"pixel scale: {pixel_scale_um:.4f} μm/px")


if __name__ == "__main__":
    main()
