# %%
"""
calibrate_grid_pos_per_pos.py
-----------------------------
各 base pos (PosN) について、対応する PosN_x+0_y+0 を個別に参照して
ECC 中心補正を行い、0.1 μm 名目 grid を展開する。

【処理の流れ】
  For each PosN in TIMELAPSE_POS（BG_COPY_FROM のキーを除く）:
    1. 中心補正用 reconstruction:
       CALIB_GRID/PosN_x+0_y+0 の output_phase が未作成なら自動再構成。
       (x+0_y+0 のみ。全 grid 点の再構成はしない。)
    2. 中心補正: ECC( REF_GRID/PosN_x+0_y+0, CALIB_GRID/PosN_x+0_y+0 )
       - _tilt_correct + 固定 ecc_vmin/ecc_vmax
       - MAD 外れ値除去 (thresh=5.0)
       - 補正量を PosN 座標に適用 (calibrate_grid_pos.py と同じ符号規約)

  BG_COPY_FROM で指定した BG pos（例: Pos0）には、
  対応するサンプル pos（例: Pos1）の補正量をそのままコピーする。

  出力:
    BASE_DIR/_per_pos_ecc_corrected.pos  : 補正済み base pos
    BASE_DIR/_per_pos_ecc_grid.pos       : 名目 0.1 μm grid 展開 (snake scan)
    BASE_DIR/per_pos_ecc_log.json        : 補正ログ

【符号規約】
  findTransformECC(ref, sample) → (tx, ty)
    tx = warp[0,2] : 画像 X (col) 方向のずれ [px]
    ty = warp[1,2] : 画像 Y (row) 方向のずれ [px]
  drift_stage_x_um = sx_sign * ty * pixel_scale_um   (画像Y → ステージX)
  drift_stage_y_um = sy_sign * tx * pixel_scale_um   (画像X → ステージY)
  pos_correct = -drift  (ドリフトを打ち消す方向)
  ※ calibrate_grid_pos.py L389-394 と完全同一
"""

import json
import copy
import re
import sys
import numpy as np
import tifffile
import cv2
from pathlib import Path

# ============================================================
# ★ 設定パラメータ — 毎回ここを編集する
# ============================================================

# 出力先ディレクトリ
BASE_DIR = r"D:\AquisitionData\Kitagishi\260331"

# 補正する timelapse.pos（base positions のみ含む）
TIMELAPSE_POS = r"D:\AquisitionData\Kitagishi\260331\timelapse.pos"

# 今日の calibration データフォルダ
# grid データ（Pos1_x+0_y+0）のとき CALIB_SUFFIX = "x+0_y+0"
# focus/timelapse データ（Pos1）のとき CALIB_SUFFIX = ""
CALIB_SUFFIX   = ""
CALIB_GRID_DIR = r"E:\Acuisition\kitagishi\260331\focus_check_3"

# Day-1 固定参照 grid フォルダ
# PosN_x+0_y+0/output_phase/*_phase.tif が再構成済みであること
REF_GRID_DIR   = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"

# z インデックス（img_000000000_ph_{Z:03d}_phase.tif）
REF_Z_INDEX   = 10   # REF 側（再構成済みファイルに合わせる）
CALIB_Z_INDEX = 10   # CALIB 側（z=0=index10 の in-focus フレーム）

# CALIB_GRID の BG base label（再構成時の BG 差し引きに使う Pos0_x+0_y+0）
CALIB_BG_BASE_LABEL = "Pos0"

# channel_rois.json が per-pos に存在しない場合は channel_crop.py --detect を自動実行する

# drift_config.json（光学パラメータ・符号・ECC vmin/vmax・crop 設定）
DRIFT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"

# ECC / tilt 補正パラメータ
TILT_CROP_H = 270   # tilt 補正用 X 幅 [px]（compute_pos_shifts.py と同値）
ECC_CROP_H  = 80    # ECC に使う crop X 幅 [px]
MAD_THRESH  = 5.0   # チャネル間 MAD 外れ値閾値

# グリッド展開パラメータ（generate_grid_pos.py と一致させる）
X_STEP = 0.1   # [μm]  ステージ X → 画像 Y
Y_STEP = 0.1   # [μm]  ステージ Y → 画像 X
X_HALF = 4     # → 合計 9 点/軸 → 81 点/Pos
Y_HALF = 4

# BG pos の補正量コピー設定
# キー: BG の LABEL、値: 補正量をコピーするサンプル pos の LABEL
# 例: Pos0 には Pos1 の補正量をそのまま適用する
BG_COPY_FROM = {"Pos0": "Pos1"}

# ============================================================


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_rect_roi(img, cy, cx, crop_w, crop_h):
    """calibrate_grid_positions.py と同一実装。"""
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


def _tilt_correct(img_f64, cy, cx, crop_w, crop_h_out, fit_right: bool = False):
    """
    compute_pos_shifts.py と同一実装。
    big crop (TILT_CROP_H cols) → 背景側1/3 slope+intercept fit → 補正 → 中央 crop_h_out cols。
    fit_right=False: 左1/3（Pos_num < POS_SPLIT）、True: 右1/3（Pos_num >= POS_SPLIT）。
    """
    big   = extract_rect_roi(img_f64, cy, cx, crop_w, TILT_CROP_H).astype(np.float64)
    x     = np.arange(TILT_CROP_H, dtype=np.float64)
    prof  = big.mean(axis=0)
    fit_n = max(1, TILT_CROP_H // 3)
    if fit_right:
        a, b = np.polyfit(x[-fit_n:], prof[-fit_n:], 1)
    else:
        a, b = np.polyfit(x[:fit_n], prof[:fit_n], 1)
    corrected = big - (a * x + b)[np.newaxis, :]
    start = (TILT_CROP_H - crop_h_out) // 2
    return corrected[:, start : start + crop_h_out]


def to_uint8_fixed(img, vmin, vmax):
    """固定 vmin/vmax で uint8 化（percentile normalization なし）。"""
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    """(tx, ty, correlation) を返す。失敗時は None。"""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


def _mad(arr):
    m = np.median(arr)
    return float(np.median(np.abs(arr - m)))


def _remove_outliers_mad(values, thresh):
    arr = np.array(values, dtype=np.float64)
    md = _mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - np.median(arr)) > thresh * md


def get_crops_u8(img_f64, rois, n_channels, vmin, vmax, fit_right: bool = False):
    """全チャネルの tilt 補正済み uint8 crop を返す。"""
    return [
        to_uint8_fixed(
            _tilt_correct(img_f64, rois[ch]["cy"], rois[ch]["cx"],
                          rois[ch]["crop_w"], ECC_CROP_H, fit_right),
            vmin, vmax,
        )
        for ch in range(n_channels)
    ]


def load_phase_image(grid_dir, label, z_index):
    """output_phase/img_000000000_ph_{z_index:03d}_phase.tif を読む。"""
    path = (Path(grid_dir) / label / "output_phase"
            / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return tifffile.imread(str(path)).astype(np.float64)


# ============================================================
# 位相再構成（calibrate_grid_pos.py から移植、per-Pos crop 対応）
# ============================================================

def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None,
                      pos_num: int = 1) -> np.ndarray:
    """QPI 位相再構成。pos_num で crop_before / crop_after を切り替える。"""
    script_dir = Path(cfg["script_dir"])
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    pos_split = cfg.get("pos_split", 3)
    crop = tuple(cfg["crop_before"]) if pos_num < pos_split else tuple(cfg["crop_after"])
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
        print(f"    BG: {bg_path.parent.name}/{bg_path.name}")
    else:
        print("    BG なし")
    return phase


def postprocess_phase(phase: np.ndarray, cfg: dict) -> np.ndarray:
    """calibrate_grid_pos.py と同一。平均除去 + Gaussian 勾配除去。"""
    from scipy.ndimage import gaussian_filter
    h_p, w_p = phase.shape
    region = phase[1:h_p - 1, 1:w_p // 2]
    if region.size > 0:
        phase = phase - np.mean(region)
    sigma = cfg.get("gradient_sigma", 0)
    if sigma > 0:
        phase = phase - gaussian_filter(phase, sigma=sigma, mode="nearest")
    return phase


def ensure_center_reconstructed(calib_grid_dir: str, label: str, z_index: int,
                                 bg_base_label: str, cfg: dict,
                                 bg_suffix: str = "x+0_y+0") -> Path:
    """
    CALIB_GRID/label/output_phase/img_..._phase.tif が存在しなければ再構成する。
    label 例: "Pos1_x+0_y+0"（grid）または "Pos1"（focus/timelapse）。再構成済みパスを返す。
    bg_suffix: BG ディレクトリのサフィックス（grid: "x+0_y+0"、focus: ""）
    """
    out_path = (Path(calib_grid_dir) / label / "output_phase"
                / f"img_000000000_ph_{z_index:03d}_phase.tif")
    if out_path.exists():
        return out_path

    print(f"  [reconstruction] {label} → 自動再構成")
    raw_path = (Path(calib_grid_dir) / label
                / f"img_000000000_ph_{z_index:03d}.tif")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw 画像が見つかりません: {raw_path}")

    bg_label = f"{bg_base_label}_{bg_suffix}" if bg_suffix else bg_base_label
    bg_raw   = Path(calib_grid_dir) / bg_label / f"img_000000000_ph_{z_index:03d}.tif"
    if not bg_raw.exists():
        print(f"    WARNING: BG raw が見つかりません: {bg_raw} → BG なし")
        bg_raw = None

    m = re.match(r"Pos(\d+)", label)
    pos_num = int(m.group(1)) if m else 1

    phase = reconstruct_phase(raw_path, cfg, bg_raw, pos_num)
    phase = postprocess_phase(phase, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), phase.astype(np.float32))
    print(f"    保存: {out_path}")
    return out_path


# ============================================================
# 中心補正 ECC
# ============================================================

def center_ecc(ref_img, calib_img, rois, n_channels, vmin, vmax, fit_right: bool = False):
    """
    ECC → MAD 外れ値除去 → 平均。
    Returns: (tx_avg, ty_avg, per_ch_list) | (None, None, per_ch_list) 全失敗時。
    tx: 画像 X (col) [px], ty: 画像 Y (row) [px]
    """
    ref_crops   = get_crops_u8(ref_img,   rois, n_channels, vmin, vmax, fit_right)
    calib_crops = get_crops_u8(calib_img, rois, n_channels, vmin, vmax, fit_right)

    tx_list, ty_list, corr_list = [], [], []
    per_ch = []
    for ch in range(n_channels):
        res = ecc_align(ref_crops[ch], calib_crops[ch])
        if res is None:
            per_ch.append({"ch": ch, "tx": None, "ty": None,
                           "corr": None, "excluded": True, "reason": "ecc_failed"})
        else:
            tx, ty, corr = res
            tx_list.append(tx); ty_list.append(ty); corr_list.append(corr)
            per_ch.append({"ch": ch, "tx": tx, "ty": ty,
                           "corr": corr, "excluded": False, "reason": None})

    if not tx_list:
        return None, None, per_ch

    n_raw = len(tx_list)
    if n_raw >= 3:
        is_out = _remove_outliers_mad(tx_list, MAD_THRESH) | _remove_outliers_mad(ty_list, MAD_THRESH)
    else:
        is_out = np.zeros(n_raw, dtype=bool)

    valid_indices = [i for i, c in enumerate(per_ch) if not c["excluded"]]
    for k, idx in enumerate(valid_indices):
        if is_out[k]:
            per_ch[idx]["excluded"] = True
            per_ch[idx]["reason"] = "mad_outlier"

    used_mask = ~is_out
    if not np.any(used_mask):
        # 全部外れ値 → 全部使う（calibrate_grid_pos.py L364-365 相当）
        used_mask = np.ones(n_raw, dtype=bool)
        for idx in valid_indices:
            per_ch[idx]["excluded"] = False
            per_ch[idx]["reason"] = None

    tx_avg   = float(np.mean(np.array(tx_list)[used_mask]))
    ty_avg   = float(np.mean(np.array(ty_list)[used_mask]))
    n_used   = int(np.sum(used_mask))
    corr_avg = float(np.mean(np.array(corr_list)[used_mask]))
    print(f"  ECC: tx={tx_avg:+.3f}px  ty={ty_avg:+.3f}px  "
          f"使用 {n_used}/{n_raw}ch  corr={corr_avg:.4f}")
    return tx_avg, ty_avg, per_ch


# ============================================================
# メイン
# ============================================================

def _run_channel_detect(pos_dir: Path):
    """channel_crop.py --detect を output_phase/ の phase 画像に対して実行する。
    out_dir = img_dir / "channels" なので --dir に output_phase/ を渡すことで
    output_phase/channels/channel_rois.json に出力される。
    """
    import subprocess
    script = Path(__file__).resolve().parent / "channel_crop.py"
    output_phase_dir = pos_dir / "output_phase"
    cmd = [sys.executable, str(script),
           "--dir", str(output_phase_dir),
           "--pattern", "img_*_ph_000_phase.tif",
           "--detect"]
    print(f"  実行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"  [!] channel_crop.py --detect が失敗 (returncode={result.returncode})")


def main():
    cfg            = load_config(DRIFT_CONFIG)
    vmin           = cfg.get("ecc_vmin", -5.0)
    vmax           = cfg.get("ecc_vmax",  2.0)
    pixel_scale_um = cfg["pixel_scale_um"]
    sx_sign        = cfg.get("shift_sign_x", 1)
    sy_sign        = cfg.get("shift_sign_y", 1)
    pos_split      = cfg.get("pos_split", 33)

    print(f"vmin={vmin}  vmax={vmax}")
    print(f"pixel_scale: {pixel_scale_um:.5f} μm/px  sx_sign={sx_sign}  sy_sign={sy_sign}")

    with open(TIMELAPSE_POS, "r") as f:
        pos_data = json.load(f)
    positions = pos_data["POSITIONS"]
    print(f"timelapse.pos: {len(positions)} positions")

    # label → pos dict（BG コピー時に座標を更新するため）
    pos_by_label = {p["LABEL"]: p for p in positions}

    # label → 補正量 dict（BG コピー用に記録）
    correction_by_label: dict[str, dict] = {}

    bg_labels = set(BG_COPY_FROM.keys())

    log_entries = []

    # ---- サンプル pos の中心補正 ----
    for pos in positions:
        base_label = pos["LABEL"]
        if base_label in bg_labels:
            continue   # BG は後でコピー

        ref_label   = f"{base_label}_x+0_y+0"
        calib_label = f"{base_label}_{CALIB_SUFFIX}" if CALIB_SUFFIX else base_label
        print(f"\n=== {base_label} ===")

        # Step 1: 必要なら CALIB 側のみ再構成
        try:
            ensure_center_reconstructed(
                CALIB_GRID_DIR, calib_label, CALIB_Z_INDEX,
                CALIB_BG_BASE_LABEL, cfg, bg_suffix=CALIB_SUFFIX,
            )
        except FileNotFoundError as e:
            print(f"  [reconstruction] 失敗 → スキップ: {e}")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": f"reconstruction failed: {e}"})
            continue

        # Step 2: 中心補正 ECC
        try:
            ref_img   = load_phase_image(REF_GRID_DIR,   ref_label,   REF_Z_INDEX)
            calib_img = load_phase_image(CALIB_GRID_DIR, calib_label, CALIB_Z_INDEX)
        except FileNotFoundError as e:
            print(f"  [中心補正] 画像読み込み失敗 → スキップ: {e}")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": str(e)})
            continue

        pos_num   = int(re.match(r"Pos(\d+)", base_label).group(1))
        fit_right = pos_num >= pos_split

        # per-pos channel_rois.json を REF_GRID_DIR から読む
        # 存在しなければ channel_crop.py --detect を自動実行して生成する
        perpos_rois_path = (Path(REF_GRID_DIR) / ref_label
                            / "output_phase" / "channels" / "channel_rois.json")
        if not perpos_rois_path.exists():
            print(f"  [channel_detect] channel_rois.json なし → 自動 detect: {ref_label}")
            _run_channel_detect(Path(REF_GRID_DIR) / ref_label)
        if not perpos_rois_path.exists():
            raise FileNotFoundError(
                f"channel_rois.json の生成に失敗しました: {perpos_rois_path}\n"
                f"手動で実行: python scripts/channel_crop.py --dir \"{Path(REF_GRID_DIR) / ref_label}\" --detect"
            )
        with open(perpos_rois_path, encoding="utf-8") as f:
            rois = json.load(f)
        print(f"  channel_rois: {len(rois)} ch ({ref_label})")
        n_ch = len(rois)

        tx_avg, ty_avg, per_ch = center_ecc(ref_img, calib_img, rois, n_ch, vmin, vmax,
                                            fit_right=fit_right)

        if tx_avg is None:
            print(f"  [中心補正] 全チャネル ECC 失敗 → スキップ")
            log_entries.append({"base_label": base_label, "center_correction": None,
                                 "error": "all_ecc_failed", "per_channel": per_ch})
            continue

        # 符号規約: calibrate_grid_pos.py L389-394 と完全同一
        drift_stage_x_um = sx_sign * ty_avg * pixel_scale_um   # 画像Y → ステージX
        drift_stage_y_um = sy_sign * tx_avg * pixel_scale_um   # 画像X → ステージY
        pos_correct_x_um = -drift_stage_x_um
        pos_correct_y_um = -drift_stage_y_um

        print(f"  補正: stage_X={pos_correct_x_um:+.4f}μm  stage_Y={pos_correct_y_um:+.4f}μm")

        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] += pos_correct_x_um
                dev["Y"] += pos_correct_y_um

        correction = {
            "tx_avg_px": tx_avg,
            "ty_avg_px": ty_avg,
            "drift_stage_x_um": drift_stage_x_um,
            "drift_stage_y_um": drift_stage_y_um,
            "pos_correct_x_um": pos_correct_x_um,
            "pos_correct_y_um": pos_correct_y_um,
            "n_channels_total": n_ch,
            "per_channel": per_ch,
        }
        correction_by_label[base_label] = correction
        log_entries.append({"base_label": base_label, "center_correction": correction})

    # ---- BG pos に対応サンプルの補正量をコピー ----
    for bg_label, src_label in BG_COPY_FROM.items():
        print(f"\n=== {bg_label} (BG: {src_label} の補正量をコピー) ===")
        if src_label not in correction_by_label:
            print(f"  WARNING: {src_label} の補正量が未計算 → {bg_label} はスキップ")
            log_entries.append({"base_label": bg_label,
                                 "center_correction": None,
                                 "copied_from": src_label,
                                 "error": "source correction not available"})
            continue

        src_corr = correction_by_label[src_label]
        pos = pos_by_label.get(bg_label)
        if pos is None:
            print(f"  WARNING: {bg_label} が timelapse.pos に見つかりません → スキップ")
            continue

        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                dev["X"] += src_corr["pos_correct_x_um"]
                dev["Y"] += src_corr["pos_correct_y_um"]

        print(f"  補正: stage_X={src_corr['pos_correct_x_um']:+.4f}μm"
              f"  stage_Y={src_corr['pos_correct_y_um']:+.4f}μm  (コピー元: {src_label})")
        log_entries.append({
            "base_label": bg_label,
            "copied_from": src_label,
            "center_correction": {
                k: v for k, v in src_corr.items() if k != "per_channel"
            },
        })

    # ---- 補正済み base pos 保存 ----
    base_dir = Path(BASE_DIR)
    out_corrected = base_dir / "_per_pos_ecc_corrected.pos"
    with open(out_corrected, "w") as f:
        json.dump(pos_data, f, indent=3)
    print(f"\n補正済み base pos: {out_corrected}  ({len(positions)} positions)")

    # ---- grid 展開（snake scan: generate_grid_pos.py と完全一致） ----
    new_positions = []
    for pos in positions:
        base_x = base_y = base_z = 0.0
        for dev in pos["DEVICES"]:
            if dev["DEVICE"] == "XYStage":
                base_x = dev["X"]
                base_y = dev["Y"]
            elif dev["DEVICE"] == "TIPFSOffset":
                base_z = dev["X"]

        for xi in range(-X_HALF, X_HALF + 1):
            row = xi + X_HALF   # 0-indexed
            yi_range = (range(-Y_HALF, Y_HALF + 1) if row % 2 == 0
                        else range(Y_HALF, -Y_HALF - 1, -1))
            for yi in yi_range:
                new_pos = copy.deepcopy(pos)
                new_pos["LABEL"] = f"{pos['LABEL']}_x{xi:+d}_y{yi:+d}"
                for dev in new_pos["DEVICES"]:
                    if dev["DEVICE"] == "XYStage":
                        dev["X"] = base_x + xi * X_STEP
                        dev["Y"] = base_y + yi * Y_STEP
                    elif dev["DEVICE"] == "TIPFSOffset":
                        dev["X"] = base_z
                new_positions.append(new_pos)

    pos_data["POSITIONS"] = new_positions
    out_grid = base_dir / "_per_pos_ecc_grid.pos"
    with open(out_grid, "w") as f:
        json.dump(pos_data, f, indent=3)
    n_base = len(positions)
    print(f"grid 展開後: {out_grid}  "
          f"({len(new_positions)} = {n_base} × {(2*X_HALF+1)*(2*Y_HALF+1)})")

    # ---- ログ保存 ----
    out_log = base_dir / "per_pos_ecc_log.json"
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump({
            "timelapse_pos":   str(TIMELAPSE_POS),
            "calib_grid_dir":  str(CALIB_GRID_DIR),
            "ref_grid_dir":    str(REF_GRID_DIR),
            "ref_z_index":     REF_Z_INDEX,
            "calib_z_index":   CALIB_Z_INDEX,
            "ecc_vmin":        vmin,
            "ecc_vmax":        vmax,
            "tilt_crop_h":     TILT_CROP_H,
            "ecc_crop_h":      ECC_CROP_H,
            "mad_thresh":      MAD_THRESH,
            "x_step_um":       X_STEP,
            "y_step_um":       Y_STEP,
            "x_half":          X_HALF,
            "y_half":          Y_HALF,
            "bg_copy_from":    BG_COPY_FROM,
            "per_pos":         log_entries,
        }, f, indent=2, ensure_ascii=False)
    print(f"ログ: {out_log}")
    print("\n--- 完了 ---")


if __name__ == "__main__":
    main()

# %%
