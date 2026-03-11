"""
compute_drift_online.py
-----------------------
タイムラプス1タイムポイント分のXYドリフト量を計算する。
realtime_drift_mda.bsh から各タイムポイント後にサブプロセスとして呼ばれる。

使い方 (Beanshell が呼び出す):
    python compute_drift_online.py \
        --timepoint 5 \
        --sample-raw "D:/path/Pos1/img_000000005_ph_000.tif" \
        --bg-raw     "D:/path/Pos0/img_000000005_ph_000.tif" \
        --config     "C:/Users/QPI/Documents/QPI_Omni/drift_session/drift_config.json"

出力:
    drift_state.txt に結果を書き込む（Beanshell が読む）

ドリフト計算戦略:
  - T=0: grid(0,0) 再構成済み位相画像 vs 現フレーム → ECC
  - T>0: 前フレームのcrop（prev_frame_crops.tif）vs 現フレーム → ECC（安定・小シフト）
  - 両方とも累積 = T=0 基準のトータルドリフトをtrackする
  - ジャンプ（JUMP_THRESH）を超えたら補正しない（ECC失敗と判断）
"""

import sys
import json
import argparse
import numpy as np
import tifffile
import cv2
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
                jump: bool):
    """drift_state.txt を書き込む（Beanshell が key=value 形式で読む）"""
    lines = [
        "# drift_state.txt - written by compute_drift_online.py",
        f"STATUS={'correction_ready' if valid else 'correction_skipped'}",
        f"TIMEPOINT={t}",
        f"DX_UM={dx_um:.6f}",
        f"DY_UM={dy_um:.6f}",
        f"CUMULATIVE_DX_UM={cum_dx:.6f}",
        f"CUMULATIVE_DY_UM={cum_dy:.6f}",
        f"CORRECTION_VALID={'true' if valid else 'false'}",
        f"ECC_CORRELATION={corr:.6f}",
        f"JUMP_DETECTED={'true' if jump else 'false'}",
        f"TIMESTAMP={datetime.now().isoformat()}",
    ]
    with open(state_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def read_state(state_path: str) -> dict:
    """前回の状態（累積ドリフト値など）を読む"""
    result = {
        "cumulative_dx_um": 0.0,
        "cumulative_dy_um": 0.0,
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
    except Exception:
        pass
    return result


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


def to_uint8(img, vmin, vmax):
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8, max_iter=10000, epsilon=1e-6):
    """ECC アライメント。(tx, ty, correlation) を返す。失敗時は None。"""
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, epsilon)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria
        )
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None) -> np.ndarray:
    """QPI 位相再構成。bg_path があれば差分を返す。"""
    script_dir = Path(cfg["script_dir"])
    sys.path.insert(0, str(script_dir))

    from PIL import Image
    from qpi import QPIParameters, get_field
    from skimage.restoration import unwrap_phase
    from optical_config import OFFAXIS_CENTER, WAVELENGTH, NA, PIXELSIZE

    # クロップ領域はターゲットPos（ref_pos）に合わせる。
    # BG（Pos0）も同じcropで再構成する必要があるため、bg_pathにも同じcropを使う。
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

    print(f"[T={t}] compute_drift_online.py start")
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

    # ---- grid(0,0) 参照 crops（prepare_drift_session.py で生成済み）----
    grid_ref_crops = tifffile.imread(str(ref_crops_path)).astype(np.float64)
    # shape: (n_channels, H, W) or (H, W) if n_channels==1
    if grid_ref_crops.ndim == 2:
        grid_ref_crops = grid_ref_crops[np.newaxis, ...]

    # 常に grid(0,0) を基準に比較する。
    # 前フレーム比較（incremental）はステージ補正が入ると
    # 意図的なステージ移動をドリフトとして誤検出するため使わない。
    use_prev = False
    prev_crops = grid_ref_crops.copy()
    print(f"  Using grid(0,0) as reference")

    # ---- QPI 位相再構成 ----
    print("  Phase reconstruction...")
    try:
        phase = reconstruct_phase(sample_raw, cfg, bg_raw)
    except Exception as ex:
        print(f"ERROR: Phase reconstruction failed: {ex}")
        import traceback; traceback.print_exc()
        write_state(state_path, t, 0.0, 0.0, 0.0, 0.0, False, 0.0, False)
        sys.exit(1)

    # 平均除去（mean subtraction でオフセット正規化）
    h_p, w_p = phase.shape
    region = phase[1:h_p-1, 1:w_p//2]   # 同じ領域を pipeline_full.py と統一
    if region.size > 0:
        phase -= np.mean(region)

    # ---- Gaussian 空間グラジェント除去（prepare_drift_session.py と同一処理）----
    gradient_sigma = cfg.get("gradient_sigma", 0)
    if gradient_sigma > 0:
        from scipy.ndimage import gaussian_filter
        bg = gaussian_filter(phase, sigma=gradient_sigma, mode='nearest')
        phase = phase - bg
        print(f"  Gaussian gradient removal: sigma={gradient_sigma}px")

    # ---- 再構成済み位相画像を保存（pipeline_full.py と同じ命名規則）----
    # sample_raw.parent / "output_phase" / "img_XXXXXXXXX_ph_000_phase.tif"
    out_phase_dir = sample_raw.parent / "output_phase"
    out_phase_dir.mkdir(exist_ok=True)
    out_phase_path = out_phase_dir / (sample_raw.stem + "_phase.tif")
    tifffile.imwrite(str(out_phase_path), phase.astype(np.float32))
    print(f"  Phase saved: {out_phase_path.name}")

    # ---- 各チャネルのROI cropを切り出し + backsub ----
    current_crops = []
    for ch_idx, roi in enumerate(rois):
        crop = extract_rect_roi(phase, roi["cy"], roi["cx"], roi["crop_w"], roi["crop_h"])
        offset = compute_backsub_offset(crop, cfg)
        crop = crop + offset
        current_crops.append(crop)

    # ---- チャネルごとに ECC ----
    tx_list, ty_list, corr_list = [], [], []
    for ch_idx in range(min(n_channels, len(current_crops))):
        ref_crop  = prev_crops[ch_idx] if ch_idx < len(prev_crops) else prev_crops[-1]
        cur_crop  = current_crops[ch_idx]

        ref_u8 = to_uint8(ref_crop, vmin, vmax)
        cur_u8 = to_uint8(cur_crop, vmin, vmax)
        result = ecc_align(ref_u8, cur_u8)
        if result is not None:
            tx, ty, corr = result
            tx_list.append(tx)
            ty_list.append(ty)
            corr_list.append(corr)
            print(f"    ch{ch_idx:02d}: tx={tx:+.3f}px ty={ty:+.3f}px corr={corr:.4f}")
        else:
            print(f"    ch{ch_idx:02d}: ECC failed")

    if not tx_list:
        print("ERROR: ECC failed on all channels")
        prev_state = read_state(state_path)
        write_state(state_path, t, 0.0, 0.0,
                    prev_state["cumulative_dx_um"], prev_state["cumulative_dy_um"],
                    False, 0.0, False)
        sys.exit(0)

    # チャネル平均
    tx_avg  = float(np.mean(tx_list))
    ty_avg  = float(np.mean(ty_list))
    corr_avg = float(np.mean(corr_list))
    print(f"  ECC平均: tx={tx_avg:+.4f}px  ty={ty_avg:+.4f}px  corr={corr_avg:.4f}")

    # ---- 符号・スケール変換（pixel → μm、画像軸 → ステージ軸）----
    # ECC findTransformECC(ref, current): templateImage ≈ warpAffine(inputImage, W)
    # W = [[1,0,tx],[0,1,ty]]  →  current は ref に対して (tx, ty) px ずれている
    # shift_x (画像X方向) = tx → ステージY補正
    # shift_y (画像Y方向) = ty → ステージX補正
    # grid_subtract.py と同じ符号規則:
    #   dx_stage_um (ステージX) = SHIFT_SIGN_X * shift_y * pixel_scale_um
    #   dy_stage_um (ステージY) = SHIFT_SIGN_Y * shift_x * pixel_scale_um
    # ステージをこの分だけ引けばサンプルが元の位置に戻る

    shift_x = tx_avg  # 画像X方向ずれ [px]
    shift_y = ty_avg  # 画像Y方向ずれ [px]

    # 前フレーム基準の場合: これが「前フレームからの増分ドリフト」
    # grid(0,0) 基準の場合: これが「grid(0,0)からのトータルドリフト」
    # 補正ステージ移動量（ステージを+方向に動かすと画像が-方向にずれる前提）
    correction_stage_x_um = sx_sign * shift_y * pixel_scale_um  # ステージX方向補正 [μm]
    correction_stage_y_um = sy_sign * shift_x * pixel_scale_um  # ステージY方向補正 [μm]

    # ---- 累積ドリフト計算 ----
    prev_state = read_state(state_path)
    cum_dx = prev_state["cumulative_dx_um"] + correction_stage_x_um
    cum_dy = prev_state["cumulative_dy_um"] + correction_stage_y_um

    print(f"  Correction: stage_x={correction_stage_x_um:+.4f}um  stage_y={correction_stage_y_um:+.4f}um")
    print(f"  Cumulative: stage_x={cum_dx:+.4f}um  stage_y={cum_dy:+.4f}um")

    # ---- ジャンプ検出 ----
    step_um = (correction_stage_x_um**2 + correction_stage_y_um**2) ** 0.5
    total_um = (cum_dx**2 + cum_dy**2) ** 0.5
    jump = (step_um > jump_thresh) or (total_um > max_total)

    if jump:
        print(f"  [JUMP] step={step_um:.3f}um (thresh={jump_thresh}um), "
              f"total={total_um:.3f}um (max={max_total}um) -> skipped")
        # 累積値は更新しない
        write_state(state_path, t,
                    correction_stage_x_um, correction_stage_y_um,
                    prev_state["cumulative_dx_um"], prev_state["cumulative_dy_um"],
                    False, corr_avg, True)
    else:
        # ---- prev_frame_crops.tif を更新（次タイムポイントのincrementalに使う）----
        current_crops_arr = np.stack([c.astype(np.float32) for c in current_crops], axis=0)
        tifffile.imwrite(str(prev_crops_path), current_crops_arr)
        write_state(state_path, t,
                    correction_stage_x_um, correction_stage_y_um,
                    cum_dx, cum_dy,
                    True, corr_avg, False)
        print(f"  [OK] Correction valid, prev_frame_crops.tif updated")

    # ---- ログ追記 ----
    log_entry = {
        "timepoint":            t,
        "timestamp":            datetime.now().isoformat(),
        "sample_raw":           str(sample_raw),
        "bg_raw":               str(bg_raw) if bg_raw else None,
        "used_prev_frame":      use_prev,
        "n_channels_used":      len(tx_list),
        "tx_avg_px":            tx_avg,
        "ty_avg_px":            ty_avg,
        "ecc_correlation":      corr_avg,
        "correction_stage_x_um": correction_stage_x_um,
        "correction_stage_y_um": correction_stage_y_um,
        "cumulative_dx_um":     cum_dx,
        "cumulative_dy_um":     cum_dy,
        "jump_detected":        jump,
        "correction_valid":     not jump,
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
