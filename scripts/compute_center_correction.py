"""
compute_center_correction.py
----------------------------
BeanShell から呼ばれる:
  python compute_center_correction.py Pos1 <snap_tif_path>

  argv[1] = base label (e.g. "Pos1")
  argv[2] = BeanShell が保存した raw snap TIF のパス

出力:
  <snap_tif_path の grandparent>/center_correction_{label}.json
  <snap_tif_path の grandparent>/center_correction_{label}.txt  <- BeanShell 用

txt フォーマット:
  success          (or "failed")
  +0.1234          (pos_correct_x_um)
  -0.0567          (pos_correct_y_um)
"""

import sys
import json
import re
import numpy as np
import tifffile
import cv2
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# ★ 設定（calibrate_and_acquire_grid.bsh と一致させる）
# ============================================================
REF_GRID_DIR      = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
REF_Z_INDEX       = 10
DRIFT_CONFIG      = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"
TILT_CROP_H       = 270
ECC_CROP_H        = 80
MAD_THRESH        = 5.0
# ============================================================


def find_channel_rois(grid_dir: str, label: str):
    """Try both x+0_y+0 and x-0_y-0 naming conventions."""
    for center_name in [f"{label}_x+0_y+0", f"{label}_x-0_y-0"]:
        p = (Path(grid_dir) / center_name / "output_phase" / "channels" / "channel_rois.json")
        if p.exists():
            return p
    return None


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def _tilt_correct(img_f64, cy, cx, crop_w, crop_h_out, fit_right: bool = False):
    h, w  = img_f64.shape
    if (cx - TILT_CROP_H // 2) < 0 or (cx + TILT_CROP_H // 2) > w:
        return extract_rect_roi(img_f64, cy, cx, crop_w, crop_h_out).astype(np.float64)
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
    clipped = np.clip(img, vmin, vmax)
    return ((clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def ecc_align(ref_u8, tl_u8):
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-8)
    try:
        corr, warp_matrix = cv2.findTransformECC(
            ref_u8, tl_u8, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        return float(warp_matrix[0, 2]), float(warp_matrix[1, 2]), float(corr)
    except Exception:
        return None


def _mad(arr):
    return float(np.median(np.abs(arr - np.median(arr))))


def _remove_outliers_mad(values, thresh):
    arr = np.array(values, dtype=np.float64)
    md = _mad(arr)
    if md == 0:
        return np.zeros(len(arr), dtype=bool)
    return np.abs(arr - np.median(arr)) > thresh * md


def reconstruct_phase(raw_path: Path, cfg: dict, bg_path: Path = None,
                      pos_num: int = 1) -> np.ndarray:
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
        print(f"  BG: {bg_path.name}", flush=True)
    else:
        print("  No BG", flush=True)
    return phase


def postprocess_phase(phase: np.ndarray, cfg: dict) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    h_p, w_p = phase.shape
    region = phase[1:h_p - 1, 1:w_p // 2]
    if region.size > 0:
        phase = phase - np.mean(region)
    sigma = cfg.get("gradient_sigma", 0)
    if sigma > 0:
        phase = phase - gaussian_filter(phase, sigma=sigma, mode="nearest")
    return phase


def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_center_correction.py Pos1 <snap_tif_path>",
              flush=True)
        sys.exit(1)

    label     = sys.argv[1]
    snap_path = Path(sys.argv[2])
    # BeanShell は ECC_SNAP_DIR\PosN_x+0_y+0\img_...tif を渡す
    # txt 出力先は ECC_SNAP_DIR（grandparent of snap_path）
    out_dir   = snap_path.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"center_correction_{label}.json"
    txt_path  = out_dir / f"center_correction_{label}.txt"

    print(f"=== ECC: {label} ===", flush=True)

    def write_fail(msg):
        result = {"label": label, "success": False, "error": msg}
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        txt_path.write_text(f"failed\n{msg}\n")

    try:
        cfg        = load_config(DRIFT_CONFIG)
        vmin       = cfg.get("ecc_vmin", -5.0)
        vmax       = cfg.get("ecc_vmax",  2.0)
        sx_sign    = cfg.get("shift_sign_x", 1)
        sy_sign    = cfg.get("shift_sign_y", 1)
        px_scale   = cfg["pixel_scale_um"]

        channel_rois_path = find_channel_rois(REF_GRID_DIR, label)
        if channel_rois_path is None:
            raise FileNotFoundError(
                f"channel_rois.json not found for {label} in {REF_GRID_DIR}")
        with open(channel_rois_path, encoding="utf-8") as f:
            rois = json.load(f)
        n_channels = len(rois)
        print(f"  channels: {n_channels}  vmin={vmin}  vmax={vmax}  "
              f"rois: {channel_rois_path.parent.parent.parent.name}", flush=True)

        # --- reconstruct snap (ph_000) ---
        if not snap_path.exists():
            raise FileNotFoundError(f"snap not found: {snap_path}")

        print(f"  Reconstructing snap...", flush=True)
        m = re.match(r"Pos(\d+)", label)
        pos_num = int(m.group(1)) if m else 1
        pos_split  = cfg.get("pos_split", 3)
        fit_right  = (pos_num >= pos_split)

        # Find Pos0 BG raw from REF grid (same BG used to build phase_ref)
        bg_raw_path = None
        for bg_center_name in ["Pos0_x+0_y+0", "Pos0_x-0_y-0"]:
            p = Path(REF_GRID_DIR) / bg_center_name / f"img_000000000_ph_{REF_Z_INDEX:03d}.tif"
            if p.exists():
                bg_raw_path = p
                break
        if bg_raw_path is None:
            print("  WARNING: Pos0 BG raw not found in REF grid, reconstructing without BG",
                  flush=True)

        phase_calib = reconstruct_phase(snap_path, cfg, bg_path=bg_raw_path, pos_num=pos_num)
        phase_calib = postprocess_phase(phase_calib, cfg)

        # --- load REF (try x+0_y+0 then x-0_y-0) ---
        ref_path = None
        for center_name in [f"{label}_x+0_y+0", f"{label}_x-0_y-0"]:
            p = (Path(REF_GRID_DIR) / center_name / "output_phase"
                 / f"img_000000000_ph_{REF_Z_INDEX:03d}_phase.tif")
            if p.exists():
                ref_path = p
                break
        if ref_path is None:
            raise FileNotFoundError(f"REF not found for {label} in {REF_GRID_DIR}")
        phase_ref = tifffile.imread(str(ref_path)).astype(np.float64)

        # --- per-channel ECC ---
        tx_list, ty_list, corr_list = [], [], []
        per_ch = []
        for ch in range(n_channels):
            roi = rois[ch]
            ref_crop   = to_uint8_fixed(
                _tilt_correct(phase_ref,   roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H,
                              fit_right=fit_right),
                vmin, vmax)
            calib_crop = to_uint8_fixed(
                _tilt_correct(phase_calib, roi["cy"], roi["cx"], roi["crop_w"], ECC_CROP_H,
                              fit_right=fit_right),
                vmin, vmax)
            res = ecc_align(ref_crop, calib_crop)
            if res is None:
                per_ch.append({"ch": ch, "excluded": True, "reason": "ecc_failed"})
            else:
                tx, ty, corr = res
                tx_list.append(tx); ty_list.append(ty); corr_list.append(corr)
                per_ch.append({"ch": ch, "tx": tx, "ty": ty,
                               "corr": corr, "excluded": False})

        if not tx_list:
            raise RuntimeError("All channels ECC failed")

        n_raw = len(tx_list)
        if n_raw >= 3:
            is_out = (_remove_outliers_mad(tx_list, MAD_THRESH)
                      | _remove_outliers_mad(ty_list, MAD_THRESH))
        else:
            is_out = np.zeros(n_raw, dtype=bool)

        used_mask = ~is_out
        if not np.any(used_mask):
            used_mask = np.ones(n_raw, dtype=bool)

        tx_avg   = float(np.mean(np.array(tx_list)[used_mask]))
        ty_avg   = float(np.mean(np.array(ty_list)[used_mask]))
        n_used   = int(np.sum(used_mask))
        corr_avg = float(np.mean(np.array(corr_list)[used_mask]))
        print(f"  tx={tx_avg:+.3f}px  ty={ty_avg:+.3f}px  "
              f"n={n_used}/{n_raw}  corr={corr_avg:.4f}", flush=True)

        # 符号規約: calibrate_grid_pos.py L389-394 と完全同一
        drift_x = sx_sign * ty_avg * px_scale
        drift_y = sy_sign * tx_avg * px_scale
        pos_correct_x_um = -drift_x
        pos_correct_y_um = -drift_y
        print(f"  correction: X={pos_correct_x_um:+.4f} um"
              f"  Y={pos_correct_y_um:+.4f} um", flush=True)

        result = {
            "label": label, "success": True,
            "pos_correct_x_um": pos_correct_x_um,
            "pos_correct_y_um": pos_correct_y_um,
            "n_channels_used": n_used, "mean_correlation": corr_avg,
            "per_channel": per_ch,
        }
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        txt_path.write_text(
            f"success\n{pos_correct_x_um:+.6f}\n{pos_correct_y_um:+.6f}\n")
        print(f"  Saved: {txt_path}", flush=True)

    except Exception as e:
        import traceback
        print(f"  ERROR: {e}", flush=True)
        traceback.print_exc()
        write_fail(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
