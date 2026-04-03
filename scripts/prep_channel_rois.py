"""
prep_channel_rois.py
--------------------
Run BEFORE calibrate_and_acquire_grid.bsh and align_timelapse_pos.py.

For each sample PosN in timelapse.pos:
  1. Reconstruct Pos0_x+0_y+0 (BG, no BG subtraction) if not done
  2. Reconstruct PosN center (x+0_y+0 or x-0_y-0) using Pos0 as BG if not done
  3. Run channel_crop.py --detect to generate per-PosN channel_rois.json

Output: REF_GRID_DIR/PosN_center/output_phase/channels/channel_rois.json
"""

import sys
import json
import re
import subprocess
import numpy as np
import tifffile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

# ============================================================
# Settings
# ============================================================
TIMELAPSE_POS  = r"D:\AquisitionData\Kitagishi\260331\timelapse.pos"
REF_GRID_DIR   = r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
DRIFT_CONFIG   = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"
BG_LABEL       = "Pos0"    # background position (no channel detection needed)
CALIB_Z_INDEX  = 10        # z-index used for reconstruction and channel detection
SCRIPT_DIR     = Path(__file__).parent
CHANNEL_CROP   = SCRIPT_DIR / "channel_crop.py"
# ============================================================


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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
        print(f"    BG: {bg_path.parent.name}", flush=True)
    else:
        print("    No BG", flush=True)
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


def _find_center_dir(grid_dir: str, label: str):
    """Return Path to center directory, trying x+0_y+0 then x-0_y-0."""
    for center_name in [f"{label}_x+0_y+0", f"{label}_x-0_y-0"]:
        p = Path(grid_dir) / center_name
        if p.exists():
            return p
    return None


def ensure_reconstructed(center_dir: Path, z_index: int, bg_raw: Path,
                          pos_num: int, cfg: dict) -> Path:
    """Reconstruct center phase image if not already done."""
    out_path = center_dir / "output_phase" / f"img_000000000_ph_{z_index:03d}_phase.tif"
    if out_path.exists():
        print(f"    already reconstructed: {out_path.parent.name}", flush=True)
        return out_path

    raw_path = center_dir / f"img_000000000_ph_{z_index:03d}.tif"
    if not raw_path.exists():
        raise FileNotFoundError(f"raw not found: {raw_path}")

    print(f"    reconstructing {center_dir.name} ...", flush=True)
    phase = reconstruct_phase(raw_path, cfg,
                               bg_path=bg_raw if (bg_raw and bg_raw.exists()) else None,
                               pos_num=pos_num)
    phase = postprocess_phase(phase, cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(out_path), phase.astype(np.float32))
    print(f"    saved: {out_path}", flush=True)
    return out_path


def run_channel_detect(output_phase_dir: Path, z_index: int) -> bool:
    """Run channel_crop.py --detect on output_phase_dir using z_index phase image."""
    rois_path = output_phase_dir / "channels" / "channel_rois.json"
    if rois_path.exists():
        print(f"    channel_rois.json already exists", flush=True)
        return True

    pattern = f"*_ph_{z_index:03d}_phase.tif"
    cmd = [sys.executable, str(CHANNEL_CROP),
           "--dir", str(output_phase_dir),
           "--detect",
           "--pattern", pattern]
    print(f"    running channel_crop.py --detect (pattern={pattern})", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            print(f"      {line}", flush=True)
    if result.returncode != 0:
        print(f"    ERROR: channel_crop failed:\n{result.stderr}", flush=True)
        return False
    if rois_path.exists():
        with open(rois_path, encoding="utf-8") as f:
            rois = json.load(f)
        print(f"    channel_rois.json: {len(rois)} channels", flush=True)
        return True
    else:
        print(f"    ERROR: channel_rois.json not created", flush=True)
        return False


def main():
    cfg = load_config(DRIFT_CONFIG)

    with open(TIMELAPSE_POS, "r") as f:
        pos_data = json.load(f)
    positions = pos_data["POSITIONS"]
    sample_labels = [p["LABEL"] for p in positions if p["LABEL"] != BG_LABEL]
    print(f"timelapse.pos: {len(positions)} positions  "
          f"sample: {len(sample_labels)}  BG: {BG_LABEL}", flush=True)
    print(f"REF_GRID_DIR: {REF_GRID_DIR}", flush=True)

    # Step 1: Reconstruct BG (Pos0) without BG subtraction
    print(f"\n--- Step 1: Reconstruct BG ({BG_LABEL}) ---", flush=True)
    bg_center = _find_center_dir(REF_GRID_DIR, BG_LABEL)
    bg_phase_path = None
    if bg_center is None:
        print(f"  WARNING: {BG_LABEL} center dir not found in {REF_GRID_DIR}", flush=True)
        print(f"  Other Pos will be reconstructed without BG subtraction", flush=True)
    else:
        try:
            bg_phase_path = ensure_reconstructed(
                bg_center, CALIB_Z_INDEX, bg_raw=None, pos_num=0, cfg=cfg)
        except FileNotFoundError as e:
            print(f"  WARNING: {e}", flush=True)

    # Step 2: Process each sample Pos
    results = []
    for label in sample_labels:
        print(f"\n--- {label} ---", flush=True)
        center_dir = _find_center_dir(REF_GRID_DIR, label)
        if center_dir is None:
            print(f"  ERROR: center dir not found for {label} in {REF_GRID_DIR}", flush=True)
            results.append({"label": label, "status": "center_dir_not_found"})
            continue

        print(f"  center: {center_dir.name}", flush=True)
        m = re.match(r"Pos(\d+)", label)
        pos_num = int(m.group(1)) if m else 1

        # Reconstruct
        bg_raw = None
        if bg_phase_path is not None:
            # Find raw BG file for reconstruction (not the phase)
            bg_raw_candidate = (
                bg_phase_path.parent.parent  # center_dir of Pos0
                / f"img_000000000_ph_{CALIB_Z_INDEX:03d}.tif"
            )
            bg_raw = bg_raw_candidate if bg_raw_candidate.exists() else None

        try:
            ensure_reconstructed(center_dir, CALIB_Z_INDEX, bg_raw, pos_num, cfg)
        except FileNotFoundError as e:
            print(f"  ERROR: reconstruction failed: {e}", flush=True)
            results.append({"label": label, "status": "recon_failed", "error": str(e)})
            continue

        output_phase_dir = center_dir / "output_phase"
        ok = run_channel_detect(output_phase_dir, CALIB_Z_INDEX)
        results.append({"label": label,
                        "center": center_dir.name,
                        "status": "ok" if ok else "detect_failed"})

    # Summary
    print(f"\n{'='*50}", flush=True)
    print(f"Summary:", flush=True)
    for r in results:
        status = r["status"]
        label = r["label"]
        center = r.get("center", "?")
        print(f"  {label:8s}  {center:20s}  {status}", flush=True)
    n_ok = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{n_ok}/{len(results)} Pos completed successfully", flush=True)
    if n_ok < len(results):
        print("Check errors above before running bsh or align_timelapse_pos.py", flush=True)


if __name__ == "__main__":
    main()
