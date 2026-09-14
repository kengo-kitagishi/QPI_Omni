"""bench_recalibrate.py -- regenerate grid_calibration with a swapped ECC estimator

Drives calibrate_grid_positions.py in-process (it uses ThreadPoolExecutor, so a
parent-process monkeypatch DOES reach the work) to regenerate the per-grid-point
offset table with a different per-channel estimator and/or z plane, WITHOUT
editing production source and WITHOUT overwriting the production
grid_calibration_*.json (output goes to an explicit isolated path).

Motivation: the production grid_calibration was measured with ECC on uint8 input
at grid z=5. ECC-uint8 carries a ~+0.087px systematic X bias (see the
ground-truth bench), which is baked into every grid point's offset and shows up
as a SYSTEMATIC offset in the subtraction. Re-measuring with float ECC (bias
removed) at the correct z plane should remove that systematic offset.

Estimators:
  ecc_float : float32 input to ecc_align (uint8 quantization bypassed)  [default]
  sg        : NCC + SG subpixel peak (Qiita) on float32
  ecc       : uint8 ECC (current production behaviour; for control)

Usage:
    python scripts/bench_recalibrate.py --estimator ecc_float --grid-z 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_subtract_ab import sg_ncc_align

DEFAULT_CONFIG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--pos", type=int, default=1)
    p.add_argument("--grid-z", type=int, default=8)
    p.add_argument("--estimator", default="ecc_float",
                   choices=["ecc_float", "sg", "ecc"])
    p.add_argument("--out", default=None,
                   help="isolated output json (default: <save_dir>/../ecc_sg_ab/"
                        "grid_calibration_Pos{N}_z{Z}_{estimator}.json)")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pos, gz = args.pos, args.grid_z
    grid_dir = cfg["grid_dir"]

    out = (Path(args.out) if args.out else
           Path(cfg["save_dir"]).parent / "ecc_sg_ab"
           / f"grid_calibration_Pos{pos}_z{gz}_{args.estimator}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Guard: never write into the grid dir (no overwrite of production calibration).
    if Path(out).resolve().parent == Path(grid_dir).resolve():
        raise RuntimeError("refusing to write calibration into the grid dir (no overwrite)")

    import calibrate_grid_positions as cgp
    cgp.GRID_DIR = grid_dir
    cgp.BASE_LABEL = f"Pos{pos}"
    cgp.GRID_Z_INDEX = gz
    cgp.POS_SPLIT = cfg["pos_split"]
    cgp.TILT_CROP_H = cfg["tilt_crop_h"]
    cgp.ECC_CROP_H = cfg["ecc_crop_h"]
    cgp.VMIN, cgp.VMAX = cfg["ecc_vmin"], cfg["ecc_vmax"]
    cgp.X_STEP = cfg["crop_sub_x_step_um"]
    cgp.Y_STEP = cfg["crop_sub_y_step_um"]
    cgp.OUTPUT_JSON = str(out)
    cgp.N_GRID_THREADS = 8
    cgp._save_calibration_figures = lambda *a, **k: None  # skip Drive figure writes

    # --- swap the estimator (ThreadPool -> in-process monkeypatch works) ---
    if args.estimator in ("ecc_float", "sg"):
        # bypass uint8 quantization: feed float32 crops to the estimator
        cgp.to_uint8 = lambda img, vmin=None, vmax=None: img.astype(np.float32)
    if args.estimator == "sg":
        cgp.ecc_align = sg_ncc_align

    print(f"Recalibrate Pos{pos}  grid_z={gz}  estimator={args.estimator}")
    print(f"  grid_dir: {grid_dir}")
    print(f"  output (isolated): {out}\n")
    cgp.main()
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
