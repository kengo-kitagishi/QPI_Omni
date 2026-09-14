"""Inspect grid_calibration error_DY across every Pos in the
spawn-style figure-hub run, and decompose the residual into:
 - mean (DC bias on assumed pixel_scale)
 - slope vs xi (cross-coupling, camera-stage rotation)
 - slope vs yi (sensitivity to the other axis)
 - rms residual (random ECC noise)
The "real" pixel_scale_um implied by the slope of measured_dy vs the
nominal driver axis (xi) gives a direct check on the assumed value.

Targets the data .npz files saved by calibrate_grid_positions.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

INBOX_RUN = Path(
    r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
    r"\2026-05-19\spawn\20260519T060323Z_825f2a"
)


def main() -> None:
    if not INBOX_RUN.is_dir():
        print(f"[FATAL] missing run dir: {INBOX_RUN}", file=sys.stderr)
        sys.exit(1)

    dy_npz = sorted(INBOX_RUN.glob("spawn__Grid_calibration_error_DY_*_data.npz"))
    print(f"[info] {len(dy_npz)} Pos with error_DY data")

    rows = []
    for npz in dy_npz:
        # Pos label is embedded in the filename:
        # spawn__Grid_calibration_error_DY_<POS>__<run>__fNNN_data.npz
        stem = npz.stem
        # Find "<POS>__"
        try:
            pos_label = stem.split("error_DY_", 1)[1].split("__", 1)[0]
        except IndexError:
            pos_label = "?"
        try:
            d = np.load(npz)
            err_dy = d["err_dy_um"]
            xi_vals = d["xi_vals"]
            yi_vals = d["yi_vals"]
        except Exception as e:
            print(f"  [warn] {pos_label}: load failed {e}", file=sys.stderr)
            continue

        finite = np.isfinite(err_dy)
        if not finite.any():
            continue
        # Slope per axis
        XI, YI = np.meshgrid(xi_vals, yi_vals)  # err_dy[iy, ix] indexing
        x_flat = XI[finite].astype(float)
        y_flat = YI[finite].astype(float)
        e_flat = err_dy[finite].astype(float)
        slope_xi = float(np.polyfit(x_flat, e_flat, 1)[0])  # um per xi-step
        slope_yi = float(np.polyfit(y_flat, e_flat, 1)[0])
        rows.append({
            "pos": pos_label,
            "n_pts": int(finite.sum()),
            "mean_um": float(np.mean(e_flat)),
            "rms_um": float(np.sqrt(np.mean(e_flat**2))),
            "abs_max_um": float(np.max(np.abs(e_flat))),
            "slope_err_dy_per_xi_um": slope_xi,
            "slope_err_dy_per_yi_um": slope_yi,
        })

    df = pd.DataFrame(rows).sort_values("pos", key=lambda s: s.str.extract(r"(\d+)").astype(int).iloc[:, 0])
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    print(df.to_string(index=False))

    print("\n--- summary across Pos ---")
    for col in ("mean_um", "rms_um", "abs_max_um", "slope_err_dy_per_xi_um", "slope_err_dy_per_yi_um"):
        v = df[col].values
        print(f"{col:>32s}: median={np.median(v):+.4f}  "
              f"mean={np.mean(v):+.4f}  std={np.std(v):.4f}")

    # Sample one Pos json to recover the assumed pixel_scale_um
    sample_json = next(INBOX_RUN.glob("spawn__Grid_calibration_error_DY_*.json"), None)
    if sample_json:
        meta = json.loads(sample_json.read_text(encoding="utf-8"))
        psc = meta.get("params", {}).get("pixel_scale_um")
        if psc:
            print(f"\nassumed pixel_scale_um = {psc:.6f} um/px")

    # The slope of err_dy vs xi tells us how much extra (or missing) drift in dy per xi step.
    # X_STEP=Y_STEP=0.1 um (per grid_subtract.py); SHIFT_SIGN_X=SHIFT_SIGN_Y=-1.
    # Nominal model:  nominal_dy_px = SHIFT_SIGN_X * xi * X_STEP / pixel_scale_um
    # Empirical measured_dy_px = (nominal + err_dy_px) where err_dy_px = err_dy_um / pixel_scale_um.
    # If the *true* pixel_scale_um is psc_true, then measured_dy_um per xi-step = -X_STEP * (psc_assumed/psc_true).
    # So:  err_dy_um per xi  =  measured_dy_um - nominal_dy_um  =  -X_STEP * (psc_assumed/psc_true - 1).
    # =>  psc_true / psc_assumed = X_STEP / (X_STEP + (-err_dy_um/per_xi_slope)*(-1)).
    # Easier: |slope_err_dy_per_xi_um| should be 0 if scale & rotation are perfect.
    # When non-zero, it conflates scale error AND rotation error (stage X -> camera Y leak).
    # The pure-scale ratio comes from |slope of measured_dy_um per yi step| compared to Y_STEP.
    print("\nNote: slope_err_dy_per_xi_um is mostly a stage-vs-camera ROTATION signature;"
          "\n      pure pixel-scale error shows up as a deviation in measured_dy_um per yi from Y_STEP.")


if __name__ == "__main__":
    main()
