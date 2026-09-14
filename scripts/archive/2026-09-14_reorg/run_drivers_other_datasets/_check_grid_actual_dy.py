"""Load the raw err_dy_um array and reconstruct the *actual* dy_um per xi
to settle the question: how many um does the image move per commanded
0.1 um stage step? The answer dictates whether pixel_scale_um is off."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

INBOX_RUN = Path(
    r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
    r"\2026-05-19\spawn\20260519T060323Z_825f2a"
)
X_STEP = 0.1  # commanded step (um) per grid_subtract.py
SHIFT_SIGN_X = -1
SHIFT_SIGN_Y = -1


def main() -> None:
    rows = []
    for npz_path in sorted(INBOX_RUN.glob("spawn__Grid_calibration_error_DY_*_data.npz")):
        pos_label = npz_path.stem.split("error_DY_", 1)[1].split("__", 1)[0]
        json_path = next(INBOX_RUN.glob(
            f"spawn__Grid_calibration_error_DY_{pos_label}__*.json"), None)
        psc_assumed = None
        if json_path:
            meta = json.loads(json_path.read_text(encoding="utf-8"))
            psc_assumed = meta["params"]["pixel_scale_um"]

        d = np.load(npz_path)
        err_dy_um = d["err_dy_um"]
        xi_vals = d["xi_vals"]
        yi_vals = d["yi_vals"]
        XI, _YI = np.meshgrid(xi_vals, yi_vals)

        # nominal_dy_um = SHIFT_SIGN_X * xi * X_STEP   (the model's assumption)
        nominal_dy_um = SHIFT_SIGN_X * XI * X_STEP
        actual_dy_um = nominal_dy_um + err_dy_um  # because err = actual - nominal

        finite = np.isfinite(err_dy_um)
        x_flat = XI[finite].astype(float)
        nom_flat = nominal_dy_um[finite].astype(float)
        act_flat = actual_dy_um[finite].astype(float)
        err_flat = err_dy_um[finite].astype(float)

        slope_act = float(np.polyfit(x_flat, act_flat, 1)[0])
        slope_nom = float(np.polyfit(x_flat, nom_flat, 1)[0])
        slope_err = float(np.polyfit(x_flat, err_flat, 1)[0])

        # actual_dy at xi=+4 (the user's "0.4 um commanded" case)
        mask4p = finite & (XI == 4)
        mask4n = finite & (XI == -4)
        mean_act_xi_pos4 = float(np.mean(actual_dy_um[mask4p])) if mask4p.any() else float("nan")
        mean_act_xi_neg4 = float(np.mean(actual_dy_um[mask4n])) if mask4n.any() else float("nan")

        # Empirical pixel_scale assuming X_STEP correct:
        # actual_dy_px per xi = (actual_dy_um per xi) / psc_assumed
        # nominal_dy_px per xi (true) = -X_STEP / psc_true  (with SHIFT=-1)
        # so |psc_true| = X_STEP / |actual_dy_px per xi|
        actual_dy_px_per_xi = slope_act / psc_assumed if psc_assumed else float("nan")
        psc_est = X_STEP / abs(actual_dy_px_per_xi) if actual_dy_px_per_xi else float("nan")

        rows.append({
            "pos": pos_label,
            "psc_assumed_um_px": psc_assumed,
            "nominal_um_per_xi": slope_nom,
            "actual_um_per_xi": slope_act,
            "err_um_per_xi": slope_err,
            "actual_dy_um@xi_+4": mean_act_xi_pos4,
            "actual_dy_um@xi_-4": mean_act_xi_neg4,
            "implied_psc_TRUE_um_px": psc_est,
            "ratio_assumed_over_true": psc_assumed / psc_est if psc_est else float("nan"),
        })

    df = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:+.4f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))
    print()
    for col in ("nominal_um_per_xi", "actual_um_per_xi", "err_um_per_xi",
                "actual_dy_um@xi_+4", "actual_dy_um@xi_-4",
                "implied_psc_TRUE_um_px", "ratio_assumed_over_true"):
        v = df[col].values
        print(f"{col:>32s}: median={np.median(v):+.4f}  mean={np.mean(v):+.4f}  std={np.std(v):.4f}")


if __name__ == "__main__":
    main()
