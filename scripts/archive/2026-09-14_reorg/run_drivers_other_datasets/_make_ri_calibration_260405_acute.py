"""One-off: emit a manually-constructed RI calibration JSON for the
260405_acute_z18_200h dataset, using the values the user supplied (wo_2 =
1.33503, wo_0 = 1.33274) plus a linearly-interpolated wo_0p01 entry.

The JSON schema matches what ri_calibration.load_calibration() and
calibrate_ri.py expect. Writes to
<GRID_2PER_DIR>/ri_calibration_results.json by convention.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

GRID_2PER_DIR = Path(r"F:\260405_acute_z18_200h\grid_2pergluc_60ms_1")
GRID_0PER_DIR = Path(r"F:\260405_acute_z18_200h\grid_0pergluc_60ms_1")

N_MILIQ = 1.3312     # literature, 658 nm, 25 C
N_ETOH = 1.3588      # literature, 100% ethanol, 658 nm, 25 C

WO_2 = 1.33503       # user-specified, 2% glucose
WO_0 = 1.33274       # user-specified, 0% glucose

# 0.01% glucose: linear interpolation by glucose concentration
GLUC_FRAC = 0.01 / 2.0
WO_0P01 = WO_0 + GLUC_FRAC * (WO_2 - WO_0)


def main() -> None:
    if not GRID_2PER_DIR.is_dir():
        raise SystemExit(f"GRID_2PER_DIR not found: {GRID_2PER_DIR}")

    now_local = datetime.now(timezone.utc).astimezone()
    timestamp_iso = now_local.isoformat(timespec="seconds")
    timestamp_id = now_local.strftime("%Y%m%dT%H%M%S")
    calibration_id = f"260405_acute_z18_200h_manual_{timestamp_id}"

    entry = {
        "calibration_id": calibration_id,
        "calibrated_at": timestamp_iso,
        "session": "260405_acute_z18_200h",
        "method": "manual (user-supplied wo_2 / wo_0; wo_0p01 linearly interpolated by glucose concentration)",
        "wavelength_nm": 658.0,
        "git_commit": None,
        "reference": {
            "n_miliq": N_MILIQ,
            "n_etoh": N_ETOH,
            "source": "literature @658 nm 25 C",
        },
        "media": {
            "wo_milliq": N_MILIQ,
            "wo_0":      WO_0,
            "wo_0p01":   WO_0P01,
            "wo_2":      WO_2,
            "wo_etoh":   N_ETOH,
        },
        "raw": None,
        "exclusions": {
            "skip_edge_channels": True,
            "excluded_pos": [],
        },
        "channel_depth_um": None,
        "config": {
            "grid_2per_dir": str(GRID_2PER_DIR),
            "grid_0per_dir": str(GRID_0PER_DIR),
            "glucose_interpolation": {
                "wo_0_glucose_pct": 0.0,
                "wo_2_glucose_pct": 2.0,
                "wo_0p01_glucose_pct": 0.01,
                "linear_in": "glucose_pct",
            },
        },
        "per_pos": [],
        "notes": (
            "Manually constructed for 260405_acute_z18_200h. wo_2 (1.33503) "
            "and wo_0 (1.33274) come from user knowledge of medium RI at "
            "658 nm. wo_0p01 = wo_0 + (0.01/2.0) * (wo_2 - wo_0) — a linear "
            "interpolation by glucose concentration. n_miliq is used as the "
            "protein-density baseline (frame-independent)."
        ),
    }

    out_path = GRID_2PER_DIR / "ri_calibration_results.json"
    history = {
        "schema_version": "1.0",
        "active": calibration_id,
        "calibrations": [entry],
    }
    out_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"  calibration_id: {calibration_id}")
    print(f"  wo_2:    {WO_2}")
    print(f"  wo_0p01: {WO_0P01:.7f}")
    print(f"  wo_0:    {WO_0}")
    print(f"  n_miliq: {N_MILIQ}")


if __name__ == "__main__":
    main()
