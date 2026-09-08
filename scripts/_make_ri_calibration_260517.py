"""One-off: emit a manually-constructed RI calibration JSON for the
260517 2per_0055per_0per_2per dataset.

Reuses the lab-standard medium RI values at 658 nm (wo_2 = 1.33503,
wo_0 = 1.33274) that were measured for 260405 and already reused for the
260426 dataset. Adds a `wo_0p0055` token for the 0.0055% glucose epoch:
per user instruction its RI is identical to wo_0, but the distinct token
keeps the `media_name` column distinguishable downstream (the tracker's
medium_name_at_frame returns the token verbatim without touching media_ri).

Schema matches ri_calibration.load_calibration() / calibrate_ri.py.
Writes to <GRID_2PER_DIR>/ri_calibration_results.json by convention.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

GRID_2PER_DIR = Path(r"F:\260517\grid_2pergluc_2")

N_MILIQ = 1.3312     # literature, 658 nm, 25 C
N_ETOH = 1.3588      # literature, 100% ethanol, 658 nm, 25 C

WO_2 = 1.33503       # lab standard, 2% glucose (reused from 260405)
WO_0 = 1.33274       # lab standard, 0% glucose (reused from 260405)

# 0.0055% glucose epoch: per user, use the SAME RI as 0% glucose. Only the
# media_name label (wo_0p0055) differs so the period stays distinguishable.
WO_0P0055 = WO_0


def main() -> None:
    if not GRID_2PER_DIR.is_dir():
        raise SystemExit(f"GRID_2PER_DIR not found: {GRID_2PER_DIR}")

    now_local = datetime.now(timezone.utc).astimezone()
    timestamp_iso = now_local.isoformat(timespec="seconds")
    timestamp_id = now_local.strftime("%Y%m%dT%H%M%S")
    calibration_id = f"260517_2per_0055per_0per_2per_manual_{timestamp_id}"

    entry = {
        "calibration_id": calibration_id,
        "calibrated_at": timestamp_iso,
        "session": "260517_2per_0055per_0per_2per",
        "method": (
            "manual (lab-standard wo_2 / wo_0 reused from 260405; "
            "wo_0p0055 = wo_0 per user — distinct label, identical RI)"
        ),
        "wavelength_nm": 658.0,
        "git_commit": None,
        "reference": {
            "n_miliq": N_MILIQ,
            "n_etoh": N_ETOH,
            "source": "literature @658 nm 25 C",
        },
        "media": {
            "wo_milliq":  N_MILIQ,
            "wo_0":       WO_0,
            "wo_0p0055":  WO_0P0055,
            "wo_2":       WO_2,
            "wo_etoh":    N_ETOH,
        },
        "raw": None,
        "exclusions": {
            "skip_edge_channels": True,
            "excluded_pos": [],
        },
        "channel_depth_um": None,
        "config": {
            "grid_2per_dir": str(GRID_2PER_DIR),
            "media_schedule_example": "0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2",
            "glucose_epochs": {
                "wo_2_glucose_pct": 2.0,
                "wo_0p0055_glucose_pct": 0.0055,
                "wo_0_glucose_pct": 0.0,
            },
        },
        "per_pos": [],
        "notes": (
            "Manually constructed for 260517. wo_2 (1.33503) and wo_0 "
            "(1.33274) are the lab-standard medium RI at 658 nm reused from "
            "260405 (and 260426). The 0.0055% glucose epoch uses wo_0p0055, "
            "whose RI equals wo_0 by user instruction; the separate token "
            "only keeps the media_name column distinguishable. n_miliq is the "
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
    print(f"  wo_2:       {WO_2}")
    print(f"  wo_0p0055:  {WO_0P0055}  (== wo_0)")
    print(f"  wo_0:       {WO_0}")
    print(f"  n_miliq:    {N_MILIQ}")


if __name__ == "__main__":
    main()
