"""
RI calibration loader / lookup utilities.

Reads append-history JSON written by calibrate_ri.py:

    {
      "schema_version": "1.0",
      "active": "<calibration_id>",
      "calibrations": [
        {
          "calibration_id": "0per_gluc_20260426-130154",
          "calibrated_at": "2026-04-26T13:01:54+09:00",
          "reference": {"n_miliq": 1.3312, "n_etoh": 1.3588, "source": "..."},
          "media": {"wo_milliq": 1.3312, "wo_0": 1.332744,
                    "wo_2": 1.335029, "wo_etoh": 1.3588},
          "raw": {...}, "config": {...}, "exclusions": {...},
          "channel_depth_um": 4.88,
          "git_commit": "cf8bb2d",
          "notes": ""
        },
        ...
      ]
    }

Provides:
  load_calibration(path, calibration_id=None) -> RICalibration
  parse_media_schedule("0:wo_2,575:wo_0,1439:wo_2") -> [(0,'wo_2'), ...]
  n_medium_at_frame(frame, schedule, media_ri) -> float
  mean_ri_to_protein_mg_ml(mean_ri, n_milliq, alpha) -> float
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_ALPHA_RI = 0.00018  # mL/mg, generic protein


@dataclass
class RICalibration:
    calibration_id: str
    calibrated_at: str
    media: dict[str, float]
    n_miliq: float
    n_etoh: float | None = None
    raw: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    source_path: Path | None = None


def load_calibration(
    path: str | Path,
    calibration_id: str | None = None,
) -> RICalibration:
    """Load one calibration entry from the append-history JSON.

    If `calibration_id` is None, uses the entry pointed to by `active`.
    """
    p = Path(path).expanduser()
    data = json.loads(p.read_text(encoding="utf-8"))

    if "calibrations" not in data:
        # Legacy single-entry schema (from old calibrate_ri.py).
        # Wrap it on the fly so downstream code is uniform.
        entry = _legacy_to_entry(data)
        cals = [entry]
        target_id = entry["calibration_id"]
    else:
        cals = data.get("calibrations") or []
        if not cals:
            raise ValueError(f"No calibrations in {p}")
        target_id = calibration_id or data.get("active")
        if not target_id:
            raise ValueError(
                f"No active calibration in {p} and no calibration_id given"
            )

    entry = next((c for c in cals if c.get("calibration_id") == target_id), None)
    if entry is None:
        ids = [c.get("calibration_id") for c in cals]
        raise KeyError(
            f"calibration_id {target_id!r} not in {p}. Available: {ids}"
        )

    ref = entry.get("reference", {}) or {}
    media = dict(entry.get("media", {}) or {})
    n_miliq = ref.get("n_miliq", media.get("wo_milliq"))
    if n_miliq is None:
        raise ValueError(
            f"Calibration {target_id!r} has no reference.n_miliq nor media.wo_milliq"
        )

    return RICalibration(
        calibration_id=entry["calibration_id"],
        calibrated_at=entry.get("calibrated_at", ""),
        media=media,
        n_miliq=float(n_miliq),
        n_etoh=float(ref["n_etoh"]) if "n_etoh" in ref else None,
        raw=dict(entry.get("raw", {}) or {}),
        meta={
            k: v
            for k, v in entry.items()
            if k not in {"reference", "media", "raw"}
        },
        source_path=p,
    )


def _legacy_to_entry(data: dict) -> dict:
    """Convert old single-entry calibrate_ri.py JSON to the new entry schema."""
    media = {}
    if "n_2per" in data and data["n_2per"] is not None:
        media["wo_2"] = float(data["n_2per"])
    if "n_0per" in data and data["n_0per"] is not None:
        media["wo_0"] = float(data["n_0per"])
    if "n_miliq_ref" in data:
        media.setdefault("wo_milliq", float(data["n_miliq_ref"]))
    if "n_etoh_ref" in data:
        media.setdefault("wo_etoh", float(data["n_etoh_ref"]))
    return {
        "calibration_id": "legacy",
        "calibrated_at": "",
        "reference": {
            "n_miliq": data.get("n_miliq_ref"),
            "n_etoh": data.get("n_etoh_ref"),
        },
        "media": media,
        "raw": {
            "S_miliq_rad_px": data.get("total_miliq_sum"),
            "S_etoh_rad_px": data.get("total_etoh_sum"),
            "V_total_rad_px": data.get("V_total"),
            "n_mask_pixels": data.get("total_mask_pixels"),
        },
    }


def parse_media_schedule(
    schedule: str | Sequence | None,
) -> list[tuple[int, str]]:
    """Parse '0:wo_2,575:wo_0,1439:wo_2' or [(0,'wo_2'),...] into a sorted list.

    Must start at frame 0.
    """
    if schedule is None or schedule == "":
        return []
    if isinstance(schedule, str):
        items: list[tuple[int, str]] = []
        for tok in schedule.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if ":" not in tok:
                raise ValueError(
                    f"media_schedule token {tok!r} must be 'frame:medium'"
                )
            f, name = tok.split(":", 1)
            items.append((int(f.strip()), name.strip()))
    else:
        items = [(int(f), str(name)) for f, name in schedule]
    items.sort(key=lambda x: x[0])
    if not items or items[0][0] != 0:
        raise ValueError(
            f"media_schedule must start at frame 0; got {items!r}"
        )
    return items


def n_medium_at_frame(
    frame: float | int,
    schedule: Sequence[tuple[int, str]],
    media_ri: dict[str, float],
) -> float:
    """Step-function lookup of medium RI at `frame`.

    schedule: sorted list like [(0,'wo_2'), (575,'wo_0'), (1439,'wo_2')]
    media_ri: {'wo_2': 1.335..., 'wo_0': 1.332..., ...}
    """
    if not schedule:
        raise ValueError("Empty media_schedule")
    name = schedule[0][1]
    for f, n in schedule:
        if frame >= f:
            name = n
        else:
            break
    if name not in media_ri:
        raise KeyError(
            f"Medium {name!r} not in calibration media: {sorted(media_ri)}"
        )
    return float(media_ri[name])


def mean_ri_to_protein_mg_ml(
    mean_ri: float,
    n_milliq: float,
    alpha_ri: float = DEFAULT_ALPHA_RI,
) -> float:
    """Protein concentration [mg/mL] using MilliQ as baseline.

        protein = (mean_RI - n_milliq) / alpha
    """
    if not (alpha_ri and alpha_ri > 0):
        return float("nan")
    return (float(mean_ri) - float(n_milliq)) / float(alpha_ri)
