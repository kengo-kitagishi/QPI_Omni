"""write_corrected_lineage.py — drop-in corrected run-dirs for the mother.

The downstream figure scripts read a lineage_data3D.csv (+ sibling clist.csv)
and use the rank=1 (mother) rows' short_axis_um / long_axis_um / volume_um3_rod
/ mean_ri / mass_pg. This builds corrected copies of those run-dirs where, for
the mother rows only, those columns are replaced by the mask-direct medial-axis
measurement, so the figure scripts produce corrected figures with NO logic
change — only where they resolve the run dir (handled by qpi_paths).

Two volume variants are emitted so they can be compared (user: "do both"):
  - rod     : volume_um3_rod = 2-axis rod formula      (mean_ri/mass from it)
  - profile : volume_um3_rod = solid-of-revolution vol (mean_ri/mass from it)
In each variant the STANDARD column names hold that variant's values, so the
downstream scripts need no column renaming — they just point at the chosen set.

Output: results/260517/corrected_lineage_<variant>/<pos>_<ch>/
            lineage_data3D.csv   (mother corrected; daughters untouched)
            clist.csv            (copied from the original run dir)

Usage:
    python scripts/write_corrected_lineage.py --gold-standard --variant rod
    python scripts/write_corrected_lineage.py --gold-standard --variant profile
    python scripts/write_corrected_lineage.py --all-revived --variant both
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import find_lineage_csv, find_clist_csv, results_dir  # noqa: E402
from gold_standard import (  # noqa: E402
    select_gold_standard, list_phase1_survivors, figure_cohort,
)

# variant -> (volume_src_col, ri_src_col, mass_src_col) in the recompute CSV
VARIANT_SRC = {
    "rod":     ("volume_um3_rod",     "mean_ri",         "mass_pg"),
    "profile": ("volume_profile_um3", "mean_ri_profile", "mass_pg_profile"),
}


def correct_one(pos: str, ch: str, method: str, variant: str,
                rec_dir: Path, base_out: Path) -> str:
    rec_path = rec_dir / f"{pos}_{ch}.csv"
    lin_path = find_lineage_csv(pos, ch)
    if lin_path is None or not rec_path.exists():
        return "missing"
    lin = pd.read_csv(lin_path)
    rec = pd.read_csv(rec_path)
    rec = rec[rec["mode"] == method].set_index("frame")
    if rec.empty:
        return "no_mode"

    vol_src, ri_src, mass_src = VARIANT_SRC[variant]
    out = lin.copy()
    is_mother = out["rank"] == 1

    # map recompute values onto mother frames; keep original where missing
    def _apply(dst_col, src_col):
        if src_col not in rec.columns:
            return
        mapped = out.loc[is_mother, "frame"].map(rec[src_col])
        ok = mapped.notna().to_numpy()
        idx = out.index[is_mother][ok]
        out.loc[idx, dst_col] = mapped.dropna().to_numpy()

    _apply("short_axis_um", "short_axis_um")
    _apply("long_axis_um", "long_axis_um")
    _apply("volume_um3_rod", vol_src)
    _apply("mean_ri", ri_src)
    _apply("mass_pg", mass_src)

    # carry cross-section quality for downstream QC filtering
    out["multi_xsec_frac"] = np.nan
    if "multi_xsec_frac" in rec.columns:
        m = out.loc[is_mother, "frame"].map(rec["multi_xsec_frac"])
        out.loc[out.index[is_mother], "multi_xsec_frac"] = m.to_numpy()

    out_dir = base_out / f"{pos}_{ch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "lineage_data3D.csv", index=False)
    clist_path = find_clist_csv(lin_path)
    if clist_path and clist_path.exists():
        shutil.copy2(clist_path, out_dir / "clist.csv")
    return "ok"


def resolve_channels(args):
    if args.channels:
        return [tuple(t.split("_", 1)) for t in args.channels]
    if args.gold_standard:
        return select_gold_standard(verbose=True)
    if args.all_revived:
        return [pc for pc in list_phase1_survivors() if find_lineage_csv(*pc)]
    if args.figure_cohort:
        return [pc for pc in figure_cohort() if find_lineage_csv(*pc)]
    raise SystemExit("specify --gold-standard, --all-revived, --figure-cohort, or --channels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-standard", action="store_true")
    ap.add_argument("--all-revived", action="store_true")
    ap.add_argument("--figure-cohort", action="store_true")
    ap.add_argument("--channels", nargs="+")
    ap.add_argument("--method", default="medial_axis",
                    choices=["medial_axis", "supersegger_adaptive", "skimage_legacy"])
    ap.add_argument("--variant", default="both", choices=["rod", "profile", "both"])
    args = ap.parse_args()

    channels = resolve_channels(args)
    rec_dir = results_dir() / "recomputed_axes"
    variants = ["rod", "profile"] if args.variant == "both" else [args.variant]
    for variant in variants:
        base_out = results_dir() / f"corrected_lineage_{variant}"
        counts = {}
        for pos, ch in channels:
            st = correct_one(pos, ch, args.method, variant, rec_dir, base_out)
            counts[st] = counts.get(st, 0) + 1
        print(f"[{variant}] -> {base_out}: " +
              "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
