"""regenerate_all_figures.py — re-run every downstream figure script on corrected data.

Runs each analysis/figure script (everything in the 2026-06-10 inbox EXCEPT
per_channel_figures and batch_figures) with QPI_USE_CORRECTED=1 so they read the
mask-direct corrected mother lineage instead of the biased skimage values. The
plotting logic of those scripts is untouched — only their lineage data source is
redirected (via qpi_paths.resolve_lineage_csv).

For each volume variant (rod / profile) a separate pass is run so the two can be
compared. Figures land in the figure-hub inbox as usual (save_figure).

Usage:
    python scripts/regenerate_all_figures.py --variant rod
    python scripts/regenerate_all_figures.py --variant both
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

# Order matters only for readability; each is independent.
DOWNSTREAM_SCRIPTS = [
    "gold_standard_4way_comparison",      # Task D validation (reads recompute CSVs)
    "gold_standard_rod_corrected_cycle",
    "gold_standard_minor_axis_cycle",
    "gold_standard_phase1_homeostasis",
    "overlay_mean_sd_band_full_timecourse",
    "overlay_mean_sd_band_phase1_normalized_ri",
    "overlay_gold_standard_and_phase1_dead",
    "overlay_elongation_death_pair",
    "grid_revived_mother_individual",
    "grid_phase2_dead_mother_individual",
    "analyze_starvation_entry_cell_cycle",
]


def run_one(script: str, env: dict, timeout: int = 1800):
    path = SCRIPTS_DIR / f"{script}.py"
    if not path.exists():
        return (script, "missing", 0.0, "no such script")
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(SCRIPTS_DIR.parent), env=env,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return (script, "timeout", time.time() - t0, "")
    dt = time.time() - t0
    if proc.returncode == 0:
        return (script, "ok", dt, "")
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return (script, "FAIL", dt, "\n".join(tail[-6:]))


def run_variant(variant: str, scripts: list[str]) -> list:
    env = dict(os.environ)
    env["QPI_USE_CORRECTED"] = "1"
    env["QPI_VOLUME_VARIANT"] = variant
    # keep figures quiet w.r.t. Notion to avoid per-figure network calls
    env.setdefault("QPI_FIGURE_LOGGER_NOTION", "0")
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n===== variant={variant} (QPI_USE_CORRECTED=1) =====", flush=True)
    results = []
    for s in scripts:
        script, status, dt, err = run_one(s, env)
        flag = {"ok": "OK", "FAIL": "FAIL", "timeout": "TIMEOUT",
                "missing": "MISSING"}.get(status, status)
        print(f"  [{flag:7s}] {script}  ({dt:.0f}s)", flush=True)
        if err:
            for line in err.splitlines():
                print(f"           | {line}", flush=True)
        results.append((variant, script, status, dt, err))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="both", choices=["rod", "profile", "both"])
    ap.add_argument("--only", nargs="+", help="run only these scripts (by name)")
    args = ap.parse_args()
    scripts = args.only if args.only else DOWNSTREAM_SCRIPTS
    variants = ["rod", "profile"] if args.variant == "both" else [args.variant]

    all_results = []
    for v in variants:
        all_results += run_variant(v, scripts)

    n_ok = sum(1 for r in all_results if r[2] == "ok")
    print(f"\n===== summary: {n_ok}/{len(all_results)} script-runs OK =====")
    for variant, script, status, dt, _err in all_results:
        if status != "ok":
            print(f"  {status:8s} {variant:8s} {script}")


if __name__ == "__main__":
    main()
