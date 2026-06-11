"""batch_recompute_axes.py — run recompute_axes_from_masks over many channels.

Drives ``recompute_channel`` across a channel set and writes one CSV per channel
under results/260517/recomputed_axes/, plus an index CSV. Mask I/O (per-frame
TIFFs on the external drive) dominates, so channels are processed with a small
thread pool — tifffile releases the GIL during decode.

Usage:
    python scripts/batch_recompute_axes.py --gold-standard
    python scripts/batch_recompute_axes.py --gold-standard --frame-max 2200
    python scripts/batch_recompute_axes.py --channels Pos27_ch06 Pos1_ch05
    python scripts/batch_recompute_axes.py --all-revived          # ~130 ch
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from qpi_paths import results_dir, find_lineage_csv  # noqa: E402
from recompute_axes_from_masks import recompute_channel, DEFAULT_MODES  # noqa: E402
from gold_standard import (  # noqa: E402
    select_gold_standard, list_phase1_survivors, list_all_cells, figure_cohort,
)

FULL_LAST_FRAME = 3747  # a CSV reaching this frame is a complete full-timecourse run


def resolve_channels(args) -> list[tuple[str, str]]:
    if args.channels:
        out = []
        for token in args.channels:
            pos, ch = token.split("_", 1)
            out.append((pos, ch))
        return out
    if args.gold_standard:
        return select_gold_standard(verbose=True)
    if args.all_revived:
        return [pc for pc in list_phase1_survivors() if find_lineage_csv(*pc)]
    if args.all_cells:
        # superset: every status=cells channel with lineage (gold/revived/dead)
        return [pc for pc in list_all_cells() if find_lineage_csv(*pc)]
    if args.figure_cohort:
        # exact union of channels any downstream figure needs (157)
        return [pc for pc in figure_cohort() if find_lineage_csv(*pc)]
    raise SystemExit("specify --gold-standard, --all-revived, --all-cells, "
                     "--figure-cohort, or --channels")


def already_full(out_path: Path) -> bool:
    """True if out_path is a finished full-timecourse recompute (reaches frame 3747)."""
    if not out_path.exists():
        return False
    try:
        import pandas as _pd
        fmax = _pd.read_csv(out_path, usecols=["frame"])["frame"].max()
        return int(fmax) >= FULL_LAST_FRAME
    except Exception:
        return False


def run_one(pos, ch, modes, frame_max, out_dir):
    out_path = out_dir / f"{pos}_{ch}.csv"
    try:
        df = recompute_channel(pos, ch, modes, frame_max, progress_every=0)
    except Exception as e:  # noqa: BLE001
        return (pos, ch, "error", str(e), 0)
    if df is None:
        return (pos, ch, "no_output", "", 0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return (pos, ch, "ok", str(out_path), int(df["frame"].nunique()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-standard", action="store_true")
    ap.add_argument("--all-revived", action="store_true")
    ap.add_argument("--all-cells", action="store_true")
    ap.add_argument("--figure-cohort", action="store_true",
                    help="union of channels all downstream figures need (157)")
    ap.add_argument("--channels", nargs="+")
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--frame-max", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip channels whose full-timecourse CSV already exists")
    args = ap.parse_args()

    channels = resolve_channels(args)
    out_dir = results_dir() / "recomputed_axes"
    if args.skip_existing:
        before = len(channels)
        channels = [(p, c) for p, c in channels
                    if not already_full(out_dir / f"{p}_{c}.csv")]
        print(f"skip-existing: {before - len(channels)} already done, "
              f"{len(channels)} remaining")
    print(f"recomputing {len(channels)} channels -> {out_dir} "
          f"(workers={args.workers}, frame_max={args.frame_max})")

    index_rows = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, p, c, args.modes, args.frame_max, out_dir): (p, c)
                for p, c in channels}
        for fut in as_completed(futs):
            pos, ch, status, info, n_frames = fut.result()
            done += 1
            index_rows.append({"pos": pos, "ch": ch, "status": status,
                               "n_frames": n_frames, "info": info})
            print(f"[{done}/{len(channels)}] {pos}_{ch}: {status} ({n_frames} frames)",
                  file=sys.stderr)

    index = pd.DataFrame(index_rows).sort_values(["pos", "ch"])
    index_path = out_dir / "_index.csv"
    index.to_csv(index_path, index=False)
    n_ok = int((index["status"] == "ok").sum())
    print(f"\ndone: {n_ok}/{len(channels)} ok -> index {index_path}")


if __name__ == "__main__":
    main()
