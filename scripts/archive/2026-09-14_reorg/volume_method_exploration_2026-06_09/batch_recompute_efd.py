"""batch_recompute_efd.py — recompute the gold-standard + phase1-dead cohort
with the adopted EFD-contour-intersection volume column added.

Reuses recompute_axes_from_masks.recompute_channel (mask-direct, threaded HDD
reads within a channel; channels processed one at a time = single-stream HDD).
Only the medial_axis mode is requested (it carries the efd_* columns), and only
phase1 frames (<= frame-max) are processed since the overlay is phase1-only.

Output: results/260517/recomputed_axes_efd/<pos>_<ch>.csv  (resumable: skips
existing). Does NOT touch the existing recomputed_axes/ CSVs.

Usage:
    python scripts/batch_recompute_efd.py                 # all 51 channels
    python scripts/batch_recompute_efd.py --frame-max 2018 --read-workers 4
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from recompute_axes_from_masks import recompute_channel  # noqa: E402
from qpi_paths import results_dir  # noqa: E402
import overlay_gold_standard_and_phase1_dead as ov  # noqa: E402


def cohort() -> list[tuple[str, str]]:
    gold = ov.select_gold_standard()
    dead = [(p, c) for p, c, _ in ov.select_phase1_dead_sorted_by_death()]
    seen, out = set(), []
    for pc in list(gold) + dead:
        if pc not in seen:
            seen.add(pc)
            out.append(pc)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame-max", type=int, default=2018,
                    help="phase1 end frame (<=168 h); overlay is phase1-only")
    ap.add_argument("--read-workers", type=int, default=4)
    ap.add_argument("--channels", nargs="+", default=None,
                    help="explicit Pos_ch list (default: gold + phase1-dead cohort)")
    ap.add_argument("--preview", action="store_true",
                    help="small subset: 3 gold + 3 phase1-dead (early/mid/late "
                         "death) to gauge how much EFD changes the figure")
    ap.add_argument("--revived", action="store_true",
                    help="phase2 survivors (revived mothers) — full timecourse")
    ap.add_argument("--full-cohort", action="store_true",
                    help="union of revived + never_revived + phase2_dead (full timecourse)")
    ap.add_argument("--out-dir", default=None,
                    help="output dir (default recomputed_axes_efd; use "
                         "recomputed_axes_efd_full for the full-timecourse revived run)")
    args = ap.parse_args()

    if args.channels:
        chans = [tuple(t.split("_", 1)) for t in args.channels]
    elif args.preview:
        gold = ov.select_gold_standard()[:3]
        dead = [(p, c) for p, c, _ in ov.select_phase1_dead_sorted_by_death()]
        pick_dead = [dead[0], dead[len(dead) // 2], dead[-1]] if len(dead) >= 3 else dead
        chans = list(gold) + pick_dead
    elif args.revived:
        import overlay_mean_sd_band_full_timecourse as ov2
        chans = ov2.list_revived_mothers()
    elif args.full_cohort:
        import overlay_mean_sd_band_full_timecourse as ov2
        seen, chans = set(), []
        groups = [ov2.list_revived_mothers(), ov2.list_never_revived_mothers()]
        try:
            import grid_phase2_dead_mother_individual as gpd
            groups.append([tuple(x) for x in gpd.list_phase2_dead_channels()])
        except Exception:
            pass
        for g in groups:
            for pc in g:
                if pc not in seen:
                    seen.add(pc); chans.append(pc)
    else:
        chans = cohort()
    out_dir = Path(args.out_dir) if args.out_dir else results_dir() / "recomputed_axes_efd"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"cohort: {len(chans)} channels -> {out_dir}")

    t_all = time.time()
    done = skipped = failed = 0
    for i, (pos, ch) in enumerate(chans, 1):
        out_path = out_dir / f"{pos}_{ch}.csv"
        if out_path.exists():
            skipped += 1
            print(f"[{i}/{len(chans)}] {pos}_{ch}: skip (exists)")
            continue
        t0 = time.time()
        try:
            df = recompute_channel(pos, ch, modes=["medial_axis"],
                                   frame_max=args.frame_max,
                                   read_workers=args.read_workers,
                                   progress_every=500)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(chans)}] {pos}_{ch}: FAILED {e}", file=sys.stderr)
            continue
        if df is None or df.empty:
            failed += 1
            print(f"[{i}/{len(chans)}] {pos}_{ch}: no output", file=sys.stderr)
            continue
        df.to_csv(out_path, index=False)
        done += 1
        print(f"[{i}/{len(chans)}] {pos}_{ch}: {df['frame'].nunique()} frames "
              f"in {time.time()-t0:.0f}s -> {out_path.name}", flush=True)

    print(f"\nDONE: {done} written, {skipped} skipped, {failed} failed in "
          f"{(time.time()-t_all)/60:.1f} min -> {out_dir}")


if __name__ == "__main__":
    main()
