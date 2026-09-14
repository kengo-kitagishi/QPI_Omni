"""_batch_swelldeath_movies.py — render the QPI swelling/elongation-death movies.

Drives scripts/_animate_vol_ri_f_ch.py over every curated phase1-dead lineage:

  * swelling-death lineages  = DEATH_WINDOW keys, minus ELONGATION, minus the
    gold_standard.MANUAL_EXCLUDE channels -> movie WITH the n-2 non-mother
    daughter tracked;
  * elongation-cascade lineages (ELONGATION = Pos20_ch06, Pos30_ch04) -> the
    same movie WITHOUT the daughter (--no-daughter).

The last-division anchor for each lineage is the curated DEATH_WINDOW[key][1].
Lineage lists are imported (not re-listed) so the single source of truth stays
gold_standard.MANUAL_EXCLUDE / _fig_predeath_growthrate.DEATH_WINDOW.

Run (omnipose env has imageio_ffmpeg + ffmpeg):
  python scripts/_batch_swelldeath_movies.py --dry-run        # preview the list
  python scripts/_batch_swelldeath_movies.py --out-dir D:/swelldeath_movies
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fig_predeath_growthrate import DEATH_WINDOW, ELONGATION  # noqa: E402
from _fig_panelA_cellcycle import data_root  # noqa: E402
from gold_standard import MANUAL_EXCLUDE  # noqa: E402

ANIM = Path(__file__).parent / "_animate_vol_ri_f_ch.py"

# Per-lineage daughter overrides (manual curation; replaces the auto n-2 default
# for that lineage). Keys: "Pos_ch". Values may set any of daughter_div_back /
# daughter_cell_id / daughter_div_frame (passed through to the animation script).
DAUGHTER_OVERRIDE: dict[str, dict] = {
    # 2026-06-25 manual curation (user, lineage by lineage). div_back counts
    # divisions before the curated last division: 2 = default (n-2), 3 = one
    # earlier (n-3), 1 = one later (n-1).
    "Pos45_ch02": {"daughter_div_back": 3},   # n-3 (n-2 track corrupted to 3747)
    "Pos32_ch05": {"daughter_div_back": 1},   # n-1 (one later)
    "Pos26_ch03": {"daughter_div_back": 3},   # n-3 (one earlier)
    "Pos17_ch09": {"daughter_div_back": 3},   # n-3 (one earlier)
    "Pos11_ch06": {"daughter_div_back": 3},   # n-3 (one earlier)
}


def lineage_lists() -> tuple[list[tuple[str, str, int]], list[tuple[str, str, int]]]:
    """(swelling, elongation) as (pos, ch, last_div_frame) tuples."""
    swelling, elong = [], []
    for key, (_f0, f1) in DEATH_WINDOW.items():
        pos, ch = key.split("_", 1)
        if (pos, ch) in MANUAL_EXCLUDE:
            continue
        if key in ELONGATION:
            elong.append((pos, ch, int(f1)))
        else:
            swelling.append((pos, ch, int(f1)))
    swelling.sort()
    elong.sort()
    return swelling, elong


def channel_dir(pos: str, ch: str) -> Path:
    return data_root(pos) / ch


def render(pos: str, ch: str, last_div: int, *, with_daughter: bool,
           out_dir: Path, before_h: float, after_h: float, fps: int,
           skip_existing: bool, scale: dict, override: dict | None = None
           ) -> tuple[str, str]:
    tag = "swelldeath" if with_daughter else "elongdeath"
    out = out_dir / (f"{pos}_{ch}_{tag}_lastdiv_-{before_h:g}h+{after_h:g}h_"
                     f"inferno_{fps}fps.mp4")
    cd = channel_dir(pos, ch)
    if not cd.is_dir():
        return (f"{pos}_{ch}", "skip: channel dir missing")
    if not (cd / "inference_out" / "lineage_out" / "lineage_data3D.csv").exists():
        return (f"{pos}_{ch}", "skip: no lineage CSV")
    if skip_existing and out.exists() and out.stat().st_size > 0:
        return (f"{pos}_{ch}", f"skip: exists ({out.name})")
    cmd = [sys.executable, str(ANIM),
           "--channel-dir", str(cd), "--last-div-frame", str(last_div),
           "--before-h", str(before_h), "--after-h", str(after_h),
           "--fps", str(fps), "--out", str(out),
           "--ri-min", str(scale["ri_min"]), "--ri-max", str(scale["ri_max"]),
           "--mass-min", str(scale["mass_min"]), "--mass-max", str(scale["mass_max"]),
           "--mode-label", "swelling death" if with_daughter
           else "elongation cascade death"]
    if not with_daughter:
        cmd.append("--no-daughter")
    ov = override or {}
    if "daughter_div_back" in ov:
        cmd += ["--daughter-div-back", str(ov["daughter_div_back"])]
    if "daughter_cell_id" in ov:
        cmd += ["--daughter-cell-id", str(ov["daughter_cell_id"])]
    if "daughter_div_frame" in ov:
        cmd += ["--daughter-div-frame", str(ov["daughter_div_frame"])]
    print(f"\n=== {pos}_{ch}  last_div={last_div}  "
          f"{'WITH' if with_daughter else 'NO'} daughter"
          f"{' [override ' + str(ov) + ']' if ov else ''} -> {out.name}",
          flush=True)
    r = subprocess.run(cmd)
    return (f"{pos}_{ch}", "ok" if r.returncode == 0 else
            f"FAIL (exit {r.returncode})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("D:/swelldeath_movies"))
    ap.add_argument("--before-h", type=float, default=30.0)
    ap.add_argument("--after-h", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=12)
    # shared fixed y-scale across all swelling lineages (elongation clips on top).
    # 2026-06-25: user chose the "standard" profile.
    ap.add_argument("--ri-min", type=float, default=1.360)
    ap.add_argument("--ri-max", type=float, default=1.402)
    ap.add_argument("--mass-min", type=float, default=0.0)
    ap.add_argument("--mass-max", type=float, default=50.0)
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip lineages whose output MP4 already exists")
    ap.add_argument("--only", default=None,
                    help="comma-separated 'Pos_ch' keys to render (subset)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the lineage lists and exit")
    args = ap.parse_args()

    swelling, elong = lineage_lists()
    only = set(args.only.split(",")) if args.only else None

    print(f"swelling-death lineages (WITH daughter): {len(swelling)}")
    for pos, ch, f1 in swelling:
        print(f"  {pos}_{ch}  last_div={f1}")
    print(f"\nelongation-cascade lineages (NO daughter): {len(elong)}")
    for pos, ch, f1 in elong:
        print(f"  {pos}_{ch}  last_div={f1}")
    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    scale = {"ri_min": args.ri_min, "ri_max": args.ri_max,
             "mass_min": args.mass_min, "mass_max": args.mass_max}
    results = []
    for pos, ch, f1 in swelling:
        if only and f"{pos}_{ch}" not in only:
            continue
        results.append(render(pos, ch, f1, with_daughter=True,
                              out_dir=args.out_dir, before_h=args.before_h,
                              after_h=args.after_h, fps=args.fps,
                              skip_existing=args.skip_existing, scale=scale,
                              override=DAUGHTER_OVERRIDE.get(f"{pos}_{ch}")))
    for pos, ch, f1 in elong:
        if only and f"{pos}_{ch}" not in only:
            continue
        results.append(render(pos, ch, f1, with_daughter=False,
                              out_dir=args.out_dir, before_h=args.before_h,
                              after_h=args.after_h, fps=args.fps,
                              skip_existing=args.skip_existing, scale=scale))

    print("\n==================== summary ====================")
    for name, status in results:
        print(f"  {name:16s} {status}")
    n_ok = sum(1 for _, s in results if s == "ok")
    print(f"  rendered {n_ok}/{len(results)} (rest skipped/failed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
