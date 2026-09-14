"""_batch_phase2_movies.py — render phase2 (refeed) revival/death movies.

Drives _animate_phase2_cells.py over the user-curated phase2 cohorts, all
anchored at the 2% recovery (frame 2893; window -10..+30 h = frames 2773..3253)
with a shared fixed y-scale (mean RI [1.350, 1.410], dry mass [0, 50]):

  * REVIVED  (multi-cell: mother + other cells until oob) -- clean "all cells
    revived" lineages, Pos spread out;
  * CONTROL  (multi-cell) -- never_revived / died_starvation;
  * ELONG_TIP (mother-only) -- elongated mother where only the tip cell revives.

Run:
  python scripts/_batch_phase2_movies.py --dry-run
  python scripts/_batch_phase2_movies.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _fig_panelA_cellcycle import data_root  # noqa: E402

ANIM = Path(__file__).parent / "_animate_phase2_cells.py"

REFEED = 2893               # 2% recovery anchor (t=0); window 2773..3253
RI_MIN, RI_MAX = 1.350, 1.410
MASS_MIN, MASS_MAX = 0.0, 50.0

# --- user-curated 2026-06-26 ---
REVIVED = [("Pos5", "ch10"), ("Pos6", "ch08"), ("Pos7", "ch00"), ("Pos9", "ch06"),
           ("Pos11", "ch05"), ("Pos20", "ch05"), ("Pos24", "ch10"),
           ("Pos34", "ch03"), ("Pos40", "ch07"), ("Pos44", "ch02")]
CONTROL = [("Pos1", "ch10"),                       # never_revived
           ("Pos3", "ch01"), ("Pos3", "ch05"), ("Pos3", "ch09"),
           ("Pos6", "ch03"), ("Pos22", "ch02"), ("Pos41", "ch01")]  # died_starvation
ELONG_TIP = [("Pos12", "ch03"), ("Pos14", "ch02")]   # mother-only


def channel_dir(pos, ch):
    return data_root(pos) / ch


def render(pos, ch, *, mother_only, tag, label, out_dir, fps, skip_existing):
    out = out_dir / (f"{pos}_{ch}_{tag}_refeed_-10h+30h_inferno_{fps}fps.mp4")
    cd = channel_dir(pos, ch)
    if not (cd / "inference_out" / "lineage_out" / "lineage_data3D.csv").exists():
        return (f"{pos}_{ch}", "skip: no lineage CSV")
    if skip_existing and out.exists() and out.stat().st_size > 0:
        return (f"{pos}_{ch}", "skip: exists")
    cmd = [sys.executable, str(ANIM), "--channel-dir", str(cd),
           "--refeed-frame", str(REFEED),
           "--ri-min", str(RI_MIN), "--ri-max", str(RI_MAX),
           "--mass-min", str(MASS_MIN), "--mass-max", str(MASS_MAX),
           "--fps", str(fps), "--mode-label", label, "--out", str(out)]
    if mother_only:
        cmd.append("--mother-only")
    print(f"\n=== {pos}_{ch}  {tag}  {'mother-only' if mother_only else 'multi'} "
          f"-> {out.name}", flush=True)
    r = subprocess.run(cmd)
    return (f"{pos}_{ch}", "ok" if r.returncode == 0 else f"FAIL {r.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("D:/phase2_movies"))
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--only", default=None, help="comma 'Pos_ch' subset")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    jobs = ([(p, c, False, "p2revival", "phase2 revival") for p, c in REVIVED]
            + [(p, c, False, "p2dead", "phase2 dead (never revived)")
               for p, c in CONTROL]
            + [(p, c, True, "p2elongtip", "phase2 elongation tip-revival")
               for p, c in ELONG_TIP])
    only = set(args.only.split(",")) if args.only else None

    print(f"refeed={REFEED}  window {REFEED-120}..{REFEED+360}  "
          f"RI[{RI_MIN},{RI_MAX}] mass[{MASS_MIN},{MASS_MAX}]")
    for p, c, mo, tag, _ in jobs:
        print(f"  {p}_{c:6s} {tag:11s} {'mother-only' if mo else 'multi-cell'}")
    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p, c, mo, tag, label in jobs:
        if only and f"{p}_{c}" not in only:
            continue
        results.append(render(p, c, mother_only=mo, tag=tag, label=label,
                              out_dir=args.out_dir, fps=args.fps,
                              skip_existing=args.skip_existing))
    print("\n==================== summary ====================")
    for name, status in results:
        print(f"  {name:16s} {status}")
    print(f"  ok {sum(1 for _, s in results if s == 'ok')}/{len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
