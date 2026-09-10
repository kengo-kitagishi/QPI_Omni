"""rollback_drift_state.py -- undo stage corrections computed from bad frames.

When a timepoint's holograms are unusable (a knock, an earthquake, a bubble), the
reconstruction is garbage, the per-channel match scores collapse, and the drift
loop still computes a correction from whatever it measured -- moving the stage by
up to a few microns on noise. The next healthy frame would have to recapture from
that displaced position, which can exceed the estimator's capture range.

This rewrites drift_state from a known-good timepoint in drift_log, so the
BeanShell's resume block puts every position back where it was before the bad
frame (it sets corrX[i] = baseX[i] - CUMULATIVE_DX_UM_i). Only the numeric values
are replaced; the file's structure and any key this script does not know about
are preserved.

EMA carry-over is not restored: with correction_ema_alpha = 1.0 the previous EMA
is unused. The script refuses to run if the config says otherwise, rather than
silently resuming with a stale filter state.

Run only while the acquisition is STOPPED.

Usage
-----
    python scripts/rollback_drift_state.py --to-timepoint 28
    python scripts/rollback_drift_state.py --to-timepoint 28 --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

SESSION_DIR = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session")
SUFFIX = "_zstack"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--to-timepoint", type=int, required=True,
                   help="timepoint whose cumulative corrections become the new state")
    p.add_argument("--session-dir", default=str(SESSION_DIR))
    p.add_argument("--suffix", default=SUFFIX)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    session = Path(args.session_dir)
    state_path = session / f"drift_state{args.suffix}.txt"
    log_path = session / f"drift_log{args.suffix}.json"
    cfg_path = session / f"drift_config{args.suffix}.json"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    alpha = cfg.get("correction_ema_alpha", 1.0)
    if alpha != 1.0:
        raise SystemExit(
            f"correction_ema_alpha is {alpha}, not 1.0: the EMA carries state this "
            f"script cannot reconstruct from the log. Roll back by hand.")

    log = json.loads(log_path.read_text(encoding="utf-8"))
    recs = [r for r in log if r["timepoint"] == args.to_timepoint]
    if not recs:
        raise SystemExit(f"no log record for T={args.to_timepoint}; "
                         f"have {sorted({r['timepoint'] for r in log})}")
    rec = recs[-1]
    good = {p["pos_idx"]: (p["cumulative_dx_um"], p["cumulative_dy_um"])
            for p in rec["positions"]}
    print(f"T={args.to_timepoint}: {len(good)} positions in the log")

    text = state_path.read_text(encoding="utf-8")
    cur = {}
    for line in text.splitlines():
        m = re.match(r"^CUMULATIVE_D([XY])_UM_(\d+)=(.*)$", line)
        if m:
            cur.setdefault(int(m.group(2)), {})[m.group(1)] = float(m.group(3))
    missing = sorted(set(cur) - set(good))
    if missing:
        print(f"  note: {len(missing)} position(s) in the state have no T="
              f"{args.to_timepoint} record and keep their current value: {missing[:10]}")

    moved = []
    for idx, (dx, dy) in sorted(good.items()):
        if idx in cur:
            moved.append((idx, cur[idx]["X"] - dx, cur[idx]["Y"] - dy))
    moved.sort(key=lambda t: -max(abs(t[1]), abs(t[2])))
    print(f"\nstage will move back by (worst 8 of {len(moved)}):")
    for idx, ddx, ddy in moved[:8]:
        print(f"  Pos{idx:<4d} dx={ddx:+7.3f} dy={ddy:+7.3f} um")
    big = sum(1 for _, ddx, ddy in moved if max(abs(ddx), abs(ddy)) > 0.5)
    print(f"  positions moving more than 0.5 um: {big}")

    def replace(line: str) -> str:
        m = re.match(r"^CUMULATIVE_D([XY])_UM_(\d+)=", line)
        if m and int(m.group(2)) in good:
            axis, idx = m.group(1), int(m.group(2))
            v = good[idx][0 if axis == "X" else 1]
            return f"CUMULATIVE_D{axis}_UM_{idx}={v:.6f}"
        if line.startswith("TIMEPOINT="):
            return f"TIMEPOINT={args.to_timepoint}"
        return line

    new_text = "\n".join(replace(l) for l in text.splitlines()) + "\n"

    if args.dry_run:
        print("\n--dry-run: state not written")
        return

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup = state_path.with_name(f"{state_path.stem}_prerollback_{stamp}.txt")
    shutil.copy2(state_path, backup)
    state_path.write_text(new_text, encoding="utf-8")
    print(f"\nbackup : {backup.name}")
    print(f"written: {state_path.name}  TIMEPOINT={args.to_timepoint}")
    print(f"\nNext: Run the BeanShell (FORCE_FRESH_START must stay false).")
    print(f"      It will print '[RESUME] Resuming from T={args.to_timepoint + 1}' and "
          f"drive every stage back to its T={args.to_timepoint} position.")


if __name__ == "__main__":
    main()
