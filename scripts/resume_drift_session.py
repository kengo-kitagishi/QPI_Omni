"""resume_drift_session.py -- rewrite the drift config without losing the run.

`prepare_drift_session.py` re-initialises drift_state and drift_log, which is what
you want for a fresh run and exactly what you must not do when continuing an
acquisition: the BeanShell reads TIMEPOINT and the per-Pos CUMULATIVE_DX/DY_UM
out of the state file to pick up where it left off, and wiping them restarts at
T=0 with the stage corrections thrown away.

This script lets a parameter change (a new estimator, a new threshold) reach the
config while the run continues:

  1. copy drift_state / drift_log aside
  2. run prepare_drift_session.py (new config)
  3. put the state and log back, with TIMEPOINT set so the BeanShell resumes at
     the requested timepoint (it starts at TIMEPOINT + 1)

Run it only while the acquisition is STOPPED -- the BeanShell rereads the state
file every timepoint, and prepare_drift_session truncates it mid-flight.

Afterwards, in realtime_drift_mda_zstack.bsh:
    FORCE_FRESH_START = false        <- otherwise the resume block is skipped

Usage
-----
    python scripts/resume_drift_session.py --resume-at 27
    python scripts/resume_drift_session.py --resume-at 27 --dry-run
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SESSION_DIR = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session")
SUFFIX = "_zstack"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume-at", type=int, required=True,
                   help="timepoint the acquisition should continue from")
    p.add_argument("--session-dir", default=str(SESSION_DIR))
    p.add_argument("--suffix", default=SUFFIX)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def read_state(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def set_timepoint(text: str, timepoint: int) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("TIMEPOINT="):
            lines[i] = f"TIMEPOINT={timepoint}"
            break
    else:
        raise RuntimeError("state file has no TIMEPOINT line")
    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    session = Path(args.session_dir)
    state_path = session / f"drift_state{args.suffix}.txt"
    log_path = session / f"drift_log{args.suffix}.json"
    cfg_path = session / f"drift_config{args.suffix}.json"
    script_dir = Path(__file__).resolve().parent

    if not state_path.exists():
        raise FileNotFoundError(f"no state file at {state_path}; nothing to resume")

    state_txt = state_path.read_text(encoding="utf-8")
    state = read_state(state_path)
    last_tp = int(state.get("TIMEPOINT", -1))
    n_pos = sum(1 for k in state if k.startswith("CUMULATIVE_DX_UM_"))
    print(f"current state : TIMEPOINT={last_tp}  STATUS={state.get('STATUS')}  "
          f"per-Pos entries={n_pos}")
    print(f"requested     : resume at T={args.resume_at}  "
          f"(state TIMEPOINT will be set to {args.resume_at - 1})")
    if n_pos == 0:
        raise RuntimeError("state carries no per-Pos cumulative corrections; "
                           "resuming would restart the stage corrections from zero")

    old_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backups = {}
    for p in (state_path, log_path):
        if p.exists():
            b = p.with_name(f"{p.stem}_preresume_{stamp}{p.suffix}")
            backups[p] = b
            if not args.dry_run:
                shutil.copy2(p, b)
            print(f"backup        : {p.name} -> {b.name}")

    if args.dry_run:
        print("\n--dry-run: prepare_drift_session.py NOT run, nothing restored")
        return

    print("\n--- prepare_drift_session.py ---")
    rc = subprocess.run([sys.executable, str(script_dir / "prepare_drift_session.py")],
                        cwd=str(script_dir.parent)).returncode
    if rc != 0:
        # The backups still hold the live run; restore them before giving up.
        for p, b in backups.items():
            shutil.copy2(b, p)
        raise SystemExit(f"prepare_drift_session.py failed (exit {rc}); state restored")

    state_path.write_text(set_timepoint(state_txt, args.resume_at - 1), encoding="utf-8")
    print(f"\nstate restored: TIMEPOINT={args.resume_at - 1} "
          f"(+ {n_pos} per-Pos cumulative corrections)")
    if log_path in backups:
        shutil.copy2(backups[log_path], log_path)
        print(f"log restored  : {log_path.name} (new records append to the old ones)")

    new_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    changed = {k: (old_cfg.get(k), new_cfg.get(k))
               for k in sorted(set(old_cfg) | set(new_cfg))
               if old_cfg.get(k) != new_cfg.get(k)}
    print("\nconfig changes:")
    for k, (a, b) in changed.items():
        print(f"  {k}: {a!r} -> {b!r}")
    if not changed:
        print("  (none)")

    print(f"\nNext: set FORCE_FRESH_START = false in realtime_drift_mda_zstack.bsh, "
          f"then Run.\n      The BeanShell will print '[RESUME] Resuming from T={args.resume_at}'.")


if __name__ == "__main__":
    main()
