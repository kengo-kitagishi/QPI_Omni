"""Resilient driver for Phase B (Pos47-104): repeatedly runs the GPU chain
(_chain_seg_260517.py) until every position is complete or max attempts is hit.

The GPU segmentation crashes intermittently (driver / BrokenProcessPool /
access-violation). Each chain run skips already-complete positions (per the
chain's per-channel already_done) and processes the rest; if it dies mid-way,
this wrapper sees remaining > 0 and restarts it. Self-heals through soft crashes
that kill the chain but not this wrapper. A hard crash that kills all python
(e.g. PC reboot) stops this too — recover by re-launching this script.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
CHAIN = SCRIPTS / "_chain_seg_260517.py"
LOG = SCRIPTS / "_chain_resilient_260517.log"

# import the chain module (defines ROOT, Z, already_done) without running it
_spec = importlib.util.spec_from_file_location("chain260517", str(CHAIN))
chain = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(chain)


def _log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def remaining(start: int, end: int) -> list[int]:
    out = []
    for n in range(start, end + 1):
        z = chain.ROOT / f"Pos{n}" / "output_phase" / "channels" / "crop_sub_rawraw" / chain.Z
        if z.is_dir() and not chain.already_done(z):
            out.append(n)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=47)
    ap.add_argument("--end", type=int, default=104)
    ap.add_argument("--ch-workers", type=int, default=1)
    ap.add_argument("--max-attempts", type=int, default=40)
    ap.add_argument("--sleep", type=int, default=30)
    args = ap.parse_args()

    LOG.write_text(f"=== resilient Phase B START {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n",
                   encoding="utf-8")
    for attempt in range(1, args.max_attempts + 1):
        rem = remaining(args.start, args.end)
        _log(f"attempt {attempt}/{args.max_attempts}: {len(rem)} Pos remaining "
             f"(first: {rem[:12]})")
        if not rem:
            _log("ALL DONE")
            break
        cmd = [sys.executable, "-u", str(CHAIN),
               "--start", str(args.start), "--end", str(args.end),
               "--ch-workers", str(args.ch_workers)]
        _log(f"launching chain: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        _log(f"attempt {attempt} chain exited rc={rc}")
        # detect a position that did not advance (deterministic crash) for visibility
        still = remaining(args.start, args.end)
        if still and still == rem:
            _log(f"WARNING: no progress this attempt; lead Pos {still[0]} may crash deterministically")
        time.sleep(args.sleep)
    else:
        _log(f"GAVE UP after {args.max_attempts} attempts")
    _log(f"final remaining: {remaining(args.start, args.end)}")
    _log("=== resilient Phase B END ===")


if __name__ == "__main__":
    main()
