"""_wait_retrack_then_publish_260517.py - wait for the running
_retrack_260517_newmodel.py chain to finish, then publish the master dataset.

Polls the chain log for its END line. If the chain process disappears without
writing END (crash / reboot) nothing is published and this exits with code 2 so
the partial state is never frozen as a master by mistake.

Usage (normally launched detached right after the chain):
    python scripts/_wait_retrack_then_publish_260517.py [--interval-sec 300] [--tag vYYYYMMDD_newmodel]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PY = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
CHAIN_LOG = SCRIPTS / "_retrack_260517_newmodel.log"
CHAIN_PID = SCRIPTS / "_retrack_260517_newmodel.pid"
END_LINE = "=== 260517 new-model re-track END ==="
LOG = SCRIPTS / "_wait_retrack_then_publish_260517.log"


def _log(msg: str) -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return True  # be conservative: assume alive if we cannot tell
    return str(pid) in out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--interval-sec", type=float, default=300.0)
    ap.add_argument("--max-days", type=float, default=7.0)
    ap.add_argument("--tag", default=None, help="passed to publish_master_260517.py")
    args = ap.parse_args()

    pid = int(CHAIN_PID.read_text().strip()) if CHAIN_PID.exists() else None
    _log(f"waiting for chain END (pid={pid}, log={CHAIN_LOG.name}, every {args.interval_sec:.0f}s)")
    t0 = time.time()
    while True:
        text = CHAIN_LOG.read_text(encoding="utf-8", errors="replace") if CHAIN_LOG.exists() else ""
        if END_LINE in text:
            _log("chain END detected")
            break
        if pid is not None and not pid_alive(pid):
            _log(f"chain pid {pid} is gone but no END line - NOT publishing (exit 2)")
            return 2
        if time.time() - t0 > args.max_days * 86400:
            _log("gave up waiting (max-days reached)")
            return 3
        time.sleep(args.interval_sec)

    # Rebuild the consolidated tables with the current code (the chain process
    # loaded its copy of the module when it started), then freeze the master.
    cons = [PY, "-u", str(SCRIPTS / "_retrack_260517_newmodel.py"), "--consolidate-only"]
    _log("launching: " + " ".join(cons))
    rc = subprocess.run(cons).returncode
    _log(f"consolidate rc={rc}")
    if rc != 0:
        _log("consolidation failed - NOT publishing")
        return rc

    cmd = [PY, "-u", str(SCRIPTS / "publish_master_260517.py")]
    if args.tag:
        cmd += ["--tag", args.tag]
    _log("launching: " + " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    _log(f"publish rc={rc}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
