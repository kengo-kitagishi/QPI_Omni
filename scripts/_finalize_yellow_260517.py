"""_finalize_yellow_260517.py - wait for the two running jobs, sweep, consolidate, publish.

Jobs it waits for (polling the logs every 5 min):
  * _chain_tiltfix_260517.py      -> "=== tiltfix chain END ==="   (Pos53..104: refit, seg, track)
  * _retrack_260517_newmodel.py   -> "=== 260517 new-model re-track END ===" (Pos1..52 re-track)
If a job's process disappears without its END line, this exits with code 2 and publishes nothing.

Then, in order:
  1. sweep  : _retrack_260517_newmodel.py --start 53 --end 104   (re-tracks the channels that were
              tracked with the pre-yellow tracker before the switch; the production marker is
              volume_um3_efd, so only those are redone)
  2. consolidate (includes division_qc_260517 per channel)
  3. publish_master_260517.py --tag <tag>   (phase-1 package built inside)
  4. SUPERSEDED note next to v20260911_newmodel

Usage:
    python scripts/_finalize_yellow_260517.py [--tag v20260916_yellow] [--interval-sec 300]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
PY = r"C:\Users\QPI\anaconda3\envs\omnipose\python.exe"
LOG = SCRIPTS / "_finalize_yellow_260517.log"
JOBS = {
    "tiltfix": (SCRIPTS / "_chain_tiltfix_260517.log", SCRIPTS / "_chain_tiltfix_260517.pid", "=== tiltfix chain END ==="),
    "retrack1-52": (SCRIPTS / "_retrack_260517_newmodel.log", SCRIPTS / "_retrack_260517_newmodel.pid",
                    "=== 260517 new-model re-track END ==="),
}


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
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return True
    return str(pid) in out


def _run(cmd: list[str]) -> int:
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(cmd, env=e).returncode


def wait_jobs(interval: float, max_days: float) -> bool:
    t0 = time.time()
    done = set()
    while len(done) < len(JOBS):
        for name, (log, pidf, end_line) in JOBS.items():
            if name in done:
                continue
            text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
            if end_line in text:
                _log(f"{name}: END detected")
                done.add(name)
                continue
            pid = int(pidf.read_text().strip()) if pidf.exists() else None
            if pid is not None and not pid_alive(pid):
                _log(f"{name}: pid {pid} gone without END line - aborting (exit 2)")
                return False
        if len(done) < len(JOBS):
            if time.time() - t0 > max_days * 86400:
                _log("gave up waiting")
                return False
            time.sleep(interval)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--interval-sec", type=float, default=300.0)
    ap.add_argument("--max-days", type=float, default=7.0)
    ap.add_argument("--skip-wait", action="store_true")
    args = ap.parse_args()
    _log("=== finalize START ===")
    if not args.skip_wait and not wait_jobs(args.interval_sec, args.max_days):
        return 2

    # 1. sweep Pos53..104 (log rotated so the earlier runs stay readable)
    rl = SCRIPTS / "_retrack_260517_newmodel.log"
    if rl.exists():
        rl.rename(SCRIPTS / f"_retrack_260517_newmodel.log.before_sweep_{time.strftime('%Y%m%dT%H%M%S')}")
    rc = _run([PY, "-u", str(SCRIPTS / "_retrack_260517_newmodel.py"), "--start", "53", "--end", "104", "--no-consolidate"])
    _log(f"sweep 53-104 rc={rc}")
    if rc != 0:
        return rc
    # 2. consolidate (+ division QC)
    rc = _run([PY, "-u", str(SCRIPTS / "_retrack_260517_newmodel.py"), "--consolidate-only"])
    _log(f"consolidate rc={rc}")
    if rc != 0:
        return rc
    # 3. publish
    tag = args.tag or time.strftime("v%Y%m%d_yellow")
    rc = _run([PY, "-u", str(SCRIPTS / "publish_master_260517.py"), "--tag", tag])
    _log(f"publish {tag} rc={rc}")
    if rc != 0:
        return rc
    # 4. supersede note
    note = Path(r"D:\QPI_master\260517") / "v20260911_newmodel.SUPERSEDED.txt"
    try:
        note.write_text(
            f"Superseded by {tag} ({time.strftime('%Y-%m-%d')}).\n"
            "- Pos53..104 phase crops had the linear tilt fitted on the cell side (grid_subtract fit_right=False "
            "for all Pos); re-flattened on the right third (refit_tilt_right_260517.py), masks and lineages of "
            "Pos53..104 regenerated.\n"
            "- Cell geometry changed for ALL positions: yellow-contour (EFD K=6, -0.5 px) rod and section volumes "
            "replace the medial-axis rod/profile volumes; skimage/profile columns removed.\n"
            "- Division calls carry a mass/volume validation (divisions_qc).\n", encoding="utf-8")
    except Exception as e:
        _log(f"WARN: could not write SUPERSEDED note: {e!r}")
    _log("=== finalize END ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
