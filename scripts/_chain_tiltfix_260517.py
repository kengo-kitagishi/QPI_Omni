"""_chain_tiltfix_260517.py - redo Pos53..Pos104 of 260517 after the tilt-side fix,
end to end, and publish a new master.

Per position (three pipelined stages, each position flows refit -> seg -> track):
  1. refit  : refit_tilt_right_260517.py --pos N     (H: crops -> D:\\260517_tiltfix, right-third fit)
  2. seg    : run_seg_260517_gpu.py  (SEG_RAW_ROOT=D:\\260517_tiltfix, SEG_OUT_ROOT=D:\\260517_seg)
              the old masks + lineage of the position are moved to D:\\260517_seg_oldtilt\\PosN first
  3. track  : central_cell_lineage_tracker via _retrack_260517_newmodel (raw_root_for -> D:\\260517_tiltfix)
Then: consolidate all 104 Pos -> publish_master_260517.py --tag <tag> (phase-1 package is built inside).

Resumable: refit uses _REFIT_DONE markers, seg uses per-channel _DONE markers, tracking skips
production lineages (is_production). Re-launch the same command after a crash / reboot.

Usage:
    python scripts/_chain_tiltfix_260517.py                    # Pos 53..104, then publish
    python scripts/_chain_tiltfix_260517.py --start 53 --end 60 --no-publish
    python scripts/_chain_tiltfix_260517.py --publish-only --tag v20260915_tiltfix
"""
from __future__ import annotations

import argparse
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import _retrack_260517_newmodel as chain  # noqa: E402

PY = chain.PY
LOG = SCRIPTS / "_chain_tiltfix_260517.log"
OLD_MASK_ROOT = Path(r"D:\260517_seg_oldtilt")
SEG_WORKERS = int(os.environ.get("SEG_WORKERS", "6"))
_lock = threading.Lock()


def _log(msg: str) -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with _lock:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def _run(cmd: list[str], env: dict | None = None) -> int:
    e = dict(os.environ)
    e["PYTHONIOENCODING"] = "utf-8"
    if env:
        e.update(env)
    return subprocess.run(cmd, env=e).returncode


# ---------------------------------------------------------------- stage 1: refit
def refit_done(n: int) -> bool:
    return (chain.RAW_ROOT_TILTFIX / f"Pos{n}" / "_REFIT_DONE").exists()


def stage_refit(n: int) -> bool:
    if refit_done(n):
        _log(f"refit Pos{n}: already done")
        return True
    t0 = time.time()
    rc = _run([PY, "-u", str(SCRIPTS / "refit_tilt_right_260517.py"), "--pos", str(n)])
    _log(f"refit Pos{n}: rc={rc} ({time.time() - t0:.0f}s)")
    return rc == 0 and refit_done(n)


# ---------------------------------------------------------------- stage 2: seg
def _retire_old_masks(n: int) -> None:
    """Move the wrong-tilt masks + lineage of PosN out of the mask root (kept for comparison)."""
    old = chain.MASK_ROOT / f"Pos{n}"
    if not old.exists():
        return
    z = old / chain.REL
    # only retire if this looks like an old run (any channel lacks a TILTFIX marker)
    if (old / "_TILTFIX_SEG").exists():
        return
    dst = OLD_MASK_ROOT / f"Pos{n}"
    if dst.exists():
        dst = OLD_MASK_ROOT / f"Pos{n}_{time.strftime('%Y%m%dT%H%M%S')}"
    OLD_MASK_ROOT.mkdir(parents=True, exist_ok=True)
    shutil.move(str(old), str(dst))
    _log(f"seg Pos{n}: old masks/lineage moved -> {dst}")


def seg_done(n: int) -> bool:
    z = chain.MASK_ROOT / f"Pos{n}" / chain.REL
    src = chain.RAW_ROOT_TILTFIX / f"Pos{n}" / chain.REL
    if not (z.is_dir() and src.is_dir()):
        return False
    chs = [p.name for p in src.iterdir() if p.is_dir() and p.name.startswith("ch")]
    return bool(chs) and all((z / c / "inference_out" / "_DONE").exists() for c in chs)


def stage_seg(n: int) -> bool:
    _retire_old_masks(n)
    if seg_done(n):
        _log(f"seg Pos{n}: already done")
        return True
    t0 = time.time()
    env = {"SEG_RAW_ROOT": str(chain.RAW_ROOT_TILTFIX), "SEG_OUT_ROOT": str(chain.MASK_ROOT),
           "SEG_POS_START": str(n), "SEG_POS_END": str(n), "SEG_WORKERS": str(SEG_WORKERS)}
    rc = _run([PY, "-u", str(SCRIPTS / "run_seg_260517_gpu.py")], env=env)
    ok = rc == 0 and seg_done(n)
    if ok:
        (chain.MASK_ROOT / f"Pos{n}" / "_TILTFIX_SEG").write_text(
            f"masks from D:\\260517_tiltfix crops, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    _log(f"seg Pos{n}: rc={rc} ok={ok} ({time.time() - t0:.0f}s)")
    return ok


# ---------------------------------------------------------------- stage 3: track
def stage_track(n: int) -> bool:
    targets, n_done, n_empty = chain.channel_worklist(n, n, force=False)
    _log(f"track Pos{n}: {len(targets)} channels to track ({n_done} done, {n_empty} empty)")
    t0 = time.time()
    n_err = 0
    for pos, mask_ch, raw_ch in targets:
        t1 = time.time()
        rc = chain.run_tracker(mask_ch, raw_ch)
        if rc != 0:
            n_err += 1
            _log(f"!! track {pos}/{mask_ch.name} rc={rc} ({time.time() - t1:.0f}s)")
        else:
            _log(f"ok track {pos}/{mask_ch.name} ({time.time() - t1:.0f}s)")
    _log(f"track Pos{n}: done, errors={n_err} ({time.time() - t0:.0f}s)")
    return True


# ---------------------------------------------------------------- pipeline
def run_pipeline(poss: list[int]) -> None:
    q_seg: queue.Queue = queue.Queue()
    q_trk: queue.Queue = queue.Queue()
    failures: list[str] = []

    def refit_worker():
        for n in poss:
            if stage_refit(n):
                q_seg.put(n)
            else:
                failures.append(f"refit Pos{n}")
        q_seg.put(None)

    def seg_worker():
        while True:
            n = q_seg.get()
            if n is None:
                q_trk.put(None)
                return
            if stage_seg(n):
                q_trk.put(n)
            else:
                failures.append(f"seg Pos{n}")

    def track_worker():
        while True:
            n = q_trk.get()
            if n is None:
                return
            try:
                stage_track(n)
            except Exception as e:
                failures.append(f"track Pos{n}: {e!r}")
                _log(f"!! track Pos{n} raised {e!r}")

    threads = [threading.Thread(target=f, name=nm, daemon=True)
               for f, nm in ((refit_worker, "refit"), (seg_worker, "seg"), (track_worker, "track"))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        _log("FAILURES: " + "; ".join(failures))
    else:
        _log("all positions refit + segmented + tracked")


def publish(tag: str) -> None:
    _log("consolidating all positions")
    rc = _run([PY, "-u", str(SCRIPTS / "_retrack_260517_newmodel.py"), "--consolidate-only"])
    _log(f"consolidate rc={rc}")
    if rc != 0:
        _log("consolidation failed - NOT publishing")
        return
    rc = _run([PY, "-u", str(SCRIPTS / "publish_master_260517.py"), "--tag", tag])
    _log(f"publish {tag} rc={rc}")
    if rc == 0:
        note = Path(r"D:\QPI_master\260517") / "v20260911_newmodel.SUPERSEDED.txt"
        try:
            note.write_text(
                f"Superseded by {tag} ({time.strftime('%Y-%m-%d')}): Pos53..104 phase crops had the linear tilt "
                "fitted on the cell side (grid_subtract fit_right=False for all Pos), biasing total_phase, RI and "
                "mass of those positions; masks and lineages of Pos53..104 were regenerated from re-flattened "
                "crops (refit_tilt_right_260517.py). Pos1..52 are unchanged.\n", encoding="utf-8")
        except Exception as e:
            _log(f"WARN: could not write SUPERSEDED note: {e!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=chain.TILTFIX_POS_MIN)
    ap.add_argument("--end", type=int, default=104)
    ap.add_argument("--tag", default=None, help="master tag (default v<YYYYMMDD>_tiltfix at publish time)")
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--publish-only", action="store_true")
    args = ap.parse_args()
    _log(f"=== tiltfix chain START Pos {args.start}..{args.end} publish={not args.no_publish} ===")
    if not args.publish_only:
        run_pipeline(list(range(args.start, args.end + 1)))
    if not args.no_publish:
        publish(args.tag or time.strftime("v%Y%m%d_tiltfix"))
    _log("=== tiltfix chain END ===")


if __name__ == "__main__":
    main()
