"""
batch_pos_shifts_posparallel_260517.py
--------------------------------------
Pos-level-parallel driver for compute_pos_shifts (260517, offline, z=8).

Why this design (vs batch_compute_pos_shifts_260517.py):
  The single-Pos ProcessPool-over-frames path pickles the ~2 GB per-channel
  `stacks` to every worker on spawn. On Windows a single multiprocessing pickle
  write near/over 2 GB fails with OSError [Errno 22] Invalid argument, so
  large-drift Pos (more grid cells -> bigger payload) crash (Pos10/14/15/49...).

  Fix = parallelise across Pos instead of across frames:
    - Each Pos runs in its OWN process (independent GIL -> true multi-core),
      loading its own data (NO cross-process pickle of stacks -> no [Errno 22]).
    - Within a Pos we force a ThreadPool (monkeypatch ProcessPoolExecutor ->
      ThreadPoolExecutor): threads share memory (no pickle, low RAM) and
      cv2.findTransformECC releases the GIL so the ECC still parallelises.
    - Run K Pos concurrently to saturate the cores.

  Identical ECC math/results as the ProcessPool path (frames are computed
  independently either way); only the parallel topology changes.

Resumable: skips a Pos whose pos_shifts_cal.json already covers all frames.

Usage:
  Orchestrator:  python batch_pos_shifts_posparallel_260517.py
  (internal)     python batch_pos_shifts_posparallel_260517.py --pos N
"""
import sys
import os
import json
import time
import argparse
import subprocess
from pathlib import Path

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))

# ============================================================
TL_ROOT   = Path(r"E:\260517\2per_0055per_0per_2per")
POS_START = 1
POS_END   = 104

K_CONCURRENT    = 12   # Pos processes at once (~K * 2.5 GB RAM; 64 GB box)
THREADS_PER_POS = 3    # ThreadPool threads inside each Pos (memory-shared; fills cores)
LOG_DIR = _SD.parent / "drift_session" / "posparallel_logs"
# ============================================================


def _expected_frames(n):
    op = TL_ROOT / f"Pos{n}" / "z000" / "output_phase"
    return len(list(op.glob("img_*_ph_000_phase.tif")))


def _needs_run(n):
    op = TL_ROOT / f"Pos{n}" / "z000" / "output_phase"
    if not op.is_dir():
        return False  # nothing to do
    exp = _expected_frames(n)
    if exp == 0:
        return False
    p = op / "channels" / "pos_shifts_cal.json"
    if not p.exists():
        return True
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return d.get("n_frames", 0) < exp
    except Exception:
        return True


def worker(n):
    """Single-Pos worker: ThreadPool inside, run compute_pos_shifts via the
    existing run_pos() config. Fresh process => memory released on exit."""
    import concurrent.futures as cf
    cf.ProcessPoolExecutor = cf.ThreadPoolExecutor   # avoid the 2 GB pickle
    import batch_compute_pos_shifts_260517 as B
    B.N_WORKERS_ECC = THREADS_PER_POS
    return B.run_pos(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=int, default=None)
    args = ap.parse_args()

    if args.pos is not None:
        try:
            r = worker(args.pos)
        except Exception as e:
            import traceback
            traceback.print_exc()
            r = f"EXC:{e}"
        print(f"[single] Pos{args.pos} -> {r}", flush=True)
        sys.exit(0 if str(r).startswith(("ok", "skip")) else 1)

    # --- Orchestrator: sliding window of K Pos subprocesses, multi-pass retry ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def run_batch(todo):
        """Run the given Pos list through a K-wide subprocess window once."""
        running = {}   # Popen -> (pos, start_time, logfile)
        done_ok, done_fail = [], []
        idx = 0

        def launch(n):
            lf = open(LOG_DIR / f"Pos{n}.log", "w")
            p = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "--pos", str(n)],
                stdout=lf, stderr=subprocess.STDOUT,
            )
            running[p] = (n, time.time(), lf)

        while idx < len(todo) or running:
            while idx < len(todo) and len(running) < K_CONCURRENT:
                launch(todo[idx]); idx += 1
            time.sleep(3)
            for p in list(running):
                rc = p.poll()
                if rc is None:
                    continue
                n, st, lf = running.pop(p)
                lf.close()
                (done_ok if rc == 0 else done_fail).append(n)
                print(f"  Pos{n} rc={rc} ({(time.time()-st)/60:.1f} min)  "
                      f"[done {len(done_ok)+len(done_fail)}/{len(todo)}, "
                      f"running {len(running)}]", flush=True)
        return done_ok, done_fail

    MAX_PASSES = 3
    for pass_i in range(1, MAX_PASSES + 1):
        todo = [n for n in range(POS_START, POS_END + 1) if _needs_run(n)]
        if not todo:
            print(f"All Pos complete.", flush=True)
            break
        print(f"\n=== PASS {pass_i}/{MAX_PASSES}: {len(todo)} Pos to run "
              f"(K={K_CONCURRENT}, threads/Pos={THREADS_PER_POS}) ===", flush=True)
        print(f"todo: {todo}", flush=True)
        ok, fail = run_batch(todo)
        print(f"[pass {pass_i}] ok={len(ok)} fail={len(fail)}", flush=True)

    remaining = [n for n in range(POS_START, POS_END + 1) if _needs_run(n)]
    print(f"\nDONE in {(time.time()-t0)/60:.1f} min.", flush=True)
    if remaining:
        print(f"STILL INCOMPLETE after {MAX_PASSES} passes: {remaining}", flush=True)
    else:
        print("All Pos have complete pos_shifts_cal.json.", flush=True)


if __name__ == "__main__":
    main()
