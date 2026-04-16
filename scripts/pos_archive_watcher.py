"""
pos_archive_watcher.py
----------------------
Watch batch_pipeline_progress.json and archive each Pos to F: after it
finishes all steps (0per=true). Then delete the original from D:.

Flow per Pos:
  1. Detect 0per=true in progress log
  2. Copy D:\...\PosN\ -> F:\260405\ph_260405\PosN\
  3. Verify: file count + cumulative size match
  4. Delete D:\...\PosN\

Notes:
  - Pos in SKIP_POS are handled manually in another session -> skip here.
  - Re-running is safe: completed transfers are recorded in STATE_FILE.
  - Copy failures never trigger the delete step.
"""
import json
import shutil
import sys
import time
from pathlib import Path
from datetime import datetime

SRC_ROOT = Path(r"D:\AquisitionData\Kitagishi\260405\ph_260405")
DST_ROOT = Path(r"F:\260405\ph_260405")
PROGRESS_LOG = SRC_ROOT / "batch_pipeline_progress.json"
STATE_FILE = SRC_ROOT / "pos_archive_watcher.json"

POS_MIN = 10
POS_MAX = 64
SKIP_POS = {6, 7, 8}      # handled manually in another session
POLL_INTERVAL = 30        # seconds


def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {"archived": []}


def save_state(state):
    STATE_FILE.write_text(
        json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def dir_stats(d: Path):
    """Return (file_count, total_size)."""
    n = 0
    sz = 0
    for p in d.rglob("*"):
        if p.is_file():
            n += 1
            sz += p.stat().st_size
    return n, sz


def archive_pos(pos_num: int) -> bool:
    src = SRC_ROOT / f"Pos{pos_num}"
    dst = DST_ROOT / f"Pos{pos_num}"

    if not src.is_dir():
        log(f"Pos{pos_num}: src missing, skip")
        return False

    log(f"Pos{pos_num}: start copy {src} -> {dst}")
    t0 = time.time()

    # If dst already exists with content, refuse to overwrite (safety)
    if dst.exists() and any(dst.iterdir()):
        log(f"Pos{pos_num}: dst already has content, ABORT (manual check needed)")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        shutil.copytree(str(src), str(dst), dirs_exist_ok=True)
    except Exception as e:
        log(f"Pos{pos_num}: copy FAILED: {e}")
        return False

    # Verify
    src_n, src_sz = dir_stats(src)
    dst_n, dst_sz = dir_stats(dst)
    if src_n != dst_n or src_sz != dst_sz:
        log(f"Pos{pos_num}: VERIFY FAILED "
            f"src=(n={src_n}, sz={src_sz}) dst=(n={dst_n}, sz={dst_sz}). "
            f"Not deleting src.")
        return False

    dt = time.time() - t0
    log(f"Pos{pos_num}: copy OK ({src_n} files, {src_sz/1e9:.1f} GB, {dt/60:.1f} min)")

    # Delete src
    try:
        shutil.rmtree(str(src))
        log(f"Pos{pos_num}: deleted src")
    except Exception as e:
        log(f"Pos{pos_num}: delete FAILED (copy OK): {e}")
        return False

    return True


def main():
    log(f"Watcher start. src={SRC_ROOT} dst={DST_ROOT}")
    log(f"  Pos range: {POS_MIN}-{POS_MAX}, skip={sorted(SKIP_POS)}")

    if not DST_ROOT.parent.exists():
        log(f"ERROR: dst parent {DST_ROOT.parent} does not exist")
        sys.exit(1)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    while True:
        state = load_state()
        archived = set(state["archived"])

        if not PROGRESS_LOG.exists():
            log("progress log missing, waiting...")
            time.sleep(POLL_INTERVAL)
            continue

        try:
            prog = json.loads(PROGRESS_LOG.read_text(encoding="utf-8"))
        except Exception as e:
            log(f"progress log read error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        candidates = []
        for label, steps in prog.items():
            if not label.startswith("Pos"):
                continue
            try:
                n = int(label[3:])
            except ValueError:
                continue
            if n in SKIP_POS or n < POS_MIN or n > POS_MAX:
                continue
            if n in archived:
                continue
            if steps.get("0per") is True:
                candidates.append(n)

        for n in sorted(candidates):
            ok = archive_pos(n)
            if ok:
                state["archived"].append(n)
                save_state(state)
            else:
                log(f"Pos{n}: will retry next cycle")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Watcher stopped.")
