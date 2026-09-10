"""
run_recon_batches_260908.py

Reconstruct, detect and calibrate the raw folders left over from the 260908
grid session, one after another.

Each raw folder carries its own Pos0 BG and must therefore reconstruct into
its own output tree -- the BG is subtracted from the samples acquired
alongside it, so BGs are never shared across acquisitions.

The D: folder (ye_grid_0p05_1) runs last: its disk is the one the timelapse
writes to. Finished batches are re-verified and skipped, so the driver can
simply be run again to pick up a batch that was added later.

Run:
    python run_recon_batches_260908.py
    python run_recon_batches_260908.py --dry-run
"""
import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
WATCHER = SCRIPT_DIR / "watch_grid_then_recon_260908.py"

BATCHES = [
    # (raw, output) -- newest acquisition first: it is the one still missing
    # from every downstream step.
    (r"C:\260908\ye_grid_0p05_3", r"E:\260908\ye_grid_0p05_3"),
    (r"C:\260908\ye_grid_0p05_2", r"E:\260908\ye_grid_0p05_2"),
    # Pos0 + Pos50-63. Its partial Pos64 is parked in _partial_Pos64. This is
    # the disk the timelapse writes to, so it goes last.
    (r"D:\AquisitionData\Kitagishi\260908\ye_grid_0p05_1", r"E:\260908\ye_grid_0p05_1"),
]

LOG_PATH = Path(r"C:\260908\run_recon_batches.log")


def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def patch(name, value):
    text = WATCHER.read_text(encoding="utf-8")
    pat = re.compile(rf"^({re.escape(name)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
    m = pat.search(text)
    if not m:
        raise RuntimeError(f"{WATCHER.name}: '{name} = ...' not found")
    if m.group(2).strip() != value:
        log(f"  {name}: {m.group(2).strip()}  ->  {value}")
    WATCHER.write_text(pat.sub(lambda mm: f"{mm.group(1)}{value}{mm.group(3)}",
                               text, count=1), encoding="utf-8")


def ensure_bg_placeholders(raw, out):
    """Let a batch whose Pos0 raw is gone reconstruct against its saved BG.

    batch_reconstruction_grid only needs the Pos0_x*_y* directory names to
    exist under the raw tree: with no image files inside, the BG pass reports
    "all exist, skipped" and every target frame reads its BG phase from
    <out>/Pos0_.../output_phase_raw. So when the raw Pos0 was deleted after
    its reconstruction, empty directories stand in for it. Nothing is written
    into them and the watcher ignores image-less point folders.
    """
    raw, out = Path(raw), Path(out)
    if any(raw.glob("Pos0_x*_y*/img_*.tif")):
        return 0
    made = 0
    for xi in range(-5, 6):
        for yi in range(-5, 6):
            name = f"Pos0_x{xi:+d}_y{yi:+d}"
            bg = out / name / "output_phase_raw"
            if not any(bg.glob("img_*_ph_*_phase.tif")):
                raise RuntimeError(f"{name}: no raw and no reconstructed BG in {out}")
            d = raw / name
            if not d.exists():
                d.mkdir()
                made += 1
    if made:
        log(f"  Pos0 raw absent; created {made} empty placeholder dirs so the "
            f"reconstruction reads the BG from {out}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    results = []
    for raw, out in BATCHES:
        log("=" * 70)
        log(f"BATCH {raw}  ->  {out}")
        log("=" * 70)
        patch("WATCH_DIR", f'Path(r"{raw}")')
        patch("OUTPUT_DIR", f'Path(r"{out}")')
        ensure_bg_placeholders(raw, out)

        cmd = [sys.executable, str(WATCHER), "--skip-wait"]
        if args.dry_run:
            cmd.append("--dry-run")
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(SCRIPT_DIR)).returncode
        log(f"{raw}: exit {rc} after {(time.time()-t0)/3600:.2f} h")
        results.append((raw, rc))
        if rc != 0:
            log(f"{raw} FAILED -- stopping so the raw is not left half handled")
            break

    log("=" * 70)
    for raw, rc in results:
        log(f"{raw}: {'OK' if rc == 0 else f'FAILED ({rc})'}")
    log("=" * 70)
    sys.exit(0 if all(rc == 0 for _, rc in results) else 1)


if __name__ == "__main__":
    main()
