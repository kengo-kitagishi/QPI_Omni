"""Free F: drive by deleting Pos intermediates that are no longer needed:

  - output_phase_raw/  (only used by correct_0pergluc; once it has written
                       crop_sub_rawraw/correct_0pergluc_log.json we don't
                       need it any more)
  - output_phase/*.tif (top-level BG-subtracted full-frame phase; channels/
                       subdir is preserved because crop_sub_rawraw lives
                       there)

Both are gated on a Pos having a populated channels/crop_sub_rawraw/
directory, so we don't accidentally wipe upstream output of an in-flight Pos.

Run as a one-off when F: runs out of space."""
from __future__ import annotations
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(r"F:\260405_acute_z18_200h\ph_260405")
KEEP_LATEST = 0  # for safety; set > 0 to keep the N most recent Pos untouched


def _rm(p: Path) -> int:
    if not p.exists():
        return 0
    size = 0
    try:
        for f in p.rglob("*"):
            try:
                size += f.stat().st_size
            except Exception:
                pass
        shutil.rmtree(p, ignore_errors=True)
    except Exception:
        pass
    return size


def cleanup_pos(pos: Path) -> tuple[str, int, int]:
    """Returns (pos_name, raw_freed_bytes, op_tifs_freed_bytes)."""
    cs = pos / "output_phase" / "channels" / "crop_sub_rawraw"
    if not cs.is_dir() or not any(cs.iterdir()):
        return (pos.name, 0, 0)

    # output_phase_raw — full directory
    raw_dir = pos / "output_phase_raw"
    raw_freed = _rm(raw_dir)

    # output_phase/*.tif at top level (not subdirs)
    op_dir = pos / "output_phase"
    op_freed = 0
    if op_dir.is_dir():
        for f in op_dir.glob("*.tif"):
            try:
                op_freed += f.stat().st_size
                f.unlink()
            except Exception:
                pass

    return (pos.name, raw_freed, op_freed)


def main() -> None:
    poses = sorted(
        (p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("Pos") and p.name != "Pos0"),
        key=lambda p: int(p.name.removeprefix("Pos")),
    )
    if KEEP_LATEST > 0:
        poses = poses[:-KEEP_LATEST]
    print(f"[info] {len(poses)} Pos candidates for cleanup")

    total_raw = 0
    total_op = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(cleanup_pos, p): p for p in poses}
        done = 0
        for fut in as_completed(futs):
            name, raw_freed, op_freed = fut.result()
            total_raw += raw_freed
            total_op += op_freed
            done += 1
            if (raw_freed + op_freed) > 0:
                print(f"  [{done:>3}/{len(poses)}] {name}: "
                      f"raw={raw_freed/1e9:.2f} GB  op_tifs={op_freed/1e9:.2f} GB", flush=True)
    print(f"[done] freed: output_phase_raw {total_raw/1e9:.1f} GB, "
          f"output_phase/*.tif {total_op/1e9:.1f} GB  "
          f"({(total_raw + total_op)/1e9:.1f} GB total) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
