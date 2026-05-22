"""Re-run step0 (recon of timelapse frames) for the Pos that lost
output_phase/*.tif and output_phase_raw/* during the earlier cleanup,
without touching steps 1-4 (which already wrote crop_sub_rawraw).

Reads Pos0 BG cache (still on disk) and the raw holos
(Pos<N>/img_*_ph_000.tif). Writes back:
  Pos<N>/output_phase/*.tif       (BG-subtracted)
  Pos<N>/output_phase_raw/*.tif   (raw phase)

Runs in parallel with the analysis chain — both use the GPU but
cellpose is light and the recon FFTs leave room on the A5000. Target
Pos can be filtered via --start / --end / --only."""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import tifffile
from tqdm import tqdm

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from batch_reconstruction_grid import reconstruct_from_holo

TIMELAPSE_ROOT = Path(r"F:\260405_acute_z18_200h\ph_260405")
BG_CACHE_BEFORE = TIMELAPSE_ROOT / "Pos0" / "bg_phase_before"
BG_CACHE_AFTER = TIMELAPSE_ROOT / "Pos0" / "bg_phase_after"

POS_SPLIT = 33
CROP_BEFORE = (0, 2048, 400, 2448)   # Pos < POS_SPLIT
CROP_AFTER = (0, 2048, 0, 2048)      # Pos >= POS_SPLIT


def get_crop(pos_num: int):
    return CROP_BEFORE if pos_num < POS_SPLIT else CROP_AFTER


_BG_SHM = None
_BG_CACHE: dict = {}


def _init_worker(shm_name, shape, dtype_str, stems):
    global _BG_SHM, _BG_CACHE
    _BG_SHM = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_BG_SHM.buf)
    _BG_CACHE = {stem: arr[i] for i, stem in enumerate(stems)}


def _build_shm(cache_dir: Path):
    files = sorted(cache_dir.glob("*_phase.tif"))
    if not files:
        raise FileNotFoundError(cache_dir)
    first = tifffile.imread(str(files[0]))
    dtype = first.dtype
    shape = (len(files), first.shape[0], first.shape[1])
    nbytes = int(np.prod(shape)) * dtype.itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    arr[0] = first
    stems = [files[0].stem]
    for i, f in enumerate(files[1:], start=1):
        arr[i] = tifffile.imread(str(f))
        stems.append(f.stem)
    return shm, (shm.name, shape, str(dtype), stems)


def _release_shm(shm):
    if shm is not None:
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass


def _one_frame(args):
    tgt_path_str, bg_stem, crop, out_phase_path, out_raw_path, pos_num = args
    out_phase_path = Path(out_phase_path)
    out_raw_path = Path(out_raw_path)
    if out_phase_path.exists() and out_raw_path.exists():
        return True
    try:
        tgt_phase = reconstruct_from_holo(tgt_path_str, crop)
        # raw
        if not out_raw_path.exists():
            out_raw_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(out_raw_path), tgt_phase.astype(np.float32))
        # BG-subtracted
        if not out_phase_path.exists():
            bg_phase = _BG_CACHE[bg_stem].astype(np.float64)
            phase = tgt_phase - bg_phase
            h, w = phase.shape
            if pos_num < POS_SPLIT:
                region = phase[1:h - 1, 1:w // 2]
            else:
                region = phase[1:h - 1, w // 2:w - 1]
            if region.size > 0:
                phase -= float(np.mean(region))
            out_phase_path.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(str(out_phase_path), phase.astype(np.float32))
        return True
    except Exception as e:
        print(f"[ERROR] {Path(tgt_path_str).name}: {e}", flush=True)
        return False


def process_pos(pos_num: int, n_workers: int) -> None:
    pos_dir = TIMELAPSE_ROOT / f"Pos{pos_num}"
    if not pos_dir.is_dir():
        print(f"[skip] Pos{pos_num}: dir not found", flush=True)
        return
    crop = get_crop(pos_num)
    cache_dir = BG_CACHE_BEFORE if pos_num < POS_SPLIT else BG_CACHE_AFTER

    raw_holos = sorted(pos_dir.glob("img_*_ph_000.tif"))
    if not raw_holos:
        print(f"[skip] Pos{pos_num}: no raw holos", flush=True)
        return

    out_phase_dir = pos_dir / "output_phase"
    out_raw_dir = pos_dir / "output_phase_raw"

    tasks = []
    for raw_path in raw_holos:
        stem = raw_path.stem  # img_000000123_ph_000
        bg_stem = f"{stem}_phase"  # bg cache files end with _phase.tif
        out_phase = out_phase_dir / f"{stem}_phase.tif"
        out_raw = out_raw_dir / f"{stem}_phase.tif"
        tasks.append((str(raw_path), bg_stem, crop, str(out_phase), str(out_raw), pos_num))

    # filter already-done
    to_do = [t for t in tasks
             if not (Path(t[3]).exists() and Path(t[4]).exists())]
    print(f"[Pos{pos_num}] crop={crop}  to_recon={len(to_do)}/{len(tasks)}", flush=True)
    if not to_do:
        return

    shm, initargs = _build_shm(cache_dir)
    try:
        ok = err = 0
        with ProcessPoolExecutor(max_workers=n_workers,
                                 initializer=_init_worker,
                                 initargs=initargs) as ex:
            futures = [ex.submit(_one_frame, t) for t in to_do]
            for f in tqdm(as_completed(futures), total=len(futures),
                          desc=f"Pos{pos_num}"):
                if f.result():
                    ok += 1
                else:
                    err += 1
        print(f"[Pos{pos_num}] done: ok={ok} err={err}", flush=True)
    finally:
        _release_shm(shm)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=26)
    ap.add_argument("--only", type=int, nargs="*", default=None,
                    help="explicit Pos numbers (overrides --start/--end)")
    ap.add_argument("--workers", type=int, default=4,
                    help="frames in parallel within each Pos (default 4; "
                         "kept low so the analysis chain still has GPU bandwidth)")
    args = ap.parse_args()

    targets = args.only if args.only else list(range(args.start, args.end + 1))
    print(f"[info] recovering output_phase{{,_raw}} for Pos: {targets}", flush=True)
    for pos in targets:
        process_pos(pos, args.workers)
    print("[done] all targets processed", flush=True)


if __name__ == "__main__":
    main()
