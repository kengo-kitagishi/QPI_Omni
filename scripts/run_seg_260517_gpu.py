# -*- coding: utf-8 -*-
"""run_seg_260517_gpu.py - GPU Omnipose segmentation driver for 260517.

Repo copy of the other session's ``run_seg_260517_fast2.py`` that produced the
2026-09-07..09 masks (archived in the master under inputs/). Same checkpoint,
EVAL parameters and empty-frame gate, so masks stay comparable; only the roots
became configurable:

  SEG_RAW_ROOT   phase crops root   (default H:\\260517\\2per_0055per_0per_2per_crop_sub)
  SEG_OUT_ROOT   mask root          (default D:\\260517_seg)
  SEG_POS_START / SEG_POS_END / SEG_WORKERS / SEG_MAX_FILES  as before

Behaviour:
- reads channel crops, gates on phase intensity (skips the model on empty frames)
- writes ONLY <frame>_masks.tif for frames with >=1 detected cell
- ProcessPoolExecutor over channels; model loaded once per worker (GPU only)
- per-channel _DONE marker for crash-resume (SEG_MAX_FILES disables resume)
"""
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

H_ROOT = os.environ.get("SEG_RAW_ROOT", r"H:\260517\2per_0055per_0per_2per_crop_sub")
OUT_ROOT = os.environ.get("SEG_OUT_ROOT", r"D:\260517_seg")
MODEL = (r"C:\Users\QPI\Desktop\train\omni_model_d20\models"
         r"\cellpose_residual_on_style_on_concatenation_off_omni_abstract"
         r"_nclasses_3_nchan_1_dim_2_omni_model_d20_2026_09_07_12_41_31.782047")
REL = ("output_phase", "channels", "crop_sub_rawraw", "z000")
GATE_HI, GATE_MIN_PX = 0.7, 40
DIAM = 20

POS_START = int(os.environ.get("SEG_POS_START", "1"))
POS_END = int(os.environ.get("SEG_POS_END", "104"))
N_WORKERS = int(os.environ.get("SEG_WORKERS", "6"))
_mf = os.environ.get("SEG_MAX_FILES")
MAX_FILES = int(_mf) if _mf else None

EVAL = dict(channels=None, channel_axis=None, diameter=DIAM, normalize=True, tile=False,
            omni=True, verbose=False, flow_threshold=0.11, mask_threshold=0, min_size=10,
            net_avg=False)

_model = None


def _init():
    global _model
    import logging
    logging.getLogger("cellpose_omni").setLevel(logging.WARNING)
    from cellpose_omni.models import CellposeModel
    _model = CellposeModel(gpu=True, pretrained_model=MODEL, omni=True, nchan=1, nclasses=3, dim=2)


def process_channel(args):
    pos, chdir_str = args
    chdir = Path(chdir_str)
    outdir = Path(OUT_ROOT) / chdir.relative_to(H_ROOT) / "inference_out"
    done_marker = outdir / "_DONE"
    if done_marker.exists() and not MAX_FILES:
        return (pos, chdir.name, "skip-done", 0, 0)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(chdir.glob("img_*_ph_000_phase.tif"),
                   key=lambda p: int(re.search(r"img_(\d+)_", p.name).group(1)))
    if MAX_FILES:
        files = files[:MAX_FILES]
    nc = ng = 0
    for f in files:
        try:
            img = tifffile.imread(str(f)).astype(np.float32)
        except Exception:
            continue
        if int((img > GATE_HI).sum()) < GATE_MIN_PX:   # cheap empty gate: no model, no file
            ng += 1
            continue
        try:
            m = _model.eval([img], **EVAL)[0][0]
        except Exception:
            ng += 1
            continue
        if m is None or int(np.asarray(m).max()) == 0:  # model found nothing: no file
            ng += 1
            continue
        tifffile.imwrite(str(outdir / f"{f.stem}_masks.tif"), np.asarray(m).astype(np.uint16))
        nc += 1
    if not MAX_FILES:
        done_marker.write_text("done")
    return (pos, chdir.name, "ok", nc, ng)


def main():
    tasks = []
    for pos in range(POS_START, POS_END + 1):
        base = Path(H_ROOT) / f"Pos{pos}" / Path(*REL)
        if not base.is_dir():
            continue
        for ch in sorted([d for d in base.glob("ch*") if d.is_dir()], key=lambda p: int(p.name[2:])):
            tasks.append((pos, str(ch)))
    print(f"channels={len(tasks)} workers={N_WORKERS} pos={POS_START}-{POS_END} "
          f"max_files={MAX_FILES} raw={H_ROOT} out={OUT_ROOT}", flush=True)
    t0 = time.time()
    done = tot_c = tot_g = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=_init) as ex:
        futs = [ex.submit(process_channel, t) for t in tasks]
        for fut in as_completed(futs):
            pos, ch, st, c, g = fut.result()
            done += 1
            tot_c += c
            tot_g += g
            el = time.time() - t0
            fps = (tot_c + tot_g) / max(el, 1e-9)
            print(f"[{done}/{len(tasks)}] Pos{pos} {ch}: {st} cell={c} gated={g} "
                  f"| {el:.0f}s {fps:.1f} frames/s", flush=True)
    el = time.time() - t0
    print(f"DONE {done} ch in {el:.0f}s | cell_frames={tot_c} gated={tot_g} "
          f"| {(tot_c + tot_g) / max(el, 1e-9):.1f} frames/s", flush=True)


if __name__ == "__main__":
    main()
