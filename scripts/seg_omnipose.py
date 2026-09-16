# -*- coding: utf-8 -*-
"""seg_omnipose.py - GPU Omnipose segmentation of mother-machine trap crops, resumable per channel.

Generic successor of run_seg_260517_gpu.py (same empty-frame gate, same eval defaults, same
outputs), with the dataset layout and the checkpoint given on the command line so that
run_dataset_pipeline.py can drive it from datasets/<id>.yaml.

    python scripts/seg_omnipose.py --raw-root <phase root> --mask-root <mask root> \\
        --channel-rel output_phase/channels/crop_sub_rawraw/z000 \\
        --model models/omni_model_d20_2026_09_07_12_41_31.782047 \\
        [--pos-start 1 --pos-end 104] [--workers 6] [--max-files N] \\
        [--eval-json '{"diameter": 20, ...}'] [--gate-hi 0.7 --gate-min-px 40] [--phase-glob img_*_ph_000_phase.tif]

Layout
    <raw_root>/PosN/<channel_rel>/chNN/<phase_glob>                          phase crops (radian, float32)
    <mask_root>/PosN/<channel_rel>/chNN/inference_out/<stem>_masks.tif      uint16 labels, only frames with >= 1 cell
    <mask_root>/PosN/<channel_rel>/chNN/inference_out/_DONE                  per-channel marker (skip on re-run; --max-files never writes it)

Behaviour
    - frames with fewer than gate-min-px pixels above gate-hi rad skip the model (empty trap); no file is written
    - ProcessPoolExecutor over channels, one model per worker, GPU only (CPU masks differ; do not add a CPU path)
    - nothing is written under raw_root
"""
from __future__ import annotations

import base64
import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile

EVAL_DEFAULT = dict(channels=None, channel_axis=None, diameter=20, normalize=True, tile=False,
                    omni=True, verbose=False, flow_threshold=0.11, mask_threshold=0, min_size=10,
                    net_avg=False)
_model = None
_cfg: dict = {}


# cellpose_omni imports its GUI module when the package is imported, and that module downloads GUI
# assets into ~/.omnipose/ whenever they are missing. Two of those URLs answer 404 (seen 2026-09-15 on
# a fresh PC: gui/logo.png and gui/gamma.svg upstream), so `import cellpose_omni` itself raises. Both
# files only decorate the GUI window: 1x1 stand-ins let the import through. The test images it also
# fetches (docs/test_files/) still download. Same guard in environment/bootstrap_windows.ps1,
# environment/check_env.py and scripts/seg_omnipose.py.
_GUI_ASSETS_B64 = {
    "logo.png": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
    "gamma.svg": "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxIiBoZWlnaHQ9IjEiLz4=",
}


def _ensure_gui_icon() -> None:
    d = Path.home() / ".omnipose"
    for name, b64 in _GUI_ASSETS_B64.items():
        f = d / name
        if not f.is_file():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(base64.b64decode(b64))


def _ensure_cuda_dlls_on_path() -> None:
    """Put the env's bin/ on PATH so NVRTC can load nvrtc-builtins64_*.dll.

    From the second inference on, torch's TorchScript fuser compiles CUDA kernels at run time
    with NVRTC, and NVRTC finds its builtins DLL through PATH. The conda env keeps that DLL in
    <env>/bin, which is on PATH only after `conda activate`; the kit starts <env>/python.exe
    directly, so without this every inference after the first raised "failed to open
    nvrtc-builtins64_118.dll" (seen 2026-09-15 on the microscope PC). Same helper in
    environment/check_env.py and scripts/seg_omnipose.py.
    """
    prefix = Path(sys.prefix)
    parts = os.environ.get("PATH", "").split(os.pathsep)
    for d in (prefix / "Library" / "bin", prefix / "bin"):
        if d.is_dir() and str(d) not in parts:
            parts.insert(0, str(d))
    os.environ["PATH"] = os.pathsep.join(parts)


def _init(model_path: str, eval_params: dict, gate_hi: float, gate_min_px: int,
          raw_root: str, mask_root: str, phase_glob: str, max_files):
    global _model, _cfg
    _ensure_cuda_dlls_on_path()
    import logging
    logging.getLogger("cellpose_omni").setLevel(logging.WARNING)
    _ensure_gui_icon()
    from cellpose_omni.models import CellposeModel
    _model = CellposeModel(gpu=True, pretrained_model=model_path, omni=True, nchan=1, nclasses=3, dim=2)
    _cfg = dict(eval=eval_params, gate_hi=gate_hi, gate_min_px=gate_min_px, raw_root=Path(raw_root),
                mask_root=Path(mask_root), phase_glob=phase_glob, max_files=max_files)


def _frame_no(p: Path) -> int:
    m = re.search(r"img_0*(\d+)", p.name)
    return int(m.group(1)) if m else -1


def process_channel(args):
    pos, chdir_str = args
    chdir = Path(chdir_str)
    outdir = _cfg["mask_root"] / chdir.relative_to(_cfg["raw_root"]) / "inference_out"
    done_marker = outdir / "_DONE"
    max_files = _cfg["max_files"]
    if done_marker.exists() and not max_files:
        return (pos, chdir.name, "skip-done", 0, 0, 0, "", 0)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(chdir.glob(_cfg["phase_glob"]), key=_frame_no)
    if max_files:
        files = files[:max_files]
    # A frame that cannot be read or makes the model raise is an error, not an empty trap: it is
    # counted apart, reported, and keeps _DONE from being written so a re-run retries the channel.
    # (Counting those as "gated" hid a missing-DLL failure on every frame after the first.)
    nc = ng = ne = nt = 0
    first_err = ""
    for f in files:
        try:
            img = tifffile.imread(str(f)).astype(np.float32)
        except Exception as e:  # noqa: BLE001
            ne += 1
            first_err = first_err or f"{f.name}: read {e!r}"
            continue
        if int((img > _cfg["gate_hi"]).sum()) < _cfg["gate_min_px"]:
            ng += 1
            continue
        try:
            m = _model.eval([img], **_cfg["eval"])[0][0]
        except ValueError as e:
            if "n_neighbors <= n_samples_fit" not in str(e):
                ne += 1
                first_err = first_err or f"{f.name}: eval {str(e).splitlines()[0][:160]}"
                continue
            # Omnipose post-processing takes 50 nearest neighbours over the foreground pixels and
            # raises when a frame holds fewer (an object of < 50 px: a cell tip at the trap edge,
            # debris). Deterministic per frame, so it would fail every re-run; the pre-2026-09-15
            # code counted it as an empty frame and 260517 was published that way. 260908: 96 of
            # ~500k frames over 12 channels. Counted apart as "tiny", no mask, _DONE allowed.
            nt += 1
            continue
        except Exception as e:  # noqa: BLE001
            ne += 1
            first_err = first_err or f"{f.name}: eval {str(e).splitlines()[0][:160]}"
            continue
        if m is None or int(np.asarray(m).max()) == 0:
            ng += 1
            continue
        tifffile.imwrite(str(outdir / f"{f.stem}_masks.tif"), np.asarray(m).astype(np.uint16))
        nc += 1
    if not max_files and ne == 0:
        done_marker.write_text("done")
    return (pos, chdir.name, "ok" if ne == 0 else "ERROR", nc, ng + nt, ne, first_err, nt)


def load_channel_filter(path):
    """{(pos, "chNN")} from a selection file, or None when no path is given.

    Accepts what channel_contact_sheet.py writes: "PosN chNN" lines (.selected.txt) or
    "pos,ch" rows with a header (.selected.csv). Blank lines and # comments are ignored.
    """
    if not path:
        return None
    keep = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.lower().startswith("pos,ch"):
            continue
        m = re.match(r"Pos(\d+)[\s,]+(ch\d+)$", line) or re.match(r"(\d+)[\s,]+(ch\d+)$", line)
        if not m:
            raise ValueError(f"{path}: cannot read channel line {line!r}")
        keep.add((int(m.group(1)), m.group(2)))
    if not keep:
        raise ValueError(f"{path}: no channels listed")
    return keep


def channel_dirs(raw_root: Path, rel: Path, pos_start: int, pos_end: int,
                 keep=None) -> list[tuple[int, str]]:
    tasks = []
    for pos in range(pos_start, pos_end + 1):
        base = raw_root / f"Pos{pos}" / rel
        if not base.is_dir():
            continue
        chs = sorted((d for d in base.glob("ch*") if d.is_dir()), key=lambda p: int(p.name[2:]))
        tasks += [(pos, str(c)) for c in chs if keep is None or (pos, c.name) in keep]
    return tasks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--mask-root", required=True)
    ap.add_argument("--channel-rel", required=True, help="e.g. output_phase/channels/crop_sub_rawraw/z000")
    ap.add_argument("--model", required=True)
    ap.add_argument("--pos-start", type=int, default=1)
    ap.add_argument("--pos-end", type=int, default=999)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SEG_WORKERS", "6")))
    ap.add_argument("--max-files", type=int, default=None, help="smoke test: first N frames per channel, no _DONE")
    ap.add_argument("--eval-json", default=None, help="JSON object merged over the default eval parameters")
    ap.add_argument("--gate-hi", type=float, default=0.7)
    ap.add_argument("--gate-min-px", type=int, default=40)
    ap.add_argument("--phase-glob", default="img_*_ph_000_phase.tif")
    ap.add_argument("--channels-file", default=None,
                    help="only these channels: PosN chNN lines or pos,ch rows "
                         "(channel_contact_sheet.py --serve writes both)")
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # noqa: BLE001
        pass
    raw_root, mask_root = Path(a.raw_root), Path(a.mask_root)
    model = Path(a.model)
    if not model.exists():
        print(f"model not found: {model}", flush=True)
        return 2
    eval_params = dict(EVAL_DEFAULT)
    if a.eval_json:
        eval_params.update(json.loads(a.eval_json))
    keep = load_channel_filter(a.channels_file)
    tasks = channel_dirs(raw_root, Path(a.channel_rel), a.pos_start, a.pos_end, keep)
    if keep is not None:
        print(f"channel filter: {a.channels_file} ({len(keep)} channels listed)", flush=True)
    print(f"channels={len(tasks)} workers={a.workers} pos={a.pos_start}-{a.pos_end} max_files={a.max_files} "
          f"raw={raw_root} out={mask_root} model={model.name}", flush=True)
    if not tasks:
        return 0
    _ensure_cuda_dlls_on_path()  # before the pool: spawned workers inherit os.environ
    t0 = time.time()
    done = tot_c = tot_g = tot_e = 0
    initargs = (str(model), eval_params, a.gate_hi, a.gate_min_px, str(raw_root), str(mask_root),
                a.phase_glob, a.max_files)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=initargs) as ex:
        futs = [ex.submit(process_channel, t) for t in tasks]
        for fut in as_completed(futs):
            pos, ch, st, c, g, e, err, tiny = fut.result()
            done += 1
            tot_c += c
            tot_g += g
            tot_e += e
            el = time.time() - t0
            print(f"[{done}/{len(tasks)}] Pos{pos} {ch}: {st} cell={c} gated={g} (tiny={tiny}) errors={e} "
                  f"| {el:.0f}s {(tot_c + tot_g + tot_e) / max(el, 1e-9):.1f} frames/s"
                  + (f" | first error: {err}" if e else ""), flush=True)
    el = time.time() - t0
    print(f"DONE {done} ch in {el:.0f}s | cell_frames={tot_c} gated={tot_g} errors={tot_e} "
          f"| {(tot_c + tot_g + tot_e) / max(el, 1e-9):.1f} frames/s", flush=True)
    return 1 if tot_e else 0


if __name__ == "__main__":
    sys.exit(main())
