"""One-shot driver: load the d20 Omnipose model once, then segment every ch* dir.

Mirrors the eval params and per-file outputs of 07_segmentation.py but reuses the
in-memory model across all 17 channel directories to amortize GPU init.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import tifffile
from cellpose_omni import io
from cellpose_omni.models import CellposeModel

MODEL_PATH = (
    r"C:\Users\QPI\Desktop\train\omni_model_d20\models\\"
    r"cellpose_residual_on_style_on_concatenation_off_omni_abstract_"
    r"nclasses_3_nchan_1_dim_2_omni_model_d20_2026_05_01_19_01_52.321350"
)

CH_DIRS: list[Path] = []
for ch in sorted(Path(r"f:/260514/Pos50/output_phase/channels/crop_sub_rawraw").iterdir()):
    if ch.is_dir() and ch.name.startswith("ch"):
        CH_DIRS.append(ch)
for ch in sorted(Path(r"f:/260514/Pos51/output_phase/channels/crop_sub_raw_raw").iterdir()):
    if ch.is_dir() and ch.name.startswith("ch"):
        CH_DIRS.append(ch)

EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=20,
    normalize=True,
    tile=False,
    net_avg=True,
    omni=True,
    verbose=False,
    flow_threshold=0.4,
    mask_threshold=0,
    min_size=10,
)

print(f"[driver] loading model: {MODEL_PATH}", flush=True)
model = CellposeModel(
    gpu=True, pretrained_model=MODEL_PATH, omni=True, nchan=1, nclasses=3, dim=2
)
print("[driver] model loaded", flush=True)


def segment_dir(indir: Path) -> dict:
    outdir = indir / "inference_out"
    outdir.mkdir(parents=True, exist_ok=True)
    files = io.get_image_files(str(indir), mask_filter="_masks", look_one_level_down=False)
    if not isinstance(files, (list, tuple)):
        files = list(files)
    proc = []
    for f in files:
        if isinstance(f, (list, tuple)):
            if f:
                proc.append(f[0])
        else:
            proc.append(f)
    files = proc
    print(f"[{indir}] {len(files)} files", flush=True)
    n_proc = n_skip = n_err = 0
    for i, f in enumerate(files, 1):
        try:
            img = tifffile.imread(f)
        except Exception as e:
            print(f"  [{i}/{len(files)}] read fail: {e}", flush=True)
            n_err += 1
            continue
        try:
            masks, _, _ = model.eval([img], **EVAL_PARAMS)
        except ValueError as e:
            empty = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(outdir / (Path(f).stem + "_masks.tif"), empty)
            tifffile.imwrite(outdir / (Path(f).stem + "_binary.tif"),
                             np.full_like(empty, 255, dtype=np.uint8))
            n_skip += 1
            continue
        except Exception as e:
            print(f"  [{i}/{len(files)}] eval err: {e}", flush=True)
            traceback.print_exc()
            empty = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(outdir / (Path(f).stem + "_masks.tif"), empty)
            tifffile.imwrite(outdir / (Path(f).stem + "_binary.tif"),
                             np.full_like(empty, 255, dtype=np.uint8))
            n_err += 1
            continue
        if masks is None or np.max(masks) == 0:
            empty = np.zeros_like(img, dtype=np.uint16)
            tifffile.imwrite(outdir / (Path(f).stem + "_masks.tif"), empty)
            tifffile.imwrite(outdir / (Path(f).stem + "_binary.tif"),
                             np.full_like(empty, 255, dtype=np.uint8))
            n_skip += 1
            continue
        m = masks[0].astype(np.uint16)
        tifffile.imwrite(outdir / (Path(f).stem + "_masks.tif"), m)
        border = ((m != np.roll(m, 1, 0)) | (m != np.roll(m, -1, 0)) |
                  (m != np.roll(m, 1, 1)) | (m != np.roll(m, -1, 1))) & (m > 0)
        tifffile.imwrite(outdir / (Path(f).stem + "_binary.tif"),
                         np.where(border, 0, 255).astype(np.uint8))
        n_proc += 1
        if i % 200 == 0:
            print(f"  [{indir.parent.parent.parent.parent.name}/{indir.name}] "
                  f"{i}/{len(files)} (proc={n_proc} skip={n_skip} err={n_err})",
                  flush=True)
    return dict(total=len(files), processed=n_proc, skipped=n_skip, errors=n_err)


summary: dict[str, dict] = {}
for ch in CH_DIRS:
    pos_name = ch.parent.parent.parent.parent.name
    label = f"{pos_name}/{ch.name}"
    print(f"\n=== {label} ===", flush=True)
    try:
        summary[label] = segment_dir(ch)
    except Exception as e:
        print(f"!! {label} crashed: {e}", flush=True)
        traceback.print_exc()
        summary[label] = {"error": str(e)}

print("\n=== SUMMARY ===", flush=True)
for k, v in summary.items():
    print(f"  {k}: {v}", flush=True)
