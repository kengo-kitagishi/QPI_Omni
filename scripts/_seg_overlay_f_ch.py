"""Segment f:/ch02 and f:/ch07 (frames 288-1152 inclusive) with the latest d20
Omnipose model, save _masks.tif / _binary.tif, and write a red mask-outline
overlay PNG per frame.

Output:
    <ch>/inference_out/img_*_masks.tif
    <ch>/inference_out/img_*_binary.tif
    <ch>/inference_out/overlays/img_*_overlay.png
"""
from __future__ import annotations

import re
import traceback
from pathlib import Path

import numpy as np
import tifffile
from cellpose_omni.models import CellposeModel
from PIL import Image

MODEL_PATH = (
    r"C:\Users\QPI\Desktop\train\omni_model_d20\models\\"
    r"cellpose_residual_on_style_on_concatenation_off_omni_abstract_"
    r"nclasses_3_nchan_1_dim_2_omni_model_d20_2026_05_01_19_01_52.321350"
)

FRAME_MIN, FRAME_MAX = 288, 1152  # inclusive

JOBS: list[tuple[Path, str]] = [
    (Path(r"f:/ch02"), "ph_000"),
    (Path(r"f:/ch07"), "ph_001"),
]

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

_FRAME_RX = re.compile(r"img_(\d+)_ph_\d+\.tif$", re.IGNORECASE)


def frame_index(path: Path) -> int | None:
    m = _FRAME_RX.search(path.name)
    return int(m.group(1)) if m else None


def to_uint8_phase(arr: np.ndarray, lo: float = -0.5, hi: float = 2.0) -> np.ndarray:
    a = np.clip((arr.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    return (a * 255).astype(np.uint8)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    return (
        ((mask != np.roll(mask, 1, 0)) |
         (mask != np.roll(mask, -1, 0)) |
         (mask != np.roll(mask, 1, 1)) |
         (mask != np.roll(mask, -1, 1))) & (mask > 0)
    )


def render_overlay(phase: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    gray = to_uint8_phase(phase)
    rgb = np.stack([gray, gray, gray], axis=-1)
    outline = mask_outline(mask)
    rgb[outline] = (255, 0, 0)
    Image.fromarray(rgb, mode="RGB").save(out_path, format="PNG", optimize=False)


print(f"[driver] loading model: {MODEL_PATH}", flush=True)
model = CellposeModel(
    gpu=True, pretrained_model=MODEL_PATH, omni=True, nchan=1, nclasses=3, dim=2
)
print("[driver] model loaded", flush=True)


def process(indir: Path, ph_tag: str) -> dict:
    outdir = indir / "inference_out"
    overlay_dir = outdir / "overlays"
    outdir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    all_tifs = sorted(indir.glob(f"img_*_{ph_tag}.tif"))
    selected = [
        p for p in all_tifs
        if (fi := frame_index(p)) is not None and FRAME_MIN <= fi <= FRAME_MAX
    ]
    print(f"\n=== {indir} ({ph_tag}) ===", flush=True)
    print(f"  matched {len(selected)} frames in [{FRAME_MIN}, {FRAME_MAX}]", flush=True)

    proc = skip = err = 0
    for i, f in enumerate(selected, 1):
        try:
            phase = tifffile.imread(str(f))
        except Exception as e:
            print(f"  [{i}/{len(selected)}] read fail: {e}", flush=True)
            err += 1
            continue
        if phase.ndim != 2:
            phase = phase.squeeze()
        try:
            masks, _, _ = model.eval([phase], **EVAL_PARAMS)
        except ValueError:
            empty = np.zeros_like(phase, dtype=np.uint16)
            tifffile.imwrite(outdir / (f.stem + "_masks.tif"), empty)
            tifffile.imwrite(outdir / (f.stem + "_binary.tif"),
                             np.full_like(empty, 255, dtype=np.uint8))
            render_overlay(phase, empty, overlay_dir / (f.stem + "_overlay.png"))
            skip += 1
            continue
        except Exception as e:
            print(f"  [{i}/{len(selected)}] eval err: {e}", flush=True)
            traceback.print_exc()
            err += 1
            continue
        m = masks[0].astype(np.uint16) if (masks is not None and np.max(masks) > 0) \
            else np.zeros_like(phase, dtype=np.uint16)
        tifffile.imwrite(outdir / (f.stem + "_masks.tif"), m)
        border = mask_outline(m)
        tifffile.imwrite(outdir / (f.stem + "_binary.tif"),
                         np.where(border, 0, 255).astype(np.uint8))
        render_overlay(phase, m, overlay_dir / (f.stem + "_overlay.png"))
        if int(m.max()) == 0:
            skip += 1
        else:
            proc += 1
        if i % 100 == 0:
            print(f"  {indir.name}: {i}/{len(selected)} (proc={proc} skip={skip} err={err})",
                  flush=True)
    return {"total": len(selected), "processed": proc, "skipped": skip, "errors": err}


summary: dict[str, dict] = {}
for indir, ph_tag in JOBS:
    try:
        summary[str(indir)] = process(indir, ph_tag)
    except Exception as e:
        print(f"!! {indir} crashed: {e}", flush=True)
        traceback.print_exc()
        summary[str(indir)] = {"error": str(e)}

print("\n=== SUMMARY ===", flush=True)
for k, v in summary.items():
    print(f"  {k}: {v}", flush=True)
