"""Export every phase TIFF in Pos51/ch10 as a viridis-colored PNG.

Output:
    <ch10>/viridis_pngs/frame_<INDEX>.png   (one per phase tif)
    <ch10>/viridis_pngs/_colorbar.png       (standalone viridis colorbar)

vmin/vmax follow the intensity_kymograph defaults (-0.5 .. 2.0) so the
per-frame frames and the kymograph share a colormap.
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from matplotlib import cm
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from PIL import Image

CH_DIR = Path(r"f:/260514/Pos51/output_phase/channels/crop_sub_raw_raw/ch10")
OUT_DIR = CH_DIR / "viridis_pngs"
VMIN = -0.5
VMAX = 2.0

_FRAME_RX = re.compile(r"img_(\d+)_ph_000\.tif$", re.IGNORECASE)


def render_viridis_png(arr: np.ndarray, out_path: Path) -> None:
    norm = np.clip((arr.astype(np.float32) - VMIN) / (VMAX - VMIN), 0.0, 1.0)
    rgba = (cm.get_cmap("viridis")(norm) * 255).astype(np.uint8)
    Image.fromarray(rgba, mode="RGBA").save(out_path, format="PNG", optimize=False)


def save_colorbar(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.2, 3.4), constrained_layout=True)
    cb = ColorbarBase(
        ax,
        cmap=plt.get_cmap("viridis"),
        norm=Normalize(vmin=VMIN, vmax=VMAX),
        orientation="vertical",
    )
    cb.set_label("Phase (rad)")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tifs = sorted(CH_DIR.glob("img_*_ph_000.tif"))
    if not tifs:
        print(f"no phase tifs in {CH_DIR}")
        return 1
    print(f"{len(tifs)} phase tifs -> {OUT_DIR}")
    for i, tif in enumerate(tifs, 1):
        m = _FRAME_RX.search(tif.name)
        idx = m.group(1) if m else f"{i:07d}"
        arr = tifffile.imread(str(tif))
        if arr.ndim != 2:
            arr = arr.squeeze()
        render_viridis_png(arr, OUT_DIR / f"frame_{idx}.png")
        if i % 200 == 0:
            print(f"  {i}/{len(tifs)}", flush=True)
    save_colorbar(OUT_DIR / "_colorbar.png")
    print(f"done. wrote {len(tifs)} frames + 1 colorbar to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
