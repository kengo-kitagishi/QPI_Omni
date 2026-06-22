"""
build_inferno_hyperstack.py
---------------------------
Assemble per-Pos ImageJ hyperstacks (Z x T) from the online crop_sub_rawraw
delta TIFs, with the channels tiled vertically into one frame and an inferno
0-1.6 rad colormap baked in (RGB). Open in ImageJ -> scroll Z and T sliders.

Source TIFs (background-anchored deltas, written by compute_drift_online Phase B):
  {ROOT}/{Pos}/output_phase/channels/crop_sub_rawraw/z{zzz}/ch{cc}/img_{t}_ph_{zzz}.tif

Output (one ImageJ-openable RGB hyperstack per Pos):
  {OUT_DIR}/{Pos}_inferno_z{nz}_t{nt}.tif   (axes TZYXS, uint8 RGB)

Re-run any time as more timepoints accumulate. Never touches the source TIFs.
"""
import re
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.cm as cm

ROOT = Path(r"E:\260617\2per_corr_zstack_3_crop_sub")
OUT_DIR = Path(r"E:\260617\inferno_hyperstacks")
POSITIONS = None   # None -> all PosN found under ROOT (sorted); or a list of labels
VMIN, VMAX = 0.15, 1.95
MIN_FRAME = 2   # drop T=0,1 (initial drift-correction settling)
CMAP = cm.get_cmap("inferno")
GAP = 2  # px separator between tiled channels


def frame_index(p):
    m = re.search(r"img_0*([0-9]+)_ph", p.name)
    return int(m.group(1)) if m else -1


def build_one(pos):
    base = ROOT / pos / "output_phase" / "channels" / "crop_sub_rawraw"
    if not base.exists():
        print(f"  {pos}: no crop_sub_rawraw"); return
    z_dirs = sorted([d for d in base.iterdir() if d.is_dir() and re.match(r"z\d+$", d.name)],
                    key=lambda d: int(d.name[1:]))
    if not z_dirs:
        print(f"  {pos}: no z dirs"); return
    ch_dirs = sorted([d for d in z_dirs[0].iterdir() if d.is_dir() and re.match(r"ch\d+$", d.name)],
                     key=lambda d: int(d.name[2:]))
    # frame indices common to z0/ch0 (assume same across z/ch)
    frames = sorted(f for f in {frame_index(p) for p in (z_dirs[0] / ch_dirs[0].name).glob("*.tif")}
                    if f >= MIN_FRAME)
    if not frames:
        print(f"  {pos}: no frames"); return
    # crop size from one tif
    sample = next((z_dirs[0] / ch_dirs[0].name).glob("*.tif"))
    h, w = tifffile.imread(str(sample)).shape
    nz, nt, nch = len(z_dirs), len(frames), len(ch_dirs)
    tiled_h = nch * h + (nch - 1) * GAP
    # [T, Z, Y, X, 3] uint8 RGB -- inferno baked in over [VMIN, VMAX]
    stack = np.zeros((nt, nz, tiled_h, w, 3), dtype=np.uint8)
    for zi, zd in enumerate(z_dirs):
        for ti, t in enumerate(frames):
            for ci, cd in enumerate(ch_dirs):
                fp = zd / cd.name / f"img_{t:09d}_ph_{zd.name[1:].zfill(3)}.tif"
                if not fp.exists():
                    continue
                img = tifffile.imread(str(fp)).astype(np.float64)
                norm = np.clip((img - VMIN) / (VMAX - VMIN), 0, 1)
                rgb = (CMAP(norm)[..., :3] * 255).astype(np.uint8)
                y0 = ci * (h + GAP)
                stack[ti, zi, y0:y0 + h, :, :] = rgb
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{pos}_inferno.tif"   # stable name -> re-runs overwrite
    tifffile.imwrite(str(out), stack, imagej=True, metadata={"axes": "TZYXS"})
    print(f"  {pos}: T={nt} Z={nz} ch={nch} ({h}x{w}) -> {out.name}")


def main():
    positions = POSITIONS
    if positions is None:
        positions = sorted([d.name for d in ROOT.iterdir()
                            if d.is_dir() and re.match(r"Pos\d+$", d.name)],
                           key=lambda s: int(s[3:]))
    print(f"inferno hyperstack (vmin={VMIN}, vmax={VMAX}) from {ROOT}  [{len(positions)} pos]")
    for pos in positions:
        build_one(pos)
    print(f"\nOpen in ImageJ: {OUT_DIR}\\*_inferno_*.tif  (scroll Z and T sliders)")


if __name__ == "__main__":
    main()
