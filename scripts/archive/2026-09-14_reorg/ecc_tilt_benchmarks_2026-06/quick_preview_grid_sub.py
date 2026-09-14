"""
quick_preview_grid_sub.py — Display N frames from output_phase_grid_sub with viridis colormap
"""
from pathlib import Path
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# ── Settings ─────────────────────────────────────────
OUT_DIR = Path(r"C:\ph_1\Pos1\output_phase_grid_sub")
N_SHOW  = 9       # Number of frames to display (equally spaced sampling)
VMIN    = -0.5
VMAX    =  0.5
# ──────────────────────────────────────────────────────

frames = sorted(OUT_DIR.glob("*_phase.tif"))
if not frames:
    raise FileNotFoundError(f"no frames in {OUT_DIR}")

indices = np.linspace(0, len(frames) - 1, N_SHOW, dtype=int)
cols = 3
rows = (N_SHOW + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
axes = axes.flatten()

for ax, idx in zip(axes, indices):
    img = tifffile.imread(str(frames[idx])).astype(np.float32)
    im = ax.imshow(img, cmap="viridis", vmin=VMIN, vmax=VMAX)
    ax.set_title(f"frame {idx}", fontsize=9)
    ax.axis("off")

for ax in axes[len(indices):]:
    ax.axis("off")

fig.colorbar(im, ax=axes[:len(indices)], shrink=0.6, label="phase (rad)")
fig.suptitle(f"grid_sub preview  vmin={VMIN}  vmax={VMAX}  (total {len(frames)} frames)", fontsize=11)
plt.tight_layout()
plt.show()
