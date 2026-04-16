# %%
import os
import glob
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# === Settings ===
input_dir = r"C:\Users\QPI\Desktop\align_demo\output\subtracted"
output_dir = os.path.join(input_dir, "colormap")
os.makedirs(output_dir, exist_ok=True)

# Colormap settings
vmin = -0.1
vmax = 1.7
cmap = "RdBu_r"

# Normalization centered at 0 (white)
norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

# === Process all TIF files ===
tif_files = glob.glob(os.path.join(input_dir, "*.tif"))

for path in tif_files:
    # Load image
    img = tifffile.imread(path).astype(float)

    # Output filename
    fname = os.path.splitext(os.path.basename(path))[0]
    out_png = os.path.join(output_dir, f"{fname}_RdBu.png")

    # Draw
    plt.figure(figsize=(6, 5))
    plt.imshow(img, cmap=cmap, norm=norm)
    plt.colorbar(label="Intensity (a.u.)")
    plt.title(fname)
    plt.axis("off")

    # Save
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", out_png)

print("=== Done ===")

# %%
