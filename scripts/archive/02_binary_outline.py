# %% 
import os
import glob
import numpy as np
import tifffile

outdir = r"G:\250910_0\Pos4\output_phase\aligned_left_center\diff_from_first\roi_subtracted_only_float32\inference_out"
mask_files = sorted(glob.glob(os.path.join(outdir, "*_masks.tif")))

if not mask_files:
    raise FileNotFoundError(f"No mask file found in {outdir}")

for i, mpath in enumerate(mask_files, 1):
    msk = tifffile.imread(mpath)
    border = ((msk != np.roll(msk, 1, 0)) |
              (msk != np.roll(msk, -1, 0)) |
              (msk != np.roll(msk, 1, 1)) |
              (msk != np.roll(msk, -1, 1))) & (msk > 0)
    binary = np.where(border, 0, 255).astype(np.uint8)
    out_path = mpath.replace("_masks.tif", "_binary.tif")
    tifffile.imwrite(out_path, binary)
    print(f"[{i}/{len(mask_files)}] saved {os.path.basename(out_path)}")

print("Done.")

# %%
