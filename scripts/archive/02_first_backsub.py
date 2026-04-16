# %%
import os, glob
import numpy as np
import tifffile as tiff

# ====== Input ======
#seq_dir  = r"G:\250910_0\Pos4\output_phase\crop_150_300"  #251030_0_Pos4
#seq_dir = r"G:\250910_0\Pos1\output_phase\aligned_left_center\crop_150_300" #251101_0_Pos1
#seq_dir = r"G:\250910_0\Pos3\output_phase\aligned_left_center\crop_150_300" #251101_0_Pos3
# seq_dir = r"F:\250815_kk\ph_1\Pos1\output_phase\aligned_left_center\crop_150_300" #251102_0.0055_Pos1
#seq_dir = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\crop_150_300" #251102_0.0055_Pos2
#seq_dir = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\crop_150_300" #251102_0.0055_Pos3
#seq_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300" #251102_0_Pos4
#seq_dir = r"F:\250611_kk\ph_1\Pos1\output_phase\crop_150_300"
#seq_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\crop_150_300"
seq_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300"

#mask_dir = r"G:\250910_0\Pos4\output_phase\crop_150_300\mask_for_first_backsub"  #251030_0_Pos4
#mask_dir = r"G:\250910_0\Pos1\output_phase\aligned_left_center\mask_for_first_backsub"  #251101_0_Pos1
#mask_dir = r"G:\250910_0\Pos3\output_phase\aligned_left_center\mask_for_first_backsub"  #251101_0_Pos3
#mask_dir = r"F:\250815_kk\ph_1\Pos1\output_phase\aligned_left_center\mask_for_first_backsub" #251102_0.0055_Pos1
#mask_dir = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\mask_for_first_backsub" #251102_0.0055_Pos2
#mask_dir = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\mask_for_first_backsub" #251102_0.0055_Pos3
#mask_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\mask_for_first_backsub" #251102_0_Pos4
#mask_dir = r"F:\250611_kk\ph_1\Pos1\output_phase\mask_for_first_backsub" #251103_0.01_Pos1
#mask_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\mask_for_first_backsub"
mask_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\mask_for_first_backsub"


out_dir  = os.path.join(seq_dir, "subtracted_by_maskmean_float32")

os.makedirs(out_dir, exist_ok=True)

# ====== File mapping ======
frames = sorted(glob.glob(os.path.join(seq_dir, "*.tif")))
masks  = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))

if not frames:
    raise FileNotFoundError(f"Sequence images not found: {seq_dir}")
if len(frames) != len(masks):
    print(f"WARNING: count mismatch -> images={len(frames)}, masks={len(masks)}")

# ====== Process each pair ======
processed = 0
skipped = 0

for f, m in zip(frames, masks):
    IMG = tiff.imread(f).astype(np.float32)
    MSK_raw = tiff.imread(m)

    if MSK_raw.ndim > 2:
        MSK_raw = MSK_raw[..., 0]
    MSK = (MSK_raw > 0)

    # Size check
    if IMG.shape[:2] != MSK.shape[:2]:
        print(f"[SKIP] Size mismatch: {os.path.basename(f)} {IMG.shape} vs {MSK.shape}")
        skipped += 1
        continue

    if not MSK.any():
        print(f"[SKIP] No mask region: {os.path.basename(f)}")
        skipped += 1
        continue

    # Calculate mean of mask region (white)
    mean_in_mask = float(IMG[MSK].mean())

    # Subtract mean from entire image
    OUT = IMG - mean_in_mask

    # Save
    base = os.path.splitext(os.path.basename(f))[0]
    out_path = os.path.join(out_dir, f"{base}_subtracted.tif")
    tiff.imwrite(out_path, OUT.astype(np.float32))

    processed += 1
    print(f"[OK] {os.path.basename(f)} mean_in_mask={mean_in_mask:.6f}")

print(f"Done: {processed} images (skipped: {skipped}) -> {out_dir}")

# %%
