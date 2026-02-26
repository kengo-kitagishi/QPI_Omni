# %%
import os
import pandas as pd
import numpy as np
import tifffile
import cv2
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
# %%
#image_dir = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"F:\250611_kk\ph_1\Pos1\output_phase\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\crop_150_300\subtracted_by_maskmean_float32"
image_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300\subtracted_by_maskmean_float32"
output_dir = os.path.join(image_dir, "aligned_left_center/crop_128_256")

aligned_dir = output_dir
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")


#aligned_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\aligned_left_center\crop_128_256\diff_from_first\subtracted_by_maskmean_float32"
#output_diff_dir = os.path.join(aligned_dir, "diff_from_first")

os.makedirs(output_diff_dir, exist_ok=True)

# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
assert len(image_paths) > 1, "画像が1枚しか見つかりませんでした。"

# === 1枚目を基準として読み込む ===
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# === 差分画像を保存 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)

print("✅ 差分画像を保存しました:", output_diff_dir)

# %%
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)
output_colormap_tif_dir = os.path.join(aligned_dir, "diff_colormap_tif")
os.makedirs(output_colormap_tif_dir, exist_ok=True)


# === ファイル読み込みとソート ===
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

vmin, vmax = -0.5, 2.0

# === 差分＋カラーマップ処理 ===
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img

    # 指定範囲でクリップして正規化（0-255）
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)

    # カラーマップ適用（JET）
    color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)

    # 保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_fixed.tif"
    tifffile.imwrite(os.path.join(output_colormap_tif_dir, fname), color_mapped)

print("✅ カラーマップ画像を保存しました（範囲固定: -1〜3）→", output_colormap_dir)

# %%
