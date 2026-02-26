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

# csv_path = r"G:\250910_0\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\Results.csv" #251101_0_pos1
#csv_path = r"G:\250910_0\Pos3\output_phase\aligned_left_center\mask_for_first_backsub\Results.csv" #251101_0_pos3
#csv_path = r"F:\250815_kk\ph_1\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\Results.csv" #251102_0.0055_pos1
#csv_path = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\Results.csv" #251102_0.0055_pos1
#csv_path = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\Results.csv" #251102_0.0055_pos1
#csv_path = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32\Results.csv"
#csv_path = r"F:\250611_kk\ph_1\Pos1\output_phase\crop_150_300\subtracted_by_maskmean_float32\Results.csv"
csv_path = r"F:\250611_kk\ph_1\Pos2\output_phase\crop_150_300\subtracted_by_maskmean_float32\Results.csv"
csv_path = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300\subtracted_by_maskmean_float32\Results.csv"

#image_dir = r"G:\250910_0\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32" #251101_0_Pos1
#image_dir = r"F:\250815_kk\ph_1\Pos1\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"G:\250815_kk\ph_1\Pos2\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"G:\250815_kk\ph_1\Pos3\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"H:\250910_0\Pos4\output_phase\aligned_left_center\crop_150_300\subtracted_by_maskmean_float32"
#image_dir = r"F:\250611_kk\ph_1\Pos1\output_phase\crop_150_300\subtracted_by_maskmean_float32"
image_dir = r"F:\250611_kk\ph_1\Pos2\output_phase\crop_150_300\subtracted_by_maskmean_float32"
image_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300\subtracted_by_maskmean_float32"

output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# === CSV読み込みとSlice順ソート ===
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

# === 画像読み込み（自然順ソート） ===
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
assert len(image_paths) == len(dx), f"画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません。"

# === アライメント処理 ===
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print("アライメント完了", output_dir)
# %%
output_dir = os.path.join(image_dir, "aligned_left_center/crop_128_256")
aligned_dir = output_dir
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
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
