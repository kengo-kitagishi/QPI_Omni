# %%
"""
アライメントと差分解析スクリプト

CSVベースのアライメント、1枚目からの差分画像生成、カラーマップ作成
"""

import os
import numpy as np
import pandas as pd
import tifffile
import cv2
from glob import glob
from natsort import natsorted
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# %%
# ==================== パラメータ設定 ====================

# パス設定（使用時に変更）
csv_path = "/Volumes/QPI/ph_1/Pos3/output_phase/Results.csv"
image_dir = "/Volumes/QPI/ph_1/Pos3/output_phase"

# アライメント方法選択
# "left_center": 左辺の中点
# "right_center": 右辺の中点
# "center": 矩形の中心
alignment_point = "left_center"

# カラーマップ設定
vmin, vmax = -0.5, 3.0
colormap = "viridis"  # viridis, plasma, JET, RdBu_r など

print("=" * 80)
print("アライメントと差分解析")
print("=" * 80)
print(f"\nCSVパス: {csv_path}")
print(f"画像ディレクトリ: {image_dir}")
print(f"アライメント基準: {alignment_point}")

# %%
# ==================== セル1: CSVベースアライメント（左辺中点） ====================

print("\n" + "=" * 80)
print("CSVベースアライメント（左辺中点）")
print("=" * 80)

# 出力ディレクトリ
output_dir = os.path.join(image_dir, "aligned_left_center")
os.makedirs(output_dir, exist_ok=True)

# CSV読み込みとSlice順ソート
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

print(f"\nCSV行数: {len(df)}")

# 左辺の中点の座標を取得
x = df["BX"]
y = df["BY"] + df["Height"] / 2

# 参照フレーム（最初のスライス）の座標
x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

print(f"基準位置: ({x0:.2f}, {y0:.2f})")
print(f"最大シフト量: dx={dx.abs().max():.2f}, dy={dy.abs().max():.2f}")

# 画像読み込み（自然順ソート）
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))

# 'aligned_'を含むファイルを除外
image_paths = [p for p in image_paths if 'aligned' not in os.path.basename(p)]

print(f"画像数: {len(image_paths)}")

if len(image_paths) != len(dx):
    print(f"⚠️  警告: 画像数（{len(image_paths)}）とCSV行数（{len(dx)}）が一致しません")
    print("   CSV行数に合わせて処理します")
    image_paths = image_paths[:len(dx)]

# アライメント処理
print("\nアライメント処理中...")
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print(f"✅ アライメント完了: {output_dir}")

# %%
# ==================== セル2: CSVベースアライメント（右辺中点） ====================

print("\n" + "=" * 80)
print("CSVベースアライメント（右辺中点）")
print("=" * 80)

# 出力ディレクトリ
output_dir = os.path.join(image_dir, "aligned_right_center")
os.makedirs(output_dir, exist_ok=True)

# CSV読み込み
df = pd.read_csv(csv_path)
df = df.sort_values("Slice").reset_index(drop=True)

# 右辺の中点座標を計算
x = df["BX"] + df["Width"]
y = df["BY"] + df["Height"] / 2

x0, y0 = x.iloc[0], y.iloc[0]
dx = x - x0
dy = y - y0

print(f"基準位置: ({x0:.2f}, {y0:.2f})")
print(f"最大シフト量: dx={dx.abs().max():.2f}, dy={dy.abs().max():.2f}")

# 画像読み込み
image_paths = natsorted(glob(os.path.join(image_dir, "*.tif")))
image_paths = [p for p in image_paths if 'aligned' not in os.path.basename(p)]
image_paths = image_paths[:len(dx)]

# アライメント処理
print("\nアライメント処理中...")
for i, path in enumerate(image_paths):
    img = tifffile.imread(path)
    M = np.float32([[1, 0, -dx.iloc[i]], [0, 1, -dy.iloc[i]]])
    aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    fname = os.path.basename(path)
    tifffile.imwrite(os.path.join(output_dir, fname), aligned.astype(np.float32))

print(f"✅ アライメント完了: {output_dir}")

# %%
# ==================== セル3: 1枚目からの差分画像生成 ====================

print("\n" + "=" * 80)
print("1枚目からの差分画像生成")
print("=" * 80)

# アライメント済みディレクトリ（使用時に変更）
aligned_dir = os.path.join(image_dir, "aligned_left_center")
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
os.makedirs(output_diff_dir, exist_ok=True)

# ファイル読み込みとソート
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))

if len(image_paths) < 2:
    print("❌ 画像が2枚未満です")
else:
    print(f"画像数: {len(image_paths)}")
    
    # 1枚目を基準として読み込む
    ref_img = tifffile.imread(image_paths[0]).astype(np.float32)
    print(f"基準画像: {os.path.basename(image_paths[0])}")
    
    # 差分画像を保存
    print("\n差分計算中...")
    for i, path in enumerate(image_paths[1:], start=1):
        img = tifffile.imread(path).astype(np.float32)
        diff = img - ref_img
        fname = os.path.basename(path)
        tifffile.imwrite(os.path.join(output_diff_dir, fname), diff)
    
    print(f"✅ 差分画像を保存しました: {output_diff_dir}")

# %%
# ==================== セル4: カラーマップ付き差分画像生成 ====================

print("\n" + "=" * 80)
print("カラーマップ付き差分画像生成")
print("=" * 80)

# 出力ディレクトリ
output_colormap_dir = os.path.join(aligned_dir, "diff_colormap")
os.makedirs(output_colormap_dir, exist_ok=True)

# ファイル読み込み
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

print(f"差分表示範囲: [{vmin}, {vmax}]")
print(f"カラーマップ: {colormap}")

# カラーマップ設定
if colormap.lower() == "jet":
    cmap_cv = cv2.COLORMAP_JET
elif colormap.lower() == "viridis":
    cmap_cv = cv2.COLORMAP_VIRIDIS
elif colormap.lower() == "hot":
    cmap_cv = cv2.COLORMAP_HOT
else:
    cmap_cv = cv2.COLORMAP_JET

# 差分 + カラーマップ処理
print("\nカラーマップ生成中...")
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    
    # 指定範囲でクリップして正規化（0-255）
    diff_clipped = np.clip(diff, vmin, vmax)
    diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    
    # カラーマップ適用
    color_mapped = cv2.applyColorMap(diff_norm, cmap_cv)
    
    # 保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap.png"
    cv2.imwrite(os.path.join(output_colormap_dir, fname), color_mapped)

print(f"✅ カラーマップ画像を保存しました: {output_colormap_dir}")

# %%
# ==================== セル5: Matplotlib版カラーマップ（高品質） ====================

print("\n" + "=" * 80)
print("Matplotlib版カラーマップ（高品質）")
print("=" * 80)

# 出力ディレクトリ
output_colormap_mpl_dir = os.path.join(aligned_dir, "diff_colormap_matplotlib")
os.makedirs(output_colormap_mpl_dir, exist_ok=True)

# カラーマップ設定
cmap_mpl = cm.get_cmap(colormap)
norm = Normalize(vmin=vmin, vmax=vmax)

# ファイル読み込み
image_paths = natsorted(glob(os.path.join(aligned_dir, "*.tif")))
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

print(f"カラーマップ: {colormap}")
print(f"差分表示範囲: [{vmin}, {vmax}]")

# 差分 + カラーマップ処理
print("\nMatplotlibカラーマップ生成中...")
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img
    
    # 正規化してカラーマップに変換（RGBA）
    rgba_image = cmap_mpl(norm(diff))  # shape: (H, W, 4)
    rgb_image = (rgba_image[..., :3] * 255).astype(np.uint8)
    
    # TIF保存
    fname = os.path.splitext(os.path.basename(path))[0] + "_cmap_mpl.tif"
    tifffile.imwrite(os.path.join(output_colormap_mpl_dir, fname), rgb_image)

print(f"✅ Matplotlibカラーマップ画像を保存しました: {output_colormap_mpl_dir}")

# %%
# ==================== 完了 ====================

print("\n" + "=" * 80)
print("アライメント・差分解析スクリプト完了")
print("=" * 80)

print("\n【使い方】")
print("  1. パラメータ設定セルで、CSVパスと画像ディレクトリを設定")
print("  2. 必要なセルを実行:")
print("     - セル1: 左辺中点でアライメント")
print("     - セル2: 右辺中点でアライメント")
print("     - セル3: 1枚目からの差分計算")
print("     - セル4: OpenCV版カラーマップ")
print("     - セル5: Matplotlib版カラーマップ（高品質）")

print("\n【次のステップ】")
print("  - 差分画像の確認")
print("  - 時系列解析")
print("  - 統計解析")

# %%

