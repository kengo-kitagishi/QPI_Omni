# 厚みマップとRI計算 - 完全ガイド

**作成日**: 2025-12-23  
**目的**: ROIセットから厚みマップを生成し、定量位相画像と組み合わせてRI（屈折率）を計算

---

## 🎯 概要

このワークフローは、以下を実現します：

1. **ROIセット → 3D再構成 → 厚みマップ（z-stack数の2D画像）**
2. **厚みマップ + 位相差画像 → RI計算**
3. **Mean RI, Total RIの時系列追跡**

### 出力される厚みマップ

各XYピクセル位置での**Z方向の占有スライス数**を表す2D画像：

```
厚みマップ (thickness_map):
  - 形式: 2D float32 TIFF
  - 値: 各ピクセルでのZ方向スライス数
  - 例: 値が8.0 = そのピクセル位置で8スライス分の厚み
```

これは**24_elip_volume.pyのzstack.tif**と同等の情報です！

---

## 🚀 基本的な使い方

### ステップ1: 厚みマップを生成

```python
from timeseries_volume_from_roiset import TimeSeriesVolumeTracker

# Trackerを作成
tracker = TimeSeriesVolumeTracker(
    roi_zip_path="RoiSet.zip",
    voxel_xy=0.08625,  # XYピクセルサイズ (um)
    voxel_z=0.08625,   # Z方向ステップ (um) ← XYと同じにすることも可能
    radius_enlarge=1.0,
    image_width=512,
    image_height=512
)

# 体積 + 厚みマップを追跡
results_df = tracker.track_volume_timeseries(
    max_frames=None,  # 全フレーム処理
    save_thickness_maps=True  # 厚みマップを保存
)

# 結果を保存
tracker.save_results('output_dir')
```

**出力**:
```
output_dir/
├── volume_timeseries.csv           # 体積データ
├── thickness_maps/                 # 個別の厚みマップ
│   ├── 0085-0024-0136_thickness.tif
│   ├── 0086-0024-0136_thickness.tif
│   └── ...
└── thickness_stack_all_frames.tif  # 全フレームの統合スタック
```

---

### ステップ2: RI計算（位相差画像がある場合）

```python
# 位相差画像ディレクトリを指定
ri_results = tracker.compute_ri_from_phase_images(
    phase_image_dir='path/to/phase_images/',
    wavelength_nm=663,      # 波長 (nm)
    n_medium=1.333          # 培地の屈折率
)

# RI結果を保存
tracker.save_ri_results('output_dir')
```

**出力**:
```
output_dir/
├── ri_statistics.csv    # Mean RI, Total RIなど
├── ri_summary.txt       # 統計サマリー
└── ri_maps/             # 個別のRIマップ
    ├── 0085-0024-0136_ri_map.tif
    └── ...
```

---

## 📊 厚みマップの詳細

### 厚みマップとは

**定義**: 各XYピクセル位置でのZ方向の「占有スライス数」

```
例:
  ピクセル (100, 150):
    - Z方向で10スライス分の細胞が存在
    → thickness_map[150, 100] = 10.0
```

### 実際の厚み（µm）への変換

```python
# 厚みマップを読み込み
thickness_map = tifffile.imread('thickness_maps/0085-0024-0136_thickness.tif')

# スライス数 → 実際の厚み (um)
voxel_z = 0.08625  # um/slice
thickness_um = thickness_map * voxel_z

print(f"Max thickness: {np.max(thickness_um):.2f} um")
```

### 可視化

```python
import matplotlib.pyplot as plt
import tifffile

# 厚みマップを読み込み
thickness_map = tifffile.imread('thickness_maps/0085-0024-0136_thickness.tif')

# 可視化
plt.figure(figsize=(10, 8))
plt.imshow(thickness_map, cmap='hot', interpolation='nearest')
plt.colorbar(label='Thickness (z-slices)')
plt.title('Cell Thickness Map')
plt.xlabel('X (pixels)')
plt.ylabel('Y (pixels)')
plt.savefig('thickness_visualization.png', dpi=300)
plt.show()
```

---

## 🔬 RI計算の原理

### 物理式

位相差と屈折率の関係：

```
φ = (2π/λ) × (n_sample - n_medium) × h
```

ここで：
- φ: 位相差（ラジアン）
- λ: 波長（µm）
- n_sample: サンプルの屈折率
- n_medium: 培地の屈折率
- h: 厚み（µm）

屈折率を求める：

```
n_sample = n_medium + (φ × λ) / (2π × h)
```

### 実装

```python
# パラメータ
wavelength_nm = 663  # HeNe レーザー
wavelength_um = wavelength_nm / 1000.0  # 0.663 um
n_medium = 1.333  # 水/培地

# 厚みマップ（スライス数 → um）
thickness_um = thickness_map * voxel_z

# 位相差画像（ラジアン）
phase_image = tifffile.imread('phase_image.tif')

# RI計算
n_sample = n_medium + (phase_image * wavelength_um) / (2 * np.pi * thickness_um)

# マスク内のみ
mask = thickness_um > 0
mean_ri = np.mean(n_sample[mask])
total_ri = np.sum(n_sample[mask] - n_medium)

print(f"Mean RI: {mean_ri:.4f}")
print(f"Total RI: {total_ri:.2f}")
```

---

## 📈 batch_analysis.pyとの対応

### 28_batch_analysis.pyの出力

```python
# batch_analysis.pyは以下を計算:
- Mean RI: セル内の平均屈折率
- Total RI: セル全体の積分屈折率
- RI map: 各ピクセルのRI分布
```

### 本ツールの出力

```python
# timeseries_volume_from_roiset.pyも同じ:
ri_results = tracker.compute_ri_from_phase_images(...)

# 各フレームで:
- mean_ri: Mean RI
- total_ri: Total RI  
- ri_map: RI分布マップ（TIFF）
```

### CSVデータ

**ri_statistics.csv**:
```csv
time_index,roi_name,mean_ri,median_ri,std_ri,total_ri,n_pixels
0,0085-0024-0136.roi,1.3850,1.3845,0.0012,125.50,5432
1,0086-0024-0136.roi,1.3852,1.3848,0.0011,126.20,5401
```

---

## 🎯 応用例

### 1. 時系列のMean RI変化をプロット

```python
import pandas as pd
import matplotlib.pyplot as plt

# RI統計を読み込み
ri_df = pd.read_csv('output_dir/ri_statistics.csv')

# プロット
plt.figure(figsize=(12, 6))

# Mean RI
plt.subplot(1, 2, 1)
plt.plot(ri_df['time_index'], ri_df['mean_ri'], 'o-')
plt.xlabel('Time (frame)')
plt.ylabel('Mean RI')
plt.title('Mean RI Over Time')
plt.grid(True, alpha=0.3)

# Total RI
plt.subplot(1, 2, 2)
plt.plot(ri_df['time_index'], ri_df['total_ri'], 'o-')
plt.xlabel('Time (frame)')
plt.ylabel('Total RI')
plt.title('Total RI Over Time')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ri_timeseries.png', dpi=300)
plt.show()
```

### 2. 厚みマップとRIマップの比較

```python
import tifffile
import matplotlib.pyplot as plt

# データ読み込み
thickness_map = tifffile.imread('thickness_maps/0085-0024-0136_thickness.tif')
ri_map = tifffile.imread('ri_maps/0085-0024-0136_ri_map.tif')

# 並べて表示
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 厚みマップ
ax = axes[0]
im = ax.imshow(thickness_map, cmap='hot')
ax.set_title('Thickness Map (z-slices)', fontsize=14)
plt.colorbar(im, ax=ax, label='Slices')

# RIマップ
ax = axes[1]
im = ax.imshow(ri_map, cmap='viridis', vmin=1.33, vmax=1.40)
ax.set_title('RI Map', fontsize=14)
plt.colorbar(im, ax=ax, label='RI')

# RI vs Thickness散布図
ax = axes[2]
mask = thickness_map > 0
ax.scatter(thickness_map[mask], ri_map[mask], alpha=0.1, s=1)
ax.set_xlabel('Thickness (slices)')
ax.set_ylabel('RI')
ax.set_title('RI vs Thickness', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thickness_ri_comparison.png', dpi=300)
plt.show()
```

### 3. 質量濃度への変換

```python
# RI → 質量濃度
# C [mg/ml] = (RI - RI_medium) / α
# α = 0.0018 ml/mg (タンパク質の比屈折率増分)

alpha_ri = 0.0018  # ml/mg

# 質量濃度マップ
concentration_map = (ri_map - n_medium) / alpha_ri

# マスク内の平均
mask = thickness_map > 0
mean_concentration = np.mean(concentration_map[mask])

print(f"Mean concentration: {mean_concentration:.2f} mg/ml")

# 総質量
# 体積を計算
voxel_volume = (voxel_xy ** 2) * voxel_z  # um^3
cell_volume_um3 = np.sum(thickness_map * voxel_xy * voxel_xy * voxel_z)
cell_volume_ml = cell_volume_um3 * 1e-15  # um^3 → ml

# 総質量
total_mass_mg = mean_concentration * cell_volume_ml
total_mass_pg = total_mass_mg * 1e9  # mg → pg

print(f"Cell volume: {cell_volume_um3:.2f} um^3")
print(f"Total mass: {total_mass_pg:.2f} pg")
```

---

## ⚙️ パラメータ調整

### voxel_z の設定

**重要**: `voxel_z`は厚みの分解能を決定します。

#### ケース1: XYと同じ（等方的）
```python
voxel_xy = 0.08625
voxel_z = 0.08625  # XYと同じ
```
- **利点**: 等方的、計算が直感的
- **欠点**: Z方向のスライス数が多くなる

#### ケース2: Z方向を粗く
```python
voxel_xy = 0.08625
voxel_z = 0.3  # XYの約3.5倍
```
- **利点**: スライス数が少なく、高速
- **欠点**: Z方向の分解能が低い

#### 推奨設定

```python
# 細胞の厚み ~3-5 um
# 希望するスライス数 ~10-15枚

voxel_z = 細胞の厚み / 希望スライス数

例: 
voxel_z = 3.0 / 10 = 0.3 um/slice
```

---

## 🔧 トラブルシューティング

### 問題1: 厚みマップが保存されない

**原因**: `save_thickness_maps=False`

**解決策**:
```python
results_df = tracker.track_volume_timeseries(
    save_thickness_maps=True  # Trueにする
)
```

### 問題2: RI計算でエラー

**原因**: 位相差画像と厚みマップのサイズが合わない

**解決策**:
```python
# 画像サイズを確認
phase = tifffile.imread('phase_image.tif')
thickness = tifffile.imread('thickness_map.tif')

print(f"Phase: {phase.shape}")
print(f"Thickness: {thickness.shape}")

# 必要に応じてリサイズ
from skimage.transform import resize
thickness_resized = resize(thickness, phase.shape, order=1)
```

### 問題3: RIが異常値

**原因**: 位相差画像の単位が間違っている

**確認**:
```python
# 位相差画像はラジアン単位である必要がある
# 典型的な範囲: 0 〜 2π (0 〜 6.28)

phase = tifffile.imread('phase_image.tif')
print(f"Phase range: {np.min(phase):.2f} - {np.max(phase):.2f}")

# もし0-255などの場合、変換が必要:
phase_radians = phase * (2 * np.pi / 255)
```

---

## 📚 関連ファイル

| ファイル | 機能 |
|---------|------|
| `timeseries_volume_from_roiset.py` | メインスクリプト |
| `24_elip_volume.py` | 元の楕円体積計算（参考） |
| `28_batch_analysis.py` | バッチRI解析（参考） |
| `25_Roiset_from_zstack.py` | Z-stackからROI作成 |

---

## 🎓 まとめ

### キーポイント

✅ **ROIセット** → **厚みマップ**（各XY位置のZ占有数）  
✅ **厚みマップ** + **位相差画像** → **RI計算**  
✅ **Mean RI, Total RI** の時系列追跡  
✅ **batch_analysis.py互換** の出力形式  

### ワークフロー

```
ROIセット (.zip)
    ↓
3D再構成
    ↓
厚みマップ (thickness_map.tif)
    ↓ + 位相差画像
RI計算
    ↓
Mean RI, Total RI, RI map
```

### 次のステップ

1. **質量濃度**: RI → 濃度への変換
2. **総質量**: 体積 × 濃度
3. **分子数推定**: 質量 / 分子量

このツールで、**ROIセットから細胞の物理量（厚み、RI、質量）を完全に定量化**できます！

---

**作成日**: 2025-12-23  
**バージョン**: 1.0  
**連絡先**: QPI_omni Project

