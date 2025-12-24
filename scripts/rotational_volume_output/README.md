# 回転対称体積推定 - 実行結果

## 📋 概要

このディレクトリには、Odermatt et al. (2021) eLife の回転対称体積推定アルゴリズムを用いた解析結果が含まれています。

**アルゴリズム**: 反復的中心線・断面線更新（最大3回、収束閾値0.5ピクセル）  
**セクション間隔**: 0.25 µm (論文準拠)  
**解析細胞数**: 100個  
**参考論文**: [Odermatt et al. (2021) eLife 10:e64901](https://elifesciences.org/articles/64901)

---

## 📂 ファイル構造

```
rotational_volume_output/
├── rotational_volume_timeseries.csv          # 体積・表面積データ
├── rotational_volume_summary.txt             # 統計サマリー
├── thickness_stack_all_frames.tif            # 全フレームの厚みマップ (100, 512, 512)
├── thickness_maps/                           # 個別の厚みマップ (100ファイル)
├── visualizations/                           # 中心線・断面線の可視化 (100ファイル)
├── ri_statistics.csv                         # RI統計（オプション）
├── ri_summary.txt                            # RIサマリー（オプション）
└── ri_maps/                                  # RIマップ（オプション）
```

---

## 📊 解析結果サマリー

### 体積統計

| 統計量 | 値 (µm³) |
|--------|----------|
| 平均   | 125.51   |
| 中央値 | 122.51   |
| 標準偏差 | 28.95   |
| 最小値 | 86.08    |
| 最大値 | 275.31   |

### 表面積統計

| 統計量 | 値 (µm²) |
|--------|----------|
| 平均   | 125.97   |
| 標準偏差 | 23.07   |

### アルゴリズム設定

- **最大反復回数**: 3回
- **収束閾値**: 0.5ピクセル
- **セクション間隔**: 0.25 µm (250 nm)
- **ピクセルサイズ**: 0.348 µm

---

## 🔍 データファイルの説明

### 1. rotational_volume_timeseries.csv

時系列の体積・表面積データ

**カラム**:
- `volume_um3`: 体積 (µm³)
- `surface_area_um2`: 表面積 (µm²)
- `n_sections`: 断面数
- `mean_radius_um`: 平均半径 (µm)
- `max_radius_um`: 最大半径 (µm)
- `length_um`: 細胞長 (µm)
- `area_2d`: 2D投影面積 (ピクセル²)
- `time_point`: フレーム番号
- `time_index`: 時系列インデックス
- `cell_index`: 細胞インデックス
- `roi_name`: ROIファイル名

**読み込み例**:
```python
import pandas as pd
df = pd.read_csv('rotational_volume_timeseries.csv')
print(df.head())
```

### 2. thickness_stack_all_frames.tif

全フレームの厚みマップを統合したTIFFスタック

**形状**: (100, 512, 512) = (フレーム数, 高さ, 幅)  
**データ型**: float32  
**単位**: ピクセル数（Z方向の占有スライス数）

**読み込み例**:
```python
import tifffile
import matplotlib.pyplot as plt

stack = tifffile.imread('thickness_stack_all_frames.tif')
print(f"Shape: {stack.shape}")

# 最初のフレームを表示
plt.imshow(stack[0], cmap='viridis')
plt.colorbar(label='Thickness (pixels)')
plt.title('Thickness Map - Frame 0')
plt.show()
```

**実際の厚み（µm）への変換**:
```python
pixel_size_um = 0.348
thickness_um = stack * pixel_size_um
```

### 3. thickness_maps/

個別のフレームごとの厚みマップ

**ファイル名**: `{frame}-{x}-{y}_thickness.tif`  
例: `0085-0024-0136_thickness.tif`

**読み込み例**:
```python
import tifffile
thickness = tifffile.imread('thickness_maps/0085-0024-0136_thickness.tif')
print(f"Max thickness: {thickness.max():.1f} pixels")
```

### 4. visualizations/

断面線と中心線の可視化画像

**ファイル名**: `{frame}-{x}-{y}_visualization.png`

**可視化内容**:
- 🔵 **輪郭** (青線): 細胞の境界
- 🔴 **長軸** (赤線): 初期の長軸（最小外接矩形）
- 🟢 **中心線** (緑線): 反復更新後の最終中心線
- 🔷 **断面線** (シアン線): 中心線に垂直な断面
- 🟡 **回転対称円** (黄色): 各断面での半径

**例**:
![Visualization Example](visualizations/0085-0024-0136_visualization.png)

### 5. ri_statistics.csv (オプション)

位相差画像から計算したRI統計

**カラム**:
- `time_index`: 時系列インデックス
- `time_point`: フレーム番号
- `frame_num`: 位相差画像のフレーム番号
- `roi_name`: ROIファイル名
- `mean_ri`: 平均屈折率
- `median_ri`: 中央値屈折率
- `std_ri`: 屈折率の標準偏差
- `total_ri`: 全RI（積分値）
- `n_pixels`: ピクセル数

**読み込み例**:
```python
import pandas as pd
ri_df = pd.read_csv('ri_statistics.csv')

# 時系列プロット
import matplotlib.pyplot as plt
plt.plot(ri_df['time_index'], ri_df['mean_ri'])
plt.xlabel('Time (frame)')
plt.ylabel('Mean RI')
plt.title('RI Time-series')
plt.show()
```

### 6. ri_maps/

個別のフレームごとのRIマップ

**ファイル名**: `{frame}-{x}-{y}_ri_map.tif`  
**データ型**: float32  
**単位**: 屈折率（無次元）

**読み込み例**:
```python
import tifffile
import matplotlib.pyplot as plt

ri_map = tifffile.imread('ri_maps/0085-0024-0136_ri_map.tif')

plt.imshow(ri_map, cmap='jet', vmin=1.33, vmax=1.40)
plt.colorbar(label='Refractive Index')
plt.title('RI Map')
plt.show()
```

---

## 🔬 データ解析の例

### 例1: 体積の時系列プロット

```python
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('rotational_volume_timeseries.csv')

# プロット
plt.figure(figsize=(12, 6))
plt.plot(df['time_index'], df['volume_um3'], 'o-', alpha=0.7)
plt.xlabel('Time (frame)', fontsize=12)
plt.ylabel('Volume (µm³)', fontsize=12)
plt.title('Cell Volume Time-series', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volume_timeseries.png', dpi=300)
plt.show()
```

### 例2: 体積分布のヒストグラム

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('rotational_volume_timeseries.csv')

plt.figure(figsize=(10, 6))
plt.hist(df['volume_um3'], bins=20, edgecolor='black', alpha=0.7)
plt.axvline(df['volume_um3'].mean(), color='r', linestyle='--', 
           linewidth=2, label=f'Mean = {df["volume_um3"].mean():.1f} µm³')
plt.axvline(df['volume_um3'].median(), color='g', linestyle='--', 
           linewidth=2, label=f'Median = {df["volume_um3"].median():.1f} µm³')
plt.xlabel('Volume (µm³)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Volume Distribution', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volume_histogram.png', dpi=300)
plt.show()
```

### 例3: 体積 vs 表面積の相関

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_csv('rotational_volume_timeseries.csv')

# 相関係数
corr, p_value = pearsonr(df['volume_um3'], df['surface_area_um2'])

plt.figure(figsize=(8, 8))
plt.scatter(df['volume_um3'], df['surface_area_um2'], alpha=0.5, s=50)
plt.xlabel('Volume (µm³)', fontsize=12)
plt.ylabel('Surface Area (µm²)', fontsize=12)
plt.title(f'Volume vs Surface Area\nr = {corr:.3f}, p = {p_value:.2e}', 
         fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('volume_vs_surface_area.png', dpi=300)
plt.show()
```

### 例4: 厚みマップの統計解析

```python
import tifffile
import numpy as np
import matplotlib.pyplot as plt

# スタック読み込み
stack = tifffile.imread('thickness_stack_all_frames.tif')
pixel_size_um = 0.348

# ピクセル → µm
stack_um = stack * pixel_size_um

# 各フレームの統計
mean_thickness = np.mean(stack_um, axis=(1, 2))
max_thickness = np.max(stack_um, axis=(1, 2))

# プロット
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

axes[0].plot(mean_thickness, 'o-', alpha=0.7)
axes[0].set_xlabel('Frame', fontsize=12)
axes[0].set_ylabel('Mean Thickness (µm)', fontsize=12)
axes[0].set_title('Mean Thickness Time-series', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

axes[1].plot(max_thickness, 'o-', alpha=0.7, color='orange')
axes[1].set_xlabel('Frame', fontsize=12)
axes[1].set_ylabel('Max Thickness (µm)', fontsize=12)
axes[1].set_title('Max Thickness Time-series', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('thickness_timeseries.png', dpi=300)
plt.show()
```

### 例5: RIの時系列解析（オプション）

```python
import pandas as pd
import matplotlib.pyplot as plt

# RI統計読み込み
ri_df = pd.read_csv('ri_statistics.csv')

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Mean RI
axes[0].plot(ri_df['time_index'], ri_df['mean_ri'], 'o-', alpha=0.7)
axes[0].axhline(y=1.333, color='r', linestyle='--', label='Medium RI')
axes[0].set_xlabel('Time (frame)', fontsize=12)
axes[0].set_ylabel('Mean RI', fontsize=12)
axes[0].set_title('Mean Refractive Index Time-series', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Total RI
axes[1].plot(ri_df['time_index'], ri_df['total_ri'], 'o-', alpha=0.7, color='green')
axes[1].set_xlabel('Time (frame)', fontsize=12)
axes[1].set_ylabel('Total RI', fontsize=12)
axes[1].set_title('Total RI Time-series (Dry Mass Proxy)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ri_timeseries.png', dpi=300)
plt.show()
```

---

## 🎨 可視化の見方

### 断面線・中心線の可視化画像

各可視化画像には以下の要素が含まれています：

1. **輪郭（青線）**: 細胞の境界線
   - `measure.find_contours()` で抽出

2. **長軸（赤線）**: 初期の長軸
   - 最小外接矩形の長辺
   - 反復更新の開始点

3. **中心線（緑線、点付き）**: 最終的な中心線
   - 反復更新後の結果
   - 各断面の中点を通る

4. **断面線（シアン線）**: 中心線に垂直な断面
   - 250 nm間隔
   - 中心線に垂直に更新される

5. **回転対称円（黄色、半透明）**: 各断面での半径
   - 断面線と輪郭の交点から計算
   - 回転対称を仮定した体積計算に使用

**良好な解析の指標**:
- ✅ 中心線が細胞の中央を通っている
- ✅ 断面線が中心線に垂直
- ✅ 回転対称円が細胞の形状に適合
- ✅ 長軸（赤）と中心線（緑）に大きな差がない

**問題がある場合の指標**:
- ❌ 中心線が細胞からはみ出している
- ❌ 断面線が不規則
- ❌ 回転対称円が細胞から大きくはみ出す

---

## 📝 メタデータ

### 実験条件

- **ピクセルサイズ**: 0.348 µm/pixel
- **画像サイズ**: 512 × 512 pixels
- **解析フレーム数**: 100 フレーム
- **フレーム範囲**: 85-184

### アルゴリズムパラメータ

- **セクション間隔**: 0.25 µm (250 nm)
- **最大反復回数**: 3回
- **収束閾値**: 0.5 pixels
- **体積計算**: 回転対称（各断面を円形と仮定）

### RI計算パラメータ（オプション）

- **波長**: 663 nm（赤色レーザー）
- **培地屈折率**: 1.333（水）
- **計算式**: RI = n_medium + (φ × λ) / (2π × thickness)

---

## 🔗 関連ファイル

### スクリプト
- `../31_roiset_rotational_volume.py`: メイン解析スクリプト
- `../30_demo_rotational_symmetry_volume.py`: デモスクリプト

### ドキュメント
- `../../docs/workflows/rotational_symmetry_volume_workflow.md`: 詳細なワークフロー
- `../../docs/workflows/thickness_map_and_ri_calculation.md`: 厚みマップとRI計算
- `../../docs/workflows/pomegranate_reconstruction_summary.md`: Pomegranate法との比較

---

## 📖 引用

このアルゴリズムを使用した場合は、以下の論文を引用してください：

```
Odermatt, P. D., Miettinen, T. P., Lemière, J., Kang, J. H., Bostan, E., 
Manalis, S. R., ... & Chang, F. (2021). Variations of intracellular density 
during the cell cycle arise from tip-growth regulation in fission yeast. 
eLife, 10, e64901. https://doi.org/10.7554/eLife.64901
```

---

**生成日**: 2024年12月24日  
**スクリプトバージョン**: 1.0  
**QPI_omni プロジェクト**

