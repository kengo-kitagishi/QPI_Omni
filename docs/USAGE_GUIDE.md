# QPI解析 使い方ガイド

このドキュメントは、QPI解析システムの使い方、クイックスタート、トラブルシューティングをまとめたものです。

---

## 目次

1. [クイックスタート](#1-クイックスタート)
2. [手法別の使い方](#2-手法別の使い方)
3. [パラメータリファレンス](#3-パラメータリファレンス)
4. [よく使うコマンド](#4-よく使うコマンド)
5. [トラブルシューティング](#5-トラブルシューティング)
6. [出力ファイルの見方](#6-出力ファイルの見方)

---

## 1. クイックスタート

### 1.1 基本的なワークフロー

```bash
# 1. scriptsディレクトリに移動
cd c:\Users\QPI\Documents\QPI_omni\scripts

# 2. 単一条件で実行
python 24_ellipse_volume.py

# 3. バッチ実行（12条件）
python 27_compare_volume_estimation_methods.py

# 4. 時系列プロット
python 30_plot_filtered_conditions.py

# 5. Pomegranate 3D再構成
python 29_Pomegranate_from_roiset.py

# 6. 回転対称体積推定
python 31_roiset_rotational_volume.py
```

### 1.2 最速で結果を得る

**楕円近似（最速）**:
```bash
cd scripts
python 24_ellipse_volume.py
```

実行時間: 約1-2分

**回転対称（最高精度）**:
```bash
cd scripts
python 31_roiset_rotational_volume.py
```

実行時間: 約2-3分（100フレーム）

---

## 2. 手法別の使い方

### 2.1 楕円・Feret径近似（24_ellipse_volume.py）

#### 基本的な使い方

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "ellipse_volume", Path("24_ellipse_volume.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TimeSeriesDensityMapper = module.TimeSeriesDensityMapper

# アナライザーを作成
mapper = TimeSeriesDensityMapper(
    results_csv="Results.csv",
    image_directory="path/to/phase_images",
    shape_type='ellipse',      # 'ellipse' or 'feret'
    subpixel_sampling=5        # 1, 5, 10
)

# 解析実行
mapper.process_all_rois(max_rois=None)
```

#### コマンドライン実行

```bash
# 楕円近似、サブピクセル5
python 24_ellipse_volume.py

# スクリプト内でパラメータを変更:
# SHAPE_TYPE = 'feret'
# SUBPIXEL_SAMPLING = 10
```

#### パラメータの選択

| 用途 | shape_type | subpixel_sampling |
|------|-----------|-------------------|
| 高速処理 | ellipse | 1 |
| バランス（推奨） | ellipse | 5 |
| 細長い細胞 | feret | 5 |
| 最高精度 | feret | 10 |

#### 出力ディレクトリ

```
scripts/
└── timeseries_density_output_{shape}_{subpixel}/
    ├── density_tiff/           # 濃度マップ
    ├── visualizations/         # 可視化
    ├── csv_data/               # 個別CSV
    └── all_rois_summary.csv    # サマリー
```

---

### 2.2 Pomegranate 3D再構成

#### 基本的な使い方

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "pomegranate_volume", Path("29_Pomegranate_from_roiset.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TimeSeriesVolumeTracker = module.TimeSeriesVolumeTracker

# Trackerを作成
tracker = TimeSeriesVolumeTracker(
    roi_zip_path="RoiSet.zip",
    voxel_xy=0.08625,  # XYピクセルサイズ (µm)
    voxel_z=0.3,       # Z方向ステップ (µm)
    radius_enlarge=1.0,
    image_width=512,
    image_height=512
)

# 体積を追跡
results_df = tracker.track_volume_timeseries(
    max_frames=None,            # 全フレーム
    save_thickness_maps=True    # 厚みマップを保存
)

# プロット
tracker.plot_volume_timeseries('volume_plot.png')

# 保存
tracker.save_results('timeseries_volume_output')
```

#### RI計算（オプション）

```python
# 位相差画像からRI計算
ri_results = tracker.compute_ri_from_phase_images(
    phase_image_dir='path/to/phase_images/',
    wavelength_nm=663,
    n_medium=1.333
)

# RI結果を保存
tracker.save_ri_results('timeseries_volume_output')
```

#### コマンドライン実行

```bash
cd scripts
python 29_Pomegranate_from_roiset.py
```

#### 出力ディレクトリ

```
timeseries_volume_output/
├── volume_timeseries.csv           # 体積データ
├── volume_summary.txt              # サマリー
├── volume_plot.png                 # プロット
├── thickness_maps/                 # 個別厚みマップ
│   └── *.tif
├── thickness_stack_all_frames.tif  # 統合スタック
├── ri_statistics.csv               # RI統計（オプション）
└── ri_maps/                        # RIマップ（オプション）
    └── *.tif
```

---

### 2.3 回転対称体積推定（31_roiset_rotational_volume.py）

#### 基本的な使い方

```python
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "rotational_volume", Path("31_roiset_rotational_volume.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
RotationalSymmetryROIAnalyzer = module.RotationalSymmetryROIAnalyzer

# アナライザーを作成
analyzer = RotationalSymmetryROIAnalyzer(
    roi_zip_path="RoiSet.zip",
    pixel_size_um=0.348,
    section_interval_um=0.25,  # 250 nm
    max_iterations=3,
    convergence_tolerance=0.5
)

# 解析実行
results_df = analyzer.analyze_timeseries(
    max_frames=100,             # 最初の100フレーム
    save_visualizations=True,   # 可視化を保存
    save_thickness_maps=True    # 厚みマップを保存
)

# 結果を保存
analyzer.save_results('rotational_volume_output')
analyzer.save_visualizations('rotational_volume_output', format='png')
analyzer.plot_results('rotational_volume_plot.png')
```

#### RI計算（オプション）

```python
# 位相差画像からRI計算
analyzer.compute_ri_from_phase_images(
    phase_image_dir,
    wavelength_nm=663,
    n_medium=1.333
)
analyzer.save_ri_results('rotational_volume_output')
```

#### コマンドライン実行

```bash
cd scripts
python 31_roiset_rotational_volume.py
```

#### よく使う設定

**高速モード（可視化なし）**:
```python
results_df = analyzer.analyze_timeseries(
    max_frames=100,
    save_visualizations=False,  # 可視化を無効化
    save_thickness_maps=True
)
```

**全フレーム処理**:
```python
results_df = analyzer.analyze_timeseries(
    max_frames=None,  # 全フレーム
    save_visualizations=False,
    save_thickness_maps=True
)
```

**カスタムパラメータ**:
```python
analyzer = RotationalSymmetryROIAnalyzer(
    roi_zip_path="RoiSet.zip",
    pixel_size_um=0.08625,      # 異なるピクセルサイズ
    section_interval_um=0.5,    # 大きい間隔（高速化）
    max_iterations=5,           # より多くの反復
    convergence_tolerance=1.0   # 緩い収束条件
)
```

#### 出力ディレクトリ

```
rotational_volume_output/
├── rotational_volume_timeseries.csv     # 体積データ
├── rotational_volume_summary.txt        # サマリー
├── rotational_volume_plot.png           # プロット
├── thickness_stack_all_frames.tif       # 厚みマップスタック
├── thickness_maps/                      # 個別厚みマップ
│   └── *.tif
├── visualizations/                      # 可視化
│   └── *.png
├── ri_statistics.csv                    # RI統計（オプション）
└── ri_maps/                             # RIマップ（オプション）
    └── *.tif
```

---

### 2.4 バッチ解析（27_compare_volume_estimation_methods.py）

#### 基本的な使い方

複数のパラメータ組み合わせで自動実行：

```python
# 実行条件
csv_files = [
    r"c:\...\Results_enlarge.csv",
    r"c:\...\Results_enlarge_interpolate.csv"
]

shape_types = ['ellipse', 'feret']
subpixel_samplings = [1, 5, 10]

# 全組み合わせで実行
# 2 CSVs × 2 shapes × 3 subpixels = 12条件
```

#### コマンドライン実行

```bash
cd scripts
python 27_compare_volume_estimation_methods.py
```

実行時間: 約1-2時間（12条件、各条件約5-10分）

#### 実行条件

| CSV | Shape | Subpixel | 出力フォルダ |
|-----|-------|----------|--------------|
| enlarge | ellipse | 1 | `timeseries_density_output_enlarge_ellipse_subpixel1` |
| enlarge | ellipse | 5 | `timeseries_density_output_enlarge_ellipse_subpixel5` |
| enlarge | ellipse | 10 | `timeseries_density_output_enlarge_ellipse_subpixel10` |
| enlarge | feret | 1 | `timeseries_density_output_enlarge_feret_subpixel1` |
| enlarge | feret | 5 | `timeseries_density_output_enlarge_feret_subpixel5` |
| enlarge | feret | 10 | `timeseries_density_output_enlarge_feret_subpixel10` |
| ... | ... | ... | ... |

---

### 2.5 ImageJ連携

#### ROI前処理（スムージング）

```imagej
// ImageJマクロ
// Gap Closure & Smoothing
gap = 2;
run("Enlarge...", "enlarge=" + gap + " pixel");
run("Enlarge...", "enlarge=-" + gap + " pixel");

// ROIを保存
roiManager("Save", "RoiSet_smoothed.zip");
```

効果：
- 小さな穴や凹みを埋める
- 滑らかな輪郭

#### 2D→3D再構成（Pomegranate）

```imagej
// マクロを実行
run("2D_to_3D_reconstruction.ijm");

// パラメータ入力
voxelXY = 0.1;  // um/pixel
voxelZ = 0.3;   // um/slice
radiusEnlarge = 1.0;
```

---

## 3. パラメータリファレンス

### 3.1 実験パラメータ

| パラメータ | 記号 | デフォルト値 | 単位 | 説明 |
|-----------|------|-------------|------|------|
| レーザー波長 | wavelength_nm | 663 | nm | オフアクシスホログラフィー用光源 |
| 培地屈折率 | n_medium | 1.333 | - | 培地（主に水）の屈折率 |
| ピクセルサイズ | pixel_size_um | 0.348 | µm | 再構成画像のピクセルサイズ |
| 比屈折率増分 | alpha_ri | 0.00018 | ml/mg | タンパク質の標準値 |

### 3.2 解析パラメータ

| パラメータ | 選択肢 | 説明 | 推奨値 |
|-----------|--------|------|--------|
| shape_type | 'ellipse' / 'feret' | 形状近似方法 | 'ellipse' |
| subpixel_sampling | 1 / 5 / 10 | サブピクセル分割数 | 5 |
| max_rois | int / None | 処理するROI数 | None（全ROI） |

### 3.3 Pomegranateパラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| voxel_xy | 0.08625 | XYピクセルサイズ (µm) |
| voxel_z | 0.3 | Z方向ステップ (µm) |
| radius_enlarge | 1.0 | 半径拡張パラメータ |

### 3.4 回転対称パラメータ

| パラメータ | デフォルト値 | 説明 | 推奨範囲 |
|-----------|-------------|------|---------|
| pixel_size_um | 0.348 | ピクセルサイズ (µm) | 0.05-1.0 |
| section_interval_um | 0.25 | 断面間隔 (µm) | 0.1-0.5 |
| max_iterations | 3 | 最大反復回数 | 2-5 |
| convergence_tolerance | 0.5 | 収束閾値 (pixels) | 0.1-2.0 |

---

## 4. よく使うコマンド

### 4.1 実行コマンド

```bash
# 単体実行（楕円近似）
cd scripts
python 24_ellipse_volume.py

# バッチ実行
python 27_compare_volume_estimation_methods.py

# Pomegranate
python 29_Pomegranate_from_roiset.py

# 回転対称
python 31_roiset_rotational_volume.py

# 時系列プロット
python 30_plot_filtered_conditions.py
```

### 4.2 ファイル検索

```bash
# 特定ROIの可視化を探す
ls timeseries_density_output_*/visualizations/ROI_0000*

# 時系列プロットを探す
ls timeseries_plots_*/timeseries_volume_ri_mass.png

# 体積データを探す
ls */volume_timeseries.csv
ls */rotational_volume_timeseries.csv
```

### 4.3 データ確認（Python）

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVを読み込み
df = pd.read_csv('timeseries_volume_output/volume_timeseries.csv')

# 基本統計
print(df['volume_um3'].describe())

# プロット
plt.plot(df['time_index'], df['volume_um3'], 'o-')
plt.xlabel('Time (frame)')
plt.ylabel('Volume (µm³)')
plt.show()
```

### 4.4 データ確認（ImageJ）

```imagej
// 厚みマップスタックを開く
File > Open > thickness_stack_all_frames.tif

// スライダーで各フレームを確認
// Image > Adjust > Brightness/Contrast で表示を調整
```

---

## 5. トラブルシューティング

### 5.1 実行が遅い

**症状**: 処理に時間がかかりすぎる

**解決策**:
```python
# 1. サブピクセルサンプリングを減らす
SUBPIXEL_SAMPLING = 1  # 5 → 1

# 2. 処理するROI数を制限
MAX_ROIS = 10  # テスト用

# 3. 可視化を無効化
save_visualizations=False
```

### 5.2 メモリ不足

**症状**: `MemoryError` が発生

**解決策**:
```python
# 1. フレーム数を減らす
max_frames=50  # 100 → 50

# 2. 可視化を無効化
save_visualizations=False

# 3. 厚みマップを無効化
save_thickness_maps=False

# 4. 分割実行
# 0-100フレーム、100-200フレームなど
```

### 5.3 ROIが読み込めない

**症状**: `Successfully parsed: 0 ROIs`

**解決策**:
```python
# ROIファイルを確認
import zipfile
with zipfile.ZipFile('RoiSet.zip', 'r') as zf:
    print(zf.namelist()[:5])  # 最初の5個を表示

# ROIファイル形式を確認
# 対応形式: Polygon, Rectangle, Oval, Freehand, Traced
```

### 5.4 位相差画像が見つからない

**症状**: `Warning: Phase image directory not found`

**解決策**:
```python
# パスを確認
import os
phase_dir = r"c:\Users\QPI\Documents\QPI_omni\data\align_demo\bg_corr_aligned\aligned"
print(f"Exists: {os.path.exists(phase_dir)}")

# ファイル数を確認
if os.path.exists(phase_dir):
    print(f"Files: {len(os.listdir(phase_dir))}")
```

### 5.5 収束しない（回転対称）

**症状**: すべての反復が実行される

**解決策**:
```python
# 収束閾値を緩める
analyzer = RotationalSymmetryROIAnalyzer(
    ...
    convergence_tolerance=1.0  # 0.5 → 1.0
)

# または反復回数を増やす
max_iterations=5  # 3 → 5
```

### 5.6 マージ失敗（バッチ解析）

**症状**: CSVファイルのマージでエラー

**解決策**:
```python
# roi_indexカラムを確認
df = pd.read_csv('Results.csv')
print(df['roi_index'].head())

# 重複を確認
print(df['roi_index'].duplicated().sum())
```

### 5.7 体積が異常値

**症状**: 体積が負の値、または異常に大きい/小さい

**チェック項目**:
```python
# 1. ピクセルサイズを確認
PIXEL_SIZE_UM = 0.348  # 正しい値か？

# 2. ROIパラメータを確認
print(f"Major: {major}, Minor: {minor}")

# 3. 厚みマップを確認
import tifffile
thickness = tifffile.imread('thickness_maps/0085_thickness.tif')
print(f"Min: {thickness.min()}, Max: {thickness.max()}")

# 4. 可視化で確認
# visualizations/*.png を開いて視覚的に確認
```

---

## 6. 出力ファイルの見方

### 6.1 体積データ（CSV）

**ファイル**: `volume_timeseries.csv`, `rotational_volume_timeseries.csv`

主要カラム：

| カラム | 説明 | 単位 |
|--------|------|------|
| `time_index` | 時間インデックス（0始まり） | - |
| `time_point` | フレーム番号 | - |
| `volume_um3` | 体積 | µm³ |
| `area_2d` | 2D面積 | pixels |
| `max_radius` | 最大半径 | pixels |

**Pythonで確認**:
```python
import pandas as pd
df = pd.read_csv('volume_timeseries.csv')

# 基本統計
print(df['volume_um3'].describe())

# 時系列プロット
import matplotlib.pyplot as plt
plt.plot(df['time_index'], df['volume_um3'], 'o-')
plt.show()
```

### 6.2 厚みマップ（TIFF）

**ファイル**: `thickness_maps/*.tif`, `thickness_stack_all_frames.tif`

各ピクセルの値 = Z方向のスライス数（float）

**ImageJで確認**:
```imagej
File > Open > thickness_stack_all_frames.tif
Image > Adjust > Brightness/Contrast
```

**Pythonで確認**:
```python
import tifffile
import numpy as np

stack = tifffile.imread('thickness_stack_all_frames.tif')

# 統計
print(f"Shape: {stack.shape}")  # (T, Y, X)
print(f"Min: {stack.min():.1f} slices")
print(f"Max: {stack.max():.1f} slices")
print(f"Mean: {stack.mean():.1f} slices")
```

### 6.3 RI統計（CSV）

**ファイル**: `ri_statistics.csv`

主要カラム：

| カラム | 説明 | 単位 |
|--------|------|------|
| `time_index` | 時間インデックス | - |
| `mean_ri` | 平均RI | - |
| `total_ri` | 総RI（積分） | - |
| `min_ri` | 最小RI | - |
| `max_ri` | 最大RI | - |

**典型的な値**:
- 培地RI: 1.333
- 細胞内平均RI: 1.35 - 1.39
- タンパク質濃度: 100 - 400 mg/ml

### 6.4 サマリーファイル（TXT）

**ファイル**: `volume_summary.txt`, `rotational_volume_summary.txt`

内容：
- 処理したROI数
- 体積の統計（平均、中央値、標準偏差、範囲）
- 実行時間
- パラメータ

**例**:
```
Rotational Symmetry Volume Analysis Summary
============================================
Total ROIs processed: 100

Volume Statistics:
  Mean: 125.51 µm³
  Median: 120.34 µm³
  Std Dev: 28.95 µm³
  Min: 86.08 µm³
  Max: 275.31 µm³
```

### 6.5 可視化（PNG）

**ファイル**: `visualizations/*.png`, `volume_plot.png`

**個別フレームの可視化**（回転対称）:
- 🔵 輪郭（青線）
- 🔴 長軸（赤線）
- 🟢 中心線（緑線）
- 🔷 断面線（シアン線）
- 🟡 回転対称円（黄色）

**時系列プロット**:
- Volume vs Time
- Mean RI vs Time
- Total Mass vs Time
- 体積分布の変化

---

## 7. データ解析の例

### 7.1 複数手法の比較

```python
import pandas as pd
import matplotlib.pyplot as plt

# データを読み込み
ellipse_df = pd.read_csv('timeseries_density_output_ellipse_subpixel5/all_rois_summary.csv')
feret_df = pd.read_csv('timeseries_density_output_feret_subpixel5/all_rois_summary.csv')
pomegranate_df = pd.read_csv('timeseries_volume_output/volume_timeseries.csv')
rotational_df = pd.read_csv('rotational_volume_output/rotational_volume_timeseries.csv')

# プロット
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ellipse_df['time_index'], ellipse_df['volume_um3'], 'o-', label='Ellipse')
ax.plot(feret_df['time_index'], feret_df['volume_um3'], 's-', label='Feret')
ax.plot(pomegranate_df['time_index'], pomegranate_df['volume_um3'], '^-', label='Pomegranate')
ax.plot(rotational_df['time_index'], rotational_df['volume_um3'], 'd-', label='Rotational')
ax.set_xlabel('Time (frame)')
ax.set_ylabel('Volume (µm³)')
ax.legend()
plt.tight_layout()
plt.savefig('method_comparison.png', dpi=300)
plt.show()
```

### 7.2 相関分析

```python
import pandas as pd
import numpy as np

# データをマージ
df1 = pd.read_csv('timeseries_volume_output/volume_timeseries.csv')
df2 = pd.read_csv('rotational_volume_output/rotational_volume_timeseries.csv')

merged = pd.merge(df1, df2, on='time_point', suffixes=('_pom', '_rot'))

# 相関
corr = merged[['volume_um3_pom', 'volume_um3_rot']].corr()
print(corr)

# 散布図
import matplotlib.pyplot as plt
plt.scatter(merged['volume_um3_pom'], merged['volume_um3_rot'], alpha=0.5)
plt.xlabel('Pomegranate Volume (µm³)')
plt.ylabel('Rotational Volume (µm³)')
plt.plot([50, 300], [50, 300], 'r--', label='y=x')
plt.legend()
plt.show()
```

### 7.3 時系列統計

```python
import pandas as pd
import numpy as np

df = pd.read_csv('volume_timeseries.csv')

# 時間窓での統計
window_size = 10
df['volume_rolling_mean'] = df['volume_um3'].rolling(window=window_size).mean()
df['volume_rolling_std'] = df['volume_um3'].rolling(window=window_size).std()

# プロット
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(df['time_index'], df['volume_um3'], 'o', alpha=0.3, label='Raw')
ax.plot(df['time_index'], df['volume_rolling_mean'], '-', linewidth=2, label=f'Rolling Mean (n={window_size})')
ax.fill_between(df['time_index'], 
                df['volume_rolling_mean'] - df['volume_rolling_std'],
                df['volume_rolling_mean'] + df['volume_rolling_std'],
                alpha=0.2, label='±1 SD')
ax.set_xlabel('Time (frame)')
ax.set_ylabel('Volume (µm³)')
ax.legend()
plt.show()
```

---

## 8. ヒント・ベストプラクティス

### 8.1 最初は少ないフレームで試す

```python
# 最初は10フレームで試して、問題がないか確認
results_df = analyzer.analyze_timeseries(max_frames=10)
```

### 8.2 可視化で結果を確認

可視化画像で以下を確認：
- ✅ 中心線が細胞の中央を通っているか
- ✅ 断面線が中心線に垂直か
- ✅ 回転対称円が細胞に適合しているか

### 8.3 厚みマップの妥当性チェック

```python
import tifffile
import numpy as np

stack = tifffile.imread('thickness_stack_all_frames.tif')

# 統計を確認
print(f"Min: {stack.min():.1f} slices")
print(f"Max: {stack.max():.1f} slices")
print(f"Mean: {stack.mean():.1f} slices")

# 妥当な範囲か確認（通常5-30スライス程度）
```

### 8.4 バッチ処理

複数のROIセットを処理：
```python
roi_sets = ['RoiSet1.zip', 'RoiSet2.zip', 'RoiSet3.zip']

for roi_set in roi_sets:
    analyzer = RotationalSymmetryROIAnalyzer(roi_zip_path=roi_set, ...)
    results_df = analyzer.analyze_timeseries(...)
    
    # 出力ディレクトリを分ける
    output_dir = f'output_{roi_set.replace(".zip", "")}'
    analyzer.save_results(output_dir)
```

### 8.5 結果のバックアップ

```bash
# 重要な結果をバックアップ
mkdir backup_2025-12-24
cp -r *_output/ backup_2025-12-24/
```

---

**最終更新**: 2025-12-24  
**プロジェクト**: QPI_omni  
**著者**: AI Assistant
