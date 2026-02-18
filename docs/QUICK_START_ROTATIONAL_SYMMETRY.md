# 🚀 回転対称体積推定 - クイックスタートガイド

このガイドでは、Odermatt et al. (2021) の回転対称体積推定アルゴリズムを使って、ROIセットから体積・厚みマップ・RIを計算する方法を説明します。

---

## ⚡ 最速で実行

```bash
# 1. scriptsディレクトリに移動
cd scripts

# 2. スクリプトを実行（100フレーム、可視化あり）
python 31_roiset_rotational_volume.py
```

**実行時間**: 約2-3分（100フレーム）

---

## 📋 前提条件

### 必要なファイル

1. **ROIセット**: `scripts/RoiSet.zip`
   - ImageJで作成したROIの集合
   - ファイル名形式: `{frame}-{x}-{y}.roi`

2. **位相差画像**（オプション、RI計算に必要）:
   - `data/align_demo/bg_corr_aligned/aligned/*.tif`
   - フレーム番号がROIと一致している必要あり

### 必要なライブラリ

```python
import numpy
import pandas
import matplotlib
import scipy
import skimage
import tifffile
import cv2
```

確認コマンド:
```bash
python -c "import numpy, pandas, matplotlib, scipy, skimage, tifffile, cv2; print('OK')"
```

---

## 🎯 基本的な使い方

### 方法1: デフォルト設定で実行

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
    pixel_size_um=0.348,       # ピクセルサイズ
    section_interval_um=0.25,  # 250 nm
    max_iterations=3,          # 反復回数
    convergence_tolerance=0.5  # 収束閾値
)

# 解析実行
results_df = analyzer.analyze_timeseries(
    max_frames=100,            # 最初の100フレーム
    save_visualizations=True,  # 可視化を保存
    save_thickness_maps=True   # 厚みマップを保存
)

# 結果を保存
analyzer.save_results('rotational_volume_output')
analyzer.save_visualizations('rotational_volume_output', format='png')
analyzer.plot_results('rotational_volume_plot.png')
```

### 方法2: コマンドラインから実行

```bash
cd scripts
python 31_roiset_rotational_volume.py
```

スクリプト内で設定を変更する場合は、`main()`関数を編集：

```python
def main():
    analyzer = RotationalSymmetryROIAnalyzer(
        roi_zip_path=r"c:\Users\QPI\Documents\QPI_omni\scripts\RoiSet.zip",
        pixel_size_um=0.348,  # ← ここを変更
        section_interval_um=0.25,
        max_iterations=3,
        convergence_tolerance=0.5
    )
    
    results_df = analyzer.analyze_timeseries(
        max_frames=100,  # ← フレーム数を変更
        save_visualizations=True,
        save_thickness_maps=True
    )
    # ...
```

---

## 📂 出力ファイル

### 出力ディレクトリ構造

```
scripts/rotational_volume_output/
├── rotational_volume_timeseries.csv     # 体積データ
├── rotational_volume_summary.txt        # サマリー
├── thickness_stack_all_frames.tif       # 厚みマップスタック
├── thickness_maps/                      # 個別厚みマップ
│   └── *.tif (100ファイル)
└── visualizations/                      # 可視化
    └── *.png (100ファイル)
```

### すぐに確認できるファイル

1. **`rotational_volume_summary.txt`**: 統計サマリー
   ```bash
   cat rotational_volume_output/rotational_volume_summary.txt
   ```

2. **`rotational_volume_plot.png`**: 体積の時系列プロット
   - 画像ビューアーで開く

3. **`visualizations/*.png`**: 断面線・中心線の可視化
   - 各フレームの解析結果を確認

---

## 🔧 よく使う設定

### 設定1: 全フレームを解析

```python
results_df = analyzer.analyze_timeseries(
    max_frames=None,  # 全フレーム
    save_visualizations=False,  # メモリ節約
    save_thickness_maps=True
)
```

### 設定2: 高速モード（可視化なし）

```python
results_df = analyzer.analyze_timeseries(
    max_frames=100,
    save_visualizations=False,  # 可視化を無効化
    save_thickness_maps=True
)
```

### 設定3: メモリ節約モード

```python
# 少ないフレーム数で実行
results_df = analyzer.analyze_timeseries(
    max_frames=50,  # 50フレームのみ
    save_visualizations=False,
    save_thickness_maps=False  # 厚みマップも無効化
)
```

### 設定4: カスタムパラメータ

```python
analyzer = RotationalSymmetryROIAnalyzer(
    roi_zip_path="RoiSet.zip",
    pixel_size_um=0.08625,     # 異なるピクセルサイズ
    section_interval_um=0.5,    # 大きい間隔（高速化）
    max_iterations=5,           # より多くの反復
    convergence_tolerance=1.0   # 緩い収束条件
)
```

---

## 📊 結果の簡単な確認

### Pythonで確認

```python
import pandas as pd
import matplotlib.pyplot as plt

# CSVを読み込み
df = pd.read_csv('rotational_volume_output/rotational_volume_timeseries.csv')

# 基本統計
print(df['volume_um3'].describe())

# 簡単なプロット
plt.plot(df['time_index'], df['volume_um3'], 'o-')
plt.xlabel('Time (frame)')
plt.ylabel('Volume (µm³)')
plt.title('Volume Time-series')
plt.show()
```

### ImageJで確認

```
# 厚みマップスタックを開く
File > Open > rotational_volume_output/thickness_stack_all_frames.tif

# スライダーで各フレームを確認
# Image > Adjust > Brightness/Contrast で表示を調整
```

---

## ⚙️ RI計算の追加（オプション）

位相差画像がある場合、RI（屈折率）を計算できます。

### ステップ1: 位相差画像の準備

位相差画像を以下のディレクトリに配置：
```
data/align_demo/bg_corr_aligned/aligned/
├── subtracted_by_maskmean_float320085_bg_corr_aligned.tif
├── subtracted_by_maskmean_float320086_bg_corr_aligned.tif
└── ...
```

### ステップ2: スクリプトのRI計算部分を有効化

`31_roiset_rotational_volume.py`の`main()`関数で、以下のコメントを外す：

```python
def main():
    # ... (解析実行) ...
    
    # RI計算（コメントを外す）
    phase_dir = os.path.join(os.path.dirname(__file__), "..", "data", 
                            "align_demo", "bg_corr_aligned", "aligned")
    phase_dir = os.path.abspath(phase_dir)
    
    if os.path.exists(phase_dir):
        analyzer.compute_ri_from_phase_images(
            phase_dir, 
            wavelength_nm=663,    # 波長
            n_medium=1.333        # 培地の屈折率
        )
        analyzer.save_ri_results('rotational_volume_output')
```

### ステップ3: 実行

```bash
python 31_roiset_rotational_volume.py
```

### ステップ4: RI結果の確認

```python
import pandas as pd
ri_df = pd.read_csv('rotational_volume_output/ri_statistics.csv')
print(ri_df[['time_index', 'mean_ri', 'total_ri']].head())
```

---

## 🐛 トラブルシューティング

### 問題1: ROIが読み込めない

**エラー**:
```
Successfully parsed: 0 ROIs
```

**解決**:
```python
# ROIファイルを確認
import zipfile
with zipfile.ZipFile('RoiSet.zip', 'r') as zf:
    print(zf.namelist()[:5])  # 最初の5個を表示
```

### 問題2: メモリ不足

**エラー**:
```
MemoryError
```

**解決**:
```python
# フレーム数を減らす
results_df = analyzer.analyze_timeseries(
    max_frames=50,  # 100 → 50
    save_visualizations=False  # 可視化を無効化
)
```

### 問題3: 位相差画像が見つからない

**エラー**:
```
Warning: Phase image directory not found
```

**解決**:
```python
# パスを確認
import os
phase_dir = r"c:\Users\QPI\Documents\QPI_omni\data\align_demo\bg_corr_aligned\aligned"
print(f"Exists: {os.path.exists(phase_dir)}")
print(f"Files: {len(os.listdir(phase_dir)) if os.path.exists(phase_dir) else 0}")
```

### 問題4: 収束しない

**症状**: すべての反復が実行される

**解決**:
```python
# 収束閾値を緩める
analyzer = RotationalSymmetryROIAnalyzer(
    ...
    convergence_tolerance=1.0  # 0.5 → 1.0
)
```

---

## 📚 次のステップ

### 詳細な解析

より詳細な解析方法は以下を参照：
- `docs/workflows/rotational_symmetry_volume_workflow.md`: 完全なワークフロー
- `scripts/rotational_volume_output/README.md`: 出力ファイルの詳細

### 他の手法との比較

```python
# Pomegranate法と比較
import pandas as pd

pomegranate_df = pd.read_csv('timeseries_volume_output/volume_timeseries.csv')
rotational_df = pd.read_csv('rotational_volume_output/rotational_volume_timeseries.csv')

# マージして比較
merged = pd.merge(pomegranate_df, rotational_df, 
                 on='time_point', suffixes=('_pom', '_rot'))

# 相関
print(merged[['volume_um3_pom', 'volume_um3_rot']].corr())
```

### カスタマイズ

アルゴリズムのカスタマイズ方法：
1. **セクション間隔の変更**: `section_interval_um` を調整
2. **反復回数の変更**: `max_iterations` を調整
3. **収束条件の変更**: `convergence_tolerance` を調整
4. **出力形式の変更**: `format='tiff'` でTIFF形式の可視化

---

## 💡 ヒント

### ヒント1: 最初は少ないフレームで試す

```python
# 最初は10フレームで試して、問題がないか確認
results_df = analyzer.analyze_timeseries(max_frames=10)
```

### ヒント2: 可視化で結果を確認

可視化画像で以下を確認：
- ✅ 中心線が細胞の中央を通っているか
- ✅ 断面線が中心線に垂直か
- ✅ 回転対称円が細胞に適合しているか

### ヒント3: 厚みマップの妥当性チェック

```python
import tifffile
import numpy as np

stack = tifffile.imread('rotational_volume_output/thickness_stack_all_frames.tif')

# 統計を確認
print(f"Min: {stack.min():.1f} pixels")
print(f"Max: {stack.max():.1f} pixels")
print(f"Mean: {stack.mean():.1f} pixels")

# 妥当な範囲か確認（通常5-30ピクセル程度）
```

### ヒント4: バッチ処理

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

---

## 📞 サポート

### 詳細ドキュメント
- **完全なワークフロー**: `docs/workflows/rotational_symmetry_volume_workflow.md`
- **出力ファイルの説明**: `scripts/rotational_volume_output/README.md`
- **Pomegranate法**: `docs/workflows/pomegranate_reconstruction_summary.md`

### 参考論文
Odermatt et al. (2021). "Variations of intracellular density during the cell cycle arise from tip-growth regulation in fission yeast." eLife 10:e64901. 
https://doi.org/10.7554/eLife.64901

---

**作成日**: 2024年12月24日  
**QPI_omni プロジェクト**  
**バージョン**: 1.0
