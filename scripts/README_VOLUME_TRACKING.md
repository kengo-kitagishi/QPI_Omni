# Time-series Volume Tracking - Quick Start

## 🎯 目的

**ImageJ ROIセット（.zip）から細胞の3D体積を推定し、時系列の変化を追跡**

---

## 📦 ファイル構成

```
scripts/
├── timeseries_volume_from_roiset.py  # メインスクリプト
├── 2D_to_3D_reconstruction.ijm       # ImageJマクロ版
├── 2D_to_3D_reconstruction_analysis.py  # 基本アルゴリズム
├── RoiSet.zip                         # あなたのROIセット
└── demo_output/                       # デモ実行結果
    └── 3D_reconstruction.tif          # 3D z-stack例

docs/
├── workflows/
│   ├── timeseries_volume_tracking_guide.md  # 詳細ガイド
│   └── pomegranate_reconstruction_summary.md  # アルゴリズム解説
└── notes/
    └── pomegranate_algorithm_theory.md  # 理論の詳細
```

---

## 🚀 クイックスタート

### ステップ1: スクリプトを実行

```bash
cd c:\Users\QPI\Documents\QPI_omni\scripts
python timeseries_volume_from_roiset.py
```

### ステップ2: 結果を確認

```
timeseries_volume_output/
├── volume_timeseries.csv      # 全データ
├── volume_summary.txt         # 統計サマリー
└── timeseries_volume_plot.png # グラフ
```

---

## 📊 出力例

### CSVデータ
```csv
roi_name,area_2d,max_radius,z_slices,total_voxels,volume_um3,time_point,time_index,cell_index
0085-0024-0136.roi,270,6.0,8,901,2.01,85,0,0
0086-0024-0136.roi,262,6.0,8,901,2.01,86,1,0
0087-0024-0136.roi,284,6.0,8,931,2.08,87,2,0
```

### 体積時系列
```
Frame 85: 2.01 um^3
Frame 86: 2.01 um^3
Frame 87: 2.08 um^3  ← +3.5% 増加
Frame 88: 2.08 um^3
Frame 89: 2.02 um^3
```

---

## 🔧 カスタマイズ

### あなたのROIセットで実行

```python
from timeseries_volume_from_roiset import TimeSeriesVolumeTracker

# 1. Trackerを作成
tracker = TimeSeriesVolumeTracker(
    roi_zip_path="あなたの RoiSet.zip",
    voxel_xy=0.08625,    # あなたのピクセルサイズに変更
    voxel_z=0.3,         # あなたのZ間隔に変更
    image_width=512,     # あなたの画像サイズに変更
    image_height=512
)

# 2. 体積を追跡
results_df = tracker.track_volume_timeseries()

# 3. プロット
tracker.plot_volume_timeseries('my_volume_plot.png')

# 4. 保存
tracker.save_results('my_output')
```

### 全フレームを処理

```python
# max_framesを指定しない = 全フレーム処理
results_df = tracker.track_volume_timeseries()
```

---

## 📐 パラメータの決め方

### Voxelサイズ (重要!)

**XY方向**:
```
voxel_xy = カメラのピクセルサイズ (um) / 倍率

例: 
- カメラ: 6.5 um/pixel
- 対物レンズ: 100×
→ voxel_xy = 6.5 / 100 = 0.065 um/pixel
```

**Z方向**:
```
voxel_z = 細胞の厚み (um) / 希望するスライス数

例:
- 細胞厚: ~3 um
- スライス数: 10枚欲しい
→ voxel_z = 3 / 10 = 0.3 um/slice
```

---

## 🔬 アルゴリズム（簡単版）

```
ROI (輪郭線)
    ↓
1. 2Dマスクに変換
    ↓
2. Distance Transform
   (各点から境界までの距離 = 半径)
    ↓
3. Skeleton
   (中心軸を抽出)
    ↓
4. 球体の断面積で3D展開
   r(z) = √(R² - z²)
    ↓
5. 体積 = voxel数 × voxelサイズ
```

詳細: `docs/workflows/pomegranate_reconstruction_summary.md`

---

## 📈 結果の見方

### グラフ（4パネル）

1. **左上**: 個別細胞の体積変化
   - 各線が1つの細胞
   - 複数細胞の追跡を確認

2. **右上**: 平均体積変化（±標準偏差）
   - 全体のトレンドを把握
   - 青い帯 = ばらつき

3. **左下**: 体積分布の変化
   - 時間経過での分布の変化
   - バイオリンプロット

4. **右下**: 2D面積 vs 3D体積
   - 相関を確認（R²値）
   - 外れ値の検出

### CSV解析

```python
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv('timeseries_volume_output/volume_timeseries.csv')

# 特定の細胞をプロット
cell_0 = df[df['cell_index'] == 0]
plt.plot(cell_0['time_index'], cell_0['volume_um3'])
plt.xlabel('Time (frame)')
plt.ylabel('Volume (um^3)')
plt.savefig('cell_0_volume.png')

# 統計
print(f"Mean volume: {df['volume_um3'].mean():.2f} um^3")
print(f"Volume range: {df['volume_um3'].min():.2f} - {df['volume_um3'].max():.2f} um^3")
```

---

## ⚠️ よくある問題

### 問題1: "Total processed: 0 cells"

**原因**: ROI形式が未対応、または画像サイズが合わない

**解決策**:
```python
# 画像サイズを実際のサイズに変更
tracker = TimeSeriesVolumeTracker(
    ...,
    image_width=1024,   # ImageJで確認
    image_height=1024
)
```

### 問題2: 体積が異常に大きい/小さい

**原因**: voxelサイズの設定ミス

**解決策**:
- ImageJで画像を開く
- `Analyze > Set Scale` で実際のスケールを確認
- `voxel_xy` を正しい値に修正

### 問題3: メモリエラー

**原因**: ROI数が多すぎる

**解決策**:
```python
# 段階的に処理
for i in range(0, total_frames, 100):
    results = tracker.track_volume_timeseries(max_frames=100)
    results.to_csv(f'results_part_{i}.csv')
```

---

## 🎓 次のステップ

### 1. 細胞成長速度の計算

```python
# 体積変化率
df['volume_change'] = df.groupby('cell_index')['volume_um3'].diff()
df['growth_rate'] = df['volume_change'] / time_per_frame  # um^3/min
```

### 2. 細胞周期の検出

```python
from scipy.signal import find_peaks

# 体積のピーク検出
peaks, _ = find_peaks(df['volume_um3'], prominence=0.5)
print(f"Division events detected at frames: {df.iloc[peaks]['time_index'].values}")
```

### 3. 他のツールと連携

- **TrackMate**: 細胞追跡
- **CellProfiler**: 高度な画像解析
- **napari**: 3D可視化

---

## 📚 詳細ドキュメント

| ファイル | 内容 |
|---------|------|
| `timeseries_volume_tracking_guide.md` | 完全な使用ガイド |
| `pomegranate_reconstruction_summary.md` | アルゴリズムの詳細 |
| `pomegranate_algorithm_theory.md` | 数学的理論 |

---

## ✅ まとめ

このツールで、以下ができます：

✅ **ImageJ ROIセット** → **3D体積**  
✅ **時系列追跡** で体積変化を定量化  
✅ **自動計算** で大量のデータを効率的に処理  
✅ **可視化** で直感的に理解  

**たった1コマンドで、細胞の3D体積変化を追跡できます！**

```bash
python timeseries_volume_from_roiset.py
```

---

**作成日**: 2025-12-23  
**バージョン**: 1.0  
**連絡先**: QPI_omni Project

