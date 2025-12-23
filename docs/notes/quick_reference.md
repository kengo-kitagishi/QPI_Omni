# クイックリファレンス

よく使うコマンドやパラメータの早見表

## 🚀 実行コマンド

### 単体実行
```bash
cd scripts
python 24_elip_volume.py
```

### バッチ実行
```bash
cd scripts
python 28_batch_analysis.py
```

---

## 🔧 よく使うパラメータ

### QPI実験パラメータ
```python
WAVELENGTH_NM = 663        # レーザー波長
N_MEDIUM = 1.333           # 培地屈折率
PIXEL_SIZE_UM = 0.348      # ピクセルサイズ
ALPHA_RI = 0.00018         # 比屈折率増分
```

### 解析パラメータ
```python
SHAPE_TYPE = 'ellipse'     # または 'feret'
SUBPIXEL_SAMPLING = 5      # 1, 5, 10
MAX_ROIS = None            # 全ROI（テスト時は5）
```

---

## 📊 計算式

### RI
```
n_sample = n_medium + (φ × λ) / (2π × thickness)
```

### 濃度
```
C [mg/ml] = (RI - RI_medium) / α
```

### Total Mass
```
Total mass [pg] = Σ(C [mg/ml] × V [µm³])
```

---

## 📂 出力ディレクトリ

```
scripts/
├── timeseries_density_output_{shape}_{subpixel}/
│   ├── density_tiff/
│   ├── visualizations/
│   ├── csv_data/
│   └── all_rois_summary.csv
└── timeseries_plots_{shape}_{subpixel}/
    └── timeseries_volume_ri_mass.png
```

---

## 🔍 よく使う検索

### ファイル検索
```bash
# 特定ROIの可視化を探す
ls timeseries_density_output_*/visualizations/ROI_0000*

# 時系列プロットを探す
ls timeseries_plots_*/timeseries_volume_ri_mass.png
```

### コード内検索
```bash
# Total Massの計算箇所を探す
grep -n "total_mass" 24_elip_volume.py

# サブピクセルサンプリングの実装を探す
grep -n "subpixel_sampling" 24_elip_volume.py
```

---

## 🐛 トラブルシューティング

### 実行が遅い
→ `SUBPIXEL_SAMPLING = 1` でテスト

### メモリ不足
→ `MAX_ROIS = 5` で分割実行

### マージ失敗
→ `Results.csv`の`roi_index`を確認

---

最終更新: 2025-12-23
