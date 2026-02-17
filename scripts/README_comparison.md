# 条件比較と再処理スクリプト

## 30_simple_mean_ri_analysis.py - シンプルなmean RI計算 ⭐NEW⭐

**ピクセルごとにRIを計算せず**、全位相の合計を回転体体積で割ってmean RIを求める、よりシンプルな方法です。

### 計算式

```
mean_RI = n_medium + (total_phase × λ × pixel_area) / (2π × volume)
```

ここで:
- `total_phase`: マスク内の全ピクセルの位相値の合計 (rad)
- `volume`: rod shape回転体の体積 (µm³)
- `λ`: 波長 (nm → µm)
- `pixel_area`: ピクセル面積 (µm²)
- `n_medium`: 培地の屈折率

### 使用方法

```bash
# カレントディレクトリの全条件を処理
python scripts/30_simple_mean_ri_analysis.py

# G:\test_dens_estで処理
python scripts/30_simple_mean_ri_analysis.py -d G:\test_dens_est

# 特定の条件のみ
python scripts/30_simple_mean_ri_analysis.py -c "*ellipse*subpixel10*"

# 条件リストのみ表示
python scripts/30_simple_mean_ri_analysis.py --list-only
```

### 出力

各条件に対応する`timeseries_plots_*_simple_mean_ri/`ディレクトリが生成されます：

- **timeseries_simple_mean_ri.png**: 時系列プロット
  - Volume vs Frame
  - Mean RI vs Frame
  - Total Mass vs Frame

- **simple_mean_ri_summary.csv**: ROI/Frameごとのサマリー
  - mean_ri, volume_um3, total_phase_rad, mean_concentration_mg_ml, total_mass_pg

- **simple_mean_ri_all_conditions_summary.csv**: 全条件の統合サマリー

### コマンドライン引数

| オプション | 短縮形 | デフォルト | 説明 |
|-----------|--------|-----------|------|
| `--base-dir` | `-d` | `.` | 基準ディレクトリ |
| `--conditions` | `-c` | 全条件 | 処理する条件（複数指定可、ワイルドカード可） |
| `--pixel-size` | - | `0.348` | ピクセルサイズ（µm） |
| `--wavelength` | - | `663` | 波長（nm） |
| `--n-medium` | - | `1.333` | 培地の屈折率 |
| `--alpha-ri` | - | `0.00018` | 比屈折率増分（ml/mg） |
| `--voxel-z` | - | `0.3` | Z方向のボクセルサイズ（µm） |
| `--list-only` | - | False | 条件リストのみ表示 |

---

## 28_compare_all_conditions.py - 全条件の結果比較

全条件の`all_rois_summary.csv`を読み込んで、体積・質量・RIを比較します。

### 使用方法

```bash
cd C:\Users\QPI\Documents\QPI_omni
python scripts/28_compare_all_conditions.py
```

### 出力

`condition_comparison/`ディレクトリに以下が生成されます：

- **all_conditions_timeseries.png**: 全条件の時系列プロット
  - 体積 vs 時間
  - 質量 vs 時間
  - RI vs 時間

- **condition_comparison_bars.png**: 条件ごとの平均値バープロット
  - 平均体積
  - 平均質量
  - 平均RI

- **condition_summary_statistics.csv**: 条件ごとの統計サマリー
  - 平均、標準偏差、ROI数など

- **heatmap_continuous.png**: Continuousモードのヒートマップ
  - Shape type × Subpixel sampling

- **heatmap_discrete_*.png**: Discreteモードのヒートマップ
  - Discretize method × Subpixel sampling
  - Shape typeごとに分割

---

## 29_reprocess_with_thickness_filter.py - 1ピクセル未満をフィルタリング

既存の結果から、1ピクセル未満の厚みを持つピクセルをマスク外として再処理します。

### 使用方法

**基本的な使い方（カレントディレクトリの全条件を処理）**:
```bash
cd C:\Users\QPI\Documents\QPI_omni
python scripts/29_reprocess_with_thickness_filter.py
```

**G:\test_dens_estで実行する場合**:
```bash
# 方法1: カレントディレクトリを移動
cd G:\test_dens_est
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py

# 方法2: -dオプションでディレクトリ指定
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -d G:\test_dens_est

# 方法3: 確認なしで実行
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -d G:\test_dens_est -y
```

**特定の条件のみ処理**:
```bash
# 1つの条件のみ
python scripts/29_reprocess_with_thickness_filter.py -c timeseries_density_output_ellipse_subpixel5

# 複数の条件（ワイルドカード）
python scripts/29_reprocess_with_thickness_filter.py -c timeseries_density_output_ellipse*

# ellipseのsubpixel5のみ
python scripts/29_reprocess_with_thickness_filter.py -c "*ellipse*subpixel5*"

# 複数の条件を明示的に指定
python scripts/29_reprocess_with_thickness_filter.py -c timeseries_density_output_ellipse_subpixel5 timeseries_density_output_feret_subpixel5
```

**パラメータをカスタマイズ**:
```bash
# 0.5ピクセル未満を除外
python scripts/29_reprocess_with_thickness_filter.py -t 0.5

# Z方向のボクセルサイズを変更
python scripts/29_reprocess_with_thickness_filter.py --voxel-z 0.25

# 複数のパラメータを変更
python scripts/29_reprocess_with_thickness_filter.py -t 1.5 --pixel-size 0.35 --wavelength 532
```

**条件リストのみ表示**:
```bash
# 処理せずに条件リストだけ確認
python scripts/29_reprocess_with_thickness_filter.py --list-only

# 特定のディレクトリの条件リストを確認
python scripts/29_reprocess_with_thickness_filter.py -d G:\test_dens_est --list-only
```

### コマンドライン引数

| オプション | 短縮形 | デフォルト | 説明 |
|-----------|--------|-----------|------|
| `--base-dir` | `-d` | `.` | 基準ディレクトリ |
| `--conditions` | `-c` | 全条件 | 処理する条件（複数指定可、ワイルドカード可） |
| `--min-thickness` | `-t` | `1.0` | 最小厚み閾値（ピクセル単位） |
| `--pixel-size` | - | `0.348` | ピクセルサイズ（µm） |
| `--wavelength` | - | `663` | 波長（nm） |
| `--n-medium` | - | `1.333` | 培地の屈折率 |
| `--alpha-ri` | - | `0.00018` | 比屈折率増分（ml/mg） |
| `--voxel-z` | - | `0.3` | Z方向のボクセルサイズ（µm） |
| `--yes` | `-y` | False | 確認なしで実行 |
| `--list-only` | - | False | 条件リストのみ表示 |

### ヘルプ表示

```bash
python scripts/29_reprocess_with_thickness_filter.py --help
```

### 処理内容

1. 各条件ディレクトリの`density_tiff/*_zstack.tif`を読み込み
2. `min_thickness_px=1.0`でフィルタリング
3. RI、質量濃度、体積、質量を再計算
4. 新しいディレクトリ`*_filtered_1.0px/`に保存

### 出力

各条件ディレクトリに対応する`*_filtered_1.0px/`ディレクトリが生成されます：

```
timeseries_density_output_ellipse_subpixel5/
timeseries_density_output_ellipse_subpixel5_filtered_1.0px/  ← 新規作成
  ├── density_tiff/
  │   ├── ROI_0000_Frame_0001_zstack.tif      (フィルタリング済み)
  │   ├── ROI_0000_Frame_0001_ri.tif          (再計算)
  │   ├── ROI_0000_Frame_0001_concentration.tif
  │   └── ...
  ├── csv_data/
  │   ├── ROI_0000_Frame_0001_pixel_data.csv  (フィルタリング済み)
  │   └── ...
  └── filtering_summary.csv                    (フィルタリング統計)
```

**filtering_summary.csv**の内容：
- `pixels_before`: フィルタリング前のピクセル数
- `pixels_after`: フィルタリング後のピクセル数
- `pixels_filtered`: 除外されたピクセル数
- `filter_ratio`: 除外率（%）
- `volume_um3`: 再計算された体積
- `total_mass_pg`: 再計算された質量
- など

**reprocessed_all_conditions_summary.csv**: 全条件の統合サマリー

---

## 実行例

### 🌟 実行例0: シンプルなmean RI計算（推奨！）

```bash
# G:\test_dens_estで全条件をシンプルな方法で処理
python C:\Users\QPI\Documents\QPI_omni\scripts\30_simple_mean_ri_analysis.py -d G:\test_dens_est

# 特定の条件のみ（例: ellipse + subpixel10）
python C:\Users\QPI\Documents\QPI_omni\scripts\30_simple_mean_ri_analysis.py -d G:\test_dens_est -c "*ellipse*subpixel10*"

# フィルタリング済みデータで処理
python C:\Users\QPI\Documents\QPI_omni\scripts\30_simple_mean_ri_analysis.py -d G:\test_dens_est -c "*filtered_1.0px"
```

### 実行例1: G:\test_dens_estで全条件を処理

```bash
# まず条件リストを確認
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -d G:\test_dens_est --list-only

# 確認後、全条件を処理（確認なし）
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -d G:\test_dens_est -y

# 結果を比較
cd G:\test_dens_est
python C:\Users\QPI\Documents\QPI_omni\scripts\28_compare_all_conditions.py
```

### 実行例2: 特定の条件のみ処理

```bash
cd G:\test_dens_est

# ellipseの条件のみ処理
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -c "*ellipse*" -y

# subpixel5の条件のみ処理
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -c "*subpixel5*" -y
```

### 実行例3: カスタムパラメータで処理

```bash
# 0.5ピクセル未満を無視、Z方向0.25µm
python scripts/29_reprocess_with_thickness_filter.py -t 0.5 --voxel-z 0.25 -y

# 2ピクセル未満を無視（より保守的）
python scripts/29_reprocess_with_thickness_filter.py -t 2.0 -y
```

### 実行例4: フィルタリング前後を比較

```bash
# 1. フィルタリング前の結果を比較
python scripts/28_compare_all_conditions.py

# 2. フィルタリング実行
python scripts/29_reprocess_with_thickness_filter.py -y

# 3. フィルタリング後の結果を確認
# filtering_summary.csv を all_rois_summary.csv にコピーしてから28.pyを実行
# または、28.pyを修正して *_filtered_1.0px ディレクトリを対象にする
```

---

## よくある質問

### Q0: ピクセルごとのRI計算とシンプルなmean RI計算の違いは？

A: 2つの方法があります：

**従来の方法（ピクセルごと）**:
- 各ピクセルで `RI = n_medium + (phase × λ) / (2π × thickness)` を計算
- 全ピクセルの平均を取る
- 厚みが薄いピクセルでRIが過大評価される可能性

**シンプルな方法（30.py）**:
- `mean_RI = n_medium + (total_phase × λ) / (2π × volume)`
- 全体の位相を全体の体積で割る
- より安定した値が得られる（推奨）

### Q1: G:\test_dens_estのデータを処理したい

A: 以下の方法のいずれかを使用してください：

**方法1: カレントディレクトリを移動**
```bash
cd G:\test_dens_est
python C:\Users\QPI\Documents\QPI_omni\scripts\28_compare_all_conditions.py
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py
```

**方法2: -dオプションで指定（29.pyのみ対応）**
```bash
python C:\Users\QPI\Documents\QPI_omni\scripts\29_reprocess_with_thickness_filter.py -d G:\test_dens_est
```

**方法3: 28.pyも対応させる**
28.pyの`load_all_conditions()`に引数を追加：
```python
df = load_all_conditions(base_dir='G:\\test_dens_est')
```

### Q2: 特定の条件だけ処理したい

A: `-c`オプションを使用してください：

```bash
# ellipseだけ
python scripts/29_reprocess_with_thickness_filter.py -c "*ellipse*"

# subpixel5だけ
python scripts/29_reprocess_with_thickness_filter.py -c "*subpixel5*"

# ellipseかつsubpixel5
python scripts/29_reprocess_with_thickness_filter.py -c "*ellipse*subpixel5*"

# 複数の条件を明示的に指定
python scripts/29_reprocess_with_thickness_filter.py -c \
    timeseries_density_output_ellipse_subpixel5 \
    timeseries_density_output_feret_subpixel5

# まず--list-onlyで確認してから実行
python scripts/29_reprocess_with_thickness_filter.py -c "*ellipse*" --list-only
python scripts/29_reprocess_with_thickness_filter.py -c "*ellipse*" -y
```

### Q3: 閾値を変更したい（例: 0.5ピクセル）

A: `-t`オプションを使用してください：

```bash
# 0.5ピクセル未満を無視
python scripts/29_reprocess_with_thickness_filter.py -t 0.5

# 2.0ピクセル未満を無視（より保守的）
python scripts/29_reprocess_with_thickness_filter.py -t 2.0

# G:\test_dens_estで0.5ピクセル閾値を適用
python scripts/29_reprocess_with_thickness_filter.py -d G:\test_dens_est -t 0.5 -y
```

### Q4: フィルタリング前後を比較したい

A: 28.pyを2回実行：

```bash
# フィルタリング前
python scripts/28_compare_all_conditions.py

# フィルタリング後（スクリプト修正が必要）
# pattern = 'timeseries_density_output_*_filtered_1.0px' に変更
python scripts/28_compare_all_conditions.py
```

---

## トラブルシューティング

### エラー: "No data found!"

- `timeseries_density_output_*`ディレクトリが存在するか確認
- `all_rois_summary.csv`が各ディレクトリに存在するか確認
- 正しいディレクトリで実行しているか確認

### エラー: "No z-stack files found"

- `density_tiff/`ディレクトリが存在するか確認
- `*_zstack.tif`ファイルが存在するか確認

### メモリエラー

- 条件数が多い場合、一部ずつ処理
- または`MAX_ROIS`で制限をかける

