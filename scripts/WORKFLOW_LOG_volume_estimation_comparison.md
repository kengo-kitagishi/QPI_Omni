# ワークログ: 体積推定メソッド比較システムの構築

**作成日**: 2025-12-24  
**目的**: 複数のパラメータ組み合わせでQPI体積推定を行い、結果を比較するバッチシステムの構築

---

## 📋 目次

1. [背景と問題点](#背景と問題点)
2. [実装した機能](#実装した機能)
3. [修正の詳細](#修正の詳細)
4. [最終的なファイル構成](#最終的なファイル構成)
5. [使用方法](#使用方法)
6. [実行条件の詳細](#実行条件の詳細)

---

## 背景と問題点

### 初期状態
- `24_elip_volume.py`: 単一条件でQPI体積推定を実行するスクリプト
- `28_batch_analysis.py`: 複数条件でバッチ実行するスクリプト（初期バージョン）

### 発見された問題

#### 問題1: `dir_suffix`変数が未定義エラー
**エラー内容**:
```
NameError: name 'dir_suffix' is not defined
```

**原因**:
- `24_elip_volume.py`の90行目で`dir_suffix`をローカル変数として定義
- 801行目で再度使用しようとしたが、スコープ外だった

**発生箇所**:
```python
# 90行目: __init__メソッド内
dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"

# 801行目: メイン実行ブロック内（スコープ外）
plot_output_dir = f"timeseries_plots_{dir_suffix}"  # エラー！
```

#### 問題2: CSVファイルによる出力フォルダ名の区別ができない
**状況**:
- 2つのCSVファイル（`Results_enlarge.csv`, `Results_enlarge_interpolate.csv`）を使用
- 出力フォルダ名が同じになり、結果が上書きされる

#### 問題3: バッチ実行が6条件のみ（12条件にしたい）
**現状**: 1 CSV × 2 shape_types × 3 subpixel_samplings = 6条件  
**目標**: 2 CSVs × 2 shape_types × 3 subpixel_samplings = 12条件

#### 問題4: ハードコードされたパラメータによる上書き
**原因**:
```python
# 24_elip_volume.py の764-768行目
SHAPE_TYPE = 'feret'         # 常にferetで上書き！
SUBPIXEL_SAMPLING = 5        # 常に5で上書き！
```

**結果**: バッチ実行で`ellipse`や`subpixel=1`を指定しても、常に`feret + subpixel5`が実行される

---

## 実装した機能

### 機能1: CSVファイル名からの自動サフィックス抽出
- CSVファイル名を解析し、自動的に識別子を抽出
- 例: `Results_enlarge.csv` → `enlarge`
- 例: `Results_enlarge_interpolate.csv` → `enlarge_interpolate`

### 機能2: 手動サフィックス指定のサポート
- 必要に応じて手動でカスタムサフィックスを指定可能
- 自動抽出と手動指定の両方に対応

### 機能3: 複数CSVファイルの一括処理
- 2つ以上のCSVファイルをリストで指定
- 各CSVに対してすべてのパラメータ組み合わせを実行

### 機能4: パラメータの柔軟な管理
- バッチ実行時: 渡されたパラメータを使用
- 単独実行時: デフォルト値を使用

---

## 修正の詳細

### 修正1: `dir_suffix`のスコープ問題を解決

**ファイル**: `24_elip_volume.py`

**変更箇所**: 90行目
```python
# 修正前
dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
self.output_dir = f"timeseries_density_output_{dir_suffix}"

# 修正後
self.dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
self.output_dir = f"timeseries_density_output_{self.dir_suffix}"
```

**変更箇所**: 801行目
```python
# 修正前
plot_output_dir = f"timeseries_plots_{dir_suffix}"

# 修正後
plot_output_dir = f"timeseries_plots_{mapper.dir_suffix}"
```

### 修正2: CSVサフィックス機能の実装

**ファイル**: `24_elip_volume.py`

**変更箇所**: コンストラクタの引数に追加（23-50行目付近）
```python
def __init__(self, results_csv, image_directory, 
             wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
             alpha_ri=0.0018, shape_type='ellipse', subpixel_sampling=5,
             csv_suffix=None):  # 新規追加
```

**変更箇所**: パラメータドキュメント（50行目付近に追加）
```python
csv_suffix : str, optional
    出力フォルダ名に追加するサフィックス。デフォルト: None
    Noneの場合、CSVファイル名から自動抽出（例: Results_enlarge.csv → enlarge）
    手動で指定する場合: 'enlarge', 'interpolate', 'custom_name'など
```

**変更箇所**: サフィックス抽出ロジック（__init__メソッド内、71-89行目付近）
```python
# CSVサフィックスを決定（手動指定 or 自動抽出）
if csv_suffix is not None:
    self.csv_suffix = csv_suffix
else:
    # CSVファイル名から自動抽出 (例: Results_enlarge.csv → enlarge)
    csv_filename = os.path.basename(results_csv)
    csv_name_without_ext = os.path.splitext(csv_filename)[0]  # Results_enlarge
    # "Results_"の後の部分を取得（あれば）
    if '_' in csv_name_without_ext:
        parts = csv_name_without_ext.split('_', 1)  # ['Results', 'enlarge']
        if len(parts) > 1 and parts[1]:
            self.csv_suffix = parts[1]
        else:
            self.csv_suffix = None
    else:
        self.csv_suffix = None
```

**変更箇所**: dir_suffixの生成ロジック（95-99行目付近）
```python
# 出力ディレクトリ（パラメータに応じた名前）
if self.csv_suffix:
    self.dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}_{self.csv_suffix}"
else:
    self.dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
self.output_dir = f"timeseries_density_output_{self.dir_suffix}"
```

**変更箇所**: mapperインスタンス作成時（792-803行目付近）
```python
mapper = TimeSeriesDensityMapper(
    results_csv=RESULTS_CSV,
    image_directory=IMAGE_DIRECTORY,
    wavelength_nm=WAVELENGTH_NM,
    n_medium=N_MEDIUM,
    pixel_size_um=PIXEL_SIZE_UM,
    alpha_ri=ALPHA_RI,
    shape_type=SHAPE_TYPE,
    subpixel_sampling=SUBPIXEL_SAMPLING,
    csv_suffix=CSV_SUFFIX  # 新規追加
)
```

### 修正3: バッチスクリプトで複数CSV対応

**ファイル**: `compare_volume_estimation_methods.py`（旧`28_batch_analysis.py`）

**変更箇所**: run_analysis関数の引数（39-48行目）
```python
# 修正前
def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory, 
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois):

# 修正後
def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory, 
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois, csv_suffix=None):
    """
    csv_suffix : str, optional
        出力フォルダ名に追加するサフィックス。デフォルト: None
        Noneの場合、CSVファイル名から自動抽出
    """
```

**変更箇所**: globals_dictに追加（61-73行目）
```python
globals_dict = {
    '__name__': '__main__',
    'RESULTS_CSV': results_csv,
    'IMAGE_DIRECTORY': image_directory,
    'WAVELENGTH_NM': wavelength_nm,
    'N_MEDIUM': n_medium,
    'PIXEL_SIZE_UM': pixel_size_um,
    'ALPHA_RI': alpha_ri,
    'SHAPE_TYPE': shape_type,
    'SUBPIXEL_SAMPLING': subpixel_sampling,
    'MAX_ROIS': max_rois,
    'CSV_SUFFIX': csv_suffix,  # 新規追加
}
```

**変更箇所**: CSVファイルの指定方法（109-113行目）
```python
# 修正前
RESULTS_CSV = r"C:\...\Results_enlarge_interpolate.csv"

# 修正後
RESULTS_CSVS = [
    r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge.csv",
    r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge_interpolate.csv"
]
```

**変更箇所**: CSV_SUFFIXパラメータの追加（119-126行目）
```python
# === CSVサフィックス（出力フォルダ名の識別用）===
# オプション1: 自動抽出（Noneを指定）
#   - CSVファイル名から自動で抽出されます
#   - Results_enlarge.csv → 'enlarge'
#   - Results_enlarge_interpolate.csv → 'enlarge_interpolate'
# オプション2: 手動指定（文字列を指定）
#   - 例: CSV_SUFFIX = 'my_custom_name'
CSV_SUFFIX = None  # Noneで自動抽出、または手動で文字列を指定
```

**変更箇所**: バッチ実行開始メッセージ（137-147行目）
```python
# 修正前
print(f"# Total combinations: {len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)}")

# 修正後
total_combos = len(RESULTS_CSVS) * len(SHAPE_TYPES) * len(SUBPIXEL_SAMPLINGS)
print(f"# Total combinations: {total_combos}")
print(f"#   CSV files: {len(RESULTS_CSVS)}")
print(f"#   Shape types: {len(SHAPE_TYPES)}")
print(f"#   Subpixel samplings: {len(SUBPIXEL_SAMPLINGS)}")
```

**変更箇所**: 実行ループの構造（148-189行目）
```python
# 修正前
for i, shape_type in enumerate(SHAPE_TYPES, 1):
    for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
        combo_num = (i-1) * len(SUBPIXEL_SAMPLINGS) + j
        ...

# 修正後
combo_num = 0
for csv_idx, results_csv in enumerate(RESULTS_CSVS, 1):
    csv_name = os.path.basename(results_csv)
    print(f"\nProcessing CSV {csv_idx}/{len(RESULTS_CSVS)}: {csv_name}")
    
    for i, shape_type in enumerate(SHAPE_TYPES, 1):
        for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
            combo_num += 1
            
            print(f"# Combination {combo_num}/{total_combos}")
            print(f"#   CSV: {csv_name}")
            print(f"#   Shape: {shape_type}")
            print(f"#   Subpixel: {subpixel_sampling}×{subpixel_sampling}")
            ...
```

**変更箇所**: resultsリストの構造（170-175行目）
```python
# 修正前
results.append({
    'shape_type': shape_type,
    'subpixel_sampling': subpixel_sampling,
    'success': success
})

# 修正後
results.append({
    'csv_file': csv_name,
    'shape_type': shape_type,
    'subpixel_sampling': subpixel_sampling,
    'success': success
})
```

**変更箇所**: サマリー表示（200-241行目）
```python
# CSVファイル名を含めた表示
for result in results:
    status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
    csv_short = result['csv_file'].replace('Results_', '').replace('.csv', '')
    print(f"  {csv_short:20s} | {result['shape_type']:8s} | subpixel{result['subpixel_sampling']:2d} : {status}")

# CSVファイルごとにグループ化して出力ディレクトリを表示
for results_csv in RESULTS_CSVS:
    csv_name = os.path.basename(results_csv)
    csv_short = csv_name.replace('Results_', '').replace('.csv', '')
    print(f"\n  [{csv_short}]")
    
    for result in results:
        if result['csv_file'] == csv_name:
            # CSVファイル名から自動抽出されるサフィックスを推定
            csv_name_without_ext = os.path.splitext(csv_name)[0]
            if '_' in csv_name_without_ext:
                parts = csv_name_without_ext.split('_', 1)
                csv_suffix_auto = parts[1] if len(parts) > 1 and parts[1] else None
            else:
                csv_suffix_auto = None
            
            if csv_suffix_auto:
                dir_suffix = f"{result['shape_type']}_subpixel{result['subpixel_sampling']}_{csv_suffix_auto}"
            else:
                dir_suffix = f"{result['shape_type']}_subpixel{result['subpixel_sampling']}"
            
            print(f"    - timeseries_density_output_{dir_suffix}/")
            print(f"    - timeseries_plots_{dir_suffix}/")
```

### 修正4: ハードコードされたパラメータの条件付き代入

**ファイル**: `24_elip_volume.py`

**変更箇所**: 全パラメータを条件付き代入に変更（742-773行目）

```python
# 修正前（ハードコード）
RESULTS_CSV = r"C:\Users\QPI\Desktop\...\Results_enlarge.csv"
IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\...\subtracted"
WAVELENGTH_NM = 663
N_MEDIUM = 1.333
PIXEL_SIZE_UM = 0.348
ALPHA_RI = 0.00018
SHAPE_TYPE = 'feret'
SUBPIXEL_SAMPLING = 5
MAX_ROIS = None

# 修正後（条件付き代入）
# バッチ実行時はglobals()から、単独実行時はデフォルト値を使用
RESULTS_CSV = r"C:\Users\QPI\Desktop\...\Results_enlarge.csv" if 'RESULTS_CSV' not in globals() else globals()['RESULTS_CSV']
IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\...\subtracted" if 'IMAGE_DIRECTORY' not in globals() else globals()['IMAGE_DIRECTORY']
WAVELENGTH_NM = 663 if 'WAVELENGTH_NM' not in globals() else globals()['WAVELENGTH_NM']
N_MEDIUM = 1.333 if 'N_MEDIUM' not in globals() else globals()['N_MEDIUM']
PIXEL_SIZE_UM = 0.348 if 'PIXEL_SIZE_UM' not in globals() else globals()['PIXEL_SIZE_UM']
ALPHA_RI = 0.00018 if 'ALPHA_RI' not in globals() else globals()['ALPHA_RI']
SHAPE_TYPE = 'feret' if 'SHAPE_TYPE' not in globals() else globals()['SHAPE_TYPE']
SUBPIXEL_SAMPLING = 5 if 'SUBPIXEL_SAMPLING' not in globals() else globals()['SUBPIXEL_SAMPLING']
MAX_ROIS = None if 'MAX_ROIS' not in globals() else globals()['MAX_ROIS']
CSV_SUFFIX = None if 'CSV_SUFFIX' not in globals() else globals()['CSV_SUFFIX']
```

**重要**: この修正により、バッチ実行時に渡されたパラメータが正しく使用されるようになりました。

### 修正5: ファイル名の変更

**実行コマンド**:
```powershell
cd c:\Users\QPI\Documents\QPI_omni\scripts
Move-Item -Path "28_batch_analysis.py" -Destination "compare_volume_estimation_methods.py"
```

**理由**: ファイル名から機能が明確に分かるようにするため

---

## 最終的なファイル構成

```
scripts/
├── 24_elip_volume.py                    # 単一条件での体積推定スクリプト
├── compare_volume_estimation_methods.py # バッチ比較スクリプト（旧28_batch_analysis.py）
│
├── timeseries_density_output_ellipse_subpixel1_enlarge/
├── timeseries_density_output_ellipse_subpixel5_enlarge/
├── timeseries_density_output_ellipse_subpixel10_enlarge/
├── timeseries_density_output_feret_subpixel1_enlarge/
├── timeseries_density_output_feret_subpixel5_enlarge/
├── timeseries_density_output_feret_subpixel10_enlarge/
├── timeseries_density_output_ellipse_subpixel1_enlarge_interpolate/
├── timeseries_density_output_ellipse_subpixel5_enlarge_interpolate/
├── timeseries_density_output_ellipse_subpixel10_enlarge_interpolate/
├── timeseries_density_output_feret_subpixel1_enlarge_interpolate/
├── timeseries_density_output_feret_subpixel5_enlarge_interpolate/
├── timeseries_density_output_feret_subpixel10_enlarge_interpolate/
│
├── timeseries_plots_ellipse_subpixel1_enlarge/
├── timeseries_plots_ellipse_subpixel5_enlarge/
... (同様に12個のプロットフォルダ)
```

---

## 使用方法

### 1. 単一条件での実行（24_elip_volume.py）

**直接実行する場合**:
```python
# 24_elip_volume.pyを直接実行
python 24_elip_volume.py
```

スクリプト内のデフォルト値が使用されます：
- `RESULTS_CSV`: Results_enlarge.csv
- `SHAPE_TYPE`: feret
- `SUBPIXEL_SAMPLING`: 5

**カスタマイズして実行する場合**:
```python
from importlib.machinery import SourceFileLoader

# パラメータを指定
globals_dict = {
    'RESULTS_CSV': r'C:\path\to\your\Results.csv',
    'IMAGE_DIRECTORY': r'C:\path\to\images',
    'SHAPE_TYPE': 'ellipse',
    'SUBPIXEL_SAMPLING': 10,
    'CSV_SUFFIX': 'my_experiment',
    # ... 他のパラメータ
}

# スクリプトを読み込んで実行
loader = SourceFileLoader('module', '24_elip_volume.py')
module = loader.load_module()
```

### 2. バッチ比較実行（compare_volume_estimation_methods.py）

**基本的な実行**:
```python
python compare_volume_estimation_methods.py
```

**カスタマイズ方法**:

ファイルを開いて以下の部分を編集：

```python
# CSVファイルを指定（111-113行目）
RESULTS_CSVS = [
    r"C:\Users\QPI\Desktop\...\Results_enlarge.csv",
    r"C:\Users\QPI\Desktop\...\Results_enlarge_interpolate.csv"
]

# 共通パラメータ（115-120行目）
WAVELENGTH_NM = 663
N_MEDIUM = 1.333
PIXEL_SIZE_UM = 0.348
ALPHA_RI = 0.00018
MAX_ROIS = 5  # テスト実行。Noneで全ROI処理

# CSVサフィックス（126行目）
CSV_SUFFIX = None  # Noneで自動抽出

# 条件の組み合わせ（129-130行目）
SHAPE_TYPES = ['ellipse', 'feret']
SUBPIXEL_SAMPLINGS = [1, 5, 10]
```

**実行時の出力例**:
```
################################################################################
# BATCH ANALYSIS START
# Start time: 2025-12-24 14:00:00
# Total combinations: 12
#   CSV files: 2
#   Shape types: 2
#   Subpixel samplings: 3
################################################################################

================================================================================
Processing CSV 1/2: Results_enlarge.csv
================================================================================

################################################################################
# Combination 1/12
#   CSV: Results_enlarge.csv
#   Shape: ellipse
#   Subpixel: 1×1
################################################################################

... (処理が進む)

################################################################################
# BATCH ANALYSIS COMPLETE
# End time: 2025-12-24 15:30:00
# Total elapsed time: 90.0 minutes (1.50 hours)
################################################################################

Results summary:
================================================================================
  enlarge              | ellipse  | subpixel 1 : ✅ SUCCESS
  enlarge              | ellipse  | subpixel 5 : ✅ SUCCESS
  enlarge              | ellipse  | subpixel10 : ✅ SUCCESS
  enlarge              | feret    | subpixel 1 : ✅ SUCCESS
  enlarge              | feret    | subpixel 5 : ✅ SUCCESS
  enlarge              | feret    | subpixel10 : ✅ SUCCESS
  enlarge_interpolate  | ellipse  | subpixel 1 : ✅ SUCCESS
  enlarge_interpolate  | ellipse  | subpixel 5 : ✅ SUCCESS
  enlarge_interpolate  | ellipse  | subpixel10 : ✅ SUCCESS
  enlarge_interpolate  | feret    | subpixel 1 : ✅ SUCCESS
  enlarge_interpolate  | feret    | subpixel 5 : ✅ SUCCESS
  enlarge_interpolate  | feret    | subpixel10 : ✅ SUCCESS
================================================================================

Success rate: 12/12 (100.0%)
```

---

## 実行条件の詳細

### 12条件の内訳

| # | CSV | Shape | Subpixel | 出力フォルダ名 |
|---|-----|-------|----------|---------------|
| 1 | enlarge | ellipse | 1×1 | `timeseries_density_output_ellipse_subpixel1_enlarge` |
| 2 | enlarge | ellipse | 5×5 | `timeseries_density_output_ellipse_subpixel5_enlarge` |
| 3 | enlarge | ellipse | 10×10 | `timeseries_density_output_ellipse_subpixel10_enlarge` |
| 4 | enlarge | feret | 1×1 | `timeseries_density_output_feret_subpixel1_enlarge` |
| 5 | enlarge | feret | 5×5 | `timeseries_density_output_feret_subpixel5_enlarge` |
| 6 | enlarge | feret | 10×10 | `timeseries_density_output_feret_subpixel10_enlarge` |
| 7 | enlarge_interpolate | ellipse | 1×1 | `timeseries_density_output_ellipse_subpixel1_enlarge_interpolate` |
| 8 | enlarge_interpolate | ellipse | 5×5 | `timeseries_density_output_ellipse_subpixel5_enlarge_interpolate` |
| 9 | enlarge_interpolate | ellipse | 10×10 | `timeseries_density_output_ellipse_subpixel10_enlarge_interpolate` |
| 10 | enlarge_interpolate | feret | 1×1 | `timeseries_density_output_feret_subpixel1_enlarge_interpolate` |
| 11 | enlarge_interpolate | feret | 5×5 | `timeseries_density_output_feret_subpixel5_enlarge_interpolate` |
| 12 | enlarge_interpolate | feret | 10×10 | `timeseries_density_output_feret_subpixel10_enlarge_interpolate` |

### パラメータの意味

#### Shape Type
- **ellipse**: ImageJの`Major`, `Minor`, `Angle`を使用した楕円近似
- **feret**: ImageJの`Feret`, `MinFeret`, `FeretAngle`を使用したFeret径近似

#### Subpixel Sampling
- **1×1**: ピクセル中心のみをサンプリング（高速だが端で精度低）
- **5×5**: 各ピクセルを5×5に分割してサンプリング（推奨、バランス良い）
- **10×10**: 各ピクセルを10×10に分割してサンプリング（高精度だが遅い）

#### CSV Type
- **enlarge**: 元のResults.csvを拡大したもの
- **enlarge_interpolate**: 拡大+補間処理を適用したもの

### 出力データ

各条件の出力フォルダには以下が含まれます：

```
timeseries_density_output_[条件]/
├── csv_data/
│   ├── ROI_XXXX_Frame_YYYY_pixel_data.csv      # ピクセルごとのデータ
│   └── ROI_XXXX_Frame_YYYY_parameters.csv      # ROIパラメータ
├── density_tiff/
│   ├── ROI_XXXX_Frame_YYYY_ri.tif              # 屈折率マップ
│   ├── ROI_XXXX_Frame_YYYY_concentration.tif   # 濃度マップ
│   ├── ROI_XXXX_Frame_YYYY_zstack.tif          # 厚みマップ
│   └── ROI_XXXX_Frame_YYYY_phase.tif           # 位相マップ
├── visualizations/
│   └── ROI_XXXX_Frame_YYYY_visualization.png   # 可視化画像
└── all_rois_summary.csv                         # 全ROIのサマリー

timeseries_plots_[条件]/
└── timeseries_volume_ri_mass.png                # 時系列プロット
```

---

## トラブルシューティング

### 問題: 実行しても常に`feret + subpixel5`が実行される

**原因**: Jupyter NotebookのKernelが古いバージョンをキャッシュしている

**解決方法**:
1. Jupyter Notebookの場合: **Kernel → Restart** を実行
2. 再度スクリプトを実行

### 問題: `dir_suffix`が未定義エラー

**確認事項**:
- `24_elip_volume.py`の90行目が`self.dir_suffix = ...`になっているか
- `24_elip_volume.py`の801行目が`mapper.dir_suffix`を使っているか

### 問題: 出力フォルダが上書きされる

**確認事項**:
- `CSV_SUFFIX`パラメータが正しく設定されているか
- CSVファイル名に識別可能な部分（例: `_enlarge`, `_interpolate`）が含まれているか

### 問題: バッチ実行で6条件しか実行されない

**確認事項**:
- `compare_volume_estimation_methods.py`の`RESULTS_CSVS`が**リスト**になっているか
- 古いバージョンの`28_batch_analysis.py`を実行していないか

---

## 再現手順

この会話の内容を完全に再現するには、以下の手順を実行してください：

### ステップ1: dir_suffix問題の修正

```python
# 24_elip_volume.py の90行目を編集
# 変更前:
dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"

# 変更後:
self.dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
```

```python
# 24_elip_volume.py の801行目を編集
# 変更前:
plot_output_dir = f"timeseries_plots_{dir_suffix}"

# 変更後:
plot_output_dir = f"timeseries_plots_{mapper.dir_suffix}"
```

### ステップ2: CSVサフィックス機能の追加

```python
# 24_elip_volume.py のコンストラクタに引数を追加（23行目付近）
def __init__(self, results_csv, image_directory, 
             wavelength_nm=663, n_medium=1.333, pixel_size_um=0.348,
             alpha_ri=0.0018, shape_type='ellipse', subpixel_sampling=5,
             csv_suffix=None):  # この行を追加
```

```python
# 24_elip_volume.py の__init__内（71-99行目付近）にロジックを追加
# CSVサフィックスを決定（手動指定 or 自動抽出）
if csv_suffix is not None:
    self.csv_suffix = csv_suffix
else:
    # CSVファイル名から自動抽出
    csv_filename = os.path.basename(results_csv)
    csv_name_without_ext = os.path.splitext(csv_filename)[0]
    if '_' in csv_name_without_ext:
        parts = csv_name_without_ext.split('_', 1)
        if len(parts) > 1 and parts[1]:
            self.csv_suffix = parts[1]
        else:
            self.csv_suffix = None
    else:
        self.csv_suffix = None

# ... (既存のコード)

# 出力ディレクトリの生成ロジックを修正
if self.csv_suffix:
    self.dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}_{self.csv_suffix}"
else:
    self.dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
self.output_dir = f"timeseries_density_output_{self.dir_suffix}"
```

```python
# 24_elip_volume.py の mapper作成部分にcsv_suffixを追加（792-803行目付近）
mapper = TimeSeriesDensityMapper(
    results_csv=RESULTS_CSV,
    image_directory=IMAGE_DIRECTORY,
    wavelength_nm=WAVELENGTH_NM,
    n_medium=N_MEDIUM,
    pixel_size_um=PIXEL_SIZE_UM,
    alpha_ri=ALPHA_RI,
    shape_type=SHAPE_TYPE,
    subpixel_sampling=SUBPIXEL_SAMPLING,
    csv_suffix=CSV_SUFFIX  # この行を追加
)
```

### ステップ3: バッチスクリプトの修正

```python
# compare_volume_estimation_methods.py（旧28_batch_analysis.py）の編集

# run_analysis関数にcsv_suffix引数を追加（39行目）
def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory, 
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois, csv_suffix=None):

# globals_dictにCSV_SUFFIXを追加（72行目）
globals_dict = {
    # ... 既存のキー ...
    'CSV_SUFFIX': csv_suffix,  # この行を追加
}

# CSVファイルをリストに変更（111-113行目）
RESULTS_CSVS = [
    r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge.csv",
    r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge_interpolate.csv"
]

# CSV_SUFFIXパラメータを追加（126行目）
CSV_SUFFIX = None

# ループ構造を変更（148行目以降）
combo_num = 0
for csv_idx, results_csv in enumerate(RESULTS_CSVS, 1):
    csv_name = os.path.basename(results_csv)
    
    for i, shape_type in enumerate(SHAPE_TYPES, 1):
        for j, subpixel_sampling in enumerate(SUBPIXEL_SAMPLINGS, 1):
            combo_num += 1
            # ... 既存のコード ...
            
            success = run_analysis(
                # ... 既存の引数 ...
                csv_suffix=CSV_SUFFIX  # この行を追加
            )
            
            results.append({
                'csv_file': csv_name,  # この行を追加
                'shape_type': shape_type,
                'subpixel_sampling': subpixel_sampling,
                'success': success
            })
```

### ステップ4: ハードコードパラメータの修正

```python
# 24_elip_volume.py のメイン実行ブロック（742-773行目）を全て条件付き代入に変更

RESULTS_CSV = r"C:\...\Results_enlarge.csv" if 'RESULTS_CSV' not in globals() else globals()['RESULTS_CSV']
IMAGE_DIRECTORY = r"C:\...\subtracted" if 'IMAGE_DIRECTORY' not in globals() else globals()['IMAGE_DIRECTORY']
WAVELENGTH_NM = 663 if 'WAVELENGTH_NM' not in globals() else globals()['WAVELENGTH_NM']
N_MEDIUM = 1.333 if 'N_MEDIUM' not in globals() else globals()['N_MEDIUM']
PIXEL_SIZE_UM = 0.348 if 'PIXEL_SIZE_UM' not in globals() else globals()['PIXEL_SIZE_UM']
ALPHA_RI = 0.00018 if 'ALPHA_RI' not in globals() else globals()['ALPHA_RI']
SHAPE_TYPE = 'feret' if 'SHAPE_TYPE' not in globals() else globals()['SHAPE_TYPE']
SUBPIXEL_SAMPLING = 5 if 'SUBPIXEL_SAMPLING' not in globals() else globals()['SUBPIXEL_SAMPLING']
MAX_ROIS = None if 'MAX_ROIS' not in globals() else globals()['MAX_ROIS']
CSV_SUFFIX = None if 'CSV_SUFFIX' not in globals() else globals()['CSV_SUFFIX']
```

### ステップ5: ファイル名の変更

```powershell
cd c:\Users\QPI\Documents\QPI_omni\scripts
Move-Item -Path "28_batch_analysis.py" -Destination "compare_volume_estimation_methods.py"
```

### ステップ6: 実行

```python
python compare_volume_estimation_methods.py
```

---

## まとめ

この一連の修正により、以下が実現されました：

1. ✅ **エラーの解消**: `dir_suffix`未定義エラーの修正
2. ✅ **自動識別**: CSVファイル名からの自動サフィックス抽出
3. ✅ **柔軟性**: 手動サフィックス指定のサポート
4. ✅ **拡張性**: 複数CSVファイルの一括処理（6→12条件）
5. ✅ **再現性**: バッチ実行時のパラメータが正しく反映される
6. ✅ **可読性**: ファイル名から機能が明確に分かる

これにより、異なる条件での体積推定結果を系統的に比較できるようになりました。

---

**文書作成者**: AI Assistant  
**最終更新**: 2026-02-04  
**バージョン**: 1.1

---

# アライメント処理の改善

**更新日**: 2026-02-04  
**対象スクリプト**: `21_calc_alignment.py`, `22_ecc_alignment.py`, `19_gausian_backsub.py`

---

## 📋 実施した改善項目

### 1. アライメント基準と引き算基準の分離

#### 背景
- 従来: アライメント計算と引き算に同じ基準画像（通常0番）を使用
- 問題: 時系列でドリフトがある場合、最初のフレームが適切な基準でない可能性

#### 実装内容
**21_calc_alignment.py**:
- パラメータを分離:
  - `alignment_reference_index`: アライメント計算の基準（デフォルト: 1200）
  - `subtraction_reference_index`: 引き算の基準（デフォルト: 0）
- JSON出力に両方のインデックスを保存（後方互換性あり）

**22_ecc_alignment.py**:
- JSONから両方のインデックスを読み込み
- 引き算基準を`subtraction_reference_index`から取得
- 後方互換性を維持（古いJSONにも対応）

#### メリット
- 中間フレーム（例: 1200番）でアライメント → ドリフト補正に有効
- 最初のフレーム（0番）で引き算 → バックグラウンド除去
- 柔軟な基準選択が可能

#### 使用例
```python
# 21_calc_alignment.py
alignment_reference_index=1200,   # 中間フレームでアライメント
subtraction_reference_index=0,    # 最初のフレームで引き算
```

---

### 2. PNG保存のオプション化と高速化

#### 背景
- 問題: 3168ファイル全てにPNG保存（dpi=300）で数時間かかる
- 原因: TIF保存とPNG保存が同じループ内で実行

#### 実装内容

**21_calc_alignment.py**:
- 処理を2フェーズに分離:
  - **フェーズ1**: TIF保存のみ（高速、数分）
  - **フェーズ2**: PNG保存（オプション、save_png=True時のみ）
- 新パラメータ:
  - `save_png=False`: PNG保存の有無（デフォルト: False）
  - `png_dpi=150`: 解像度（デフォルト: 150、軽量）
  - `png_sample_interval=1`: サンプリング間隔（1=全保存、10=10枚に1枚）

**19_gausian_backsub.py**:
- 同様の構造に変更
- `main()`関数にPNG保存パラメータを追加

**22_ecc_alignment.py**:
- PNG保存を分離してオプション化
- サンプリング機能を追加

#### メリット
- 通常実行（save_png=False）: 数分で完了
- 確認用（png_sample_interval=10）: 数十分
- 全PNG保存時も効率的

#### 使用例
```python
# 高速モード（推奨）
save_png=False  # PNG不要 → 数分

# 確認用
save_png=True, png_sample_interval=10  # 10枚に1枚 → 数十分

# 全保存
save_png=True, png_dpi=150, png_sample_interval=1  # → 数時間
```

---

### 3. ECCアルゴリズムの精度向上

#### 背景
- ECC (Enhanced Correlation Coefficient) の収束基準を調整
- 相関値 ~0.97 を観測

#### 実装内容
収束基準（criteria）を変更:
```python
# 変更前
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50000, 1e-6)

# 変更後（バランス型）
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100000, 1e-7)
```

- 最大反復回数: 50,000 → 100,000（2倍）
- 収束閾値(epsilon): 1e-6 → 1e-7（10倍厳しく）

#### 期待される効果
- アライメント精度の向上
- 相関値の安定化
- 計算時間: 約1.5〜2倍（許容範囲）

---

### 4. uint8正規化の改善

#### 背景
- 従来の実装: 各画像ごとにmin-max正規化
- 問題:
  - 画像間で値の対応が取れない
  - 外れ値1つで全体のスケールが変わる
  - アライメント精度に悪影響

#### 従来の実装
```python
def to_uint8(img):
    img_min = np.min(img)  # 画像ごとに異なる
    img_max = np.max(img)  # 画像ごとに異なる
    normalized = (img - img_min) / (img_max - img_min)
    return (normalized * 255).astype(np.uint8)
```

#### 新しい実装
```python
def to_uint8(img, vmin=-5.0, vmax=2.0):
    """
    固定範囲でuint8に変換（アライメント用）
    
    Parameters:
    -----------
    vmin, vmax : float
        クリッピング範囲（位相画像のrad値）
        デフォルト: -5.0 ~ 2.0 rad（実測値域）
    """
    # クリッピング（外れ値除去）
    clipped = np.clip(img, vmin, vmax)
    
    # 0-255に正規化
    normalized = (clipped - vmin) / (vmax - vmin)
    
    return (normalized * 255).astype(np.uint8)
```

#### 重要な考察

**255ギリギリまで使うメリットは実はない**:
- ECCは勾配ベースのアルゴリズム
- 重要なのは**画像間での一貫性**
- 絶対値よりも**相対的な変化**が重要
- むしろ固定範囲での一貫した変換が重要

**ユーザーの状況**:
- 画像内に外れ値がある
- 画像間のばらつきは小さい
- 位相画像の実測値域: **-5 ~ 2 rad**

#### メリット
1. **一貫性**: 全画像で同じピクセル値が同じuint8値に
2. **外れ値耐性**: 異常値が自動的にクリッピング
3. **精度向上**: 画像間の対応が正確に取れる
4. **安定性**: ノイズや外れ値の影響が小さい

#### 注意事項
- 19_gaussian_backsub.pyの範囲（-1.1 ~ 1.5）はバックグラウンドピークの探索範囲
- 画像全体の値域とは異なる
- 実際の値域（-5 ~ 2 rad）を使用することが重要

---

### 5. フォルダ作成の確実化

#### 背景
- 稀にフォルダ作成が失敗し、ファイル保存時にエラーが発生

#### 実装内容
- 初期のフォルダ作成時にエラーハンドリングを追加
- 各ファイル保存の直前でもフォルダの存在を確認
- 成功/失敗のメッセージを表示

```python
# 初期フォルダ作成
try:
    os.makedirs(aligned_folder, exist_ok=True)
    os.makedirs(subtracted_folder, exist_ok=True)
    os.makedirs(colored_folder, exist_ok=True)
    print(f"\n✅ フォルダ作成成功:")
except Exception as e:
    print(f"\n❌ エラー: フォルダ作成に失敗しました")
    return None

# ファイル保存前の確認
os.makedirs(subtracted_folder, exist_ok=True)  # 念のため再確認
io.imsave(subtracted_path, subtracted.astype(np.float32))
```

---

### 6. 日本語フォント対応

#### 背景
- matplotlibのデフォルトフォント（DejaVu Sans）に日本語が含まれない
- PNG保存時に警告が発生

#### 実装内容
```python
# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'  # Windowsの場合
plt.rcParams['axes.unicode_minus'] = False  # マイナス記号の文字化け対策
```

---

## 処理フロー（改善後）

### 21_calc_alignment.py（ステップ1）
```
1. ファイルリスト取得
2. アライメント基準画像（1200番）を読み込み
3. 全画像を1200番に合わせてアライメント計算
4. アライメント済み画像をTIF保存
5. 変換行列をJSON保存（両方のインデックス含む）
6. 引き算基準画像（0番、アライメント済み）を取得
7. 全画像から0番を引き算
8. 差分TIFを保存
9. [オプション] PNG保存（save_png=True時のみ）
```

### 22_ecc_alignment.py（ステップ2）
```
1. JSONから変換行列と両方のインデックスを読み込み
2. ターゲット画像に変換行列を適用
3. アライメント済み画像をTIF保存
4. 引き算基準画像を取得
5. 差分TIFを保存
6. [オプション] PNG保存（save_png=True時のみ）
```

---

## 実行例

### 基本的な実行
```python
# 21_calc_alignment.py
transforms = step1_calculate_and_subtract_fixed(
    empty_channel_folder=r"F:\251212\ph_1\Pos10\10_7\bg_corr",
    output_folder=r"F:\251212\ph_1\Pos10\10_7\aligned",
    output_json=r"F:\251212\ph_1\Pos10\10_7\alignment_transforms.json",
    alignment_reference_index=1200,   # アライメント基準
    subtraction_reference_index=0,    # 引き算基準
    method='ecc',
    save_png=False  # 高速モード
)
```

### PNG保存が必要な場合
```python
# 確認用（10枚に1枚）
save_png=True,
png_sample_interval=10,
png_dpi=150
```

---

## まとめ

この一連の改善により、以下が実現されました：

1. ✅ **柔軟性**: アライメント基準と引き算基準の独立指定
2. ✅ **高速化**: PNG保存のオプション化で数分に短縮
3. ✅ **精度向上**: ECC基準の最適化 + 固定範囲正規化
4. ✅ **一貫性**: 全画像で統一された変換
5. ✅ **外れ値耐性**: クリッピングによる安定化
6. ✅ **安定性**: フォルダ作成の確実化
7. ✅ **日本語対応**: フォント設定の追加

これにより、位相画像のアライメント処理が高速化・高精度化され、外れ値にも強くなりました。

---

**記録者**: AI Assistant  
**更新日**: 2026-02-04







