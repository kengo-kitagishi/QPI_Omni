# QPI解析ワークフロー完全ログ（2025-12-23）

## 概要

このログは、QPI（Quantitative Phase Imaging）解析パイプラインの開発・改良作業の完全な記録です。
上から順に実行することで、同じ結果を再現できます。

## 目次

1. [初期要求：時系列データのプロット](#1-初期要求時系列データのプロット)
2. [Total Mass計算の追加](#2-total-mass計算の追加)
3. [プロット表示の改良](#3-プロット表示の改良)
4. [Feret径ベースのマスク生成](#4-feret径ベースのマスク生成)
5. [サブピクセルサンプリング](#5-サブピクセルサンプリング)
6. [出力ディレクトリの整理](#6-出力ディレクトリの整理)
7. [スクリプトの統合](#7-スクリプトの統合)
8. [バッチ解析の実装](#8-バッチ解析の実装)
9. [ImageJでのROI処理](#9-imagejでのroi処理)
10. [最終実行](#10-最終実行)

---

## 1. 初期要求：時系列データのプロット

### 要求
体積変化、平均密度変化、Total Mass変化を時系列でプロットしたい。

### 実装

#### 1.1 Total Mass計算を`24_elip_volume.py`に追加

**場所**: `24_elip_volume.py`の`process_roi`メソッド

**追加コード**:
```python
# Total massを計算
# Total mass [pg] = Σ(concentration [mg/ml] × pixel_volume [µm³])
# 単位変換: 1 mg/ml = 1 pg/µm³
pixel_volumes = thickness_um[mask] * pixel_area_um2  # 各ピクセルの体積 [µm³]
total_mass_pg = np.sum(concentration_map[mask] * pixel_volumes)  # [pg]

print(f"  Calculating total mass...")
print(f"    Total mass: {total_mass_pg:.2f} pg")
print(f"    Mean concentration: {concentration_map[mask].mean():.2f} mg/ml")
```

**statsに追加**:
```python
'total_mass_pg': float(total_mass_pg),
```

#### 1.2 時系列プロット機能を`27_timeseries_plot.py`に追加

**追加したプロット**:
- Volume vs Time
- Mean RI vs Time
- Total Mass vs Time（新規）

**レイアウト**: 3行3列（9パネル）→ 後で3行1列に変更

---

## 2. Total Mass計算の追加

### 計算式

```python
Total mass [pg] = Σ(concentration [mg/ml] × thickness [µm] × pixel_area [µm²])
```

### 単位変換
- 1 mg/ml = 1 mg/cm³ = 1 pg/µm³

---

## 3. プロット表示の改良

### 3.1 エラーバーとトレンドラインの削除

**要求**: ±SDとトレンドラインは不要

**変更内容**: `27_timeseries_plot.py`の全プロット

**Before**:
```python
ax.errorbar(time_centers, means, yerr=stds, 
           fmt='o-', capsize=5, label='Mean ± SD')
# トレンドライン
z = np.polyfit(time_centers, means, 1)
ax.plot(time_centers, p(time_centers), "--", label=f'Trend: {z[0]:.2f}')
```

**After**:
```python
ax.plot(time_centers, means, 
       'o-', linewidth=2.5, label='Mean', zorder=10)
```

### 3.2 プロットレイアウトの変更

**要求**: 横並び→縦並びに変更

**Before**: 1行3列（24×6）
```python
fig = plt.figure(figsize=(24, 6))
gs = GridSpec(1, 3, figure=fig)
```

**After**: 3行1列（14×14）
```python
fig = plt.figure(figsize=(14, 14))
gs = GridSpec(3, 1, figure=fig)
```

### 3.3 マーカーの削除

**要求**: 点が大きすぎるので線のみで表示

**変更**: `'o-'` → `'-'`

```python
# Before
ax.plot(time_centers, means, 'o-', markersize=10, ...)

# After
ax.plot(time_centers, means, '-', linewidth=2.5, ...)
```

### 3.4 Mean RIのY軸範囲設定

**要求**: min 1.3, max 1.5

**実装**: 後に1.36-1.40に調整
```python
ax2.set_ylim(1.36, 1.40)
```

---

## 4. Feret径ベースのマスク生成

### 背景
現在は楕円（Major/Minor）近似のみ。Feret径を使った近似も追加したい。

### 実装

#### 4.1 `create_rod_zstack_map`メソッドに`shape_type`パラメータ追加

**場所**: `24_elip_volume.py`

```python
def create_rod_zstack_map(self, roi_params, image_shape, shape_type='ellipse'):
    """
    Parameters:
    -----------
    shape_type : str
        'ellipse': Major/Minor/Angleを使用（楕円近似）
        'feret': Feret/MinFeret/FeretAngleを使用（Feret径近似）
    """
    if shape_type == 'feret':
        length = roi_params.get('Feret', roi_params.get('Major'))
        width = roi_params.get('MinFeret', roi_params.get('Minor'))
        angle = roi_params.get('FeretAngle', roi_params.get('Angle'))
    else:  # 'ellipse'
        length = roi_params['Major']
        width = roi_params['Minor']
        angle = roi_params['Angle']
```

#### 4.2 パラメータ設定追加

```python
SHAPE_TYPE = 'ellipse'  # または 'feret'
```

### ImageJ設定（Feret径を使用する場合）

```
Analyze > Set Measurements... 
☑ Feret's Diameter
```

---

## 5. サブピクセルサンプリング

### 問題
マスク端のRIの値が大きすぎる。ピクセル中心での二値判定では精度が不足。

### 解決策
ピクセル内でN×Nサブピクセルサンプリングを行い、平均厚みを計算。

### 実装

#### 5.1 `create_rod_zstack_map`メソッドに`subpixel_sampling`パラメータ追加

```python
def create_rod_zstack_map(self, roi_params, image_shape, shape_type='ellipse', 
                          subpixel_sampling=5):
    """
    Parameters:
    -----------
    subpixel_sampling : int
        ピクセル内のサブサンプリング数（N×N）
        1: ピクセル中心のみ（高速だが精度低）
        5: 5×5サブピクセル（推奨）
        10: 10×10サブピクセル（高精度だが遅い）
    """
```

#### 5.2 サブピクセルループの実装

```python
# サブピクセルオフセットを計算
if subpixel_sampling > 1:
    offsets = np.linspace(0.5/subpixel_sampling, 
                         1 - 0.5/subpixel_sampling, 
                         subpixel_sampling) - 0.5
else:
    offsets = np.array([0.0])

# 各ピクセルについてサブピクセルごとに計算
for py in range(img_height):
    for px in range(img_width):
        thickness_sum = 0.0
        valid_subpixels = 0
        
        for dy_offset in offsets:
            for dx_offset in offsets:
                # サブピクセル中心座標
                px_sub = px + 0.5 + dx_offset
                py_sub = py + 0.5 + dy_offset
                
                # 厚みを計算...
                if inside_roi:
                    thickness_sum += thickness
                    valid_subpixels += 1
        
        # ピクセル内の平均厚み
        if valid_subpixels > 0:
            zstack_map[py, px] = thickness_sum / valid_subpixels
```

#### 5.3 パラメータ設定追加

```python
SUBPIXEL_SAMPLING = 5  # 1, 5, 10
```

### 計算時間の目安

| Sampling | 計算時間 | 精度 |
|----------|---------|------|
| 1×1 | 1× | 低 |
| 5×5 | 25× | 高 |
| 10×10 | 100× | 最高 |

---

## 6. 出力ディレクトリの整理

### 要求
使用しているパラメータ（shape_type, subpixel_sampling）を出力ディレクトリ名に含めたい。

### 実装

#### 6.1 `24_elip_volume.py`の出力ディレクトリ命名

```python
# 出力ディレクトリ（パラメータに応じた名前）
dir_suffix = f"{self.shape_type}_subpixel{self.subpixel_sampling}"
self.output_dir = f"timeseries_density_output_{dir_suffix}"
```

**例**:
- `timeseries_density_output_ellipse_subpixel5/`
- `timeseries_density_output_feret_subpixel10/`

#### 6.2 `27_timeseries_plot.py`の対応

```python
# 24_elip_volume.pyで使用したパラメータを指定
SHAPE_TYPE = 'feret'
SUBPIXEL_SAMPLING = 5

# パラメータに基づいてディレクトリを自動生成
dir_suffix = f"{SHAPE_TYPE}_subpixel{SUBPIXEL_SAMPLING}"
BASE_OUTPUT_DIR = f"timeseries_density_output_{dir_suffix}"
CSV_DIR = os.path.join(BASE_OUTPUT_DIR, "csv_data")
OUTPUT_DIR = f"timeseries_plots_{dir_suffix}"
```

---

## 7. スクリプトの統合

### 要求
24と27を統合したほうが見やすく、実行回数も減らせる。

### 実装

#### 7.1 時系列プロット機能を`24_elip_volume.py`に統合

**場所**: `24_elip_volume.py`のメイン実行部分の最後

```python
mapper.process_all_rois(max_rois=MAX_ROIS)

# ===== 時系列プロット生成 =====
print(f"\n{'#'*80}")
print(f"# Generating time-series plots...")
print(f"{'#'*80}\n")

# all_rois_summary.csvを読み込み
summary_path = os.path.join(mapper.output_dir, "all_rois_summary.csv")

if os.path.exists(summary_path):
    df_summary = pd.read_csv(summary_path)
    df_summary['time_h'] = df_summary['frame_number'] / 12.0
    
    # 時系列プロット用の出力ディレクトリ
    plot_output_dir = f"timeseries_plots_{dir_suffix}"
    os.makedirs(plot_output_dir, exist_ok=True)
    
    # 時間ビン作成と統計計算
    time_bin_h = 1.0
    # ... (プロット生成コード)
    
    # 3つのプロット（Volume, RI, Mass）を縦並びで作成
    fig = plt.figure(figsize=(14, 14))
    gs = GridSpec(3, 1, figure=fig, hspace=0.3, wspace=0.3)
    # ... (各プロットを作成)
```

#### 7.2 `27_timeseries_plot.py`の削除

```bash
rm c:\Users\QPI\Documents\QPI_omni\scripts\27_timeseries_plot.py
```

### ワークフロー（統合後）

**以前**（2ステップ）:
```bash
python 24_elip_volume.py
python 27_timeseries_plot.py
```

**現在**（1ステップ）:
```bash
python 24_elip_volume.py
```

---

## 8. バッチ解析の実装

### 要求
全パラメータ組み合わせを網羅的に実行したい：
- SHAPE_TYPE: `ellipse`, `feret`
- SUBPIXEL_SAMPLING: `1`, `5`, `10`

合計: 6通り

### 実装

#### 8.1 `28_batch_analysis.py`作成

```python
#!/usr/bin/env python3
"""
バッチ解析：全パラメータ組み合わせを網羅的に実行

実行する組み合わせ:
1. ellipse + subpixel1
2. ellipse + subpixel5
3. ellipse + subpixel10
4. feret + subpixel1
5. feret + subpixel5
6. feret + subpixel10
"""

def run_analysis(shape_type, subpixel_sampling, results_csv, image_directory, 
                 wavelength_nm, n_medium, pixel_size_um, alpha_ri, max_rois):
    """指定されたパラメータで解析を実行"""
    
    # グローバル変数として渡す
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
    }
    
    # 24_elip_volume.pyを読み込んで実行
    with open('24_elip_volume.py', 'r', encoding='utf-8') as f:
        code = f.read()
    
    code = code.replace('if __name__ == "__main__":', 'if True:')
    exec(code, globals_dict)
```

#### 8.2 パラメータ設定

```python
# 共通パラメータ
RESULTS_CSV = r"C:\Users\QPI\Desktop\...\Results_enlarge_interpolate.csv"
IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\...\subtracted"

WAVELENGTH_NM = 663
N_MEDIUM = 1.333
PIXEL_SIZE_UM = 0.348
ALPHA_RI = 0.00018

MAX_ROIS = 5  # テスト実行（Noneで全ROI）

# パラメータの組み合わせ
SHAPE_TYPES = ['ellipse', 'feret']
SUBPIXEL_SAMPLINGS = [1, 5, 10]
```

#### 8.3 実行

```bash
python 28_batch_analysis.py
```

### 出力ディレクトリ（全6通り）

```
timeseries_density_output_ellipse_subpixel1/
timeseries_density_output_ellipse_subpixel5/
timeseries_density_output_ellipse_subpixel10/
timeseries_density_output_feret_subpixel1/
timeseries_density_output_feret_subpixel5/
timeseries_density_output_feret_subpixel10/

timeseries_plots_ellipse_subpixel1/
timeseries_plots_ellipse_subpixel5/
timeseries_plots_ellipse_subpixel10/
timeseries_plots_feret_subpixel1/
timeseries_plots_feret_subpixel5/
timeseries_plots_feret_subpixel10/
```

---

## 9. ImageJでのROI処理

### 目的
ROIを1ピクセル縮小＋スムージング処理を行い、より正確な形状を得る。

### ImageJ Macroスクリプト

```javascript
// ROI Manager内の全てのROIを処理
count = roiManager("count");

for (i = 0; i < count; i++) {
    roiManager("select", i);
    
    // 1ピクセル縮める
    run("Enlarge...", "enlarge=-1");
    
    // 滑らかにする（閉曲線に適している）
    run("Interpolate", "interval=1 smooth");
    
    // ROI Managerを更新
    roiManager("update");
}

roiManager("deselect");
```

### 実行手順

1. ImageJでROI Managerを開く
2. ROIsetを読み込む
3. 上記マクロを実行
4. `Measure` → `Results_enlarge_interpolate.csv`として保存

---

## 10. 最終実行

### 10.1 パラメータ設定

**`28_batch_analysis.py`**:
```python
RESULTS_CSV = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted\inference_out\Results_enlarge_interpolate.csv"
IMAGE_DIRECTORY = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"

MAX_ROIS = 5  # テスト: 5 ROIs、本番: None
SHAPE_TYPES = ['ellipse', 'feret']
SUBPIXEL_SAMPLINGS = [1, 5, 10]
```

### 10.2 実行

```bash
# テスト実行（5 ROIsのみ）
python 28_batch_analysis.py

# 本番実行（全ROIs）
# MAX_ROIS = None に変更してから実行
python 28_batch_analysis.py
```

### 10.3 出力確認

```bash
# ディレクトリ構造確認
ls -la timeseries_density_output_*/
ls -la timeseries_plots_*/

# 可視化画像確認
ls timeseries_density_output_feret_subpixel5/visualizations/

# 時系列プロット確認
ls timeseries_plots_*/timeseries_volume_ri_mass.png
```

---

## ファイル一覧

### 主要スクリプト

1. **`24_elip_volume.py`** (902行)
   - TimeSeriesDensityMapperクラス
   - ROI解析（RI、体積、Total Mass）
   - 時系列プロット自動生成
   - Ellipse/Feret近似
   - サブピクセルサンプリング

2. **`28_batch_analysis.py`** (183行)
   - バッチ解析スクリプト
   - 全パラメータ組み合わせを自動実行

3. **`27_timeseries_plot.py`** (削除済み)
   - 24_elip_volume.pyに統合されたため削除

### 設定ファイル

- **`config.yaml`**: 設定ファイル（存在する場合）

### 出力ディレクトリ

- `timeseries_density_output_{shape_type}_subpixel{N}/`
  - `density_tiff/`: TIFF画像（RI、濃度、厚み、位相）
  - `visualizations/`: 可視化画像（8パネル）
  - `csv_data/`: CSV データ（パラメータ、ピクセルデータ）
  - `all_rois_summary.csv`: 全ROIのサマリー

- `timeseries_plots_{shape_type}_subpixel{N}/`
  - `timeseries_volume_ri_mass.png`: 時系列プロット（3パネル）

---

## パラメータ一覧

### QPI実験パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `WAVELENGTH_NM` | 663 | レーザー波長（nm） |
| `N_MEDIUM` | 1.333 | 培地の屈折率 |
| `PIXEL_SIZE_UM` | 0.348 | ピクセルサイズ（µm）<br>507×507再構成画像用 |
| `ALPHA_RI` | 0.00018 | 比屈折率増分（ml/mg） |

### 解析パラメータ

| パラメータ | 選択肢 | 説明 |
|-----------|--------|------|
| `SHAPE_TYPE` | `'ellipse'` or `'feret'` | ROI形状近似方法 |
| `SUBPIXEL_SAMPLING` | `1`, `5`, `10` | サブピクセルサンプリング数 |
| `MAX_ROIS` | `None` or 整数 | 処理するROI数（Noneで全ROI） |

---

## 計算式まとめ

### 屈折率（RI）計算

```
n_sample = n_medium + (φ × λ) / (2π × thickness)
```

- `φ`: 位相差（radians）
- `λ`: 波長（µm）
- `thickness`: 厚み（µm）

### 質量濃度計算

```
C [mg/ml] = (RI - RI_medium) / α
```

- `α`: 比屈折率増分（ml/mg）

### Total Mass計算

```
Total mass [pg] = Σ(C [mg/ml] × V [µm³])
```

- 単位変換: 1 mg/ml = 1 pg/µm³

### 体積計算

```
Volume [µm³] = Σ(thickness [µm] × pixel_area [µm²])
```

---

## トラブルシューティング

### 問題1: マージが失敗する（0/N ROIs have volume/RI data）

**原因**: `roi_index`のマッチング失敗

**解決策**: 
- `Results.csv`のインデックスと`parameters.csv`の`roi_index`が一致しているか確認
- `frame_number`でのマッチングも試す

### 問題2: 計算が遅い

**原因**: サブピクセルサンプリング数が大きい

**解決策**:
- `SUBPIXEL_SAMPLING = 1` でテスト
- 必要に応じて `5` または `10` に増やす

### 問題3: メモリ不足

**原因**: 大量のROI + 高解像度画像

**解決策**:
- `MAX_ROIS` を設定して分割実行
- サブピクセルサンプリングを減らす

---

## 今後の改善案

1. **並列処理**
   - マルチプロセッシングで複数ROIを同時処理
   - バッチ解析の並列化

2. **GPUアクセラレーション**
   - サブピクセルサンプリングをGPUで高速化

3. **3D可視化**
   - 厚みマップの3D表示

4. **統計解析の拡充**
   - 相関分析
   - 時系列の統計的有意性検定

5. **GUI化**
   - パラメータ設定用のGUI
   - リアルタイムプレビュー

---

## 参考文献

### QPI関連
- 位相差顕微鏡の原理
- 屈折率と質量濃度の関係

### ImageJ
- ROI Manager の使い方
- Macro 言語リファレンス

### Python ライブラリ
- NumPy, Pandas, Matplotlib
- scikit-image, SciPy
- tifffile, Pillow

---

## バージョン履歴

| 日付 | バージョン | 変更内容 |
|------|-----------|---------|
| 2025-12-23 | 1.0 | 初版作成 |
| | | - Total Mass計算追加 |
| | | - 時系列プロット統合 |
| | | - Feret径対応 |
| | | - サブピクセルサンプリング |
| | | - バッチ解析機能 |

---

## 連絡先・サポート

質問や問題がある場合は、このログを参照して再現手順を確認してください。

---

## ライセンス

このワークフローは研究目的で使用されています。

---

**End of Log**

最終更新: 2025-12-23 21:30 JST

