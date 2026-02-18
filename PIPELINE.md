# QPI画像解析パイプライン詳細マニュアル

本ドキュメントでは、ホログラム画像から細胞の体積・密度情報を取得するまでの全工程を詳細に説明します。

## 目次

1. [パイプライン全体フロー](#パイプライン全体フロー)
2. [ステップ1: ホログラム位相再構成](#ステップ1-ホログラム位相再構成)
3. [ステップ2: 領域Crop](#ステップ2-領域crop)
4. [ステップ3: Background Subtraction](#ステップ3-background-subtraction)
5. [ステップ4: Alignment計算](#ステップ4-alignment計算)
6. [ステップ5: Alignment適用](#ステップ5-alignment適用)
7. [ステップ6: Diff from First](#ステップ6-diff-from-first)
8. [ステップ7: Omnipose Training準備](#ステップ7-omnipose-training準備)
9. [ステップ8: Segmentation](#ステップ8-segmentation)
10. [ステップ9: ROI作成（Fiji）](#ステップ9-roi作成fiji)
11. [ステップ10: 体積・密度解析](#ステップ10-体積密度解析)
12. [重要な注意事項](#重要な注意事項)
13. [その他のスクリプト](#その他のスクリプト)

---

## パイプライン全体フロー

```
┌──────────────────────────────────────────────────────────────┐
│  外付けHDD上のホログラム画像シークエンス                         │
│  (Pos0: 背景, Pos1-N: 測定位置)                                │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ1: ホログラム位相再構成                                 │
│  - スクリプト: 09_single_reconstruction.py (確認用)             │
│               10_batch_reconstruction.py (バッチ処理)          │
│  - 出力: output_phase/*.tif                                   │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ2: 領域Crop（手動）                                    │
│  - サイズ: 約150×50ピクセル                                    │
│  - 注: 現状はスクリプト内で手動指定                              │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ3: Background Subtraction (ガウス法)                  │
│  - スクリプト: 19_gaussian_backsub.py                          │
│  - 出力: *_bg_corr.tif                                        │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ4: Alignment計算 (ECC法)                              │
│  - スクリプト: 21_calc_alignment.py                           │
│  - 入力: 空チャネルの背景補正済み画像                      │
│  - 出力: alignment_transforms.json                            │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ5: Alignment適用                                      │
│  - スクリプト: 22_ecc_alignment.py                            │
│  - 入力: 細胞入りチャネル + JSON                                │
│  - 出力: *_aligned.tif                                        │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ6: Diff from First                                   │
│  - スクリプト: 04_diff_from_first.py                          │
│  - 処理: 1枚目(空チャネル)を全画像から減算                       │
│  - 出力: チャネル構造除去後の細胞のみの位相情報                   │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ7: Omnipose Training準備                              │
│  7-1. Omnipose GUIで正解データ作成 (~3000細胞)                  │
│  7-2. マスク変換 (06_seg_npy_to_masks.py)                      │
│  7-3. Data Augmentation (12, 26)                             │
│  7-4. モデルTraining (08_train.py)                            │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ8: Segmentation                                      │
│  - スクリプト: 07_segmentation.py                              │
│  - 出力: *_masks.tif, *_binary.tif                           │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ9: ROI作成 (Fiji)                                     │
│  - ツール: Fiji (ImageJ)                                      │
│  - 出力: ROI.zip, Results.csv                                │
└──────────────────┬───────────────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ステップ10: 体積・密度解析                                     │
│  - スクリプト: 24_ellipse_volume.py                              │
│  - 出力: 密度マップ、体積・密度時系列データ                       │
└──────────────────────────────────────────────────────────────┘
```

---

## ステップ1: ホログラム位相再構成

### 目的
外付けHDD上のオフアクシスホログラム画像から位相情報を再構成します。

### 使用スクリプト
- **確認用**: `scripts/09_single_reconstruction.py` - 1枚の画像で動作確認
- **バッチ処理**: `scripts/10_batch_reconstruction.py` - 複数位置の一括処理

### 入力
- 外付けハードディスク上のホログラム画像シークエンス (.tif)
- ディレクトリ構造例:
  ```
  外付けHDD/
  ├── Pos0/          # 背景
  │   ├── img_000000000_Default_000.tif
  │   ├── img_000000001_Default_000.tif
  │   └── ...
  ├── Pos1/          # 測定位置1
  ├── Pos2/          # 測定位置2
  └── ...
  ```

### 出力
- 位相再構成画像: `output_phase/*_phase.tif` (float32形式)
- カラーマップ画像: `output_colormap/*_colormap.png`
ここでcolormapのパラメータを考えた方がいいし、一緒にcolormapを保存すると時間がかかりすぎる感じがする。
### 主要パラメータ

#### 光学系パラメータ
```python
WAVELENGTH = 663e-9        # 波長 (663 nm)
NA = 0.95                  # 開口数
PIXELSIZE = 3.45e-6 / 40   # ピクセルサイズ (カメラピクセル / 対物レンズ倍率)
OFFAXIS_CENTER = (845, 772) # オフアクシス中心座標 ★要確認★
```

#### パラメータ確認方法

1. **`09_single_reconstruction.py`を開く**
2. **FFT画像を表示してピーク位置を確認**:
   ```python
   cb = CursorVisualizer(np.log(np.abs(img_fft)))
   cb.run()
   ```
3. **カーソルを動かしてオフアクシスピークの座標を記録**
4. **`OFFAXIS_CENTER`を更新**

### 実行手順

#### 1枚目の確認（09_single_reconstruction.py）

```bash
cd scripts
python 09_single_reconstruction.py
```

スクリプト内で以下を編集:
```python
# 画像パスを指定
path = r"F:\250611_kk\ph_1\Pos1\img_000000001_Default_000.tif"
path_bg = r"F:\250611_kk\ph_1\Pos0\img_000000001_Default_000.tif"

# オフアクシス中心を確認・設定
offaxis_center = (845, 772)  # FFTで確認した座標
```

確認事項:
- FFT画像でオフアクシスピークが円内に収まっているか
- 位相画像が適切に再構成されているか
- 背景差分後の位相値が適切な範囲か（-0.1〜数rad程度）

#### バッチ処理（10_batch_reconstruction.py）

全ての測定位置を一括処理します:

```bash
python 10_batch_reconstruction.py
```

スクリプト内で以下を編集:
```python
# ベースディレクトリ
BASE_DIR = r"F:\250611_kk\ph_1"
BG_DIR = os.path.join(BASE_DIR, "Pos0")  # 背景画像

# 処理範囲を指定
for pos_idx in range(1, 27):  # Pos1〜Pos26
    pos_name = f"Pos{pos_idx}"
    TARGET_DIR = os.path.join(BASE_DIR, pos_name)
    # ...
```

出力:
- 各Posフォルダ内に`output_phase/`と`output_colormap/`が作成されます
- 位相画像: `output_phase/img_*_phase.tif`

### 注意事項

1. **OFFAXIS_CENTERは実験条件ごとに異なります**
   - 光学系の調整により変化
   - 必ず最初に確認すること

2. **背景画像（Pos0）が必須**
   - 各時刻に対応する背景画像が必要
   - ファイル名が一致している必要がある

3. **Crop処理について**
   - スクリプト内でコメントアウトされているCrop処理がある
   - 必要に応じて有効化（例: `img = img[8:2056,416:2464]`）

### 将来の改善予定

- **OFFAXIS_CENTER等のパラメータをYAMLで一括管理**
- **Pos指定を自動化**
- **再構成画像の命名規則を統一**

---

## ステップ2: 領域Crop

### 目的
解析対象領域を絞り込む。

### サイズ
約**150×50ピクセル**（細胞が収まる最小サイズ）

### 実施方法

現状では、各スクリプト内で手動で指定します。

例（10_batch_reconstruction.py内）:
```python
# Crop処理を有効化
angle_nobg = angle - angle_bg
angle_nobg = angle_nobg[y_start:y_end, x_start:x_end]  # 範囲を指定
```

具体的な座標例:
```python
# 左チャネル
angle_nobg = angle_nobg[1:507, 1:253]

# 右チャネル  
angle_nobg = angle_nobg[1:507, 254:507]
```

### 将来の改善予定
- 自動的にチャネル領域を検出してCrop
- 設定ファイルで座標を一括管理

---

## ステップ3: Background Subtraction

### 目的
ヒストグラムのガウスフィッティングにより、背景（培地）のシグナルを0に合わせ、細胞とデバイスのシグナルのみを取り出します。

### 使用スクリプト
`scripts/19_gaussian_backsub.py`

### 入力
- Crop済みの位相再構成画像 (.tif)

### 出力
- 背景補正済み画像: `*_bg_corr.tif` (float32)
- ヒストグラムプロット: `*_histogram.png`
- 補正前後比較: `*_comparison.png`

### 主要パラメータ

```python
minPhase = -1.1       # ピーク検出の下限（rad）
hist_min = -1.1       # ヒストグラムの最小値（rad）
hist_max = 1.5        # ヒストグラムの最大値（rad）
n_bins = 512          # ヒストグラムのビン数（少ないほど滑らか）
smooth_window = 20    # スムージングのウィンドウサイズ
```

### 処理の流れ

1. **ヒストグラム作成**: 位相値の分布を固定範囲（-1.1〜1.5 rad）で取得
2. **スムージング**: 2回の移動平均フィルタでノイズ除去
3. **ピーク検出**: -1.1 rad以上の範囲で最大値を探索
4. **ガウスフィッティング**: ピーク周辺±300ビンでガウス関数をフィット
5. **補正**: ガウスの平均値を0にシフト

### 実行手順

```bash
cd scripts
python 19_gaussian_backsub.py
```

スクリプト内で入力・出力フォルダを指定:
```python
# 入力フォルダと出力フォルダの設定
input_folder = Path(r"C:\Users\QPI\Desktop\align_demo")
output_folder = input_folder / "bg_corr"
```

### 出力例

**ヒストグラム例**:
```
処理中: img_000000100_phase.tif
画像範囲: -0.950 ~ 1.234 rad
ピーク位置: -0.123 rad
ガウスフィット - 平均: -0.1145 rad
              標準偏差: 0.0892 rad
補正値: 0.1145 rad
補正後の範囲: -0.835 ~ 1.349 rad
保存完了: img_000000100_phase_bg_corr.tif
```

### パラメータ調整のヒント

- **ピークが検出されない場合**: `minPhase`を小さくする
- **ガウスフィットが失敗する場合**: `n_bins`や`smooth_window`を調整
- **補正が過剰/不足の場合**: `hist_min`/`hist_max`の範囲を見直す
ここのパラメータはどうすればいいのか考える。
---

## ステップ4: Alignment計算

### 目的
時系列画像間のドリフト（位置ずれ）を補正するため、空チャネル（細胞なし）の画像を使ってアライメント変換行列を計算します。

### 使用スクリプト
`scripts/21_calc_alignment.py`

### 入力
- 空チャネル（Pos0）の背景補正済み画像フォルダ

### 出力
- **アライメント変換行列JSON**: `alignment_transforms.json`
- アライメント済み画像: `aligned/*_aligned.tif`
- 差分画像: `subtracted/*_subtracted.tif`
- カラーマップ画像: `colored/*_colored.png`

### 主要パラメータ

```python
reference_index = 0        # 基準画像のインデックス（通常は最初の画像）
method = 'ecc'             # アライメント手法（ECC法を推奨）
vmin = -0.1                # カラーマップの最小値
vmax = 1.7                 # カラーマップの最大値
cmap = 'RdBu_r'           # カラーマップ
```

### アライメント手法

#### ECC (Enhanced Correlation Coefficient)
- OpenCVの`cv2.findTransformECC()`を使用
- 高精度な並進変換を計算
- 収束条件: `(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5000, 1e-6)`

#### Phase Correlation（代替手法）
- scikit-imageの`registration.phase_cross_correlation()`を使用
- サブピクセル精度でシフト量を計算

### 実行手順

```bash
cd scripts
python 21_calc_alignment.py
```

スクリプト内で設定:
```python
transforms = step1_calculate_and_subtract_fixed(
    empty_channel_folder=r"C:\Users\QPI\Desktop\align_demo\empty_channel\bg_corr",
    output_folder=r"C:\Users\QPI\Desktop\align_demo\empty_channel_aligned",
    output_json=r"C:\Users\QPI\Desktop\align_demo\alignment_transforms.json",
    reference_index=0,
    method='ecc',
    vmin=-0.1,
    vmax=1.7,
    cmap='RdBu_r'
)
```

### JSONファイルの構造

```json
{
  "reference_index": 0,
  "reference_filename": "empty0001_bg_corr.tif",
  "method": "ecc",
  "total_files": 1000,
  "successful_transforms": 998,
  "transforms": [
    {
      "index": 0,
      "filename": "empty0001_bg_corr.tif",
      "warp_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
      "shift_y": 0.0,
      "shift_x": 0.0,
      "correlation": 1.0
    },
    {
      "index": 1,
      "filename": "empty0002_bg_corr.tif",
      "warp_matrix": [[1.0, 0.0, -0.15], [0.0, 1.0, 0.23]],
      "shift_y": 0.23,
      "shift_x": -0.15,
      "correlation": 0.9987
    },
    ...
  ]
}
```

### 出力の確認

スクリプトは処理中に以下の情報を表示します:
```
シフト量統計:
  Y: 平均=0.12px, 標準偏差=0.45px, 範囲=[-1.23, 2.34]
  X: 平均=-0.08px, 標準偏差=0.38px, 範囲=[-1.45, 1.89]
```

### 注意事項

1. **空チャネルの画像を必ず使用**
   - 細胞が入っていると正しくアライメントできません
   - チャネル構造など固定された構造が必要

2. **重複ファイルのチェック**
   - スクリプトは自動的に重複ファイルを検出・スキップします

3. **相関値の確認**
   - `correlation`が低い（<0.95など）画像は要注意

---

## ステップ5: Alignment適用

### 目的
ステップ4で計算した変換行列を、細胞入りチャネルの画像に適用します。

### 使用スクリプト
`scripts/22_ecc_alignment.py`

### 入力
- 細胞入りチャネルの背景補正済み画像フォルダ
- ステップ4で生成した`alignment_transforms.json`

### 出力
- アライメント済み画像: `aligned/*_aligned.tif`
- 差分画像: `subtracted/*_subtracted.tif`
- カラーマップ画像: `colored/*_colored.png`

### ファイルマッチング

スクリプトは**ファイル名の数字部分（下4桁）**でマッチングします。

例:
- 空チャネル: `empty0001_bg_corr.tif`
- 細胞チャネル: `subtracted_by_maskmean_float320001_bg_corr.tif`
- → 数字 `0001` でマッチ

### 実行手順

```bash
cd scripts
python 22_ecc_alignment.py
```

スクリプト内で設定:
```python
count = step2_apply_by_filename_number(
    target_folder=r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr",
    json_path=r"C:\Users\QPI\Desktop\align_demo\alignment_transforms.json",
    output_folder=r"C:\Users\QPI\Desktop\align_demo\from_outputphase\aligned",
    vmin=-0.1,
    vmax=1.7,
    cmap='RdBu_r'
)
```

### 処理の流れ

1. **JSONから変換行列を読み込み**
2. **ファイル名の数字部分を抽出**
3. **マッチングを確認**
   ```
   ✅ マッチング例（最初の5個）:
     subtracted_0001_bg_corr.tif
       ↓ 数字0001でマッチ
     empty0001_bg_corr.tif
       シフト: Y=0.23, X=-0.15
   ```
4. **変換行列を適用** (cv2.warpAffine)
5. **差分計算** (基準画像との差分)
6. **カラーマップ生成**

### マッチング失敗時の対処

マッチングできないファイルがある場合:
```
❌ 変換行列が見つからないファイル（最初の10個）:
  subtracted_0999_bg_corr.tif (数字: 0999)
```

対処法:
- ファイル名の命名規則を確認
- 空チャネルと細胞チャネルのファイル数が一致するか確認

---

## ステップ6: Diff from First

### 目的
アライメント済み画像シークエンスの1枚目（空チャネル）を全ての画像から減算し、チャネル構造のシグナルを除去して細胞のみの位相情報を抽出します。

### 使用スクリプト
`scripts/04_diff_from_first.py`

### 入力
- アライメント済み画像シークエンス (`*_aligned.tif`)

### 出力
- 差分画像: `diff_from_first/*.tif`
- カラーマップ画像: `diff_colormap/*.tif` (オプション)

### 処理の流れ

```python
# 1枚目を基準として読み込む
ref_img = tifffile.imread(image_paths[0]).astype(np.float32)

# 2枚目以降の差分を計算
for i, path in enumerate(image_paths[1:], start=1):
    img = tifffile.imread(path).astype(np.float32)
    diff = img - ref_img  # 差分計算
    tifffile.imwrite(output_path, diff)
```

### 実行手順

```bash
cd scripts
python 04_diff_from_first.py
```

スクリプト内で設定:
```python
# 入力ディレクトリ（アライメント済み画像）
image_dir = r"F:\250611_kk\ph_1\Pos3\output_phase\crop_150_300\subtracted_by_maskmean_float32"
aligned_dir = image_dir  # または特定のalignedフォルダ

# 出力ディレクトリ
output_diff_dir = os.path.join(aligned_dir, "diff_from_first")
```

### カラーマップ生成（オプション）

差分画像を視覚化する場合:
```python
vmin, vmax = -0.5, 2.0  # 表示範囲を指定

# JETカラーマップを適用
diff_clipped = np.clip(diff, vmin, vmax)
diff_norm = ((diff_clipped - vmin) / (vmax - vmin) * 255).astype(np.uint8)
color_mapped = cv2.applyColorMap(diff_norm, cv2.COLORMAP_JET)
```

### 結果の確認

```
✅ 差分画像を保存しました: F:\250611_kk\ph_1\Pos3\...\diff_from_first
✅ カラーマップ画像を保存しました（範囲固定: -0.5〜2.0）
```

### 注意事項

1. **1枚目は空チャネルであること**
   - アライメント処理で1枚目が基準になっている前提
   - 細胞が入っていると正しく差分が取れません

2. **位相のラッピングに注意**
   - 大きな位相差がある場合は2πの不定性に注意

---

## ステップ7: Omnipose Training準備

このステップでは、細胞セグメンテーション用のOmniposeモデルをトレーニングします。

### 7-1. 正解データ作成（Omnipose GUI）

#### 目的
約3000細胞分の正解セグメンテーションデータを作成します。

#### 使用ツール
Omnipose GUI（Cellpose GUIのOmnipose版）

#### 手順

1. **Omnipose GUIの起動**
   ```bash
   conda activate omnipose
   python -m cellpose
   ```

2. **画像の読み込み**
   - `File` → `Load Images` で差分画像を読み込み
   - 複数枚の画像を選択

3. **セグメンテーション**
   - 手動で細胞領域を描画、またはモデルで予測後に修正
   - 各細胞に異なるラベル番号が割り当てられる

4. **保存**
   - `File` → `Save masks` で保存
   - 出力: `*_seg.npy` (正解マスク + メタデータ)

#### Tips

- **効率的な作業のために**:
  - まず少数の画像で作業に慣れる
  - プリトレーニングモデルで予測→手動修正の方が早い
  - キーボードショートカットを活用

- **品質管理**:
  - 細胞の境界を正確に
  - 重なった細胞も個別にラベリング
  - 小さすぎる/大きすぎるオブジェクトは除外

### 7-2. マスク変換

#### 目的
Omnipose GUIで作成した`*_seg.npy`を、トレーニングに必要な`*_masks.tif`に変換します。

#### 使用スクリプト
`scripts/06_seg_npy_to_masks.py`

#### 実行手順

```bash
cd scripts
python 06_seg_npy_to_masks.py
```

スクリプト内で設定:
```python
# 指定フォルダ直下の *_seg.npy から *_masks.tif を生成
convert_seg_to_masks(r"C:\Users\QPI\Desktop\train")
```

#### 処理の流れ

1. `*_seg.npy`を読み込み
2. Omnipose API (`io.save_masks`) で`*_cp_masks.tif`を生成
3. `*_cp_masks.tif` → `*_masks.tif` にリネーム

#### 出力例
```
[OK] img_000000100_phase_masks.tif
[OK] img_000000200_phase_masks.tif
...
Done.
```

### 7-3. Data Augmentation

#### 目的
トレーニングデータを増やし、モデルの汎化性能を向上させます。

#### 使用スクリプト
- `scripts/12_vertical_flip.py` - 上下反転
- `scripts/26_horizontal_flip.py` - 左右反転

#### 実行手順

```bash
# 上下反転
python 12_vertical_flip.py

# 左右反転  
python 26_horizontal_flip.py
```

各スクリプト内で設定:
```python
# 対象フォルダ
folder = r"C:\Users\QPI\Desktop\verti_flip_train"
```

#### 処理内容

**上下反転**:
```python
flipped = np.flipud(img)
out_path = os.path.join(folder, f"flippedud_{filename}")
```

**左右反転**:
```python
flipped = np.fliplr(img)
out_path = os.path.join(folder, f"flippedlr_{filename}")
```

#### 注意事項

1. **画像とマスクの両方を反転**
   - `img_000000100_phase.tif` → `flippedud_img_000000100_phase.tif`
   - `img_000000100_phase_masks.tif` → `flippedud_img_000000100_phase_masks.tif`

2. **命名規則を統一**
   - プレフィックスに`flippedud_`や`flippedlr_`を付ける
   - マスクファイルの対応関係を維持

### 7-4. モデルTraining

#### 目的
準備したデータを使ってOmniposeモデルをトレーニングします。

#### 使用スクリプト
`scripts/08_train.py`

#### 主要パラメータ

```python
train_dir = r"C:\Users\QPI\Desktop\verti_flip_train"  # トレーニングデータ

# モデル設定
use_gpu = True
nchan = 1              # グレースケール
nclasses = 3           # Omniposeのクラス数
diameter = 20          # 細胞の典型的な直径（ピクセル）

# トレーニング設定
learning_rate = 0.0001
batch_size = 2
save_every = 100       # 100エポックごとに保存
n_epochs = 3000        # 総エポック数
crop_size = (32, 96)   # クロップサイズ (t, y, x)

# ★重要★
normalize = False      # 正規化OFF（前処理済みのため）
rescale = False        # リスケールOFF
```

#### 実行手順

```bash
cd scripts
python 08_train.py
```

#### トレーニングの流れ

1. **ファイルペアの収集**
   ```python
   image_paths = sorted(glob.glob(os.path.join(train_dir, "*.tif")))
   # *_masks.tif 以外の画像ファイル
   image_paths = [p for p in image_paths if not p.endswith("_masks.tif")]
   
   # 対応するマスクを探す
   mask_paths = [p.replace(".tif", "_masks.tif") for p in image_paths]
   ```

2. **データ読み込みと型変換**
   ```python
   imgs = [tifffile.imread(p).astype(np.float32) for p in image_paths]
   masks = [tifffile.imread(p).astype(np.int32) for p in mask_paths]
   ```

3. **モデル作成**
   ```python
   model = models.CellposeModel(
       gpu=use_gpu,
       pretrained_model=None,  # またはプリトレーニングモデルのパス
       omni=True,
       nchan=nchan,
       nclasses=nclasses
   )
   ```

4. **トレーニング実行**
   ```python
   model.train(
       train_data=imgs,
       train_labels=masks,
       train_links=train_links,     # [None] * len(masks)
       train_files=train_files,     # ファイル名リスト
       channels=None,
       normalize=False,              # ★正規化OFF★
       save_path=save_dir,
       save_every=save_every,
       learning_rate=learning_rate,
       min_train_masks=1,
       n_epochs=n_epochs,
       batch_size=batch_size,
       rescale=False
   )
   ```

#### 出力

モデルは以下の形式で保存されます:
```
save_dir/models/
└── cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_2_omni_model_2025_11_05_19_14_41.656097
```

#### トレーニングの監視

- エポックごとにlossが表示される
- 100エポックごとに中間モデルが保存される
- GPUメモリ使用量を監視（batch_sizeを調整）

#### トラブルシューティング

**メモリ不足エラー**:
- `batch_size`を1に減らす
- `crop_size`を小さくする

**正規化の警告**:
- **必ず`normalize=False`に設定**
- 前処理で既に位相画像は適切な範囲に調整済み

**収束しない**:
- `learning_rate`を調整（0.0001〜0.001）
- トレーニングデータの品質を確認

---

## ステップ8: Segmentation

### 目的
トレーニングしたOmniposeモデルを使って、差分画像シークエンスから細胞領域を検出します。

### 使用スクリプト
`scripts/07_segmentation.py`

### 入力
- 差分画像シークエンス（Crop済み、細胞のみの位相情報）
- トレーニング済みOmniposeモデル

### 出力
- セグメンテーションマスク: `*_masks.tif` (uint16)
- 二値化輪郭画像: `*_binary.tif` (uint8)
- オーバーレイ画像: `*_overlay.tif` (RGB)

### 主要パラメータ

```python
# 入力設定
indir = r"C:\Users\QPI\Desktop\align_demo\from_outputphase\bg_corr\subtracted"
outdir = os.path.join(indir, "inference_out")

# モデルパス
model_path = r"C:\Users\QPI\Desktop\verti_flip_train\omni_model\models\cellpose_..."

# 推論設定
USE_GPU = True
NCHAN = 1
NCLASSES = 3

# Omnipose eval パラメータ
EVAL_PARAMS = dict(
    channels=None,
    channel_axis=None,
    diameter=30,               # 細胞の直径（ピクセル）
    normalize=False,           # ★正規化OFF★
    tile=False,                # タイル処理なし（一貫性のため）
    net_avg=True,              # 複数ネットの平均
    omni=True,
    flow_threshold=0.11,       # フロー閾値（低めで検出を確保）
    mask_threshold=0,          # マスク閾値
    min_size=10                # 最小サイズ（小さなゴミ除去）
)
```

### 実行手順

```bash
cd scripts
python 07_segmentation.py
```

### 処理の流れ

1. **ファイルリスト取得**
2. **モデル読み込み**
   ```python
   model = CellposeModel(
       gpu=USE_GPU,
       pretrained_model=model_path,
       omni=True,
       nchan=NCHAN,
       nclasses=NCLASSES,
       dim=2
   )
   ```

3. **画像ごとに推論**
   ```python
   for f in files:
       img = tifffile.imread(f)
       masks, flows, _ = model.eval([img], **EVAL_PARAMS)
       
       # マスク保存
       tifffile.imwrite(outpath, masks[0].astype(np.uint16))
   ```

4. **輪郭抽出**
   ```python
   m = masks[0]
   border = ((m != np.roll(m,  1, 0)) |
             (m != np.roll(m, -1, 0)) |
             (m != np.roll(m,  1, 1)) |
             (m != np.roll(m, -1, 1))) & (m > 0)
   
   # 白背景、黒線
   binary = np.where(border, 0, 255).astype(np.uint8)
   ```

### 出力例

```
[1/1000] img_000000100_phase.tif
[2/1000] img_000000101_phase.tif
...
=== Inference summary ===
Total files : 1000
Processed   : 998
Skipped     : 2 (no cells or kNN issue)
Errors      : 0
Saved results to: C:\Users\QPI\Desktop\...\inference_out
```

### トラブルシューティング

**kNN（k近傍法）エラー**:
```
⚠ Skipping due to ValueError: not enough points for kNN
```
対処:
- `flow_threshold`を下げる（0.11 → 0.08など）
- `mask_threshold`を下げる

**検出されない細胞が多い**:
- `diameter`を調整（実際の細胞サイズに合わせる）
- トレーニングデータの品質を確認
- エポック数を増やして再トレーニング

**検出が不安定**:
- `tile=True`を試す（ただし境界で不一致が生じる可能性）
- `net_avg=False`にして単一ネットで推論

---

## ステップ9: ROI作成（Fiji）

### 目的
セグメンテーション結果から細胞の輪郭情報（楕円パラメータ）を取得します。

### 使用ツール
**Fiji (ImageJ)**

### 入力
- 二値化輪郭画像: `*_binary.tif`

### 出力
- **ROI.zip**: ROI Manager で作成したROIセット
- **Results.csv**: 各ROIの楕円パラメータ（X, Y, Major, Minor, Angle等）

### 手順

#### 1. Fijiの起動と画像読み込み

```
File → Import → Image Sequence...
```

二値化画像のフォルダを選択。

#### 2. 輪郭からROIを作成

**手動の場合**:
```
- Elliptical Selection Tool を選択
- 各細胞を囲む楕円を描画
- ROI Manager (Analyze → Tools → ROI Manager) に追加 [t]
```

**自動の場合**:
```
Process → Binary → Options... 
  → Black background にチェック
Analyze → Analyze Particles...
  → Size: 100-Infinity (小さいゴミを除外)
  → Show: Outlines
  → Add to Manager にチェック
```

#### 3. 楕円フィッティング

各ROIに対して楕円をフィット:
```
ROI Manager → More → Fit Ellipse
```

#### 4. 平滑化（オプション）

より滑らかなマスクが必要な場合:
```
ROI Manager → More → Interpolate...
  → Interval: 1
```

#### 5. 測定

```
Analyze → Set Measurements...
  → Area, Centroid, Fit ellipse, Slice number にチェック

ROI Manager → Measure
```

#### 6. 保存

```
# ROIセットを保存
ROI Manager → More → Save... → ROI.zip

# 測定結果を保存
File → Save As → Results... → Results.csv
```

### Results.csvの構造

| カラム名 | 説明 |
|---------|------|
| Label | ROIのラベル（画像名含む） |
| Slice | スライス番号（フレーム番号に対応） |
| Area | 面積（ピクセル^2） |
| X | 中心X座標 |
| Y | 中心Y座標 |
| Major | 長軸（ピクセル） |
| Minor | 短軸（ピクセル） |
| Angle | 角度（度） |

例:
```csv
,Label,Slice,Area,X,Y,Major,Minor,Angle
0,img_000000100_phase_binary.tif:0001,100,1234.5,125.3,45.2,52.4,28.1,12.3
1,img_000000100_phase_binary.tif:0002,100,1156.2,180.7,50.1,48.9,26.5,85.7
...
```

### 注意事項

1. **スライス番号の確認**
   - Sliceカラムが正しくフレーム番号に対応しているか確認

2. **ROIの品質管理**
   - 明らかに間違ったROIは削除
   - 細胞が重なっている場合は個別にROIを作成

3. **平滑化の使い分け**
   - 体積計算には平滑化したマスクが有効
   - 形態解析には元の輪郭を使用

---

## ステップ10: 体積・密度解析

### 目的
ROIパラメータと差分画像から、Rod-shaped細胞の3D形状を再構成し、体積・密度情報を時系列で追跡します。

### 使用スクリプト
`scripts/24_ellipse_volume.py`

### 入力
- **Results.csv**: Fijiから出力されたROIパラメータ
- **差分画像シークエンス**: 細胞のみの位相情報

### 出力
- **密度マップTIFF**: `density_tiff/ROI_XXXX_Frame_YYYY_density.tif`
- **Z-stackマップTIFF**: `density_tiff/ROI_XXXX_Frame_YYYY_zstack.tif`
- **ピクセルデータCSV**: `csv_data/ROI_XXXX_Frame_YYYY_pixel_data.csv`
- **パラメータCSV**: `csv_data/ROI_XXXX_Frame_YYYY_parameters.csv`
- **可視化PNG**: `visualizations/ROI_XXXX_Frame_YYYY_visualization.png`
- **サマリーCSV**: `all_rois_summary.csv`

### 主要パラメータ

```python
# 入力設定
RESULTS_CSV = "/path/to/Results.csv"
IMAGE_DIRECTORY = "/path/to/subtracted_images"

# QPI実験パラメータ
WAVELENGTH_NM = 663       # レーザー波長 (nm)
N_MEDIUM = 1.333          # 培地の屈折率
PIXEL_SIZE_UM = 0.348     # ピクセルサイズ (µm)
ALPHA_RI = 0.00018        # 比屈折率増分 (ml/mg)

# 解析パラメータ
SHAPE_TYPE = 'ellipse'    # 'ellipse' or 'feret' (形状近似方法)
SUBPIXEL_SAMPLING = 5     # 1, 5, 10 (サブピクセルサンプリング数)

# 処理するROI数（テスト用）
MAX_ROIS = None           # None で全ROI処理
```

### 新機能（2025-12-23追加）

#### Total Mass計算
細胞の総質量を時系列で追跡：
```python
Total mass [pg] = Σ(concentration [mg/ml] × pixel_volume [µm³])
# 単位変換: 1 mg/ml = 1 pg/µm³
```

#### Feret径ベースのマスク生成
楕円近似（Major/Minor/Angle）に加えて、Feret径（Feret/MinFeret/FeretAngle）による形状近似に対応。より不規則な形状の細胞に適用可能。

#### サブピクセルサンプリング
マスク端での精度向上のため、ピクセル内でN×Nサブピクセルサンプリングを実施：
- `1×1`: ピクセル中心のみ（高速）
- `5×5`: 推奨設定（精度と速度のバランス）
- `10×10`: 最高精度（計算時間増）

#### 時系列プロット自動生成
解析完了後、以下の時系列プロットを自動生成：
- Volume vs Time
- Mean RI vs Time
- Total Mass vs Time

出力先: `timeseries_plots_{shape_type}_subpixel{N}/timeseries_volume_ri_mass.png`

### 処理アルゴリズム

#### 1. Rod-shaped細胞の3D形状モデル

細胞を「円柱 + 両端の半球」としてモデル化:

```
       半球    円柱部    半球
        _______________
       /               \
      |                 |
      |                 |
       \_______________ /
      
      ← r →← h →← r →
      
r: 半径 = Minor / 2
h: 円柱の高さ = Major - 2r
```

#### 2. Z-stackカウントマップ生成

各ピクセル(x, y)について、z方向の厚みを計算:

```python
# ローカル座標系に変換（回転を考慮）
x_local = dx * cos(angle) + dy * sin(angle)
y_local = -dx * sin(angle) + dy * cos(angle)

# y方向の距離
dist_from_axis = abs(y_local)

if dist_from_axis > r:
    thickness = 0  # 細胞外
else:
    # 円柱部分
    if abs(x_local) <= h / 2:
        z_half = sqrt(r^2 - y_local^2)
        thickness = 2 * z_half
    # 半球部分
    else:
        dist_sq = x_from_sphere_center^2 + y_local^2
        if dist_sq <= r^2:
            z_half_sphere = sqrt(r^2 - dist_sq)
            thickness = 2 * z_half_sphere
```

#### 3. 密度計算

```python
# 補正項を加える
correction_term = zstack_map * CORRECTION_FACTOR

# 密度 = (観測位相 + 補正) / Z-stack厚み
density_map = (image + correction_term) / zstack_map
```

**補正の意味**:
- `diff_from_first`で培地の情報を引きすぎた分を補正
- 培地の屈折率変化を考慮

### 実行手順

```bash
cd scripts
python 24_ellipse_volume.py
```

スクリプト内で設定:
```python
# パラメータ設定
RESULTS_CSV = "/mnt/user-data/uploads/Results.csv"
IMAGE_DIRECTORY = "/path/to/subtracted"
CORRECTION_FACTOR = 0.02
MAX_ROIS = 10  # テスト実行（Noneで全ROI）

# ワークフロー実行
mapper = TimeSeriesDensityMapper(
    results_csv=RESULTS_CSV,
    image_directory=IMAGE_DIRECTORY,
    correction_factor=CORRECTION_FACTOR
)

mapper.process_all_rois(max_rois=MAX_ROIS)
```

### 出力ファイルの詳細

#### pixel_data.csv
各ピクセルのデータ:
```csv
X_pixel,Y_pixel,Z_stack_count,Original_value,Correction,Density
125,45,28.5,0.234,0.570,0.0282
126,45,29.1,0.241,0.582,0.0283
...
```

#### parameters.csv
ROIのパラメータとサマリー:
```csv
roi_index,frame_number,X,Y,Major,Minor,Angle,zstack_max,zstack_mean,density_mean,density_min,density_max,num_pixels,correction_factor
0,100,125.3,45.2,52.4,28.1,12.3,28.5,24.2,0.0285,0.0180,0.0350,1234,0.02
```

#### all_rois_summary.csv
全ROIのサマリー:
```csv
roi_index,frame_number,X,Y,Major,Minor,Angle,zstack_max,zstack_mean,density_mean,num_pixels
0,100,125.3,45.2,52.4,28.1,12.3,28.5,24.2,0.0285,1234
1,100,180.7,50.1,48.9,26.5,85.7,26.8,22.9,0.0278,1156
...
```

### 可視化

スクリプトは各ROIについて6パネルの可視化図を生成:

1. **Original Image + ROI**: 元画像とROIアウトライン
2. **Z-stack Count Map**: Z方向の厚みマップ
3. **Medium Correction**: 培地補正項の分布
4. **Density Map**: 計算された密度マップ
5. **Density vs Z-stack**: 散布図
6. **Value Distribution**: ヒストグラム

### 体積・密度の時系列解析

サマリーCSVを使って時系列解析:

```python
import pandas as pd
import matplotlib.pyplot as plt

# データ読み込み
df = pd.read_csv("all_rois_summary.csv")

# 特定の細胞を追跡（例: roi_index=0の時系列）
cell_data = df[df['roi_index'] == 0].sort_values('frame_number')

# 体積の推定（楕円体として近似）
a = cell_data['Major'] / 2
b = cell_data['Minor'] / 2
volume = (4/3) * np.pi * a * b * b

# プロット
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.plot(cell_data['frame_number'], volume)
plt.xlabel('Frame')
plt.ylabel('Volume (pixels^3)')
plt.title('Volume over time')

plt.subplot(132)
plt.plot(cell_data['frame_number'], cell_data['density_mean'])
plt.xlabel('Frame')
plt.ylabel('Mean density')
plt.title('Density over time')

plt.subplot(133)
plt.plot(cell_data['frame_number'], volume * cell_data['density_mean'])
plt.xlabel('Frame')
plt.ylabel('Total mass (a.u.)')
plt.title('Total mass over time')

plt.tight_layout()
plt.show()
```

### パラメータ調整

#### CORRECTION_FACTOR
- **デフォルト**: 0.02
- **調整方法**: 理論的な密度値と比較
- 細胞の密度が培地より高いことを前提（正の値）

#### 理論密度の計算

培地の屈折率を n_medium、細胞の屈折率を n_cell とすると:

```
理論密度 = 1 + CORRECTION_FACTOR
```

文献値と比較して調整します。

### 注意事項

1. **フレーム番号のマッチング**
   - Results.csvの`Slice`と画像ファイル名のフレーム番号が一致していること
   - スクリプトは自動的にファイル名から番号を抽出

2. **ROIの品質**
   - 楕円フィッティングが適切に行われていること
   - 細胞の長軸・短軸が正しく検出されていること

3. **計算時間**
   - 各ROIの処理に数秒〜数十秒
   - 大量のROI（数千個）を処理する場合は時間がかかる

---

## 重要な注意事項

### ファイル命名規則

パイプライン全体で以下の命名規則を使用します:

| 処理段階 | ファイル名 |
|---------|-----------|
| 元画像 | `img_000000XXX_Default_000.tif` |
| 位相再構成 | `img_000000XXX_Default_000_phase.tif` |
| 背景補正 | `*_bg_corr.tif` |
| アライメント | `*_aligned.tif` |
| 差分画像 | `*_subtracted.tif` |
| セグメンテーションマスク | `*_masks.tif` |
| 輪郭画像 | `*_binary.tif` |

**注意**:
- Omniposeが出力する`*_cp_masks.tif`は自動的に`*_masks.tif`にリネームされます
- ファイル名の数字部分（下4桁）でマッチングが行われます

### データの流れ

```
外付けHDD
  └→ output_phase/        # ステップ1の出力
      └→ bg_corr/         # ステップ3の出力
          └→ aligned/     # ステップ5の出力
              └→ diff_from_first/  # ステップ6の出力
                  └→ inference_out/  # ステップ8の出力
```

### 将来の改善予定

#### パラメータ管理の自動化

現在はスクリプト内で個別に設定していますが、将来的には:

```yaml
# config.yaml での一括管理
experiments:
  exp_20250611:
    wavelength: 663e-9
    NA: 0.95
    pixelsize: 8.625e-8
    offaxis_center: [845, 772]
    positions:
      - Pos0: {role: background}
      - Pos1: {crop: [1, 507, 1, 253]}
      - Pos2: {crop: [1, 507, 254, 507]}
```

#### 自動化の拡張

- Crop処理の自動検出
- チャネル領域の自動認識
- パラメータの自動最適化

---

## その他のスクリプト

以下のスクリプトもリポジトリに含まれていますが、現在のメインパイプラインでは使用していません。
必要に応じて参照してください。

### 00_contours.py
輪郭抽出の代替実装

### 01_QPI_analysis.py
QPI解析の基本機能（qpi.pyと連携）

### 02_binary_outline.py
二値化と輪郭線抽出

### 02_first_backsub.py
別の背景除去手法（ガウス法以前の実装）

### 03_alignment.py
別のアライメント手法（ECC以前の実装）

### 05_2nd_backsub.py
2回目の背景除去（多段階処理用）

### 11_delete_mask_seg.py
不要なマスクファイルの削除ユーティリティ

### 13_collect_files.py
ファイル操作ユーティリティ

### 14_medium_diff.py
培地差分処理

### 15_fluo_segment.py
蛍光画像のセグメンテーション

### 16_phasecor_ali.py, 17_batch_phasecor_ali.py
位相相関アライメント

### 18_phase_to_colormap.py
位相画像をカラーマップに変換

### 20_test_alignment_methods.py
アライメント手法のテスト

### 23_plot_summary.py
図の作成ユーティリティ

### 25_roiset_from_zstack.py
Z-stackからROIセットを作成。屈折率（RI）マップと質量濃度マップの可視化にも対応。

### 28_batch_analysis.py
**バッチ解析スクリプト（2025-12-23追加）**

全パラメータ組み合わせ（`SHAPE_TYPE` × `SUBPIXEL_SAMPLING`）を網羅的に実行：
- 組み合わせ例: ellipse + subpixel1, ellipse + subpixel5, ..., feret + subpixel10
- 各組み合わせで`24_ellipse_volume.py`を自動実行
- 実行時間と成功/失敗を記録
- 出力ディレクトリもパラメータごとに自動命名

実行方法：
```bash
cd scripts
python 28_batch_analysis.py
```

### arrconv.py, subtract_mean.py
配列変換・平均値除去ユーティリティ

### qpi.py, qpi_analysis.py
QPI処理の基本関数群

### CursorVisualizer.py
インタラクティブな画像表示ツール

---

## 参考文献

1. **Omnipose**  
   Cutler, K. J. et al. "Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation." *Nature Methods* 19, 1438–1448 (2022).

2. **Cellpose**  
   Stringer, C. et al. "Cellpose: a generalist algorithm for cellular segmentation." *Nature Methods* 18, 100–106 (2021).

3. **ECC Alignment**  
   Evangelidis, G. D. & Psarakis, E. Z. "Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization." *IEEE Transactions on Pattern Analysis and Machine Intelligence* 30, 1858–1865 (2008).

4. **Quantitative Phase Imaging**  
   Park, Y., Depeursinge, C. & Popescu, G. "Quantitative phase imaging in biomedicine." *Nature Photonics* 12, 578–589 (2018).

---

## トラブルシューティング

### Q1: 位相再構成がうまくいかない

**症状**: 位相画像にノイズが多い、または構造が見えない

**対処法**:
1. `OFFAXIS_CENTER`を再確認
   - `09_single_reconstruction.py`でFFT画像を確認
   - ピークが円内に収まっているか確認
2. 背景画像が正しく選択されているか確認
3. `NA`や`WAVELENGTH`のパラメータを確認

### Q2: アライメントがずれる

**症状**: 差分画像に構造物が残る

**対処法**:
1. 空チャネル（Pos0）の画像を使用しているか確認
2. ECC法のiterations数を増やす
3. Phase Correlation法を試す

### Q3: セグメンテーションの精度が低い

**症状**: 細胞が検出されない、または過検出

**対処法**:
1. トレーニングデータを増やす（3000細胞以上推奨）
2. `diameter`パラメータを実際の細胞サイズに合わせる
3. `flow_threshold`と`mask_threshold`を調整
4. エポック数を増やして再トレーニング
5. **`normalize=False`が設定されていることを確認**

### Q4: ファイルマッチングが失敗する

**症状**: アライメント適用時に「マッチングしない」エラー

**対処法**:
1. ファイル名の命名規則を確認
2. 空チャネルと細胞チャネルで同じ数字部分があるか確認
3. スクリプト内の`extract_number_from_filename()`関数を確認

### Q5: 体積・密度計算で異常値が出る

**症状**: 密度が負の値、または極端に大きい

**対処法**:
1. `CORRECTION_FACTOR`を調整
2. 差分画像の品質を確認（ステップ6）
3. ROIの楕円フィッティングが正しいか確認
4. 位相のアンラッピングが正しいか確認

---

## サポート

質問やバグ報告は以下まで:
- GitHub Issues: [リンク]
- Email: [連絡先]

---

## 更新履歴

### 2025-12-23: 時系列解析機能の大幅拡張
- **Total Mass計算**: 細胞の総質量（pg）を時系列で追跡
- **Feret径対応**: 楕円近似に加えてFeret径ベースのマスク生成に対応
- **サブピクセルサンプリング**: マスク端の精度向上（1×1, 5×5, 10×10）
- **時系列プロット統合**: Volume, RI, Total Massの自動可視化
- **バッチ解析**: 全パラメータ組み合わせの自動実行（`28_batch_analysis.py`）
- **ドキュメント整備**: `docs/workflows/`, `docs/notes/` ディレクトリ追加

詳細: [`docs/workflows/2025-12-23_timeseries_total_mass.md`](docs/workflows/2025-12-23_timeseries_total_mass.md)

### 初版
- パイプライン基本機能の実装
- ホログラム再構成、アライメント、セグメンテーション、体積・密度解析

---

**最終更新**: 2025-12-23

