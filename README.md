# QPI_Omni: 定量位相イメージング解析パイプライン

## 概要

本プロジェクトは、オフアクシスホログラフィック顕微鏡による定量位相イメージング（Quantitative Phase Imaging: QPI）データを解析し、細胞の形態・体積・密度変化を時系列で追跡するためのPythonベースの画像解析パイプラインです。

### 主要機能

- **ホログラム位相再構成**: オフアクシスホログラム画像から位相情報を再構成
- **背景補正**: ガウスフィッティングによる背景シグナルの自動除去
- **画像アライメント**: ECC法による時系列画像の位置合わせ
- **細胞セグメンテーション**: Omniposeを用いた高精度な細胞領域検出
- **体積・密度解析**: Rod-shaped細胞の3D形状再構成と密度計算
- **Total Mass追跡**: 細胞の総質量を時系列で定量（2025-12-23追加）
- **バッチ解析**: 複数パラメータの網羅的自動実行（2025-12-23追加）

## 環境構築

### 必要なソフトウェア

- Python 3.10
- Anaconda/Miniconda
- Fiji (ImageJ) - ROI作成用

### 環境セットアップ

```bash
# conda環境の作成
conda env create -f env.yml
conda activate omnipose

# 必要に応じて追加パッケージをインストール
pip install cellpose-omni tifffile scikit-image opencv-python natsort
```

## ディレクトリ構成

```
QPI_Omni/
├── scripts/           # 解析スクリプト
│   ├── 09_single_reconstruction.py    # 単一画像の位相再構成
│   ├── 10_batch_reconstruction.py     # バッチ位相再構成
│   ├── 19_gausian_backsub.py          # 背景補正
│   ├── 21_calc_alignment.py           # アライメント計算
│   ├── 22_ecc_alignment.py            # アライメント適用
│   ├── 04_diff_from_first.py          # 差分画像生成
│   ├── 06_seg.ny_to_masks.py          # マスク変換
│   ├── 08_train.py                    # Omniposeトレーニング
│   ├── 07_segmentaion.py              # セグメンテーション
│   ├── 12_vertical_flip.py            # 上下反転（Augmentation）
│   ├── 26_horizontal_flip.py          # 左右反転（Augmentation）
│   ├── 24_elip_volume.py              # 体積・密度・Total Mass解析
│   ├── 28_batch_analysis.py           # バッチ解析（全パラメータ組み合わせ）
│   ├── config.yaml                    # 設定ファイル
│   └── qpi.py                         # QPIユーティリティ
├── docs/              # ドキュメント
│   ├── README.md                      # ドキュメント索引
│   ├── workflows/                     # 詳細なワークフローログ
│   └── notes/                         # クイックリファレンス・メモ
├── results/           # 解析結果出力
│   └── inbox/         # 一時ファイル
├── env.yml            # conda環境定義
├── README.md          # 本ファイル
└── PIPELINE.md        # 詳細なパイプライン説明

外部（外付けHDD）:
└── 生データ/          # 顕微鏡からのホログラム画像シークエンス
    ├── Pos0/          # 背景（空チャネル）
    ├── Pos1/          # 測定位置1
    ├── Pos2/          # 測定位置2
    └── ...
```

## クイックスタート

### 1. 位相再構成画像の確認

まず、1枚の画像でオフアクシス中心座標を確認します：

```bash
cd scripts
python 09_single_reconstruction.py
```

スクリプト内の`offaxis_center`パラメータを調整し、適切な位相画像が得られるようにします。

### 2. バッチ処理による位相再構成

```bash
python 10_batch_reconstruction.py
```

全ての位置（Pos1〜PosN）の画像シークエンスを一括で位相再構成します。

### 3. 背景補正

```bash
python 19_gausian_backsub.py
```

ガウスフィッティングにより背景シグナルを0に合わせます。

### 4. アライメントと差分処理

```bash
# Step 1: アライメント計算（空チャネル）
python 21_calc_alignment.py

# Step 2: アライメント適用（細胞入りチャネル）
python 22_ecc_alignment.py

# Step 3: 1枚目との差分
python 04_diff_from_first.py
```

### 5. セグメンテーションとトレーニング

詳細は[PIPELINE.md](PIPELINE.md)の「ステップ7」を参照してください。

### 6. 体積・密度解析

Fijiで作成したROIとResults.csvを用いて解析します：

```bash
# 単体実行
python 24_elip_volume.py

# バッチ実行（全パラメータ組み合わせ）
python 28_batch_analysis.py
```

**新機能（2025-12-23追加）**:
- Total Mass計算（質量濃度 × 体積）
- Feret径ベースのマスク生成（楕円近似に加えて）
- サブピクセルサンプリング（1×1, 5×5, 10×10）で精度向上
- 時系列プロット自動生成（Volume, RI, Total Mass）

## ドキュメント

### パイプライン詳細
完全な解析パイプラインの詳細は **[PIPELINE.md](PIPELINE.md)** を参照してください。

### ワークフローログ
開発・解析作業の詳細ログは **[docs/](docs/)** ディレクトリを参照：
- **[docs/workflows/](docs/workflows/)**: 日付別の詳細ワークフローログ
- **[docs/notes/](docs/notes/)**: クイックリファレンス・メモ

**最新ログ**: [2025-12-23: 時系列解析機能の大幅拡張](docs/workflows/2025-12-23_timeseries_total_mass.md)

## 設定

- `scripts/config.yaml`: データパスやOmniposeパラメータの設定
- 各スクリプト内の`OFFAXIS_CENTER`、`WAVELENGTH`等のパラメータ

## トラブルシューティング

### よくある問題

1. **位相再構成がうまくいかない**
   - `offaxis_center`の座標を再確認してください
   - `09_single_reconstruction.py`でFFT画像を表示し、ピークの位置を確認

2. **アライメントがずれる**
   - 空チャネル（Pos0）の画像が正しく選択されているか確認
   - ECCのiterationsやtolerance等のパラメータを調整

3. **セグメンテーションの精度が低い**
   - トレーニングデータを追加（3000細胞以上推奨）
   - Data Augmentationを確実に実行
   - `normalize=False`が設定されていることを確認

## 引用

このプロジェクトでは以下のツールを使用しています：

- **Omnipose**: Cutler, K. J. et al. "Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation." Nature Methods (2022).
- **Cellpose**: Stringer, C. et al. "Cellpose: a generalist algorithm for cellular segmentation." Nature Methods (2021).

## ライセンス

[プロジェクトのライセンスを記載]

## 連絡先

[連絡先情報を記載]
