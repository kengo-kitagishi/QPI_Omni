---
name: Central Cell Track Figure 実装
overview: run_omnipose_chm_batch.py が作る `Pos*/output_phase/channels/crop_sub_rawraw/chXX/` とその直下 `inference_out/` の対応関係を前提に、raw と `*_masks.tif` を basename でペアリングし、各 frame で中心に最も近い代表細胞を選んで Overlay Strip・Contour Montage・Shape Trace を最初の完成物として実装する。lineage tree や厳密 tracking は扱わない。
todos:
  - id: task_00_layout
    content: スクリプト配置とCLIエントリ設計
    status: pending
  - id: task_01_io
    content: run_omnipose_chm_batch.py 構造のデータ探索とraw-maskペアリング
    status: pending
  - id: task_02_regions
    content: frameごとの領域特徴量抽出
    status: pending
  - id: task_03_seed
    content: 各frameで中心に最も近いmask選択
    status: pending
  - id: task_04_track
    content: frame独立のcentral-cell series構成
    status: pending
  - id: task_05_export
    content: track中間成果物の保存
    status: pending
  - id: task_06_geometry
    content: cropと回転整列の共通幾何処理
    status: pending
  - id: task_07_style
    content: figure共通スタイルとsave_figure統合
    status: pending
  - id: task_08_overlay
    content: Overlay Strip figure実装
    status: pending
  - id: task_09_contour
    content: Contour Montage figure実装
    status: pending
  - id: task_10_shape
    content: Shape Trace figure実装
    status: pending
  - id: task_11_test
    content: MVP検証と回帰用サンプル整備
    status: pending
  - id: task_12_ghost
    content: Ghost Contour Overlay追加
    status: pending
  - id: task_13_kymo_intensity
    content: Intensity Kymograph追加
    status: pending
  - id: task_14_kymo_mask
    content: Mask Kymograph追加
    status: pending
  - id: task_15_qc
    content: QC figureと終了理由可視化
    status: pending
  - id: task_16_docs
    content: 使用例と運用ドキュメント作成
    status: pending
---

# Central Cell Track Figure 実装タスク分解

## フェーズの考え方

最初の完成目標は以下の 3 点です。

- `run_omnipose_chm_batch.py` 由来の 1 ch 分の raw series と `inference_out/*_masks.tif` から frame ごとの `central-cell series` を再現可能に抽出できる
- `track_summary.csv` と `track_metadata.json` を保存できる
- `Overlay Strip`、`Contour Montage`、`Shape Trace` を `save_figure()` 経由で保存できる

これを `MVP` とし、kymograph や QC はその次のフェーズに回す。

## Phase 0: 土台作り

### Task 00. スクリプト配置と CLI エントリ設計

目的:

- 実装の置き場所を固定し、`Pos/ch` 単位の処理入口を明確にする

実装内容:

- 仮のメインスクリプトを `scripts/central_cell_track_figures.py` として定義
- 補助モジュール候補を決める
  - `scripts/central_cell_track_io.py`
  - `scripts/central_cell_track_selection.py`
  - `scripts/central_cell_track_geometry.py`
  - `scripts/central_cell_track_plotting.py`
- CLI 引数の骨格だけ先に作る
- 入力単位を以下のどちらかで扱う設計にする
  - `--indir /.../Pos3/output_phase/channels/crop_sub_rawraw/ch00`
  - `--pos-root /.../Pos3 --channel 0`

完了条件:

- `python scripts/central_cell_track_figures.py --help` が動く
- 必須引数、任意引数、figure 種別、`Pos/ch` 指定方法の骨格が見える

依存:

- なし

## Phase 1: 入力と中心細胞選択

### Task 01. run_omnipose_chm_batch.py 構造のデータ探索と raw-mask ペアリング

目的:

- `run_omnipose_chm_batch.py` が作る raw と `inference_out/*_masks.tif` の対応関係を明示的に解決し、frame series として安全に扱えるようにする

実装内容:

- `chXX` ディレクトリを入力として受け取る
- raw 画像は `chXX/*.tif` から収集する
- mask 画像は `chXX/inference_out/*_masks.tif` から収集する
- basename 対応でペアリングする
  - raw: `foo.tif`
  - mask: `inference_out/foo_masks.tif`
- `*_binary.tif` は補助出力として認識だけし、初版では利用しない
- `run_omnipose_chm_batch.py` の `NO_CELL_PIXEL_STREAK` により途中で mask が無い tail がありうることを許容する
- ペアリング結果を frame table として保持する
  - raw_path
  - mask_path
  - frame_name
  - frame_index
- `frame_index` はまずソート済みファイル順を使い、必要なら basename から数値抽出を拡張する
- mask は `(Y, X)` 単枚ラベル画像としてロードする
- binary mask が来ても label 化できるようにしておく

完了条件:

- `chXX` から raw と mask の対応表を作れる
- 欠損 mask や途中打ち切り channel があっても落ちない
- 各 frame の mask が label image として読み込める
- エラー時のメッセージが明確

依存:

- Task 00

### Task 02. frame ごとの領域特徴量抽出

目的:

- 各 frame の mask 候補から中心細胞選択と figure 生成に必要な特徴量を得る

実装内容:

- `regionprops_table` か `regionprops` で特徴量抽出
- 最低限の列を DataFrame 化
  - label
  - area
  - centroid
  - bbox
  - major/minor axis
  - orientation
  - eccentricity
  - solidity
- `touches_border` 判定追加
- `min_area` で候補除外
- 各 frame の画像中心 `(cx, cy)` を列として保持
- 各候補に対し `distance_to_center_px` を計算する

完了条件:

- 任意 frame について候補領域表を返せる
- border-touch と小領域除外が動く
- 中心距離が数値として確認できる

依存:

- Task 01

### Task 03. 各 frame で中心に最も近い mask 選択

目的:

- 各 frame における代表細胞を単純かつ頑健に選ぶ

実装内容:

- 各 frame 内候補から中心距離最小の領域を選ぶ
- 必要なら border penalty を追加
- 選ばれた mask の feature を record 化

選択ロジックの初版:

1. `min_area` を満たす候補だけ残す
2. `exclude_border=true` なら border-touch 候補を外す
3. 各候補について `distance_to_center_px = sqrt((cx - x)^2 + (cy - y)^2)` を計算
4. 最小の候補をその frame の代表細胞とする

補足:

- これはかなり簡単に実装できる
- 実質的には `regionprops` の centroid を取って `argmin` するだけ
- この方式では frame 間の同一性は保証しない
- ただし figure 用途では十分な可能性が高い

完了条件:

- 各 frame に採用 label が一意に決まる
- 候補なし frame を適切にスキップできる
- 選ばれた代表細胞の中心距離をログや CSV で確認できる

依存:

- Task 02

### Task 04. frame 独立の central-cell series 構成

目的:

- Task 03 の結果を、図生成に使える時系列レコードへまとめる

実装内容:

- frame 順に代表細胞 record を並べる
- 候補なし frame は `missing_cell` として記録する
- raw が欠ける frame は `missing_raw` として扱う
- mask がない tail は `unavailable_mask` として扱う
- confidence 低下時は `selection_flag` を記録する
- 将来 tracking を足したくなっても差し替えやすいよう、選択ロジックと series 構成は分離する

完了条件:

- `frame -> selected label or missing` が全 frame について得られる
- 欠損理由が `selection_flag` に残る
- 図生成がこの series だけで動く

依存:

- Task 03

### Task 05. track 中間成果物の保存

目的:

- track 結果を可視化とは独立に再利用可能にする

実装内容:

- `track_summary.csv`
- `track_metadata.json`
- `track_masks.tif`
- `track_crops.tif`
- run id / 入力パス / パラメータ / 中心細胞選択ルール情報の保存
- 入力元として以下を必ず記録
  - `pos_root`
  - `channel_dir`
  - `inference_out_dir`
  - raw file list
  - mask file list

完了条件:

- 1 回の実行で上記成果物が `results/central_cell_track/<run_id>/` に出る
- CSV と JSON だけ読めば figure 再生成条件がわかる

依存:

- Task 04

## Phase 2: 幾何共通処理

### Task 06. crop と回転整列の共通幾何処理

目的:

- 各 figure が同じ crop / 向き / スケールで描画されるようにする

実装内容:

- tight crop + margin
- fixed-size crop
- major axis に基づく回転整列
- contour の aligned 座標生成
- raw/mask 両方に同じ transform を適用

完了条件:

- 任意 frame について `crop`, `aligned_crop`, `contour` が取得できる
- 主要 figure がこの共通処理だけで描ける

依存:

- Task 05

## Phase 3: 図生成基盤

### Task 07. figure 共通スタイルと `save_figure()` 統合

目的:

- 図の見た目と保存方法を統一する

実装内容:

- `matplotlib` rcParams/preset
- 共通色、線幅、フォントサイズ、panel label 設定
- scale bar 描画ユーティリティ
- `save_figure()` 用 wrapper
- `manuscript`, `presentation`, `qc` preset

完了条件:

- figure 生成コードが個別に保存ロジックを持たない
- PNG/PDF/SVG を同じ API で保存できる

依存:

- Task 00

## Phase 4: MVP 図

### Task 08. Overlay Strip figure 実装

目的:

- raw 上に代表細胞輪郭を重ねた主力 figure を作る

実装内容:

- timepoint のサンプリング
- raw crop 横並び
- contour overlay
- time label
- scale bar

完了条件:

- raw があるときに publication 向けの strip figure が生成される
- raw が `chXX/*.tif` と `inference_out/*.tif` のペアリング前提で動く
- raw がない場合は明確に unavailable を返す

依存:

- Task 06
- Task 07

### Task 09. Contour Montage figure 実装

目的:

- 背景に依存せず形状変化を示す

実装内容:

- contour only / faint fill の 2 モード
- timepoint 横並び
- 共通スケール

完了条件:

- raw なしでも高品質 figure が生成される
- SVG/PDF 出力で輪郭がきれいに保たれる

依存:

- Task 06
- Task 07

### Task 10. Shape Trace figure 実装

目的:

- 形態指標の経時変化を定量表示する

実装内容:

- `track_summary.csv` を用いた時系列プロット
- デフォルト指標
  - area
  - major axis
  - minor axis
  - aspect ratio
- `missing_cell` や `selection_flag` の注記

完了条件:

- 1 コマンドで shape summary 図が出る
- 欠損 frame があっても破綻しない

依存:

- Task 05
- Task 07

### Task 11. MVP 検証と回帰用サンプル整備

目的:

- MVP の品質を固定し、次の機能追加で壊れないようにする

実装内容:

- 小さいサンプル `Pos/ch` を 1 セット選ぶ
- 中心代表細胞選択結果を目視確認
- 別細胞に切り替わっても用途上問題ないかを確認
- 3 つの figure を golden output と比較
- 最低限の pytest または smoke test を追加

完了条件:

- 実データ 1 例で MVP が最後まで通る
- 再実行で同じ成果物が得られる

依存:

- Task 08
- Task 09
- Task 10

## Phase 5: 拡張 figure

### Task 12. Ghost Contour Overlay 追加

目的:

- 1 パネルで形状変化を凝縮表示する

実装内容:

- aligned contour の重ね描き
- time gradient か alpha gradient を選択可能にする

完了条件:

- 古い輪郭と新しい輪郭の差が視覚的に判別できる

依存:

- Task 06
- Task 07

### Task 13. Intensity Kymograph 追加

目的:

- 長軸方向の signal 分布の時間変化を示す

実装内容:

- aligned raw crop の longitudinal resampling
- transverse averaging
- 2D heatmap 描画
- 共通 intensity scaling

完了条件:

- raw stack ありのときに再現可能な kymograph が出る
- 向きの揺れで模様が壊れにくい

依存:

- Task 06
- Task 07

### Task 14. Mask Kymograph 追加

目的:

- raw がなくても形態変化を kymograph 的に示す

実装内容:

- aligned mask から longitudinal position ごとの幅を計算
- `time x position` heatmap 描画

完了条件:

- raw なしでも kymograph に相当する図を出せる

依存:

- Task 06
- Task 07

## Phase 6: QC と運用

### Task 15. QC figure と終了理由可視化

目的:

- 中心代表細胞選択の妥当性確認を早くする

実装内容:

- 原画像全体上の centroid 列
- distance-to-center 時系列
- selection flag 一覧
- 欠損 frame 表示

完了条件:

- 選択が怪しいときに QC figure だけ見れば状況判断できる

依存:

- Task 04
- Task 07

### Task 16. 使用例と運用ドキュメント作成

目的:

- 後から迷わず再利用できるようにする

実装内容:

- 使い方の README 追加
- 実行例コマンド
- figure 種別一覧
- frame 独立選択であることを明記
- raw なし時の制約明記

完了条件:

- 他人が 1 回読めば動かせる
- `lineage tree` や厳密 tracking ではないことが明確

依存:

- Task 11 以降

## 推奨着手順

実装順は次の通り。

1. Task 00
2. Task 01
3. Task 02
4. Task 03
5. Task 04
6. Task 05
7. Task 07
8. Task 06
9. Task 08
10. Task 09
11. Task 10
12. Task 11
13. Task 12
14. Task 13
15. Task 14
16. Task 15
17. Task 16

## 実装上の重要判断

- `lineage` という語はコードやドキュメントで使わない
- 初版では frame 間 tracking をしない
- final figure は `matplotlib` で描く
- figure 保存は `save_figure()` を使う
- viewer は後回しにする
- 入力の正本は `run_omnipose_chm_batch.py` の出力構造とする
- `chXX` 直下 raw と `chXX/inference_out` の basename 対応を基本ルールにする

## MVP の Definition of Done

以下が満たされたら MVP 完了とする。

1. `chXX/inference_out/*_masks.tif` のみで `Contour Montage` と `Shape Trace` が出る
2. `chXX/*.tif` と対応 mask があれば `Overlay Strip` が出る
3. `track_summary.csv` と `track_metadata.json` が保存される
4. 各 frame の中心代表細胞選択が主要サンプルで妥当に見える
5. 図が `save_figure()` 経由で保存される
