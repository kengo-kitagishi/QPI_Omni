# Central Cell Track Figure 仕様書

## 1. 目的

`run_omnipose_chm_batch.py` が作る raw と `*_masks.tif` の対応関係を入力として、各 frame で画像中心に最も近い代表細胞を 1 つ選び、その系列から論文・発表向けの高品質 figure を再現可能に生成する。

本仕様の主眼は以下の 2 点とする。

- OmniSegger の lineage 依存可視化の一部を、`central-cell track` ベースで代替する
- interactive viewer より先に、静的で高品質な図生成を実現する

## 2. 用語

- `central-cell series`
  - 各 frame で画像中心に最も近い代表細胞を 1 つ選んだ系列
- `track record`
  - 各 frame における採用マスクと特徴量の記録

注意:

- 本仕様では `lineage tree` は扱わない
- 本仕様は厳密な tracking ではなく、frame ごとの中心代表細胞選択である
- 外部向けの図・説明では `lineage` ではなく `central cell series` または `single-cell representative series` と表記する

## 3. スコープ

### 3.1 含む

- `run_omnipose_chm_batch.py` 出力構造からの raw / mask 読み込み
- 各 frame における代表細胞 1 個の自動抽出
- raw 画像がある場合の overlay / kymograph 生成
- mask のみでも成立する figure 生成
- figure の PDF/SVG/PNG 出力
- 図保存時の `scripts/figure_logger.py` 利用

### 3.2 含まない

- true lineage tree
- mother / daughter 分岐の厳密推定
- multi-cell population 解析
- GUI viewer の実装
- manual correction GUI

## 4. 入力仕様

### 4.1 必須入力

- `chXX` ディレクトリ
  - 例: `Pos3/output_phase/channels/crop_sub_rawraw/ch00`
- `inference_out`
  - `chXX/inference_out/` にある `*_masks.tif`

`run_omnipose_chm_batch.py` の前提構造:

```text
Pos3/
  output_phase/
    channels/
      crop_sub_rawraw/
        ch00/
          frame001.tif
          frame002.tif
          ...
          inference_out/
            frame001_masks.tif
            frame001_binary.tif
            frame002_masks.tif
            frame002_binary.tif
            ...
```

raw と mask は basename で対応付ける。

- raw: `frame001.tif`
- mask: `inference_out/frame001_masks.tif`

### 4.2 任意入力

- `config.yml` または CLI 引数
  - pixel size
  - time interval
  - scale bar 長
  - figure preset

### 4.3 前提条件

- `*_masks.tif` は各 frame ごとに 1 細胞以上を含むとは限らない
- `run_omnipose_chm_batch.py` の `NO_CELL_PIXEL_STREAK` により channel の後半に mask が無いことがある
- raw がある場合は `chXX/*.tif` を使う
- raw が無い場合でも、mask 形状ベースの figure は生成可能
- time 間で label ID が維持されている必要はない

## 5. 中心細胞選択仕様

### 5.1 基本方針

各 frame を独立に扱い、その frame 内の候補マスクのうち画像中心 `(X/2, Y/2)` に最も近いものを採用する。

この方式は実装が単純で、figure 用途では十分である可能性が高い。一方で、時系列途中で別細胞へ切り替わる可能性は残るため、厳密な同一細胞 tracking とはみなさない。

### 5.2 frame 内候補の前処理

各 frame に対して以下を行う。

- binary mask なら connected components により label 化
- `regionprops` により特徴量算出
- 極小領域の除外
- border-touch mask の除外または低優先化

最低限使う特徴量:

- label id
- area
- centroid
- bbox
- major_axis_length
- minor_axis_length
- orientation
- eccentricity
- solidity

### 5.3 採用 mask の選択

各 frame において、以下の score が最小のマスクを選ぶ。

`score = d_center + border_penalty + small_area_penalty`

ここで:

- `d_center`: 画像中心と centroid の距離
- `border_penalty`: 枠接触時の固定加算
- `small_area_penalty`: `min_area` 未満に近い場合の加算

初版では単純化のため、実装上は以下でも可とする。

- `min_area` を満たす候補のうち、中心距離最小

### 5.4 例外処理

- 候補が 0 個ならその frame は `missing_cell`
- border-touch を除外した結果候補が 0 個なら、必要に応じて border-touch を再許可する
- 小さすぎる debris しか無い場合は `low_confidence`

### 5.5 将来拡張

必要になった場合のみ、以下を追加可能とする。

- overlap ベース tracking
- centroid continuity penalty
- split / merge の検出

## 6. データ出力仕様

### 6.1 中間成果物

`results/central_cell_track/<run_id>/` に以下を保存する。

- `track_summary.csv`
  - frame ごとの基本特徴量
- `track_masks.tif`
  - 採用 mask のみを残した stack
- `track_crops.tif`
  - crop stack
- `track_metadata.json`
  - 入力パス、パラメータ、中心細胞選択ルール

### 6.2 `track_summary.csv` の必須列

- `frame`
- `label`
- `tracked`
- `selection_flag`
- `centroid_y`
- `centroid_x`
- `area_px`
- `major_axis_px`
- `minor_axis_px`
- `orientation_rad`
- `eccentricity`
- `solidity`
- `bbox_ymin`
- `bbox_xmin`
- `bbox_ymax`
- `bbox_xmax`
- `distance_to_center_px`

raw がある場合は追加:

- `mean_intensity`
- `median_intensity`
- `integrated_intensity`

## 7. 幾何標準化

figure 品質を揃えるため、選択された代表細胞に対し以下を共通処理として持つ。

### 7.1 crop

- `tight crop`
  - bbox + margin
- `fixed-size crop`
  - 全 frame 共通サイズ

初版 figure は `fixed-size crop` を標準とする。

### 7.2 回転整列

- 各 frame の major axis に基づき回転
- 長軸を水平方向へ揃える
- 左右反転は初版では行わない

### 7.3 座標系

以下の 3 系統を持つ。

- original frame 座標
- crop 座標
- aligned crop 座標

kymograph と ghost contour では `aligned crop` を使用する。

## 8. figure テンプレート

優先度順に実装する。

### 8.0 図の基本方針

図は `QC 用` と `見せる用` を明確に分ける。

- `QC figure`
  - 目的は妥当性確認
  - 情報量優先
  - 美しさは二次的
- `Presentation / Manuscript figure`
  - 目的は結果提示
  - 見やすさと見た目を最優先
  - 不要な補助情報は載せない

見せる用の図では以下を原則とする。

- 余計な軸・目盛りは消す
- contour はベクタ線で描く
- raw のコントラストは frame 間で固定する
- crop サイズと向きは図内で統一する
- panel ごとの見た目のぶれをなくす

### 8.1 F1: Overlay Strip

目的:

- raw 画像上で代表細胞の輪郭変化を見せる

入力:

- track
- raw stack

表示:

- 複数 timepoint の crop を横並び
- 輪郭をベクタ線で重ねる
- time label
- scale bar

見た目仕様:

- 背景は raw grayscale
- contour 色は単色
  - manuscript: 黒または暖色系 1 色
  - presentation: 背景とのコントラストが強い色 1 色
- 各 panel は正方形または固定アスペクト
- 画像の外枠は消す
- time label は左上
- scale bar は右下の最後の panel のみ

出力:

- PNG
- PDF
- SVG

優先度:

- 最優先

### 8.2 F2: Contour Montage

目的:

- 背景なしで形状変化のみを強調する

入力:

- track mask

表示:

- contour のみ
- 同一スケール
- 必要に応じて filled mask を薄灰で表示

見た目仕様:

- 背景は白
- contour は黒または濃色
- fill を使う場合は薄いグレー 1 色
- panel 間の spacing を小さくし、シリーズ感を強く出す

優先度:

- 最優先

### 8.3 F3: Ghost Contour Overlay

目的:

- 時間変化を 1 パネルで直感的に示す

入力:

- aligned contour series

表示:

- 全 timepoint の contour を同一座標へ重ね描き
- 時間で色を変えるか、古いものを薄くする

見た目仕様:

- 背景は白
- colormap は時間順が直感的にわかるものを使う
- 輪郭線は細め
- 開始時刻と終了時刻だけ注記する

優先度:

- 高

### 8.4 F4: Shape Trace

目的:

- 形態指標の経時変化を定量表示する

入力:

- `track_summary.csv`

表示候補:

- area
- major axis
- minor axis
- aspect ratio
- eccentricity
- solidity

見た目仕様:

- 1 行 2 列または 2 行 2 列の小パネル構成
- 線色は 1 系列につき 1 色で統一
- 点は基本なし、必要なら小マーカー
- `missing_cell` 区間は薄灰帯または欠損表示

優先度:

- 最優先

### 8.5 F5: Intensity Kymograph

目的:

- 長軸に沿った signal 分布の時間変化を示す

入力:

- raw stack
- aligned crop
- selected mask

処理:

1. aligned crop に変換
2. 長軸方向に再サンプリング
3. 短軸方向で平均または max projection
4. `time x longitudinal_position` の 2D 画像を生成

見た目仕様:

- heatmap は perceptually uniform colormap を使う
- colorbar は最小限
- x 軸は normalized cell axis
- y 軸は time
- 可能なら上に代表 contour か average width を添える

優先度:

- 高

### 8.6 F6: Mask Kymograph

目的:

- raw が無い場合でも形の時間変化を kymograph 風に見せる

入力:

- aligned mask

処理:

- 各 longitudinal position に対する mask 幅を時系列化

見た目仕様:

- 背景は白
- heatmap は単色系 colormap
- 幅 0 は白に近い色で表示
- 外形変化が一目で見えるコントラストにする

優先度:

- 高

### 8.7 F7: Cell Tower

目的:

- 形と signal を frame ごとに積み上げて見せる

入力:

- aligned crop stack

見た目仕様:

- 各 row を frame、各 column を 1 panel とするのではなく、
  1 列積みまたは短いグリッドで「流れ」が読めるようにする
- 縦長になりすぎる場合は一定間隔サンプリングを使う

優先度:

- 中

### 8.8 F8: QC Figure

目的:

- 中心代表細胞選択の妥当性を確認する

表示候補:

- 原画像全体の中での採用 centroid
- distance-to-center の時系列
- selection flag
- 除外された候補数

見た目仕様:

- 情報優先でよい
- presentation/manuscript 用 preset とは別に `qc` preset を持つ
- 候補 mask と採用 mask の違いがわかる色分けを行う

優先度:

- 中

## 8.9 推奨する最初の見せる図セット

最初に実装・運用する見せる図は以下の 3 種に固定する。

### Set A: 最小セット

- F1 `Overlay Strip`
- F2 `Contour Montage`
- F4 `Shape Trace`

用途:

- 最初の結果共有
- figure の品質評価
- 実装の土台確認

### Set B: 論文候補セット

- F1 `Overlay Strip`
- F3 `Ghost Contour Overlay`
- F4 `Shape Trace`
- F5 `Intensity Kymograph`

用途:

- 形態変化 + signal 変化を 1 組で見せる

## 8.10 図ごとの推奨パネル数

- `Overlay Strip`
  - 6 panel を標準
  - 長い時系列では 8 panel まで
- `Contour Montage`
  - 6-10 panel
- `Ghost Contour Overlay`
  - 全 frame 使用可
- `Shape Trace`
  - 3-4 指標までを標準
- `Intensity Kymograph`
  - 1 panel を基本、補助 panel を足しても 2 panel まで

## 8.11 panel 選択ルール

timepoint の選択は以下を基本とする。

- 全 frame から等間隔サンプリング
- `missing_cell` frame は除外
- 開始・中盤・終盤を必ず含む

将来拡張:

- area 変化の大きい frame を優先表示
- user 指定 frame を優先表示

## 9. 図の品質要件

### 9.1 描画方針

- 最終図は `matplotlib` で描画する
- contour は `find_contours` などでベクタ線として描画する
- frame ごとの intensity 自動スケーリングは原則禁止
- `vmin/vmax` を固定または robust percentile 固定とする

### 9.2 共通デザイン

- publication 用フォントサイズを preset 化
- line width を統一
- colorblind-safe palette を採用
- 背景は白を基本
- panel label を自動付与
- scale bar を必須化

### 9.2.1 preset

#### `manuscript`

- figure width: 170 mm 相当を標準
- font size: 7-9 pt
- line width: 0.8-1.2 pt
- 背景: 白
- 装飾: 最小限

#### `presentation`

- figure width: 1920 px 想定
- font size: 12-18 pt
- line width: 1.5-2.5 pt
- コントラスト強め
- ラベルは遠目でも読める大きさ

#### `qc`

- 見た目より情報量優先
- 凡例・注釈を多めに許容
- 補助ガイド線や候補表示を許容

### 9.2.2 フォント

- 既定の sans-serif を使う場合でも、preset 内で統一する
- 論文用では過度に装飾的なフォントは使わない
- 数値ラベルと panel label の視認性を優先する

### 9.2.3 色

- 見せる図は原則 1 図 1 主色
- raw 背景 + contour の組み合わせでは色数を増やしすぎない
- kymograph は可読性重視で perceptually uniform colormap を使う

### 9.2.4 scale bar

- すべての見せる図に scale bar を入れる
- Overlay Strip は最後の panel のみ
- Montage / Ghost / Tower は図全体で 1 回のみ

### 9.2.5 余白

- subplot の隙間は狭め
- ただし panel label と time label が干渉しないこと
- figure 全体に十分な外余白を確保して切れを防ぐ

### 9.3 保存

保存は `scripts/figure_logger.py` の `save_figure()` を使用する。

必須:

- `description`
- `params`

保存形式:

- `png`
- `pdf`
- `svg`

## 10. CLI 仕様

スクリプト名の仮案:

- `scripts/central_cell_track_figures.py`

基本例:

```bash
python scripts/central_cell_track_figures.py \
  --indir /path/to/Pos3/output_phase/channels/crop_sub_rawraw/ch00 \
  --outdir /path/to/output \
  --figure overlay_strip contour_montage shape_trace intensity_kymograph
```

主要引数:

- `--indir`
- `--pos-root`
- `--channel`
- `--outdir`
- `--min-area`
- `--exclude-border`
- `--crop-margin`
- `--align-major-axis`
- `--time-interval-min`
- `--pixel-size-um`
- `--figure`
- `--preset manuscript|presentation|qc`
- `--no-save`

## 11. 実装優先順位

### Phase 1: MVP

- raw / mask ペアリング
- frame ごとの中心代表細胞選択
- central-cell series 構成
- `track_summary.csv`
- F1 `Overlay Strip`
- F2 `Contour Montage`
- F4 `Shape Trace`

### Phase 2: Figure 拡張

- 回転整列
- F3 `Ghost Contour Overlay`
- F5 `Intensity Kymograph`
- F6 `Mask Kymograph`

### Phase 3: QC と堅牢化

- F8 `QC Figure`
- selection flag の明示
- 欠損 frame の扱い改善
- preset の整備

## 12. 検証項目

### 12.1 中心代表細胞選択の妥当性

- 各 frame で選ばれた細胞が本当に中心近傍か
- 別細胞に切り替わっても用途上問題ないか
- border cell や debris を拾っていないか

### 12.2 figure の妥当性

- contour が mask と一致するか
- scale bar が正しいか
- frame 間の intensity 比較が可能な表示か
- crop サイズが各 figure で一貫しているか

### 12.3 再現性

- 同一入力で同一 figure が再生成できる
- `track_metadata.json` から条件が追跡可能

## 13. リスクと対策

### 13.1 中心細胞が途中で画面外に出る

対策:

- `missing_cell` として記録
- QC figure に selection flag を表示

### 13.2 密な画角で隣接細胞へ切り替わる

対策:

- QC figure で確認
- border-touch 候補の低優先化

### 13.3 同一細胞とは限らない

対策:

- ドキュメントと図注で `center-nearest representative cell` と明記する
- 厳密 tracking と混同しない

### 13.4 raw stack が無い

対策:

- contour / mask / shape trace / mask kymograph を主出力にする

## 14. 将来拡張

- daughter 継承モードの追加
- 複数 position にまたがる代表細胞比較
- population consensus figure
- napari viewer による QC
- track correction の半手動モード

## 15. 受け入れ条件

以下を満たせば本仕様の MVP は達成とみなす。

1. 各 frame で中心に最も近い mask を安定に 1 つ選べる
2. `track_summary.csv` を再現可能に出力できる
3. `Overlay Strip`, `Contour Montage`, `Shape Trace` を保存できる
4. 図保存が `save_figure()` 経由で行われる
5. 出力上で厳密な tracking や `lineage tree` を解いたかのような誤解を招かない
