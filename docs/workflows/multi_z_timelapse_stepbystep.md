# Multi-z タイムラプス Step-by-Step ガイド

> 対象: `realtime_drift_mda_zstack.bsh` を使う z-stack 付きリアルタイム drift 補正タイムラプス
> 最終更新: 2026-04-27

このドキュメントは、QPI で **per-Pos / per-timepoint で z-stack を取りながらリアルタイム drift 補正をかけ、後処理で grid 基準引き算と 0% グルコース培地 RI 補正までかける** 一連の手順を、スクリプト単位で順番にまとめたもの。

z-stack 化の差分はわずか 3 点（後述）。それ以外は single-z タイムラプスと共通スクリプトで動く。

---

## 全体像

```
[Day-1 一回だけ]
  REF_GRID 撮影 (例: grid_2pergluc_60ms_1)
  └─ batch_reconstruction_grid.py        Day-1 固定基準を全 z 再構成
  └─ channel_crop.py --detect            各 PosN_x+0_y+0 で channel ROI 検出

[毎実験日 — 取得前]
  CALIB_GRID 撮影 (今日の小 grid)
  ├─ focus_check_subtract.py             焦点 z を目視確認
  ├─ align_timelapse_pos.py              timelapse.pos を REF_GRID にアラインして timelapse_aligned.pos
  ├─ calibrate_grid_pos_per_pos.py       PosN ごとに grid_calibration_PosN.json (測定済 px オフセット)
  └─ prepare_drift_session.py            drift_config_zstack.json + positions_zstack.csv

[取得中 — Micro-Manager]
  realtime_drift_mda_zstack.bsh
  └─ compute_drift_online.py (各 t で呼ばれる)
       ・全 Pos × 全 z 取得 (img_{t}_ph_{z:03d}.tif)
       ・ECC で drift 測定 → Kalman → ステージ補正
       ・任意で output_phase_raw/ に位相プリ再構成
       ・pos_shifts_cal_online.json をフレームごとに追記

[取得後 — 後処理]
  ├─ extract_timelapse_delta.py          各 Pos で「grid(0,0) に最も近いフレーム」から
  │                                       z ごとに delta_z{Z:03d}.tif を作る
  └─ batch_pipeline_all_pos.py           各 Pos に対して以下を回す:
       Step 0: Reconstruction (raw → output_phase, Pos0 BG)
       Step 1: copy channel_rois.json
       Step 2: compute_pos_shifts.py     (フレームごとの ECC shift)
       Step 3: grid_subtract.py          (raw-raw 引き算)
       Step 4: correct_0pergluc.py       (0% gluc 期間の medium-RI 補正)
```

z-stack 化に伴う差分:
1. 取得 BSH が `realtime_drift_mda.bsh` → `realtime_drift_mda_zstack.bsh`
2. raw 保存階層が `PosN/z{z:03d}/img_*_ph_{z:03d}.tif`（プリ再構成有効時は `PosN/z{z:03d}/output_phase_raw/`）
3. 後処理の `extract_timelapse_delta.py` が **z ごとに delta TIF** を出力し、`correct_0pergluc.py` がそれを per-z で消費する

---

## Phase A: Day-1 基準セットアップ（1 度だけ）

実験キャンペーンの初日に固定基準を作る。以後の毎実験日では「ここで作った基準と今日の小 calibration grid の差分」を ECC で測って補正する設計のため、**REF_GRID は移動・上書きしない**。

### Step A-1. REF_GRID 撮影

例: `grid_2pergluc_60ms_1`。光路が確定した状態で `generate_grid_pos.py` 系で展開した 0.1 µm 刻みの grid pos を MM で取得する。

各 `PosN_x{i}_y{j}` フォルダに z-stack（例: 11 枚, 0.4 µm step, z=5 が中央 0.0 µm）の生ホログラム `img_000000000_ph_{z:03d}.tif` が入る。

### Step A-2. `batch_reconstruction_grid.py` — REF_GRID 全再構成

```bash
python scripts/batch_reconstruction_grid.py
```

- 入力: `GRID_DIR/PosN_x{i}_y{j}/img_000000000_ph_{z:03d}.tif`
- BG: `BG_BASE_LABEL`（既定 `Pos0`）の同 `(xi, yi, z)` を背景として引く
- 出力: `PosN_x{i}_y{j}/output_phase/img_000000000_ph_{z:03d}_phase.tif`

ポイント:
- `POS_SPLIT` で **Pos 番号によって crop（左/右チャネル）を切り替える**。データセットによって左右が入れ替わるので、必ず実画像で確認する。
- 光学定数は `optical_config.py` で一元管理。

### Step A-3. `channel_crop.py --detect` — channel ROI 検出

各サンプル `PosN_x+0_y+0/output_phase/` 内のチャネルを検出して、`output_phase/channels/channel_rois.json` を書き出す。後段の ECC は **すべてこの ROI** を共通で使うため、後の取得・解析の正本になる。

```bash
python scripts/channel_crop.py --detect
```

> `channel_rois.json` がないと `prepare_drift_session.py` が abort する。

---

## Phase B: 毎実験日の取得前準備

REF_GRID は固定したまま、その日の温度ドリフト・ステージ位置ずれを吸収するために、**今日の小 calibration grid (CALIB_GRID) を 1 回撮る**。以降のステップはこの今日の grid と Day-1 REF_GRID の比較で動く。

### Step B-1. CALIB_GRID 撮影

`timelapse.pos` を MDA で 1 timepoint だけ走らせ、`Pos0`（BG）と `Pos1, Pos2, ...`（各サンプル）の中央 (`x+0_y+0`) を取得する。z-stack 1 枚（または焦点 z 周辺数枚）でよい。

### Step B-2. `focus_check_subtract.py` — 焦点 z の目視確認

```bash
python scripts/focus_check_subtract.py
```

- 入力: `FOCUS_DIR/PosN`（細胞入り z-stack）と `GRID_DIR/PosN_x+0_y+0`（背景基準）
- 処理: PosN ごとに ECC で 1 回アラインし、その shift を **全 z に適用**して引き算
- 出力: `OUTPUT_DIR/PosN/z{z:03d}.tif`（ImageJ で z-stack として開いて中身を見る）

判断: 「どの z で細胞のエッジがピシッと出ているか」を見て **焦点 z を 1 つ決める**。方針は全 Pos 共通固定。

### Step B-3. `align_timelapse_pos.py` — `timelapse.pos` をアライン

REF_GRID は Day-1 のもの、CALIB_GRID は今日のもの、という前提で、各 PosN ごとに `REF_GRID/PosN_x+0_y+0` と `CALIB_GRID/PosN_x+0_y+0` を ECC で比較し、ステージ XY を補正する。

```bash
python scripts/align_timelapse_pos.py
```

- 入力: `TIMELAPSE_POS`（補正前 .pos）, `REF_GRID_DIR`, `CALIB_GRID_DIR`
- 処理:
  1. CALIB_GRID 中央が未再構成なら自動再構成（BG = `Pos0_x+0_y+0`）
  2. PosN ごと、`channel_rois.json` の全チャネルで ECC → MAD アウトライア除去 → 平均
  3. `drift_stage_x_um = sx_sign * ty_avg * pixel_scale_um` などで sign 適用
  4. `pos_correct = -drift` を XYStage に加算
  5. BG Pos (`Pos0`) は対応サンプル Pos（既定 `Pos1`）の補正をコピー
- 出力: `BASE_DIR/timelapse_aligned.pos` + `align_timelapse_log.json`

> sign 規約: `findTransformECC(ref, sample) -> (tx, ty)` を画像 X→ステージ Y、画像 Y→ステージ X に変換するのは `calibrate_grid_pos.py` と同じ。詳細は同ファイル L389-394 のコメント参照。

**MM 操作:** 出力された `timelapse_aligned.pos` を MM にロードする。これでタイムラプス開始時点で各 PosN は grid x+0_y+0 の近傍に再センタリングされる。

### Step B-4. `focus_check_subtract.py` 再確認（任意）

`timelapse_aligned.pos` で再撮影した z-stack に対し、もう一度 `focus_check_subtract.py` を回して焦点 z がずれていないか確認する。

### Step B-5. `calibrate_grid_pos_per_pos.py` — Pos ごとの grid calibration

各 PosN について **「nominal の 0.1 µm grid」と「実測 px オフセット」の対応表**（`grid_calibration_PosN.json`）を作る。後処理 (`grid_subtract.py`) と取得時 (`compute_drift_online.py`) はどちらもこの実測値を使って最近接 grid を選ぶ。

```bash
python scripts/calibrate_grid_pos_per_pos.py
```

- 出力: `GRID_DIR/grid_calibration_PosN.json`（PosN ごと）
- 並列化したい場合は `parallel_calibrate.py` を使う。

> このファイルがない PosN があると `prepare_drift_session.py` が abort する。

### Step B-6. `prepare_drift_session.py` — drift session 設定書き出し

`drift_config_zstack.json` と `positions_zstack.csv`、`drift_state_zstack.txt`、`drift_log_zstack.json`、`drift_kf_state_zstack.json` を一括で生成する。

```bash
python scripts/prepare_drift_session.py
```

事前に `prepare_drift_session.py` 上部で**毎実験ごとに編集する箇所**:
- `POSITIONS_FILE` → `timelapse_aligned.pos`
- `GRID_DIR` → 今日の小 calibration grid（または REF_GRID; per-pos calibration が見られる方）
- `SAVE_DIR` → MM のタイムラプス保存先
- `BG_POS_INDEX`（既定 0 = Pos0）
- `N_TIMEPOINTS` / `INTERVAL_SEC` / `EXPOSURE_MS`
- **z-stack パラメタ:** `N_Z_SLICES`, `Z_STEP_UM`, `Z_START_UM`
- `RAW_TL_Z_INDEX`（drift 計測に使う z スライス。例: 5）
- `CROP_SUB_ROOT`（オンライン crop-sub 保存先）

確認内容:
1. `.pos` を読んで `positions_zstack.csv` を作る
2. 各 サンプル Pos に `channel_rois.json` と `grid_calibration_PosN.json` が揃っているか検証 → 欠けていれば abort
3. `drift_config_zstack.json` に光学定数・sign・Kalman パラメタ・z-stack パラメタを書き出す
4. 状態ファイルを初期化（前回の `drift_log_*.json` はタイムスタンプ付きでアーカイブ）

> `N_Z_SLICES > 1` のときファイル名末尾に `_zstack` がつく（`drift_config_zstack.json` 等）。後段の BSH もこちらを読むよう設定する。

---

## Phase C: 取得（Micro-Manager）

### Step C-1. `realtime_drift_mda_zstack.bsh` を MM Script Panel から起動

1. MM 1.4 の Script Panel を開く
2. `scripts/realtime_drift_mda_zstack.bsh` を読み込む
3. ファイル先頭の `CONFIG_FILE` を `prepare_drift_session.py` が出力した `drift_config_zstack.json` のパスに書き換える
4. `FORCE_FRESH_START = true` で T=0 から、前回続きから再開する場合は `false` + `RESUME_SAVE_DIR` を指定
5. Run

BSH 側のループ:
```
for t in 0..N_TIMEPOINTS-1:
    for pos in positions:
        XY 移動 → settle
        for z in 0..N_Z_SLICES-1:
            TIPFSOffset 移動
            画像取得 → img_{t}_ph_{z:03d}.tif として PosN/ 直下に保存
    python compute_drift_online.py --timepoint t --config drift_config_zstack.json
    interval まで待機
```

### Step C-2. `compute_drift_online.py`（t ごとに自動実行）

各 t で次を Pos ごとに並列実行する:

1. **生ホログラムを ON-the-fly で位相再構成**（Pos0 BG 引き算 + tilt 補正）
2. **`channel_rois.json` の各チャネルで ECC** を grid(0,0) リファレンスに対して計算（pass 1）
3. 必要なら最近接 grid (xi, yi) を選び直して **pass 2 / pass 3** で再計算
4. MAD でアウトライア除去後、チャネル平均
5. ECC sign を適用してステージ µm 単位に変換 → **Kalman フィルタ**（位置のみランダムウォーク, Q/R は drift_config に書かれた測定値ベース）
6. 補正値を `drift_state.txt` に書く（次の t で BSH が読む）
7. `pos_shifts_cal_online.json` にフレーム結果を追記
8. オプションで `output_phase_raw/` に生位相をプリ再構成して保存（`enable_crop_sub_save = true` の場合）

> Kalman の Q/R は実測値（`KF_Q_TY_NM2 = 291`, `KF_R_TY_NM2 = 91` など）から K≈0.80 になるよう調整済み。値は `prepare_drift_session.py` 上で変更できる。

> 取得中の様子は `01_realtime_visibility_monitor.py` などで監視できる。

---

## Phase D: 取得後処理

実験終了後、`SAVE_DIR/PosN/` 下に `img_{t}_ph_{z:03d}.tif`（生ホログラム） + 任意で `output_phase_raw/img_*_ph_{z:03d}_phase.tif`（プリ再構成位相） + `output_phase/channels/pos_shifts_cal_online.json`（フレームごと shift 履歴）が揃った状態。

### Step D-1. `extract_timelapse_delta.py` — 全 Pos の per-z delta TIF 抽出

0% グルコース培地に切り替えた後、その期間中の medium RI 差を引くために使う **delta_z{Z:03d}.tif** を z ごとに作る。`grid_0per` 用の別 grid 撮影が不要になる代わりに、タイムラプス自身の「grid(0,0) に最も近い 1 フレーム」を使ってその差分を作る。

```bash
python scripts/extract_timelapse_delta.py
```

- 入力: `TIMELAPSE_ROOT/PosN/`（取得結果）, `GRID_2PER_DIR`（Day-1 REF_GRID）, `pos_shifts_cal_online.json`
- 処理（PosN ごとに）:
  1. `pos_shifts_cal_online.json` を読み、`shift_x_avg/shift_y_avg` のノルムが最小のフレーム index を選ぶ
  2. 各 `(tl_z, grid_z)` ペアについて、そのフレームの timelapse 生位相と REF_GRID `Pos*_x+0_y+0` の同 `grid_z` 位相をロード
  3. timelapse 側を `apply_inverse_shift_warp(sx, sy)` で grid(0,0) に揃え、引き算
  4. `delta_z{grid_z:03d}.tif` を `PosN/output_phase/channels/delta_timelapse/` に保存
- 出力: `PosN/output_phase/channels/delta_timelapse/delta_z{Z:03d}.tif` × n_z + ログ JSON
- 副産物: `figure_logger` 経由で per-Pos の delta マップ可視化図

> `Z_PAIRS = [(i, i) for i in range(11)]` のように **timelapse z と grid z の物理 z (µm) を一致させる必要がある**ことに注意。撮影設計時点で揃えておく。

### Step D-2. `batch_pipeline_all_pos.py` — 全 Pos まとめて 4 ステップ実行

```bash
python scripts/batch_pipeline_all_pos.py
# または: python scripts/batch_pipeline_all_pos.py --skip-grid-0per
```

ファイル先頭の編集ポイント:
- `TIMELAPSE_ROOT`, `GRID_2PER_DIR`, `GRID_0PER_DIR`
- `POS_START`, `POS_END`（既定で Pos0 = BG はスキップ）
- `GLUCOSE_0_START`, `GLUCOSE_0_END`（0% グルコース期間のフレーム index, end は exclusive）
- `RAW_TL_Z_INDEX`, `RAW_GRID_Z_INDEX`（解析対象 z スライス）
- `SHIFT_SIGN_X`, `SHIFT_SIGN_Y`（ECC sign）
- ECC パラメタ（`TILT_CROP_H`, `ECC_CROP_H`, `OUTLIER_MAD_THRESH`, `ECC_MIN_CORR`, `VMIN/VMAX`）

このスクリプトは Pos ごとに以下 4 ステップを順番に実行する。

#### Step D-2-0: Reconstruction（生 → 位相）

- 入力: `PosN/img_{t}_ph_{z:03d}.tif`
- BG: `Pos0` の同フレームの位相を `Pos0/bg_phase_before/` または `bg_phase_after/` にキャッシュして再利用
- 出力: `PosN/output_phase/img_{t}_ph_{z:03d}_phase.tif`
- 並列度: `N_WORKERS_RECON`（既定 24）

> z-stack 取得の場合、`PosN/z{z:03d}/output_phase_raw/` にプリ再構成があれば `extract_timelapse_delta.py` 側はそちらも自動的に拾う。

#### Step D-2-1: `channel_rois.json` のコピー

REF_GRID 側の `PosN_x+0_y+0/output_phase/channels/channel_rois.json` を `PosN/output_phase/channels/` にコピー。以降の per-frame ECC で同じ ROI を使う。

#### Step D-2-2: `compute_pos_shifts.py` — フレームごとの ECC shift

```bash
python scripts/compute_pos_shifts.py
```

- 全フレームの位相に対して `channel_rois.json` の各チャネルで ECC をかけ、grid(0,0) に対する `shift_x`, `shift_y` を計算
- MAD アウトライア除去（`OUTLIER_MAD_THRESH = 5.0`）+ ECC 相関閾値（`ECC_MIN_CORR = 0.96`）
- 出力: `PosN/output_phase/channels/pos_shifts.json`

> オンライン版 `pos_shifts_cal_online.json` と区別。後処理ではより慎重に出した `pos_shifts.json` を使うのがデフォルト。

#### Step D-2-3: `grid_subtract.py` — raw-raw 引き算

各フレームについて、`grid_calibration_PosN.json` を使って **shift に最も近い grid (xi, yi)** を選ぶ。

- 残差シフトを `apply_inverse_shift_warp` で適用
- 該当 `(xi, yi)` の REF_GRID 位相を引いて `grid_sub` を得る
- 出力: `PosN/output_phase/grid_sub/img_{t}_ph_{z:03d}_phase.tif`

> 「raw-raw」は raw 位相どうしを直接引くという意味。背景補正 (gauss/iarpls) の前段にあたる。

#### Step D-2-4: `correct_0pergluc.py` — 0% gluc 期間の medium-RI 補正

`GLUCOSE_0_START ≤ t < GLUCOSE_0_END` のフレームのみが対象。

- per-Pos: `delta_timelapse/delta_z{Z:03d}.tif`（Step D-1 の出力）を**そのフレームの cal(xi,yi) に合わせて warp** → tilt 補正 → 引き算
- 出力: `PosN/output_phase/grid_sub_0per_corrected/img_{t}_ph_{z:03d}_phase.tif`
- 並列度: `N_PARALLEL_FRAMES`（既定 8）

これで **2% gluc → 0% gluc に切り替えた区間でも、培地 RI 差が乗らない位相画像**が得られる。

### Step D-3.（任意）背景プロファイル除去 — `timelapse_iarpls_bgsub.py` / `timelapse_plane_bgsub.py`

`grid_sub` 後にも残る左右非対称な傾斜や緩やかなプロファイルを除く。

- `timelapse_plane_bgsub.py`: BG ROI で平面 fit → 引く
- `timelapse_iarpls_bgsub.py`: BG ROI 列平均に **pspline_iarpls** をフィット、左半分のテール傾きで右半分を線形外挿、全列に適用
  - 出力: `PosN/output_phase_iarpls/img_{t}_ph_{z:03d}_phase.tif`（既定 40×440 の crop）

> どちらを採用するかは「画像端の差」と「細胞ピクセルへの影響」で評価する（`docs/ANALYSIS_FLOW_CURRENT.md` フェーズ5 参照）。

### Step D-4. QC

```bash
python scripts/visualize_timelapse_qc.py
```

`pos_shifts.json` / `pos_shifts_cal_online.json` / 各種 log JSON を読んで、各 Pos の shift 履歴・チャネル除外率・Kalman gain などを figure として出す。

---

## チェックリスト（毎実験日）

取得前:
- [ ] CALIB_GRID 撮影済み（今日の小 grid）
- [ ] `focus_check_subtract.py` で焦点 z を決定し、`prepare_drift_session.py` の `RAW_TL_Z_INDEX` と一致
- [ ] `align_timelapse_pos.py` の出力 `timelapse_aligned.pos` を MM にロード
- [ ] `calibrate_grid_pos_per_pos.py` を全 Pos 走らせ、`grid_calibration_PosN.json` 揃っている
- [ ] `prepare_drift_session.py` を成功で終了（abort なし）
- [ ] `realtime_drift_mda_zstack.bsh` の `CONFIG_FILE` が `drift_config_zstack.json` を指している
- [ ] `FORCE_FRESH_START = true`（再開時は `false` + `RESUME_SAVE_DIR`）

取得後:
- [ ] `pos_shifts_cal_online.json` が全 Pos に存在
- [ ] `extract_timelapse_delta.py` が全 Pos 成功（`delta_timelapse/delta_z*.tif` × n_z）
- [ ] `batch_pipeline_all_pos.py` が `POS_START`〜`POS_END` 全部完走
- [ ] `correct_0pergluc.py` の出力フレーム数 = `GLUCOSE_0_END - GLUCOSE_0_START`

---

## ファイルマップ早見表

| パス | 中身 |
|---|---|
| `REF_GRID_DIR/PosN_x{i}_y{j}/output_phase/img_*_ph_{z:03d}_phase.tif` | Day-1 基準位相（全 z, 全 grid 点） |
| `REF_GRID_DIR/PosN_x+0_y+0/output_phase/channels/channel_rois.json` | per-Pos チャネル ROI（正本） |
| `REF_GRID_DIR/grid_calibration_PosN.json` | per-Pos の grid (xi,yi) → 実測 px オフセット表 |
| `BASE_DIR/timelapse_aligned.pos` | 当日 ECC 補正済みの MM 用 .pos |
| `SESSION_DIR/drift_config_zstack.json` | 取得・後処理共通の設定 |
| `SESSION_DIR/drift_state_zstack.txt` | 取得中の状態（BSH と Python の同期） |
| `SAVE_DIR/PosN/img_{t}_ph_{z:03d}.tif` | 取得した生ホログラム |
| `SAVE_DIR/PosN/output_phase/channels/pos_shifts_cal_online.json` | オンライン ECC ログ |
| `SAVE_DIR/PosN/output_phase/img_{t}_ph_{z:03d}_phase.tif` | 後処理で再構成した位相 |
| `SAVE_DIR/PosN/output_phase/channels/pos_shifts.json` | 後処理の per-frame ECC shift |
| `SAVE_DIR/PosN/output_phase/channels/delta_timelapse/delta_z{Z:03d}.tif` | per-z medium-RI 補正用 delta |
| `SAVE_DIR/PosN/output_phase/grid_sub/...` | grid 引き算済み位相 |
| `SAVE_DIR/PosN/output_phase/grid_sub_0per_corrected/...` | 0% gluc 補正済み位相 |

---

## 関連ドキュメント

- `docs/ANALYSIS_FLOW_CURRENT.md` — single-z 含むタイムラプス全体の解析フロー
- `docs/workflows/2025-12-23_timeseries_total_mass.md` — 体積・乾燥質量時系列追跡
- 各スクリプトの docstring が一次資料。記載と実装が食い違う場合はスクリプトを正とする。
