# QPI タイムラプス実験 — 統合プロトコル

> **対象**: タイプB 段階的飢餓実験（2% → Low% → 0% → 2% グルコース）における QPI 定量位相タイムラプス
> **読者**: Mother Machine + QPI 装置を扱える経験者
> **更新方針**: スクリプトの実装が変わったらこの doc を真っ先に更新する。`ANALYSIS_FLOW_CURRENT.md` `PIPELINE.md` 等の旧 doc は廃止しこの doc に一本化

---

## 0. 全体フロー（要約）

```
[Day 0]
1. ステージに Mother Machine を載せる、2per gluc を流す
2. 光学系調整（visibility ≈ 0.75）
3. Micro-Manager で timelapse.pos 作成（Pos0 = BG / Pos1..N = 細胞 ch）
4. python generate_grid_pos.py            → grid.pos（各 Pos の周囲 9×9=81 点）
5. MM で grid.pos を load → 2per gluc で grid 撮影（z 11 slice, exp 60ms）
6. python batch_reconstruction_grid.py    → 全 81 点 × 11 z を再構成
7. python prep_channel_rois.py            → channel_rois.json（per Pos）
8. python calibrate_grid_positions.py     → grid_calibration_{pos}.json

[Day 0 evening]
9. 培地を 0per / Lowper に切替 → MM で短い z-stack timelapse
10. python extract_timelapse_delta.py     → delta_z*.tif（per Pos / per z）

[Day 1]
11. 細胞ローディング → 焦点 z 決定 → 培地を 2per に戻す
12. python prepare_drift_session.py       → drift_config.json + state files
13. MM Script Panel で realtime_drift_mda.bsh を Run
    → 内部で compute_drift_online.py が立ち上がり、各 frame を ECC drift 補正 + grid_subtract
14. 培地切替（2 → Low → 0 → 2）は手動オペレーション、フレーム番号を必ず記録

[Day 2+]
15. python correct_0pergluc.py            → 0% 期間 frame に delta を warp 引き算
16. (任意) Omnipose GUI で training 画像を作成 → 学習
17. python 07_segmentation.py             → mask
18. (任意) python calibrate_ri.py         → 培地 RI を MilliQ + EtOH 2点法で校正
19. ImageJ で細胞 ROI tracking → Results.csv
20. python 32_simple_ellipse_ri.py        → 細胞 RI / dry mass
21. python central_cell_lineage_tracker.py → 系譜・dry mass 時系列
22. python qpi_fig_03_lineage_analysis.py → 解析図
```

---

## 1. データ保存先の規約

スクリプト内で混在しているので、**新規実験ごとに以下に揃える**。

| 用途 | 場所 | 例 |
|---|---|---|
| Micro-Manager 生 hologram（細胞・grid・delta すべて） | `D:\AquisitionData\Kitagishi\YYMMDD\<exp_name>\` | `D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1\` |
| timelapse.pos / grid.pos | `C:\YYMMDD\` | `C:\260423\timelapse.pos` |
| drift session 設定一式 | `C:\Users\QPI\Documents\QPI_Omni\drift_session\` | 固定 |
| 解析中間ファイル（output_phase, channels/, crop_sub/） | 生 hologram と同一 Pos 配下 | `…\Pos1\output_phase\channels\` |

`<exp_name>` 命名例:
- `2per_gridgluc_1`（grid 撮影、2% 培地）
- `0per_zstack_1`（delta 用 0% z-stack）
- `Lowper_zstack_1`（delta 用 Low% z-stack）
- `ph_1`（本番タイムラプス）

---

## 2. Phase 1 — 撮影前準備

1. Mother Machine をステージにマウント、2per gluc を流す
2. 光学系調整：`scripts/01_realtime_visibility_monitor.py` を立ち上げ、visibility が ≈ 0.75 に乗るまで参照系を調整

---

## 3. Phase 2 — timelapse.pos 作成

Micro-Manager で Position List を作る。

- **Pos0**: 細胞のいない流路（BG）
- **Pos1 〜 PosN**: 細胞測定対象の channel（n_channels 個）

保存先: `C:\YYMMDD\timelapse.pos`

> Pos 数の目安: 12（Pos0 + 11 細胞）。`drift_config.json` の `n_channels` と一致させる。

---

## 4. Phase 3 — Grid 撮影と校正

### 4.1 `generate_grid_pos.py` — grid.pos 生成

**何をするか**:
1. `timelapse.pos` を読み込む
2. 各 Pos の `XYStage` 座標を中心に、`(xi, yi) ∈ [-4, +4] × [-4, +4]` で 9×9=81 点を展開
3. **snake scan 順**（行ごとに yi の進行方向を反転）で並べる → ステージのバックラッシュ低減
4. `Z` （`TIPFSOffset`）は元の値を保持
5. 新ラベル: `{base_label}_x{xi:+d}_y{yi:+d}` （例 `Pos1_x+0_y+0`, `Pos1_x-1_y+2`）
6. JSON として `grid.pos` に書き出し

**編集箇所** (`scripts/generate_grid_pos.py`):
```python
INPUT_POS  = r"C:\260423\timelapse.pos"
OUTPUT_POS = r"C:\260423\grid.pos"
X_STEP = 0.1   # um
Y_STEP = 0.1   # um
X_HALF = 4    # 9 points per axis (±0.4 um)
Y_HALF = 4
```

**実行**:
```bash
python scripts/generate_grid_pos.py
```

**出力**: `C:\YYMMDD\grid.pos`（位置数 = 元 Pos × 81）

---

### 4.2 Grid 撮影（Micro-Manager）

- `grid.pos` を Position List にロード
- z stack: **z = -2.0 〜 +2.0 µm, 0.4 µm step（11 slices）**
- exposure: **60 ms**
- channel: `ph`
- 培地: **2% glucose（2per gluc）**
- 出力: `D:\AquisitionData\Kitagishi\YYMMDD\grid_2pergluc_1\PosN_x±i_y±j\img_000000000_ph_{z:03d}.tif`

---

### 4.3 `batch_reconstruction_grid.py` — Grid 位相再構成

**何をするか**:
1. `GRID_DIR` 配下を走査し、`{base_label}_x{xi:+d}_y{yi:+d}` パターンの Pos フォルダを集める
2. Pos 番号で crop 領域を切替（`pos_number < POS_SPLIT` → CROP_BEFORE, else CROP_AFTER）
3. 各 hologram 1枚ずつ：
   - PIL で読み込み → crop
   - `qpi.get_field()`：オフアクシス FFT → サイドバンド中心化 → LP filter → IFFT で複素場
   - `np.angle()` → `skimage.restoration.unwrap_phase()` でラップ解除
4. **BG 引き算**（同 (xi, yi) の `Pos0_x{xi}_y{yi}` を BG として引く）
5. 端部の mean を引いて 0-mean に揃える（POS_SPLIT で左右どちら半分を使うか切替）
6. `output_phase/img_000000000_ph_{z:03d}_phase.tif` に float32 で書く
7. GPU+CPU パイプライン: 1 GPU producer（FFT）+ N CPU consumer（unwrap + save）でスループット確保

**編集箇所** (`scripts/batch_reconstruction_grid.py`):
```python
GRID_DIR = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
BG_BASE_LABEL = "Pos0"
TARGET_BASE_LABELS = None    # None = Pos0 以外全て
TARGET_COORDS = None         # None = 全 81 点
Z_INDICES = None             # None = 全 11 z
POS_SPLIT = 52               # ステージ位置に応じた crop 切替
SKIP_IF_EXISTS = True
```

光学定数は `scripts/optical_config.py` から自動読み込み（`OFFAXIS_CENTER`, `WAVELENGTH`, `NA`, `PIXELSIZE`）。

**実行**:
```bash
python scripts/batch_reconstruction_grid.py
```

**出力**: `…\grid_2pergluc_1\PosN_x±i_y±j\output_phase\img_000000000_ph_{z:03d}_phase.tif`

---

### 4.4 `prep_channel_rois.py` — channel_rois.json 生成

**何をするか**:
1. `timelapse.pos` を読み、Pos0（BG）以外の sample Pos を列挙
2. 各 PosN について `{REF_GRID_DIR}/PosN_x+0_y+0` を中心位置として特定
3. もし `output_phase/img_000000000_ph_{Z:03d}_phase.tif` が無ければ：
   - Pos0 を BG として recon（`reconstruct_phase`：FFT → サイドバンド → LP → IFFT → unwrap）
   - PosN center を recon
4. `output_phase` 上で `channel_crop.py --detect` を subprocess 起動
   → 位相画像のチャネル軸プロファイル（peak detection）から各 channel ROI の中心 (cx, cy) と crop 幅/高 を検出
5. `PosN_x+0_y+0/output_phase/channels/channel_rois.json` を保存（list of `{cx, cy, crop_w, crop_h}`）

**編集箇所** (`scripts/prep_channel_rois.py`):
```python
TIMELAPSE_POS  = r"C:\260423\timelapse.pos"
REF_GRID_DIR   = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
DRIFT_CONFIG   = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json"
BG_LABEL       = "Pos0"
CALIB_Z_INDEX  = 5    # grid 中央 z（11 slices なら index 5 = 0.0 µm）
```

**実行**:
```bash
python scripts/prep_channel_rois.py
```

**出力（per Pos）**:
```
…\grid_2pergluc_1\PosN_x+0_y+0\output_phase\channels\channel_rois.json
```

---

### 4.5 `calibrate_grid_positions.py` — Grid 実位置の校正

**何をするか（per Pos 実行）**:
1. `BASE_LABEL` の Pos の 81 点フォルダを走査
2. `(0, 0)` 点の position phase 画像をリファレンスとしてロード
3. 各チャネル ROI について：
   - `extract_rect_roi(cy, cx, CROP_W, TILT_CROP_H)` で大きめ crop を取り
   - `tilt_fit_crop`: 端領域から 2D の slope (a, b) と intercept c を最小二乗フィット → `phase - (a·x + b·y + c)` で傾き＋オフセット除去 → 中心 `ECC_CROP_H` 幅に再 crop
   - `to_uint8(VMIN, VMAX)` で 8-bit に正規化（ECC は uint8 を要求）
4. 81 点の各 (xi, yi) について同じ前処理 → `(0, 0)` との `cv2.findTransformECC` を全チャネルで走らせ、translation `(tx, ty)` と相関を取得
5. チャネル間平均 `actual_dx = mean(-tx)`, `actual_dy = mean(-ty)` を計算（`cur` 側の content は `(-tx, -ty)` だけ動いて見える）
6. nominal（理論）位置 `nominal_dx = SHIFT_SIGN_Y * yi * Y_STEP / pixel_scale_um` と比較
7. 結果を `grid_calibration_{base_label}.json` として保存:
   ```json
   {
     "(xi,yi)": {
        "actual_dx_px": ..., "actual_dy_px": ...,
        "nominal_dx_px": ..., "nominal_dy_px": ...,
        "error_dx_px":  ..., "error_dy_px": ...,
        "mean_correlation": ...
     }
   }
   ```

> ※ ヘッダ docstring に書かれている BFS chaining は最適化用の代替実装で、現行の `main()` は 81 点全部を直接 (0,0) と ECC する **direct comparison** モード。両者の出力 JSON は同じスキーマ。

**編集箇所** (`scripts/calibrate_grid_positions.py`):
```python
GRID_DIR          = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
BASE_LABEL        = "Pos1"     # ★ Pos 番号を変えて Pos1〜PosN 全てに実行
GRID_Z_INDEX      = 5
CHANNEL_ROIS_JSON = r"…\PosN_x+0_y+0\output_phase\channels\channel_rois.json"
VMIN = -5.0
VMAX =  2.0
TILT_CROP_H = 270
ECC_CROP_H  = 80
POS_SPLIT   = 52
OUTPUT_JSON = None    # None → GRID_DIR/grid_calibration_{BASE_LABEL}.json
```

**実行（全 Pos まとめて回す例）**:
```bash
# Pos1〜Pos11 を順に
for i in 1 2 3 4 5 6 7 8 9 10 11; do
  python -c "
import sys; sys.path.insert(0,'scripts')
import calibrate_grid_positions as m
m.BASE_LABEL = f'Pos$i'
m.CHANNEL_ROIS_JSON = rf'D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1\Pos$i_x+0_y+0\output_phase\channels\channel_rois.json'
m.main()
"
done
```
（または `parallel_calibrate.py` で並列化）

**出力**: `…\grid_2pergluc_1\grid_calibration_PosN.json`（per Pos）

---

## 5. Phase 4 — Delta 撮影（培地 RI 差の補正用）

### 5.1 Delta 用 timelapse 撮影

**目的**: 0% / Low% glucose の培地が流れているときの「同じ流路の同じ位置 + 同じ z」での位相 vs 2% grid の差分を取り、後で本番 timelapse の対応 frame に引き算する。

手順:
1. 培地を 0per gluc に切替 → 5 min 待機（流路置換完了）
2. MM で `timelapse.pos`（grid.pos ではなく元 Pos）をロード
3. z-stack 短時間 timelapse 撮影:
   - z = -2.0 〜 +2.0 µm, 0.4 µm step（grid と同じ 11 slice）
   - exposure 60 ms, interval 5 min, 5〜10 frame で十分
   - 出力: `D:\AquisitionData\Kitagishi\YYMMDD\0per_zstack_1\PosN\z{z:03d}\img_*_ph_*.tif`
     または `…\PosN\img_*_ph_*.tif`（single-z レイアウト）
4. Low% についても同様に撮影 → `Lowper_zstack_1`

---

### 5.2 `extract_timelapse_delta.py` — Delta TIF 生成

**何をするか**:
1. `TIMELAPSE_ROOT` 配下から `PosN`（N ≥ 1）を自動検出
2. 各 Pos について `pos_shifts JSON` を読み、frame ごとの shift magnitude が最小（= grid(0,0) に最も近い）frame を選出
3. `Z_PAIRS = [(tl_z, grid_z), …]` の各組について：
   - timelapse 側 frame の生 hologram を recon（`output_phase_raw` があればロード、無ければ `reconstruct_from_holo` で on-the-fly）
   - 同 z の `GRID_2PER_DIR/PosN_x+0_y+0` の grid recon 画像をロード
   - `apply_inverse_shift_warp` で timelapse frame を grid(0,0) 座標系に warp
   - `delta = warped_timelapse - grid_2per`
   - `delta_z{grid_z:03d}.tif` として保存（511×511 float32）

**編集箇所** (`scripts/extract_timelapse_delta.py`):
```python
TIMELAPSE_ROOT = r"D:\AquisitionData\Kitagishi\260423\0per_zstack_1"
GRID_2PER_DIR  = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
POS_START = 1
POS_END   = None
SHIFTS_FILENAME = "pos_shifts_cal_online.json"
Z_PAIRS = [(i, i) for i in range(11)]   # 11-slice 1:1
FRAME_RANGE = None
OUTPUT_SUBDIR = "delta_timelapse"   # ★ Low% 用は "delta_lowper" 等に変える
```

**実行**:
```bash
# 0per
python scripts/extract_timelapse_delta.py
# Low% 用に TIMELAPSE_ROOT と OUTPUT_SUBDIR を書き換えて再実行
```

**出力**:
```
…\0per_zstack_1\PosN\output_phase\channels\delta_timelapse\delta_z{Z:03d}.tif
```

→ Phase 7 の `correct_0pergluc.py` の `DELTA_TIFS_DIR` でこのパスを指す。

---

### 5.3 `calibrate_ri.py` — 培地 RI の絶対値校正（任意 / 数ヶ月に1回）

**何をするか**: MilliQ と EtOH の既知 RI（n_miliq=1.3312, n_etoh=1.3588 @ 658nm 25℃）から 2 点法で `n_2per` を逆算。

理論:
```
S = Σ delta[channel_mask] = (n_medium - n_2per) · V_total
n_2per = (S_miliq · n_etoh - S_etoh · n_miliq) / (S_miliq - S_etoh)
```

手順:
1. MilliQ を device に注入 → 5 min 待機 → `timelapse.pos` で多 z timelapse 撮影
2. EtOH を注入 → 5 min 待機 → 同条件で撮影
3. MilliQ / EtOH それぞれで `extract_timelapse_delta.py` を実行（`OUTPUT_SUBDIR` を `delta_miliq` / `delta_etoh` に変える）
4. `python scripts/calibrate_ri.py`

**編集箇所** (`scripts/calibrate_ri.py`):
```python
GRID_2PER_DIR = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
MILIQ_SESSION = r"E:\260424\miliq"
ETOH_SESSION  = r"E:\260424\etoh"
MILIQ_DELTA_SUBDIR = "output_phase/channels/delta_miliq"
ETOH_DELTA_SUBDIR  = "output_phase/channels/delta_etoh"
DELTA_Z = 5
POS_NUMBERS = list(range(1, 12))
N_MILIQ = 1.3312
N_ETOH  = 1.3588        # ★ 希釈 EtOH の場合は要更新
SKIP_EDGE_CHANNELS = True
```

**処理内容**:
1. 各 Pos の `Pos{N}_x+0_y+0/output_phase/img_000000000_ph_{Z:03d}_phase.tif`（grid_2per BG引き済み phase）をロード → channel_rois.json と組み合わせ、各チャネルで `extract_rect_roi → apply_2pi_tilt_crop → mask = (phase < -1.0)` を作る
2. MilliQ delta TIF, EtOH delta TIF をロード → 同じ tilt 補正 → mask 内合計 `S_miliq`, `S_etoh`
3. 上式で `n_2per` を逆算 → 32_simple_ellipse_ri.py の `N_MEDIUM` に反映

**出力**: 標準出力に `n_2per` 推定値（コンソール）

---

## 6. Phase 5 — 細胞ローディング・焦点 z 決定

1. 細胞をロード
2. focus 用に短時間 z-stack timelapse → 目視で焦点 z を決定
3. MM の Position List で各 Pos の z を更新（決まった focus z にそろえる）
4. 培地を 2per gluc に戻す

---

## 7. Phase 6 — 本番タイムラプス（リアルタイム drift 補正付き）

### 7.1 `prepare_drift_session.py` — drift session 準備

**何をするか**:
1. `POSITIONS_FILE`（timelapse.pos）を読み、`positions.csv`（Beanshell が消費する簡易フォーマット）を生成
2. `GRID_DIR` 配下に `grid_calibration_{pos_label}.json` が **全 sample Pos 分** 存在することを検証（無いと abort）
3. `drift_config.json` に全パラメータをまとめて書き出す
4. `drift_state.txt`（progress カウンタ）と `drift_log.json` を初期化

**編集箇所** (`scripts/prepare_drift_session.py`):
```python
POSITIONS_FILE = r"C:\260423\timelapse.pos"
GRID_DIR       = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
GRID_Z_INDEX   = 5
SESSION_DIR    = r"C:\Users\QPI\Documents\QPI_Omni\drift_session"
SAVE_DIR       = r"D:\AquisitionData\Kitagishi\260423\ph_1"   # ★ 本番出力先
BG_POS_INDEX   = 0
N_TIMEPOINTS   = 3168           # ★ 総 frame 数（培地切替 timeline に応じて）
INTERVAL_SEC   = 300            # 5 min
EXPOSURE_MS    = 60.0
SETTLE_MS      = 150
```

**実行**:
```bash
python scripts/prepare_drift_session.py
```

**出力**:
```
C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_config.json
                                              \positions.csv
                                              \drift_state.txt
                                              \drift_log.json
                                              \drift_kf_state.json
```

---

### 7.2 `realtime_drift_mda.bsh` — Micro-Manager から起動

**位置づけ**: MM の Script Panel で動かす Beanshell。MM 内蔵の MDA は使わず、このスクリプトが acquisition と drift correction を統合制御する。

**起動手順**:
1. MM Script Panel を開く
2. `scripts/realtime_drift_mda.bsh` をロード
3. ファイル冒頭の `CONFIG_FILE` を `drift_config.json` のパスに合わせる：
   ```java
   String CONFIG_FILE = "C:/Users/QPI/Documents/QPI_Omni/drift_session/drift_config.json";
   boolean FORCE_FRESH_START = true;
   ```
4. **Run**

**bsh が内部でやること**（要点）:
- positions.csv からポジション一覧をロード
- 各 timepoint で：
  1. 全 Pos を順に巡回 → exposure → 1 frame 取得（XYStage と TIPFSOffset を補正済み座標で動かす）
  2. `Runtime.getRuntime().exec("python compute_drift_online.py --timepoint T --config drift_config.json")` で drift correction を起動
  3. `compute_drift_online.py` が次 timepoint の補正量を `drift_log.json` に書き、次回はそれをポジションに加味

中断は **Stop** ボタン。再開は `FORCE_FRESH_START = false` にして再 Run。

---

### 7.3 `compute_drift_online.py` — Drift correction 内部処理

**何をするか（per Pos, parallel ProcessPoolExecutor）**:
1. 当該 timepoint で取得された hologram を読む
2. recon（CROP, FFT, LP, IFFT, unwrap）→ output_phase
3. 各チャネル ROI で `tilt_fit_crop`（slope+intercept フィットで傾き除去）して局所 crop を取得
4. `to_uint8(VMIN, VMAX)` で 8-bit 化
5. **Grid (0,0) refernce** との `cv2.findTransformECC` で `(tx, ty)` を計算 → 全チャネル中央値を取る
6. `(tx, ty)` を `grid_calibration_{pos}.json` の `actual_dx/dy` テーブルと突き合わせ、最も近い grid 点 `(xi, yi)` を引き当てる
7. **Kalman filter**（位置・速度のランダムウォークモデル）で測定値を平滑化:
   - `kf_step_posonly_nm()`: `x_k = x_{k-1} + w_k`, `z_k = x_k + v_k`
   - Q（プロセス分散）, R（観測分散）は `drift_config.json` 内
8. 平滑化された補正量を **次 timepoint** の XYStage 指令値に加算（feedforward）
9. `crop_sub_root\PosN\chXX\` に **grid_subtract 済み** crop（細胞のみ位相）を保存
10. `drift_log.json` に各 timepoint の `(tx, ty, corr, kf_pos)` を記録

> ※ `STEP_GAUSSIAN_BACKSUB` は廃止。tilt_correct（slope+intercept）で背景・傾きとも処理する方針。

**設定の主要パラメータ**（`drift_config.json` 内、`prepare_drift_session.py` から自動生成）:
| key | 意味 | 既定 |
|---|---|---|
| `n_timepoints` | 総 frame 数 | 3168 |
| `interval_sec` | frame 間隔 | 300 |
| `pixel_scale_um` | 再構成画像 1 px の物理サイズ | 0.346 µm/px |
| `tilt_crop_h` / `ecc_crop_h` | 傾き fit 用 / ECC 用の crop 幅 | 270 / 80 |
| `ecc_vmin` / `ecc_vmax` | uint8 正規化レンジ | -5.0 / 2.0 |
| `enable_crop_sub_save` | `crop_sub_root` への ch_subtracted 保存 | true |
| `crop_sub_root` | ch_subtracted 出力先 | `C:\YYMMDD\online_crop_sub` |
| `kf_Q_ty_nm2` / `kf_Q_tx_nm2` | KF プロセス分散 | 291 / 877 |
| `kf_R_ty_nm2` / `kf_R_tx_nm2` | KF 観測分散 | 91 / 274 |
| `max_total_corr_um` | 累積補正の絶対上限 | 15 µm |
| `enable_third_pass` | 3-pass ECC（精度↑） | true |

---

### 7.4 培地切り替えオペレーション

- 切替は手動（バルブ操作）。所要時間は流路置換 5 min 程度
- **必ずフレーム番号を記録する**（後段の `correct_0pergluc.py` の `GLUCOSE_*_START/END` で必要）
- 例: `2per: 0–287, Lowper: 288–575, 0per: 576–1151, 2per: 1152–`（数値は実験ごとに TBD）

---

## 8. Phase 7 — オフライン補正

### 8.1 `correct_0pergluc.py` — 0% 期間 frame に delta を引き算

**何をするか**:
1. `OUTPUT_DIR`（compute_drift_online が吐いた `crop_sub_rawraw` 配下）の各 frame を走査
2. `grid_subtract_log.json` から **その frame に紐付く grid 点** `(xi, yi)` を取得
3. `DELTA_TIFS_DIR/delta_z{GRID_Z_INDEX:03d}.tif` をロード（`extract_timelapse_delta.py` の出力）
4. `grid_calibration_{base_label}.json` から `(cal_dx(xi,yi), cal_dy(xi,yi))` を取り、`apply_inverse_shift_warp` で delta を当該 frame の座標系に warp
5. `extract_rect_roi(40 × 270)` → `apply_2pi_tilt_crop` で 40 × 180 に crop
6. ch_subtracted frame から warp+crop された delta を引き算
7. `crop_sub_rawraw_0per_corr/chXX/*.tif` 等に上書き保存

**条件**: `GLUCOSE_0_START ≤ frame_index < GLUCOSE_0_END` の frame のみ処理。

**編集箇所** (`scripts/correct_0pergluc.py`):
```python
PH_SESSION_ROOT       = r"D:\AquisitionData\Kitagishi\260423\ph_1"
CHANNEL_OUTPUT_SUBDIR = "crop_sub_rawraw"
POS_NUMBERS_TO_RUN    = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
GRID_2PER_DIR         = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1"
GRID_CALIBRATION_JSON = r"D:\AquisitionData\Kitagishi\260423\grid_2pergluc_1\grid_calibration_Pos1.json"
GLUCOSE_0_START       = 576    # ★ 本番タイムラプスの 0% 期間 frame（要記録から反映）
GLUCOSE_0_END         = 1152
DELTA_TIFS_DIR        = r"D:\AquisitionData\Kitagishi\260423\0per_zstack_1\Pos1\output_phase\channels\delta_timelapse"
GRID_Z_INDEX          = 5
N_PARALLEL_FRAMES     = 8
```

> Low% 期間も同様に補正するなら、`DELTA_TIFS_DIR` を Low% delta に切り替えて、`GLUCOSE_0_START/END` を Low% 期間に書き換えて再実行する。

**実行**:
```bash
python scripts/correct_0pergluc.py
```

**出力**: `…\PosN\output_phase\channels\crop_sub_rawraw_0per_corr\chXX\*.tif`

---

## 9. Phase 8 — セグメンテーション

### 9.1 Omnipose GUI で training データ作成

実験ごとに条件（培地・撮影条件・光学系）が変わるので、**実験ごとに学習し直す**方針。

1. `correct_0pergluc.py` 出力（or 補正不要なら ch_subtracted）から代表的な frame を抽出
2. Omnipose GUI でアノテーション（〜数百〜数千 細胞分の ground truth mask 作成）
3. 必要なら `12_vertical_flip.py` `26_horizontal_flip.py` でデータ拡張
4. `08_train.py` で学習（`normalize=False, rescale=False` が重要）
5. モデルを `C:\Users\QPI\Desktop\train\omni_model\models\` に保存

---

### 9.2 `07_segmentation.py` — Omnipose 推論

**何をするか**:
1. `indir` 配下の TIF をリストアップ
2. `CellposeModel(pretrained_model=model_path, omni=True, nchan=1, nclasses=3)` をロード（GPU 使用）
3. 各 image を `tifffile.imread` → `model.eval()` で mask 推論
   - `diameter=30, normalize=True, tile=False, omni=True`
   - `flow_threshold=0.11`（low: 検出点の確保）
   - `mask_threshold=0`, `min_size=10`
4. `outdir = indir/inference_out/` に `*_masks.tif`（uint16 ラベル）を保存
5. エラー frame は空 mask で書く

**編集箇所** (`scripts/07_segmentation.py`):
```python
indir = r"D:\AquisitionData\Kitagishi\260423\ph_1\Pos2\output_phase\channels\crop_sub_rawraw_0per_corr\ch01"
model_path = r"C:\Users\QPI\Desktop\train\omni_model\models\cellpose_residual_..._2026_04_13_10_54_41.173761"
USE_GPU = True
```

**実行（全 Pos × 全 ch を回す例）**:
```bash
for POS in 1 2 3 4 5 6 7 8 9 10 11; do
  for CH in 00 01 02 03 04 05 06 07 08 09 10 11; do
    python -c "
import sys; sys.path.insert(0,'scripts')
import importlib, runpy
import os
os.environ['INDIR'] = rf'D:\AquisitionData\Kitagishi\260423\ph_1\Pos$POS\output_phase\channels\crop_sub_rawraw_0per_corr\ch$CH'
runpy.run_path('scripts/07_segmentation.py')
"
  done
done
```
※ `07_segmentation.py` 側で `indir` を環境変数から読むように1行書き換えると上のループが綺麗になる。

**出力**: `…\chXX\inference_out\*_masks.tif`

---

## 10. Phase 9 — 細胞 RI / dry mass

### 10.1 ImageJ での ROI tracking → Results.csv

各細胞を 1-by-1 で trace（Mother Machine のチャネル奥の細胞を継承）し、各 frame で楕円 fit パラメータを取得 → `Results.csv` に書き出す:

| 列 | 意味 |
|---|---|
| Label | 細胞 ID + frame |
| Major | 楕円長径 [px] |
| Minor | 楕円短径 [px] |
| X, Y | 重心座標 |
| Angle | 楕円の傾き |
| Slice | frame index（1-indexed）|
| Area | mask 面積 [px²] |

> 自動 tracking は `central_cell_lineage_tracker.py` でカバーされるが、人手で詳細に追いたい場合や training に使う場合は ImageJ。

---

### 10.2 `32_simple_ellipse_ri.py` — Rod 体積近似で細胞 RI

**何をするか**:
1. `Results.csv` を読み、ROI ごとに `Slice` から対応する `*_subtracted.tif` をロード
2. `make_ellipse_mask(X, Y, Major, Minor, Angle)` で楕円マスクを生成
3. **Rod shape 体積**（カプセル型）を計算:
   ```
   length = Major × pixel_size_um
   width  = Minor × pixel_size_um
   r = width / 2
   h = length - 2r
   V = (4/3)πr³ + πr²·h    (h ≥ 0; h<0 なら球)
   ```
4. **位相積分から Δn を算出**:
   ```
   Σφ = Σ phase[mask] · pixel_area
   Δn = (Σφ · λ) / (2π · V)
   n_cell = n_medium + Δn
   ```
5. **dry mass**:
   ```
   m = Δn · V / α      (α = 0.18 mL/g = 0.00018 mL/mg)
   ```
6. 各 ROI の (n_cell, dry_mass, V, length, width) を時系列で plot → `figure_logger.save_figure()` で保存

**編集箇所** (`scripts/32_simple_ellipse_ri.py`):
```python
RESULTS_CSV  = r"D:\AquisitionData\Kitagishi\260423\analysis\Pos2_ch01\Results.csv"
IMAGE_DIR    = r"D:\AquisitionData\Kitagishi\260423\ph_1\Pos2\output_phase\channels\crop_sub_rawraw_0per_corr\ch01"
OUTPUT_FILE  = "simple_mean_ri.png"

PIXEL_SIZE_UM = 0.348
WAVELENGTH_NM = 658
N_MEDIUM      = 1.333    # ★ calibrate_ri.py の結果 (n_2per) で更新。培地切替で時刻ごとに変えるなら別実装が必要
ALPHA_RI      = 0.00018  # mL/mg (= 0.18 mL/g)
```

**実行**:
```bash
python scripts/32_simple_ellipse_ri.py
```

**出力**:
- `figure_logger` 経由で `results/figures/` + figure inbox JSON
- 標準出力に各 ROI の RI / dry_mass

---

## 11. Phase 10 — 図生成

### 11.1 `central_cell_lineage_tracker.py` — Mother machine 系譜トラッキング

**何をするか**:
1. `--indir` 配下の `*.tif`（生位相）と `inference_out/*_masks.tif`（segmentation）をペアでロード
2. 各 frame で mask の重心を計算 → 画像 x 中心からの距離で **rank** をつける（rank 0 = 最も中央 = mother）
3. **2-pointer rank iteration** で frame 間の ID 伝播:
   - `area_ratio > DIV_AREA_RATIO_MIN` → 同一 ID（continuation）
   - `curr[r] + curr[r+1] ≈ prev[r_prev]` → 分裂（inner=parent, outer=new daughter）
   - どれにも該当しない → continuation + `is_outlier` フラグ
4. mother の子孫だけを系譜木に残す（frame 0 の rank ≥ 2 は bookkeeping のみ）
5. 各細胞・各 frame で楕円 fit → Major, Minor → Rod 体積 → mean phase → RI / dry mass
6. 出力:
   - `lineage_table.csv`: per-frame per-cell metrics
   - `lineage_cells.json`: per-cell birth/death/parent
   - `figure_logger` 経由で系譜木 PDF + volume / RI 時系列図

**実行**:
```bash
python scripts/central_cell_lineage_tracker.py \
  --indir /Volumes/2604/260423/ph_1/Pos9/output_phase/channels/crop_sub_rawraw_0per_corr/ch00 \
  --pixel-size-um 0.348 \
  --time-interval-min 5
```

**出力**:
```
…/inference_out/lineage_out/lineage_table.csv
                            /lineage_cells.json
results/figures/<figure_id>_lineage_tree.pdf
                <figure_id>_volume_timeseries.pdf
                <figure_id>_RI_timeseries.pdf
```

---

### 11.2 `qpi_fig_03_lineage_analysis.py` — 系譜解析図

**何をするか**: ImageJ ROI Results.csv（or `lineage_table.csv`）から以下 10 種の解析:
1. 個別細胞の area / RI 時系列（分裂イベント検出付き）
2. 集団 mean ± SEM（area / RI）
3. Birth size vs Added size（sizer / adder / timer 分類）
4. Birth RI vs Added RI（dry mass homeostasis）
5. 分裂間隔ヒストグラム
6. 分裂間隔 per generation
7. 細胞周期で揃えた trajectory（Area / RI / Dry mass）
8. RI 分布ヒストグラム + Gaussian fit
9. Density homeostasis（birth RI vs ΔRI）
10. Growth rate（dArea/dt, dMass/dt）

**実行**:
```bash
python scripts/qpi_fig_03_lineage_analysis.py
```

各種図を `figure_logger` 経由で保存。

---

### 11.3 補助図

- `qpi_fig_01_reconstruction_procedure.py`: QPI hologram → 位相再構成プロシージャの 6-panel 教科書図（thesis 用）
- `qpi_fig_02_visibility.py`: hologram から visibility 計算プロシージャの 6-panel 図（thesis 用）

これらは「実験ごとに必ず」ではなく、**論文・thesis の section 用に必要時のみ実行**。

---

## Appendix A — 出力ファイル一覧

| Phase | 出力 | 場所 |
|---|---|---|
| 4.1 | grid.pos | `C:\YYMMDD\` |
| 4.3 | grid output_phase TIF | `…\grid_2pergluc_1\PosN_x±i_y±j\output_phase\img_*_ph_*_phase.tif` |
| 4.4 | channel_rois.json | `…\PosN_x+0_y+0\output_phase\channels\` |
| 4.5 | grid_calibration_PosN.json | `…\grid_2pergluc_1\` |
| 5.2 | delta_z{Z:03d}.tif | `…\0per_zstack_1\PosN\output_phase\channels\delta_timelapse\` |
| 7.1 | drift_config.json + state | `C:\Users\QPI\Documents\QPI_Omni\drift_session\` |
| 7.3 | ch_subtracted TIF | `…\ph_1\PosN\output_phase\channels\crop_sub_rawraw\chXX\` |
| 7.3 | drift_log.json | `…\drift_session\` |
| 8.1 | 0% 補正済み TIF | `…\PosN\…\crop_sub_rawraw_0per_corr\chXX\` |
| 9.2 | mask | `…\chXX\inference_out\*_masks.tif` |
| 10.2 | RI / dry mass 図 + JSON | `results/figures/`, `figure_inbox/` |
| 11.1 | lineage_table.csv / lineage_cells.json | `…\chXX\inference_out\lineage_out\` |

---

## Appendix B — QPI 理論（最低限）

**位相再構成（オフアクシス干渉）**
1. 入力: hologram `I(x, y)` = `|R + S|² = |R|² + |S|² + 2|R||S|·cos(2π·k_c·x + φ(x,y))`
2. 2D FFT → 周波数空間で 0次成分とサイドバンドが分離
3. サイドバンドを (0, 0) 中心に shift（`OFFAXIS_CENTER` の指定箇所）
4. LP filter で ±NA/λ より外を 0 に
5. 2D IFFT → 複素場 `E(x, y) = A(x, y) · exp(iφ(x, y))`
6. `np.angle(E)` → 位相、`unwrap_phase` でラップ解除

**Dry mass（Barer 1952, Davies & Wilkins 1952）**
```
m_dry = (1/α) · ∫∫ Δn(x, y) · A_pixel  dxdy
       = (1/α) · (λ / 2π) · ∫∫ φ(x, y) · A_pixel  dxdy
α ≈ 0.18 mL/g  (specific refractive increment, 平均的タンパク質)
```

**細胞内平均 RI**
```
n_cell = n_medium + ΣΔn[mask] · A_pixel / V_total
V_total: rod shape (capsule) approximation (Phase 10.2)
```

---

## Appendix C — 廃止された旧 doc / 旧パイプライン

このプロトコルへの一本化に伴い、以下を削除:

- `PIPELINE.md` — 旧パイプライン（`channel_crop` → `gaussian_backsub` → `compute_pos_shifts` → `grid_subtract` の分割実行）。`compute_drift_online.py` で統合済み。
- `docs/USAGE_GUIDE.md` — 旧 24/31/32 系の使い方
- `docs/METHODS.md` — 理論記述。本 doc Appendix B に圧縮
- `docs/ANALYSIS_FLOW_CURRENT.md` — 本 doc が後継
- `docs/QUICK_START_ROTATIONAL_SYMMETRY.md`, `docs/CHANGELOG_ROTATIONAL_SYMMETRY.md`, `docs/workflows/rotational_symmetry_volume_workflow.md` — 回転対称体積推定（採用していない手法）
- `docs/COMMUNITY_REPRO_GUIDE.md` — 旧コミュニティ向け再現ガイド
- `docs/Mac_Laptop_Cursor_Setup.md` — セットアップ作業ログ
- `docs/workflows/timeseries_volume_tracking_guide.md`, `2025-12-23_timeseries_total_mass.md`, `thickness_map_and_ri_calculation.md`, `micromanager_realtime_visibility.md` — 旧 workflow doc
- `README.md` — 古く誤情報あり

`gaussian_backsub` 自体（`scripts/19_gaussian_backsub.py`）は比較診断用に残置するが、本番パイプラインでは使わない。tilt_correct（slope+intercept fit、`tilt_utils.tilt_fit_crop`）が標準。

---

## TBD（実験ごとに埋める）

- 各培地段階の **frame 範囲**（Phase 7.4 のメモ + Phase 8 の `GLUCOSE_*_START/END`）
- 培地切替の正確な **タイミング表**
- Omnipose model のパス（毎回学習し直すので毎回更新）
- ImageJ ROI Results.csv の保存先
