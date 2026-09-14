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
15. python correct_0pergluc.py            → 0% 期間 frame に delta を warp 引き算（必要なときだけ）
16. (任意) Omnipose GUI で training 画像を作成 → 08_train.py で学習 → 採用 checkpoint を models/ に置く
17. (任意) python calibrate_ri.py         → 培地 RI を MilliQ + EtOH 2点法で校正
18. datasets/<YYMMDD>.yaml を書く          → パス・培地切替 frame・RI 校正・bad frames・model
19. python scripts/run_dataset_pipeline.py datasets/<YYMMDD>.yaml --plan   → 何が走るか確認
20. python scripts/run_dataset_pipeline.py datasets/<YYMMDD>.yaml --stages seg,track,qc,consolidate,publish --tag v<YYYYMMDD>_<note>
    → seg（GPU）→ 全 ch の lineage tracking → 分裂 QC → 集約 → master 公開（<master_root>/<tag>/）
21. 解析・図は master から読む（qpi_paths.resolve_lineage_csv）。論文用の窓は build_phase1_dataset_260517.py
```

Python は `environment/` の pinned env（Windows: `powershell -ExecutionPolicy Bypass -File environment\bootstrap_windows.ps1`）。

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

1. `correct_0pergluc.py` 出力（補正不要なら `crop_sub_rawraw`）から代表的な frame を抽出する。`sample_train_frames.py` / `_build_trainset_260517.py` が 2% 増殖期・回復期を密に、飢餓期を疎に、複数 Pos × ch から取る（端 trap ch00 / ch11 は入れない）
2. Omnipose GUI でアノテーション（数百〜数千細胞分）
3. 必要なら `26_horizontal_flip.py` でデータ拡張
4. `08_train.py` で学習（`normalize=False, rescale=False`）。checkpoint は `C:\Users\QPI\Desktop\train\omni_model_d20\models\` に出る。`checkpoint_eval.py` / `checkpoint_overlay_runner.py` で epoch を選ぶ
5. 採用した checkpoint を **`models/`（リポジトリ直下）に短い名前でコピーし、`models/MODELS.json` に sha256・学習日・学習セット・eval を書く**。`datasets/<YYMMDD>.yaml` の `segmentation.model` はこの短い名前を指す

現行（260517、2026-09-07 学習）: `models/omni_model_d20_2026_09_07_12_41_31.782047`

---

### 9.2 `seg_omnipose.py` — Omnipose 推論（GPU のみ）

`run_dataset_pipeline.py --stages seg` から Pos ごとに起動される。単体でも動く。

- 入力: `<raw_root>/PosN/<channel_rel>/chNN/img_*_ph_000_phase.tif`（位相 crop、radian）
- 出力: `<mask_root>/PosN/<channel_rel>/chNN/inference_out/img_*_ph_000_phase_masks.tif`（uint16 ラベル）。**細胞が検出された frame だけ**書く。ch が終わると `_DONE`（再実行時は skip）
- 空フレーム gate: 位相 > 0.7 rad の画素が 40 未満なら model を呼ばない
- eval: `diameter=20, normalize=True, tile=False, omni=True, flow_threshold=0.11, mask_threshold=0, min_size=10, net_avg=False`（yaml の `segmentation.eval`）
- ch 単位の ProcessPool（既定 6 worker、worker ごとに model を 1 回ロード）。**CPU では走らせない**（backend で mask が変わる）
- 260517 の Pos1〜104 を出した driver は `run_seg_260517_gpu.py` にそのまま残してある（同じ eval・同じ gate）

```powershell
<env>\python.exe scripts\seg_omnipose.py --raw-root D:\AquisitionData\Kitagishi\YYMMDD\ph_1 --mask-root E:\YYMMDD_seg `
    --channel-rel output_phase/channels/crop_sub_rawraw --model models\omni_model_d20_2026_09_07_12_41_31.782047 `
    --pos-start 1 --pos-end 12 --workers 6
```

`07_segmentation.py` は 1 ディレクトリを手で切るときの旧 CLI（`--indir --model-path --frame-min/--frame-max`）。

---

## 10. Phase 9 — 系譜 tracking と細胞ごとの RI / dry mass / volume

### 10.1 `central_cell_lineage_tracker.py`

ch ごとに 1 回。`run_dataset_pipeline.py --stages track` が masks のある ch だけを回す（production 済みの ch は skip）。

**入力**: `--indir <mask ch>`（`inference_out/*_masks.tif`）、`--raw-dir <phase ch>`（位相 crop。mask と別ディスクでよい。H: のような読み取り専用ディスクには何も書かない）

**処理**:
1. drift session の `bad_frames.json` にある frame を linking の前に除外（測定値は `lineage_bad_frames.csv` に出す）
2. 画像端に触れる mask を linking から落とす（trap から出た細胞）
3. 各 frame の mask を trap 奥からの距離で rank 付け（rank 1 = mother）。frame 間の対応は面積で決める: 面積比 > 0.68 なら同一細胞、`|(a+b) − prev| / prev < 0.30` なら分裂（内側が親・外側が娘）、どちらでもなければ `is_outlier=True` の行として残す（ID は切れない。3 frame 規則: 比 < 0.30・> 1.50・< 1/1.8 も outlier）
4. mother の子孫を系譜木（`in_tree`）に入れる。木の外の細胞も全 frame 測る
5. 各細胞・各 frame で **黄色輪郭**（`mask_volume_schematic.efd_section_geometry`: mask 境界を EFD K=6 で平滑化し 0.5 px 内側へ縮め、中心線の中点更新を 1 回）から長軸・短軸・体積を取る
   - `volume_um3_rod`: 黄色の長軸・短軸からのカプセル
   - `volume_um3_efd`: 黄色の弦の回転体積分 Σπ(w/2)²Δs（**採用**）
6. RI と dry mass: `--ri-calibration` の JSON と `--media-schedule`（絶対 img 番号 → wo_* の対応）から frame ごとの n_medium を決め、`Δn = Σφ · λ / (2π · V)`、`n_cell = n_medium + Δn`、`m = Δn · V / α`（α = 0.00018 mL/mg）。`mean_ri` / `mass_pg` / `density_pg_um3` は rod 体積、`mean_ri_efd` / `mass_pg_efd` / `density_pg_um3_efd` は efd 体積から
7. `--frame-min 2` で img_0 / img_1 を落とし、img_2 を time 0 h にする

**出力** (`<mask ch>/inference_out/lineage_out/`): `lineage_data3D.csv`（細胞 × frame）、`clist.csv`（細胞ごと）、`lineage_cells.json`、`lineage_bad_frames.csv`、`bad_frames_used.json`、`lineage_run_params.json`。列は `docs/LINEAGE_DATAFRAME_SCHEMA.md`

medial-axis・profile・skimage の楕円近似は出さない（2026-09-14 決定）。手法の対比は `_fig_volume_method_comparison_260517.py` の図だけに残す（黄色 efd は旧 medial profile の約 0.80 倍。差は −0.5 px の縮めが全部）。ImageJ の ROI tracking と `32_simple_ellipse_ri.py` は使わない（archive）。

---

### 10.2 `division_qc_260517.py` — 分裂判定の検証

tracker の分裂判定は 1 frame の面積だけなので、一時的な mask 分裂が偽の娘を作る。各候補を親の `mass_pg_efd` / `volume_um3_efd` の前後比で検証する:

- ±1 frame に outlier がなければ direct 採用
- あれば ±8 frame の有効 2〜3 点の中央値で post/pre mass 0.25〜0.78、volume 0.25〜0.85、両比の差 ≤ 0.25 なら rescued
- 有効点が 2 未満なら insufficient、範囲外なら rejected、検証済みイベントから 12 frame 以内の rescued は duplicate

出力 `divisions_qc.csv`（per ch: `validated`, `method`, 比）、集約 `all_cells_divisions_qc.csv.gz`。`run_dataset_pipeline.py --stages qc` と consolidate の中で自動実行（`lineage_data3D.csv` より新しい結果があれば skip）。

---

### 10.3 集約と master 公開 — `run_dataset_pipeline.py`

```powershell
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --stages consolidate,publish --tag v<YYYYMMDD>_<note>
```

- consolidate: production 判定（`lineage_run_params.json` の media_schedule / frame_min が yaml と一致し、CSV に `volume_um3_efd` 列がある）を満たす ch を `<mask_root>/_lineage_consolidated/` に連結（`all_cells_lineage_data3D.csv.gz`, `all_cells_clist.csv.gz`, `all_cells_lineage_bad_frames.csv.gz`, `all_cells_divisions_qc.csv.gz`, `channel_index.csv`, `manifest.json`）
- publish: `<master_root>/<tag>/` に凍結。`consolidated/`、`per_channel/PosN/chNN/`、`inputs/`（RI 校正・bad_frames・dataset yaml・channel classification・model checkpoint）、`code/`（tracker・幾何モジュール・driver・git HEAD）、`qc/`、`MANIFEST.json`、`SHA256SUMS.txt`（LF）、`README.md`、`SCHEMA.md`。全ファイル読み取り専用。`LATEST.txt` を更新。G: のミラーは per_channel 抜き
- **解析はすべて master から読む**。`qpi_paths.resolve_lineage_csv()` が master を最優先で解決する（`QPI_LINEAGE_MASTER=<tag>` で版固定、`QPI_LINEAGE_SOURCE=inbox` で旧 inbox データ）。作業ツリー `<mask_root>` を直接読む解析は書かない

260517 は固有 chain（`_retrack_260517_newmodel.py` → `_chain_tiltfix_260517.py` → `_finalize_yellow_260517.py` → `publish_master_260517.py`）で公開している。処理は driver と同じで、`datasets/260517.yaml` が同じ設定を記録している。

---

### 10.4 派生パッケージ（論文用の窓）

`build_phase1_dataset_260517.py` が master から `derived/phase1_img0002-2017/` を生成（publish 時に自動。yaml の `derived`）。窓は **img_2〜img_2017 = 2016 frames = 168 h**。img_2018 は培地切替（予定 img_2019）の光学的影響を既に受けている（母細胞 total_phase +7%、52 Pos 中 41 Pos）ため除外。再 tracking はせず、per-frame 行は master のまま、per-cell 要約だけ窓内で再計算し打ち切りフラグ（`present_at_window_start` / `alive_at_window_end`）を付ける。中身: cells_frames / cells / divisions（`validated` 付き）/ channels（`edge_channel`, `analysis_recommended`）/ excluded_frames / bad_frame_measurements + README / SCHEMA / MANIFEST / SHA256。

端 trap **ch00 / ch11 は master に残すが解析対象から外す**（`channels.csv: analysis_recommended`。2026-09-14 決定）。

---

## 11. Phase 10 — QC と図

- `_fig_mother_lineage_qc_260517.py`: Pos ごとに全 ch の mother lineage（分裂線つき）を並べ、tracking の破綻を目で見る
- `_fig_switch_frame_check_260517.py`: 培地切替 frame 前後の mother total_phase / RI（切替の光学的影響が何 frame 前から出るか）
- `lineage_html_gallery_260517.py --source master`: lineage ごとに mean_ri / 位相積分 mass / volume の 3 段 plot を共通 time 軸で HTML に並べる（有効 = 濃青、outlier = 赤 ×、border = 橙 △、drift 除外 = 紫 ◇、検証済み分裂 = 灰の縦線、cycle ごとの ln(mass) 直線 fit = 赤線。y 範囲 1.37–1.40 / 0–50 pg / 0–120 µm³）
- 論文図 `_fig_*.py` は黄色 master と `channels.csv: analysis_recommended` から組み直す。図は `figure_logger.save_figure(data=, caption=)` で保存し、JSON サイドカーに context を書く（`docs/FIGURE_SPEC.md`）
- `qpi_fig_01_reconstruction_procedure.py` / `qpi_fig_02_visibility.py` は thesis の手法図（必要時のみ）

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
| 7.3 | 位相 crop（ch_subtracted TIF） | `…\ph_1\PosN\output_phase\channels\crop_sub_rawraw\[z000\]chXX\` |
| 7.3 | drift_log.json / grid_subtract_log.json / bad_frames.json | `…\drift_session\` |
| 8.1 | 0% 補正済み TIF | `…\PosN\…\crop_sub_rawraw_0per_corr\chXX\` |
| 9.2 | mask + `_DONE` | `<mask_root>\PosN\…\chXX\inference_out\*_masks.tif` |
| 10.1 | lineage_data3D.csv / clist.csv / lineage_cells.json / lineage_run_params.json | `<mask_root>\PosN\…\chXX\inference_out\lineage_out\` |
| 10.2 | divisions_qc.csv | 同上 |
| 10.3 | all_cells_*.csv.gz / channel_index.csv / manifest.json | `<mask_root>\_lineage_consolidated\` |
| 10.3 | master（読み取り専用） | `<master_root>\<tag>\` + `LATEST.txt`、ミラー `G:\共有ドライブ\wakamotolab_meeting\kitagishi\data_master\<YYMMDD>\<tag>\` |
| 10.4 | phase-1 パッケージ | `<master_root>\<tag>\derived\phase1_img0002-2017\` |
| 11 | 図 + JSON サイドカー | `results/figures/`, figure-hub inbox |

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
V_total: 黄色輪郭の回転体積分 volume_um3_efd（10.1）。volume_um3_rod は同じ輪郭の長軸・短軸からのカプセル
```

---

## Appendix C — 廃止された旧 doc / 旧パイプライン

このプロトコルへの一本化に伴い、以下を削除（2026-09-09 実施。`docs/README.md`・`scripts/README_VOLUME_TRACKING.md`・`AGENTS.md.bak.*` も同時に削除）:

- `PIPELINE.md` — 旧パイプライン（`channel_crop` → `gaussian_backsub` → `compute_pos_shifts` → `grid_subtract` の分割実行）。`compute_drift_online.py` で統合済み。
- `docs/USAGE_GUIDE.md` — 旧 24/31/32 系の使い方
- `docs/METHODS.md` — 理論記述。本 doc Appendix B に圧縮
- `docs/ANALYSIS_FLOW_CURRENT.md` — 本 doc が後継
- `docs/QUICK_START_ROTATIONAL_SYMMETRY.md`, `docs/CHANGELOG_ROTATIONAL_SYMMETRY.md`, `docs/workflows/rotational_symmetry_volume_workflow.md` — 回転対称体積推定（採用していない手法）
- `docs/COMMUNITY_REPRO_GUIDE.md` — 旧コミュニティ向け再現ガイド
- `docs/Mac_Laptop_Cursor_Setup.md` — セットアップ作業ログ
- `docs/workflows/timeseries_volume_tracking_guide.md`, `2025-12-23_timeseries_total_mass.md`, `thickness_map_and_ri_calculation.md`, `micromanager_realtime_visibility.md` — 旧 workflow doc
- `README.md` — 古く誤情報あり（本 doc を指す短い索引に置き換え）

2026-09-14 に、`19_gaussian_backsub.py`・`32_simple_ellipse_ri.py`・`36_align_and_subtract_timelapse.py`・`10_batch_reconstruction_new.py`・`qpi_fig_03_lineage_analysis.py` を含む 260 本の旧スクリプトを `scripts/archive/2026-09-14_reorg/` へ退避した（索引は `scripts/README.md`、退避理由は archive 側の README）。tilt 補正は `tilt_utils.tilt_fit_crop`（slope + intercept fit）が標準。

---

## 実験ごとに決めること（`datasets/<YYMMDD>.yaml` に書く）

- 各培地段階の **frame 範囲**（`tracking.media_schedule`。Phase 7.4 のメモ、`correct_0pergluc.py` の `GLUCOSE_*_START/END` と同じ数値）
- RI 校正 JSON（`tracking.ri_calibration`、Phase 5.3）と drift session の `bad_frames.json`（`tracking.bad_frames`）
- Omnipose checkpoint（`segmentation.model`、`models/` の短い名前）と学習セットの記録（`models/MODELS.json`）
- `raw_root` / `mask_root` / `master_root` / `channel_rel`（z-stack timelapse なら `z000` を含む。0% 補正後なら `crop_sub_rawraw_0per_corr`）
- tilt fit の側（Pos によって trap の向きが違うなら `batch_grid_subtract_260517.py` の `TILT_POS_SPLIT`。crop の切替 `POS_SPLIT` とは別の境界）

### 260517 の記録

- model: `models/omni_model_d20_2026_09_07_12_41_31.782047`（2026-09-07 学習）。masks `D:\260517_seg\PosN\output_phase\channels\crop_sub_rawraw\z000\chNN\inference_out`、位相 crop は `H:\260517\2per_0055per_0per_2per_crop_sub`（読み取り専用）
- 培地: `0:wo_2,2019:wo_0p0055,2307:wo_0,2885:wo_2`（0.0055% は 0% の RI を使う）。RI 校正 `H:\260517\grid_2pergluc_2\ri_calibration_results.json`（wo_2 = 1.33503、wo_0 = wo_0p0055 = 1.33274、n_milliq = 1.3312）
- **tilt fit の側**: Pos ≤52 は開口端が左なので左 1/3、Pos ≥53 は trap が鏡像（細胞が左・開口端が右）なので右 1/3。2026-09-14 に、grid_subtract が `PosN\z000` から Pos 番号を読めず全 Pos を左 fit していたことが判明（Pos ≥53 の背景が −0.1〜−1.7 rad に沈み total_phase / RI / mass が偏る。mask はほぼ不変）。保存 crop は fit 幅 270 そのままなので `refit_tilt_right_260517.py` で右 1/3 に fit し直した結果は元パイプラインの `fit_right=True` と同一。修正後の位相は `D:\260517_tiltfix`（yaml の `raw_root_overrides`）。grid_subtract は Pos 番号が判別できないとき停止するようにした
- 培地切替の影響: img_2018 で母細胞の total_phase +7〜8%・RI +0.0013（52 Pos 中 41 Pos）。論文用の窓は img_2〜img_2017
- master: `D:\QPI_master\260517\<tag>\`（`v20260911_newmodel` → tilt fix と黄色幾何を反映した `v20260915_yellow` に置き換え予定）
