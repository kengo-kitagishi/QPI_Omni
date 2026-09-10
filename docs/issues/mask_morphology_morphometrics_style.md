# Replace skimage second-moment axis fit with Morphometrics-style width measurement in `mask_morphology.py`

> このファイルは GitHub Issue 用本文。解析 PC で以下のように issue を立てる:
>
> ```bash
> gh issue create --repo kengo-kitagishi/QPI_Omni \
>     --label "enhancement,someday" \
>     --title "Replace skimage second-moment axis fit with Morphometrics-style width measurement in mask_morphology.py" \
>     --body-file docs/issues/mask_morphology_morphometrics_style.md
> ```

---

## TL;DR

現在 `central_cell_lineage_tracker.py` が `lineage_data3D.csv` に書き込んでいる `short_axis_um` / `long_axis_um` / `volume_um3_rod` は、`skimage.measure.regionprops` の二次モーメント楕円フィット (`major_axis_length` / `minor_axis_length`) を素直に使っている。これは S. pombe の rod 形状に対して **aspect ratio 依存の系統的 bias** を持ち、cycle aligned 解析で偽の幅増加 (+6 %) を生む。

理論モデル (`scripts/rod_axis_correction.py`) で補正したところ「+6 % のうち +2.6 % は二次モーメント artifact、残り +3.5 % は biological」と判定された。本 issue は **mask データから直接 Morphometrics 風の幅測定 (rotate-and-project + adaptive trim、および medial-axis perpendicular width)** を実装し、理論補正と実測を直接比較して補正の妥当性を確定させることが目的。

完了後の状態:
- `scripts/mask_morphology.py` が **3 通りの幅測定モード** (`fixed_trim`, `adaptive_trim`, `medial_axis`) を持つ
- driver `scripts/recompute_axes_from_masks.py` が segmentation mask + lineage_data3D を入力に、補正済み長軸/短軸/体積を CSV 出力する
- gold-standard mother 31 channel で **skimage / rod_corrected / mask_morphology(adaptive_trim) / mask_morphology(medial_axis)** の 4 way 比較プロット
- 真の cycle 内 short axis 変動率を確定 (期待値 +3.5 % 前後)

---

## Background

### 現状の数値 (gold-standard 31 ch, 1618 cycles, phase1 only)

| 指標 | skimage (raw) | rod-corrected (理論) | Δ% raw | Δ% corrected |
|---|---|---|---|---|
| short axis [μm] | 4.11 → 4.36 | 3.76 → 3.89 | +5.96 % | +3.47 % |
| long axis [μm] | 8.31 → 14.88 | 7.86 → 13.63 | +56.19 % | +53.29 % |
| rod volume [μm³] | 92 → 201 | 74 → 147 | +74.36 % | +66.86 % |

文献整合: corrected の cycle 末 volume 147 μm³ ↔ S. pombe div volume 文献値 ~140 μm³ で 3 % 以内一致。

### Bias の原因

`skimage.measure.regionprops().minor_axis_length = 4 √(λ_min)` (covariance matrix 最小固有値)。理想 ellipse なら真の幅を返すが、rod (cylinder + 2 semicircular caps) に対しては aspect ratio AR = L/(2r) 依存で過大評価:

- AR=1 (球): 1.000 × 真の幅
- AR=2: 1.089 × 真の幅
- AR=3: 1.113 × 真の幅
- AR=5: 1.131 × 真の幅
- AR→∞: 1.155 × 真の幅 (理論極限)

cycle 内で AR が 2.0 → 3.4 と変動するので、skimage minor axis も同方向に変動 → 偽の幅増加。

### 関連論文

- Ursell et al. 2017 BMC Biology, DOI: 10.1186/s12915-017-0348-8 (Morphometrics)
- Stylianidou et al. 2016 (SuperSegger): rotate-and-project の元祖
- 数学的等価性: medial axis に垂直な弦長で width 測定 ≡ Morphometrics 風 ≡ 二次モーメント artifact を持たない

---

## Goal

1. mask データから直接、二次モーメント artifact を持たない width 測定を実装
2. gold-standard mother で 4 way 比較し **rod_axis_correction の妥当性を実測で検証**
3. (将来) 全 mother (revived + never_revived) で再計算し、phase2 dynamics 解析を補正版データで再評価

---

## Implementation plan

### Task A: `mask_morphology.py` に **adaptive endcap trim** を追加

現在 `_measure_single_cell()` は固定 `endcap_trim_frac=0.10` を使う (line 137-138 付近)。短い細胞では cap が trim 範囲を超えて中央側に侵入し、width が過小評価される (cycle 開始時の bias の一因)。

**変更**: `endcap_trim_frac` パラメータに以下のいずれかを許容:
- `float`: 固定値 (現状の挙動、後方互換)
- `"adaptive"` (文字列): cap 長 ≈ rough_short_axis_px / 2 と仮定して **動的に trim 範囲を決める**

**adaptive アルゴリズム**:
```python
# rough short axis: 一旦中央 80% で短軸推定
n_wp = len(width_profile)
trim0 = int(0.10 * n_wp)
if n_wp > 2 * trim0 + 1:
    rough_short = float(np.mean(width_profile[trim0:n_wp - trim0]))
else:
    rough_short = float(np.mean(width_profile))
# cap 長 ≈ rough_short / 2 (半円 cap radius)
cap_len_px = rough_short / 2.0
trim = int(np.clip(cap_len_px, 1, n_wp // 3))
if n_wp > 2 * trim + 1:
    core = width_profile[trim:-trim]
else:
    core = width_profile
short_axis_px = float(np.mean(core))
```

### Task B: `mask_morphology.py` に **medial-axis-based perpendicular width** メソッドを追加

新関数:
```python
def measure_single_cell_medial(
    binary_mask: np.ndarray,
    label: int = 0,
    pixel_size_um: float = 1.0,
    smoothing_window_frac: float = 0.15,
    plateau_top_frac: float = 0.50,
) -> CellMorphology:
    """Width measurement using a smoothed mid-column profile,
    mathematically equivalent to Morphometrics' medial-axis
    perpendicular width when the cell is roughly straight (true for S. pombe
    mothers in chamber).
    """
```

**アルゴリズム** (medial axis 風だが計算的にロバスト):
1. mask を long axis 水平に回転 (既存 `_measure_single_cell` の rotation ロジックを再利用)
2. 各 column で **mask の y 中心 (centroid) と幅** を測定
3. y 中心列を **smoothing window で平滑化** (default: long axis の 15 %, Gaussian or moving-average)
4. 隣接 column の y 中心差分 → local tangent angle θ(x) = arctan(dy/dx)
5. 各 column で **rotated width / cos(θ)** = perpendicular width
6. width profile の中央 plateau (top 50 % の連続 region) を `short_axis` として採用
7. 中央 plateau の弧長 (∫ √(1 + (dy/dx)²) dx) + 両端 cap 長 = `long_axis`

このアプローチは Morphometrics と数学的に等価で、S. pombe の直線 rod ならほぼ完全一致するはず。

**バリエーション**: 実装が複雑になるなら medial axis を **distance transform の ridge** から抽出する案もある (`scipy.ndimage.distance_transform_edt`). しかし上記 column-centroid 法が S. pombe には十分。

### Task C: driver script `scripts/recompute_axes_from_masks.py`

**入力**:
- inference_out/*_masks.tif (per-frame label image)
- lineage_data3D.csv (cell_id とフレーム対応)
- pixel_size_um (`lineage_run_params.json` から)

**処理**:
1. mask tif シーケンスをロード (frame 順、glob `*_masks.tif`)
2. 各 frame で lineage_data3D を見て rank=1 cell の cell_id を取得
3. mask 内で label == cell_id の region を切り出す
4. 3 つのモードで axis 測定:
   - `mode='skimage_legacy'`: skimage regionprops major/minor axis (cross-check 用)
   - `mode='supersegger_adaptive'`: `measure_single_cell(endcap_trim_frac='adaptive')`
   - `mode='medial_axis'`: `measure_single_cell_medial()`
5. 各モードの結果を frame 別に CSV 出力

**出力 CSV スキーマ** (`<pos>_<ch>_recomputed_axes.csv`):
```
frame, cell_id, mode,
short_axis_um, long_axis_um, volume_um3_rod,
short_axis_px, long_axis_px, area_px
```

**CLI**:
```bash
python scripts/recompute_axes_from_masks.py \
    --pos Pos27 --ch ch06 \
    --modes skimage_legacy supersegger_adaptive medial_axis \
    --out results/260517/recomputed_axes/Pos27_ch06.csv
```

**batch モード**: `--gold-standard` フラグで gold-standard 31 channel 全部処理 (`select_gold_standard()` を使う)。出力は `results/260517/recomputed_axes/<pos>_<ch>.csv` 群と index CSV。

### Task D: 4 way 比較プロット `scripts/gold_standard_4way_comparison.py`

`scripts/gold_standard_rod_corrected_cycle.py` をベースに拡張。
- 縦軸: short axis / long axis / rod volume (3 panel 縦並び)
- 4 曲線を 1 panel に重ね:
  - **skimage_raw** (現状)
  - **rod_corrected** (理論補正)
  - **supersegger_adaptive** (実測)
  - **medial_axis** (実測、Morphometrics 等価)
- cycle aligned mean ± SD band (色分け、band は薄め)
- 各曲線について cycle 内変動率 (max − min) / mean を legend に表示
- title: "Validation of rod-axis correction: theoretical vs measured"

**期待される結果**:
- supersegger_adaptive と medial_axis がほぼ重なる (実測同士の一致)
- rod_corrected が両者にほぼ一致 (理論モデル妥当性)
- skimage_raw のみ +6 % 増加方向にずれる

---

## File changes

| ファイル | 種類 | 内容 |
|---|---|---|
| `scripts/mask_morphology.py` | 改修 | `_measure_single_cell()` の adaptive trim 対応、新規 `measure_single_cell_medial()` 追加 |
| `scripts/recompute_axes_from_masks.py` | 新規 | mask + lineage CSV → 補正済み軸 CSV |
| `scripts/gold_standard_4way_comparison.py` | 新規 | 4 way 比較プロット |
| `tests/test_mask_morphology_correction.py` | 新規 (任意) | synthetic rod mask で各手法を unit test |

---

## Reference: 既存リソース

- **`scripts/rod_axis_correction.py`** (このリポジトリ): 理論補正、lookup table ベース。`correction_factors(major, minor) → (true_L, true_2r)` および `true_rod_volume_um3()`。**self test 済み (AR=1〜50 で真値完全復元)**
- **`scripts/gold_standard_rod_corrected_cycle.py`** (このリポジトリ): raw vs rod_corrected 比較プロット。Task D の雛形
- **`scripts/overlay_gold_standard_and_phase1_dead.py:select_gold_standard()`**: gold-standard mother 選択 (interval [2.5, 5.0] h + MANUAL_EXCLUDE 35 ch)
- **`scripts/overlay_gold_standard_and_phase1_dead.py:find_lineage_csv()`**: per-channel lineage CSV path (per_channel_figures / batch_figures 両対応)
- **`scripts/analyze_starvation_entry_cell_cycle.py:rank1_division_frames()`**: rank=1 cell_id 切替 frame と clist daughter births を統合した division 検出 (chamber 両モード対応、必ずこれを使うこと)
- **`scripts/central_cell_lineage_tracker.py:248-264`**: 現状の axis 計算ロジック (`p.major_axis_length`, `p.minor_axis_length` をそのまま採用) — これが置き換え対象

---

## Constants / 制約

- `PHASE1_END_FRAME = 2018` (frame 2019 から 0.0055 % glucose)
- `LOW_MASS_PG = 10.0` (mass < 10 pg の row は outlier 扱い)
- `TIME_INTERVAL_MIN = 5.0`, `FRAMES_PER_HOUR = 12.0`
- `pixel_size_um` は `lineage_run_params.json` の `pixel_size_um` を使う (推定 0.345 μm/px)

## MANUAL_EXCLUDE (35 ch, 解析時に除外する)

`scripts/overlay_gold_standard_and_phase1_dead.py:70-99` を参照。

---

## Validation plan

### Phase 1: self test (mask データなしで実装後すぐ)

合成 mask で各手法を検証:
- 真の長さ L, 真の幅 2r を与えて rod mask を合成 (cylinder + 2 semicircular caps)
- 各手法を適用
- 真値との誤差を AR=1, 2, 3, 5, 10 でテーブル化
- 合格条件: medial_axis と supersegger_adaptive で **誤差 ±1 % 以内**

**合成 rod mask コード例**:
```python
def synthesize_rod_mask(L_px: float, r_px: float, canvas_pad: int = 10):
    H = int(2 * r_px) + 2 * canvas_pad
    W = int(L_px) + 2 * canvas_pad
    yy, xx = np.indices((H, W))
    cy = H / 2
    cyl_x0, cyl_x1 = canvas_pad + r_px, canvas_pad + L_px - r_px
    cylinder = (xx >= cyl_x0) & (xx <= cyl_x1) & (np.abs(yy - cy) <= r_px)
    left_cap = ((xx - cyl_x0) ** 2 + (yy - cy) ** 2 <= r_px ** 2) & (xx <= cyl_x0)
    right_cap = ((xx - cyl_x1) ** 2 + (yy - cy) ** 2 <= r_px ** 2) & (xx >= cyl_x1)
    return cylinder | left_cap | right_cap
```

### Phase 2: 1 channel pilot

Pos27_ch06 (gold-standard、cycles=51) で:
- 4 way 比較を per-frame でプロット
- supersegger_adaptive ≈ medial_axis ≈ rod_corrected であることを目視確認
- ズレが大きいフレームを mask overlay で原因究明

### Phase 3: gold-standard 展開

31 channel 全部処理し、Task D の 4 way cycle aligned plot を生成。

### Phase 4 (任意): 全 mother 再計算

revived + never_revived 全部 (≈ 130 ch) で再計算し、phase2 dynamics 解析を補正版データで再評価。

---

## Done criteria

- [ ] `mask_morphology.py` `endcap_trim_frac='adaptive'` 動作
- [ ] `mask_morphology.py` `measure_single_cell_medial()` 実装
- [ ] 合成 rod mask self test で medial_axis 誤差 ≤ 1 %
- [ ] `recompute_axes_from_masks.py` 1 channel で動作
- [ ] `gold_standard_4way_comparison.py` 31 channel で 1 PDF 出力
- [ ] cycle 内 short axis 変動率: supersegger_adaptive / medial_axis / rod_corrected が ±0.5 % 以内で一致
- [ ] (任意) PR description にサマリ表 (skimage / rod_corrected / measured の比較)

---

## Out of scope (この issue では扱わない)

- mass / RI 計算式 (`calc_optical_metrics`) の再評価 → volume が変わると c = mass/volume も変わるので RI 再計算が必要だが、別 issue で扱う
- 2 番目の細胞 (rank=2) の lineage 拡張 → 別 issue
- phase2 dynamics の deep dive (divergence point 同定) → 別 issue

---

## Notes for the implementer (Claude on the analysis PC)

- `/Users/kitak/QPI_Omni/scripts/rod_axis_correction.py` は **数学的検証済み**。self test で AR=1〜50 で真値完全復元 (`recovered_2r/2=1.0000`, `recovered_L/L=1.0000`)。これと measured を比較すれば理論補正の妥当性が確定する
- mask データは `inference_out/*_masks.tif` (Omnisegger 出力)。label image (uint16) で、各 label が cell_id と一致する想定
- 既存の `per_channel_figures.py:mother_division_frames()` は使わないこと (cell_id=0 終了後の division を取りこぼす bug あり、issue 別件)
- 全コードは `scripts/figure_logger.py:save_figure()` で保存する規約。figure_hub inbox に PNG + JSON サイドカーが落ちる
- 出力 figure 例: `inbox/2026-06-10/gold_standard_rod_corrected_cycle/20260610T071441Z_2a98b8/f001.png` (raw vs rod_corrected 2 line × 3 panel)。Task D はこれに 2 曲線追加する形

---

## 既に解析環境にあるはずの主要ファイル (push 済み)

- `scripts/rod_axis_correction.py` — 理論補正モジュール (self test 済み)
- `scripts/gold_standard_rod_corrected_cycle.py` — Task D の雛形
- `scripts/gold_standard_minor_axis_cycle.py` — short axis cycle aligned
- `scripts/gold_standard_phase1_homeostasis.py` — phase1 homeostasis 解析
- `scripts/analyze_starvation_entry_cell_cycle.py` — `rank1_division_frames()` を含む
- `scripts/overlay_gold_standard_and_phase1_dead.py` — `select_gold_standard()`, `MANUAL_EXCLUDE`
- `scripts/mask_morphology.py` — 改修対象 (SuperSegger style 実装済み)
