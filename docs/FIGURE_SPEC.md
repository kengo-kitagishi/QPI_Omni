# FIGURE_SPEC — publication-grade figures (QPI_Omni)

目的: 保存する**全図が「単体で再現・再利用できる」**こと。
満たす標準: FAIR (Wilkinson et al. 2016, *Sci Data*) / journal **Source Data** (EMBO・Nature・eLife) /
**Ten Simple Rules for Better Figures** (Rougier et al. 2014, *PLoS Comput Biol*) /
**Error bars in experimental biology** (Cumming, Fidler, Vaux 2007, *JCB*) / **MDAR checklist** (2021).

---

## 0. 図は必ず3点セットで保存
1. **図本体**（pdf/svg 優先、png は ≥300 dpi）
2. **source-data ファイル**（プロットした実数値そのもの。csv＋npz）
3. **JSON サイドカー**: provenance（caller_file, git commit(+dirty), params, run_id, runtime）＋ **caption**

## 1. Source data — FAIR ＋ journal Source Data
- 図と**必ず一緒に保存**（任意にしない）。script が `data=` を渡さなくても、figure_logger が
  matplotlib の artists（line / scatter / bar / hist）から実数値を**自動抽出**して保存する（fallback）。
- **開いた形式**: csv（ヘッダ付）＋ npz。`data_keys` に全列/配列を列挙。
- 各列は §3-★ の操作的定義を持つ（数値だけで解釈可能＝Reusable）。
- Findable/Accessible: 一意 `run_id` ＋ manifest 登録、inbox に同梱。

## 2. 見た目 — Ten Simple Rules
- **3層インフラ**を使う: `paper.mplstyle`（地）＋ `qpi_colors`（色の意味）＋ `qpi_plots`（作法）。
  jet/rainbow 禁止・Okabe-Ito・上右 spine 除去・軸に単位。
- **caption は必須**（caption の無い図は未完成扱い）。

## 3. Caption — 自己完結・journal grade（必須項目）
1. **タイトル文**: 内容/主張を1文で宣言（例 "Intracellular density varies across the cell cycle."）
2. **パネル別 (A)(B)(C)…**: 各パネルが何をプロットしたか（量・軸）
3. **★ 操作的定義（全プロット・全量に必須）**: 各量を「生データからどう計算したか」で定義する。
   軸ラベルではない。複数量があれば各量に1つずつ。例:
   - cycle-mean RI = [birth, div−1] の各フレームの全細胞 mean RI を平均
   - added = value(div_frame) − value(birth_frame)。div_frame = 次の娘の birth = 分裂後
   - volume = EFD 回転体推定（輪郭を楕円フーリエ近似→中心線直交断面の回転体積分）
   - dry mass = (λ/2πα)·∬φ dA（位相の面積積分）
4. **視覚要素を全部定義**: 色 / 線種 / マーカー / 影帯 / フィット線 / 矢印
5. **誤差バー（Cumming-Vaux）**: 何を表すか明示（SD / SEM / 95%CI）。未定義で出さない
6. **n を単位付き・群ごと**: cells / cycles / time points / experiments（生物 vs 技術反復も該当時 MDAR）
7. **統計検定名（MDAR）**: Pearson / KS / t-test 等 ＋ 有意マーカーがあれば
8. **条件（MDAR）**: 生物種・株/遺伝子型・培地・温度・増殖 phase
9. **画像なら**: scale bar・時間間隔 (min/frame)・representative かどうか
10. **略語定義**（SD, EMM 等）＋ "see also"（関連図 / Methods / supplement）＋ data availability（source-data）
11. 数値は**データ/params 由来のみ**。文献引用は**確実な時だけ**（不確かなら引かない）

## 4. Provenance / 再現性 — MDAR ＋ FAIR
- サイドカーに caller_file・git commit(+dirty)・params・run_id・runtime（figure_logger が記録済）。
- コード可用性: 生成 script が記録 commit でリポジトリにある。

## 5. 強制（任意にしない）
- **infra 強制**（figure_logger）: source-data 保存（自動抽出 fallback）・provenance・軸 introspection。
- **チェックリスト強制**（生成時の agent/人）: §3 の caption（特に ★操作的定義・誤差/n/検定）。
- **チェッカ**: inbox で `data_file` 空 or `caption` 欠落の図を flag。

---

## 標準 → 要件 対応表
| 標準 | 満たす箇所 |
|---|---|
| FAIR (Wilkinson 2016) | §1 開形式+keyed+同梱+provenance、§3-★ 操作的定義（Reusable）|
| Source Data (EMBO/Nature/eLife) | §1 図ごとに実数値を必ず保存 |
| Ten Simple Rules (Rougier 2014) | §2 3層インフラ＋caption 必須 |
| Error bars (Cumming/Vaux 2007) | §3.5 誤差定義＋§3.6 n |
| MDAR (2021) | §3.6–3.8 n/検定/条件＋§4 provenance/コード可用性 |

---

## 型別の体裁見本（該当タイプを真似る）

**顕微鏡像/モンタージュ**
> fission yeast cells in EMM (2% glucose) were imaged in time lapse in a Mother Machine and optical phase delay maps were extracted by QPI. Shown are images of a representative cell traversing the cell cycle from (left) cell birth to (right) the last frame before division (15 min/frame). [必要なら scale bar を追記]

**ヒストグラム/分布**
> Distribution of division intervals (interdivision times) for normally dividing mother cells in EMM (2% glucose) at 30 °C (n = 1268 cycles from 31 mother cells; mean ± SD = 3.26 ± 0.36 h).

**時系列オーバーレイ（群比較）**
> Two dying lineages (Pos20_ch06, orange; Pos30_ch04, blue) overlaid on 26 surviving lineages (gray): mother-cell mean RI, dry mass, and volume vs time (0–168 h; EMM 2% glucose, 30 °C). Both elongate before death (volume and dry mass exceed the surviving band) with rising mean RI; survivors stay near mean RI ≈ 1.37 throughout. Lines, per-lineage traces.

**多パネルの周期トレンド＋解釈（最も網羅的な手本）**
> Intracellular density varies across the cell cycle. (A) Wild-type *S. pombe* imaged in time lapse in a Mother Machine; phase-shift maps by QPI; a representative cell from (left) birth to (right) the last frame before division (15 min/frame). (B,C,D,E) Volume (B), dry mass (C), mean RI (D), and dry-mass concentration (E) aligned by relative cell-cycle progression (0 = birth, 1 = last frame before division); curves, mean; shaded regions, ±1 SD (n = 1306 cycles, 26 mother cells; EFD volume estimate). Volume and dry mass each roughly double; concentration declines ~7% through the cycle and recovers toward division. Dry-mass cell-to-cell CV = 11.7% (birth), 9.0% (before division). SD, standard deviation.

**散布/相関（操作的定義を必ず明記）**
> Birth-size homeostasis across generations in normally dividing mother cells (n = 1306 cycles, 26 mothers; EFD volume). Operational definition: the endpoint is the *next* division, so the plotted "added" = value(div_frame) − value(birth_frame) = birth(N+1) − birth(N) (consecutive post-division sizes), **not** within-cycle growth; x = birth value; dashed line, OLS fit. Pearson, all p < 1×10⁻⁴⁸ (volume r = −0.54, slope = −0.57; dry mass r = −0.39; mean RI r = −0.42): larger or denser births are followed by smaller next births — return toward the population mean across generations.
