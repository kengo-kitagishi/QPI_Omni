# Lineage tracker / survival analysis DataFrame schema

`central_cell_lineage_tracker.py` と `lineage_survival_analysis.py` が書き出す 3 つの CSV の列を列挙したリファレンス。いずれも `<channel_dir>/inference_out/lineage_out/` に出力される。

- `clist.csv` … cell 1 行のサマリ（SuperSegger 風）
- `lineage_data3D.csv` … cell × frame の 2D long-format（per-frame metrics）
- `survival_summary.csv` … in-tree cell に対する recovery 期 fate 判定＋飢餓直前メトリクス

---

## 共通用語

| 用語 | 意味 |
|------|------|
| cell_id | tracker が内部で連番発行する cell の一意 ID（0 = mother） |
| in_tree | mother（cell_id=0）を根とする子孫の系統樹に属するか |
| rank | あるフレームでの mother からの順位（0 = mother 側、大きいほど open end 側） |
| outlier frame | 面積 or 体積が前後フレームと整合しないフレーム（セグメンテーション瞬断など） |
| 分裂点 | parent の rank に new daughter が挿入されたフレーム |

時間系列の物理量（µm, µm², µm³, pg）は `optical_config.PIXELSIZE` と 4/3 π を使った楕円体近似から算出。frame ↔ h 変換は `time_interval_min`（既定 5 分＝ 1 フレーム）を使う。

---

## clist.csv  — per-cell summary

1 行 = 1 cell。tracker が全フレーム処理後に `_build_clist_rows()` で生成。

### 系譜

| 列 | 意味 |
|----|------|
| cell_id | この cell の ID。0 = mother |
| mother_id | 親 cell の ID。mother（cell_id=0）は -1 |
| daughter1_id | 1 番目の daughter の cell_id（なければ -1） |
| daughter2_id | 2 番目の daughter の cell_id（なければ -1） |
| generation | mother=0、その娘=1、…（mother からの世代） |
| in_tree | True なら mother 子孫系統樹に属する |

### ライフサイクル（フレーム）

| 列 | 意味 |
|----|------|
| n_frames | 観測された総フレーム数 |
| n_outliers | そのうち outlier フラグがついたフレーム数 |
| birth_frame | 誕生フレーム（mother は -1） |
| death_frame | 最終観測フレーム |
| age_frames | death_frame - birth_frame |
| birth_time_h | birth_frame を時間換算 |
| death_time_h | 同上 |
| age_h | 寿命（h） |

### birth / death 時の形態・RI

すべて birth_frame / death_frame 単フレームの値。

| 列 | 意味 |
|----|------|
| long_axis_birth_um / long_axis_death_um | 楕円近似 major 軸（µm） |
| short_axis_birth_um / short_axis_death_um | 楕円近似 minor 軸（µm） |
| area_birth_um2 / area_death_um2 | 2D 面積（µm²） |
| volume_birth_um3 / volume_death_um3 | rod 体積近似（µm³） |
| mean_ri_birth / mean_ri_death | 細胞内 mean refractive index |
| mass_birth_pg / mass_death_pg | dry mass（pg）＝ ∫Δn dA を mass に換算 |

### 位置・画像端

| 列 | 意味 |
|----|------|
| x_pos_birth_px / y_pos_birth_px | 重心画素座標（birth 時） |
| x_pos_death_px / y_pos_death_px | 重心画素座標（death 時） |
| dist_to_edge_birth_px | birth 時に最も近い画像端までのピクセル距離 |

### 成長指標

| 列 | 意味 |
|----|------|
| dL_max_um_per_frame | 生涯で最大の長軸伸長率 |
| dL_min_um_per_frame | 最小の長軸伸長率（しばしば負＝分裂で縮む） |
| L_death_over_birth | long_axis_death / long_axis_birth |

### rank / 生涯平均

| 列 | 意味 |
|----|------|
| rank_birth | birth_frame での rank |
| rank_death | death_frame での rank |
| mean_volume_um3 | 生涯の volume_um3_rod 平均（outlier/border 除外） |
| mean_ri_over_life | 生涯の mean_ri 平均（同） |
| mean_mass_pg | 生涯の mass_pg 平均（同） |

---

## lineage_data3D.csv  — per-frame × per-cell long format

1 行 = 1 cell × 1 frame。`_build_data3D_rows()` で生成。

| 列 | 意味 |
|----|------|
| cell_id | cell の一意 ID |
| parent_id | 親 cell の ID（mother は -1） |
| in_tree | mother 子孫に属するか |
| birth_frame | この cell の誕生フレーム（行に関わらず一定） |
| death_frame | この cell の最終フレーム（一定） |
| frame | そのレコードが対応するフレーム番号 |
| time_h | frame × time_interval_min / 60 |
| rank | そのフレームでの rank |
| area_px | セグメンテーションマスクの面積（ピクセル） |
| area_um2 | 面積（µm²） |
| long_axis_um | 楕円近似 major 軸（µm） |
| short_axis_um | 楕円近似 minor 軸（µm） |
| centroid_x_px / centroid_y_px | 重心（画素） |
| total_phase | ROI 内の位相積分値（rad·px） |
| volume_um3_rod | rod 体積近似（µm³） |
| mean_ri | ROI 内の mean refractive index |
| mass_pg | dry mass（pg） |
| is_outlier | 3-frame outlier rule で弾かれたフレームか |
| touches_border | マスクが画像端に接しているか |

**利用上の注意**

- `is_outlier = True` や `touches_border = True` のフレームは `volume/mass/mean_ri` の値が信用できない。プロットや slope フィットでは除外する運用。
- 分裂の発生は「同じ `cell_id` の death_frame 直後に `parent_id == その cell_id` の new row が現れる」ことで確認できる。

---

## survival_summary.csv  — in-tree cell の fate / 飢餓期 state

`lineage_survival_analysis.py` の `build_fate_table()` 出力。in-tree ではない cell は除外、さらに「親の outlier フレームまたはその直後に birth」と判定された false-division も除外。

### 系譜・ライフサイクル

| 列 | 意味 |
|----|------|
| cell_id | cell の一意 ID |
| parent_id | 親 cell の ID（mother は -1） |
| in_tree | 常に True（false は排除） |
| generation | mother=0, 娘=1, … |
| birth_frame / death_frame | 誕生・最終フレーム |
| birth_time_h | birth_frame を時間換算 |
| birth_epoch | `"pre" / "starv" / "rec"` のいずれか。birth_frame が属するメディア epoch |

### 分裂情報

| 列 | 意味 |
|----|------|
| divided_in_rec | 回復期（frame ≥ REC_START_FRAME）に自身の daughter を少なくとも 1 つ生んだか |
| n_divisions_in_rec | 回復期に生まれた自身の daughter 数 |

### recovery lag / elongation rate

| 列 | 意味 |
|----|------|
| recovery_lag_frames | 回復開始から自身の最初の分裂までのフレーム差（分裂しなければ NaN） |
| recovery_lag_h | 同上、時間換算 |
| recovery_elong_um_per_h | 回復期の long_axis_um 傾き（per cell-cycle セグメントの平均）。NaN なら「十分なデータなし」 |
| n_recovery_points | elongation rate 計算に使った frame 数 |
| n_recovery_segments | 回復期に分裂で区切られた何セグメントから平均を取ったか |

`recovery_elong_um_per_h` の計算手順:
1. 回復期の frame（≥ REC_START_FRAME）だけ残す
2. outlier / border フレームを除外
3. 子の birth_frame で区切って cell-cycle ごとのセグメントに分割
4. 各セグメントで long_axis_um の線形フィット（seg 内 n ≥ 4 必要）
5. 全セグメント総計で REC_SLOPE_MIN_POINTS（既定 10）以上あれば、各セグメント slope の平均を返す

### 飢餓直前（pre-starvation）ベースライン

STARV_START_FRAME の直前 PRE_WINDOW_FRAMES（既定 100 フレーム ≒ 8.3h）の平均。

| 列 | 意味 |
|----|------|
| pre_starv_volume_um3 | 体積平均（outlier / border 除外） |
| pre_starv_mass_pg | dry mass 平均 |
| pre_starv_mean_ri | mean RI 平均 |

飢餓開始後に生まれた cell では全て NaN になる（window 内にデータがないため）。

### fate ラベル（暫定）

| 値 | 判定条件 |
|----|----------|
| `alive` | recovery_elong_um_per_h ≥ 0.3 µm/h |
| `dead` | recovery_elong_um_per_h < 0.3 µm/h |
| `no_rec_data` | recovery_elong_um_per_h が NaN（回復期の有効データなし） |

閾値 0.3 µm/h は暫定値。現データセット（260405 Pos9）では生死が明確に分かれないため、将来 bimodal な elongation 分布が得られる実験でフィッティングし直す。

survival DataFrame の `.attrs` には参考として以下が入る:

| attr | 意味 |
|------|------|
| elong_alive_thr | 閾値（= 0.3） |
| elong_threshold_median | 観測された elongation rate の中央値 |
| elong_lower_bound | 観測された elongation rate の最小値 |

---

## 関連スクリプトと追加の入出力

| スクリプト | 主な出力 |
|------------|----------|
| `central_cell_lineage_tracker.py` | `clist.csv`, `lineage_data3D.csv`, `lineage_cells.json`, figure-hub inbox に系統樹・時系列 PDF |
| `central_cell_lineage_overlay.py` | per-frame PNG overlay（lineage 色付け） |
| `lineage_survival_analysis.py` | `survival_summary.csv`, figure-hub inbox に mother trajectory / fate tree / scatter / hist PDF |

MEDIA_SWITCHES / STARV_START_FRAME / REC_START_FRAME などのパラメータは `lineage_survival_analysis.py` の先頭定数を見る。
