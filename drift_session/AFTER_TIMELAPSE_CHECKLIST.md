# タイムラプス終了後チェックリスト
## 目的: ステージノイズ vs. KF velocity feedforward の切り分け

---

## Step 1: drift_log バックアップ

```bash
copy drift_session\drift_log.json drift_session\drift_log_kf_vel_on_260327.json
```

---

## Step 2: ステージ再現性テスト（顕微鏡室で実行）

**前提**: タイムラプス終了直後、試料はそのまま、フォーカスも維持した状態で実行。

1. Micro-Manager の Script Panel を開く
2. `scripts/test_stage_repeatability.bsh` を読み込む
3. CONFIG_FILE・N_REPS・TEST_DIR を確認（デフォルトで OK）
4. Run → 約 10-15 分で完了
5. 結果は `C:/stage_repeatability_test/repeatability_results.json` と図に出力される

### 結果の読み方

| temporal std (image X→stage Y) | 解釈 |
|---|---|
| ≈ 27 nm（ECC精度と同程度） | ステージは精確 → KFか生物学が原因 |
| ≈ 112 nm（タイムラプス残差と同程度） | **ステージ位置決めノイズが支配的** |
| 40–80 nm（中間） | ステージ + KF の複合 |

---

## Step 3: KF velocity 無効化テスト（次のタイムラプスで実施）

`drift_session/drift_config.json` を以下のように変更：

```json
"kf_Q_vel_nm2": 0.0,
```

他のパラメータは変えない。次の実験の最初の数時間のdrift_logを今回と比較する。

| 比較 | 解釈 |
|---|---|
| velocity=0で残差が変わらない（≈112nm） | KF関係ない → ステージノイズが主因 |
| velocity=0で残差が大幅に減少（<60nm） | KF velocity feedforward が主因 |

---

## Step 4: Step 2–3 の結果で対応方針を決める

### 4A: ステージノイズ主因だった場合

- 現在のKFパラメータのままで良い（velocity項はフィルタとして機能している）
- `kf_R_nm2` を増やすことで、ノイズ測定値の影響を減らせる
  ```json
  "kf_R_nm2": 40000.0
  ```
- 根本解決はエンコーダ付きステージか、ステップモーター化

### 4B: KF velocity 主因だった場合

- `kf_Q_vel_nm2: 0.0` のまま継続
- または velocity feedforward を pos のみに変更（要コード修正）:
  ```
  scripts/compute_drift_online.py: 719行目
  kf_ff_ty_nm = pos_ty_new  # + vel_ty_new  ← velocityを外す
  kf_ff_tx_nm = pos_tx_new  # + vel_tx_new
  ```

### 4C: 両方が混在していた場合

- velocity=0 + R増加（両方適用）
- `kf_Q_vel_nm2: 0.0` + `kf_R_nm2: 25000.0` から試す

---

## 参考: 現在のタイムラプス（260327）の診断サマリー

| 指標 | 値 |
|---|---|
| ECC単一測定精度（チャネル間σ） | X→stageY: 27nm、Y→stageX: 8nm |
| タイムラプス残差 std（T=1–28） | image X: 112nm、image Y: 130nm |
| 残差の自己相関 lag1 | image X: 0.11（ほぼ白色ノイズ） |
| 残差の安定性 | 時間変化なし（制御ループ不安定ではない） |
| corr(cum_y[t], tx[t+1]) | -0.30（過補正の痕跡） |

**現状最有力仮説**: ステージ位置決めノイズ（±0.1–0.2µm/回）が主因。
KF velocity feedforward が二次的に増幅している可能性あり。
