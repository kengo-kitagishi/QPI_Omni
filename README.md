# QPI_Omni

分裂酵母 *S. pombe* のグルコース飢餓応答を、定量位相イメージング（QPI）+ Mother Machine で
1細胞ずつ追跡する解析リポジトリ。

## 新しい PC で clone してから解析まで

```powershell
git clone https://github.com/kengo-kitagishi/QPI_Omni.git
cd QPI_Omni
powershell -ExecutionPolicy Bypass -File environment\bootstrap_windows.ps1     # Miniconda + pinned env + GPU/model check
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --plan   # 何が走るか
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml          # seg -> tracking -> 分裂 QC -> 集約
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --stages consolidate,publish --tag v<YYYYMMDD>_<note>
```

`<env>` は `%USERPROFILE%\miniconda3\envs\omnipose`（Anaconda がある PC では `anaconda3\envs\omnipose`）。
撮影側（grid 校正・drift session・`compute_drift_online.py`）は [docs/PROTOCOL_TIMELAPSE.md](docs/PROTOCOL_TIMELAPSE.md) §2〜8。

## ドキュメント（正本）

- [docs/PROTOCOL_TIMELAPSE.md](docs/PROTOCOL_TIMELAPSE.md) — 実験〜解析の統合プロトコル。旧 doc はこの doc に一本化済み（Appendix C）。
- [docs/LINEAGE_DATAFRAME_SCHEMA.md](docs/LINEAGE_DATAFRAME_SCHEMA.md) — tracker 出力 / master CSV の列定義。
- [docs/FIGURE_SPEC.md](docs/FIGURE_SPEC.md) — 図の保存規約（caption / source data）。
- [docs/CENTRAL_CELL_TRACK_FIGURE_SPEC.md](docs/CENTRAL_CELL_TRACK_FIGURE_SPEC.md) — central-cell track 図の仕様。
- [SETUP.md](SETUP.md) — 環境構築（Python 環境・Claude Code / Cursor・gh）。
- [scripts/README.md](scripts/README.md) — スクリプト索引（役割別）。退避した旧コードは `scripts/archive/`。
- 各スクリプトの docstring が一次資料。記載と実装が食い違う場合はスクリプトを正とする。

## リポジトリの構成

| 場所 | 中身 |
|---|---|
| `environment/` | pinned conda env（explicit spec・pip pins・bootstrap・check） |
| `models/` | production に使った Omnipose checkpoint と `MODELS.json`（sha256・学習セット・eval） |
| `datasets/` | 実験ごとの yaml（パス・培地切替 frame・RI 校正・bad frames・model） |
| `scripts/` | 解析スクリプト。`run_dataset_pipeline.py` が seg 以降の入口 |
| `docs/` | プロトコル・スキーマ・図の規約 |

## 主要スクリプト（`scripts/`）

| 役割 | スクリプト |
|---|---|
| データセット 1 本を master まで | `run_dataset_pipeline.py`（yaml 駆動。seg / track / qc / consolidate / publish、再開可） |
| Omnipose 推論（GPU） / 学習 | `seg_omnipose.py` / `08_train.py` |
| lineage tracking（黄色輪郭の volume / RI / mass、全細胞） | `central_cell_lineage_tracker.py --indir <mask ch> --raw-dir <phase ch>` |
| 分裂判定の検証 | `division_qc_260517.py` |
| 論文用の窓の派生パッケージ | `build_phase1_dataset_260517.py` |
| master の解決（解析はここから読む） | `qpi_paths.py` |
| QC 図・lineage HTML | `_fig_mother_lineage_qc_260517.py`, `_fig_switch_frame_check_260517.py`, `lineage_html_gallery_260517.py` |
| 図の保存 | `figure_logger.py`（`save_figure(data=, caption=)`。`plt.savefig()` 直接呼び出しは避ける） |

## 引用

- Omnipose: Cutler, K. J. et al. *Nature Methods* (2022).
- Cellpose: Stringer, C. et al. *Nature Methods* (2021).
