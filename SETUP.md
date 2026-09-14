# QPI_omni セットアップ手順

## 1. リポジトリ

```powershell
git clone https://github.com/kengo-kitagishi/QPI_Omni.git
cd QPI_Omni
# 2回目以降
git pull
```

## 2. Python 環境（解析・セグメンテーション）

Windows + NVIDIA GPU の PC では `environment/bootstrap_windows.ps1` を 1 回実行する（管理者権限は不要）。

```powershell
powershell -ExecutionPolicy Bypass -File environment\bootstrap_windows.ps1
```

やること: conda が無ければ Miniconda を per-user で `%USERPROFILE%\miniconda3` に入れる（PATH と
レジストリは変えないので、Micro-Manager の BeanShell が呼ぶ CPython 3.11 はそのまま）→
`environment/omnipose_win64_explicit.txt` から `omnipose` env を作る → pip の pin を入れる →
`environment/check_env.py` で CUDA・checkpoint・推論を確認する。初回は約 3 GB のダウンロードで 15〜30 分。

解析用 Python はこれ以降 `%USERPROFILE%\miniconda3\envs\omnipose\python.exe`
（解析用 PC のように Anaconda がある場合は `anaconda3\envs\omnipose\python.exe`）。
`scripts/` のスクリプトはすべてこの interpreter で実行する。中身と更新手順は
[environment/README.md](environment/README.md)。

既存の env を確かめるだけなら:

```powershell
<env>\python.exe environment\check_env.py --model models\omni_model_d20_2026_09_07_12_41_31.782047
```

Mac / Linux には bootstrap は無い。`environment/omnipose_win64_full.yml` を元に手で作る
（GPU が無いと segmentation は走らせない。図・集計だけならそれで足りる）。

## 3. 解析の入口

```powershell
<env>\python.exe scripts\run_dataset_pipeline.py datasets\<YYMMDD>.yaml --plan
```

実験ごとの設定は `datasets/<YYMMDD>.yaml`（`datasets/TEMPLATE.yaml` から作る）。
手順の正本は [docs/PROTOCOL_TIMELAPSE.md](docs/PROTOCOL_TIMELAPSE.md)。

## 4. Node.js（Notion MCP に必要）

[https://nodejs.org](https://nodejs.org) の LTS 版を入れて `node --version` で確認。

## 5. gh CLI（GitHub Issues の自動登録に必要）

```powershell
winget install --id GitHub.cli
gh auth login     # GitHub.com -> HTTPS -> Login with a web browser
```

Mac は `brew install gh`。インストール後はアプリを再起動して PATH を読み直す。

## 6. Claude Code CLI / Cursor

```powershell
npm install -g @anthropic-ai/claude-code
```

プロジェクトフォルダで `claude` を起動する。`CLAUDE.md`（リポジトリ直下）が clone で付いてくるので追加設定は不要。
Notion MCP は claude.ai の connector で繋ぐ（トークンをファイルに置く旧方式 `.cursor/mcp.json` / `.claude.json` は不要）。
予定・タスクは Notion の Tasks DB で管理する（ClickUp は 2026 年に廃止）。

## 主要ファイル

| ファイル | 役割 |
|---|---|
| `environment/` | pinned conda env と bootstrap |
| `models/MODELS.json` | production checkpoint の台帳（sha256・学習セット・eval） |
| `datasets/<YYMMDD>.yaml` | 実験ごとの設定 |
| `scripts/run_dataset_pipeline.py` | seg → tracking → 分裂 QC → 集約 → master 公開 |
| `scripts/figure_logger.py` | 図の保存 + `docs/EXPERIMENT_LOG.md` への自動追記 |
| `docs/EXPERIMENT_LOG.md` | 実験ログ（figure_logger.py が自動更新） |
