---
name: Gitignore作成とGit設定
overview: GitHubにpushする前に、適切な.gitignoreファイルを作成して、データファイルや結果ファイル、キャッシュファイルなどをGitから除外する設定を行います。
todos:
  - id: create-gitignore
    content: .gitignoreファイルを作成し、データファイル、結果ファイル、キャッシュファイルなどを除外する
    status: completed
  - id: verify-git-status
    content: git statusで.gitignoreが正しく機能しているか確認する
    status: in_progress
    dependencies:
      - create-gitignore
---

# .gitign

oreファイルの作成とGit設定

## 現状

- `.gitignore`ファイルが存在しない
- 大量のデータファイル（CSV、TIF、PNG、ROI）と結果ファイルが含まれている
- `data/`ディレクトリが未追跡
- 以前追跡されていたファイルが削除されている（git statusで大量のdeletedファイル）

## 実施内容

### 1. `.gitignore`ファイルの作成

以下の内容を含む`.gitignore`を作成します：

- **データファイル**: `data/`ディレクトリ全体
- **結果ファイル**: `results/`ディレクトリ全体（READMEによると出力先）
- **Python関連**: `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`, `.Python`
- **環境ファイル**: `.env`, `.venv`, `env/`, `venv/`
- **Jupyter Notebook**: `.ipynb_checkpoints`
- **IDE設定**: `.vscode/`, `.idea/`, `*.swp`, `*.swo`
- **OS関連**: `.DS_Store`, `Thumbs.db`
- **一時ファイル**: `*.tmp`, `*.log`
- **画像処理関連**: 大きな画像ファイルや中間ファイル

### 2. Gitの状態確認

`.gitignore`作成後、`git status`で除外が正しく機能しているか確認します。

### 3. コミット対象の確認

以下のファイルはコミット対象として適切です：

- ソースコード（`.py`ファイル）
- 設定ファイル（`config.yaml`、`env.yml`）
- `README.md`
- その他のドキュメント

## 注意事項