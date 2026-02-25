---
name: Save Plans to Git
overview: 過去・将来のプランファイルをリポジトリの docs/plans/ に保存し、実装の文脈を git 履歴として残す
todos:
  - id: create-dir
    content: docs/plans/ ディレクトリを作成し、既存の23個プランファイルをコピーしてcommit
    status: in_progress
  - id: update-rule
    content: task-routing.mdc にプラン実行後に docs/plans/ へ自動コピーするルールを追加
    status: pending
isProject: false
---

# プランファイルのgit保存

## やること

1. `docs/plans/` ディレクトリを作成
2. 既存の23個のプランファイルを `c:\Users\QPI\.cursor\plans\` からコピー
3. [`task-routing.mdc`](c:\Users\QPI\Documents\QPI_omni\.cursor\rules\task-routing.mdc) にルールを追加: プラン実行後に自動コピーする

## 自動コピーのルール（task-routing.mdc に追加）

プランを実行した後（Agentモードで実装が終わったタイミング）、以下を行う：

```bash
copy "C:\Users\QPI\.cursor\plans\<planfile>.plan.md" "docs\plans\"
git add docs/plans/
```

## 既存ファイルの一括コピー

`c:\Users\QPI\.cursor\plans\` にある全23ファイルを `docs/plans/` にコピーして一括 commit する。

## ファイル名の命名規則（変更なし）

既存のファイル名（例: `アライメント後のコントラスト保存_792542d8.plan.md`）はそのまま使う。日本語ファイル名 + ハッシュ末尾で十分識別可能。