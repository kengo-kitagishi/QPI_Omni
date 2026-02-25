---
name: GitHub Issues Auto
overview: 「いずれやること」を言葉にした瞬間、Claudeが自動でGitHub Issueを作成してリンクを報告する仕組みを構築する
todos:
  - id: install-gh
    content: winget で gh CLI をインストールし gh auth login で認証
    status: completed
  - id: create-labels
    content: gh label create で someday / bug ラベルを作成
    status: completed
  - id: update-rule
    content: task-routing.mdc に自動Issue作成ルールを追加
    status: completed
isProject: false
---

# GitHub Issues 自動登録の仕組み

## やること

- `gh` CLI をインストール（winget 1コマンド）
- `gh auth login` でGitHubアカウントを認証
- `[task-routing.mdc](c:\Users\QPI\Documents\QPI_omni\.cursor\rules\task-routing.mdc)` にルールを追加

## Cursor Rule の追加内容

以下のキーワードが会話に出たとき、Claudeが自動でIssueを作成：

- `いずれ` / `そのうち` / `あとで` / `いつか`
- `実装しないといけない` / `試したい` / `忘れそう`
- `issueに挙げて`（明示的な指示）

作成後に報告するだけ（確認は取らない）：

```
Issue作成: #12 「〇〇の実装」→ https://github.com/...
```

## ラベル設計（シンプルに2種類）

- `someday` - いつかやること（解析・機能追加）
- `bug` - 直さないといけないもの

## インストール手順

```powershell
winget install --id GitHub.cli
gh auth login
```

その後、Cursor を再起動すれば私が `gh issue create` を呼べるようになる。