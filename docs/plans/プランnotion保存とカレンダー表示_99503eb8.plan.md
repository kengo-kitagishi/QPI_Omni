---
name: プランNotion保存とカレンダー表示
overview: プラン内容を Notion に保存し、Type タグで思考ログと分離。Date プロパティでカレンダー表示できるようにする。
todos: []
isProject: false
---

# プランの Notion 保存とカレンダー表示

## 現状

- プランは `docs/plans/` に git で保存されている（[task-routing.mdc](.cursor/rules/task-routing.mdc) のルール）
- Notion の QPI Research Notes には [思考]、[会話]、作業ログが混在
- 日付で「何をやったか」を見たいが、思考ログと混ざって見づらい

## 方針

**同じデータベース内で Type プロパティで分離**し、Notion のカレンダービューで日付ごとに見られるようにする。

## 1. Notion データベースに Type プロパティを追加

QPI Research Notes に **Type**（Select）プロパティを追加する（手動 or API）:

| 選択肢 | 用途 | 1日あたり |
|--------|------|-----------|
| 思考 | [思考] メモ | **1ページ**（その日の思考を追記） |
| 作業ログ | 会話まとめ・研究ノート | **1ページ**（その日の作業を追記） |
| 図 | figure_logger の自動保存 | **1ページ**（その日の図を追記） |
| プラン | 実行済みプランの記録 | 1プラン1ページ（plan.md と対応） |

**1日1ページルール**: 図・思考・作業ログは、その日の中での流れが分かるよう、1日1ページにまとめる。既存ページがあれば追記、なければ新規作成。

既存ページには後から Type を設定（手動 or 一括スクリプト）。

## 2. プラン保存フローの拡張

[task-routing.mdc](.cursor/rules/task-routing.mdc) の「プランファイルの保存ルール」を拡張:

```
1. docs/plans/ にコピー（既存）
2. Notion にプランページを作成（新規）
   - Type: プラン
   - Date: 実行日（2026年）
   - Name: プラン名（例: Subagent委譲ルール削除）
   - 本文: プランの概要・実行内容
```

## 3. カレンダー表示の使い方

Notion 側で:

1. QPI Research Notes を開く
2. ビューを追加 → **カレンダー**
3. 日付プロパティ: **Date**
4. フィルタ: **Type = プラン**（または 作業ログ など）

これで「その日に実行したプラン」をカレンダーで確認できる。

## 4. 1日1ページの実装方針

- **思考**: 既に [task-routing.mdc](.cursor/rules/task-routing.mdc) で 1日1ページ。Type=思考 を付与。
- **作業ログ**: 「Notionにまとめて」時に、当日の Type=作業ログ ページを検索。あれば追記・なければ新規。複数テーマも1ページに追記。
- **図**: figure_logger を変更。その日の Type=図 ページを検索し、あれば追記（ブロック追加）、なければ新規作成。現在は1図1ページで作成しているため、ロジック変更が必要。

## 5. 実装タスク

- **Notion に Type プロパティを追加**: 手動で Notion のデータベース設定から追加（または Notion API で database を update）
- **プラン保存スクリプト**: `scripts/notion_plan_save.py` を作成。plan.md のパスを受け取り、Notion にページ作成（Type=プラン、Date、本文）
- **task-routing.mdc のルール更新**: プラン実行完了時に Notion 保存も行う旨を追記。作業ログの1日1ページルールを明記。
- **figure_logger の変更**: その日の Type=図 ページを検索し、あれば append、なければ create。BATCH_THRESHOLD との整合も考慮。

（chat_logger は現在未使用のため対象外）

## 6. 注意点

- Notion API で database に新プロパティを追加するには、database の update 権限が必要
- 既存ページに Type が未設定だと、フィルタで「未設定」が混ざる。後から一括で設定するか、デフォルト値を検討
