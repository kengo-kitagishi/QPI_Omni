# Scheduling Policy

Last updated: 2026-09-04

## Purpose

予定・タスクの正本を Notion（`タスク管理（GTD）> Tasks`）に一本化する。
その日どう動くかは、朝の秘書（morning brief）が出した予定に従う。

2026-09-04 に ClickUp 運用を終了した。以降 ClickUp は使わない。

## Source of Truth

- タスク・予定の正本: Notion `Tasks` DB
  - <https://app.notion.com/p/82b434bf5057464e888a6b3be2bc9e87>
  - data source: `collection://7fb3f2db-e277-4883-b686-b364b2be9df7`
- その日の動き方: 朝の秘書（morning brief / `/morning`）が出した予定
- 固定予定の実体（部活・授業・TA・ご飯・学会など）: Google Calendar（参照のみ）
- 研究・実験の文脈メモ: Markdown / Obsidian / Notion 研究ノート

## Operational Rule

### Notion Tasks で持つもの

- 学振・Abstract・原稿などの締切つきタスク
- 実験チェーン（依存関係のあるもの）
- figure 修正、input backlog
- 「まだ今すぐではないが忘れたくないタスク」
- 時刻が決まった予定（`Status = Remind`、`期限` に日時）

### Google Calendar を参照するもの

- 部活・TA・授業・ご飯・学会・大会など、すでに入っている時間ブロック
- 「来週いつ空いてる？」のような空き時間の確認

## Default Behavior For Scheduling

- 「今日どこに入れるか」「明日どうするか」は、朝の秘書が出した予定を前提に考える
- こちらで勝手に一日の時間割を組み直さない
- 新しい予定・タスクは Notion Tasks に入れる（`固定` が ON のものは動かさない）
- 空き時間を答えるだけのときはタスクを作らない

## What To Avoid

- ClickUp を使うこと（運用終了）
- 朝の秘書が出した予定を無視して別の時間割を提案すること
- 低優先度の input / slide をその週に詰め込みすぎること
- 過去の私用イベントを機械的に未来へ移すこと
- 実験チェーンを依存関係なしに carryover すること

## Assistant Policy

- 普段の予定相談は、朝の秘書の予定 + Google Calendar を前提に短く返す
- 予定を作る／動かすときだけ Notion Tasks を触る
- `固定` チェックの入ったタスク（部活・授業・会議・発表）は自動で動かさない

## Revision Note

- 2026-09-04: ClickUp 運用を終了し、予定・タスク管理を Notion に一本化。
  日々の予定は朝の秘書（morning brief）が出したものに従う方針へ変更。
- 旧方針（Google Calendar + ClickUp のハイブリッド運用）は git 履歴を参照。
