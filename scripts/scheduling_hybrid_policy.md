# Scheduling Hybrid Policy

Last updated: 2026-04-01

## Purpose

日々の予定確認は Google Calendar を主に使い、タスクの意味・締切・依存関係は ClickUp で持つ。
速さと見やすさは Google Calendar、構造化と再配置は ClickUp に分担する。

## Source of Truth

- 日々の予定を見る場所: Google Calendar
- タスクの在庫・締切・依存関係: ClickUp
- 研究・実験の文脈メモ: Markdown / Obsidian

## Operational Rule

### Google Calendar を主に見るもの

- 部活
- TA
- 授業
- ご飯
- 学会
- 大会
- その日に実際に動く時間ブロック

### ClickUp で持つもの

- 学振
- Abstract
- 実験チェーン
- figure 修正
- input backlog
- 「まだ今すぐではないが忘れたくないタスク」

## Default Behavior For Scheduling

- 明日どうするか、今日どこに入れるか、という相談は Google Calendar ベースで考える
- ClickUp は毎日の細かい時間割を無理に全部持たない
- ClickUp に時間つきで入っている予定は、Google Calendar に見えている前提で扱う
- 日次の判断では Google Calendar を優先して参照する

## When ClickUp Should Be Edited

- 締切が近いタスクを再配置するとき
- 実験の依存関係を保ったまま動かすとき
- done / overdue / carryover を整理するとき
- figure / manuscript / input の backlog を月次・週次で整理するとき

## What To Avoid

- 明日の動きを決めるたびに ClickUp を重く引き直すこと
- 低優先度の input / slide をその週に詰め込みすぎること
- 過去の私用イベントを機械的に未来へ移すこと
- 実験チェーンを依存関係なしに carryover すること

## Assistant Policy

- 普段の予定相談は Google Calendar を前提に短く返す
- ClickUp は必要な時だけ使って深く触る
- 大きい変更だけを ClickUp に反映する
- 実績報告があれば、必要に応じて ClickUp の done と依存タスク再配置を行う

## Practical Interpretation

- Google Calendar = 表の予定表
- ClickUp = 裏のタスクDB

## Revision Note

この方針は、ClickUp 単独運用だと「明日何をするか」を決めるには重すぎる、という実運用上の判断に基づく。
