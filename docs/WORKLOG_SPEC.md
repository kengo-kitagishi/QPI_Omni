# WORKLOG_SPEC

## Purpose
- This file is the single source of truth for worklog format.
- If log format needs to change, update this file first.
- Target granularity is "fully reproducible by running commands top-to-bottom" while keeping section length proportional to task size.

## Scope
- Notion worklog pages (Type=`作業ログ`)
- Obsidian daily logs (`01_Daily/YYYY-MM-DD.md`)
- AI-generated work reports in QPI_Omni context

## Non-Negotiable Detail Level
- A third party must be able to reproduce the same state by following the log from top to bottom.
- Commands must be copy-pastable, include absolute paths when path dependency exists, and include expected vs actual outcomes.
- Ambiguous statements such as "did some setup" are not allowed.
- If something was planned but not executed, write `not executed` with reason.

## Section Policy (No Fixed Numbering)
- Top-level section numbering like `1..8` is not required.
- Use only the sections needed for the task; keep logs concise but reproducible.
- When the task is substantial (implementation/configuration), include all core sections below.
- When the task is small (single command/minor edit), collapse sections and keep only essentials.

### Core Sections (recommended)
- `前提`
- `背景`
- `要件定義（ユーザー要件）`
- `実装方針`
- `実装手順（Step 0, 1, 2... コマンド付き）`
- `検証手順と結果`
- `変更ファイル一覧`
- `既知の制約・注意点`
- `他PCでの再現手順`

## Authoring Rules
- Preserve user intent in concrete wording.
- Keep uncertainty explicit; do not overstate conclusions.
- Use exact file paths, labels, job names, env vars, and timestamps when relevant.
- When a section is intentionally empty, write `none`.

## Section-Level Requirements

### 前提
- Must include: OS, user, runtime assumptions, token/config source, and backup policy.

### 要件定義（ユーザー要件）
- Keep this section detailed for substantial tasks.
- Must include at least:
  - `目的` (what user wants to achieve)
  - `スコープ` (in-scope targets/files/systems)
  - `期待成果物` (what should exist after completion)
  - `受け入れ条件` (how to judge done)
  - `制約/前提` (permissions, env, API, OS limits)
- If unknown items exist, add `未確定事項` explicitly.

### 実装方針
- Keep this section detailed for substantial tasks.
- Must include at least:
  - `採用方針` (selected approach)
  - `採用理由` (why this approach)
  - `実行順序` (high-level sequence)
  - `代替案と不採用理由` (considered but not chosen)
  - `リスクと緩和策` (likely failure modes and mitigations)

### 実装手順（Step 0, 1, 2... コマンド付き）
- Must be executable from top to bottom.
- Each step must include all of:
  - `目的` (why this step exists)
  - `コマンド` (exact command block)
  - `期待結果` (what should happen)
  - `実結果` (what actually happened)
- If backup was relevant, include `Step 0: バックアップ（任意）`.
- Finish this section with a final checklist step (example: `Step N: 最終検証チェックリスト`).

### 検証手順と結果
- Each check must include command + observed output/state.
- Must include explicit pass/fail judgment.
- Must include evidence location (log path, output path, or screenshot path).

### 変更ファイル一覧
- For each changed file, include:
  - absolute path
  - what changed
  - why changed

### 他PCでの再現手順
- Must identify machine-dependent values:
  - local paths
  - env vars
  - launchd labels / scheduler settings
  - external secret/token setup points
- Must include a minimal ordered sequence for Windows/macOS differences when applicable.

## Notion Logging Rules
- Store as Type=`作業ログ`.
- Follow this spec's section policy without forcing top-level numbering.
- Entry title line format: `## HH:MM | [要約タイトル]`.
- For long operations, prioritize reproducibility over brevity.

## Markdown Template (Required Skeleton)
```md
## HH:MM | [要約タイトル]

### 前提
- 対象OS:
- 作業ユーザー:
- トークン/設定の取得元:
- バックアップ方針:

### 背景
- この作業を始めた理由と直前の状態。

### 要件定義（ユーザー要件）
- 目的:
- スコープ:
- 期待成果物:
- 受け入れ条件:
- 制約/前提:
- 未確定事項:

### 実装方針
- 採用方針:
- 採用理由:
- 実行順序:
- 代替案と不採用理由:
- リスクと緩和策:

### 実装手順（Step 0, 1, 2... コマンド付き）
#### Step 0: バックアップ（任意）
- 目的:
```bash
# command
```
- 期待結果:
- 実結果:

#### Step 1: [作業名]
- 目的:
```bash
# command
```
- 期待結果:
- 実結果:

#### Step 2: [作業名]
- 目的:
```bash
# command
```
- 期待結果:
- 実結果:

#### Step N: 最終検証チェックリスト
```bash
# check commands
```
- 期待結果:
- 実結果:

### 検証手順と結果
1. 検証項目A
```bash
# command
```
- 実結果:
2. 検証項目B
```bash
# command
```
- 実結果:
- 判定: pass / fail
- 根拠: /abs/path/to/log-or-output

### 変更ファイル一覧
- `/abs/path/fileA`: 何を変えたか / なぜ変えたか
- `/abs/path/fileB`: 何を変えたか / なぜ変えたか

### 既知の制約・注意点
- 制約A
- 注意点B

### 他PCでの再現手順
1. 依存パス・環境変数を差し替える。
2. 実装手順を上から順に実行する。
3. 検証手順を実行し、判定と根拠を確認する。
```

## Quality Gate (Before Saving to Notion)
- [ ] `前提` が埋まっている
- [ ] `要件定義` に `目的/スコープ/期待成果物/受け入れ条件/制約` がある
- [ ] `実装方針` に `採用方針/採用理由/実行順序/代替案/リスク緩和` がある
- [ ] 実装手順の各 Step に `目的/コマンド/期待結果/実結果` がある
- [ ] 検証手順に pass/fail と根拠パスがある
- [ ] 変更ファイル一覧が絶対パスで記録されている
- [ ] 他PC差分（パス・環境変数・スケジューラ設定）を明記した

## Change Control
- Do not change worklog format without explicit user approval.
- AGENTS/Cursor/Claude rules should reference this file instead of duplicating schema text.
