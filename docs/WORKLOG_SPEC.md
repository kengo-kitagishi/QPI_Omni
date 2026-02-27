# WORKLOG_SPEC

## Purpose
- This file is the single source of truth for worklog structure.
- If worklog format needs to change, update only this file first.

## Scope
- Obsidian daily logs (`01_Daily/YYYY-MM-DD.md`)
- Notion worklog pages (Type=`作業ログ`)
- AI-generated task/work reports in QPI_Omni context

## Required Sections (Order Fixed)
Every worklog should follow this order.

1. 背景 (Background)
2. 要件定義（ユーザー要件）(Requirements)
3. 実装方針 (Implementation policy)
4. 実装手順（Step 1, 2, 3… コマンド付き）(Implementation steps)
5. 検証手順と結果 (Verification)
6. 変更ファイル一覧 (Changed files)
7. 既知の制約・注意点 (Constraints/Caveats)
8. 他PCでの再現手順 (Reproduction on another PC)

## Authoring Rules
- Use concrete wording and keep user intent explicit.
- Preserve uncertainty when uncertain; do not overstate conclusions.
- Commands should be copy-pastable as much as possible.
- File paths should be specific and absolute when needed.
- If a section has no content, write `none` explicitly.

## Granularity Standard (Required)
- Logs must be reproducible by running Section 4 from top to bottom.
- In Section 4, every step must include all of:
  - `purpose`: why this step is needed.
  - `command`: exact command(s) actually used.
  - `expected`: expected output/state before execution.
  - `actual`: observed output/state after execution.
- If a step was not executed, write `not executed` with reason.
- Section 5 must include explicit pass/fail judgment and evidence path/log.
- Section 6 must include file path + what changed + why changed.
- Section 8 must call out machine-dependent values (paths, env vars, launchd labels).

## Markdown Template
```md
## HH:MM | [short title]

### 1. 背景
- What context this work started from.

### 2. 要件定義（ユーザー要件）
- Requirement A
- Requirement B

### 3. 実装方針
- Why this approach was selected.

### 4. 実装手順（Step 1, 2, 3… コマンド付き）
1. Step 1 description
   - purpose: ...
   - command: `...`
   - expected: ...
   - actual: ...
2. Step 2 description
   - purpose: ...
   - command: `...`
   - expected: ...
   - actual: ...
3. Step 3 description
   - purpose: ...
   - command: `...`
   - expected: ...
   - actual: ...

### 5. 検証手順と結果
1. Check command/result
   - command: `...`
   - actual: ...
2. Check command/result
   - command: `...`
   - actual: ...
- Outcome: pass/fail and evidence path

### 6. 変更ファイル一覧
- `/abs/path/fileA`: what changed, why changed
- `/abs/path/fileB`: what changed, why changed

### 7. 既知の制約・注意点
- Constraint A
- Caveat B

### 8. 他PCでの再現手順
1. Replace machine-specific paths.
2. Run Step 4 commands in order.
3. Run verification from Section 5.
```

## Change Control
- Do not change worklog format without explicit user approval.
- Tool-specific rules (AGENTS/Cursor/Claude preferences) should reference this file rather than duplicating schema text.
