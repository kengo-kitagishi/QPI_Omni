# clickup_helper.py

`clickup_helper.py` is a safety-first scheduling assistant for ClickUp.

## Commands

- `add`: one-shot task creation (backward compatible)
- `doctor`: auth + retrieval validation
- `agenda`: read scheduled tasks and verify normalization
- `dashboard`: generate monthly calendar + today timeline + gantt + risk checks
- `review`: tell you which existing schedules look odd or should be moved later
- `advise`: consult best slots from issue-like text
- `carryover`: re-plan unfinished tasks from yesterday
- `book`: apply one candidate from an `advise` plan (dry-run by default)
- `figfix-sync`: pull figure-fix tracker recommendations (`figure-hub`) into ClickUp (preview/apply)

For `experiment` and `code`, workflow order can be enforced using content keywords
(setup -> acquisition -> processing -> documentation).

Supported list keys (default config):

- `plan`, `input`, `experiment`, `code`, `manuscript`, `slide`, `meeting`, `competition`, `other`, `daily`
- If `auto_discover_lists=true`, missing known keys are auto-filled from workspace list metadata.

## Token loading order

1. `CLICKUP_API_TOKEN` environment variable
2. `.cursor/mcp.json` -> `mcpServers.clickup.env.CLICKUP_API_TOKEN`

## Recommended verification flow

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py doctor --deep --sample 3 --probe-tasks 50
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py doctor --discover
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py agenda --days 7 --verify --preview 20
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py dashboard --days 35
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py review --days 14 --near-days 2
```

## Issue-style consulting

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py advise \
  --text "授業登録とTA申し込みを今週中にやる" \
  --due 2026-03-12 \
  --verify
```

- `--priority auto` is default.
- Category is inferred from text (`admin`, `writing`, `experiment`, `figure`, `input`, `general`).
- List is inferred automatically with `--list auto` (default).
- `input` category uses `input_default_duration_minutes` (current config: 120m) and avoids rearrangement by default, so it is placed into free time more gently.
- When scheduling higher-priority new tasks, `input` is treated as the first deferral target in rearrangement candidates.
- `rearrange_allowed_list_keys` now includes `input` by default, so new tasks can push existing `input` tasks later.
- New schedule writes can be blocked per-list using `blocked_write_list_keys` (e.g., `["plan"]`). Blocked targets are auto-rerouted.
- `feedback/review` text is treated as deadline pressure under `--priority auto` (urgent bias).
- `competition` list (大会) is treated as no-schedule-day anchor by default; other tasks are not placed on those dates.
- Medium-exchange chain constraints are supported (`medium_flow_steps` / `medium_flow_rules`), including fixed day/hour gaps and review-time violation detection (e.g., `O/N culture -> +24h -> pre culture`).
- `T0+xxh` task names are also supported for medium-flow step inference via `medium_flow_t0_offset_map` (default: `48/96/120/168/216h` mapping).
- Split-task constraints are supported (`split_task_rules`). Examples: `glucose sln.作成` as `10m prepare + >=6h (same day) 30m collect`, `NH4Cl作成` as `30m prepare + >=1h (same day) 30m collect`, `EMM培地作成` as `30m prepare + >=2h (same day) 10m collect`.

Plan JSON is saved in:

- `/Users/kitak/QPI_Omni/scripts/.clickup_plans/`

## Safe apply

Dry-run:

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py book \
  --plan /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-YYYYMMDD-HHMMSS.json \
  --candidate 1
```

Apply:

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py book \
  --plan /Users/kitak/QPI_Omni/scripts/.clickup_plans/plan-YYYYMMDD-HHMMSS.json \
  --candidate 1 \
  --commit
```

## Daily re-plan for unfinished tasks

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py carryover
```

Apply carryover updates:

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py carryover --commit
```

## Figure Fix Tracker sync

Preview (recommended first):

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py figfix-sync --max-tasks 5
```

Apply:

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py figfix-sync --max-tasks 5 --apply
```

Notes:
- This runs `figure-hub`'s `recommend` then `clickup-sync` by default.
- Default hub root is `~/Desktop/figure-hub` (or `FIGURE_HUB_ROOT`).
- Use `--list-key manuscript` / `--list-key slide` to change destination list.

## Simple schedule sanity check

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py review
```

This reports:
- overlap problems
- tasks scheduled after due date
- low-priority tasks occupying near-term focus slots when deadline pressure exists
- no-deadline tasks consuming short-term slots
- workflow-order inversion (especially experiment/code pipeline)

## Memory log (.md)

Every `advise`, `dashboard`, and `carryover` run appends decisions/rules to:

- `/Users/kitak/QPI_Omni/scripts/schedule_companion_memory.md`

This file is meant to evolve your scheduling policy over time.

## Config

Copy `clickup_helper_config.example.json` to `clickup_helper_config.json` and customize:

- fixed schedules (`fixed_list_keys`, `fixed_keyword_patterns`)
- write blocks (`blocked_write_list_keys`)
- rearrange scope (`rearrange_allowed_list_keys`)
- experiment ordering guard (`protect_experiment_structure`)
- policy (`category_keyword_rules`, `category_priority`, `category_bias_hours`)
- workflow policy (`workflow_enabled`, `workflow_groups`, `workflow_stage_rules`, `workflow_gap_minutes`)
- horizon (`planning_days`, `review_horizon_days`, `monthly_horizon_days`, `carryover_days`)

Use another config path if needed:

```bash
python3 /Users/kitak/QPI_Omni/scripts/clickup_helper.py --config /path/to/config.json advise --text "..."
```
