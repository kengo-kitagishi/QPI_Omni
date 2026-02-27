# QPI_Omni Codex Instructions

## ClickUp routing
Use ClickUp for explicit task/schedule requests.

List IDs:
- Experiment: `901813997604`
- Code analysis: `901813997608`
- Planning: `901813997590`
- Manuscript: `901813997612`

Default time:
- Unspecified: 12:00-14:00
- Morning: 09:00-12:00
- Afternoon: 13:00-16:00

## Figure logging
When saving figures from scripts, prefer `scripts/figure_logger.py` and `save_figure()`.

## Research memo policy
For observations/hypotheses/ideas from experiments and analysis:
- Keep local source of truth in Obsidian:
  - `/Users/kitak/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD.md`
  - `/Users/kitak/Documents/Obsidian Vault/02_Research/analysis/`
- If Notion sync is required, run:
  - `/Users/kitak/Documents/Obsidian Vault/98_Automation/sync_notion_to_obsidian.py`

## Worklog format (single source)
- Canonical specification: `/Users/kitak/QPI_Omni/docs/WORKLOG_SPEC.md`
- For any worklog output (Obsidian/Notion/report), follow that spec section order.
- Required granularity: each implementation step must include `purpose`, `command`, `expected`, and `actual`.
- Verification must include pass/fail and evidence path/log.
- Do not change worklog format without explicit user approval.
- If rule files disagree, use `WORKLOG_SPEC.md` for worklog schema.

## Fallback
If ClickUp/Notion integration is unavailable, preserve logs in local markdown and explicitly report the failed integration step.
