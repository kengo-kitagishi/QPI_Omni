# Activity Logger (QPI_Omni -> Obsidian)

This setup appends work logs to Obsidian daily notes automatically.

## What gets logged

- Finished Cursor transcripts under:
  - `~/.cursor/projects/*QPI-Omni*/agent-transcripts/*/*.jsonl`
- Finished Claude Code transcripts under:
  - `~/.claude/projects/*QPI-Omni*/*.jsonl`
- Git snapshots in `/Users/kitak/QPI_Omni`

Output target:
- `/Users/kitak/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD.md`

State file:
- `.figure_history/session_activity_state.json`

## Worklog format

Transcript entries are rendered in spec format defined by:
- `/Users/kitak/QPI_Omni/docs/WORKLOG_SPEC.md`

If you need to change the section structure, update `WORKLOG_SPEC.md` first.

## Run once

```bash
cd /Users/kitak/QPI_Omni
python3 scripts/session_activity_logger.py
```

Dry run:

```bash
python3 scripts/session_activity_logger.py --dry-run
```

## Enable automatic periodic run (macOS)

Install launchd agent (default every 900 seconds):

```bash
cd /Users/kitak/QPI_Omni
bash scripts/install_activity_logger_launchd.sh
```

Custom interval (seconds):

```bash
bash scripts/install_activity_logger_launchd.sh 600
```

## Verify

```bash
launchctl list | rg activity-logger
cat .figure_history/activity_logger.out.log
cat .figure_history/activity_logger.err.log
```

## Disable

```bash
launchctl unload "$HOME/Library/LaunchAgents/com.kitak.qpi.activity-logger.plist"
```
