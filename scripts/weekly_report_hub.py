#!/usr/bin/env python3
"""weekly_report_hub.py

Generate a unified timeline index for weekly reports.

1. Run jsonl_to_obsidian.py and figure_inbox_to_obsidian.py (preprocessing)
2. Retrieve session and figure events for the target week from SQLite DB
3. Read Notion memos from the notion_sync directory
4. Scan the figure-hub library
5. Output a unified index file to Obsidian

Output:
  ~/Documents/Obsidian Vault/00_Inbox/weekly_index_YYYY-Www.md

Weekly index contents:
  - Week summary (sessions, files, bash, errors, thinking, figures)
  - Error summary (sessions with errors)
  - Daily / per-session summary
      - Request content, changed files, reproduction commands per session
      - Generation code and description per figure
      - Notion memo excerpts (background, purpose, key steps)
  - Changed file statistics (aggregated across all sessions)
  - File list for reading (for Read tool)

Usage:
  python3 scripts/weekly_report_hub.py              # generate index for this week
  python3 scripts/weekly_report_hub.py --week 2026-W10  # specified week
  python3 scripts/weekly_report_hub.py --no-preprocess  # skip preprocessing
  python3 scripts/weekly_report_hub.py --dry-run        # preview only
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_db

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

OBSIDIAN_ROOT = Path("/Users/kitak/Documents/Obsidian Vault")
OBSIDIAN_SESSIONS_DIR = OBSIDIAN_ROOT / "00_Inbox/claude_sessions"
OBSIDIAN_FIGURE_INBOX_DIR = OBSIDIAN_ROOT / "00_Inbox/figure_inbox"
OBSIDIAN_NOTION_SYNC_DIR = OBSIDIAN_ROOT / "00_Inbox/notion_sync/api"
OBSIDIAN_WEEKLY_REPORTS_DIR = OBSIDIAN_ROOT / "04_WeeklyReports"

FIGURE_HUB_LIBRARY = Path.home() / "Desktop/figure-hub/library"

JST = timezone(timedelta(hours=9))

WEEKDAY_JP = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ────────────────────────────────────────────
# Week calculation
# ────────────────────────────────────────────

def get_week_range(week_label: str | None = None) -> tuple[date, date, str]:
    if week_label:
        m = re.match(r"(\d{4})-W(\d{2})", week_label)
        if not m:
            raise ValueError(f"Invalid week_label format: {week_label} (e.g. 2026-W10)")
        year, week = int(m.group(1)), int(m.group(2))
        monday = date.fromisocalendar(year, week, 1)
    else:
        today = date.today()
        monday = today - timedelta(days=today.weekday())

    sunday = monday + timedelta(days=6)
    label = monday.strftime("%Y-W%V")
    return monday, sunday, label


# ────────────────────────────────────────────
# Frontmatter reading (fallback)
# ────────────────────────────────────────────

def read_frontmatter(md_path: Path) -> dict:
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    result = {}
    for line in fm_text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ────────────────────────────────────────────
# Notion memo parsing
# ────────────────────────────────────────────

def extract_notion_summary(md_path: Path) -> dict:
    """Extract summary information from notion_sync .md (supports WORKLOG_SPEC / thought memos).

    Returns:
        title       : page title
        notion_url  : Notion URL
        purpose     : purpose (## purpose or **purpose** or opening)
        background  : background (## background or ## what happened)
        steps       : list of step names from implementation steps (max 5)
        key_code    : key lines from code blocks (max 5 lines)
        equations   : lines containing LaTeX/equations
        sections    : list of all section headings
        raw_text    : full body text (for search)
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    # Extract and remove frontmatter
    fm = read_frontmatter(md_path)
    title = fm.get("notion_title", md_path.stem[:80])
    notion_url = fm.get("notion_url", "")

    # Body text with frontmatter removed
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            body = text[end + 4:].strip()

    # Purpose (for excerpts, short cases only)
    purpose = ""
    m = re.search(r"(?:^|\n)#{1,3}\s*目的\s*\n+(.*?)(?=\n#|\Z)", body, re.DOTALL)
    if m:
        purpose = m.group(1).strip()
    if not purpose:
        m = re.search(r"\*\*目的\*\*:?\s*(.+)", body)
        if m:
            purpose = m.group(1).strip()

    # Background (for excerpts)
    background = ""
    for heading in ("背景", "何が起きたか", "経緯", "概要"):  # TODO-JP: regex keys for matching Japanese section headings in Notion docs
        m = re.search(
            r"(?:^|\n)#{1,3}\s*" + heading + r"\s*\n+(.*?)(?=\n#|\Z)",
            body, re.DOTALL
        )
        if m:
            background = m.group(1).strip()
            break

    # Step name list (WORKLOG_SPEC: "#### Step N: ..." or "## Step N: ...")
    steps = re.findall(r"#{2,4}\s*Step\s*\d+[:\s]+(.+)", body)
    if not steps:
        # Bullet list format: "- 1. XXX"
        steps = re.findall(r"[-\d]+\.\s+(.+)", body)[:5]

    # Key lines from code blocks
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", body, re.DOTALL)
    key_code_lines = []
    for cb in code_blocks[:3]:
        for line in cb.strip().splitlines():
            line = line.strip()
            # Meaningful lines excluding comments
            if line and not line.startswith("#") and len(line) > 10:
                key_code_lines.append(line[:120])
                if len(key_code_lines) >= 5:
                    break
        if len(key_code_lines) >= 5:
            break

    # Lines containing equations/LaTeX
    equations = []
    for line in body.splitlines():
        if re.search(r"\\frac|\\sum|\\int|\\partial|\\lambda|\$[^$]+\$|≈|∑|∫|∂", line):
            equations.append(line.strip()[:120])

    # Section headings
    sections = re.findall(r"^#{1,4}\s+(.+)", body, re.MULTILINE)

    return {
        "title": title,
        "notion_url": notion_url,
        "purpose": purpose,
        "background": background,
        "steps": steps[:5],
        "key_code": key_code_lines,
        "equations": equations[:3],
        "sections": sections[:10],
        "raw_text": body,
        "full_body": body,
    }


# ────────────────────────────────────────────
# Source scanning (DB + files)
# ────────────────────────────────────────────

def scan_notion_sync(monday: date, sunday: date) -> list[dict]:
    """Scan .md files for the target week from the notion_sync directory and extract content."""
    results = []
    if not OBSIDIAN_NOTION_SYNC_DIR.exists():
        return results

    for day_offset in range(7):
        day = monday + timedelta(days=day_offset)
        day_dir = OBSIDIAN_NOTION_SYNC_DIR / day.isoformat()
        if not day_dir.exists():
            continue

        for md_path in sorted(day_dir.glob("*.md")):
            summary = extract_notion_summary(md_path)
            results.append({
                "date": day.isoformat(),
                "source": "notion-sync",
                "title": summary.get("title", md_path.stem[:60]),
                "notion_url": summary.get("notion_url", ""),
                "purpose": summary.get("purpose", ""),
                "background": summary.get("background", ""),
                "steps": summary.get("steps", []),
                "key_code": summary.get("key_code", []),
                "equations": summary.get("equations", []),
                "sections": summary.get("sections", []),
                "full_body": summary.get("full_body", ""),
                "path": str(md_path),
            })

    return results


def scan_figure_hub_library(monday: date, sunday: date) -> list[dict]:
    """Scan versions registered during the target week from the figure-hub library."""
    results = []
    if not FIGURE_HUB_LIBRARY.exists():
        return results

    for meta_path in sorted(FIGURE_HUB_LIBRARY.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        fig_id = meta.get("id", "")
        for v in meta.get("versions", []):
            added_at = v.get("added_at", "")
            if not added_at:
                continue
            try:
                added_date = date.fromisoformat(added_at[:10])
            except ValueError:
                continue

            if monday <= added_date <= sunday:
                results.append({
                    "date": added_at[:10],
                    "source": "figure-hub-library",
                    "fig_id": fig_id,
                    "version": v.get("version", ""),
                    "note": v.get("note", "")[:80],
                    "path": str(meta_path),
                })

    return results


# ────────────────────────────────────────────
# Preprocessing script execution
# ────────────────────────────────────────────

def run_preprocess(dry_run: bool) -> None:
    scripts = [
        SCRIPTS_DIR / "jsonl_to_obsidian.py",
        SCRIPTS_DIR / "figure_inbox_to_obsidian.py",
    ]
    for script in scripts:
        cmd = [sys.executable, str(script)]
        if dry_run:
            cmd.append("--dry-run")
        print(f"\n--- Running: {script.name} ---")
        try:
            subprocess.run(cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {script.name} failed (exit code {e.returncode})")


# ────────────────────────────────────────────
# Cross-reference enrichment
# ────────────────────────────────────────────

def enrich_figure_mds(conn, sessions: list[dict], figures: list[dict]) -> int:
    """Append "post-generation context" to figure .md files (enrichment state managed in DB)."""
    # Group claude-sessions by date
    sessions_by_date: dict[str, list[dict]] = {}
    for s in sessions:
        d = s.get("date", "")
        sessions_by_date.setdefault(d, []).append(s)

    enriched = 0
    for fig in figures:
        unique_key = fig.get("unique_key", "")
        fig_date = fig.get("date", "")
        fig_md = fig.get("md_path", "")

        if not fig_md or not fig_date:
            continue

        # Check if already enriched in DB (no file read needed)
        if session_db.is_figure_enriched(conn, unique_key):
            continue

        fig_path = Path(fig_md)
        if not fig_path.exists():
            continue

        existing = fig_path.read_text(encoding="utf-8")

        # Candidate sessions (same day + next 2 days)
        candidate_sessions = []
        for offset in range(3):
            try:
                check_date = (date.fromisoformat(fig_date) + timedelta(days=offset)).isoformat()
            except ValueError:
                continue
            candidate_sessions.extend(sessions_by_date.get(check_date, []))

        if not candidate_sessions:
            continue

        # Retrieve detailed session information from DB
        context_entries = []
        for session in candidate_sessions[:4]:
            sid = session.get("session_id", "")
            md_path_str = session.get("md_path", "")
            if not md_path_str:
                continue

            bash_cmds = session_db.query_session_bash_commands(conn, sid)
            edited_raw = session.get("edited_files_raw", "") or ""
            edited_files = [f for f in edited_raw.split("|||") if f]

            first_msg = session.get("first_user_message", "")
            if not first_msg and not bash_cmds and not edited_files:
                continue

            context_entries.append({
                "session_id": sid,
                "md_path": md_path_str,
                "first_user_message": first_msg,
                "bash_cmds": bash_cmds[:5],
                "edited_files": edited_files[:8],
                "thinking_blocks": session.get("thinking_blocks", 0),
                "error_count": session.get("error_count", 0),
            })

        if not context_entries:
            continue

        # Build "post-generation context" section
        ctx_lines = [
            "",
            "---",
            "",
            "## Post-generation context (what happened after viewing this figure)",
            "",
            f"> Claude Code sessions executed after this figure (generated {fig_date}):",
            "",
        ]

        for entry in context_entries:
            sid8 = entry["session_id"][:8]
            ctx_lines.append(f"### Session: `{sid8}`")
            ctx_lines.append(f"*full path*: `{entry['md_path']}`")
            ctx_lines.append("")

            if entry["first_user_message"]:
                ctx_lines.append("**Request (first message):**")
                ctx_lines.append(f"> {entry['first_user_message'][:200]}")
                ctx_lines.append("")

            if entry["bash_cmds"]:
                ctx_lines.append("**Commands (for reproduction):**")
                ctx_lines.append("```bash")
                for bc in entry["bash_cmds"]:
                    ctx_lines.append(bc["command"])
                ctx_lines.append("```")
                ctx_lines.append("")

            if entry["edited_files"]:
                ctx_lines.append("**Changed code files:**")
                for ef in entry["edited_files"]:
                    short = ef.replace("/Users/kitak/QPI_Omni/", "")
                    ctx_lines.append(f"- `{short}`")
                ctx_lines.append("")

            metrics = (f"thinking={entry['thinking_blocks']}"
                       f" errors={entry['error_count']}")
            ctx_lines.append(f"*Metrics: {metrics}*")
            ctx_lines.append("")

        context_text = "\n".join(ctx_lines)

        if "<!-- POST_FIGURE_CONTEXT -->" in existing:
            updated = existing.replace("<!-- POST_FIGURE_CONTEXT -->", context_text)
        else:
            updated = existing + context_text

        fig_path.write_text(updated, encoding="utf-8")
        session_db.mark_figure_enriched(conn, unique_key)
        conn.commit()
        enriched += 1

    print(f"  figure enrichment: appended post-generation context to {enriched} entries")
    return enriched


# ────────────────────────────────────────────
# Markdown rendering
# ────────────────────────────────────────────

def _fmt_ts(ts: str) -> str:
    if not ts:
        return "??:??"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(JST)
        return dt.strftime("%H:%M")
    except Exception:
        return "??:??"


def _short_path(path: str) -> str:
    return path.replace("/Users/kitak/QPI_Omni/", "").replace("/Users/kitak/", "~/")


def render_session_block(conn, session: dict) -> list[str]:
    """Generate a summary block for one session. Contains all information (no limits)."""
    sid = session["session_id"]
    sid8 = sid[:8]
    md_path = session.get("md_path", "")
    first_msg = session.get("first_user_message") or ""
    thinking = session.get("thinking_blocks", 0)
    errors = session.get("error_count", 0)
    bash_n = session.get("bash_command_count", 0)
    tool_n = session.get("tool_call_count", 0)
    design_notes_json = session.get("design_notes") or "[]"
    session_summary = session.get("session_summary") or ""

    # Edited files
    edited_raw = session.get("edited_files_raw") or ""
    edited_files = [f for f in edited_raw.split("|||") if f]

    # Retrieve all data from DB
    bash_cmds = session_db.query_session_bash_commands(conn, sid)
    error_cmds = [bc for bc in bash_cmds if bc["is_error"]]
    code_changes = session_db.query_session_code_changes(conn, sid)
    error_details = session_db.query_session_errors(conn, sid)

    lines = [
        f"#### Session `{sid8}` [-> full log]",
        "",
    ]

    if md_path:
        lines.append(f"**File:** `{md_path}`")
        lines.append("")

    # Request content (full text)
    if first_msg:
        lines.append("**Request (first user message):**")
        for msg_line in first_msg.splitlines():
            lines.append(f"> {msg_line}")
        lines.append("")

    # Session summary
    if session_summary:
        lines.append("**Session summary:**")
        for sl in session_summary.splitlines():
            lines.append(f"> {sl}")
        lines.append("")

    # Changed files (all)
    if edited_files:
        lines.append(f"**Changed files ({len(edited_files)}):**")
        for ef in edited_files:
            lines.append(f"- `{_short_path(ef)}`")
        lines.append("")

    # Reproduction commands (all)
    if bash_cmds:
        err_summary = f" (errors: {len(error_cmds)})" if error_cmds else ""
        lines.append(f"**Reproduction commands ({len(bash_cmds)}{err_summary}):**")
        lines.append("")
        lines.append("```bash")
        for bc in bash_cmds:
            n = bc.get("event_order", "?")
            ts = _fmt_ts(bc.get("timestamp", ""))
            err_mark = "  # WARN ERROR" if bc["is_error"] else ""
            desc = bc.get("description") or ""
            if desc:
                lines.append(f"# [{n}] {ts} - {desc}{err_mark}")
            else:
                lines.append(f"# [{n}] {ts}{err_mark}")
            lines.append(bc["command"])
            # Output preview (show only on error, or non-error with output)
            out = bc.get("output_preview") or ""
            if out:
                if bc["is_error"]:
                    for oline in out.splitlines()[:10]:
                        lines.append(f"#   {oline}")
                else:
                    preview = out[:200].replace("\n", " <- ")
                    lines.append(f"# -> {preview}")
        lines.append("```")
        lines.append("")

    # Design notes (all blocks, full text)
    try:
        design_notes = json.loads(design_notes_json)
    except Exception:
        design_notes = []
    if design_notes:
        lines.append(f"**Design notes (AI thinking / {len(design_notes)} blocks):**")
        lines.append("")
        for i, note in enumerate(design_notes, 1):
            has_math = bool(re.search(
                r"[\\$∑∫≈×÷±∂π]|\\frac|\\sum|np\.|[a-z]\s*=\s*[\d\(]", note
            ))
            math_mark = " [math]" if has_math else ""
            lines.append(f"> **[Block {i}]{math_mark}**")
            for note_line in note.splitlines():
                lines.append(f"> {note_line}")
            lines.append(">")
        lines.append("")

    # Code change details (all)
    if code_changes:
        lines.append(f"**Code change details ({len(code_changes)}):**")
        lines.append("")
        for cc in code_changes:
            fp = _short_path(cc.get("file_path", ""))
            op = cc.get("operation", "")
            ts = _fmt_ts(cc.get("timestamp", ""))
            ext = cc.get("file_path", "").rsplit(".", 1)[-1].lower()
            lang = {"py": "python", "js": "javascript", "ts": "typescript",
                    "json": "json", "md": "markdown", "sh": "bash",
                    "yaml": "yaml", "yml": "yaml"}.get(ext, "")
            lines.append(f"##### `{fp}` - {op} [{ts}]")
            old_p = cc.get("old_preview") or ""
            new_p = cc.get("new_preview") or ""
            if op == "Edit" and old_p:
                lines.append("Before:")
                lines.append(f"```{lang}")
                lines.append(old_p)
                lines.append("```")
                lines.append("After:")
                lines.append(f"```{lang}")
                lines.append(new_p)
                lines.append("```")
            elif new_p:
                lines.append(f"```{lang}")
                lines.append(new_p)
                lines.append("```")
            lines.append("")

    # Error details (all)
    if error_details:
        lines.append(f"**WARN Error details ({len(error_details)}):**")
        lines.append("")
        for err in error_details:
            n = err.get("event_order", "?")
            ts = _fmt_ts(err.get("timestamp", ""))
            tool = err.get("tool_name", "")
            err_text = (err.get("error_text") or err.get("result_preview") or "")[:400]
            lines.append(f"- `[{n}] {ts} {tool}`: {err_text[:120]}")
        lines.append("")

    # Metrics
    err_mark = f" | **WARN errors {errors}**" if errors > 0 else ""
    lines.append(f"*Metrics: thinking={thinking} | tools={tool_n} | bash={bash_n}{err_mark}*")
    lines.append("")

    # Embed full session .md (no limit, complete record including timeline)
    if md_path:
        md_file = Path(md_path)
        if md_file.exists():
            lines.append(f"**Full session text (timeline, code changes, all tool output):**")
            lines.append("")
            lines.extend(md_file.read_text(encoding="utf-8").splitlines())
            lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def _load_figure_json(fig: dict) -> dict:
    """Read and return the figure inbox JSON. Returns an empty dict on failure."""
    json_path = fig.get("json_path", "")
    if not json_path:
        md_path = fig.get("md_path", "")
        if md_path:
            json_path = md_path.replace(".md", ".json")
    if json_path:
        p = Path(json_path)
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def render_figure_block(fig: dict) -> list[str]:
    """Generate a summary block for a single figure (inline all fields of the figure inbox JSON)."""
    script = fig.get("script", "unknown")
    desc = fig.get("description", "")
    commit = fig.get("git_commit", "")
    dirty = fig.get("git_dirty", False)
    md_path = fig.get("md_path", "")
    image_name = fig.get("image_obsidian_name", "")

    dirty_mark = " *(dirty)*" if dirty else ""
    commit_str = f"`{commit[:8]}`{dirty_mark}" if commit else "N/A"

    lines = [
        f"#### [fig] `{script}` - {desc}" if desc else f"#### [fig] `{script}`",
        "",
    ]

    if image_name:
        lines.append(f"![[{image_name}]]")
        lines.append("")

    # Read full information from JSON
    data = _load_figure_json(fig)
    if data:
        params = data.get("params", {}) or {}
        diff = data.get("diff_from_last", {}) or {}
        data_file = data.get("data_file", None)
        data_keys = data.get("data_keys", []) or []
        git_info = data.get("git", {}) or {}
        changed_files = git_info.get("changed_files", []) or []

        # params table
        table_rows: list[tuple[str, str]] = []
        for k, v in params.items():
            table_rows.append((k, str(v) if v is not None else "N/A"))

        # diff_from_last
        if diff:
            diff_parts = []
            for k, v in diff.items():
                if isinstance(v, dict):
                    old = v.get("from", v.get("old", "?"))
                    new = v.get("to", v.get("new", "?"))
                    if old == "(new)":
                        diff_parts.append(f"{k}: first generation (-> {new})")
                    elif new == "(deleted)":
                        diff_parts.append(f"{k}: {old} -> (deleted)")
                    else:
                        diff_parts.append(f"{k}: {old} -> {new}")
                elif v == "(new)":
                    diff_parts.append(f"{k}: first generation")
                else:
                    diff_parts.append(f"{k}: {v}")
            diff_str = ", ".join(diff_parts) if diff_parts else "(no changes)"
        else:
            diff_str = "(no changes)"

        data_file_str = str(data_file) if data_file else "N/A"
        data_keys_str = ", ".join(data_keys) if data_keys else "N/A"

        # changed_files (up to 5)
        if changed_files:
            MAX_FILES = 5
            shown = changed_files[:MAX_FILES]
            remainder = len(changed_files) - MAX_FILES
            files_str = ", ".join(shown)
            if remainder > 0:
                files_str += f" ... (+{remainder} more)"
        else:
            files_str = "N/A"

        lines.append("| Item | Value |")
        lines.append("|------|-----|")
        for k, v in table_rows:
            lines.append(f"| {k} | {v} |")
        lines.append(f"| Change from previous figure | {diff_str} |")
        lines.append(f"| data_file | {data_file_str} |")
        lines.append(f"| data_keys | {data_keys_str} |")
        lines.append(f"| git_commit | {commit_str} |")
        lines.append(f"| Changed files | {files_str} |")
    else:
        # Fallback when JSON could not be read (legacy behavior)
        if md_path:
            lines.append(f"**File:** `{md_path}`")
        if commit:
            lines.append(f"**git commit:** {commit_str}")

    lines.append("")
    lines.append("---")
    lines.append("")

    return lines


def render_notion_block(notion: dict) -> list[str]:
    """Generate a summary block for one Notion memo. Includes full text and interpretation instructions."""
    title = notion.get("title", "(untitled)")
    notion_url = notion.get("notion_url", "")
    full_body = notion.get("full_body", "")
    path = notion.get("path", "")

    link_text = f"[Notion]({notion_url})" if notion_url else ""
    lines = [
        f"#### [note] Notion memo: \"{title}\" {link_text}",
        "",
    ]
    if path:
        lines.append(f"**File:** `{path}`")
        lines.append("")

    lines.append("**Full text (what this user is thinking / wants to do):**")
    lines.append("")
    if full_body:
        lines.append("```")
        lines.append(full_body.strip())
        lines.append("```")
        lines.append("")
        lines.append(
            "> **To Claude**: Read the above and interpret as \"this user is thinking about X / wanted to try X / is concerned about X\""
            " and weave it into the report. Do not summarize verbatim; rewrite in a form that conveys the intent."
        )
    else:
        lines.append("(no body)")

    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def render_daily_index(
    d_str: str,
    week_label: str,
    conn,
    sessions: list[dict],
    figures: list[dict],
    notions: list[dict],
    lib_entries: list[dict],
) -> str:
    """Generate the daily index Markdown for one day. Contains all information (no limits)."""
    try:
        d = date.fromisoformat(d_str)
    except ValueError:
        d = date.today()
    weekday = WEEKDAY_JP[d.weekday()]

    lines = [
        "---",
        "type: daily-index",
        f"date: {d_str}",
        f"week: {week_label}",
        f'generated_at: "{datetime.now(JST).isoformat()}"',
        "---",
        "",
        f"# {d_str} ({weekday}) daily index",
        "",
        f"> **To Claude**: This single file contains all information for {d_str}."
        " No need to read the session .md files.",
        "",
    ]

    # Sessions
    if sessions:
        lines.append(f"## Claude sessions ({len(sessions)})")
        lines.append("")
        for s in sessions:
            lines.extend(render_session_block(conn, s))

    # Figures
    if figures:
        lines.append(f"## Figures ({len(figures)})")
        lines.append("")
        for fig in figures:
            lines.extend(render_figure_block(fig))

    # Notion memos
    if notions:
        lines.append(f"## Notion memos ({len(notions)})")
        lines.append("")
        for notion in notions:
            lines.extend(render_notion_block(notion))

    # figure-hub library registrations
    if lib_entries:
        lines.append(f"## figure-hub Library registrations ({len(lib_entries)})")
        lines.append("")
        for lb in lib_entries:
            lines.append(f"### `{lb['fig_id']}` {lb['version']}")
            if lb.get("note"):
                lines.append(f"- note: {lb['note']}")
            lines.append(f"- `{lb['path']}`")
            lines.append("")

    return "\n".join(lines)


def render_weekly_index(
    week_label: str,
    monday: date,
    sunday: date,
    conn,
    sessions: list[dict],
    figures: list[dict],
    notions: list[dict],
    lib_entries: list[dict],
    daily_files: list[tuple[str, str]],
) -> str:
    """Generate the weekly index Markdown (lightweight summary + links to daily files)."""

    stats = session_db.query_week_stats(conn, monday.isoformat(), sunday.isoformat())
    file_stats = session_db.query_week_file_stats(conn, monday.isoformat(), sunday.isoformat())

    lines = []

    # -- Frontmatter --
    lines.extend([
        "---",
        "type: weekly-index",
        f"week: {week_label}",
        f"period: {monday.isoformat()} - {sunday.isoformat()}",
        f'generated_at: "{datetime.now(JST).isoformat()}"',
        "db_based: true",
        "---",
        "",
    ])

    # -- Title --
    lines.append(
        f"# Weekly index {week_label} ({monday.strftime('%m/%d')} - {sunday.strftime('%m/%d')})"
    )
    lines.append("")
    lines.append(
        "> **To Claude**: To write the weekly report, **Read the daily index files below in date order**."
        " Each daily index file contains all information for one day (bash, code changes, design notes, full Notion memo text)."
        " Do not use Notion MCP."
    )
    lines.append("")

    # -- Daily index file list --
    lines.append("## Daily index files (read with Read tool in date order)")
    lines.append("")
    for d_str, path in daily_files:
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        weekday = WEEKDAY_JP[d.weekday()]
        day_sessions = [s for s in sessions if s.get("date") == d_str]
        day_figures = [f for f in figures if f.get("date") == d_str]
        day_notions = [n for n in notions if n.get("date") == d_str]
        lines.append(
            f"- **{d_str} ({weekday})** `{path}`"
            f"  - sessions {len(day_sessions)} / figures {len(day_figures)} / Notion {len(day_notions)}"
        )
    lines.append("")

    # -- Weekly summary --
    lines.append("## Weekly summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|------|-----|")
    lines.append(f"| Claude session count | {stats.get('session_count', 0)} |")
    lines.append(f"| Unique edited files | {stats.get('unique_file_count', 0)} |")
    lines.append(f"| Bash runs | {stats.get('total_bash', 0)} |")
    lines.append(f"| Error occurrences | {stats.get('total_errors', 0)} |")
    lines.append(f"| AI thinking blocks | {stats.get('total_thinking', 0)} |")
    lines.append(f"| Total tool calls | {stats.get('total_tools', 0)} |")
    lines.append(f"| Figure generation (inbox) | {stats.get('figure_count', 0)} |")
    lines.append(f"| Finalized figure registrations (library) | {len(lib_entries)} |")
    lines.append(f"| Notion memos | {len(notions)} |")
    lines.append("")

    # -- Session list table --
    if sessions:
        lines.append("## Session list")
        lines.append("")
        lines.append("| Date | ID | thinking | bash | errors | files | request |")
        lines.append("|------|-----|------|------|-------|-------|------|")
        for s in sessions:
            d_label = s.get("date", "")[-5:]
            sid8 = s["session_id"][:8]
            thinking = s.get("thinking_blocks", 0)
            bash_n = s.get("bash_command_count", 0)
            errs = s.get("error_count", 0)
            err_mark = f"**{errs}**" if errs > 0 else str(errs)
            edited_raw = s.get("edited_files_raw") or ""
            file_n = len([f for f in edited_raw.split("|||") if f])
            msg = (s.get("first_user_message") or "")[:40]
            lines.append(
                f"| {d_label} | `{sid8}` | {thinking} | {bash_n} | {err_mark} | {file_n} | {msg} |"
            )
        lines.append("")

    # -- Error summary --
    error_sessions = [s for s in sessions if s.get("error_count", 0) > 0]
    if error_sessions:
        lines.append("## WARN Error summary")
        lines.append("")
        lines.append("| Session | Date | Error count | First user message |")
        lines.append("|----------|------|---------|----------------|")
        for s in error_sessions:
            sid8 = s["session_id"][:8]
            d_label = s.get("date", "")
            errs = s.get("error_count", 0)
            msg = (s.get("first_user_message") or "")[:50]
            lines.append(f"| `{sid8}` | {d_label} | {errs} | {msg} |")
        lines.append("")

    # -- Changed files statistics --
    if file_stats:
        lines.append("## Changed files statistics (aggregated across all sessions)")
        lines.append("")
        lines.append("| File | Change count | Related dates | Session count |")
        lines.append("|---------|---------|-------|------------|")
        for fs in file_stats[:20]:
            fp = _short_path(fs["file_path"])
            dates_str = (fs.get("dates") or "").replace(",", ", ")[:40]
            lines.append(
                f"| `{fp}` | {fs['change_count']} | {dates_str} | {fs['session_count']} |"
            )
        lines.append("")

    # -- Weekly report generation instructions --
    lines.append("---")
    lines.append("")
    lines.append("## Weekly report generation instructions")
    lines.append("")
    lines.append("Once you have finished reading all the daily index files above, generate the report according to `docs/WEEKLY_LOG_SPEC.md`.")
    lines.append("")
    lines.append("**Key specification points:**")
    lines.append("- Organize by topic (content). Do not split by session or by date")
    lines.append("- For each topic, describe the causal flow: \"design -> execution -> figure -> next change\"")
    lines.append("- Each unit should be as substantial as a Qiita article (background, procedure, code excerpts, results, lessons learned)")
    lines.append("")
    lines.append(f"**Save to:** `~/Documents/Obsidian Vault/04_WeeklyReports/{week_label}.md`")
    lines.append("")

    return "\n".join(lines)


# ────────────────────────────────────────────
# Main
# ────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    monday, sunday, week_label = get_week_range(args.week)
    print(f"Target week: {week_label} ({monday} - {sunday})")

    # Preprocessing
    if not args.no_preprocess:
        run_preprocess(args.dry_run)

    print(f"\n--- Source scan ---")

    conn = session_db.open_db()
    run_id = session_db.start_processing_run(conn, "weekly_report_hub")

    try:
        # Retrieve sessions and figures from DB
        sessions = session_db.query_week_sessions(
            conn, monday.isoformat(), sunday.isoformat()
        )
        figures = session_db.query_week_figures(
            conn, monday.isoformat(), sunday.isoformat()
        )

        # File scan (Notion and library are not in DB scope)
        notions = scan_notion_sync(monday, sunday)
        lib_entries = scan_figure_hub_library(monday, sunday)

        print(f"claude_sessions (DB): {len(sessions)} entries")
        print(f"figure_events (DB):   {len(figures)} entries")
        print(f"notion_sync:          {len(notions)} entries")
        print(f"figure_hub_library:   {len(lib_entries)} entries")

        if not sessions and not figures and not notions and not lib_entries:
            print(f"\nWarning: no events found for {week_label}.")

        # Cross-reference enrichment
        if not args.dry_run and not args.no_enrich:
            print(f"\n--- Cross-reference enrichment ---")
            enrich_figure_mds(conn, sessions, figures)

        # Group by date
        sessions_by_date: dict[str, list] = {}
        for s in sessions:
            sessions_by_date.setdefault(s["date"], []).append(s)
        figures_by_date: dict[str, list] = {}
        for f in figures:
            figures_by_date.setdefault(f["date"], []).append(f)
        notions_by_date: dict[str, list] = {}
        for n in notions:
            notions_by_date.setdefault(n["date"], []).append(n)
        libs_by_date: dict[str, list] = {}
        for lb in lib_entries:
            libs_by_date.setdefault(lb["date"], []).append(lb)

        all_dates_set = set(
            list(sessions_by_date) + list(figures_by_date) +
            list(notions_by_date) + list(libs_by_date)
        )
        # Always generate today's daily_index even if there is no activity (for daily log generation)
        today_str = datetime.now(JST).date().strftime("%Y-%m-%d")
        if monday.strftime("%Y-%m-%d") <= today_str <= sunday.strftime("%Y-%m-%d"):
            all_dates_set.add(today_str)
        all_dates = sorted(all_dates_set)

        # Generate daily index files
        daily_files: list[tuple[str, str]] = []
        inbox_dir = OBSIDIAN_ROOT / "00_Inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        for d_str in all_dates:
            day_content = render_daily_index(
                d_str, week_label, conn,
                sessions_by_date.get(d_str, []),
                figures_by_date.get(d_str, []),
                notions_by_date.get(d_str, []),
                libs_by_date.get(d_str, []),
            )
            day_path = inbox_dir / f"daily_index_{d_str}.md"
            if not args.dry_run:
                day_path.write_text(day_content, encoding="utf-8")
                print(f"  daily index: {day_path.name}"
                      f" ({len(day_content.splitlines())} lines)")
            else:
                print(f"  [dry-run] would write: {day_path.name}"
                      f" ({len(day_content.splitlines())} lines)")
            daily_files.append((d_str, str(day_path)))

        # Generate weekly index (lightweight summary)
        md_content = render_weekly_index(
            week_label, monday, sunday,
            conn, sessions, figures, notions, lib_entries,
            daily_files=daily_files,
        )

        out_path = inbox_dir / f"weekly_index_{week_label}.md"

        if args.dry_run:
            print(f"\n[dry-run] would write: {out_path}")
            total = len(sessions) + len(figures) + len(notions) + len(lib_entries)
            print(f"  total events: {total}")
            session_db.finish_processing_run(conn, run_id, success=True,
                                              notes="dry-run")
            return 0

        out_path.write_text(md_content, encoding="utf-8")

        total = len(sessions) + len(figures) + len(notions) + len(lib_entries)
        session_db.finish_processing_run(conn, run_id, processed=total, success=True)

        print(f"\nWeekly index (lightweight): {out_path}")
        print(f"  sessions: {len(sessions)} | figures: {len(figures)} |"
              f" Notion: {len(notions)} | Library: {len(lib_entries)}")
        print(f"  daily indexes: {len(daily_files)} files")
        print(f"\nNext step:")
        print(f"  In Claude Code: \"write the weekly report\""
              f" or \"write the {week_label} weekly report\"")

    except Exception as e:
        session_db.finish_processing_run(conn, run_id, success=False, notes=str(e))
        conn.close()
        raise
    finally:
        conn.close()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a unified timeline index for weekly reports")
    parser.add_argument("--week", help="Target week (e.g. 2026-W10). Defaults to this week if omitted")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip preprocessing scripts")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip appending cross-reference to figure .md files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args))
