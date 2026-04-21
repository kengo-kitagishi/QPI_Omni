#!/usr/bin/env python3
"""
session_to_notion.py

Parse Claude Code session jsonl and post work logs to Notion.

Usage:
  python3 session_to_notion.py                # Auto-select latest session
  python3 session_to_notion.py /path/to.jsonl # Specified session

Output:
  Creates a work log page in the QPI Research Notes database.
"""

import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
import urllib.request
import urllib.error

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────

NOTION_TOKEN = os.environ.get("NOTION_TOKEN", "")
NOTION_VERSION = "2022-06-28"
DATABASE_ID = "312eda96-228e-8165-9726-cd75b221357a"

SESSION_DIR = Path.home() / ".claude/projects/-Users-kitak-QPI-Omni"

# ──────────────────────────────────────────
# jsonl parsing
# ──────────────────────────────────────────

def load_entries(path: Path) -> list:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def extract_user_messages(entries: list) -> list[dict]:
    """Extract user text messages in chronological order."""
    msgs = []
    for e in entries:
        if e.get("type") != "user":
            continue
        msg = e.get("message", {})
        content = msg.get("content", "")
        ts = e.get("timestamp", "")

        # content is string = user message
        if isinstance(content, str) and content.strip():
            # Strip system-reminder tags
            text = _strip_system_tags(content)
            if text.strip():
                msgs.append({"ts": ts, "text": text.strip()})

        # content is list: extract text blocks only
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = _strip_system_tags(block["text"])
                    if text.strip():
                        msgs.append({"ts": ts, "text": text.strip()})
                        break  # 1 entry = 1 message
    return msgs


def _strip_system_tags(text: str) -> str:
    """Remove tags such as <system-reminder>...</system-reminder>."""
    import re
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[a-zA-Z-]+>.*?</[a-zA-Z-]+>", "", text, flags=re.DOTALL)
    return text.strip()


def extract_tool_uses(entries: list) -> list[dict]:
    """Extract tool calls made by the assistant in chronological order."""
    uses = []
    for e in entries:
        if e.get("type") != "assistant":
            continue
        msg = e.get("message", {})
        content = msg.get("content", [])
        ts = e.get("timestamp", "")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                uses.append({
                    "ts": ts,
                    "name": block.get("name", ""),
                    "input": block.get("input", {}),
                    "id": block.get("id", ""),
                })
    return uses


def extract_edited_files(tool_uses: list) -> list[str]:
    """Return list of file paths that were Edit/Write'd (deduplicated)."""
    files = []
    seen = set()
    for tu in tool_uses:
        if tu["name"] in ("Edit", "Write"):
            fp = tu["input"].get("file_path", "")
            if fp and fp not in seen:
                files.append(fp)
                seen.add(fp)
    return files


def extract_bash_commands(tool_uses: list) -> list[dict]:
    """Extract only Bash commands."""
    cmds = []
    for tu in tool_uses:
        if tu["name"] == "Bash":
            cmd = tu["input"].get("command", "")
            desc = tu["input"].get("description", "")
            if cmd:
                cmds.append({"ts": tu["ts"], "cmd": cmd, "desc": desc})
    return cmds


def guess_title_and_description(user_msgs: list[dict]) -> tuple[str, str]:
    """Infer title and description from the first user message."""
    if not user_msgs:
        return "[Work log] Session", ""
    first = user_msgs[0]["text"]
    # Trim to 50 chars
    desc = first[:200]
    title_raw = first[:50].replace("\n", " ")
    title = f"[Work log] {title_raw}"
    if len(first) > 50:
        title += "…"
    return title, desc


def get_session_date(entries: list) -> str:
    """Return the date from the first timestamp in the session."""
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                jst = dt.astimezone(timezone.utc)
                return jst.strftime("%Y-%m-%d")
            except Exception:
                pass
    return datetime.now().strftime("%Y-%m-%d")


# ──────────────────────────────────────────
# Notion API helpers
# ──────────────────────────────────────────

def notion_request(method: str, endpoint: str, body: dict = None) -> dict:
    url = f"https://api.notion.com/v1/{endpoint}"
    headers = {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode()
        print(f"Notion API error {e.code}: {err}", file=sys.stderr)
        raise


def _rich_text(content: str) -> list:
    # Notion rich_text has a 2000-character limit
    chunks = []
    for i in range(0, len(content), 2000):
        chunks.append({"type": "text", "text": {"content": content[i:i+2000]}})
    return chunks or [{"type": "text", "text": {"content": ""}}]


def heading2(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_2",
        "heading_2": {"rich_text": _rich_text(text[:2000])},
    }


def paragraph(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": _rich_text(text[:2000])},
    }


def bullet(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": _rich_text(text[:2000])},
    }


def code_block(text: str, lang: str = "plain text") -> dict:
    return {
        "object": "block",
        "type": "code",
        "code": {
            "rich_text": _rich_text(text[:2000]),
            "language": lang,
        },
    }


def divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


# ──────────────────────────────────────────
# Notion page creation
# ──────────────────────────────────────────

def build_page_blocks(
    user_msgs: list[dict],
    tool_uses: list[dict],
    bash_cmds: list[dict],
    edited_files: list[str],
) -> list[dict]:
    blocks = []

    # ── What was intended ──
    blocks.append(heading2("What was intended"))
    if user_msgs:
        # Show the first 5 messages
        for m in user_msgs[:5]:
            text = m["text"]
            ts_str = ""
            if m.get("ts"):
                try:
                    dt = datetime.fromisoformat(m["ts"].replace("Z", "+00:00"))
                    dt_jst = dt.astimezone(timezone(  # JST
                        __import__("datetime").timedelta(hours=9)))
                    ts_str = dt_jst.strftime("%H:%M ")
                except Exception:
                    pass
            blocks.append(bullet(f"{ts_str}{text[:300]}"))
    else:
        blocks.append(paragraph("(No user messages)"))

    blocks.append(divider())

    # ── Created/modified files ──
    blocks.append(heading2("Created/modified files"))
    if edited_files:
        for fp in edited_files:
            blocks.append(bullet(fp))
    else:
        blocks.append(paragraph("(None)"))

    blocks.append(divider())

    # ── Executed commands ──
    blocks.append(heading2("Executed commands (Bash)"))
    if bash_cmds:
        for cmd_info in bash_cmds[:30]:
            desc = cmd_info.get("desc", "")
            cmd = cmd_info["cmd"]
            label = f"# {desc}\n{cmd}" if desc else cmd
            blocks.append(code_block(label[:2000], "bash"))
    else:
        blocks.append(paragraph("(No Bash commands)"))

    blocks.append(divider())

    # ── Tool usage summary ──
    blocks.append(heading2("Tool usage summary"))
    tool_summary: dict[str, int] = {}
    for tu in tool_uses:
        tool_summary[tu["name"]] = tool_summary.get(tu["name"], 0) + 1
    if tool_summary:
        lines = [f"{name}: {count}x" for name, count in sorted(tool_summary.items(), key=lambda x: -x[1])]
        blocks.append(paragraph(" / ".join(lines)))
    else:
        blocks.append(paragraph("(None)"))

    blocks.append(divider())

    # ── Full user messages (including later ones) ──
    blocks.append(heading2("Conversation flow (user messages)"))
    for m in user_msgs:
        text = m["text"]
        ts_str = ""
        if m.get("ts"):
            try:
                dt = datetime.fromisoformat(m["ts"].replace("Z", "+00:00"))
                import datetime as dt_mod
                dt_jst = dt.astimezone(dt_mod.timezone(dt_mod.timedelta(hours=9)))
                ts_str = dt_jst.strftime("%H:%M ")
            except Exception:
                pass
        blocks.append(bullet(f"{ts_str}{text[:500]}"))

    blocks.append(divider())

    # ── Feedback ──
    blocks.append(heading2("Feedback"))
    blocks.append(paragraph("(To be filled in later)"))

    return blocks


def create_notion_page(title: str, date_str: str, description: str, blocks: list[dict],
                       script: str = "") -> dict:
    body = {
        "parent": {"database_id": DATABASE_ID},
        "properties": {
            "Name": {
                "title": [{"type": "text", "text": {"content": title[:200]}}]
            },
            "Date": {
                "date": {"start": date_str}
            },
            "Description": {
                "rich_text": [{"type": "text", "text": {"content": description[:2000]}}]
            },
            "Type": {
                "select": {"name": "作業ログ"}  # TODO-JP: Notion property
            },
        },
        # Notion API allows up to 100 blocks per request
        "children": blocks[:100],
    }
    if script:
        body["properties"]["Script"] = {
            "rich_text": [{"type": "text", "text": {"content": script[:2000]}}]
        }
    return notion_request("POST", "pages", body)


def append_blocks(page_id: str, blocks: list[dict]):
    """Append blocks in chunks of 100 when exceeding the limit."""
    for i in range(0, len(blocks), 100):
        chunk = blocks[i:i+100]
        notion_request("PATCH", f"blocks/{page_id}/children", {"children": chunk})


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def find_latest_session() -> Path:
    jsonl_files = sorted(SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not jsonl_files:
        raise FileNotFoundError(f"No session files found: {SESSION_DIR}")
    return jsonl_files[0]


def run(jsonl_path: Path | None = None):
    if jsonl_path is None:
        jsonl_path = find_latest_session()

    print(f"Session file: {jsonl_path}")
    entries = load_entries(jsonl_path)
    print(f"  Entries: {len(entries)}")

    user_msgs = extract_user_messages(entries)
    tool_uses = extract_tool_uses(entries)
    bash_cmds = extract_bash_commands(tool_uses)
    edited_files = extract_edited_files(tool_uses)

    print(f"  User messages: {len(user_msgs)}")
    print(f"  Tool calls: {len(tool_uses)} (Bash: {len(bash_cmds)})")
    print(f"  Edited files: {len(edited_files)}")

    date_str = get_session_date(entries)
    title, desc = guess_title_and_description(user_msgs)

    # Infer script name from edited files
    script_str = ", ".join(
        Path(fp).name for fp in edited_files if fp.endswith(".py")
    )

    print(f"\nCreating Notion page: {title}")
    print(f"  Date: {date_str}")

    blocks = build_page_blocks(user_msgs, tool_uses, bash_cmds, edited_files)

    # Create page with the first 100 blocks as body
    result = create_notion_page(title, date_str, desc, blocks[:100], script=script_str)
    page_id = result["id"]
    page_url = result.get("url", f"https://notion.so/{page_id.replace('-', '')}")

    # Append remaining blocks
    if len(blocks) > 100:
        append_blocks(page_id, blocks[100:])

    print(f"\nDone: {page_url}")
    return page_url


if __name__ == "__main__":
    jsonl_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(jsonl_path)
