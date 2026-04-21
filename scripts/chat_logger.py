"""
chat_logger.py  -  Background script that automatically saves Cursor conversation logs to Notion.

Usage:
    python scripts/chat_logger.py

    Or double-click start_chat_logger.bat

How it works:
    Watches C:/Users/QPI/.cursor/projects/.../agent-transcripts/
    -> If a conversation has not been updated for 15 minutes, it is treated as "finished"
    -> Parses the .jsonl and extracts the conversation title and summary
    -> Automatically saves to the QPI Research Notes database on Notion
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

# =============================================================================
# Settings
# =============================================================================
TRANSCRIPTS_DIR = Path(r"C:/Users/QPI/.cursor/projects/c-Users-QPI-Documents-QPI-omni/agent-transcripts")
NOTION_DB_ID    = "312eda96228e81659726cd75b221357a"

def _load_notion_token() -> str:
    """Load the API token from .cursor/mcp.json (gitignored file)."""
    mcp_path = Path(__file__).parent.parent / ".cursor" / "mcp.json"
    if mcp_path.exists():
        with open(mcp_path, encoding="utf-8") as f:
            config = json.load(f)
        headers_str = config.get("mcpServers", {}).get("notionApi", {}).get("env", {}).get("OPENAPI_MCP_HEADERS", "{}")
        headers = json.loads(headers_str)
        auth = headers.get("Authorization", "")
        return auth.replace("Bearer ", "").strip()
    raise RuntimeError(".cursor/mcp.json not found. Please configure the Notion API token.")

NOTION_TOKEN = _load_notion_token()
REPO_ROOT       = Path(__file__).parent.parent

INACTIVITY_MINUTES = 15   # How many minutes of no updates before treating a conversation as finished
POLL_INTERVAL_SEC  = 60   # How many seconds between folder checks

PROCESSED_LOG = REPO_ROOT / ".figure_history" / "chat_log.json"


# =============================================================================
# Utilities
# =============================================================================

def load_processed() -> set:
    if PROCESSED_LOG.exists():
        with open(PROCESSED_LOG, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_processed(processed: set):
    PROCESSED_LOG.parent.mkdir(exist_ok=True)
    with open(PROCESSED_LOG, "w", encoding="utf-8") as f:
        json.dump(list(processed), f, ensure_ascii=False, indent=2)


def parse_jsonl(jsonl_path: Path) -> dict:
    """
    Parse the .jsonl and extract information to send to Notion.
    Return value: {"title": str, "first_user": str, "summary": str, "n_turns": int, "files_touched": list}
    """
    messages = []
    try:
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    role = obj.get("role", "")
                    content = obj.get("message", {}).get("content", [])
                    text = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text += block.get("text", "")
                    if role and text:
                        messages.append({"role": role, "text": text})
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return {"title": "(parse error)", "first_user": str(e), "summary": "", "n_turns": 0, "files_touched": []}

    if not messages:
        return {"title": "(empty conversation)", "first_user": "", "summary": "", "n_turns": 0, "files_touched": []}

    # Use the first user message as the title
    first_user = next((m["text"] for m in messages if m["role"] == "user"), "")
    # If a <user_query> tag is present, extract just its contents
    if "<user_query>" in first_user:
        start = first_user.find("<user_query>") + len("<user_query>")
        end   = first_user.find("</user_query>")
        first_user = first_user[start:end].strip() if end > start else first_user[start:].strip()
    title = first_user[:80].replace("\n", " ") + ("…" if len(first_user) > 80 else "")

    # Use the assistant's last message as the summary
    assistant_msgs = [m["text"] for m in messages if m["role"] == "assistant"]
    last_assistant = assistant_msgs[-1] if assistant_msgs else ""
    summary = last_assistant[:500].replace("\n", " ") + ("…" if len(last_assistant) > 500 else "")

    # Pick up paths of created/modified files (simple heuristic)
    all_text = " ".join(m["text"] for m in messages)
    import re
    files_touched = list(set(re.findall(r'scripts/[\w_./]+\.py', all_text)))[:5]

    n_turns = sum(1 for m in messages if m["role"] == "user")

    return {
        "title":        title or "(no title)",
        "first_user":   first_user[:300],
        "summary":      summary,
        "n_turns":      n_turns,
        "files_touched": files_touched,
    }


def post_to_notion(info: dict, date_str: str):
    def rt(s): return [{"type": "text", "text": {"content": s[:2000]}}]

    files_str = ", ".join(f"`{f}`" for f in info["files_touched"]) if info["files_touched"] else "none"

    children = [
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": rt("Conversation overview")}},
        {"object": "block", "type": "quote",
         "quote": {"rich_text": rt(info["first_user"][:1000] or "(could not retrieve)")}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": rt("Last AI response")}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": rt(info["summary"] or "(could not retrieve)")}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": rt("Files touched")}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": rt(files_str)}},
    ]

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Name":        {"title": rt(f"[conversation] {info['title']}")},
            "Date":        {"date": {"start": date_str}},
            "Script":      {"rich_text": rt(f"conversation log ({info['n_turns']} turns)")},
            "Description": {"rich_text": rt(info["summary"][:500])},
            "Parameters":  {"rich_text": rt("N/A")},
            "Changed":     {"rich_text": rt(", ".join(info["files_touched"]) or "none")},
            "Figure":      {"rich_text": rt("none")},
        },
        "children": children,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        "https://api.notion.com/v1/pages",
        data=data,
        headers={
            "Authorization": f"Bearer {NOTION_TOKEN}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("url", "")


# =============================================================================
# Main loop
# =============================================================================

def find_jsonl(folder: Path) -> Path | None:
    """Return the .jsonl file inside the conversation folder."""
    for f in folder.glob("*.jsonl"):
        return f
    return None


def main():
    print(f"[chat_logger] watching: {TRANSCRIPTS_DIR}")
    print(f"[chat_logger] will save to Notion after {INACTIVITY_MINUTES} minutes of inactivity")
    print(f"[chat_logger] stop: Ctrl+C\n")

    processed = load_processed()

    while True:
        try:
            if not TRANSCRIPTS_DIR.exists():
                print(f"[chat_logger] warning: folder not found: {TRANSCRIPTS_DIR}")
                time.sleep(POLL_INTERVAL_SEC)
                continue

            for conv_folder in TRANSCRIPTS_DIR.iterdir():
                if not conv_folder.is_dir():
                    continue

                conv_id = conv_folder.name
                if conv_id in processed:
                    continue

                jsonl = find_jsonl(conv_folder)
                if jsonl is None:
                    continue

                # Check the last modification time
                mtime = jsonl.stat().st_mtime
                age_min = (time.time() - mtime) / 60

                if age_min < INACTIVITY_MINUTES:
                    continue  # conversation may still be in progress

                # Parse and save to Notion
                print(f"[chat_logger] detected end of conversation: {conv_id}")
                info = parse_jsonl(jsonl)
                date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

                try:
                    url = post_to_notion(info, date_str)
                    print(f"[chat_logger] saved to Notion: {info['title'][:50]}")
                    print(f"             -> {url}")
                    processed.add(conv_id)
                    save_processed(processed)
                except Exception as e:
                    print(f"[chat_logger] Notion save error: {e}")

        except KeyboardInterrupt:
            print("\n[chat_logger] stopped")
            sys.exit(0)
        except Exception as e:
            print(f"[chat_logger] error: {e}")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
