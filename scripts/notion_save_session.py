"""
Save the current conversation session to Notion (one-shot execution)

Usage:
    python scripts/notion_save_session.py
    python scripts/notion_save_session.py <jsonl_path>   # save the specified file
"""

import json
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = Path(r"C:\Users\QPI\.cursor\projects\c-Users-QPI-Documents-QPI-Omni\agent-transcripts")
NOTION_DB_ID = "312eda96228e81659726cd75b221357a"


def _load_notion_token() -> str:
    mcp_path = REPO_ROOT / ".cursor" / "mcp.json"
    if mcp_path.exists():
        with open(mcp_path, encoding="utf-8") as f:
            config = json.load(f)
        headers_str = config.get("mcpServers", {}).get("notionApi", {}).get("env", {}).get("OPENAPI_MCP_HEADERS", "{}")
        headers = json.loads(headers_str)
        auth = headers.get("Authorization", "")
        return auth.replace("Bearer ", "").strip()
    raise RuntimeError(".cursor/mcp.json not found. Please set the Notion API token.")


def parse_jsonl(jsonl_path: Path) -> dict:
    """Same parsing logic as chat_logger"""
    import re
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

    first_user = next((m["text"] for m in messages if m["role"] == "user"), "")
    if "<user_query>" in first_user:
        start = first_user.find("<user_query>") + len("<user_query>")
        end = first_user.find("</user_query>")
        first_user = first_user[start:end].strip() if end > start else first_user[start:].strip()
    title = first_user[:80].replace("\n", " ") + ("…" if len(first_user) > 80 else "")

    assistant_msgs = [m["text"] for m in messages if m["role"] == "assistant"]
    last_assistant = assistant_msgs[-1] if assistant_msgs else ""
    summary = last_assistant[:500].replace("\n", " ") + ("…" if len(last_assistant) > 500 else "")

    all_text = " ".join(m["text"] for m in messages)
    files_touched = list(set(re.findall(r'scripts/[\w_./]+\.py', all_text)))[:5]
    n_turns = sum(1 for m in messages if m["role"] == "user")

    return {
        "title": title or "(no title)",
        "first_user": first_user[:300],
        "summary": summary,
        "n_turns": n_turns,
        "files_touched": files_touched,
    }


def post_to_notion(info: dict, date_str: str, token: str) -> str:
    import urllib.request

    def rt(s):
        return [{"type": "text", "text": {"content": str(s)[:2000]}}]

    files_str = ", ".join(f"`{f}`" for f in info["files_touched"]) if info["files_touched"] else "None"

    children = [
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt("Conversation summary")}},
        {"object": "block", "type": "quote", "quote": {"rich_text": rt(info["first_user"][:1000] or "(could not retrieve)")}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt("Last AI response")}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt(info["summary"] or "(could not retrieve)")}},
        {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt("Files touched")}},
        {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt(files_str)}},
    ]

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Name": {"title": rt(f"[Conversation] {info['title']}")},
            "Date": {"date": {"start": date_str}},
            "Script": {"rich_text": rt(f"Conversation log ({info['n_turns']} turns)")},
            "Description": {"rich_text": rt(info["summary"][:500])},
            "Parameters": {"rich_text": rt("N/A")},
            "Changed": {"rich_text": rt(", ".join(info["files_touched"]) or "None")},
            "Figure": {"rich_text": rt("None")},
        },
        "children": children,
    }

    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        "https://api.notion.com/v1/pages",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "Notion-Version": "2022-06-28",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    return result.get("url", "")


def main():
    if len(sys.argv) >= 2:
        jsonl_path = Path(sys.argv[1])
    else:
        jsonl_files = list(TRANSCRIPTS_DIR.glob("*/*.jsonl"))
        if not jsonl_files:
            print("[notion_save_session] no .jsonl found inside agent-transcripts")
            sys.exit(1)
        jsonl_path = max(jsonl_files, key=lambda p: p.stat().st_mtime)

    print(f"[notion_save_session] target: {jsonl_path}")

    token = _load_notion_token()
    info = parse_jsonl(jsonl_path)
    date_str = datetime.now().strftime("%Y-%m-%d")

    url = post_to_notion(info, date_str, token)
    print(f"[notion_save_session] Saved to Notion")
    print(f"  -> {url}")


if __name__ == "__main__":
    main()
