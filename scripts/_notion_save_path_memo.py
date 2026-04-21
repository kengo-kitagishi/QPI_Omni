"""Save a memo about the Windows PATH setx truncation issue to Notion"""
import json
import sys
from datetime import datetime
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).parent.parent
DB_ID = "312eda96-228e-8165-9726-cd75b221357a"


def _load_headers():
    mcp_path = _REPO_ROOT / ".cursor" / "mcp.json"
    with open(mcp_path, encoding="utf-8") as f:
        config = json.load(f)
    headers_str = (
        config.get("mcpServers", {})
        .get("notionApi", {})
        .get("env", {})
        .get("OPENAPI_MCP_HEADERS", "{}")
    )
    headers = json.loads(headers_str)
    return dict(headers) | {"Content-Type": "application/json"}


def rt(s):
    return [{"type": "text", "text": {"content": s[:2000]}}]


def h2(s):
    return {"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt(s)}}


def bl(s):
    return {"object": "block", "type": "bulleted_list_item", "bulleted_list_item": {"rich_text": rt(s)}}


def p(s):
    return {"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt(s)}}


def main():
    today = datetime.now().strftime("%Y-%m-%d")
    title = "Windows PATH setx 1024-character truncation issue"

    blocks = [
        h2("What happened"),
        p("setx PATH truncated PATH at 1024 characters and everything beyond that was lost."),
        p("On the QPI account, a truncated entry 'C:\\Program Fil' remains."),
        h2("Original paths that were cut off"),
        bl("C:\\Program Fil -> originally C:\\Program Files\\Git\\cmd"),
        bl("'es\\Git\\cmd' was lost"),
        h2("Other paths that may have been cut off"),
        p("Any paths beyond 1024 characters may have been entirely removed."),
        bl("C:\\Program Files\\Git\\cmd (Git)"),
        bl("C:\\Users\\QPI\\.local\\bin (Claude Code)"),
        bl("Anaconda / Miniconda related (Python)"),
        bl("Other entries that were near the end of PATH"),
        h2("How to fix"),
        bl("The most reliable approach is to reference the PATH from the kitagishi-kengo account"),
        bl("1. On kitagishi-kengo, open Environment Variables -> Path"),
        bl("2. Copy the full list and take notes"),
        bl("3. Compare with the PATH on QPI"),
        bl("4. Add the missing paths to QPI"),
        bl("5. Delete corrupted entries like 'C:\\Program Fil' and add the correct full paths"),
        p("Alternatively, copy the entire Path from kitagishi-kengo and paste it into QPI, then add only the paths needed for QPI."),
        h2("Paths that must be added (minimum set)"),
        bl("C:\\Program Files\\Git\\cmd (Git)"),
        bl("C:\\Users\\QPI\\.local\\bin (Claude Code)"),
        h2("Caution"),
        p("Do not use setx PATH. Adding via the GUI is safer."),
    ]

    properties = {
        "Name": {"title": [{"type": "text", "text": {"content": title}}]},
        "Date": {"date": {"start": today}},
        "Type": {"select": {"name": "作業ログ"}},  # TODO-JP: Notion property name — confirm before translating
        "Script": {"rich_text": [{"type": "text", "text": {"content": "scripts/_notion_save_path_memo.py"}}]},
        "Description": {"rich_text": [{"type": "text", "text": {"content": "Cause and workaround for the Windows PATH setx 1024-character truncation"}}]},
    }

    headers = _load_headers()
    payload = {
        "parent": {"database_id": DB_ID},
        "properties": properties,
        "children": blocks,
    }

    resp = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload)
    data = resp.json()

    if resp.status_code == 200:
        pid = data["id"]
        url = f"https://www.notion.so/{pid.replace('-', '')}"
        print(f"SUCCESS: {url}")
        return 0
    else:
        print(f"ERROR {resp.status_code}: {json.dumps(data, ensure_ascii=False, indent=2)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
