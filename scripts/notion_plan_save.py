"""
Save a plan file to Notion QPI Research Notes.

Usage:
    python scripts/notion_plan_save.py <plan_file_path>
    python scripts/notion_plan_save.py "C:\\Users\\QPI\\.cursor\\plans\\プランnotion保存とカレンダー表示_99503eb8.plan.md"

If the Type property is missing, notion_setup_type.py creates it automatically, and
Type=プラン is set automatically when saving.
"""
import json
import re
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


def _parse_plan(path: Path) -> tuple[str, str, str]:
    """Parse plan.md and return (title, date_str, body_md)."""
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if match:
        front, body = match.group(1), match.group(2)
        name = ""
        for line in front.split("\n"):
            if line.startswith("name:"):
                name = line.replace("name:", "").strip()
                break
        if not name:
            name = path.stem.replace(".plan", "")
    else:
        name = path.stem.replace(".plan", "")
        body = text

    today = datetime.now().strftime("%Y-%m-%d")
    return name, today, body.strip()


def _md_to_blocks(md: str) -> list[dict]:
    """Convert Markdown to Notion blocks (simplified)."""
    blocks = []
    for line in md.split("\n"):
        line = line.rstrip()
        if not line:
            continue
        content = line[:2000]
        rt = [{"type": "text", "text": {"content": content}}]
        if line.startswith("### "):
            blocks.append({"object": "block", "type": "heading_3", "heading_3": {"rich_text": rt}})
        elif line.startswith("## "):
            blocks.append({"object": "block", "type": "heading_2", "heading_2": {"rich_text": rt}})
        elif line.startswith("# "):
            blocks.append({"object": "block", "type": "heading_1", "heading_1": {"rich_text": rt}})
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            bullet_content = line.strip()[2:][:2000]
            blocks.append({
                "object": "block",
                "type": "bulleted_list_item",
                "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": bullet_content}}]},
            })
        elif line.strip().startswith("|") and "|" in line[1:]:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt}})
        else:
            blocks.append({"object": "block", "type": "paragraph", "paragraph": {"rich_text": rt}})
    return blocks


def _find_existing_plan_page(headers: dict, full_title: str) -> str | None:
    """Search for an existing plan page with the same title. Return page_id if found."""
    try:
        resp = requests.post(
            f"https://api.notion.com/v1/databases/{DB_ID}/query",
            headers=headers,
            json={
                "filter": {
                    "property": "Name",
                    "title": {"equals": full_title},
                }
            },
        )
        if resp.status_code != 200:
            return None
        results = resp.json().get("results", [])
        if results:
            return results[0]["id"]
    except Exception:
        pass
    return None


def save_plan_to_notion(plan_path: Path) -> str | None:
    """Save a plan to Notion. Skip if a page with the same name already exists. Return URL on success."""
    if not plan_path.exists():
        print(f"ERROR: File not found: {plan_path}")
        return None

    # Add the Type property if it does not exist
    _scripts = _REPO_ROOT / "scripts"
    if str(_scripts) not in sys.path:
        sys.path.insert(0, str(_scripts))
    from notion_setup_type import ensure_type_property
    ensure_type_property()

    title, date_str, body = _parse_plan(plan_path)
    full_title = f"[プラン] {title}"  # TODO-JP: Notion property
    headers = _load_headers()

    # Duplicate check
    existing_id = _find_existing_plan_page(headers, full_title)
    if existing_id:
        existing_url = f"https://www.notion.so/{existing_id.replace('-', '')}"
        print(f"SKIP: already exists -> {existing_url}")
        return existing_url

    blocks = _md_to_blocks(body)
    if not blocks:
        blocks = [{"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"type": "text", "text": {"content": "(empty)"}}]}}]

    properties = {
        "Name": {"title": [{"type": "text", "text": {"content": full_title}}]},
        "Date": {"date": {"start": date_str}},
        "Type": {"select": {"name": "プラン"}},  # TODO-JP: Notion property
    }
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
        return url
    else:
        print(f"ERROR {resp.status_code}: {json.dumps(data, ensure_ascii=False, indent=2)}")
        return None


def _resolve_plan_path(path_arg: str) -> Path | None:
    """Resolve the plan file Path from a command line argument. Handles Japanese paths and encoding issues."""
    path_arg = path_arg.strip()
    candidates = [
        Path(path_arg),
        _REPO_ROOT / path_arg,
        _REPO_ROOT / path_arg.replace("\\", "/"),
    ]
    for p in candidates:
        if p.exists():
            return p

    # If the path does not exist: search docs/plans by the hash portion
    # - When "docs\plans\xxx_99503eb8.plan.md" is mojibake
    # - When only "99503eb8" is specified
    hash_match = re.search(r"([a-f0-9]{8})", path_arg, re.I)
    if hash_match:
        hash_part = hash_match.group(1)
        matches = list((_REPO_ROOT / "docs" / "plans").glob(f"*{hash_part}*.plan.md"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]  # If multiple, return the first one

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python notion_plan_save.py <plan_file_path>")
        print("  Example: python notion_plan_save.py docs/plans/notion_auto_memo_619be55c.plan.md")
        sys.exit(1)

    plan_file = _resolve_plan_path(sys.argv[1])
    if plan_file is None:
        print(f"ERROR: File not found: {sys.argv[1]}")
        print("  (Please check .plan.md under docs/plans/)")
        sys.exit(1)

    save_plan_to_notion(plan_file)


if __name__ == "__main__":
    main()
