"""
Add the Type property (Select) to the QPI Research Notes database.

Usage:
    python scripts/notion_setup_type.py

Does nothing if Type already exists.
Options: 思考 / 作業ログ / 図 / プラン
"""
import json
from pathlib import Path

import requests

_REPO_ROOT = Path(__file__).parent.parent
DB_ID = "312eda96-228e-8165-9726-cd75b221357a"

TYPE_OPTIONS = [
    # TODO-JP: Notion property option names (kept as-is to match the existing DB)
    {"name": "思考", "color": "gray"},
    {"name": "作業ログ", "color": "gray"},
    {"name": "図", "color": "gray"},
    {"name": "プラン", "color": "blue"},
]


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
    return dict(headers) | {"Content-Type": "application/json", "Notion-Version": "2022-06-28"}


def ensure_type_property() -> bool:
    """Add the Type property if missing. Returns True on success."""
    headers = _load_headers()

    # Fetch the current schema
    resp = requests.get(f"https://api.notion.com/v1/databases/{DB_ID}", headers=headers)
    if resp.status_code != 200:
        print(f"ERROR: failed to fetch database {resp.status_code}: {resp.text}")
        return False

    data = resp.json()
    props = data.get("properties", {})

    if "Type" in props:
        print("Type property already exists. Skipped.")
        return True

    # Add the Type property
    payload = {
        "properties": {
            "Type": {
                "select": {
                    "options": TYPE_OPTIONS,
                }
            }
        }
    }
    resp = requests.patch(f"https://api.notion.com/v1/databases/{DB_ID}", headers=headers, json=payload)
    if resp.status_code != 200:
        print(f"ERROR: failed to add property {resp.status_code}: {json.dumps(resp.json(), ensure_ascii=False, indent=2)}")
        return False

    print("SUCCESS: Type property added (options: 思考 / 作業ログ / 図 / プラン)")
    return True


if __name__ == "__main__":
    ensure_type_property()
