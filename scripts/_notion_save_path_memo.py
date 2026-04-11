"""Windows PATH 環境変数の setx トラブルメモを Notion に保存"""
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
    title = "Windows PATH setx 1024文字切り捨てトラブル"

    blocks = [
        h2("何が起きたか"),
        p("setx PATH で PATH を 1024 文字で打ち切り、それより後ろが全部消えた。"),
        p("QPI アカウントでは「C:\\Program Fil」と途中で切れたエントリが残っている。"),
        h2("切れた元のパス"),
        bl("C:\\Program Fil → 元は C:\\Program Files\\Git\\cmd"),
        bl("「es\\Git\\cmd」が消えた"),
        h2("それ以外が切れている可能性"),
        p("1024 文字より後ろにあったパスは丸ごと消えている可能性がある。"),
        bl("C:\\Program Files\\Git\\cmd（Git）"),
        bl("C:\\Users\\QPI\\.local\\bin（Claude Code）"),
        bl("Anaconda / Miniconda 関連（Python）"),
        bl("その他 PATH の後ろの方にあったもの"),
        h2("直し方"),
        bl("kitagishi-kengo の Path を参考にするのが確実"),
        bl("1. kitagishi-kengo で環境変数 → Path を開く"),
        bl("2. 一覧を全部コピーしてメモ"),
        bl("3. QPI の Path と比較"),
        bl("4. QPI に足りないパスを追加"),
        bl("5. 「C:\\Program Fil」など壊れたエントリを削除し、正しいフルパスを追加"),
        p("または kitagishi-kengo の Path を丸ごとコピーして QPI に貼り付け、QPI 用に必要なパスだけ足す。"),
        h2("追加すべきパス（最低限）"),
        bl("C:\\Program Files\\Git\\cmd（Git）"),
        bl("C:\\Users\\QPI\\.local\\bin（Claude Code）"),
        h2("注意"),
        p("setx PATH は使わない。GUI で追加する方が安全。"),
    ]

    properties = {
        "Name": {"title": [{"type": "text", "text": {"content": title}}]},
        "Date": {"date": {"start": today}},
        "Type": {"select": {"name": "作業ログ"}},
        "Script": {"rich_text": [{"type": "text", "text": {"content": "scripts/_notion_save_path_memo.py"}}]},
        "Description": {"rich_text": [{"type": "text", "text": {"content": "Windows PATH setx 1024文字切り捨ての原因と対処法"}}]},
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
