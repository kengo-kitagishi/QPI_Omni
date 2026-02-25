"""
clickup_helper.py  -  ClickUpタスク作成ユーティリティ

使い方（Claudeが内部で呼び出す）:
    python scripts/clickup_helper.py add --name "タスク名" --list experiment
    python scripts/clickup_helper.py add --name "タスク名" --list code --date 2026-02-27 --start 10 --duration 2
    python scripts/clickup_helper.py add --name "午前タスク" --list experiment --date 2026-02-27 --time morning
    python scripts/clickup_helper.py add --name "午後タスク" --list experiment --date 2026-02-27 --time afternoon
"""

import argparse
import json
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# =============================================================================
# 設定
# =============================================================================
def _load_clickup_token() -> str:
    """APIトークンを .cursor/mcp.json から読み込む（gitignore済みファイル）"""
    import json as _json
    mcp_path = Path(__file__).parent.parent / ".cursor" / "mcp.json"
    if mcp_path.exists():
        with open(mcp_path, encoding="utf-8") as f:
            config = _json.load(f)
        token = config.get("mcpServers", {}).get("clickup", {}).get("env", {}).get("CLICKUP_API_TOKEN", "")
        if token:
            return token
    raise RuntimeError(".cursor/mcp.json にCLICKUP_API_TOKENが見つかりません。")

TOKEN = _load_clickup_token()

LISTS = {
    "experiment": "901813997604",  # QPI > 3_EXPERIMENT
    "code":       "901813997608",  # QPI > 4_CODE
    "plan":       "901813997590",  # QPI > 1_PLAN
    "manuscript": "901813997612",  # QPI > 5_MANUSCRIPT
    "meeting":    "901813997648",  # MEETING > 学術発表
    "other":      "901814001256",  # OTHERs > インテリア（デフォルト）
}

TIME_PRESETS = {
    "morning":   (9,  12),   # 午前: 9:00〜12:00
    "afternoon": (13, 16),   # 午後: 13:00〜16:00
    "default":   (12, 14),   # 指定なし: 12:00〜14:00
}


# =============================================================================
# ユーティリティ
# =============================================================================

def to_jst_ms(date_str: str, hour: int, minute: int = 0) -> int:
    """日付文字列(YYYY-MM-DD)とJST時刻をUnixミリ秒に変換"""
    # JSTはUTC+9なのでhour-9でUTCに変換
    utc_hour = hour - 9
    date = datetime.strptime(date_str, "%Y-%m-%d")
    dt = datetime(date.year, date.month, date.day, utc_hour, minute, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def add_task(name: str, list_key: str, date: str, start_hour: int, end_hour: int, description: str = ""):
    list_id = LISTS.get(list_key, LISTS["experiment"])

    payload = {
        "name": name,
        "due_date": to_jst_ms(date, end_hour),
        "due_date_time": True,
        "start_date": to_jst_ms(date, start_hour),
        "start_date_time": True,
        "time_estimate": (end_hour - start_hour) * 3600 * 1000,
    }
    if description:
        payload["description"] = description

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"https://api.clickup.com/api/v2/list/{list_id}/task",
        data=data,
        headers={"Authorization": TOKEN, "Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as r:
        result = json.loads(r.read())

    print(f"作成完了: {result['name']}")
    print(f"URL: {result.get('url', '')}")
    print(f"時間: {date} {start_hour:02d}:00〜{end_hour:02d}:00 JST")
    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ClickUpタスク作成ツール")
    subparsers = parser.add_subparsers(dest="command")

    add_parser = subparsers.add_parser("add", help="タスクを追加する")
    add_parser.add_argument("--name",        required=True,  help="タスク名")
    add_parser.add_argument("--list",        default="experiment",
                            choices=list(LISTS.keys()), help="保存先リスト")
    add_parser.add_argument("--date",        default=None,   help="日付 YYYY-MM-DD（省略時は今日）")
    add_parser.add_argument("--time",        default=None,
                            choices=["morning", "afternoon", "default"], help="時間プリセット")
    add_parser.add_argument("--start",       type=int, default=None, help="開始時刻（時）")
    add_parser.add_argument("--duration",    type=int, default=None, help="所要時間（時間）")
    add_parser.add_argument("--description", default="", help="説明文")

    args = parser.parse_args()

    if args.command == "add":
        date = args.date or datetime.now().strftime("%Y-%m-%d")

        if args.start is not None and args.duration is not None:
            start_hour = args.start
            end_hour = args.start + args.duration
        elif args.time:
            start_hour, end_hour = TIME_PRESETS[args.time]
        else:
            start_hour, end_hour = TIME_PRESETS["default"]

        add_task(args.name, args.list, date, start_hour, end_hour, args.description)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
