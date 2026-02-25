"""
chat_logger.py  -  Cursorの会話ログを自動でNotionに保存するバックグラウンドスクリプト

使い方:
    python scripts/chat_logger.py

    または start_chat_logger.bat をダブルクリック

仕組み:
    C:/Users/QPI/.cursor/projects/.../agent-transcripts/ を監視
    → 会話が15分間更新されなくなったら「終了」と判断
    → .jsonlを解析して会話タイトル・要約を抽出
    → NotionのQPI Research Notesデータベースに自動保存
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
# 設定
# =============================================================================
TRANSCRIPTS_DIR = Path(r"C:/Users/QPI/.cursor/projects/c-Users-QPI-Documents-QPI-omni/agent-transcripts")
NOTION_DB_ID    = "312eda96228e81659726cd75b221357a"

def _load_notion_token() -> str:
    """APIトークンを .cursor/mcp.json から読み込む（gitignore済みファイル）"""
    mcp_path = Path(__file__).parent.parent / ".cursor" / "mcp.json"
    if mcp_path.exists():
        with open(mcp_path, encoding="utf-8") as f:
            config = json.load(f)
        headers_str = config.get("mcpServers", {}).get("notionApi", {}).get("env", {}).get("OPENAPI_MCP_HEADERS", "{}")
        headers = json.loads(headers_str)
        auth = headers.get("Authorization", "")
        return auth.replace("Bearer ", "").strip()
    raise RuntimeError(".cursor/mcp.json が見つかりません。Notion APIトークンを設定してください。")

NOTION_TOKEN = _load_notion_token()
REPO_ROOT       = Path(__file__).parent.parent

INACTIVITY_MINUTES = 15   # 何分更新がなければ会話終了とみなすか
POLL_INTERVAL_SEC  = 60   # 何秒おきにフォルダを確認するか

PROCESSED_LOG = REPO_ROOT / ".figure_history" / "chat_log.json"


# =============================================================================
# ユーティリティ
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
    .jsonlを解析してNotionに送る情報を抽出する。
    戻り値: {"title": str, "first_user": str, "summary": str, "n_turns": int, "files_touched": list}
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
        return {"title": "（解析エラー）", "first_user": str(e), "summary": "", "n_turns": 0, "files_touched": []}

    if not messages:
        return {"title": "（空の会話）", "first_user": "", "summary": "", "n_turns": 0, "files_touched": []}

    # 最初のユーザーメッセージをタイトルに
    first_user = next((m["text"] for m in messages if m["role"] == "user"), "")
    # <user_query>タグがあれば中身だけ取る
    if "<user_query>" in first_user:
        start = first_user.find("<user_query>") + len("<user_query>")
        end   = first_user.find("</user_query>")
        first_user = first_user[start:end].strip() if end > start else first_user[start:].strip()
    title = first_user[:80].replace("\n", " ") + ("…" if len(first_user) > 80 else "")

    # アシスタントの最後のメッセージを要約として使う
    assistant_msgs = [m["text"] for m in messages if m["role"] == "assistant"]
    last_assistant = assistant_msgs[-1] if assistant_msgs else ""
    summary = last_assistant[:500].replace("\n", " ") + ("…" if len(last_assistant) > 500 else "")

    # 作成・変更されたファイルのパスを拾う（簡易）
    all_text = " ".join(m["text"] for m in messages)
    import re
    files_touched = list(set(re.findall(r'scripts/[\w_./]+\.py', all_text)))[:5]

    n_turns = sum(1 for m in messages if m["role"] == "user")

    return {
        "title":        title or "（タイトルなし）",
        "first_user":   first_user[:300],
        "summary":      summary,
        "n_turns":      n_turns,
        "files_touched": files_touched,
    }


def post_to_notion(info: dict, date_str: str):
    def rt(s): return [{"type": "text", "text": {"content": s[:2000]}}]

    files_str = ", ".join(f"`{f}`" for f in info["files_touched"]) if info["files_touched"] else "なし"

    children = [
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": rt("会話の概要")}},
        {"object": "block", "type": "quote",
         "quote": {"rich_text": rt(info["first_user"][:1000] or "（取得できませんでした）")}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": rt("最後のAI応答")}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": rt(info["summary"] or "（取得できませんでした）")}},
        {"object": "block", "type": "heading_2",
         "heading_2": {"rich_text": rt("操作されたファイル")}},
        {"object": "block", "type": "paragraph",
         "paragraph": {"rich_text": rt(files_str)}},
    ]

    payload = {
        "parent": {"database_id": NOTION_DB_ID},
        "properties": {
            "Name":        {"title": rt(f"[会話] {info['title']}")},
            "Date":        {"date": {"start": date_str}},
            "Script":      {"rich_text": rt(f"会話ログ ({info['n_turns']}ターン)")},
            "Description": {"rich_text": rt(info["summary"][:500])},
            "Parameters":  {"rich_text": rt("N/A")},
            "Changed":     {"rich_text": rt(", ".join(info["files_touched"]) or "なし")},
            "Figure":      {"rich_text": rt("なし")},
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
# メインループ
# =============================================================================

def find_jsonl(folder: Path) -> Path | None:
    """会話フォルダ内の.jsonlファイルを返す"""
    for f in folder.glob("*.jsonl"):
        return f
    return None


def main():
    print(f"[chat_logger] 監視開始: {TRANSCRIPTS_DIR}")
    print(f"[chat_logger] {INACTIVITY_MINUTES}分間更新がなければNotionに保存します")
    print(f"[chat_logger] 停止: Ctrl+C\n")

    processed = load_processed()

    while True:
        try:
            if not TRANSCRIPTS_DIR.exists():
                print(f"[chat_logger] 警告: フォルダが見つかりません: {TRANSCRIPTS_DIR}")
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

                # 最終更新時刻を確認
                mtime = jsonl.stat().st_mtime
                age_min = (time.time() - mtime) / 60

                if age_min < INACTIVITY_MINUTES:
                    continue  # まだ会話中かもしれない

                # 解析してNotionに保存
                print(f"[chat_logger] 会話終了を検出: {conv_id}")
                info = parse_jsonl(jsonl)
                date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")

                try:
                    url = post_to_notion(info, date_str)
                    print(f"[chat_logger] Notionに保存: {info['title'][:50]}")
                    print(f"             → {url}")
                    processed.add(conv_id)
                    save_processed(processed)
                except Exception as e:
                    print(f"[chat_logger] Notion保存エラー: {e}")

        except KeyboardInterrupt:
            print("\n[chat_logger] 停止しました")
            sys.exit(0)
        except Exception as e:
            print(f"[chat_logger] エラー: {e}")

        time.sleep(POLL_INTERVAL_SEC)


if __name__ == "__main__":
    main()
