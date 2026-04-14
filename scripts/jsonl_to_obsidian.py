#!/usr/bin/env python3
"""jsonl_to_obsidian.py

Claude Code セッション JSONL を Obsidian の .md に変換する。

- AI の thinking ブロック（内部推論）を含む完全な記録
- ユーザー発言・AI返答・ツール実行・ツール結果をタイムスタンプ付き時系列で保存
- 増分処理（処理済みセッションはスキップ。アクティブセッションもスキップ）
- SQLite (session_db.py) にメタデータをキャッシュ

出力 .md の構成（上から）:
  1. フロントマター（session_id, date, metrics, edited_files）
  2. 変更ファイル一覧
  3. セッションサマリ（metrics table）
  4. 設計ノート（thinking ブロックの先頭抜粋）
  5. 再現コマンド一覧（Bash コマンドをコピペ可能な形式で）
  6. コード変更詳細（Edit/Write の実際の内容）
  7. エラーログ（エラーがある場合のみ）
  8. タイムライン（全会話・ツール実行の時系列）

使い方:
  python3 scripts/jsonl_to_obsidian.py              # 未処理を全部変換
  python3 scripts/jsonl_to_obsidian.py --dry-run    # ファイル書き出しせず確認のみ
  python3 scripts/jsonl_to_obsidian.py --session /path/to/file.jsonl  # 指定ファイル
  python3 scripts/jsonl_to_obsidian.py --all        # 処理済み含め全セッションを再変換

出力先:
  ~/Documents/Obsidian Vault/00_Inbox/claude_sessions/YYYY-MM-DD_<session_id[:8]>.md
"""

import argparse
import json
import re
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

# session_db を同じディレクトリから import
sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_db

# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────

SESSION_DIR = Path.home() / ".claude/projects/-Users-kitak-QPI-Omni"
SHARED_SESSION_DIR = Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/"
    "共有ドライブ/wakamotolab_meeting/kitagishi/claude_logs/QPI_Omni"
)
DEFAULT_SESSION_DIRS = [SESSION_DIR, SHARED_SESSION_DIR]
OBSIDIAN_SESSIONS_DIR = Path("/Users/kitak/Documents/Obsidian Vault/00_Inbox/claude_sessions")

# アクティブセッション（最終エントリが ACTIVE_THRESHOLD 分以内）はスキップ
ACTIVE_THRESHOLD_MINUTES = 15

JST = timezone(timedelta(hours=9))


# ────────────────────────────────────────────
# JSONL パース
# ────────────────────────────────────────────

def load_entries(path: Path) -> list:
    """JSONL を読み込む。OSError (errno 11/35: deadlock/EAGAIN) は最大3回 retry。
    最終的に失敗したら空リストを返してこのセッションだけスキップする。"""
    last_err = None
    for attempt in range(3):
        try:
            entries = []
            with open(path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            return entries
        except OSError as e:
            last_err = e
            if e.errno in (11, 35):  # EAGAIN / EWOULDBLOCK / Resource deadlock avoided
                time.sleep(0.5 * (attempt + 1))
                continue
            raise
    print(f"  WARN: {path.name} を3回 retry しても読めなかった ({last_err}). スキップします.",
          file=sys.stderr)
    return []


def _strip_tags(text: str) -> str:
    text = re.sub(r"<system-reminder>.*?</system-reminder>", "", text, flags=re.DOTALL)
    text = re.sub(r"<[a-zA-Z-]+>.*?</[a-zA-Z-]+>", "", text, flags=re.DOTALL)
    return text.strip()


def _ts_to_jst(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(JST)
    except Exception:
        return None


def _fmt_ts(ts: str) -> str:
    dt = _ts_to_jst(ts)
    if dt is None:
        return "??:??"
    return dt.strftime("%H:%M")


def get_session_date(entries: list) -> str:
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            dt = _ts_to_jst(ts)
            if dt:
                return dt.strftime("%Y-%m-%d")
    return datetime.now(JST).strftime("%Y-%m-%d")


def get_last_entry_time(entries: list) -> datetime | None:
    last_ts = None
    for e in entries:
        ts = e.get("timestamp", "")
        if ts:
            last_ts = ts
    if last_ts:
        return _ts_to_jst(last_ts)
    return None


def group_timeline_by_date(timeline: list) -> dict[str, list]:
    """timeline イベントを JST 日付ごとにグループ化する。
    タイムスタンプがない（ts=None）イベントは直前のイベントと同じ日付に入れる。
    Returns: {date_str: [events...]} を日付昇順で返す dict。"""
    groups: dict[str, list] = {}
    last_date = None
    for event in timeline:
        ts = event.get("ts")
        if ts:
            dt = _ts_to_jst(ts)
            if dt:
                last_date = dt.strftime("%Y-%m-%d")
        date_key = last_date or "unknown"
        groups.setdefault(date_key, []).append(event)
    return dict(sorted(groups.items()))


def _tool_result_content(result_block: dict) -> str:
    content = result_block.get("content", "")
    if isinstance(content, str):
        return content[:800]
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    texts.append(item.get("text", "")[:400])
                elif item.get("type") == "tool_result":
                    sub = _tool_result_content(item)
                    if sub:
                        texts.append(sub)
            elif isinstance(item, str):
                texts.append(item[:400])
        return "\n".join(texts)[:800]
    return ""


def parse_timeline(entries: list) -> list:
    # tool_result を tool_use_id でインデックス化
    tool_results: dict[str, dict] = {}
    for e in entries:
        if e.get("type") != "user":
            continue
        content = e.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                tid = block.get("tool_use_id", "")
                if tid:
                    tool_results[tid] = block

    timeline = []

    for e in entries:
        ts = e.get("timestamp", "")
        etype = e.get("type")

        if etype == "user":
            msg = e.get("message", {})
            content = msg.get("content", "")

            if isinstance(content, str):
                text = _strip_tags(content)
                if text.strip():
                    timeline.append({"ts": ts, "role": "user", "text": text})

            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        t = _strip_tags(block.get("text", ""))
                        if t.strip():
                            text_parts.append(t)
                if text_parts:
                    timeline.append({"ts": ts, "role": "user",
                                     "text": "\n".join(text_parts)})

        elif etype == "assistant":
            content = e.get("message", {}).get("content", [])
            if not isinstance(content, list):
                continue

            thinking_blocks = []
            text_blocks = []
            tool_calls = []

            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")

                if btype == "thinking":
                    t = block.get("thinking", "").strip()
                    if t:
                        thinking_blocks.append(t)

                elif btype == "text":
                    t = block.get("text", "").strip()
                    if t:
                        text_blocks.append(t)

                elif btype == "tool_use":
                    tid = block.get("id", "")
                    result = tool_results.get(tid, {})
                    tool_calls.append({
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                        "result_text": _tool_result_content(result),
                        "is_error": result.get("is_error", False),
                    })

            if thinking_blocks or text_blocks or tool_calls:
                timeline.append({
                    "ts": ts,
                    "role": "assistant",
                    "thinking": thinking_blocks,
                    "text": text_blocks,
                    "tools": tool_calls,
                })

    return timeline


# ────────────────────────────────────────────
# DB 用データ抽出
# ────────────────────────────────────────────

def extract_db_data(timeline: list) -> dict:
    """タイムラインから DB 書き込み用の構造化データを抽出する。"""
    thinking_count = 0
    user_count = 0
    assistant_count = 0
    tool_count = 0
    error_count = 0
    bash_count = 0

    first_user_message = None
    session_summary = None
    design_notes_list = []

    files: list[dict] = []
    tool_calls_db: list[dict] = []
    bash_cmds_db: list[dict] = []

    event_order = 0

    for event in timeline:
        if event["role"] == "user":
            user_count += 1
            if first_user_message is None:
                first_user_message = event["text"][:300]

        elif event["role"] == "assistant":
            assistant_count += 1
            thinking_count += len(event.get("thinking", []))

            # 設計ノート: thinking ブロックの先頭を抽出
            for tb in event.get("thinking", []):
                design_notes_list.append(tb)

            # セッションサマリ: 最初の AI テキスト
            if session_summary is None:
                for t in event.get("text", []):
                    if t.strip():
                        session_summary = t
                        break

            # ツール呼び出し
            for tc in event.get("tools", []):
                tool_count += 1
                event_order += 1
                name = tc["name"]
                inp = tc["input"]
                is_error = bool(tc.get("is_error"))
                if is_error:
                    error_count += 1

                # input_json (Bash/Edit/Write は完全保存、他は 500 文字)
                inp_str = json.dumps(inp, ensure_ascii=False)
                inp_stored = inp_str if name in ("Bash", "Edit", "Write") else inp_str[:500]

                result_preview = (tc.get("result_text", "") or "")[:500]
                error_text = result_preview if is_error else None

                tool_calls_db.append({
                    "event_order": event_order,
                    "timestamp": event["ts"],
                    "tool_name": name,
                    "input_json": inp_stored,
                    "result_preview": result_preview if not is_error else None,
                    "is_error": is_error,
                    "error_text": error_text,
                })

                # ファイル操作
                if name in ("Edit", "Write", "Read"):
                    fp = inp.get("file_path", "")
                    if fp:
                        old_prev = None
                        new_prev = None
                        if name == "Edit":
                            old_prev = (inp.get("old_string", "") or "")[:1200]
                            new_prev = (inp.get("new_string", "") or "")[:1200]
                        elif name == "Write":
                            new_prev = (inp.get("content", "") or "")[:1200]
                        files.append({
                            "file_path": fp,
                            "operation": name,
                            "old_preview": old_prev,
                            "new_preview": new_prev,
                            "event_order": event_order,
                            "timestamp": event["ts"],
                        })

                # Bash
                if name == "Bash":
                    bash_count += 1
                    cmd = inp.get("command", "")
                    desc = inp.get("description", "")
                    out_prev = (tc.get("result_text", "") or "")[:800]
                    bash_cmds_db.append({
                        "event_order": event_order,
                        "timestamp": event["ts"],
                        "command": cmd,
                        "description": desc,
                        "output_preview": out_prev,
                        "is_error": is_error,
                    })

    design_notes_json = json.dumps(design_notes_list, ensure_ascii=False) if design_notes_list else None

    return {
        "session_meta": {
            "thinking_blocks": thinking_count,
            "user_message_count": user_count,
            "assistant_message_count": assistant_count,
            "tool_call_count": tool_count,
            "error_count": error_count,
            "bash_command_count": bash_count,
            "first_user_message": first_user_message,
            "design_notes": design_notes_json,
            "session_summary": session_summary,
        },
        "files": files,
        "tool_calls": tool_calls_db,
        "bash_commands": bash_cmds_db,
    }


def write_session_to_db(
    conn,
    session_id: str, date_str: str,
    jsonl_path: Path, md_path: Path,
    timeline: list, mtime: float
) -> None:
    """セッションデータを DB に書き込む。"""
    db_data = extract_db_data(timeline)
    meta = db_data["session_meta"]

    session_db.upsert_session(conn, {
        "session_id": session_id,
        "date": date_str,
        "jsonl_path": str(jsonl_path),
        "md_path": str(md_path),
        "jsonl_mtime": mtime,
        **meta,
    })
    session_db.upsert_session_files(conn, session_id, db_data["files"])
    session_db.upsert_tool_calls(conn, session_id, db_data["tool_calls"])
    session_db.upsert_bash_commands(conn, session_id, db_data["bash_commands"])
    conn.commit()


# ────────────────────────────────────────────
# Markdown レンダリング — ヘルパーセクション
# ────────────────────────────────────────────

def _render_tool_input(name: str, inp: dict) -> str:
    if name == "Bash":
        cmd = inp.get("command", "")
        desc = inp.get("description", "")
        if desc:
            return f"`{cmd[:300]}`\n  _{desc}_"
        return f"`{cmd[:300]}`"

    elif name in ("Edit", "Write"):
        fp = inp.get("file_path", "")
        if name == "Edit":
            old = inp.get("old_string", "")[:80].replace("\n", "↵")
            new = inp.get("new_string", "")[:80].replace("\n", "↵")
            return f"`{fp}`\n  `{old}` → `{new}`"
        else:
            content_preview = inp.get("content", "")[:100].replace("\n", "↵")
            return f"`{fp}` (新規/上書き)\n  `{content_preview}...`"

    elif name == "Read":
        fp = inp.get("file_path", "")
        def _to_int(v):
            try:
                return int(str(v).strip().lstrip(",").strip())
            except (ValueError, TypeError):
                return 0
        limit = _to_int(inp.get("limit", 0))
        offset = _to_int(inp.get("offset", 0))
        extra = f" (L{offset}-{offset+limit})" if offset else ""
        return f"`{fp}`{extra}"

    elif name == "Glob":
        return f"`{inp.get('pattern', '')}` in `{inp.get('path', '.')}`"

    elif name == "Grep":
        return f"`{inp.get('pattern', '')}` in `{inp.get('path', '.')}`"

    elif name == "Task":
        return f"[{inp.get('subagent_type', '?')}] {inp.get('description', inp.get('prompt', ''))[:100]}"

    else:
        s = json.dumps(inp, ensure_ascii=False)
        return s[:200]


def _render_session_summary_section(db_data: dict) -> list[str]:
    """セッションサマリ（metrics table）セクションを生成する。"""
    meta = db_data["session_meta"]
    lines = [
        "## セッションサマリ",
        "",
        "| 指標 | 値 |",
        "|------|-----|",
        f"| 変更ファイル数 | {len(set(f['file_path'] for f in db_data['files'] if f['operation'] in ('Edit','Write')))} |",
        f"| Bash コマンド数 | {meta['bash_command_count']} |",
        f"| エラー発生数 | {meta['error_count']} |",
        f"| AI 推論ブロック数 | {meta['thinking_blocks']} |",
        f"| 全ツール呼び出し数 | {meta['tool_call_count']} |",
        f"| ユーザー発言数 | {meta['user_message_count']} |",
        "",
    ]
    return lines


def _render_design_notes_section(timeline: list) -> list[str]:
    """thinking ブロックから設計ノートを抽出するセクションを生成する。"""
    excerpts = []
    for event in timeline:
        if event["role"] != "assistant":
            continue
        for tb in event.get("thinking", []):
            if tb.strip():
                excerpts.append((event["ts"], tb))

    if not excerpts:
        return []

    lines = [
        "## 設計ノート（AI 推論ブロック抜粋）",
        "",
        "> 実装設計・数式・アルゴリズムの根拠が含まれる推論ブロック",
        "",
    ]
    for i, (ts, tb) in enumerate(excerpts, 1):
        ts_str = _fmt_ts(ts)
        # 数式っぽい行を含む場合はマーク
        has_math = bool(re.search(r"[\\$∑∫≈×÷±∂π]|\\frac|\\sum|\\int|np\.|[a-z]\s*=\s*[\d\(]", tb))
        math_mark = " 🔢" if has_math else ""
        lines.append(f"### [{i}] {ts_str}{math_mark}")
        lines.append("")
        for tline in tb.splitlines():
            lines.append(f"> {tline}")
        lines.append("")

    return lines


def _render_bash_section(bash_cmds: list[dict]) -> list[str]:
    """再現コマンド一覧セクションを生成する（コピペ可能な形式）。"""
    if not bash_cmds:
        return []

    lines = [
        "## 再現コマンド一覧",
        "",
        "> このセッションで実行された Bash コマンド（実行順・コピペ可能）",
        "",
    ]

    # エラー概要テーブル
    errors = [bc for bc in bash_cmds if bc["is_error"]]
    ok_count = len(bash_cmds) - len(errors)
    lines.append(f"| 合計 | 成功 | エラー |")
    lines.append(f"|------|------|--------|")
    lines.append(f"| {len(bash_cmds)} | {ok_count} | {len(errors)} |")
    lines.append("")

    # コマンドブロック（コピペ用）
    lines.append("```bash")
    for bc in bash_cmds:
        n = bc.get("event_order", "?")
        ts = _fmt_ts(bc.get("timestamp", ""))
        cmd = bc["command"]
        desc = bc.get("description") or ""
        err_mark = " # ⚠️ ERROR" if bc["is_error"] else ""
        if desc:
            lines.append(f"# [{n}] {ts} — {desc}{err_mark}")
        else:
            lines.append(f"# [{n}] {ts}{err_mark}")
        lines.append(cmd)
        lines.append("")
    lines.append("```")
    lines.append("")

    return lines


def _render_code_changes_section(timeline: list) -> list[str]:
    """コード変更詳細セクションを生成する（実際の old→new を表示）。"""
    changes = []
    event_order = 0
    for event in timeline:
        if event["role"] != "assistant":
            continue
        for tc in event.get("tools", []):
            event_order += 1
            name = tc["name"]
            inp = tc["input"]
            if name not in ("Edit", "Write"):
                continue
            fp = inp.get("file_path", "")
            if not fp:
                continue
            ts_str = _fmt_ts(event["ts"])

            if name == "Edit":
                old = inp.get("old_string", "") or ""
                new = inp.get("new_string", "") or ""
                changes.append({
                    "order": event_order,
                    "ts": ts_str,
                    "op": "Edit",
                    "file": fp,
                    "old": old,
                    "new": new,
                    "old_truncated": False,
                    "new_truncated": False,
                })
            elif name == "Write":
                content = inp.get("content", "") or ""
                changes.append({
                    "order": event_order,
                    "ts": ts_str,
                    "op": "Write",
                    "file": fp,
                    "new": content,
                    "new_truncated": False,
                    "old": None,
                })

    if not changes:
        return []

    # ファイルごとにグルーピング（同ファイルへの複数 Edit は折りたたむ）
    seen_files: dict[str, int] = {}
    lines = [
        "## コード変更詳細",
        "",
        "> このセッションでの全 Edit/Write 操作（実際のコード内容）",
        "",
    ]

    for ch in changes:
        fp = ch["file"]
        seen_files[fp] = seen_files.get(fp, 0) + 1
        count = seen_files[fp]

        # ファイル名を短くする（/Users/kitak/ を省略）
        short_fp = fp.replace("/Users/kitak/QPI_Omni/", "").replace("/Users/kitak/", "~/")
        header = f"### `{short_fp}` — {ch['op']} [{ch['ts']}]"
        if count > 1:
            header += f" (この操作 #{count})"
        lines.append(header)
        lines.append("")

        # 拡張子からコードブロック言語を推定
        ext = Path(fp).suffix.lower()
        lang = {"py": "python", "js": "javascript", "ts": "typescript",
                "json": "json", "md": "markdown", "sh": "bash",
                "yaml": "yaml", "yml": "yaml"}.get(ext.lstrip("."), "")

        if ch["op"] == "Edit" and ch.get("old") is not None:
            lines.append("**変更前:**")
            lines.append(f"```{lang}")
            lines.append(ch["old"])
            if ch.get("old_truncated"):
                lines.append("... (省略)")
            lines.append("```")
            lines.append("")
            lines.append("**変更後:**")
            lines.append(f"```{lang}")
            lines.append(ch["new"])
            if ch.get("new_truncated"):
                lines.append("... (省略)")
            lines.append("```")
        elif ch["op"] == "Write":
            lines.append(f"```{lang}")
            lines.append(ch["new"])
            if ch.get("new_truncated"):
                pass  # truncation removed — full content always shown
            lines.append("```")

        lines.append("")

    return lines


def _render_error_section(timeline: list) -> list[str]:
    """エラーログセクションを生成する（エラーがある場合のみ）。"""
    errors = []
    event_order = 0
    for event in timeline:
        if event["role"] != "assistant":
            continue
        for tc in event.get("tools", []):
            event_order += 1
            if tc.get("is_error"):
                errors.append({
                    "order": event_order,
                    "ts": _fmt_ts(event["ts"]),
                    "name": tc["name"],
                    "input": tc["input"],
                    "result": (tc.get("result_text", "") or ""),
                })

    if not errors:
        return []

    lines = [
        f"## エラーログ（{len(errors)} 件）",
        "",
    ]
    for err in errors:
        n = err["order"]
        ts = err["ts"]
        name = err["name"]
        inp_str = _render_tool_input(name, err["input"])
        lines.append(f"### ⚠️ [{n}] {ts} | `{name}`")
        lines.append("")
        lines.append(f"**操作:** {inp_str}")
        lines.append("")
        if err["result"]:
            lines.append("**エラー出力:**")
            lines.append("```")
            lines.append(err["result"][:500])
            lines.append("```")
        lines.append("")

    return lines


# ────────────────────────────────────────────
# Markdown レンダリング — メイン
# ────────────────────────────────────────────

def render_session_md(
    session_id: str,
    date_str: str,
    jsonl_path: Path,
    timeline: list,
    continuation_info: dict | None = None,
) -> str:
    """タイムラインを Obsidian Markdown に変換する。

    continuation_info: 日付分割セッション用。キー:
        parent_session_id: 元の session_id
        date_part: 何日目か (1, 2, ...)
        prev_date: 前日の日付文字列 (or None)
        next_date: 翌日の日付文字列 (or None)
    """
    db_data = extract_db_data(timeline)
    meta = db_data["session_meta"]

    # フロントマター用: 編集ファイル一覧
    edited_files = list({f["file_path"] for f in db_data["files"]
                         if f["operation"] in ("Edit", "Write")})

    lines = []

    # ── フロントマター ──
    lines.extend([
        "---",
        "source: claude-code-session",
        f"session_id: {session_id}",
        f"date: {date_str}",
        f"jsonl_path: {jsonl_path}",
        f"thinking_blocks: {meta['thinking_blocks']}",
        f"error_count: {meta['error_count']}",
        f"bash_command_count: {meta['bash_command_count']}",
        f"tool_call_count: {meta['tool_call_count']}",
    ])
    if continuation_info and continuation_info.get("date_part", 1) > 1:
        lines.append(f"parent_session: {continuation_info['parent_session_id']}")
        lines.append(f"date_part: {continuation_info['date_part']}")
    if edited_files:
        files_yaml = "\n".join(f'  - "{f}"' for f in edited_files[:20])
        lines.append(f"edited_files:\n{files_yaml}")
    lines.append("---")
    lines.append("")

    # ── タイトル ──
    sid_short = (continuation_info or {}).get("parent_session_id", session_id)[:8]
    lines.append(f"# セッション {date_str} | {sid_short}")
    lines.append("")

    # ── 継続リンク（冒頭） ──
    if continuation_info:
        prev_d = continuation_info.get("prev_date")
        if prev_d:
            lines.append(f"> 前日からの継続: [[{prev_d}_{sid_short}]]")
            lines.append("")

    # ── 変更ファイルサマリ ──
    if edited_files:
        lines.append("## 変更ファイル")
        for f in edited_files:
            short = f.replace("/Users/kitak/QPI_Omni/", "").replace("/Users/kitak/", "~/")
            lines.append(f"- `{short}`")
        lines.append("")

    # ── セッションサマリ ──
    lines.extend(_render_session_summary_section(db_data))

    # ── 設計ノート（thinking 抜粋） ──
    lines.extend(_render_design_notes_section(timeline))

    # ── 再現コマンド一覧 ──
    lines.extend(_render_bash_section(db_data["bash_commands"]))

    # ── コード変更詳細 ──
    lines.extend(_render_code_changes_section(timeline))

    # ── エラーログ ──
    lines.extend(_render_error_section(timeline))

    # ── タイムライン（全会話） ──
    lines.append("---")
    lines.append("")
    lines.append("## タイムライン（全会話・ツール実行）")
    lines.append("")

    for event in timeline:
        ts_str = _fmt_ts(event["ts"])

        if event["role"] == "user":
            lines.append(f"### {ts_str} | ユーザー")
            lines.append("")
            lines.append(event["text"])
            lines.append("")

        elif event["role"] == "assistant":
            lines.append(f"### {ts_str} | AI")
            lines.append("")

            # Thinking ブロック
            for thinking in event.get("thinking", []):
                lines.append("> [!quote] AI Internal Reasoning")
                for tline in thinking.splitlines():
                    lines.append(f"> {tline}")
                lines.append(">")
                lines.append("")

            # テキスト返答
            for text in event.get("text", []):
                lines.append(text)
                lines.append("")

            # ツール実行
            tools = event.get("tools", [])
            if tools:
                lines.append("**ツール実行:**")
                for tc in tools:
                    name = tc["name"]
                    inp_str = _render_tool_input(name, tc["input"])
                    error_mark = " ⚠️" if tc.get("is_error") else ""
                    lines.append(f"- `{name}`{error_mark}: {inp_str}")
                    result = tc.get("result_text", "").strip()
                    if result:
                        if tc.get("is_error"):
                            for rline in result.splitlines()[:20]:
                                lines.append(f"  > {rline}")
                        elif len(result) > 0:
                            for rline in result.splitlines():
                                lines.append(f"  > {rline}")
                lines.append("")

    # ── 継続リンク（末尾） ──
    if continuation_info:
        next_d = continuation_info.get("next_date")
        if next_d:
            sid_short_tail = (continuation_info or {}).get("parent_session_id", session_id)[:8]
            lines.append("")
            lines.append(f"> 翌日に継続: [[{next_d}_{sid_short_tail}]]")
            lines.append("")

    return "\n".join(lines)


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────

def _process_single_date(
    session_id: str, date_str: str, jsonl_path: Path,
    timeline: list, mtime: float, dry_run: bool,
    continuation_info: dict | None = None,
) -> dict | None:
    """1つの日付分の .md を生成する内部ヘルパー。"""
    sid_for_file = session_id[:8]
    if continuation_info:
        parent_sid = continuation_info.get("parent_session_id", session_id)
        sid_for_file = parent_sid[:8]

    md_content = render_session_md(
        session_id, date_str, jsonl_path, timeline,
        continuation_info=continuation_info,
    )
    out_path = OBSIDIAN_SESSIONS_DIR / f"{date_str}_{sid_for_file}.md"

    if dry_run:
        print(f"  [dry-run] would write: {out_path}")
        db_data = extract_db_data(timeline)
        meta = db_data["session_meta"]
        print(f"    events={len(timeline)}, size={len(md_content)} chars,"
              f" thinking={meta['thinking_blocks']}, errors={meta['error_count']},"
              f" bash={meta['bash_command_count']}")
    else:
        OBSIDIAN_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md_content, encoding="utf-8")
        db_data = extract_db_data(timeline)
        meta = db_data["session_meta"]
        print(f"  → {out_path.name} ({len(md_content):,} chars, {len(timeline)} events,"
              f" thinking={meta['thinking_blocks']}, errors={meta['error_count']},"
              f" bash={meta['bash_command_count']})")

    return {"out_path": out_path, "timeline": timeline,
            "date_str": date_str, "mtime": mtime,
            "session_id_for_db": session_id}


def process_session(jsonl_path: Path, dry_run: bool) -> list[dict] | None:
    """1つのセッションを処理して .md ファイルを出力する。

    日付をまたぐセッションは日付ごとに分割して複数の .md を生成する。

    Returns:
        list[dict]  各日付分の結果 (成功時、単一日でも要素1のリスト)
        None  (スキップ時)
    """
    entries = load_entries(jsonl_path)
    if not entries:
        return None

    # アクティブセッションチェック
    last_dt = get_last_entry_time(entries)
    if last_dt:
        now_jst = datetime.now(JST)
        if (now_jst - last_dt).total_seconds() < ACTIVE_THRESHOLD_MINUTES * 60:
            print(f"  SKIP (active): {jsonl_path.name}")
            return None

    try:
        mtime = jsonl_path.stat().st_mtime
    except Exception:
        mtime = 0.0

    session_id = jsonl_path.stem
    timeline = parse_timeline(entries)

    if not timeline:
        print(f"  SKIP (empty timeline): {jsonl_path.name}")
        return None

    date_groups = group_timeline_by_date(timeline)
    sorted_dates = sorted(date_groups.keys())

    # ── 単一日セッション（大多数）: 既存と同じ処理 ──
    if len(sorted_dates) == 1:
        date_str = sorted_dates[0]
        result = _process_single_date(
            session_id, date_str, jsonl_path, timeline, mtime, dry_run)
        return [result] if result else None

    # ── 複数日セッション: 日付ごとに分割 ──
    print(f"  日付分割: {len(sorted_dates)} 日にまたがるセッション ({' → '.join(sorted_dates)})")
    results = []
    for i, date_str in enumerate(sorted_dates):
        part = i + 1
        day_timeline = date_groups[date_str]

        # 合成 session_id: 1日目は元のまま、2日目以降は __dYYYYMMDD を付加
        sid = session_id if part == 1 else f"{session_id}__d{date_str.replace('-', '')}"

        cont_info = {
            "parent_session_id": session_id,
            "date_part": part,
            "prev_date": sorted_dates[i - 1] if i > 0 else None,
            "next_date": sorted_dates[i + 1] if i < len(sorted_dates) - 1 else None,
        }

        result = _process_single_date(
            sid, date_str, jsonl_path, day_timeline, mtime, dry_run,
            continuation_info=cont_info,
        )
        if result:
            results.append(result)

    return results if results else None


def _collect_jsonl_targets(session_dirs: list[Path] | None) -> list[Path]:
    """複数ディレクトリから JSONL を収集。同一 session_id は mtime が新しい方を優先。"""
    dirs = session_dirs if session_dirs else DEFAULT_SESSION_DIRS
    by_id: dict[str, tuple[Path, float]] = {}

    for d in dirs:
        path = Path(d) if not isinstance(d, Path) else d
        if not path.exists():
            continue
        for p in path.glob("*.jsonl"):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            sid = p.stem
            if sid not in by_id or mtime > by_id[sid][1]:
                by_id[sid] = (p, mtime)

    return [p for p, _ in sorted(by_id.values(), key=lambda x: x[1])]


def run(args: argparse.Namespace) -> int:
    conn = session_db.open_db()
    run_id = session_db.start_processing_run(conn, "jsonl_to_obsidian")

    # 処理対象ファイルの収集
    if args.session:
        targets = [Path(args.session)]
    else:
        dirs = None
        if getattr(args, "session_dirs", None):
            dirs = [Path(d) for d in args.session_dirs]
        targets = _collect_jsonl_targets(dirs)

    processed = 0
    skipped = 0

    try:
        for jsonl_path in targets:
            try:
                mtime = jsonl_path.stat().st_mtime
            except FileNotFoundError:
                continue

            session_id = jsonl_path.stem

            # 処理済みチェック（DB）
            if not args.all and not args.session:
                db_mtime = session_db.get_session_mtime(conn, session_id)
                if db_mtime is not None and mtime <= db_mtime:
                    skipped += 1
                    continue

            print(f"Processing: {jsonl_path.name}")
            try:
                result = process_session(jsonl_path, args.dry_run)
            except Exception as e:
                print(f"  ERROR: {jsonl_path.name} の処理で例外: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc()
                skipped += 1
                continue

            if result is not None:
                processed += 1
                if not args.dry_run:
                    # 再処理時: 古い日付分割部分を先に削除
                    session_db.delete_session_parts(conn, session_id)

                    for part_result in result:
                        db_sid = part_result.get("session_id_for_db", session_id)
                        write_session_to_db(
                            conn,
                            session_id=db_sid,
                            date_str=part_result["date_str"],
                            jsonl_path=jsonl_path,
                            md_path=part_result["out_path"],
                            timeline=part_result["timeline"],
                            mtime=part_result["mtime"],
                        )
                    # 共有フォルダ由来の JSONL は処理成功後に削除
                    if (not getattr(args, "no_delete_shared", False)
                        and str(jsonl_path.resolve()).startswith(
                            str(SHARED_SESSION_DIR.resolve())
                        )):
                        try:
                            jsonl_path.unlink()
                            print(f"  → 削除: {jsonl_path.name}")
                        except OSError as e:
                            print(f"  → 削除失敗: {e}")
            else:
                skipped += 1
                # empty-timeline / active セッションを DB に記録して次回から黙ってスキップ
                if not args.dry_run:
                    now_iso = datetime.now(JST).isoformat()
                    date_str = get_session_date(load_entries(jsonl_path))
                    session_db.upsert_session(conn, {
                        "session_id": session_id,
                        "date": date_str,
                        "jsonl_path": str(jsonl_path),
                        "md_path": None,
                        "jsonl_mtime": mtime,
                        "processed_at": now_iso,
                        "thinking_blocks": 0,
                        "user_message_count": 0,
                        "assistant_message_count": 0,
                        "tool_call_count": 0,
                        "error_count": 0,
                        "bash_command_count": 0,
                        "is_active": 0,
                        "needs_reprocess": 0,
                        "reprocess_reason": "empty_timeline",
                        "first_user_message": None,
                        "design_notes": None,
                        "session_summary": None,
                    })
                    conn.commit()

        session_db.finish_processing_run(
            conn, run_id, processed=processed, skipped=skipped, success=True
        )
    except Exception as e:
        session_db.finish_processing_run(
            conn, run_id, processed=processed, skipped=skipped,
            success=False, notes=str(e)
        )
        conn.close()
        raise
    finally:
        conn.close()

    print(f"\njsonl_to_obsidian: processed={processed}, skipped={skipped},"
          f" dry_run={args.dry_run}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="JSONL → Obsidian .md 変換（thinking ブロック・設計ノート・再現コマンド含む）"
    )
    parser.add_argument("--session", help="特定の JSONL ファイルのみ処理")
    parser.add_argument("--dry-run", action="store_true",
                        help="ファイル書き出しなしで確認")
    parser.add_argument("--all", action="store_true",
                        help="処理済み含め全セッションを再変換")
    parser.add_argument(
        "--session-dir",
        action="append",
        dest="session_dirs",
        help="JSONL ディレクトリ（複数指定可。未指定時はローカル+共有フォルダ）",
    )
    parser.add_argument(
        "--no-delete-shared",
        action="store_true",
        help="共有フォルダの JSONL を処理後も削除しない（デフォルトは削除する）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args))
