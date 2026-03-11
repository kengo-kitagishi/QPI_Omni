#!/usr/bin/env python3
"""session_db.py

Claude Code セッション・図イベントの SQLite キャッシュ/インデックス層。

- sessions        : セッションメタデータ（metrics, state flags, 設計ノート）
- session_files   : セッション内で編集されたファイル（operation, old/new preview）
- tool_calls      : 全ツール呼び出し（name, input, result, error）
- bash_commands   : Bash コマンド（command, description, output, error）
- figure_events   : figure-hub inbox の図イベント（enrichment 状態含む）
- processing_runs : 処理実行履歴（processed/skipped/error カウント）

DB パス: .figure_history/sessions.db
初回起動時に既存 JSON state ファイルから自動移行。

CLI:
  python3 scripts/session_db.py --stats      統計を表示
  python3 scripts/session_db.py --sessions   セッション一覧（直近20件）
  python3 scripts/session_db.py --week 2026-W10  週別サマリ
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator, Optional

# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / ".figure_history" / "sessions.db"

LEGACY_JSONL_STATE = REPO_ROOT / ".figure_history" / "jsonl_to_obsidian_state.json"
LEGACY_FIGURE_STATE = REPO_ROOT / ".figure_history" / "figure_inbox_obsidian_state.json"
OBSIDIAN_SESSIONS_DIR = Path("/Users/kitak/Documents/Obsidian Vault/00_Inbox/claude_sessions")

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT    PRIMARY KEY,
    date                    TEXT    NOT NULL,
    jsonl_path              TEXT    NOT NULL,
    md_path                 TEXT,
    jsonl_mtime             REAL,
    processed_at            TEXT,

    thinking_blocks         INTEGER DEFAULT 0,
    user_message_count      INTEGER DEFAULT 0,
    assistant_message_count INTEGER DEFAULT 0,
    tool_call_count         INTEGER DEFAULT 0,
    error_count             INTEGER DEFAULT 0,
    bash_command_count      INTEGER DEFAULT 0,

    is_active               INTEGER DEFAULT 0,
    needs_reprocess         INTEGER DEFAULT 0,
    reprocess_reason        TEXT,

    first_user_message      TEXT,
    design_notes            TEXT,
    session_summary         TEXT
);

CREATE TABLE IF NOT EXISTS session_files (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    file_path   TEXT    NOT NULL,
    operation   TEXT    NOT NULL,
    old_preview TEXT,
    new_preview TEXT,
    event_order INTEGER,
    timestamp   TEXT
);
CREATE INDEX IF NOT EXISTS idx_sf_path    ON session_files(file_path);
CREATE INDEX IF NOT EXISTS idx_sf_session ON session_files(session_id);
CREATE INDEX IF NOT EXISTS idx_sf_op      ON session_files(operation, file_path);

CREATE TABLE IF NOT EXISTS tool_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_order     INTEGER,
    timestamp       TEXT,
    tool_name       TEXT    NOT NULL,
    input_json      TEXT,
    result_preview  TEXT,
    is_error        INTEGER DEFAULT 0,
    error_text      TEXT
);
CREATE INDEX IF NOT EXISTS idx_tc_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tc_name    ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tc_error   ON tool_calls(is_error) WHERE is_error = 1;

CREATE TABLE IF NOT EXISTS bash_commands (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT    NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    event_order     INTEGER,
    timestamp       TEXT,
    command         TEXT    NOT NULL,
    description     TEXT,
    output_preview  TEXT,
    is_error        INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_bash_session ON bash_commands(session_id);
CREATE INDEX IF NOT EXISTS idx_bash_error   ON bash_commands(is_error) WHERE is_error = 1;

CREATE TABLE IF NOT EXISTS figure_events (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    unique_key          TEXT    UNIQUE NOT NULL,
    run_id              TEXT    NOT NULL,
    figure_index        INTEGER DEFAULT 1,
    script              TEXT,
    date                TEXT    NOT NULL,
    created_at_utc      TEXT,
    git_commit          TEXT,
    git_dirty           INTEGER DEFAULT 0,
    description         TEXT,
    json_path           TEXT    NOT NULL,
    md_path             TEXT,
    image_obsidian_name TEXT,
    processed_at        TEXT,
    enriched            INTEGER DEFAULT 0,
    enriched_at         TEXT
);
CREATE INDEX IF NOT EXISTS idx_fe_date   ON figure_events(date);
CREATE INDEX IF NOT EXISTS idx_fe_script ON figure_events(script);

CREATE TABLE IF NOT EXISTS processing_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    script          TEXT    NOT NULL,
    started_at      TEXT    NOT NULL,
    finished_at     TEXT,
    success         INTEGER DEFAULT 1,
    processed_count INTEGER DEFAULT 0,
    skipped_count   INTEGER DEFAULT 0,
    error_count     INTEGER DEFAULT 0,
    notes           TEXT
);
"""


# ────────────────────────────────────────────
# 接続管理
# ────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(JST).isoformat()


def open_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """DB を開いてスキーマ初期化を行い、接続を返す。呼び出し元が close() すること。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


@contextmanager
def get_conn(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """コンテキストマネージャ版。commit は呼び出し元の責任（ループ内でのバッチ処理向け）。"""
    conn = open_db(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_SQL)
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))
        conn.commit()
        _migrate_from_json(conn)


# ────────────────────────────────────────────
# JSON state からの移行（初回のみ）
# ────────────────────────────────────────────

def _migrate_from_json(conn: sqlite3.Connection) -> None:
    """既存 JSON state ファイルから最小限のデータを移行する（冪等）。"""
    migrated_sessions = 0
    migrated_figures = 0

    if LEGACY_JSONL_STATE.exists():
        try:
            state = json.loads(LEGACY_JSONL_STATE.read_text(encoding="utf-8"))
            for jsonl_path_str, mtime in state.get("sessions", {}).items():
                jsonl_path = Path(jsonl_path_str)
                session_id = jsonl_path.stem
                md_path = _find_md_for_session(session_id)
                date_str = _date_from_md(md_path) if md_path else ""
                if not date_str:
                    date_str = _date_from_path(jsonl_path_str)
                conn.execute(
                    """INSERT OR IGNORE INTO sessions
                       (session_id, date, jsonl_path, md_path, jsonl_mtime, processed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session_id, date_str, jsonl_path_str,
                     str(md_path) if md_path else None,
                     float(mtime), _now_iso())
                )
                migrated_sessions += 1
        except Exception as e:
            print(f"  migration warning (jsonl state): {e}")

    if LEGACY_FIGURE_STATE.exists():
        try:
            state = json.loads(LEGACY_FIGURE_STATE.read_text(encoding="utf-8"))
            for unique_key in state.get("processed_run_ids", []):
                parts = unique_key.rsplit("_f", 1)
                run_id = parts[0] if len(parts) == 2 else unique_key
                fig_idx_str = parts[1] if len(parts) == 2 else "1"
                fig_idx = int(fig_idx_str) if fig_idx_str.isdigit() else 1
                conn.execute(
                    """INSERT OR IGNORE INTO figure_events
                       (unique_key, run_id, figure_index, date, json_path, processed_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (unique_key, run_id, fig_idx, "",
                     f"(migrated:{unique_key})", _now_iso())
                )
                migrated_figures += 1
        except Exception as e:
            print(f"  migration warning (figure state): {e}")

    conn.execute(
        """INSERT INTO processing_runs (script, started_at, finished_at, notes)
           VALUES ('migration', ?, ?, ?)""",
        (_now_iso(), _now_iso(),
         f"JSON state migration: {migrated_sessions} sessions, {migrated_figures} figures")
    )
    conn.commit()

    if migrated_sessions > 0 or migrated_figures > 0:
        print(f"  DB初期化: {migrated_sessions} セッション, {migrated_figures} 図イベントを JSON state から移行")


def _find_md_for_session(session_id: str) -> Optional[Path]:
    sid8 = session_id[:8]
    if OBSIDIAN_SESSIONS_DIR.exists():
        for md in OBSIDIAN_SESSIONS_DIR.glob(f"*_{sid8}.md"):
            return md
    return None


def _date_from_md(md_path: Path) -> str:
    try:
        text = md_path.read_text(encoding="utf-8")
        for line in text.splitlines()[1:20]:
            if line.startswith("date:"):
                return line.split(":", 1)[1].strip().strip('"\'')[:10]
    except Exception:
        pass
    return ""


def _date_from_path(path_str: str) -> str:
    import re
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path_str)
    return m.group(1) if m else ""


# ────────────────────────────────────────────
# Upsert
# ────────────────────────────────────────────

def upsert_session(conn: sqlite3.Connection, data: dict) -> None:
    conn.execute(
        """INSERT INTO sessions (
               session_id, date, jsonl_path, md_path, jsonl_mtime, processed_at,
               thinking_blocks, user_message_count, assistant_message_count,
               tool_call_count, error_count, bash_command_count,
               is_active, needs_reprocess, reprocess_reason,
               first_user_message, design_notes, session_summary
           ) VALUES (
               :session_id, :date, :jsonl_path, :md_path, :jsonl_mtime, :processed_at,
               :thinking_blocks, :user_message_count, :assistant_message_count,
               :tool_call_count, :error_count, :bash_command_count,
               :is_active, :needs_reprocess, :reprocess_reason,
               :first_user_message, :design_notes, :session_summary
           )
           ON CONFLICT(session_id) DO UPDATE SET
               date = excluded.date,
               md_path = excluded.md_path,
               jsonl_mtime = excluded.jsonl_mtime,
               processed_at = excluded.processed_at,
               thinking_blocks = excluded.thinking_blocks,
               user_message_count = excluded.user_message_count,
               assistant_message_count = excluded.assistant_message_count,
               tool_call_count = excluded.tool_call_count,
               error_count = excluded.error_count,
               bash_command_count = excluded.bash_command_count,
               is_active = excluded.is_active,
               needs_reprocess = excluded.needs_reprocess,
               reprocess_reason = excluded.reprocess_reason,
               first_user_message = excluded.first_user_message,
               design_notes = excluded.design_notes,
               session_summary = excluded.session_summary""",
        {
            "session_id": data["session_id"],
            "date": data.get("date", ""),
            "jsonl_path": data.get("jsonl_path", ""),
            "md_path": data.get("md_path"),
            "jsonl_mtime": data.get("jsonl_mtime"),
            "processed_at": data.get("processed_at", _now_iso()),
            "thinking_blocks": data.get("thinking_blocks", 0),
            "user_message_count": data.get("user_message_count", 0),
            "assistant_message_count": data.get("assistant_message_count", 0),
            "tool_call_count": data.get("tool_call_count", 0),
            "error_count": data.get("error_count", 0),
            "bash_command_count": data.get("bash_command_count", 0),
            "is_active": int(data.get("is_active", False)),
            "needs_reprocess": int(data.get("needs_reprocess", False)),
            "reprocess_reason": data.get("reprocess_reason"),
            "first_user_message": data.get("first_user_message"),
            "design_notes": data.get("design_notes"),
            "session_summary": data.get("session_summary"),
        }
    )


def upsert_session_files(conn: sqlite3.Connection, session_id: str,
                         files: list[dict]) -> None:
    conn.execute("DELETE FROM session_files WHERE session_id = ?", (session_id,))
    conn.executemany(
        """INSERT INTO session_files
           (session_id, file_path, operation, old_preview, new_preview, event_order, timestamp)
           VALUES (:session_id, :file_path, :operation, :old_preview, :new_preview,
                   :event_order, :timestamp)""",
        [{"session_id": session_id,
          "file_path": f.get("file_path", ""),
          "operation": f.get("operation", ""),
          "old_preview": f.get("old_preview"),
          "new_preview": f.get("new_preview"),
          "event_order": f.get("event_order"),
          "timestamp": f.get("timestamp")} for f in files]
    )


def upsert_tool_calls(conn: sqlite3.Connection, session_id: str,
                      tool_calls: list[dict]) -> None:
    conn.execute("DELETE FROM tool_calls WHERE session_id = ?", (session_id,))
    conn.executemany(
        """INSERT INTO tool_calls
           (session_id, event_order, timestamp, tool_name, input_json,
            result_preview, is_error, error_text)
           VALUES (:session_id, :event_order, :timestamp, :tool_name, :input_json,
                   :result_preview, :is_error, :error_text)""",
        [{"session_id": session_id,
          "event_order": tc.get("event_order"),
          "timestamp": tc.get("timestamp"),
          "tool_name": tc.get("tool_name", ""),
          "input_json": tc.get("input_json"),
          "result_preview": tc.get("result_preview"),
          "is_error": int(tc.get("is_error", False)),
          "error_text": tc.get("error_text")} for tc in tool_calls]
    )


def upsert_bash_commands(conn: sqlite3.Connection, session_id: str,
                         bash_cmds: list[dict]) -> None:
    conn.execute("DELETE FROM bash_commands WHERE session_id = ?", (session_id,))
    conn.executemany(
        """INSERT INTO bash_commands
           (session_id, event_order, timestamp, command, description, output_preview, is_error)
           VALUES (:session_id, :event_order, :timestamp, :command, :description,
                   :output_preview, :is_error)""",
        [{"session_id": session_id,
          "event_order": bc.get("event_order"),
          "timestamp": bc.get("timestamp"),
          "command": bc.get("command", ""),
          "description": bc.get("description"),
          "output_preview": bc.get("output_preview"),
          "is_error": int(bc.get("is_error", False))} for bc in bash_cmds]
    )


def upsert_figure_event(conn: sqlite3.Connection, data: dict) -> None:
    conn.execute(
        """INSERT INTO figure_events (
               unique_key, run_id, figure_index, script, date, created_at_utc,
               git_commit, git_dirty, description, json_path, md_path,
               image_obsidian_name, processed_at
           ) VALUES (
               :unique_key, :run_id, :figure_index, :script, :date, :created_at_utc,
               :git_commit, :git_dirty, :description, :json_path, :md_path,
               :image_obsidian_name, :processed_at
           )
           ON CONFLICT(unique_key) DO UPDATE SET
               script = excluded.script,
               date = excluded.date,
               git_commit = excluded.git_commit,
               description = excluded.description,
               json_path = excluded.json_path,
               md_path = excluded.md_path,
               image_obsidian_name = excluded.image_obsidian_name,
               processed_at = excluded.processed_at""",
        {
            "unique_key": data["unique_key"],
            "run_id": data.get("run_id", ""),
            "figure_index": data.get("figure_index", 1),
            "script": data.get("script"),
            "date": data.get("date", ""),
            "created_at_utc": data.get("created_at_utc"),
            "git_commit": data.get("git_commit"),
            "git_dirty": int(data.get("git_dirty", False)),
            "description": data.get("description"),
            "json_path": data.get("json_path", ""),
            "md_path": data.get("md_path"),
            "image_obsidian_name": data.get("image_obsidian_name"),
            "processed_at": data.get("processed_at", _now_iso()),
        }
    )


def mark_figure_enriched(conn: sqlite3.Connection, unique_key: str) -> None:
    conn.execute(
        "UPDATE figure_events SET enriched = 1, enriched_at = ? WHERE unique_key = ?",
        (_now_iso(), unique_key)
    )


# ────────────────────────────────────────────
# State チェック
# ────────────────────────────────────────────

def get_session_mtime(conn: sqlite3.Connection, session_id: str) -> Optional[float]:
    """DB に記録された最後の処理時 mtime を返す。None = 未処理。"""
    row = conn.execute(
        "SELECT jsonl_mtime FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row["jsonl_mtime"] if row else None


def is_figure_processed(conn: sqlite3.Connection, unique_key: str) -> bool:
    """unique_key が DB に存在し、かつ移行レコードでない（実処理済み）かを返す。"""
    row = conn.execute(
        "SELECT id FROM figure_events WHERE unique_key = ? AND json_path NOT LIKE '(migrated:%'",
        (unique_key,)
    ).fetchone()
    return bool(row)


def is_figure_enriched(conn: sqlite3.Connection, unique_key: str) -> bool:
    row = conn.execute(
        "SELECT enriched FROM figure_events WHERE unique_key = ?", (unique_key,)
    ).fetchone()
    return bool(row and row["enriched"])


# ────────────────────────────────────────────
# クエリ（週次レポート用）
# ────────────────────────────────────────────

def query_week_sessions(conn: sqlite3.Connection, monday: str, sunday: str) -> list[dict]:
    """週のセッション一覧をメタデータ付きで返す。"""
    rows = conn.execute(
        """SELECT s.*,
               (SELECT GROUP_CONCAT(sf.file_path, '|||')
                FROM (SELECT DISTINCT sf2.file_path
                      FROM session_files sf2
                      WHERE sf2.session_id = s.session_id
                        AND sf2.operation IN ('Edit', 'Write')) sf) AS edited_files_raw
           FROM sessions s
           WHERE s.date BETWEEN ? AND ?
             AND s.is_active = 0
           ORDER BY s.date, s.session_id""",
        (monday, sunday)
    ).fetchall()
    return [dict(r) for r in rows]


def query_session_bash_commands(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM bash_commands WHERE session_id = ? ORDER BY event_order",
        (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def query_session_errors(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    rows = conn.execute(
        """SELECT tc.event_order, tc.timestamp, tc.tool_name,
                  tc.input_json, tc.error_text, tc.result_preview
           FROM tool_calls tc
           WHERE tc.session_id = ? AND tc.is_error = 1
           ORDER BY tc.event_order""",
        (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def query_session_code_changes(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Edit/Write 操作のファイルと内容を返す（コード変更詳細用）。"""
    rows = conn.execute(
        """SELECT sf.file_path, sf.operation, sf.old_preview, sf.new_preview,
                  sf.event_order, sf.timestamp
           FROM session_files sf
           WHERE sf.session_id = ? AND sf.operation IN ('Edit', 'Write')
           ORDER BY sf.event_order""",
        (session_id,)
    ).fetchall()
    return [dict(r) for r in rows]


def query_week_figures(conn: sqlite3.Connection, monday: str, sunday: str) -> list[dict]:
    rows = conn.execute(
        """SELECT * FROM figure_events
           WHERE date BETWEEN ? AND ?
             AND json_path NOT LIKE '(migrated:%'
           ORDER BY date, script""",
        (monday, sunday)
    ).fetchall()
    return [dict(r) for r in rows]


def query_week_file_stats(conn: sqlite3.Connection, monday: str, sunday: str) -> list[dict]:
    """週のファイル変更統計を返す。"""
    rows = conn.execute(
        """SELECT sf.file_path,
               COUNT(*) as change_count,
               GROUP_CONCAT(DISTINCT s.date) as dates,
               COUNT(DISTINCT s.session_id) as session_count
           FROM session_files sf
           JOIN sessions s ON sf.session_id = s.session_id
           WHERE s.date BETWEEN ? AND ?
             AND sf.operation IN ('Edit', 'Write')
           GROUP BY sf.file_path
           ORDER BY change_count DESC""",
        (monday, sunday)
    ).fetchall()
    return [dict(r) for r in rows]


def query_week_stats(conn: sqlite3.Connection, monday: str, sunday: str) -> dict:
    """週の集計統計を返す。"""
    row = conn.execute(
        """SELECT
               COUNT(*) as session_count,
               COALESCE(SUM(thinking_blocks), 0) as total_thinking,
               COALESCE(SUM(tool_call_count), 0) as total_tools,
               COALESCE(SUM(bash_command_count), 0) as total_bash,
               COALESCE(SUM(error_count), 0) as total_errors,
               COALESCE(SUM(user_message_count), 0) as total_user_msgs
           FROM sessions
           WHERE date BETWEEN ? AND ? AND is_active = 0""",
        (monday, sunday)
    ).fetchone()

    fig_row = conn.execute(
        """SELECT COUNT(*) as n FROM figure_events
           WHERE date BETWEEN ? AND ? AND json_path NOT LIKE '(migrated:%'""",
        (monday, sunday)
    ).fetchone()

    file_row = conn.execute(
        """SELECT COUNT(DISTINCT sf.file_path) as n
           FROM session_files sf
           JOIN sessions s ON sf.session_id = s.session_id
           WHERE s.date BETWEEN ? AND ? AND sf.operation IN ('Edit', 'Write')""",
        (monday, sunday)
    ).fetchone()

    d = dict(row) if row else {}
    d["figure_count"] = fig_row["n"] if fig_row else 0
    d["unique_file_count"] = file_row["n"] if file_row else 0
    return d


# ────────────────────────────────────────────
# 処理実行ログ
# ────────────────────────────────────────────

def start_processing_run(conn: sqlite3.Connection, script: str) -> int:
    cur = conn.execute(
        "INSERT INTO processing_runs (script, started_at) VALUES (?, ?)",
        (script, _now_iso())
    )
    conn.commit()
    return cur.lastrowid


def finish_processing_run(conn: sqlite3.Connection, run_id: int,
                          processed: int = 0, skipped: int = 0, errors: int = 0,
                          success: bool = True, notes: str = "") -> None:
    conn.execute(
        """UPDATE processing_runs SET
               finished_at = ?, success = ?, processed_count = ?,
               skipped_count = ?, error_count = ?, notes = ?
           WHERE id = ?""",
        (_now_iso(), int(success), processed, skipped, errors, notes, run_id)
    )
    conn.commit()


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

def _print_stats(conn: sqlite3.Connection) -> None:
    print("=== sessions.db 統計 ===")
    for label, sql in [
        ("セッション総数", "SELECT COUNT(*) FROM sessions"),
        ("  うち再処理待ち", "SELECT COUNT(*) FROM sessions WHERE needs_reprocess = 1"),
        ("ツール呼び出し総数", "SELECT COUNT(*) FROM tool_calls"),
        ("Bash コマンド総数", "SELECT COUNT(*) FROM bash_commands"),
        ("  うちエラー", "SELECT COUNT(*) FROM bash_commands WHERE is_error = 1"),
        ("図イベント総数", "SELECT COUNT(*) FROM figure_events WHERE json_path NOT LIKE '(migrated:%'"),
        ("  うちenrichment済み", "SELECT COUNT(*) FROM figure_events WHERE enriched = 1"),
        ("処理実行ログ件数", "SELECT COUNT(*) FROM processing_runs"),
    ]:
        n = conn.execute(sql).fetchone()[0]
        print(f"  {label}: {n}")

    print("\n--- 直近 5 件の processing_runs ---")
    rows = conn.execute(
        "SELECT * FROM processing_runs ORDER BY id DESC LIMIT 5"
    ).fetchall()
    for r in rows:
        status = "✅" if r["success"] else "❌"
        fin = (r["finished_at"] or "")[:16]
        print(f"  [{status}] {r['script']:<30} @ {r['started_at'][:16]}"
              f"  processed={r['processed_count']} skipped={r['skipped_count']}"
              f" errors={r['error_count']}")


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(description="sessions.db 管理ツール")
    parser.add_argument("--stats", action="store_true", help="統計を表示")
    parser.add_argument("--sessions", action="store_true", help="セッション一覧（直近20件）")
    parser.add_argument("--week", help="週別サマリ (例: 2026-W10)")
    parser.add_argument("--db", default=str(DB_PATH), help="DB パス")
    args = parser.parse_args()

    db_path = Path(args.db)
    with get_conn(db_path) as conn:
        if args.stats or not (args.sessions or args.week):
            _print_stats(conn)

        if args.sessions:
            rows = conn.execute(
                """SELECT session_id, date, thinking_blocks, error_count,
                          bash_command_count, tool_call_count, first_user_message
                   FROM sessions ORDER BY date DESC LIMIT 20"""
            ).fetchall()
            print("\n--- セッション一覧（直近 20 件）---")
            for r in rows:
                msg = (r["first_user_message"] or "")[:60]
                print(f"  {r['date']} | {r['session_id'][:8]}"
                      f" | thinking={r['thinking_blocks']}"
                      f" errors={r['error_count']}"
                      f" bash={r['bash_command_count']}"
                      f" | {msg}")

        if args.week:
            m = re.match(r"(\d{4})-W(\d{2})", args.week)
            if m:
                from datetime import date
                year, week = int(m.group(1)), int(m.group(2))
                monday = date.fromisocalendar(year, week, 1)
                sunday = monday + timedelta(days=6)
                stats = query_week_stats(conn, monday.isoformat(), sunday.isoformat())
                print(f"\n--- {args.week} ({monday} – {sunday}) ---")
                for k, v in stats.items():
                    print(f"  {k}: {v}")
