#!/usr/bin/env python3
"""weekly_report_hub.py

週次レポート用の統一タイムライン索引を生成する。

1. jsonl_to_obsidian.py と figure_inbox_to_obsidian.py を実行（前処理）
2. SQLite DB から対象週のセッション・図イベントを取得
3. notion_sync ディレクトリから Notion メモを読む
4. figure-hub library をスキャン
5. 全ソースを統合した索引ファイルを Obsidian に出力

出力先:
  ~/Documents/Obsidian Vault/00_Inbox/weekly_index_YYYY-Www.md

週次索引の内容:
  - 週サマリ（sessions, ファイル, bash, エラー, thinking, 図）
  - ⚠️ エラーサマリ（エラーのあったセッション）
  - 📅 日別・セッション別サマリ
      - セッションごとの依頼内容・変更ファイル・再現コマンド
      - 図ごとの生成コード・説明
      - Notion メモの内容抜粋（背景・目的・主要手順）
  - 変更ファイル統計（全セッション統合）
  - 読むべきファイル一覧（Read tool 用）

使い方:
  python3 scripts/weekly_report_hub.py              # 今週の索引を生成
  python3 scripts/weekly_report_hub.py --week 2026-W10  # 指定週
  python3 scripts/weekly_report_hub.py --no-preprocess  # 前処理スキップ
  python3 scripts/weekly_report_hub.py --dry-run        # 確認のみ
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_db

# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

OBSIDIAN_ROOT = Path("/Users/kitak/Documents/Obsidian Vault")
OBSIDIAN_SESSIONS_DIR = OBSIDIAN_ROOT / "00_Inbox/claude_sessions"
OBSIDIAN_FIGURE_INBOX_DIR = OBSIDIAN_ROOT / "00_Inbox/figure_inbox"
OBSIDIAN_NOTION_SYNC_DIR = OBSIDIAN_ROOT / "00_Inbox/notion_sync/api"
OBSIDIAN_WEEKLY_REPORTS_DIR = OBSIDIAN_ROOT / "04_WeeklyReports"

FIGURE_HUB_LIBRARY = Path.home() / "Desktop/figure-hub/library"

JST = timezone(timedelta(hours=9))

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


# ────────────────────────────────────────────
# 週の計算
# ────────────────────────────────────────────

def get_week_range(week_label: str | None = None) -> tuple[date, date, str]:
    if week_label:
        m = re.match(r"(\d{4})-W(\d{2})", week_label)
        if not m:
            raise ValueError(f"week_label の形式が不正です: {week_label} (例: 2026-W10)")
        year, week = int(m.group(1)), int(m.group(2))
        monday = date.fromisocalendar(year, week, 1)
    else:
        today = date.today()
        monday = today - timedelta(days=today.weekday())

    sunday = monday + timedelta(days=6)
    label = monday.strftime("%Y-W%V")
    return monday, sunday, label


# ────────────────────────────────────────────
# フロントマター読み取り（フォールバック用）
# ────────────────────────────────────────────

def read_frontmatter(md_path: Path) -> dict:
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    fm_text = text[3:end].strip()
    result = {}
    for line in fm_text.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip()] = v.strip().strip('"').strip("'")
    return result


# ────────────────────────────────────────────
# Notion メモのパース
# ────────────────────────────────────────────

def extract_notion_summary(md_path: Path) -> dict:
    """notion_sync .md から要約情報を抽出する（WORKLOG_SPEC / 思考メモ 対応）。

    返却:
        title       : ページタイトル
        notion_url  : Notion URL
        purpose     : 目的（## 目的 or **目的** or 冒頭）
        background  : 背景（## 背景 or ## 何が起きたか）
        steps       : 実施手順のステップ名リスト（最大5件）
        key_code    : コードブロック内の主要行（最大5行）
        equations   : LaTeX/数式が含まれる行
        sections    : 全セクション見出しリスト
        raw_text    : 本文全体（検索用）
    """
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return {}

    # フロントマター除去と取得
    fm = read_frontmatter(md_path)
    title = fm.get("notion_title", md_path.stem[:80])
    notion_url = fm.get("notion_url", "")

    # フロントマターを除いた本文
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            body = text[end + 4:].strip()

    # 目的（抜粋用・短い場合のみ）
    purpose = ""
    m = re.search(r"(?:^|\n)#{1,3}\s*目的\s*\n+(.*?)(?=\n#|\Z)", body, re.DOTALL)
    if m:
        purpose = m.group(1).strip()
    if not purpose:
        m = re.search(r"\*\*目的\*\*:?\s*(.+)", body)
        if m:
            purpose = m.group(1).strip()

    # 背景（抜粋用）
    background = ""
    for heading in ("背景", "何が起きたか", "経緯", "概要"):
        m = re.search(
            r"(?:^|\n)#{1,3}\s*" + heading + r"\s*\n+(.*?)(?=\n#|\Z)",
            body, re.DOTALL
        )
        if m:
            background = m.group(1).strip()
            break

    # ステップ名一覧（WORKLOG_SPEC: "#### Step N: ..." or "## Step N: ...")
    steps = re.findall(r"#{2,4}\s*Step\s*\d+[:\s]+(.+)", body)
    if not steps:
        # 箇条書き形式: "- 1. XXX"
        steps = re.findall(r"[-\d]+\.\s+(.+)", body)[:5]

    # コードブロック内の主要行
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", body, re.DOTALL)
    key_code_lines = []
    for cb in code_blocks[:3]:
        for line in cb.strip().splitlines():
            line = line.strip()
            # コメント以外で意味のある行
            if line and not line.startswith("#") and len(line) > 10:
                key_code_lines.append(line[:120])
                if len(key_code_lines) >= 5:
                    break
        if len(key_code_lines) >= 5:
            break

    # 数式・LaTeX を含む行
    equations = []
    for line in body.splitlines():
        if re.search(r"\\frac|\\sum|\\int|\\partial|\\lambda|\$[^$]+\$|≈|∑|∫|∂", line):
            equations.append(line.strip()[:120])

    # セクション見出し
    sections = re.findall(r"^#{1,4}\s+(.+)", body, re.MULTILINE)

    return {
        "title": title,
        "notion_url": notion_url,
        "purpose": purpose,
        "background": background,
        "steps": steps[:5],
        "key_code": key_code_lines,
        "equations": equations[:3],
        "sections": sections[:10],
        "raw_text": body,
        "full_body": body,
    }


# ────────────────────────────────────────────
# ソーススキャン（DB + ファイル）
# ────────────────────────────────────────────

def scan_notion_sync(monday: date, sunday: date) -> list[dict]:
    """notion_sync ディレクトリから対象週の .md ファイルをスキャンし、内容を抽出する。"""
    results = []
    if not OBSIDIAN_NOTION_SYNC_DIR.exists():
        return results

    for day_offset in range(7):
        day = monday + timedelta(days=day_offset)
        day_dir = OBSIDIAN_NOTION_SYNC_DIR / day.isoformat()
        if not day_dir.exists():
            continue

        for md_path in sorted(day_dir.glob("*.md")):
            summary = extract_notion_summary(md_path)
            results.append({
                "date": day.isoformat(),
                "source": "notion-sync",
                "title": summary.get("title", md_path.stem[:60]),
                "notion_url": summary.get("notion_url", ""),
                "purpose": summary.get("purpose", ""),
                "background": summary.get("background", ""),
                "steps": summary.get("steps", []),
                "key_code": summary.get("key_code", []),
                "equations": summary.get("equations", []),
                "sections": summary.get("sections", []),
                "full_body": summary.get("full_body", ""),
                "path": str(md_path),
            })

    return results


def scan_figure_hub_library(monday: date, sunday: date) -> list[dict]:
    """figure-hub library から対象週に登録されたバージョンをスキャン。"""
    results = []
    if not FIGURE_HUB_LIBRARY.exists():
        return results

    for meta_path in sorted(FIGURE_HUB_LIBRARY.glob("*/meta.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        fig_id = meta.get("id", "")
        for v in meta.get("versions", []):
            added_at = v.get("added_at", "")
            if not added_at:
                continue
            try:
                added_date = date.fromisoformat(added_at[:10])
            except ValueError:
                continue

            if monday <= added_date <= sunday:
                results.append({
                    "date": added_at[:10],
                    "source": "figure-hub-library",
                    "fig_id": fig_id,
                    "version": v.get("version", ""),
                    "note": v.get("note", "")[:80],
                    "path": str(meta_path),
                })

    return results


# ────────────────────────────────────────────
# 前処理スクリプト実行
# ────────────────────────────────────────────

def run_preprocess(dry_run: bool) -> None:
    scripts = [
        SCRIPTS_DIR / "jsonl_to_obsidian.py",
        SCRIPTS_DIR / "figure_inbox_to_obsidian.py",
    ]
    for script in scripts:
        cmd = [sys.executable, str(script)]
        if dry_run:
            cmd.append("--dry-run")
        print(f"\n--- 実行: {script.name} ---")
        try:
            subprocess.run(cmd, check=True, capture_output=False)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: {script.name} が失敗しました (exit code {e.returncode})")


# ────────────────────────────────────────────
# Cross-reference enrichment
# ────────────────────────────────────────────

def enrich_figure_mds(conn, sessions: list[dict], figures: list[dict]) -> int:
    """figure .md に「生成後の文脈」を追記する（DB で enrichment 状態を管理）。"""
    # 日付ごとに claude-session をグルーピング
    sessions_by_date: dict[str, list[dict]] = {}
    for s in sessions:
        d = s.get("date", "")
        sessions_by_date.setdefault(d, []).append(s)

    enriched = 0
    for fig in figures:
        unique_key = fig.get("unique_key", "")
        fig_date = fig.get("date", "")
        fig_md = fig.get("md_path", "")

        if not fig_md or not fig_date:
            continue

        # DB で enrichment 済みかチェック（ファイル読み込み不要）
        if session_db.is_figure_enriched(conn, unique_key):
            continue

        fig_path = Path(fig_md)
        if not fig_path.exists():
            continue

        existing = fig_path.read_text(encoding="utf-8")

        # 候補セッション（当日 + 翌2日）
        candidate_sessions = []
        for offset in range(3):
            try:
                check_date = (date.fromisoformat(fig_date) + timedelta(days=offset)).isoformat()
            except ValueError:
                continue
            candidate_sessions.extend(sessions_by_date.get(check_date, []))

        if not candidate_sessions:
            continue

        # セッションの詳細情報を DB から取得
        context_entries = []
        for session in candidate_sessions[:4]:
            sid = session.get("session_id", "")
            md_path_str = session.get("md_path", "")
            if not md_path_str:
                continue

            bash_cmds = session_db.query_session_bash_commands(conn, sid)
            edited_raw = session.get("edited_files_raw", "") or ""
            edited_files = [f for f in edited_raw.split("|||") if f]

            first_msg = session.get("first_user_message", "")
            if not first_msg and not bash_cmds and not edited_files:
                continue

            context_entries.append({
                "session_id": sid,
                "md_path": md_path_str,
                "first_user_message": first_msg,
                "bash_cmds": bash_cmds[:5],
                "edited_files": edited_files[:8],
                "thinking_blocks": session.get("thinking_blocks", 0),
                "error_count": session.get("error_count", 0),
            })

        if not context_entries:
            continue

        # "生成後の文脈" セクションを構築
        ctx_lines = [
            "",
            "---",
            "",
            "## 生成後の文脈（この図を見た後に何が起きたか）",
            "",
            f"> この図（{fig_date}生成）の後に実行された Claude Code セッション:",
            "",
        ]

        for entry in context_entries:
            sid8 = entry["session_id"][:8]
            ctx_lines.append(f"### セッション: `{sid8}`")
            ctx_lines.append(f"*full path*: `{entry['md_path']}`")
            ctx_lines.append("")

            if entry["first_user_message"]:
                ctx_lines.append("**依頼内容（最初の発言）:**")
                ctx_lines.append(f"> {entry['first_user_message'][:200]}")
                ctx_lines.append("")

            if entry["bash_cmds"]:
                ctx_lines.append("**実行コマンド（再現用）:**")
                ctx_lines.append("```bash")
                for bc in entry["bash_cmds"]:
                    ctx_lines.append(bc["command"])
                ctx_lines.append("```")
                ctx_lines.append("")

            if entry["edited_files"]:
                ctx_lines.append("**変更されたコードファイル:**")
                for ef in entry["edited_files"]:
                    short = ef.replace("/Users/kitak/QPI_Omni/", "")
                    ctx_lines.append(f"- `{short}`")
                ctx_lines.append("")

            metrics = (f"thinking={entry['thinking_blocks']}"
                       f" errors={entry['error_count']}")
            ctx_lines.append(f"*指標: {metrics}*")
            ctx_lines.append("")

        context_text = "\n".join(ctx_lines)

        if "<!-- POST_FIGURE_CONTEXT -->" in existing:
            updated = existing.replace("<!-- POST_FIGURE_CONTEXT -->", context_text)
        else:
            updated = existing + context_text

        fig_path.write_text(updated, encoding="utf-8")
        session_db.mark_figure_enriched(conn, unique_key)
        conn.commit()
        enriched += 1

    print(f"  figure enrichment: {enriched} 件に「生成後の文脈」を追記")
    return enriched


# ────────────────────────────────────────────
# Markdown レンダリング
# ────────────────────────────────────────────

def _fmt_ts(ts: str) -> str:
    if not ts:
        return "??:??"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(JST)
        return dt.strftime("%H:%M")
    except Exception:
        return "??:??"


def _short_path(path: str) -> str:
    return path.replace("/Users/kitak/QPI_Omni/", "").replace("/Users/kitak/", "~/")


def render_session_block(conn, session: dict) -> list[str]:
    """1セッション分のサマリブロックを生成する。全情報を含む（制限なし）。"""
    sid = session["session_id"]
    sid8 = sid[:8]
    md_path = session.get("md_path", "")
    first_msg = session.get("first_user_message") or ""
    thinking = session.get("thinking_blocks", 0)
    errors = session.get("error_count", 0)
    bash_n = session.get("bash_command_count", 0)
    tool_n = session.get("tool_call_count", 0)
    design_notes_json = session.get("design_notes") or "[]"
    session_summary = session.get("session_summary") or ""

    # 編集ファイル
    edited_raw = session.get("edited_files_raw") or ""
    edited_files = [f for f in edited_raw.split("|||") if f]

    # DB から全データ取得
    bash_cmds = session_db.query_session_bash_commands(conn, sid)
    error_cmds = [bc for bc in bash_cmds if bc["is_error"]]
    code_changes = session_db.query_session_code_changes(conn, sid)
    error_details = session_db.query_session_errors(conn, sid)

    lines = [
        f"#### 🤖 セッション `{sid8}` [→ フルログ]",
        "",
    ]

    if md_path:
        lines.append(f"**ファイル:** `{md_path}`")
        lines.append("")

    # 依頼内容（全文）
    if first_msg:
        lines.append("**依頼内容（最初のユーザー発言）:**")
        for msg_line in first_msg.splitlines():
            lines.append(f"> {msg_line}")
        lines.append("")

    # セッションサマリ
    if session_summary:
        lines.append("**セッションサマリ:**")
        for sl in session_summary.splitlines():
            lines.append(f"> {sl}")
        lines.append("")

    # 変更ファイル（全件）
    if edited_files:
        lines.append(f"**変更ファイル（{len(edited_files)} 件）:**")
        for ef in edited_files:
            lines.append(f"- `{_short_path(ef)}`")
        lines.append("")

    # 再現コマンド（全件）
    if bash_cmds:
        err_summary = f" （エラー {len(error_cmds)} 件）" if error_cmds else ""
        lines.append(f"**再現コマンド（{len(bash_cmds)} 件{err_summary}）:**")
        lines.append("")
        lines.append("```bash")
        for bc in bash_cmds:
            n = bc.get("event_order", "?")
            ts = _fmt_ts(bc.get("timestamp", ""))
            err_mark = "  # ⚠️ ERROR" if bc["is_error"] else ""
            desc = bc.get("description") or ""
            if desc:
                lines.append(f"# [{n}] {ts} — {desc}{err_mark}")
            else:
                lines.append(f"# [{n}] {ts}{err_mark}")
            lines.append(bc["command"])
            # 出力プレビュー（エラー時のみ表示 or 非エラーでも出力がある場合）
            out = bc.get("output_preview") or ""
            if out:
                if bc["is_error"]:
                    for oline in out.splitlines()[:10]:
                        lines.append(f"#   {oline}")
                else:
                    preview = out[:200].replace("\n", " ↵ ")
                    lines.append(f"# → {preview}")
        lines.append("```")
        lines.append("")

    # 設計ノート（全ブロック・全文）
    try:
        design_notes = json.loads(design_notes_json)
    except Exception:
        design_notes = []
    if design_notes:
        lines.append(f"**設計ノート（AI 推論 / {len(design_notes)} ブロック）:**")
        lines.append("")
        for i, note in enumerate(design_notes, 1):
            has_math = bool(re.search(
                r"[\\$∑∫≈×÷±∂π]|\\frac|\\sum|np\.|[a-z]\s*=\s*[\d\(]", note
            ))
            math_mark = " 🔢" if has_math else ""
            lines.append(f"> **[ブロック {i}]{math_mark}**")
            for note_line in note.splitlines():
                lines.append(f"> {note_line}")
            lines.append(">")
        lines.append("")

    # コード変更詳細（全件）
    if code_changes:
        lines.append(f"**コード変更詳細（{len(code_changes)} 件）:**")
        lines.append("")
        for cc in code_changes:
            fp = _short_path(cc.get("file_path", ""))
            op = cc.get("operation", "")
            ts = _fmt_ts(cc.get("timestamp", ""))
            ext = cc.get("file_path", "").rsplit(".", 1)[-1].lower()
            lang = {"py": "python", "js": "javascript", "ts": "typescript",
                    "json": "json", "md": "markdown", "sh": "bash",
                    "yaml": "yaml", "yml": "yaml"}.get(ext, "")
            lines.append(f"##### `{fp}` — {op} [{ts}]")
            old_p = cc.get("old_preview") or ""
            new_p = cc.get("new_preview") or ""
            if op == "Edit" and old_p:
                lines.append("変更前:")
                lines.append(f"```{lang}")
                lines.append(old_p)
                lines.append("```")
                lines.append("変更後:")
                lines.append(f"```{lang}")
                lines.append(new_p)
                lines.append("```")
            elif new_p:
                lines.append(f"```{lang}")
                lines.append(new_p)
                lines.append("```")
            lines.append("")

    # エラー詳細（全件）
    if error_details:
        lines.append(f"**⚠️ エラー詳細（{len(error_details)} 件）:**")
        lines.append("")
        for err in error_details:
            n = err.get("event_order", "?")
            ts = _fmt_ts(err.get("timestamp", ""))
            tool = err.get("tool_name", "")
            err_text = (err.get("error_text") or err.get("result_preview") or "")[:400]
            lines.append(f"- `[{n}] {ts} {tool}`: {err_text[:120]}")
        lines.append("")

    # 指標
    err_mark = f" | **⚠️ エラー {errors}**" if errors > 0 else ""
    lines.append(f"*指標: thinking={thinking} | tools={tool_n} | bash={bash_n}{err_mark}*")
    lines.append("")

    # セッション .md 全文を埋め込む（制限なし・タイムライン含む完全記録）
    if md_path:
        md_file = Path(md_path)
        if md_file.exists():
            lines.append(f"**セッション全文（タイムライン・コード変更・全ツール出力含む）:**")
            lines.append("")
            lines.extend(md_file.read_text(encoding="utf-8").splitlines())
            lines.append("")

    lines.append("---")
    lines.append("")

    return lines


def render_figure_block(fig: dict) -> list[str]:
    """1図分のサマリブロックを生成する。"""
    script = fig.get("script", "unknown")
    desc = fig.get("description", "")
    commit = fig.get("git_commit", "")
    dirty = fig.get("git_dirty", False)
    md_path = fig.get("md_path", "")
    image_name = fig.get("image_obsidian_name", "")

    lines = [
        f"#### 📊 図: `{script}` [→ フルメタデータ]",
        "",
    ]
    if md_path:
        lines.append(f"**ファイル:** `{md_path}`")
    if desc:
        lines.append(f"**説明:** {desc}")
    if commit:
        dirty_mark = " *(dirty)*" if dirty else ""
        lines.append(f"**git commit:** `{commit[:8]}`{dirty_mark}")
    if image_name:
        lines.append(f"**図ファイル:** `{image_name}`")
        lines.append(f"![[{image_name}]]")
    lines.append("")
    lines.append("---")
    lines.append("")

    return lines


def render_notion_block(notion: dict) -> list[str]:
    """1 Notion メモ分のサマリブロックを生成する。全文を含め、解釈指示を付ける。"""
    title = notion.get("title", "（無題）")
    notion_url = notion.get("notion_url", "")
    full_body = notion.get("full_body", "")
    path = notion.get("path", "")

    link_text = f"[Notion]({notion_url})" if notion_url else ""
    lines = [
        f"#### 📝 Notion メモ: 「{title}」 {link_text}",
        "",
    ]
    if path:
        lines.append(f"**ファイル:** `{path}`")
        lines.append("")

    lines.append("**全文（このユーザーが考えていること・やりたいこと）:**")
    lines.append("")
    if full_body:
        lines.append("```")
        lines.append(full_body.strip())
        lines.append("```")
        lines.append("")
        lines.append(
            "> **Claude へ**: 上記を読み、「このユーザーは〇〇を考えている／〇〇を試したかった／〇〇が気になっている」"
            "と解釈してレポートに織り込むこと。そのまま要約するのではなく、意図が伝わる形で書き直す。"
        )
    else:
        lines.append("（本文なし）")

    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def render_daily_index(
    d_str: str,
    week_label: str,
    conn,
    sessions: list[dict],
    figures: list[dict],
    notions: list[dict],
    lib_entries: list[dict],
) -> str:
    """1日分の日次索引 Markdown を生成する。全情報を含む（制限なし）。"""
    try:
        d = date.fromisoformat(d_str)
    except ValueError:
        d = date.today()
    weekday = WEEKDAY_JP[d.weekday()]

    lines = [
        "---",
        "type: daily-index",
        f"date: {d_str}",
        f"week: {week_label}",
        f'generated_at: "{datetime.now(JST).isoformat()}"',
        "---",
        "",
        f"# {d_str}（{weekday}）日次索引",
        "",
        f"> **Claude へ**: この1ファイルに {d_str} の全情報が含まれている。"
        " セッション .md ファイルを読む必要はない。",
        "",
    ]

    # セッション
    if sessions:
        lines.append(f"## 🤖 Claude セッション（{len(sessions)} 件）")
        lines.append("")
        for s in sessions:
            lines.extend(render_session_block(conn, s))

    # 図
    if figures:
        lines.append(f"## 📊 図（{len(figures)} 件）")
        lines.append("")
        for fig in figures:
            lines.extend(render_figure_block(fig))

    # Notion メモ
    if notions:
        lines.append(f"## 📝 Notion メモ（{len(notions)} 件）")
        lines.append("")
        for notion in notions:
            lines.extend(render_notion_block(notion))

    # figure-hub library 登録
    if lib_entries:
        lines.append(f"## 🖼️ figure-hub Library 登録（{len(lib_entries)} 件）")
        lines.append("")
        for lb in lib_entries:
            lines.append(f"### `{lb['fig_id']}` {lb['version']}")
            if lb.get("note"):
                lines.append(f"- note: {lb['note']}")
            lines.append(f"- `{lb['path']}`")
            lines.append("")

    return "\n".join(lines)


def render_weekly_index(
    week_label: str,
    monday: date,
    sunday: date,
    conn,
    sessions: list[dict],
    figures: list[dict],
    notions: list[dict],
    lib_entries: list[dict],
    daily_files: list[tuple[str, str]],
) -> str:
    """週次索引 Markdown を生成する（軽量サマリ + 日次ファイルへのリンク集）。"""

    stats = session_db.query_week_stats(conn, monday.isoformat(), sunday.isoformat())
    file_stats = session_db.query_week_file_stats(conn, monday.isoformat(), sunday.isoformat())

    lines = []

    # ── フロントマター ──
    lines.extend([
        "---",
        "type: weekly-index",
        f"week: {week_label}",
        f"period: {monday.isoformat()} - {sunday.isoformat()}",
        f'generated_at: "{datetime.now(JST).isoformat()}"',
        "db_based: true",
        "---",
        "",
    ])

    # ── タイトル ──
    lines.append(
        f"# 週次索引 {week_label}（{monday.strftime('%m/%d')} – {sunday.strftime('%m/%d')}）"
    )
    lines.append("")
    lines.append(
        "> **Claude へ**: 週次レポートを書くには下記の **日次索引ファイルを日付順に Read** すること。"
        " 日次索引1ファイルに1日分の全情報（bash・コード変更・設計ノート・Notionメモ全文）が含まれている。"
        " Notion MCP は使わない。"
    )
    lines.append("")

    # ── 日次索引ファイル一覧 ──
    lines.append("## 📅 日次索引ファイル（Read tool で日付順に読む）")
    lines.append("")
    for d_str, path in daily_files:
        try:
            d = date.fromisoformat(d_str)
        except ValueError:
            continue
        weekday = WEEKDAY_JP[d.weekday()]
        day_sessions = [s for s in sessions if s.get("date") == d_str]
        day_figures = [f for f in figures if f.get("date") == d_str]
        day_notions = [n for n in notions if n.get("date") == d_str]
        lines.append(
            f"- **{d_str}（{weekday}）** `{path}`"
            f"  — セッション {len(day_sessions)} / 図 {len(day_figures)} / Notion {len(day_notions)}"
        )
    lines.append("")

    # ── 週サマリ ──
    lines.append("## 週サマリ")
    lines.append("")
    lines.append("| 指標 | 値 |")
    lines.append("|------|-----|")
    lines.append(f"| Claude セッション数 | {stats.get('session_count', 0)} |")
    lines.append(f"| ユニーク編集ファイル数 | {stats.get('unique_file_count', 0)} |")
    lines.append(f"| Bash 実行回数 | {stats.get('total_bash', 0)} |")
    lines.append(f"| エラー発生回数 | {stats.get('total_errors', 0)} |")
    lines.append(f"| AI 推論ブロック数 | {stats.get('total_thinking', 0)} |")
    lines.append(f"| 全ツール呼び出し数 | {stats.get('total_tools', 0)} |")
    lines.append(f"| 図生成（inbox） | {stats.get('figure_count', 0)} |")
    lines.append(f"| 仕上げ図登録（library） | {len(lib_entries)} |")
    lines.append(f"| Notion メモ | {len(notions)} |")
    lines.append("")

    # ── セッション一覧表 ──
    if sessions:
        lines.append("## 🤖 セッション一覧")
        lines.append("")
        lines.append("| 日付 | ID | 推論 | bash | エラー | files | 依頼 |")
        lines.append("|------|-----|------|------|-------|-------|------|")
        for s in sessions:
            d_label = s.get("date", "")[-5:]
            sid8 = s["session_id"][:8]
            thinking = s.get("thinking_blocks", 0)
            bash_n = s.get("bash_command_count", 0)
            errs = s.get("error_count", 0)
            err_mark = f"**{errs}**" if errs > 0 else str(errs)
            edited_raw = s.get("edited_files_raw") or ""
            file_n = len([f for f in edited_raw.split("|||") if f])
            msg = (s.get("first_user_message") or "")[:40]
            lines.append(
                f"| {d_label} | `{sid8}` | {thinking} | {bash_n} | {err_mark} | {file_n} | {msg} |"
            )
        lines.append("")

    # ── エラーサマリ ──
    error_sessions = [s for s in sessions if s.get("error_count", 0) > 0]
    if error_sessions:
        lines.append("## ⚠️ エラーサマリ")
        lines.append("")
        lines.append("| セッション | 日付 | エラー数 | 最初のユーザー発言 |")
        lines.append("|----------|------|---------|----------------|")
        for s in error_sessions:
            sid8 = s["session_id"][:8]
            d_label = s.get("date", "")
            errs = s.get("error_count", 0)
            msg = (s.get("first_user_message") or "")[:50]
            lines.append(f"| `{sid8}` | {d_label} | {errs} | {msg} |")
        lines.append("")

    # ── 変更ファイル統計 ──
    if file_stats:
        lines.append("## 変更ファイル統計（全セッション統合）")
        lines.append("")
        lines.append("| ファイル | 変更回数 | 関連日 | セッション数 |")
        lines.append("|---------|---------|-------|------------|")
        for fs in file_stats[:20]:
            fp = _short_path(fs["file_path"])
            dates_str = (fs.get("dates") or "").replace(",", ", ")[:40]
            lines.append(
                f"| `{fp}` | {fs['change_count']} | {dates_str} | {fs['session_count']} |"
            )
        lines.append("")

    # ── 週次レポート生成指示 ──
    lines.append("---")
    lines.append("")
    lines.append("## 週次レポート生成の指示")
    lines.append("")
    lines.append("上記の日次索引ファイルを全て読み終えたら、`docs/WEEKLY_LOG_SPEC.md` に従ってレポートを生成する。")
    lines.append("")
    lines.append("**仕様の要点:**")
    lines.append("- トピック（内容）単位でまとめる。セッション単位・日付単位で区切らない")
    lines.append("- 各トピックで「設計 → 実行 → 図 → 次にこう変更」の因果の流れを記述")
    lines.append("- 1まとまりあたり Qiita 記事相当の厚み（背景・手順・コード抜粋・結果・学び）")
    lines.append("")
    lines.append(f"**保存先:** `~/Documents/Obsidian Vault/04_WeeklyReports/{week_label}.md`")
    lines.append("")

    return "\n".join(lines)


# ────────────────────────────────────────────
# メイン
# ────────────────────────────────────────────

def run(args: argparse.Namespace) -> int:
    monday, sunday, week_label = get_week_range(args.week)
    print(f"対象週: {week_label} ({monday} – {sunday})")

    # 前処理
    if not args.no_preprocess:
        run_preprocess(args.dry_run)

    print(f"\n--- ソーススキャン ---")

    conn = session_db.open_db()
    run_id = session_db.start_processing_run(conn, "weekly_report_hub")

    try:
        # DB からセッション・図を取得
        sessions = session_db.query_week_sessions(
            conn, monday.isoformat(), sunday.isoformat()
        )
        figures = session_db.query_week_figures(
            conn, monday.isoformat(), sunday.isoformat()
        )

        # ファイルスキャン（Notion・library は DB 対象外）
        notions = scan_notion_sync(monday, sunday)
        lib_entries = scan_figure_hub_library(monday, sunday)

        print(f"claude_sessions (DB): {len(sessions)} 件")
        print(f"figure_events (DB):   {len(figures)} 件")
        print(f"notion_sync:          {len(notions)} 件")
        print(f"figure_hub_library:   {len(lib_entries)} 件")

        if not sessions and not figures and not notions and not lib_entries:
            print(f"\n警告: {week_label} に該当するイベントが見つかりませんでした。")

        # Cross-reference enrichment
        if not args.dry_run and not args.no_enrich:
            print(f"\n--- Cross-reference enrichment ---")
            enrich_figure_mds(conn, sessions, figures)

        # 日付ごとにグループ化
        sessions_by_date: dict[str, list] = {}
        for s in sessions:
            sessions_by_date.setdefault(s["date"], []).append(s)
        figures_by_date: dict[str, list] = {}
        for f in figures:
            figures_by_date.setdefault(f["date"], []).append(f)
        notions_by_date: dict[str, list] = {}
        for n in notions:
            notions_by_date.setdefault(n["date"], []).append(n)
        libs_by_date: dict[str, list] = {}
        for lb in lib_entries:
            libs_by_date.setdefault(lb["date"], []).append(lb)

        all_dates = sorted(set(
            list(sessions_by_date) + list(figures_by_date) +
            list(notions_by_date) + list(libs_by_date)
        ))

        # 日次索引ファイルを生成
        daily_files: list[tuple[str, str]] = []
        inbox_dir = OBSIDIAN_ROOT / "00_Inbox"
        inbox_dir.mkdir(parents=True, exist_ok=True)

        for d_str in all_dates:
            day_content = render_daily_index(
                d_str, week_label, conn,
                sessions_by_date.get(d_str, []),
                figures_by_date.get(d_str, []),
                notions_by_date.get(d_str, []),
                libs_by_date.get(d_str, []),
            )
            day_path = inbox_dir / f"daily_index_{d_str}.md"
            if not args.dry_run:
                day_path.write_text(day_content, encoding="utf-8")
                print(f"  日次索引: {day_path.name}"
                      f" ({len(day_content.splitlines())} 行)")
            else:
                print(f"  [dry-run] would write: {day_path.name}"
                      f" ({len(day_content.splitlines())} 行)")
            daily_files.append((d_str, str(day_path)))

        # 週次索引（軽量サマリ）を生成
        md_content = render_weekly_index(
            week_label, monday, sunday,
            conn, sessions, figures, notions, lib_entries,
            daily_files=daily_files,
        )

        out_path = inbox_dir / f"weekly_index_{week_label}.md"

        if args.dry_run:
            print(f"\n[dry-run] would write: {out_path}")
            total = len(sessions) + len(figures) + len(notions) + len(lib_entries)
            print(f"  total events: {total}")
            session_db.finish_processing_run(conn, run_id, success=True,
                                              notes="dry-run")
            return 0

        out_path.write_text(md_content, encoding="utf-8")

        total = len(sessions) + len(figures) + len(notions) + len(lib_entries)
        session_db.finish_processing_run(conn, run_id, processed=total, success=True)

        print(f"\n週次索引（軽量）: {out_path}")
        print(f"  セッション: {len(sessions)} | 図: {len(figures)} |"
              f" Notion: {len(notions)} | Library: {len(lib_entries)}")
        print(f"  日次索引: {len(daily_files)} ファイル")
        print(f"\n次のステップ:")
        print(f"  Claude Code で: 「週次レポートを書いて」"
              f" または「{week_label} の週次レポートを書いて」")

    except Exception as e:
        session_db.finish_processing_run(conn, run_id, success=False, notes=str(e))
        conn.close()
        raise
    finally:
        conn.close()

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="週次レポート用統一タイムライン索引を生成")
    parser.add_argument("--week", help="対象週 (例: 2026-W10)。省略時は今週")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="前処理スクリプトをスキップ")
    parser.add_argument("--no-enrich", action="store_true",
                        help="figure .md への cross-reference 追記をスキップ")
    parser.add_argument("--dry-run", action="store_true",
                        help="ファイル書き出しなしで確認")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args))
