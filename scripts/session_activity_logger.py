#!/usr/bin/env python3
"""Collect QPI_Omni activity and append to Obsidian daily notes.

What this script logs:
1) Finished Cursor / Claude Code transcripts related to QPI_Omni.
2) Git state changes in QPI_Omni.

Output format for transcript entries follows docs/WORKLOG_SPEC.md.

Run once:
  python scripts/session_activity_logger.py

Install periodic run (macOS launchd):
  bash scripts/install_activity_logger_launchd.sh
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = REPO_ROOT / ".figure_history" / "session_activity_state.json"
DEFAULT_OBSIDIAN_DAILY_DIR = Path("/Users/kitak/Documents/Obsidian Vault/01_Daily")
WORKLOG_SPEC_PATH = REPO_ROOT / "docs" / "WORKLOG_SPEC.md"

CURSOR_GLOB = "/Users/kitak/.cursor/projects/*QPI-Omni*/agent-transcripts/*/*.jsonl"
CLAUDE_GLOB = "/Users/kitak/.claude/projects/*QPI-Omni*/*.jsonl"

USER_QUERY_OPEN = "<user_query>"
USER_QUERY_CLOSE = "</user_query>"
FILE_PATTERN = re.compile(r"(?:[\w./-]+\.(?:py|md|tex|json|toml|yml|yaml|sh|ipynb|csv|png|jpg|jpeg|pdf))")

COMMAND_PREFIXES = (
    "python",
    "python3",
    "bash",
    "zsh",
    "sh",
    "git",
    "launchctl",
    "cp",
    "mv",
    "sed",
    "rg",
    "cat",
    "chmod",
    "brew",
    "pip",
    "uv",
    "npm",
    "node",
    "pytest",
)


def now_local() -> dt.datetime:
    return dt.datetime.now().astimezone()


def iso_now() -> str:
    return now_local().replace(microsecond=0).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append Cursor/Claude/git activity to Obsidian daily notes")
    parser.add_argument("--repo-root", default=str(REPO_ROOT), help="QPI_Omni repository path")
    parser.add_argument("--obsidian-daily-dir", default=str(DEFAULT_OBSIDIAN_DAILY_DIR), help="Obsidian daily notes folder")
    parser.add_argument("--inactivity-minutes", type=int, default=15, help="Transcript considered finished after this idle duration")
    parser.add_argument("--dry-run", action="store_true", help="Print planned entries without writing files")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "transcripts": {},
            "git": {"head": None, "status_hash": None},
            "last_run": None,
        }
    try:
        state = read_json(path)
    except Exception:
        return {
            "transcripts": {},
            "git": {"head": None, "status_hash": None},
            "last_run": None,
        }
    if not isinstance(state, dict):
        return {
            "transcripts": {},
            "git": {"head": None, "status_hash": None},
            "last_run": None,
        }
    state.setdefault("transcripts", {})
    state.setdefault("git", {"head": None, "status_hash": None})
    state.setdefault("last_run", None)
    return state


def cleanup_user_query(text: str) -> str:
    t = text.strip()
    if USER_QUERY_OPEN in t:
        start = t.find(USER_QUERY_OPEN) + len(USER_QUERY_OPEN)
        end = t.find(USER_QUERY_CLOSE, start)
        if end == -1:
            return t[start:].strip()
        return t[start:end].strip()
    return t


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    txt = item.get("text", "")
                    if isinstance(txt, dict):
                        chunks.append(str(txt.get("value", "")))
                    else:
                        chunks.append(str(txt))
        return "".join(chunks)
    return ""


def parse_messages(jsonl_path: Path) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    with jsonl_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue

            role: Optional[str] = None
            text: str = ""

            if isinstance(obj, dict) and "role" in obj:
                role = obj.get("role")
                msg = obj.get("message", {})
                if isinstance(msg, dict):
                    text = content_to_text(msg.get("content"))

            elif isinstance(obj, dict) and obj.get("type") in {"user", "assistant"}:
                role = obj.get("type")
                msg = obj.get("message", {})
                if isinstance(msg, dict):
                    text = content_to_text(msg.get("content"))

            if role in {"user", "assistant"}:
                text = text.strip()
                if text:
                    msgs.append({"role": role, "text": text})

    return msgs


def infer_source(path: Path) -> str:
    sp = str(path)
    if "/.cursor/" in sp:
        return "Cursor"
    if "/.claude/" in sp:
        return "Claude Code"
    return "Unknown"


def _truncate(text: str, n: int) -> str:
    t = text.strip()
    if len(t) <= n:
        return t
    return t[:n].rstrip() + "..."


def _clean_command(candidate: str) -> str:
    c = candidate.strip()
    if c.startswith("$"):
        c = c[1:].strip()
    if c.startswith("`") and c.endswith("`") and len(c) >= 2:
        c = c[1:-1].strip()
    return c


def _looks_like_command(candidate: str) -> bool:
    c = _clean_command(candidate)
    if not c:
        return False
    head = c.split(maxsplit=1)[0]
    return any(head == p or head.startswith(p + ".") for p in COMMAND_PREFIXES)


def extract_command_hints(text: str, max_items: int = 8) -> List[str]:
    hints: List[str] = []
    seen = set()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        c = _clean_command(line)
        if _looks_like_command(c):
            short = _truncate(c, 180)
            if short not in seen:
                seen.add(short)
                hints.append(short)
                if len(hints) >= max_items:
                    return hints

    for m in re.finditer(r"`([^`]{3,220})`", text):
        c = _clean_command(m.group(1))
        if _looks_like_command(c):
            short = _truncate(c, 180)
            if short not in seen:
                seen.add(short)
                hints.append(short)
                if len(hints) >= max_items:
                    return hints

    return hints


def infer_constraints(text: str) -> List[str]:
    lower = text.lower()
    constraints: List[str] = []

    rules = [
        ("権限/TCC制約の可能性がある（Operation not permitted / Permission denied など）。", ["operation not permitted", "permission denied", "full disk access", "tcc"]),
        ("依存パッケージ不足の可能性がある（Module not found など）。", ["module not found", "not installed", "importerror"]),
        ("APIレート制限または接続制約の可能性がある。", ["rate limit", "429", "network error", "api error"]),
    ]

    for message, keys in rules:
        if any(k in lower for k in keys):
            constraints.append(message)

    if not constraints:
        constraints.append("会話ログからの自動抽出のため、要約内容は必要に応じて手動確認が必要。")

    return constraints


def summarize_transcript(path: Path) -> Optional[Dict[str, Any]]:
    msgs = parse_messages(path)
    if not msgs:
        return None

    users = [m["text"] for m in msgs if m["role"] == "user"]
    assistants = [m["text"] for m in msgs if m["role"] == "assistant"]

    first_user = cleanup_user_query(users[0]) if users else ""
    last_assistant = assistants[-1] if assistants else ""

    title_src = first_user or "(no user query)"
    title_line = title_src.splitlines()[0].strip()[:120]

    joined = "\n".join(m["text"] for m in msgs)
    files = sorted(set(FILE_PATTERN.findall(joined)))
    files = files[:12]

    mtime = dt.datetime.fromtimestamp(path.stat().st_mtime).astimezone()

    user_highlights = [cleanup_user_query(u) for u in users[:4]]
    user_highlights = [_truncate(u, 320) for u in user_highlights if u.strip()]

    assistant_highlights = [_truncate(a, 420) for a in assistants[-3:] if a.strip()]

    return {
        "source": infer_source(path),
        "path": str(path),
        "title": title_line,
        "first_user": first_user[:1200],
        "last_assistant": last_assistant[:1200],
        "user_highlights": user_highlights,
        "assistant_highlights": assistant_highlights,
        "command_hints": extract_command_hints(joined),
        "constraints": infer_constraints(joined),
        "files": files,
        "user_turns": len(users),
        "assistant_turns": len(assistants),
        "message_count": len(msgs),
        "mtime": mtime,
    }


def ensure_daily_note(path: Path, date_str: str, dry_run: bool) -> None:
    if path.exists():
        return
    if dry_run:
        print(f"[dry-run] would create daily note: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "---",
                "type: daily",
                f"date: {date_str}",
                "tags: [daily, research]",
                "---",
                "",
                "## One-line summary",
                "",
                "## Worklog entries (spec-driven)",
                "",
                "## Notes",
                "",
                "## Draft for group meeting",
                "",
            ]
        ),
        encoding="utf-8",
    )


def append_daily_entry(daily_file: Path, block: str, dry_run: bool) -> None:
    if dry_run:
        print("----- DAILY ENTRY -----")
        print(f"target: {daily_file}")
        print(block)
        print("-----------------------")
        return

    with daily_file.open("a", encoding="utf-8") as f:
        f.write("\n\n---\n\n")
        f.write(block.rstrip())
        f.write("\n")


def render_transcript_block(item: Dict[str, Any]) -> str:
    t = item["mtime"].strftime("%H:%M")
    requirements = item["user_highlights"] or ([item["first_user"]] if item["first_user"] else ["none"])
    policies = item["assistant_highlights"] or ([item["last_assistant"]] if item["last_assistant"] else ["none"])
    commands = item.get("command_hints", [])
    files = item.get("files", [])
    constraints = item.get("constraints", ["none"])

    lines = [
        f"## {t} | 作業ログ ({item['source']})",
        "",
        "### 1. 背景",
        f"- セッションタイトル: {item['title']}",
        f"- Transcript: `{item['path']}`",
        f"- 会話ターン: user={item['user_turns']}, assistant={item['assistant_turns']}, total={item['message_count']}",
        f"- 仕様参照: `{WORKLOG_SPEC_PATH}`",
        "",
        "### 2. 要件定義（ユーザー要件）",
    ]

    for r in requirements:
        lines.append(f"- {r}")

    lines.extend([
        "",
        "### 3. 実装方針",
    ])
    for p in policies[:4]:
        lines.append(f"- {p}")

    lines.extend([
        "",
        "### 4. 実装手順（Step 1, 2, 3… コマンド付き）",
        "1. 要件確認",
        f"   - source: `{item['path']}`",
        "2. 実装・操作",
    ])

    if commands:
        for c in commands:
            lines.append(f"   - command: `{c}`")
    else:
        lines.append("   - command: `none`")

    lines.extend([
        "3. 記録",
        "   - command: `python3 scripts/session_activity_logger.py`",
        "",
        "### 5. 検証手順と結果",
        "1. inactivity条件を満たす transcript のみ記録対象。",
        f"2. 結果: user_turns={item['user_turns']}, assistant_turns={item['assistant_turns']}, messages={item['message_count']}",
        f"3. 最終応答要約: {_truncate(item['last_assistant'] or 'none', 320)}",
        "",
        "### 6. 変更ファイル一覧",
    ])

    if files:
        for fp in files:
            lines.append(f"- `{fp}`")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "### 7. 既知の制約・注意点",
    ])
    for c in constraints:
        lines.append(f"- {c}")

    lines.extend([
        "",
        "### 8. 他PCでの再現手順",
        "1. `/Users/kitak/...` の絶対パスを対象PCのパスへ置換する。",
        "2. Step 4で記録したコマンドを上から順に実行する。",
        "3. `python3 scripts/session_activity_logger.py --dry-run` で出力確認後、本実行する。",
    ])

    return "\n".join(lines)


def run_git(args_repo_root: Path) -> Dict[str, Any]:
    def run(cmd: List[str]) -> str:
        return subprocess.check_output(cmd, text=True, cwd=args_repo_root).strip()

    head = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--short"])
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    latest = run(["git", "show", "-s", "--format=%h %ci %s", "HEAD"])

    changed_files = []
    if status:
        for line in status.splitlines():
            if not line.strip():
                continue
            changed_files.append(line)

    status_hash = hashlib.sha256(status.encode("utf-8")).hexdigest()

    return {
        "head": head,
        "branch": branch,
        "latest": latest,
        "status": status,
        "status_hash": status_hash,
        "changed_files": changed_files,
    }


def render_git_block(git_info: Dict[str, Any]) -> str:
    t = now_local().strftime("%H:%M")
    lines = [
        f"## {t} | Repo snapshot",
        "",
        "### 5. 検証手順と結果",
        f"- Branch: `{git_info['branch']}`",
        f"- HEAD: `{git_info['head'][:12]}`",
        f"- Latest commit: `{git_info['latest']}`",
        "",
        "### 6. 変更ファイル一覧",
    ]

    if git_info["changed_files"]:
        for item in git_info["changed_files"][:30]:
            lines.append(f"- `{item}`")
        if len(git_info["changed_files"]) > 30:
            lines.append(f"- `... ({len(git_info['changed_files']) - 30} more)`")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "### 7. 既知の制約・注意点",
        "- このブロックはGit状態の要約で、会話内容の要件定義や実装意図までは含まない。",
        "",
        "### 8. 他PCでの再現手順",
        "1. `cd <repo>`",
        "2. `git status --short`",
        "3. `git show -s --format=%h %ci %s HEAD`",
    ])

    return "\n".join(lines)


def transcript_candidates() -> List[Path]:
    paths: List[Path] = []
    paths.extend(Path("/").glob(CURSOR_GLOB.lstrip("/")))
    paths.extend(Path("/").glob(CLAUDE_GLOB.lstrip("/")))
    return sorted(set(paths), key=lambda p: p.stat().st_mtime)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    obsidian_daily_dir = Path(args.obsidian_daily_dir).resolve()
    idle_s = max(args.inactivity_minutes, 1) * 60

    state = load_state(STATE_FILE)
    transcript_state: Dict[str, float] = state.get("transcripts", {})

    now_ts = now_local().timestamp()

    # 1) Transcript logs
    trans_items: List[Dict[str, Any]] = []
    for p in transcript_candidates():
        sp = str(p)
        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue

        previous = float(transcript_state.get(sp, 0.0))
        if mtime <= previous:
            continue
        if now_ts - mtime < idle_s:
            continue

        info = summarize_transcript(p)
        if info:
            trans_items.append(info)
            transcript_state[sp] = mtime

    # 2) Git logs
    git_info = run_git(repo_root)
    prev_git = state.get("git", {}) if isinstance(state.get("git"), dict) else {}

    git_changed = (
        git_info["head"] != prev_git.get("head")
        or git_info["status_hash"] != prev_git.get("status_hash")
    )

    # Write daily entries grouped by date
    entries_by_date: Dict[str, List[str]] = {}

    for item in trans_items:
        date_str = item["mtime"].strftime("%Y-%m-%d")
        entries_by_date.setdefault(date_str, []).append(render_transcript_block(item))

    if git_changed:
        today = now_local().strftime("%Y-%m-%d")
        entries_by_date.setdefault(today, []).append(render_git_block(git_info))

    for date_str, blocks in sorted(entries_by_date.items()):
        daily_file = obsidian_daily_dir / f"{date_str}.md"
        ensure_daily_note(daily_file, date_str, args.dry_run)
        for block in blocks:
            append_daily_entry(daily_file, block, args.dry_run)

    # Persist state
    state["transcripts"] = transcript_state
    state["git"] = {"head": git_info["head"], "status_hash": git_info["status_hash"]}
    state["last_run"] = iso_now()

    if not args.dry_run:
        write_json(STATE_FILE, state)

    print(
        "session_activity_logger:",
        f"transcript_entries={len(trans_items)}",
        f"git_entry={1 if git_changed else 0}",
        f"dry_run={args.dry_run}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
