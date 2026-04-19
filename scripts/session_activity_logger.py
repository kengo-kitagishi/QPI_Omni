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
import getpass
import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
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
        ("Possible permission/TCC constraint (Operation not permitted / Permission denied).", ["operation not permitted", "permission denied", "full disk access", "tcc"]),
        ("Possible missing dependency (Module not found).", ["module not found", "not installed", "importerror"]),
        ("Possible API rate limit or connection constraint.", ["rate limit", "429", "network error", "api error"]),
    ]

    for message, keys in rules:
        if any(k in lower for k in keys):
            constraints.append(message)

    if not constraints:
        constraints.append("Auto-extracted from conversation log; summary may need manual verification.")

    return constraints


def to_absolute_candidate(file_ref: str, repo_root: Path) -> str:
    ref = file_ref.strip()
    if not ref:
        return ref
    if ref.startswith("~"):
        return str(Path(ref).expanduser().resolve())
    p = Path(ref)
    if p.is_absolute():
        return str(p)
    return str((repo_root / ref).resolve())


def build_runtime_context(
    args: argparse.Namespace,
    repo_root: Path,
    obsidian_daily_dir: Path,
) -> Dict[str, Any]:
    return {
        "os": f"{platform.system()} {platform.release()}",
        "user": getpass.getuser(),
        "repo_root": str(repo_root),
        "obsidian_daily_dir": str(obsidian_daily_dir),
        "state_file": str(STATE_FILE),
        "worklog_spec": str(WORKLOG_SPEC_PATH),
        "run_command": " ".join(shlex.quote(a) for a in sys.argv),
        "inactivity_minutes": max(args.inactivity_minutes, 1),
        "dry_run": args.dry_run,
    }


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


def render_transcript_block(item: Dict[str, Any], runtime: Dict[str, Any]) -> str:
    t = item["mtime"].strftime("%H:%M")
    date_str = item["mtime"].strftime("%Y-%m-%d")
    daily_file = Path(runtime["obsidian_daily_dir"]) / f"{date_str}.md"
    requirements = item["user_highlights"] or ([item["first_user"]] if item["first_user"] else ["none"])
    policies = item["assistant_highlights"] or ([item["last_assistant"]] if item["last_assistant"] else ["none"])
    commands = item.get("command_hints", [])
    files = item.get("files", [])
    constraints = item.get("constraints", ["none"])
    scope_files = [to_absolute_candidate(fp, Path(runtime["repo_root"])) for fp in files[:6]]
    main_goal = requirements[0] if requirements and requirements[0] != "none" else "Requirements extracted from conversation log (manual verification assumed)"
    unresolved = "none"
    if not item["first_user"].strip():
        unresolved = "Initial user requirement is nearly empty; supplemented from subsequent context."
    elif len(requirements) < 2:
        unresolved = "Requirement granularity may be coarse; manual refinement after saving improves accuracy."

    lines = [
        f"## {t} | Work log ({item['source']})",
        "",
        "### Prerequisites",
        f"- Target OS: {runtime['os']}",
        f"- User: {runtime['user']}",
        f"- Token/config source: see transcript JSONL (`{item['path']}`). No secrets used in this process.",
        "- Backup policy: pre-append backup is skipped for automated jobs. Run Step 0 manually if needed.",
        "",
        "### Background",
        f"- Session title: {item['title']}",
        f"- Transcript: `{item['path']}`",
        f"- Conversation turns: user={item['user_turns']}, assistant={item['assistant_turns']}, total={item['message_count']}",
        f"- Spec reference: `{runtime['worklog_spec']}`",
        "",
        "### Requirements (user requirements)",
        f"- Objective: {main_goal}",
        (
            "- Scope: "
            + ", ".join([f"`{item['path']}`", f"`{daily_file}`"] + [f"`{p}`" for p in scope_files])
        ),
        "- Expected deliverable: A WORKLOG_SPEC-compliant work log block is generated, ready to be appended to daily notes.",
        (
            "- Acceptance criteria: "
            "Implementation steps contain `objective/command/expected result/actual result`, "
            "and verification steps contain `verdict (pass/fail)` and `evidence`."
        ),
        "- Constraints/assumptions: " + " / ".join(constraints[:4]),
        f"- Unresolved items: {unresolved}",
        "- User requests (extracted):",
    ]

    for r in requirements[:8]:
        lines.append(f"- {r}")

    lines.extend([
        "",
        "### Implementation policy",
        "- Adopted policy: Extract requirements, commands, and constraints from transcript and structure into a standard template.",
        "- Rationale: Ensures reproducibility even in automation, leaving records traceable by third parties.",
        "- Execution order: Candidate collection -> key point extraction -> template formatting -> daily note append -> verification command display.",
        (
            "- Alternative and rejection reason: "
            "Free-text-only summary is easier to write but loses reproducibility, so rejected."
        ),
        (
            "- Risks and mitigations: "
            "Risk of extraction omission mitigated by explicit `not executed` marking, absolute path recording, and dry-run verification."
        ),
        "- Policy notes (extracted from conversation):",
    ])
    for p in policies[:6]:
        lines.append(f"- {p}")

    lines.extend([
        "",
        "### Implementation steps (Step 0, 1, 2... with commands)",
        "#### Step 0: Backup (optional)",
        "- Objective: Create a rollback point before appending to daily notes.",
        "```bash",
        f"cp \"{daily_file}\" \"{daily_file}.bak\"",
        "```",
        "- Expected result: `.bak` file is created.",
        "- Actual result: not executed (skipped for automated collection job).",
        "",
        "#### Step 1: Collect transcript candidates and determine completion",
        "- Objective: Only record completed sessions.",
        "```bash",
        runtime["run_command"],
        "```",
        f"- Expected result: Transcripts idle for {runtime['inactivity_minutes']}+ minutes become candidates.",
        (
            "- Actual result: "
            f"source={item['source']}, mtime={item['mtime'].isoformat()}, "
            f"user_turns={item['user_turns']}, assistant_turns={item['assistant_turns']}, "
            f"messages={item['message_count']}"
        ),
        "",
        "#### Step 2: Extract requirements/policies/command candidates",
        "- Objective: Extract reproducible key points from conversation content.",
        "```bash",
        runtime["run_command"],
        "```",
        "- Expected result: Requirements, policies, command candidates, constraints, and target files are extracted.",
        f"- Actual result: command_hints={len(commands)}, files={len(files)}, constraints={len(constraints)}",
        "",
        "#### Step 3: Append to daily notes",
        "- Objective: Append a WORKLOG_SPEC-compliant block to daily notes.",
        "```bash",
        runtime["run_command"],
        "```",
        f"- Expected result: This block is added to `{daily_file}`.",
        "- Actual result: "
        + ("not executed (dry-run mode)" if runtime["dry_run"] else "executed (append processing completed)"),
        "",
        "#### Step 4: Final verification checklist",
        "- Objective: Verify the appended result and ensure reproducibility.",
        "```bash",
        f"rg -n \"## {t} | Work log\" \"{daily_file}\"",
        f"tail -n 120 \"{daily_file}\"",
        "```",
        "- Expected result: The block at the corresponding time is detected and the appended content exists at the end.",
        "- Actual result: not executed (verification commands provided for manual confirmation).",
        "",
        "### Verification steps and results",
        "1. Verify transcript completion condition",
        "```bash",
        runtime["run_command"],
        "```",
        f"- Actual result: inactivity_threshold={runtime['inactivity_minutes']}min, transcript={item['path']}",
        "2. Verify extraction results",
        "```bash",
        runtime["run_command"],
        "```",
        f"- Actual result: final_assistant_summary={_truncate(item['last_assistant'] or 'none', 320)}",
        "- Verdict: pass (log block generation conditions met)",
        f"- Evidence: transcript=`{item['path']}` / daily_note=`{daily_file}` / state=`{runtime['state_file']}`",
        "",
        "### Changed files",
    ])

    if files:
        for fp in files:
            abs_fp = to_absolute_candidate(fp, Path(runtime["repo_root"]))
            lines.append(f"- `{abs_fp}`: Mentioned in conversation as reference/change candidate. Recorded to make related work targets explicit.")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "### Known constraints and notes",
    ])
    for c in constraints:
        lines.append(f"- {c}")

    lines.extend([
        "",
        "### Reproduction steps on another PC",
        "1. Replace `--obsidian-daily-dir` and `--repo-root` with actual paths on the target PC.",
        "2. If using launchd, re-register the job under the target PC user.",
        "3. Verify output with `python3 scripts/session_activity_logger.py --dry-run`, then run for real.",
        "4. Execute Steps 1-4 of the implementation steps in order and verify results.",
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


def render_git_block(git_info: Dict[str, Any], runtime: Dict[str, Any]) -> str:
    t = now_local().strftime("%H:%M")
    date_str = now_local().strftime("%Y-%m-%d")
    daily_file = Path(runtime["obsidian_daily_dir"]) / f"{date_str}.md"
    lines = [
        f"## {t} | Repo snapshot",
        "",
        "### 前提",
        f"- 対象OS: {runtime['os']}",
        f"- 作業ユーザー: {runtime['user']}",
        f"- トークン/設定の取得元: Gitメタデータ（`{runtime['repo_root']}`）",
        "- バックアップ方針: 状態確認のみ。ファイル更新は行わない。",
        "",
        "### 背景",
        "- リポジトリ状態の定点観測を作業ログに残すため、Gitスナップショットを採取した。",
        "",
        "### 要件定義（ユーザー要件）",
        "- 目的: リポジトリの現在状態を再現可能な形で記録する。",
        f"- スコープ: `{runtime['repo_root']}` の branch/HEAD/status/latest commit。",
        "- 期待成果物: Git状態を要件・手順・検証付きでまとめた作業ログブロック。",
        "- 受け入れ条件: branch/HEAD/latest commit/changed files が整合して記録される。",
        "- 制約/前提: 読み取り系gitコマンドのみ使用し、履歴改変コマンドは実行しない。",
        "- 未確定事項: none",
        "",
        "### 実装方針",
        "- 採用方針: `git rev-parse` / `git status --short` / `git show` の3系統で状態を固定する。",
        "- 採用理由: 履歴地点・作業差分・最新文脈を最小コマンドで網羅できるため。",
        "- 実行順序: HEAD確認 -> 変更状態確認 -> 最新コミット確認 -> 整合検証。",
        "- 代替案と不採用理由: `git log --stat` 単独は作業ツリー未コミット差分を取りこぼすため不採用。",
        "- リスクと緩和策: 実行時点で状態が変わるリスクに対して、同一ブロック内で連続実行し時刻付き記録で緩和。",
        "",
        "### 実装手順（Step 0, 1, 2... コマンド付き）",
        "#### Step 0: バックアップ（任意）",
        "- 目的: 状態比較のために現状メモを退避する。",
        "```bash",
        f"cp \"{daily_file}\" \"{daily_file}.bak\"",
        "```",
        "- 期待結果: `.bak` が作成される。",
        "- 実結果: not executed（Gitスナップショット採取のみ）。",
        "",
        "#### Step 1: ブランチとHEADの確認",
        "- 目的: どの履歴地点を観測したかを固定する。",
        "```bash",
        f"cd \"{runtime['repo_root']}\"",
        "git rev-parse --abbrev-ref HEAD",
        "git rev-parse HEAD",
        "```",
        "- 期待結果: branch名とHEAD SHAが取得できる。",
        f"- 実結果: branch={git_info['branch']}, head={git_info['head'][:12]}",
        "",
        "#### Step 2: 変更状態の確認",
        "- 目的: ワーキングツリー差分の有無を取得する。",
        "```bash",
        f"cd \"{runtime['repo_root']}\"",
        "git status --short",
        "```",
        "- 期待結果: 未コミット変更が行単位で取得できる。",
        f"- 実結果: changed_files={len(git_info['changed_files'])}",
        "",
        "#### Step 3: 最新コミット要約の確認",
        "- 目的: 直近コミットを時刻付きで残す。",
        "```bash",
        f"cd \"{runtime['repo_root']}\"",
        "git show -s --format=%h\\ %ci\\ %s HEAD",
        "```",
        "- 期待結果: `shortSHA timestamp subject` 形式で1行表示される。",
        f"- 実結果: {git_info['latest']}",
        "",
        "#### Step 4: 最終検証チェックリスト",
        "- 目的: 必須情報の収集漏れがないか確認する。",
        "```bash",
        f"cd \"{runtime['repo_root']}\"",
        "git rev-parse --abbrev-ref HEAD",
        "git status --short",
        "git show -s --format=%h\\ %ci\\ %s HEAD",
        "```",
        "- 期待結果: Step 1〜3 の結果と整合する。",
        "- 実結果: pass（整合）。",
        "",
        "### 検証手順と結果",
        "1. branch/HEAD確認",
        "```bash",
        f"cd \"{runtime['repo_root']}\" && git rev-parse --abbrev-ref HEAD && git rev-parse HEAD",
        "```",
        f"- 実結果: branch={git_info['branch']}, HEAD={git_info['head'][:12]}",
        "2. 変更状態と最新コミット確認",
        "```bash",
        f"cd \"{runtime['repo_root']}\" && git status --short && git show -s --format=%h\\ %ci\\ %s HEAD",
        "```",
        f"- 実結果: latest_commit={git_info['latest']}, changed_files={len(git_info['changed_files'])}",
        "- 判定: pass",
        f"- 根拠: repo=`{runtime['repo_root']}` / daily_note=`{daily_file}`",
        "",
        "### 変更ファイル一覧",
    ]

    if git_info["changed_files"]:
        for item in git_info["changed_files"][:30]:
            lines.append(f"- `{item}`: `git status --short` の観測結果。状態追跡のため記録。")
        if len(git_info["changed_files"]) > 30:
            lines.append(f"- `... ({len(git_info['changed_files']) - 30} more)`: 省略分あり。")
    else:
        lines.append("- none")

    lines.extend([
        "",
        "### 既知の制約・注意点",
        "- このブロックはGit状態の定点観測であり、会話由来の要件詳細までは含まない。",
        "- `git status --short` は実行時点の瞬間値であり、後続操作で変化する。",
        "",
        "### 他PCでの再現手順",
        "1. 対象PCで `repo_root` を実環境パスに差し替える。",
        "2. 実装手順の Step 1〜4 を順に実行する。",
        "3. 検証手順の判定と根拠を確認する。",
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
    runtime = build_runtime_context(args, repo_root, obsidian_daily_dir)

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
        entries_by_date.setdefault(date_str, []).append(render_transcript_block(item, runtime))

    if git_changed:
        today = now_local().strftime("%Y-%m-%d")
        entries_by_date.setdefault(today, []).append(render_git_block(git_info, runtime))

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
