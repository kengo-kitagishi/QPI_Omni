#!/usr/bin/env python3
"""generate_daily_log.py

Read daily index files (daily_index_YYYY-MM-DD.md) and
auto-generate daily logs using Claude.

Default output is multi-file (hub page + topic pages).
Use --single-file to revert to legacy format (single file).

Backend auto-detection:
  1. Prefer claude CLI (claude.ai OAuth) if available
  2. Use ANTHROPIC_API_KEY / ~/.anthropic_api_key if present

Usage:
  python3 scripts/generate_daily_log.py              # Generate today's daily log
  python3 scripts/generate_daily_log.py --date 2026-03-26  # Specific date
  python3 scripts/generate_daily_log.py --date 2026-03-26 --model claude-opus-4-6
  python3 scripts/generate_daily_log.py --run-preprocess    # Also run preprocessing (index update)
  python3 scripts/generate_daily_log.py --single-file       # Legacy format (single file)
  python3 scripts/generate_daily_log.py --backend api       # Force API key backend
  python3 scripts/generate_daily_log.py --backend cli       # Force claude CLI backend

Output:
  ~/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD.md          <- hub page
  ~/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD_topic.md    <- topic pages
"""

import argparse
import re
import shutil
import subprocess
import sys
import os
from datetime import date, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────

OBSIDIAN_ROOT = Path("/Users/kitak/Documents/Obsidian Vault")
DAILY_INDEX_DIR = OBSIDIAN_ROOT / "00_Inbox"
DAILY_LOG_DIR = OBSIDIAN_ROOT / "01_Daily"
SCRIPTS_DIR = Path(__file__).resolve().parent

# Style reference files (WEEKLY_LOG_SPEC deprecated, using style.md)
STYLE_FILES = [
    Path.home() / ".claude/skills/gm-log-compiler/references/gm_style.md",
    Path.home() / ".claude/skills/daily-log/references/daily_log_style.md",
]

DEFAULT_MODEL = "claude-opus-4-6"
JST = timezone(timedelta(hours=9))

# ────────────────────────────────────────────
# System prompt (multi-file output mode)
# ────────────────────────────────────────────

# TODO-JP: daily-log template — user-facing Japanese output
SYSTEM_PROMPT_MULTI = """\
あなたは研究者のアシスタントです。
与えられた「日次索引」（その日のClaude作業セッション・生成図・Notionメモの全記録）を読み、
スタイルガイドに従って日次ログを作成してください。

## 出力形式（厳守）

**複数ファイルを以下のマーカーで区切って出力してください。他のテキストは出力しないこと。**

<<FILE: ファイル名.md>>
ファイルの内容
<<ENDFILE>>

例:
<<FILE: 2026-03-26.md>>
# 2026-03-26（木）
- [[2026-03-26_アライメント改善]] — アライメント改善でドリフト補正精度が向上
<<ENDFILE>>
<<FILE: 2026-03-26_アライメント改善.md>>
# アライメント改善

> [[2026-03-26]] に戻る

## 背景・設計
...
<<ENDFILE>>

## ファイル構成

### ハブページ（YYYY-MM-DD.md）
- トピックへのリンク一覧（`[[YYYY-MM-DD_トピック名]]` 形式）
- 各リンクに一行説明（何をやったか・結果の一言）
- 発表用・解析外の図がある場合のみ末尾に `## 発表用・整形済み図` セクション
- 短くてよい（内容はトピックページに任せる）

### トピックページ（YYYY-MM-DD_トピック名.md）
- 先頭に必ず `> [[YYYY-MM-DD]] に戻る`
- セクション構成: `## 背景・設計` / `## 実装・実行` / `## 結果` / `## 解釈・次の一手`
- **最低 5000 日本語文字**（Qiita 記事相当の厚み）
- 図は `![[QPI_YYYY-MM-DD_script_vN.png]]` 形式で埋め込み
- 各図の下にパラメータ表（data_source, 前の図からの変更, data_file, data_keys を含む）

## 基本ルール
- トピック（内容）単位でまとめる。セッションIDや実行順で区切らない
- 「設計 → 実行 → 図 → 次にこう変更」の因果の流れを記述する
- タスクリスト形式にしない
- 「まとめ」「結論」などの形式張った見出しは使わない
- 定量的に書く。情報を落とさない
- 推測・「〜と思われる」禁止。索引から確定した値のみ記載
- 作業が少ない日は短くてよい（でっち上げ禁止）
- Notion メモは「このユーザーは〇〇を考えている」と解釈して自然に織り込む
"""

# ────────────────────────────────────────────
# System prompt (single file mode, backward compatibility)
# ────────────────────────────────────────────

# TODO-JP: daily-log template — user-facing Japanese output
SYSTEM_PROMPT_SINGLE = """\
あなたは研究者のアシスタントです。
与えられた「日次索引」（その日のClaude作業セッション・生成図・Notionメモの全記録）を読み、
スタイルガイドに従って日次ログを作成してください。

## 基本ルール
- トピック（内容）単位でまとめる。セッション単位で区切らない
- 「設計 → 実行 → 図 → 次にこう変更」の因果の流れを記述する
- 1トピックあたり Qiita 記事相当の厚み（最低 2000 日本語文字）
- タスクリスト形式にしない
- 「まとめ」「結論」などの形式張った見出しは使わない
- 定量的に書く。情報を落とさない
- 図は ![[filename]] 形式で参照する
- Notion メモは「このユーザーは〇〇を考えている」と解釈して自然に織り込む
- 作業が少ない日は短くてよい（でっち上げ禁止）
"""


# ────────────────────────────────────────────
# Prompt construction
# ────────────────────────────────────────────

def build_user_prompt(target_date: str, style_text: str, index_text: str) -> str:
    # TODO-JP: daily-log template — user-facing Japanese output
    return f"""\
# 対象日: {target_date}

# スタイルガイド（日次ログの書き方）
{style_text}

---

# 日次索引（この日の全作業記録）
{index_text}

---

上記のスタイルガイドに従って、{target_date} の日次ログを日本語で作成してください。
"""


# ────────────────────────────────────────────
# Multi-file output parsing
# ────────────────────────────────────────────

def parse_multi_file_output(text: str) -> dict[str, str]:
    """Parse <<FILE: name.md>> ... <<ENDFILE>> blocks and return {filename: content}."""
    pattern = re.compile(r'<<FILE:\s*(.+?)\s*>>\n(.*?)<<ENDFILE>>', re.DOTALL)
    return {m.group(1).strip(): m.group(2).rstrip('\n') for m in pattern.finditer(text)}


# ────────────────────────────────────────────
# Main processing
# ────────────────────────────────────────────

def run_preprocess(target_date: str) -> None:
    """Run weekly_report_hub.py to update daily_index."""
    d = date.fromisoformat(target_date)
    week_label = d.strftime("%Y-W%V")
    print(f"Preprocessing: weekly_report_hub.py --week {week_label} ...")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "weekly_report_hub.py"), "--week", week_label],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"WARNING: Preprocessing failed:\n{result.stderr[-500:]}")
    else:
        print(result.stdout[-300:])


def detect_backend(force: str = "auto") -> tuple[str, str]:
    """Detect the backend to use and return (backend, reason)."""
    if force == "cli":
        if shutil.which("claude"):
            return "cli", "cli forced"
        return "none", "--backend cli specified but claude command not found"

    if force == "api":
        api_key, reason = _find_api_key()
        if api_key:
            return "api", f"api forced ({reason})"
        return "none", "--backend api specified but API key not found"

    # auto: prefer claude CLI
    if shutil.which("claude"):
        return "cli", "claude CLI (claude.ai OAuth)"

    api_key, reason = _find_api_key()
    if api_key:
        return "api", reason

    return "none", "Neither claude CLI nor API key found"


def _find_api_key() -> tuple[str, str]:
    """Return (api_key, source_description). Returns ('', '') if not found."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key, "ANTHROPIC_API_KEY environment variable"
    key_file = Path.home() / ".anthropic_api_key"
    if key_file.exists():
        return key_file.read_text().strip(), "~/.anthropic_api_key"
    return "", ""


def _call_via_cli(model: str, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
    """Generate via claude -p subprocess. Returns (output, returncode, stderr).

    Improvements:
    - Extended env strip list (CLAUDE_* general + ANTHROPIC_PROJECT)
    - cwd=HOME to avoid inheriting parent session .claude/projects
    - timeout=600 to prevent infinite hangs
    - Returns both stdout + stderr (claude CLI may output errors to stdout)
    """
    skip_prefixes = ("CLAUDECODE", "CLAUDE_CODE", "CLAUDE_", "MCP_", "ANTHROPIC_PROJECT")
    env = {k: v for k, v in os.environ.items()
           if not any(k.startswith(p) for p in skip_prefixes)}
    env["HOME"] = os.environ.get("HOME", "/Users/kitak")

    cmd = [
        "claude", "--print",
        "--model", model,
        "--system-prompt", system_prompt,
        "--no-session-persistence",
    ]
    try:
        result = subprocess.run(
            cmd, input=user_prompt,
            capture_output=True, text=True, env=env,
            cwd=str(Path.home()), timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        return "", 124, f"TIMEOUT after 600s: {e}"

    # claude CLI may output errors to stdout, so combine both on failure
    if result.returncode != 0:
        combined_err = result.stderr
        if result.stdout and not result.stdout.strip().startswith("#"):
            combined_err = f"[stdout] {result.stdout[-2000:]}\n[stderr] {result.stderr[-2000:]}"
        return result.stdout, result.returncode, combined_err

    return result.stdout, result.returncode, result.stderr


def _call_via_api(model: str, system_prompt: str, api_key: str, user_prompt: str) -> tuple[str, int, str]:
    """Generate via anthropic SDK streaming. Returns (output, returncode, stderr)."""
    try:
        import anthropic
    except ImportError:
        return "", 1, "anthropic package not found. pip3 install anthropic"

    client = anthropic.Anthropic(api_key=api_key)
    output_parts = []
    with client.messages.stream(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            output_parts.append(text)
    print("\n")
    return "".join(output_parts), 0, ""


def _call_model(detected: str, model: str, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
    """Call the model according to the detected backend."""
    if detected == "cli":
        return _call_via_cli(model, system_prompt, user_prompt)
    else:
        api_key, _ = _find_api_key()
        return _call_via_api(model, system_prompt, api_key, user_prompt)


# Model fallback ladder: downgrade in order on rate limit
MODEL_LADDER = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]


def _is_rate_limit(output: str, stderr: str) -> bool:
    """Check if output contains a rate limit error."""
    combined = (output + "\n" + stderr).lower()
    return any(kw in combined for kw in ("rate limit", "429", "overloaded", "too many requests"))


def _call_with_fallback(
    detected: str, system_prompt: str, user_prompt: str, start_model: str
) -> tuple[str, int, str]:
    """Call with model fallback.
    rate limit -> retry same model up to 2 times (60s, 120s wait) -> downgrade to next model."""
    start_idx = 0
    for i, m in enumerate(MODEL_LADDER):
        if m == start_model:
            start_idx = i
            break
    ladder = MODEL_LADDER[start_idx:]

    for model in ladder:
        for attempt in range(3):
            print(f"  calling: {model} (attempt {attempt + 1}/3)", file=sys.stderr)
            output, rc, stderr = _call_model(detected, model, system_prompt, user_prompt)

            if rc == 0 and output.strip():
                return output, 0, ""

            if _is_rate_limit(output, stderr):
                wait = 60 * (2 ** attempt)  # 60s, 120s, 240s
                print(f"  rate limit hit ({model}), retry in {wait}s ...", file=sys.stderr)
                import time
                time.sleep(wait)
                continue

            # Non-rate-limit error -> downgrade model
            print(f"  {model} failed (rc={rc}): {(stderr or output)[-500:]}", file=sys.stderr)
            break

        print(f"  → fallback to next model", file=sys.stderr)

    return "", 1, "all models in ladder exhausted"



# exit code 2 = empty index (retrying is pointless)
EXIT_EMPTY_INDEX = 2


def _load_inputs(target_date: str) -> tuple[str, str] | None:
    """Return (index_text, style_text). Returns None on failure.
    Raises SystemExit(2) if the index is empty."""
    index_path = DAILY_INDEX_DIR / f"daily_index_{target_date}.md"
    if not index_path.exists():
        print(f"ERROR: Daily index not found: {index_path}")
        print("  Run weekly_report_hub.py first (or use --run-preprocess).")
        return None
    index_text = index_path.read_text(encoding="utf-8")
    print(f"Daily index: {index_path.name} ({len(index_text.splitlines()):,} lines, {len(index_text):,} chars)")

    # Early detection of empty index
    if len(index_text.strip()) < 500:
        print(f"ERROR: Index is empty or extremely short ({len(index_text)} chars).")
        print("  jsonl_to_obsidian / weekly_report_hub may have failed.")
        raise SystemExit(EXIT_EMPTY_INDEX)

    # Load style guides (gm_style.md + daily_log_style.md)
    style_parts = []
    for sf in STYLE_FILES:
        if sf.exists():
            content = sf.read_text(encoding="utf-8")
            style_parts.append(f"# {sf.name}\n\n{content}")
            print(f"Style: {sf.name} ({len(content):,} chars)")
        else:
            print(f"WARN: Style file not found: {sf}")
    style_text = "\n\n---\n\n".join(style_parts)
    return index_text, style_text


def generate_multi_file(target_date: str, model: str, overwrite: bool, backend: str = "auto") -> int:
    """Multi-file output mode (hub page + topic pages)."""
    detected, reason = detect_backend(backend)
    if detected == "none":
        print(f"ERROR: {reason}")
        return 1
    print(f"Backend: {detected} ({reason})")

    result = _load_inputs(target_date)
    if result is None:
        return 1
    index_text, style_text = result

    user_prompt = build_user_prompt(target_date, style_text, index_text)
    print(f"Model: {model} | Input: ~{len(user_prompt):,} chars")
    print("Generating (multi-file mode)...")

    output, returncode, stderr = _call_with_fallback(detected, SYSTEM_PROMPT_MULTI, user_prompt, model)
    if returncode != 0:
        print(f"ERROR: Model call failed:\n{stderr[-2000:]}")
        return 1

    files = parse_multi_file_output(output)
    if not files:
        # Fallback: save as single file if parsing failed
        print("WARNING: No <<FILE:...>><<ENDFILE>> blocks found. Saving as single file.")
        out_path = DAILY_LOG_DIR / f"{target_date}.md"
        DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"Saved: {out_path} ({len(output.splitlines()):,} lines)")
        return 0

    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for filename, content in files.items():
        out_path = DAILY_LOG_DIR / filename
        if out_path.exists() and not overwrite:
            ans = input(f"Overwrite? {out_path.name} [y/N] ").strip().lower()
            if ans != "y":
                print(f"  Skipped: {filename}")
                continue
        out_path.write_text(content, encoding="utf-8")
        saved.append((filename, len(content.splitlines()), len(content)))
        print(f"  Saved: {filename} ({len(content.splitlines()):,} lines / {len(content):,} chars)")

    print(f"\nDone: {len(saved)} files saved")
    for fname, lines, chars in saved:
        print(f"  ~/Documents/Obsidian Vault/01_Daily/{fname}")
    return 0


def generate_single_file(target_date: str, model: str, overwrite: bool, backend: str = "auto") -> int:
    """Single file output mode (backward compatible)."""
    detected, reason = detect_backend(backend)
    if detected == "none":
        print(f"ERROR: {reason}")
        return 1
    print(f"Backend: {detected} ({reason})")

    result = _load_inputs(target_date)
    if result is None:
        return 1
    index_text, style_text = result

    out_path = DAILY_LOG_DIR / f"{target_date}.md"
    if out_path.exists() and not overwrite:
        ans = input(f"Overwrite? {out_path} [y/N] ").strip().lower()
        if ans != "y":
            print("Cancelled. Use --overwrite to force overwrite.")
            return 0

    user_prompt = build_user_prompt(target_date, style_text, index_text)
    print(f"Model: {model} | Input: ~{len(user_prompt):,} chars")
    print("Generating (single file mode)...")

    output, returncode, stderr = _call_with_fallback(detected, SYSTEM_PROMPT_SINGLE, user_prompt, model)
    if returncode != 0:
        print(f"ERROR: {stderr[-2000:]}")
        return 1

    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"  {len(output.splitlines()):,} lines / {len(output):,} chars")
    return 0


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    from datetime import datetime
    today = datetime.now(JST).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(
        description="Daily index -> Claude API -> auto-generate daily log"
    )
    parser.add_argument("--date", default=today,
                        help=f"Target date YYYY-MM-DD (default: today {today})")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--run-preprocess", action="store_true",
                        help="Run weekly_report_hub.py to update index before generating")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files without confirmation")
    parser.add_argument("--single-file", action="store_true",
                        help="Output in legacy format (single file)")
    parser.add_argument("--backend", default="auto", choices=["auto", "cli", "api"],
                        help="Backend to use (default: auto = prefer claude CLI)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_preprocess:
        run_preprocess(args.date)
    if args.single_file:
        raise SystemExit(generate_single_file(args.date, args.model, args.overwrite, args.backend))
    else:
        raise SystemExit(generate_multi_file(args.date, args.model, args.overwrite, args.backend))
