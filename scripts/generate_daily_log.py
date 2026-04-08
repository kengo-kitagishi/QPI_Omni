#!/usr/bin/env python3
"""generate_daily_log.py

日次索引ファイル（daily_index_YYYY-MM-DD.md）を読み込み、
Claude を使って日次ログを自動生成する。

デフォルトは複数ファイル出力（ハブページ + トピックページ）。
--single-file で旧フォーマット（単一ファイル）に戻せる。

バックエンドを自動検出して切り替える:
  1. claude CLI (claude.ai OAuth) が利用可能なら優先
  2. ANTHROPIC_API_KEY / ~/.anthropic_api_key があれば API キーを使う

使い方:
  python3 scripts/generate_daily_log.py              # 今日の日次ログを生成
  python3 scripts/generate_daily_log.py --date 2026-03-26  # 指定日
  python3 scripts/generate_daily_log.py --date 2026-03-26 --model claude-opus-4-6
  python3 scripts/generate_daily_log.py --run-preprocess    # 前処理（索引更新）も実行
  python3 scripts/generate_daily_log.py --single-file       # 旧フォーマット（単一ファイル）
  python3 scripts/generate_daily_log.py --backend api       # APIキー強制使用
  python3 scripts/generate_daily_log.py --backend cli       # claude CLI 強制使用

出力先:
  ~/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD.md          ← ハブページ
  ~/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD_トピック.md ← 各トピックページ
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
# 設定
# ────────────────────────────────────────────

OBSIDIAN_ROOT = Path("/Users/kitak/Documents/Obsidian Vault")
DAILY_INDEX_DIR = OBSIDIAN_ROOT / "00_Inbox"
DAILY_LOG_DIR = OBSIDIAN_ROOT / "01_Daily"
SCRIPTS_DIR = Path(__file__).resolve().parent

# スタイル参照ファイル（WEEKLY_LOG_SPEC は廃止、style.md を使う）
STYLE_FILES = [
    Path.home() / ".claude/skills/gm-log-compiler/references/gm_style.md",
    Path.home() / ".claude/skills/daily-log/references/daily_log_style.md",
]

DEFAULT_MODEL = "claude-opus-4-6"
JST = timezone(timedelta(hours=9))

# ────────────────────────────────────────────
# システムプロンプト（複数ファイル出力モード）
# ────────────────────────────────────────────

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
# システムプロンプト（単一ファイルモード、後方互換）
# ────────────────────────────────────────────

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
# プロンプト構築
# ────────────────────────────────────────────

def build_user_prompt(target_date: str, style_text: str, index_text: str) -> str:
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
# 複数ファイル出力のパース
# ────────────────────────────────────────────

def parse_multi_file_output(text: str) -> dict[str, str]:
    """<<FILE: name.md>> ... <<ENDFILE>> ブロックを解析して {filename: content} を返す。"""
    pattern = re.compile(r'<<FILE:\s*(.+?)\s*>>\n(.*?)<<ENDFILE>>', re.DOTALL)
    return {m.group(1).strip(): m.group(2).rstrip('\n') for m in pattern.finditer(text)}


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────

def run_preprocess(target_date: str) -> None:
    """weekly_report_hub.py を実行して daily_index を更新する。"""
    d = date.fromisoformat(target_date)
    week_label = d.strftime("%Y-W%V")
    print(f"前処理: weekly_report_hub.py --week {week_label} ...")
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "weekly_report_hub.py"), "--week", week_label],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"WARNING: 前処理に失敗しました:\n{result.stderr[-500:]}")
    else:
        print(result.stdout[-300:])


def detect_backend(force: str = "auto") -> tuple[str, str]:
    """使用するバックエンドを検出して (backend, reason) を返す。"""
    if force == "cli":
        if shutil.which("claude"):
            return "cli", "cli 強制指定"
        return "none", "--backend cli を指定しましたが claude コマンドが見つかりません"

    if force == "api":
        api_key, reason = _find_api_key()
        if api_key:
            return "api", f"api 強制指定 ({reason})"
        return "none", "--backend api を指定しましたが API キーが見つかりません"

    # auto: claude CLI を優先
    if shutil.which("claude"):
        return "cli", "claude CLI (claude.ai OAuth)"

    api_key, reason = _find_api_key()
    if api_key:
        return "api", reason

    return "none", "claude CLI も API キーも見つかりません"


def _find_api_key() -> tuple[str, str]:
    """(api_key, source_description) を返す。見つからなければ ('', '')。"""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if key:
        return key, "ANTHROPIC_API_KEY 環境変数"
    key_file = Path.home() / ".anthropic_api_key"
    if key_file.exists():
        return key_file.read_text().strip(), "~/.anthropic_api_key"
    return "", ""


def _call_via_cli(model: str, system_prompt: str, user_prompt: str) -> tuple[str, int, str]:
    """claude -p サブプロセスで生成。(output, returncode, stderr) を返す。

    改善点:
    - 環境変数の strip リストを拡張（CLAUDE_* 全般 + ANTHROPIC_PROJECT）
    - cwd=HOME で親セッションの .claude/projects 継承を断つ
    - timeout=600 で無限ハングを防止
    - stdout + stderr の両方を返す（claude CLI はエラーを stdout に出す）
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

    # claude CLI はエラーを stdout に出すことがあるため、失敗時は両方を結合して返す
    if result.returncode != 0:
        combined_err = result.stderr
        if result.stdout and not result.stdout.strip().startswith("#"):
            combined_err = f"[stdout] {result.stdout[-2000:]}\n[stderr] {result.stderr[-2000:]}"
        return result.stdout, result.returncode, combined_err

    return result.stdout, result.returncode, result.stderr


def _call_via_api(model: str, system_prompt: str, api_key: str, user_prompt: str) -> tuple[str, int, str]:
    """anthropic SDK でストリーミング生成。(output, returncode, stderr) を返す。"""
    try:
        import anthropic
    except ImportError:
        return "", 1, "anthropic パッケージが見つかりません。pip3 install anthropic"

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
    """バックエンドに応じてモデルを呼び出す。"""
    if detected == "cli":
        return _call_via_cli(model, system_prompt, user_prompt)
    else:
        api_key, _ = _find_api_key()
        return _call_via_api(model, system_prompt, api_key, user_prompt)


# モデルフォールバックラダー: rate limit 時に順番にダウングレード
MODEL_LADDER = ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]


def _is_rate_limit(output: str, stderr: str) -> bool:
    """出力にレート制限エラーが含まれるかチェック。"""
    combined = (output + "\n" + stderr).lower()
    return any(kw in combined for kw in ("rate limit", "429", "overloaded", "too many requests"))


def _call_with_fallback(
    detected: str, system_prompt: str, user_prompt: str, start_model: str
) -> tuple[str, int, str]:
    """モデルフォールバック付きの呼び出し。
    rate limit → 同モデルで最大2回リトライ（60s, 120s 待ち）→ 次のモデルへ降格。"""
    start_idx = 0
    for i, m in enumerate(MODEL_LADDER):
        if m == start_model:
            start_idx = i
            break
    ladder = MODEL_LADDER[start_idx:]

    for model in ladder:
        for attempt in range(3):
            print(f"  呼び出し: {model} (attempt {attempt + 1}/3)", file=sys.stderr)
            output, rc, stderr = _call_model(detected, model, system_prompt, user_prompt)

            if rc == 0 and output.strip():
                return output, 0, ""

            if _is_rate_limit(output, stderr):
                wait = 60 * (2 ** attempt)  # 60s, 120s, 240s
                print(f"  rate limit hit ({model}), retry in {wait}s ...", file=sys.stderr)
                import time
                time.sleep(wait)
                continue

            # rate limit 以外のエラー → モデル降格
            print(f"  {model} failed (rc={rc}): {(stderr or output)[-500:]}", file=sys.stderr)
            break

        print(f"  → fallback to next model", file=sys.stderr)

    return "", 1, "all models in ladder exhausted"


MAX_INPUT_CHARS = 200_000  # claude CLI が安定して受け付けるサイズ上限


def compress_daily_index(text: str, max_chars: int) -> str:
    """巨大な daily_index をタイムラインの中間部分を省略して圧縮する。

    方針:
    - セッションヘッダ（## / ### で始まる行）は保持
    - 各セッション内のタイムラインは先頭20行＋末尾20行を残し、中間を省略
    - 図情報（figure_inbox セクション）は全件保持
    """
    lines = text.splitlines()
    result = []
    in_timeline = False
    timeline_buffer = []

    def flush_timeline():
        nonlocal timeline_buffer
        if len(timeline_buffer) <= 50:
            result.extend(timeline_buffer)
        else:
            result.extend(timeline_buffer[:20])
            omitted = len(timeline_buffer) - 40
            result.append(f"[... {omitted} events omitted ...]")
            result.extend(timeline_buffer[-20:])
        timeline_buffer = []

    for line in lines:
        # セクションヘッダで timeline flush
        if line.startswith("## ") or line.startswith("### "):
            if in_timeline:
                flush_timeline()
                in_timeline = False
            result.append(line)
        elif line.startswith("#### ") and "タイムライン" in line:
            if in_timeline:
                flush_timeline()
            result.append(line)
            in_timeline = True
        elif in_timeline:
            timeline_buffer.append(line)
        else:
            result.append(line)

    if in_timeline:
        flush_timeline()

    compressed = "\n".join(result)
    if len(compressed) > max_chars:
        # それでも大きい場合は末尾を切る
        compressed = compressed[:max_chars] + "\n[... truncated to fit size limit ...]"
    return compressed


# exit code 2 = 空索引（リトライしても無意味）
EXIT_EMPTY_INDEX = 2


def _load_inputs(target_date: str) -> tuple[str, str] | None:
    """(index_text, style_text) を返す。失敗時は None。
    空索引の場合は SystemExit(2) を発生させる。"""
    index_path = DAILY_INDEX_DIR / f"daily_index_{target_date}.md"
    if not index_path.exists():
        print(f"ERROR: 日次索引が見つかりません: {index_path}")
        print("  先に weekly_report_hub.py を実行してください（または --run-preprocess を使う）。")
        return None
    index_text = index_path.read_text(encoding="utf-8")
    print(f"日次索引: {index_path.name} ({len(index_text.splitlines()):,} 行, {len(index_text):,} 文字)")

    # 空索引の早期検出
    if len(index_text.strip()) < 500:
        print(f"ERROR: 索引が空または極端に短い ({len(index_text)} chars).")
        print("  前段の jsonl_to_obsidian / weekly_report_hub が失敗している可能性があります.")
        raise SystemExit(EXIT_EMPTY_INDEX)

    # 巨大索引の圧縮
    if len(index_text) > MAX_INPUT_CHARS:
        print(f"WARN: 索引が巨大 ({len(index_text):,} chars) — {MAX_INPUT_CHARS:,} chars に圧縮します")
        index_text = compress_daily_index(index_text, MAX_INPUT_CHARS)
        print(f"  圧縮後: {len(index_text):,} chars")

    # スタイルガイドを読み込み（gm_style.md + daily_log_style.md）
    style_parts = []
    for sf in STYLE_FILES:
        if sf.exists():
            content = sf.read_text(encoding="utf-8")
            style_parts.append(f"# {sf.name}\n\n{content}")
            print(f"スタイル: {sf.name} ({len(content):,} 文字)")
        else:
            print(f"WARN: スタイルファイルが見つかりません: {sf}")
    style_text = "\n\n---\n\n".join(style_parts)
    return index_text, style_text


def generate_multi_file(target_date: str, model: str, overwrite: bool, backend: str = "auto") -> int:
    """複数ファイル出力モード（ハブページ + トピックページ）。"""
    detected, reason = detect_backend(backend)
    if detected == "none":
        print(f"ERROR: {reason}")
        return 1
    print(f"バックエンド: {detected} ({reason})")

    result = _load_inputs(target_date)
    if result is None:
        return 1
    index_text, style_text = result

    user_prompt = build_user_prompt(target_date, style_text, index_text)
    print(f"モデル: {model} | 入力: 約 {len(user_prompt):,} 文字")
    print("生成中（複数ファイルモード）...")

    output, returncode, stderr = _call_with_fallback(detected, SYSTEM_PROMPT_MULTI, user_prompt, model)
    if returncode != 0:
        print(f"ERROR: モデル呼び出し失敗:\n{stderr[-2000:]}")
        return 1

    files = parse_multi_file_output(output)
    if not files:
        # フォールバック: パースに失敗した場合は単一ファイルとして保存
        print("WARNING: <<FILE:...>><<ENDFILE>> ブロックが見つかりません。単一ファイルとして保存します。")
        out_path = DAILY_LOG_DIR / f"{target_date}.md"
        DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output, encoding="utf-8")
        print(f"保存: {out_path} ({len(output.splitlines()):,} 行)")
        return 0

    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for filename, content in files.items():
        out_path = DAILY_LOG_DIR / filename
        if out_path.exists() and not overwrite:
            ans = input(f"上書き? {out_path.name} [y/N] ").strip().lower()
            if ans != "y":
                print(f"  スキップ: {filename}")
                continue
        out_path.write_text(content, encoding="utf-8")
        saved.append((filename, len(content.splitlines()), len(content)))
        print(f"  保存: {filename} ({len(content.splitlines()):,} 行 / {len(content):,} 文字)")

    print(f"\n完了: {len(saved)} ファイル保存")
    for fname, lines, chars in saved:
        print(f"  ~/Documents/Obsidian Vault/01_Daily/{fname}")
    return 0


def generate_single_file(target_date: str, model: str, overwrite: bool, backend: str = "auto") -> int:
    """単一ファイル出力モード（後方互換）。"""
    detected, reason = detect_backend(backend)
    if detected == "none":
        print(f"ERROR: {reason}")
        return 1
    print(f"バックエンド: {detected} ({reason})")

    result = _load_inputs(target_date)
    if result is None:
        return 1
    index_text, style_text = result

    out_path = DAILY_LOG_DIR / f"{target_date}.md"
    if out_path.exists() and not overwrite:
        ans = input(f"上書きしますか？ {out_path} [y/N] ").strip().lower()
        if ans != "y":
            print("中止しました。--overwrite で強制上書きできます。")
            return 0

    user_prompt = build_user_prompt(target_date, style_text, index_text)
    print(f"モデル: {model} | 入力: 約 {len(user_prompt):,} 文字")
    print("生成中（単一ファイルモード）...")

    output, returncode, stderr = _call_with_fallback(detected, SYSTEM_PROMPT_SINGLE, user_prompt, model)
    if returncode != 0:
        print(f"ERROR: {stderr[-2000:]}")
        return 1

    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")
    print(f"保存完了: {out_path}")
    print(f"  {len(output.splitlines()):,} 行 / {len(output):,} 文字")
    return 0


# ────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    from datetime import datetime
    today = datetime.now(JST).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(
        description="日次索引 → Claude API → 日次ログ自動生成"
    )
    parser.add_argument("--date", default=today,
                        help=f"対象日 YYYY-MM-DD（デフォルト: 今日 {today}）")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"使用モデル（デフォルト: {DEFAULT_MODEL}）")
    parser.add_argument("--run-preprocess", action="store_true",
                        help="実行前に weekly_report_hub.py で索引を更新する")
    parser.add_argument("--overwrite", action="store_true",
                        help="既存ファイルを確認なしで上書き")
    parser.add_argument("--single-file", action="store_true",
                        help="旧フォーマット（単一ファイル）で出力する")
    parser.add_argument("--backend", default="auto", choices=["auto", "cli", "api"],
                        help="使用バックエンド (デフォルト: auto = claude CLI 優先)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_preprocess:
        run_preprocess(args.date)
    if args.single_file:
        raise SystemExit(generate_single_file(args.date, args.model, args.overwrite, args.backend))
    else:
        raise SystemExit(generate_multi_file(args.date, args.model, args.overwrite, args.backend))
