#!/usr/bin/env python3
"""generate_daily_log.py

日次索引ファイル（daily_index_YYYY-MM-DD.md）を読み込み、
Claude を使って日次ログを自動生成する。

バックエンドを自動検出して切り替える:
  1. claude CLI (claude.ai OAuth) が利用可能なら優先
  2. ANTHROPIC_API_KEY / ~/.anthropic_api_key があれば API キーを使う

使い方:
  python3 scripts/generate_daily_log.py              # 今日の日次ログを生成
  python3 scripts/generate_daily_log.py --date 2026-03-14  # 指定日
  python3 scripts/generate_daily_log.py --date 2026-03-14 --model claude-opus-4-6
  python3 scripts/generate_daily_log.py --run-preprocess    # 前処理（索引更新）も実行
  python3 scripts/generate_daily_log.py --backend api       # APIキー強制使用
  python3 scripts/generate_daily_log.py --backend cli       # claude CLI 強制使用

出力先:
  ~/Documents/Obsidian Vault/01_Daily/YYYY-MM-DD.md
"""

import argparse
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
WEEKLY_LOG_SPEC = Path(__file__).resolve().parent.parent / "docs" / "WEEKLY_LOG_SPEC.md"
SCRIPTS_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = "claude-opus-4-6"
JST = timezone(timedelta(hours=9))

# ────────────────────────────────────────────
# プロンプト
# ────────────────────────────────────────────

SYSTEM_PROMPT = """\
あなたは研究者のアシスタントです。
与えられた「日次索引」（その日のClaude作業セッション・生成図・Notionメモの全記録）を読み、
WEEKLY_LOG_SPECに従って日次ログを作成してください。

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


def build_user_prompt(target_date: str, spec_text: str, index_text: str) -> str:
    return f"""\
# 対象日: {target_date}

# 仕様（WEEKLY_LOG_SPEC）
{spec_text}

---

# 日次索引（この日の全作業記録）
{index_text}

---

上記を読んで、{target_date} の日次ログを日本語で作成してください。
出力はそのまま Obsidian Markdown ファイルとして保存します。
フロントマターは不要です。見出しから始めてください。
"""


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────

def run_preprocess(target_date: str) -> None:
    """weekly_report_hub.py を実行して daily_index を更新する。"""
    from datetime import datetime
    # 対象日が属する週を計算
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
    """使用するバックエンドを検出して (backend, reason) を返す。

    Returns:
        ("cli", reason)  : claude -p を使う
        ("api", reason)  : anthropic SDK + API キーを使う
        ("none", reason) : 利用可能な認証が見つからない
    """
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


def _call_via_cli(model: str, user_prompt: str) -> tuple[str, int]:
    """claude -p サブプロセスで生成。(output, returncode) を返す。"""
    cmd = [
        "claude", "--print",
        "--model", model,
        "--system-prompt", SYSTEM_PROMPT,
        "--no-session-persistence",
    ]
    result = subprocess.run(cmd, input=user_prompt, capture_output=True, text=True)
    return result.stdout, result.returncode, result.stderr


def _call_via_api(model: str, api_key: str, user_prompt: str) -> tuple[str, int, str]:
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
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            output_parts.append(text)
    print("\n")
    return "".join(output_parts), 0, ""


def generate(target_date: str, model: str, overwrite: bool, backend: str = "auto") -> int:
    # バックエンド検出
    detected, reason = detect_backend(backend)
    if detected == "none":
        print(f"ERROR: {reason}")
        return 1
    print(f"バックエンド: {detected} ({reason})")

    # 日次索引ファイルを読む
    index_path = DAILY_INDEX_DIR / f"daily_index_{target_date}.md"
    if not index_path.exists():
        print(f"ERROR: 日次索引が見つかりません: {index_path}")
        print("  先に weekly_report_hub.py を実行してください。")
        return 1

    index_text = index_path.read_text(encoding="utf-8")
    index_lines = len(index_text.splitlines())
    print(f"日次索引: {index_path.name} ({index_lines:,} 行, {len(index_text):,} 文字)")

    # 仕様ファイルを読む
    spec_text = ""
    if WEEKLY_LOG_SPEC.exists():
        spec_text = WEEKLY_LOG_SPEC.read_text(encoding="utf-8")

    # 出力先チェック
    out_path = DAILY_LOG_DIR / f"{target_date}.md"
    if out_path.exists() and not overwrite:
        print(f"既存ファイルあり: {out_path}")
        ans = input("上書きしますか？ [y/N] ").strip().lower()
        if ans != "y":
            print("中止しました。--overwrite で強制上書きできます。")
            return 0

    user_prompt = build_user_prompt(target_date, spec_text, index_text)
    print(f"\nモデル: {model}")
    print(f"入力: 約 {len(user_prompt):,} 文字")
    print("生成中...")

    if detected == "cli":
        output, returncode, stderr = _call_via_cli(model, user_prompt)
        if returncode != 0:
            print(f"ERROR: claude コマンドが失敗しました:\n{stderr}")
            return 1
    else:
        api_key, _ = _find_api_key()
        output, returncode, stderr = _call_via_api(model, api_key, user_prompt)
        if returncode != 0:
            print(f"ERROR: API 呼び出しが失敗しました:\n{stderr}")
            return 1

    out_chars = len(output)
    out_lines = len(output.splitlines())

    # Obsidian に保存
    DAILY_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output, encoding="utf-8")

    print(f"保存完了: {out_path}")
    print(f"  {out_lines:,} 行 / {out_chars:,} 文字")
    return 0


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
    parser.add_argument("--backend", default="auto", choices=["auto", "cli", "api"],
                        help="使用バックエンド (デフォルト: auto = claude CLI 優先)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.run_preprocess:
        run_preprocess(args.date)
    raise SystemExit(generate(args.date, args.model, args.overwrite, args.backend))
