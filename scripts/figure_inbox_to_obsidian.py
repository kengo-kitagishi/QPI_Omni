#!/usr/bin/env python3
"""figure_inbox_to_obsidian.py

figure-hub inbox の JSON メタデータを Obsidian の .md に変換する。

- 「どのコードを変えた時に出た図か」（git.changed_files）を記録
- パラメータ・前回差分（diff_from_last）を記録
- 図ファイル（PNG）を Obsidian の 99_Assets/figures/ にコピー
- SQLite (session_db.py) に figure_events として記録

使い方:
  python3 scripts/figure_inbox_to_obsidian.py              # 新規分のみ変換
  python3 scripts/figure_inbox_to_obsidian.py --dry-run    # 確認のみ
  python3 scripts/figure_inbox_to_obsidian.py --all        # 全件再変換

出力先:
  ~/Documents/Obsidian Vault/00_Inbox/figure_inbox/YYYY-MM-DD_<run_id>.md
  ~/Documents/Obsidian Vault/99_Assets/figures/<published_basename>.png
"""

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_db

# ────────────────────────────────────────────
# 設定
# ────────────────────────────────────────────

INBOX_ROOT = Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/"
    "共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox"
)

OBSIDIAN_FIGURE_INBOX_DIR = Path(
    "/Users/kitak/Documents/Obsidian Vault/00_Inbox/figure_inbox"
)
OBSIDIAN_ASSETS_DIR = Path(
    "/Users/kitak/Documents/Obsidian Vault/99_Assets/figures"
)

JST = timezone(timedelta(hours=9))


# ────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────

def _ts_to_jst_str(ts: str) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        dt_jst = dt.astimezone(JST)
        return dt_jst.strftime("%Y-%m-%d %H:%M JST")
    except Exception:
        return ts


def _date_from_ts(ts: str) -> str:
    if not ts:
        return datetime.now(JST).strftime("%Y-%m-%d")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.astimezone(JST).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now(JST).strftime("%Y-%m-%d")


def _find_image_in_dir(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    images = []
    for ext in ("*.png", "*.tif", "*.tiff"):
        images.extend(directory.glob(ext))
    return sorted(images)


def _published_basename(meta: dict) -> str | None:
    pub = meta.get("published_file", "")
    if not pub:
        return None
    pub = pub.replace("\\", "/")
    return Path(pub).name


def _find_image_mac_path(meta: dict, json_path: Path) -> Path | None:
    fig_idx = meta.get("figure_index", 1)
    tag = f"f{fig_idx:03d}"
    parent = json_path.parent
    for ext in (".png", ".tif", ".tiff"):
        candidates = [p for p in parent.glob(f"*{tag}*{ext}")]
        if candidates:
            return candidates[0]
    images = _find_image_in_dir(parent)
    if images:
        return images[0]
    return None


# ────────────────────────────────────────────
# JSON → Markdown レンダリング
# ────────────────────────────────────────────

def render_figure_md(meta: dict, png_obsidian_name: str | None,
                     image_mac_path: Path | None = None) -> str:
    run_id = meta.get("run_id", "unknown")
    script = meta.get("script", "")
    description = meta.get("description", "")
    created_at = meta.get("created_at_utc", "")
    date_str = _date_from_ts(created_at)
    ts_jst = _ts_to_jst_str(created_at)

    params = meta.get("params", {})
    diff_from_last = meta.get("diff_from_last", {})
    git_info = meta.get("git", {})
    runtime = meta.get("runtime", {})
    data_info = meta.get("data_info", {})

    lines = []

    # ── フロントマター ──
    lines.extend([
        "---",
        "source: figure-inbox",
        f"script: {script}",
        f"run_id: {run_id}",
        f"date: {date_str}",
        f'created_at_utc: "{created_at}"',
    ])
    if git_info.get("commit"):
        lines.append(f'git_commit: "{git_info["commit"]}"')
    if git_info.get("dirty") is not None:
        lines.append(f"git_dirty: {str(git_info['dirty']).lower()}")
    if image_mac_path:
        lines.append(f'image_mac_path: "{image_mac_path}"')
    lines.append("---")
    lines.append("")

    # ── タイトル ──
    lines.append(f"# 図: {date_str} | {script}")
    lines.append("")
    if description:
        lines.append(f"**説明**: {description}")
        lines.append("")

    lines.append(f"**生成時刻**: {ts_jst}")
    lines.append("")

    # ── パラメータ ──
    if params:
        lines.append("## パラメータ")
        for k, v in params.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # ── 前回からの変更 ──
    if diff_from_last:
        lines.append("## 前回からの変更 (`diff_from_last`)")
        for k, v in diff_from_last.items():
            from_val = v.get("from", "(new)") if isinstance(v, dict) else v
            to_val = v.get("to", v) if isinstance(v, dict) else v
            lines.append(f"- `{k}`: `{from_val}` → `{to_val}`")
        lines.append("")

    # ── コード変更（git 情報） ──
    changed_files = git_info.get("changed_files", [])
    commit = git_info.get("commit", "")
    dirty = git_info.get("dirty", False)

    lines.append("## コード変更（この図を生成した時の git 状態）")
    if commit:
        dirty_mark = " *(unstaged changes あり)*" if dirty else ""
        lines.append(f"**git commit**: `{commit}`{dirty_mark}")
    if changed_files:
        lines.append("")
        lines.append("変更されていたファイル:")
        for f in changed_files:
            lines.append(f"- `{f}`")
    else:
        lines.append("（git.changed_files: なし）")
    lines.append("")

    # ── データ情報 ──
    if data_info:
        lines.append("## データ情報")
        for k, v in data_info.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # ── 実行環境 ──
    if runtime:
        hostname = runtime.get("hostname", "")
        python_ver = runtime.get("python", "")
        cwd = runtime.get("cwd", "")
        if hostname or python_ver:
            lines.append("## 実行環境")
            if hostname:
                lines.append(f"- hostname: `{hostname}`")
            if python_ver:
                lines.append(f"- Python: `{python_ver}`")
            if cwd:
                lines.append(f"- cwd: `{cwd}`")
            lines.append("")

    # ── 図の埋め込み ──
    if png_obsidian_name:
        lines.append("## 図")
        lines.append(f"![[{png_obsidian_name}]]")
        lines.append("")
        if image_mac_path:
            lines.append(f"*Claude Read用パス*: `{image_mac_path}`")
            lines.append("")

    inbox_file = meta.get("inbox_file", "").replace("\\", "/")
    if inbox_file:
        lines.append(f"*inbox path (Windows)*: `{inbox_file}`")

    lines.append("")
    lines.append("<!-- POST_FIGURE_CONTEXT -->")

    return "\n".join(lines)


# ────────────────────────────────────────────
# メイン処理
# ────────────────────────────────────────────

def find_inbox_jsons() -> list[Path]:
    if not INBOX_ROOT.exists():
        print(f"WARNING: inbox ディレクトリが見つかりません: {INBOX_ROOT}", flush=True)
        return []
    return sorted(INBOX_ROOT.glob("*/*/*/*.json"))


def process_json(json_path: Path, dry_run: bool) -> dict | None:
    """1つの inbox JSON を処理して .md ファイルを出力する。

    Returns:
        dict with unique_key, meta, md_path, image_name  (成功時)
        None  (失敗/スキップ時)
    """
    try:
        meta = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ERROR reading {json_path}: {e}")
        return None

    run_id = meta.get("run_id", json_path.stem)
    if not run_id:
        return None

    created_at = meta.get("created_at_utc", "")
    date_str = _date_from_ts(created_at)
    fig_idx = meta.get("figure_index", 1)
    unique_key = f"{run_id}_f{fig_idx:03d}"

    # 画像を Obsidian assets にコピー
    png_obsidian_name = None
    png_src = _find_image_mac_path(meta, json_path)
    obsidian_image_path = None

    if png_src and png_src.exists():
        pub_basename = _published_basename(meta)
        target_name = pub_basename if pub_basename else png_src.name
        target_path = OBSIDIAN_ASSETS_DIR / target_name

        if not dry_run:
            OBSIDIAN_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(png_src, target_path)
        png_obsidian_name = target_name
        obsidian_image_path = target_path if not dry_run else None

    md_content = render_figure_md(meta, png_obsidian_name,
                                  image_mac_path=obsidian_image_path)

    out_name = f"{date_str}_{unique_key}.md"
    out_path = OBSIDIAN_FIGURE_INBOX_DIR / out_name

    json_out_name = out_name.replace(".md", ".json")
    json_out_path = OBSIDIAN_FIGURE_INBOX_DIR / json_out_name

    if dry_run:
        print(f"  [dry-run] would write: {out_path}")
        print(f"  [dry-run] would write: {json_out_path}")
        if png_src:
            print(f"    PNG: {png_src.name} → {png_obsidian_name}")
        return {
            "unique_key": unique_key,
            "meta": meta,
            "md_path": out_path,
            "image_name": png_obsidian_name,
            "date_str": date_str,
            "json_path": json_path,
        }

    OBSIDIAN_FIGURE_INBOX_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md_content, encoding="utf-8")
    shutil.copy2(json_path, json_out_path)
    img_info = f", IMG→{png_obsidian_name}" if png_obsidian_name else ""
    print(f"  → {out_name}{img_info}, JSON→{json_out_name}")

    return {
        "unique_key": unique_key,
        "meta": meta,
        "md_path": out_path,
        "image_name": png_obsidian_name,
        "date_str": date_str,
        "json_path": json_path,
    }


def write_figure_to_db(conn, result: dict) -> None:
    """処理結果を figure_events テーブルに書き込む。"""
    meta = result["meta"]
    git_info = meta.get("git", {})
    session_db.upsert_figure_event(conn, {
        "unique_key": result["unique_key"],
        "run_id": meta.get("run_id", ""),
        "figure_index": meta.get("figure_index", 1),
        "script": meta.get("script"),
        "date": result["date_str"],
        "created_at_utc": meta.get("created_at_utc"),
        "git_commit": git_info.get("commit"),
        "git_dirty": bool(git_info.get("dirty", False)),
        "description": meta.get("description"),
        "json_path": str(result["json_path"]),
        "md_path": str(result["md_path"]),
        "image_obsidian_name": result["image_name"],
    })
    conn.commit()


def run(args: argparse.Namespace) -> int:
    conn = session_db.open_db()
    run_id = session_db.start_processing_run(conn, "figure_inbox_to_obsidian")

    json_files = find_inbox_jsons()
    print(f"inbox JSON ファイル数: {len(json_files)}")

    processed = 0
    skipped = 0

    try:
        for json_path in json_files:
            try:
                meta_quick = json.loads(json_path.read_text(encoding="utf-8"))
                run_id_val = meta_quick.get("run_id", json_path.stem)
                fig_idx = meta_quick.get("figure_index", 1)
            except Exception:
                run_id_val = json_path.stem
                fig_idx = 1

            unique_key = f"{run_id_val}_f{fig_idx:03d}"

            if not args.all and session_db.is_figure_processed(conn, unique_key):
                skipped += 1
                continue

            print(f"Processing: {json_path.parent.name}/{json_path.name}")
            try:
                result = process_json(json_path, args.dry_run)
            except Exception as e:
                print(f"  ERROR: {json_path.name} の処理で例外: {type(e).__name__}: {e}",
                      file=sys.stderr)
                traceback.print_exc()
                skipped += 1
                continue

            if result is not None:
                processed += 1
                if not args.dry_run:
                    write_figure_to_db(conn, result)
            else:
                skipped += 1

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

    print(f"\nfigure_inbox_to_obsidian: processed={processed}, skipped={skipped},"
          f" dry_run={args.dry_run}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="figure-hub inbox JSON → Obsidian .md 変換（SQLite 連携）"
    )
    parser.add_argument("--dry-run", action="store_true", help="ファイル書き出しなしで確認")
    parser.add_argument("--all", action="store_true", help="処理済み含め全件再変換")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(args))
