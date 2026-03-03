"""
figure_logger.py - QPI figure save utility (inbox-first).

Design goals:
- Save every generated figure at execution time into an inbox-style archive.
- Keep reproducible metadata (params, git snapshot, run_id, caller context).
- Keep backwards compatibility for existing scripts using save_figure().
- Separate "save everything" from "adopt into figure-hub".

Typical usage:
    from figure_logger import save_figure

    save_figure(
        fig,
        params={"pixel_size_um": 0.348, "n_medium": 1.333},
        description="mean RI calculation",
    )

What gets written for each call:
- inbox image:
    <inbox_root>/YYYY-MM-DD/<script>/<run_id>/<script>__<run_id>__fNNN.<fmt>
- inbox metadata json (same basename + .json)
- append-only shared manifest:
    <inbox_root>/_manifest/figure_inbox_manifest.jsonl
- session trace (local machine):
    .figure_history/session_YYYY-MM-DD.json
- experiment log entry:
    docs/EXPERIMENT_LOG.md
- optional legacy publish copy (default enabled):
    results/figures/QPI_<date>_<script>_vN.<fmt>

Environment knobs:
- QPI_FIGURE_INBOX_ROOT: override inbox root path.
- QPI_FIGURE_LOGGER_PUBLISH=0: disable legacy publish copy by default.
- QPI_FIGURE_LOGGER_NOTION=1: enable Notion logging per save.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import traceback
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# =============================================================================
# Paths / constants
# =============================================================================
_THIS_DIR = Path(__file__).parent
_REPO_ROOT = _THIS_DIR.parent
_HISTORY_DIR = _REPO_ROOT / ".figure_history"
_EXPERIMENT_LOG = _REPO_ROOT / "docs" / "EXPERIMENT_LOG.md"
_DEFAULT_PUBLISH_DIR = _REPO_ROOT / "results" / "figures"

# Shared inbox candidates (priority: env > Drive shared > Desktop mirror > local fallback).
_DEFAULT_DRIVE_HUB_INBOX = Path(
    "/Users/kitak/Library/CloudStorage/GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox"
)
_DEFAULT_DESKTOP_HUB_INBOX = Path("/Users/kitak/Desktop/figure-hub/inbox")
_FALLBACK_LOCAL_INBOX = _REPO_ROOT / "results" / "figure_inbox"
# Windows: Google Drive for Desktop（共有ドライブ）
_WINDOWS_DRIVE_HUB_INBOXES = [
    Path(f"{letter}:/共有ドライブ/wakamotolab_meeting/kitagishi/figure-hub/inbox")
    for letter in ["G", "H", "F", "I"]
]
_SHARED_MANIFEST_BASENAME = "figure_inbox_manifest.jsonl"

NOTION_DB_ID = "312eda96228e81659726cd75b221357a"

_RUN_CONTEXT = {
    "script": None,
    "run_id": None,
    "count": 0,
}
_INBOX_ENV_WARNING_SHOWN = False


# =============================================================================
# Utilities
# =============================================================================
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_token(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return out.strip("_") or "unknown"


def _looks_usable_inbox_root(path: Path) -> bool:
    """Return True if the inbox root path looks usable on this machine."""
    try:
        expanded = path.expanduser()
        return expanded.exists() or expanded.parent.exists()
    except Exception:
        return False


def _resolve_inbox_root() -> Path:
    global _INBOX_ENV_WARNING_SHOWN

    env = os.environ.get("QPI_FIGURE_INBOX_ROOT", "").strip()
    if env:
        env_path = Path(env).expanduser()
        if _looks_usable_inbox_root(env_path):
            return env_path.resolve()
        if not _INBOX_ENV_WARNING_SHOWN:
            print(
                "[figure_logger] warn: QPI_FIGURE_INBOX_ROOT is unusable; "
                "falling back to default inbox candidates.",
                file=sys.stderr,
            )
            _INBOX_ENV_WARNING_SHOWN = True

    candidates = [
        _DEFAULT_DRIVE_HUB_INBOX,
        _DEFAULT_DESKTOP_HUB_INBOX,
    ] + _WINDOWS_DRIVE_HUB_INBOXES
    for c in candidates:
        if _looks_usable_inbox_root(c):
            return c.expanduser().resolve()
    return _FALLBACK_LOCAL_INBOX.expanduser().resolve()


def _manifest_path(inbox_root: Path) -> Path:
    return inbox_root / "_manifest" / _SHARED_MANIFEST_BASENAME


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _detect_script_name() -> str:
    """Return caller script stem (without extension)."""
    for frame in traceback.extract_stack():
        path = Path(frame.filename)
        stem = path.stem
        if stem not in {"figure_logger", "<string>", "runpy"}:
            return _sanitize_token(stem)
    return "unknown"


def _detect_caller_file() -> str:
    for frame in inspect.stack():
        p = Path(frame.filename).resolve()
        if p != Path(__file__).resolve():
            return str(p)
    return ""


def _new_run_id(script_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:6]
    return f"{_sanitize_token(script_name)}_{ts}_{suffix}"


def _ensure_run_context(script_name: str) -> tuple[str, int]:
    """
    Keep a process-local run_id per script invocation session.
    Returns (run_id, figure_index starting from 1).
    """
    if _RUN_CONTEXT["script"] != script_name or _RUN_CONTEXT["run_id"] is None:
        _RUN_CONTEXT["script"] = script_name
        _RUN_CONTEXT["run_id"] = _new_run_id(script_name)
        _RUN_CONTEXT["count"] = 0

    _RUN_CONTEXT["count"] += 1
    return _RUN_CONTEXT["run_id"], int(_RUN_CONTEXT["count"])


def _load_history(script_name: str) -> dict:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    p = _HISTORY_DIR / f"{script_name}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_history(script_name: str, params: dict) -> None:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    p = _HISTORY_DIR / f"{script_name}.json"
    p.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")


def _compute_diff(prev: dict, curr: dict) -> dict:
    diff = {}
    all_keys = set(prev.keys()) | set(curr.keys())
    for key in sorted(all_keys):
        if key not in prev:
            diff[key] = {"from": "(new)", "to": curr[key]}
        elif key not in curr:
            diff[key] = {"from": prev[key], "to": "(deleted)"}
        elif prev[key] != curr[key]:
            diff[key] = {"from": prev[key], "to": curr[key]}
    return diff


def _git_snapshot() -> dict:
    try:
        commit = (
            subprocess.check_output(
                ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        commit = ""

    try:
        status = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        dirty = bool(status.strip())
        changed_files = [line[3:] for line in status.splitlines() if len(line) >= 4][:50]
    except Exception:
        dirty = False
        changed_files = []

    return {
        "commit": commit,
        "dirty": dirty,
        "changed_files": changed_files,
    }


def _detect_data_info() -> dict:
    """
    Try to infer data lineage from caller globals.
    Non-fatal: returns {} on failure.
    """
    try:
        caller_globals: dict[str, Any] = {}
        for frame_info in inspect.stack():
            if Path(frame_info.filename).resolve() != Path(__file__).resolve():
                caller_globals = frame_info.frame.f_globals
                break

        path_keywords = {"DIR", "PATH", "CSV", "FILE", "DATA", "INPUT", "ROOT"}
        candidate_paths = {
            key: str(value)
            for key, value in caller_globals.items()
            if isinstance(value, (str, Path))
            and any(kw in key.upper() for kw in path_keywords)
            and ("/" in str(value) or "\\" in str(value))
        }

        date_pat = re.compile(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})")
        measured_on = None
        for value in candidate_paths.values():
            m = date_pat.search(value)
            if m:
                measured_on = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                break

        proc_keywords = [
            "bg_corr",
            "subtracted",
            "aligned",
            "corrected",
            "filtered",
            "registered",
            "cropped",
            "denoised",
        ]
        processing = []
        for value in candidate_paths.values():
            low = value.lower()
            for kw in proc_keywords:
                if kw in low and kw not in processing:
                    processing.append(kw)

        source_path = min(candidate_paths.values(), key=len, default="")

        result = {}
        if source_path:
            result["source"] = Path(source_path).name
        if measured_on:
            result["measured_on"] = measured_on
        if processing:
            result["processing"] = " -> ".join(processing)
        return result
    except Exception:
        return {}


def _json_dump(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _append_manifest(record: dict, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_session_log(record: dict) -> None:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    date_str = record.get("date_local", datetime.now().strftime("%Y-%m-%d"))
    session_file = _HISTORY_DIR / f"session_{date_str}.json"
    if session_file.exists():
        try:
            entries = json.loads(session_file.read_text(encoding="utf-8"))
            if not isinstance(entries, list):
                entries = []
        except Exception:
            entries = []
    else:
        entries = []

    entries.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "script": record.get("script", "unknown"),
            "run_id": record.get("run_id", ""),
            "figure_index": record.get("figure_index", 0),
            "description": record.get("description", ""),
            "inbox_file": record.get("inbox_file", ""),
            "published_file": record.get("published_file", ""),
        }
    )
    session_file.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_publish_version(output_dir: Path, date_str: str, script_name: str, fmt: str) -> int:
    prefix = f"QPI_{date_str}_{script_name}_v"
    max_v = 0
    for p in output_dir.glob(f"QPI_{date_str}_{script_name}_v*.{fmt}"):
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        tail = stem[len(prefix) :]
        if tail.isdigit():
            max_v = max(max_v, int(tail))
    return max_v + 1


def _append_experiment_log(record: dict) -> None:
    _EXPERIMENT_LOG.parent.mkdir(parents=True, exist_ok=True)

    params = record.get("params", {})
    params_str = ", ".join(f"`{k}={v}`" for k, v in params.items()) or "(none)"

    diff = record.get("diff_from_last", {})
    if diff:
        diff_lines = [
            f"  - `{k}`: {v.get('from')} -> **{v.get('to')}**"
            for k, v in sorted(diff.items())
        ]
        diff_str = "\n".join(diff_lines)
    else:
        diff_str = "  - (first run / no change)"

    data_info = record.get("data_info", {})
    data_info_line = ""
    if data_info:
        parts = []
        if "raw_files" in data_info:
            files = data_info["raw_files"]
            if isinstance(files, list):
                for f in files:
                    parts.append(f"raw=`{f}`")
            else:
                parts.append(f"raw=`{files}`")
        if "measured_on" in data_info:
            parts.append(f"measured_on=`{data_info['measured_on']}`")
        if "sample_id" in data_info:
            parts.append(f"sample=`{data_info['sample_id']}`")
        if "run_id_data" in data_info:
            parts.append(f"data_run=`{data_info['run_id_data']}`")
        if "source" in data_info:
            parts.append(f"source=`{data_info['source']}`")
        if "processing" in data_info:
            parts.append(f"processing=`{data_info['processing']}`")
        if "notes" in data_info:
            parts.append(f"notes=`{data_info['notes']}`")
        if parts:
            data_info_line = f"\n**データ来歴**: {' / '.join(parts)}\n"

    inbox_rel = _to_rel_or_abs(Path(record["inbox_file"]))
    published_file = record.get("published_file", "")
    if published_file:
        published_rel = _to_rel_or_abs(Path(published_file))
    else:
        published_rel = "(disabled)"

    entry = f"""
---

## {record['date_local']} | `{record['script']}` | run `{record['run_id']}`

**説明**: {record.get('description', '')}
{data_info_line}
**パラメータ**: {params_str}

**前回からの変更点**:
{diff_str}

**Inbox**: `{inbox_rel}`
**Published**: `{published_rel}`
"""

    with _EXPERIMENT_LOG.open("a", encoding="utf-8") as f:
        f.write(entry)


def _to_rel_or_abs(path: Path) -> str:
    try:
        return path.resolve().relative_to(_REPO_ROOT.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


# =============================================================================
# Optional Notion logging
# =============================================================================
def _load_notion_token() -> str:
    mcp_path = _REPO_ROOT / ".cursor" / "mcp.json"
    if not mcp_path.exists():
        return ""
    try:
        config = json.loads(mcp_path.read_text(encoding="utf-8"))
        headers_str = (
            config.get("mcpServers", {})
            .get("notionApi", {})
            .get("env", {})
            .get("OPENAPI_MCP_HEADERS", "{}")
        )
        headers = json.loads(headers_str)
        return headers.get("Authorization", "").replace("Bearer ", "").strip()
    except Exception:
        return ""


def _notion_rt(text: str) -> list:
    return [{"type": "text", "text": {"content": str(text)[:2000]}}]


def _find_today_note_page(token: str, date_str: str, script: str) -> Optional[str]:
    """当日 + 同スクリプトのページを探して page_id を返す。なければ None。"""
    try:
        url = f"https://api.notion.com/v1/databases/{NOTION_DB_ID}/query"
        payload = {"filter": {"property": "Date", "date": {"equals": date_str}}}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Notion-Version": "2022-06-28",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())

        for page in result.get("results", []):
            props = page.get("properties", {})
            title_prop = props.get("Name", props.get("title", {}))
            page_title = "".join(
                t.get("plain_text", "") for t in title_prop.get("title", [])
            )
            if script in page_title:
                return page["id"]
    except Exception:
        return None
    return None


def _build_figure_blocks(record: dict) -> list:
    """1図分のNotion blocksを組み立てる。"""
    time_str = datetime.now().strftime("%H:%M")
    description = record.get("description", "")
    params = record.get("params", {})
    diff = record.get("diff_from_last", {})
    data_info = record.get("data_info", {})
    git_info = record.get("git", {})

    blocks: list = [
        {"object": "block", "type": "divider", "divider": {}},
        {
            "object": "block", "type": "heading_3",
            "heading_3": {"rich_text": _notion_rt(
                f"{time_str}  f{record.get('figure_index', 0):03d}  {description}"
            )},
        },
    ]

    # params
    if params:
        params_text = "  |  ".join(f"{k} = {v}" for k, v in params.items())
        blocks.append({
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _notion_rt(f"params:  {params_text}")},
        })

    # diff from last run
    if diff:
        diff_lines = "  /  ".join(
            f"{k}:  {v.get('from')} → {v.get('to')}" for k, v in diff.items()
        )
        blocks.append({
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _notion_rt(f"変更:  {diff_lines}")},
        })

    # data_source / data_info
    if data_info:
        raw_files = data_info.get("raw_files", [])
        if isinstance(raw_files, str):
            raw_files = [raw_files]
        for f in raw_files[:5]:
            blocks.append({
                "object": "block", "type": "paragraph",
                "paragraph": {"rich_text": _notion_rt(f"raw:  {f}")},
            })
        for key, label in [
            ("measured_on", "measured"),
            ("sample_id",   "sample"),
            ("run_id_data", "data_run"),
            ("notes",       "note"),
        ]:
            if key in data_info:
                blocks.append({
                    "object": "block", "type": "paragraph",
                    "paragraph": {"rich_text": _notion_rt(f"{label}:  {data_info[key]}")},
                })

    # git
    commit = git_info.get("commit", "")
    if commit:
        dirty = " (dirty)" if git_info.get("dirty") else ""
        blocks.append({
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _notion_rt(f"git:  {commit[:10]}{dirty}")},
        })

    # inbox path
    inbox = record.get("inbox_file", "")
    if inbox:
        blocks.append({
            "object": "block", "type": "paragraph",
            "paragraph": {"rich_text": _notion_rt(f"inbox:  {inbox}")},
        })

    return blocks


def _save_to_notion(record: dict) -> None:
    token = _load_notion_token()
    if not token:
        return

    try:
        date_str = record["date_local"]
        script = record["script"]
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }
        figure_blocks = _build_figure_blocks(record)

        # 当日同スクリプトのページがあれば追記
        existing_id = _find_today_note_page(token, date_str, script)
        if existing_id:
            payload = {"children": figure_blocks}
            req = urllib.request.Request(
                f"https://api.notion.com/v1/blocks/{existing_id}/children",
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="PATCH",
            )
            with urllib.request.urlopen(req, timeout=10):
                pass
            print(f"[figure_logger] Notion追記: https://www.notion.so/{existing_id.replace('-', '')}")
            return

        # なければ新規ページ作成
        payload = {
            "parent": {"database_id": NOTION_DB_ID},
            "properties": {
                "Name":        {"title": _notion_rt(f"{script}  {date_str}")},
                "Date":        {"date": {"start": date_str}},
                "Script":      {"rich_text": _notion_rt(script)},
                "Description": {"rich_text": _notion_rt(record.get("description", ""))},
                "Type":        {"select": {"name": "作業ログ"}},
                "WorkType":    {"select": {"name": "figure"}},
            },
            "children": figure_blocks,
        }
        req = urllib.request.Request(
            "https://api.notion.com/v1/pages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())
        print(f"[figure_logger] Notion保存: {result.get('url', '')}")
    except Exception as e:
        print(f"[figure_logger] Notion保存スキップ ({e})")


# =============================================================================
# Public API
# =============================================================================
def setup_autosave(description: str = "") -> None:
    """
    Call once at the top of a script to automatically save all open figures
    whenever plt.show() is called.

    Usage:
        from figure_logger import setup_autosave
        setup_autosave()                        # description auto-generated
        setup_autosave("phase analysis")        # custom description prefix
    """
    import matplotlib.pyplot as _plt

    _original_show = _plt.show
    _caller_script = _detect_script_name()

    def _autosave_show(*args, **kwargs):
        for fignum in _plt.get_fignums():
            fig = _plt.figure(fignum)
            desc = description or f"figure {fignum}"
            try:
                save_figure(fig, description=desc, script_name=_caller_script)
            except Exception as e:
                print(f"[figure_logger] autosave failed (fig {fignum}): {e}")
        _original_show(*args, **kwargs)

    _plt.show = _autosave_show
    print(f"[figure_logger] autosave enabled ({_caller_script})")


def save_figure(
    fig,
    params: Optional[dict] = None,
    description: str = "",
    script_name: Optional[str] = None,
    output_dir: Optional[os.PathLike | str] = None,
    dpi: int = 150,
    fmt: str = "png",
    publish: Optional[bool] = None,
    save_to_notion: Optional[bool] = None,
    extra_meta: Optional[dict] = None,
    data_source: Optional[dict] = None,
) -> Path:
    """
    Save a matplotlib Figure with inbox-first workflow.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object.
    params : dict, optional
        Key parameters for reproducibility.
    description : str
        Human-readable description.
    script_name : str, optional
        Caller script name. Auto-detected if omitted.
    output_dir : path-like, optional
        Legacy publish directory. Default: results/figures.
    dpi : int
        Render dpi.
    fmt : str
        Image format extension (png/pdf/svg ...).
    publish : bool, optional
        Save legacy published copy. If None, reads env QPI_FIGURE_LOGGER_PUBLISH (default True).
    save_to_notion : bool, optional
        Whether to append to Notion [図] page. If None, reads env QPI_FIGURE_LOGGER_NOTION (default False).
    extra_meta : dict, optional
        Additional metadata to include in per-figure JSON.
    data_source : dict, optional
        Raw data provenance. Explicit values override auto-detection.
        Recommended keys (all optional):
          "raw_files"   : list[str]  - absolute paths to raw measurement files
          "measured_on" : str        - measurement date (YYYY-MM-DD)
          "sample_id"   : str        - sample identifier
          "run_id_data" : str        - data acquisition run ID
          "notes"       : str        - free-form provenance notes

        Example::
            save_figure(
                fig,
                data_source={
                    "raw_files": ["/data/2026-02-28/sample01.nd2"],
                    "measured_on": "2026-02-28",
                    "sample_id": "S001",
                },
            )

    Returns
    -------
    pathlib.Path
        Published file path when publish=True, else inbox file path.
    """
    params = params or {}
    if not isinstance(params, dict):
        raise TypeError("params must be a dict")

    if publish is None:
        publish = _env_bool("QPI_FIGURE_LOGGER_PUBLISH", True)
    if save_to_notion is None:
        save_to_notion = _env_bool("QPI_FIGURE_LOGGER_NOTION", False)

    script = _sanitize_token(script_name or _detect_script_name())
    caller_file = _detect_caller_file()
    run_id, fig_index = _ensure_run_context(script)

    now_local = datetime.now()
    date_local = now_local.strftime("%Y-%m-%d")
    fmt = fmt.lower().lstrip(".")

    prev = _load_history(script)
    diff = _compute_diff(prev, params)

    inbox_root = _resolve_inbox_root()
    manifest_path = _manifest_path(inbox_root)
    inbox_dir = inbox_root / date_local / script / run_id
    inbox_dir.mkdir(parents=True, exist_ok=True)

    base = f"{script}__{run_id}__f{fig_index:03d}"
    inbox_file = inbox_dir / f"{base}.{fmt}"
    fig.savefig(inbox_file, dpi=dpi, bbox_inches="tight")

    published_file = ""
    if publish:
        out = Path(output_dir).expanduser().resolve() if output_dir else _DEFAULT_PUBLISH_DIR
        out.mkdir(parents=True, exist_ok=True)
        version = _next_publish_version(out, date_local, script, fmt)
        pub_name = f"QPI_{date_local}_{script}_v{version}.{fmt}"
        pub_path = out / pub_name
        shutil.copy2(inbox_file, pub_path)
        published_file = str(pub_path)
    else:
        pub_path = None

    meta = {
        "created_at_utc": _utc_now_iso(),
        "date_local": date_local,
        "script": script,
        "caller_file": caller_file,
        "run_id": run_id,
        "figure_index": fig_index,
        "description": description,
        "params": params,
        "diff_from_last": diff,
        "format": fmt,
        "dpi": dpi,
        "inbox_file": str(inbox_file.resolve()),
        "published_file": published_file,
        "manifest_file": str(manifest_path.resolve()),
        "data_info": {**_detect_data_info(), **(data_source or {})},
        "git": _git_snapshot(),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "hostname": socket.gethostname(),
            "cwd": str(Path.cwd().resolve()),
            "repo_root": str(_REPO_ROOT.resolve()),
        },
    }
    if extra_meta:
        meta["extra_meta"] = extra_meta

    meta_path = inbox_file.with_suffix(".json")
    _json_dump(meta_path, meta)

    _save_history(script, params)
    _append_manifest(meta, manifest_path)
    _append_session_log(meta)
    _append_experiment_log(meta)

    if save_to_notion:
        _save_to_notion(meta)

    print(f"[figure_logger] inbox saved: {inbox_file}")
    if published_file:
        print(f"[figure_logger] published copy: {published_file}")
    print(f"[figure_logger] manifest: {manifest_path}")

    return Path(published_file) if published_file else inbox_file
