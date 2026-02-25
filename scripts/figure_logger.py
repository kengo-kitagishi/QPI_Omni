"""
figure_logger.py  -  図の保存ユーティリティ

使い方:
    from figure_logger import save_figure

    save_figure(
        fig,
        params={"pixel_size_um": 0.348, "n_medium": 1.333},
        description="mean RI計算: ellipse体積でtotal phaseを割った結果",
        script_name="32_simple_ellipse_ri",   # 省略可（自動検出）
        output_dir=None,                       # 省略時は results/figures/
    )

保存されるもの:
    results/figures/QPI_2026-02-26_32_simple_ellipse_ri_v3.png
    results/figures/QPI_2026-02-26_32_simple_ellipse_ri_v3.json  (メタデータ)
    docs/EXPERIMENT_LOG.md  (自動追記)

前回実行時との差分:
    同じスクリプト名の最後の実行記録 (.figure_history/<script>.json) と比較し、
    変わったパラメータだけをファイル名と EXPERIMENT_LOG に記録する。
"""

import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path


# =============================================================================
# パス設定
# =============================================================================
_THIS_DIR = Path(__file__).parent
_REPO_ROOT = _THIS_DIR.parent
_DEFAULT_OUTPUT_DIR = _REPO_ROOT / "results" / "figures"
_HISTORY_DIR = _REPO_ROOT / ".figure_history"
_EXPERIMENT_LOG = _REPO_ROOT / "docs" / "EXPERIMENT_LOG.md"


# =============================================================================
# ユーティリティ
# =============================================================================

def _detect_script_name() -> str:
    """呼び出し元スクリプトのファイル名（拡張子なし）を返す"""
    for frame in traceback.extract_stack():
        path = Path(frame.filename)
        if path.stem not in ("figure_logger", "<string>", "runpy"):
            return path.stem
    return "unknown"


def _load_history(script_name: str) -> dict:
    """前回のパラメータ履歴を読み込む"""
    _HISTORY_DIR.mkdir(exist_ok=True)
    history_file = _HISTORY_DIR / f"{script_name}.json"
    if history_file.exists():
        with open(history_file, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_history(script_name: str, params: dict):
    """今回のパラメータを履歴として保存する"""
    _HISTORY_DIR.mkdir(exist_ok=True)
    history_file = _HISTORY_DIR / f"{script_name}.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def _compute_diff(prev: dict, curr: dict) -> dict:
    """前回から変わったパラメータだけを返す"""
    diff = {}
    all_keys = set(prev.keys()) | set(curr.keys())
    for k in all_keys:
        if k not in prev:
            diff[k] = {"from": "(新規)", "to": curr[k]}
        elif k not in curr:
            diff[k] = {"from": prev[k], "to": "(削除)"}
        elif prev[k] != curr[k]:
            diff[k] = {"from": prev[k], "to": curr[k]}
    return diff


def _make_filename(date_str: str, script_name: str, diff: dict, version: int) -> str:
    """
    ファイル名を生成する。
    差分があればそのキーと新値をファイル名に入れる（長くなりすぎないよう最大2項目）。
    差分なし / 初回はバージョン番号のみ。
    """
    parts = [f"QPI_{date_str}_{script_name}"]

    diff_items = [(k, v["to"]) for k, v in diff.items() if v.get("to") != "(削除)"]
    for k, v in diff_items[:2]:
        safe_k = k.replace(" ", "_")
        safe_v = str(v).replace(" ", "_").replace("/", "-")
        parts.append(f"{safe_k}={safe_v}")

    parts.append(f"v{version}")
    return "_".join(parts)


def _next_version(output_dir: Path, base_name: str) -> int:
    """同日・同スクリプト名のファイルが既に何枚あるか数えてバージョンを決める"""
    existing = list(output_dir.glob(f"{base_name}_v*.png"))
    return len(existing) + 1


def _append_experiment_log(
    date_str: str,
    script_name: str,
    description: str,
    params: dict,
    diff: dict,
    fig_path: Path,
):
    """docs/EXPERIMENT_LOG.md に追記する"""
    _EXPERIMENT_LOG.parent.mkdir(exist_ok=True)

    params_str = ", ".join(f"`{k}={v}`" for k, v in params.items())

    if diff:
        diff_lines = []
        for k, v in diff.items():
            diff_lines.append(f"  - `{k}`: {v.get('from')} → **{v.get('to')}**")
        diff_str = "\n".join(diff_lines)
    else:
        diff_str = "  - (初回実行 / 前回から変更なし)"

    entry = f"""
---

## {date_str} | `{script_name}`

**説明**: {description}

**パラメータ**: {params_str}

**前回からの変更点**:
{diff_str}

**図ファイル**: `{fig_path.relative_to(_REPO_ROOT).as_posix()}`
"""

    with open(_EXPERIMENT_LOG, "a", encoding="utf-8") as f:
        f.write(entry)


# =============================================================================
# メイン関数
# =============================================================================

def save_figure(
    fig,
    params: dict,
    description: str,
    script_name: str = None,
    output_dir=None,
    dpi: int = 150,
    fmt: str = "png",
) -> Path:
    """
    matplotlib Figure を保存し、EXPERIMENT_LOG.md に記録する。

    Parameters
    ----------
    fig         : matplotlib.figure.Figure
    params      : このスクリプトの主要パラメータ dict
    description : この図が何を示しているかの日本語説明
    script_name : 省略時は呼び出し元ファイル名を自動検出
    output_dir  : 省略時は results/figures/
    dpi         : 解像度（デフォルト150）
    fmt         : 保存フォーマット（デフォルト"png"）

    Returns
    -------
    Path : 保存されたファイルのパス
    """
    if script_name is None:
        script_name = _detect_script_name()

    output_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 前回との差分を計算
    prev_params = _load_history(script_name)
    diff = _compute_diff(prev_params, params)

    # ファイル名のベース部分（バージョン番号なし）
    base_name_no_ver = "_".join(
        [f"QPI_{date_str}_{script_name}"]
        + [
            f"{k.replace(' ', '_')}={str(v['to']).replace(' ', '_').replace('/', '-')}"
            for k, v in list(diff.items())[:2]
            if v.get("to") != "(削除)"
        ]
    )

    version = _next_version(output_dir, base_name_no_ver)
    filename = f"{base_name_no_ver}_v{version}.{fmt}"
    fig_path = output_dir / filename

    # 図を保存
    fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")
    print(f"[figure_logger] 保存: {fig_path}")

    # メタデータJSON
    meta = {
        "date": date_str,
        "script": script_name,
        "description": description,
        "params": params,
        "diff_from_last": diff,
        "file": str(fig_path),
    }
    meta_path = fig_path.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 履歴を更新
    _save_history(script_name, params)

    # EXPERIMENT_LOG.md に追記
    _append_experiment_log(date_str, script_name, description, params, diff, fig_path)
    print(f"[figure_logger] EXPERIMENT_LOG.md に記録しました")

    return fig_path
