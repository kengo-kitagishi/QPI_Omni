#!/usr/bin/env python3
"""
clickup_helper.py - ClickUp task/schedule utility

Backward-compatible command:
    python scripts/clickup_helper.py add --name "Task" --list experiment

New verification/consulting commands:
    python scripts/clickup_helper.py doctor
    python scripts/clickup_helper.py agenda --days 7 --verify
    python scripts/clickup_helper.py advise --text "Fix figure 1" --due 2026-03-10
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo


DEFAULT_LISTS = {
    "plan": "901813997590",        # QPI > 1_PLAN
    "input": "901813997568",       # QPI > 2_INPUT
    "experiment": "901813997604",  # QPI > 3_EXPERIMENT
    "code": "901813997608",        # QPI > 4_CODE
    "manuscript": "901813997612",  # QPI > 5_MANUSCRIPT
    "slide": "901813997621",       # QPI > 6_SLIDE
    "meeting": "901813997648",     # MEETING > Academic Presentation
    "competition": "901814044181", # T&F > Competition
    "other": "901814001256",       # OTHERs > Interior
    "daily": "901814085779",       # OTHERs > Daily
}

TIME_PRESETS = {
    "morning": (9, 12),
    "afternoon": (13, 16),
    "default": (12, 14),
}

DEFAULT_FIXED_KEYWORDS = [
    "meeting",
    "会議",
    "mtg",
    "seminar",
    "授業",
    "学会",
    "大会",
    "発表",
    "診察",
    "病院",
]

DEFAULT_DEADLINE_KEYWORDS = [
    "feedback",
    "フィードバック",
    "review",
    "レビュー",
    "添削",
    "修正依頼",
    "締切",
    "締め切り",
]

DEFAULT_NO_SCHEDULE_DAY_KEYWORDS = [
    "大会",
]

PRIORITY_SCORE = {
    "urgent": 4,
    "high": 3,
    "normal": 2,
    "low": 1,
    "none": 0,
    "": 0,
}

DEFAULT_CATEGORY_ORDER = [
    "admin",
    "writing",
    "experiment",
    "figure",
    "input",
    "general",
]


def _default_category_keyword_rules() -> Dict[str, List[str]]:
    return {
        "admin": [
            "授業",
            "ta",
            "申し込み",
            "申込",
            "registration",
            "応募",
            "申請",
            "連絡",
            "相談",
        ],
        "writing": [
            "執筆",
            "manuscript",
            "paper",
            "draft",
            "報告資料",
            "資料",
            "report",
            "slide",
            "スライド",
            "発表資料",
        ],
        "experiment": [
            "実験",
            "検証",
            "validation",
            "analysis",
            "解析",
            "autoclave",
            "オートクレーブ",
            "オークレ",
            "ac",
            "脱気",
            "degas",
            "degassing",
        ],
        "figure": [
            "figure",
            "図",
            "作図",
            "illustration",
            "凡例",
            "編集",
        ],
        "input": [
            "input",
            "inbox",
            "メモ",
            "idea",
            "later",
            "いつか",
        ],
        "general": [],
    }


def _default_category_priority() -> Dict[str, str]:
    return {
        "admin": "high",
        "writing": "high",
        "experiment": "normal",
        "figure": "normal",
        "input": "low",
        "general": "normal",
    }


def _default_category_bias_hours() -> Dict[str, int]:
    return {
        "admin": -8,
        "writing": -6,
        "experiment": 0,
        "figure": 8,
        "input": 24,
        "general": 0,
    }


def _default_priority_bias_hours() -> Dict[str, int]:
    return {
        "urgent": -24,
        "high": -8,
        "normal": 0,
        "low": 12,
    }


def _default_category_list_key() -> Dict[str, str]:
    return {
        "admin": "other",
        "writing": "manuscript",
        "experiment": "experiment",
        "figure": "manuscript",
        "input": "input",
        "general": "other",
    }


def _default_workflow_groups() -> Dict[str, List[str]]:
    return {
        "exp_code": ["experiment", "code"],
    }


def _default_workflow_stage_rules() -> List[Dict[str, object]]:
    return [
        {
            "group": "exp_code",
            "stage": "setup",
            "order": 10,
            "keywords": [
                "bonding",
                "setup",
                "準備",
                "導入",
                "install",
                "設定",
                "notion",
                "clickup",
            ],
        },
        {
            "group": "exp_code",
            "stage": "acquisition",
            "order": 20,
            "keywords": [
                "培地交換",
                "bg",
                "moving",
                "細胞",
                "取得",
                "撮影",
                "実験",
            ],
        },
        {
            "group": "exp_code",
            "stage": "processing",
            "order": 30,
            "keywords": [
                "training",
                "dataset",
                "reconstruction",
                "alignment",
                "解析",
                "analysis",
                "検証",
            ],
        },
        {
            "group": "exp_code",
            "stage": "documentation",
            "order": 40,
            "keywords": [
                "documentation",
                "report",
                "図",
                "figure",
                "まとめ",
            ],
        },
    ]


def _default_medium_flow_steps() -> List[Dict[str, object]]:
    return [
        {
            "id": "on_culture",
            "keywords": [
                "o/n culture",
                "onculture",
                "overnight culture",
                "o/n",
            ],
        },
        {
            "id": "pre_culture",
            "keywords": [
                "pre culture",
                "preculture",
                "pre-culture",
                "前培養",
            ],
        },
        {
            "id": "cell_insertion_done",
            "keywords": [
                "細胞導入完了",
                "細胞導入",
            ],
        },
        {
            "id": "change_2_2",
            "keywords": [
                "培地交換2%2%",
                "培地交換 2% 2%",
                "2%to2%",
                "2% 2%",
            ],
        },
        {
            "id": "change_2_low",
            "keywords": [
                "培地交換2%low%",
                "培地交換 2% low%",
                "2%low%",
                "2% 0.0055%",
                "2% 0.01%",
            ],
        },
        {
            "id": "change_low_0",
            "keywords": [
                "培地交換low%0%",
                "培地交換 low% 0%",
                "low%0%",
                "0.0055% 0%",
                "0.01% 0%",
                "glucose sln",
                "glucosesln",
                "mneongreen",
                "sorbitol計測",
                "sorbitol",
            ],
        },
        {
            "id": "change_0_2",
            "keywords": [
                "培地交換0%2%",
                "培地交換 0% 2%",
                "0%to2%",
                "0% 2%",
            ],
        },
    ]


def _default_medium_flow_rules() -> List[Dict[str, object]]:
    # User-fixed sequence:
    # Cell insertion done +2d -> 2%2% +2d -> 2%Low% +1d -> Low%0% +2d -> 0%2% +2d -> 2%2%
    return [
        {
            "step": "pre_culture",
            "predecessors": ["on_culture"],
            "gap_hours": 24,
        },
        {
            "step": "change_2_2",
            "predecessors": ["cell_insertion_done", "change_0_2"],
            "gap_days": 2,
        },
        {
            "step": "change_2_low",
            "predecessors": ["change_2_2"],
            "gap_days": 2,
        },
        {
            "step": "change_low_0",
            "predecessors": ["change_2_low"],
            "gap_days": 1,
        },
        {
            "step": "change_0_2",
            "predecessors": ["change_low_0"],
            "gap_days": 2,
        },
    ]


def _default_medium_flow_t0_offset_map() -> Dict[int, str]:
    # Supports names like "medium change (T0+48h)" where concentration pair is omitted.
    # Mapping is user policy: 48h->2%2%, 96h->2%Low, 120h->Low0, 168h->0%2%, 216h->2%2%.
    return {
        48: "change_2_2",
        96: "change_2_low",
        120: "change_low_0",
        168: "change_0_2",
        216: "change_2_2",
    }


def _default_split_task_rules() -> List[Dict[str, object]]:
    return [
        {
            "id": "glucose_prepare_collect",
            "list_key": "experiment",
            "keywords": [
                "glucose sln",
                "glucosesln",
                "glucose",
            ],
            "segments": [
                {"label": "prepare", "offset_minutes": 0, "duration_minutes": 10},
                {
                    "label": "collect",
                    "min_offset_minutes": 360,
                    "same_day": True,
                    "duration_minutes": 30,
                },
            ],
        },
        {
            "id": "nh4cl_prepare_collect",
            "list_key": "experiment",
            "keywords": [
                "nh4cl",
                "nh4cl作成",
                "nh4cl 作成",
            ],
            "segments": [
                {"label": "prepare", "offset_minutes": 0, "duration_minutes": 30},
                {
                    "label": "collect",
                    "min_offset_minutes": 60,
                    "same_day": True,
                    "duration_minutes": 30,
                },
            ],
        },
        {
            "id": "emm_prepare_collect",
            "list_key": "experiment",
            "keywords": [
                "emm培地",
                "emm培地作成",
                "emm 培地",
                "emm medium",
                "emm",
            ],
            "segments": [
                {"label": "prepare", "offset_minutes": 0, "duration_minutes": 30},
                {
                    "label": "collect",
                    "min_offset_minutes": 120,
                    "same_day": True,
                    "duration_minutes": 10,
                },
            ],
        },
        {
            "id": "autoclave_prepare_collect",
            "list_key": "experiment",
            "keywords": [
                "オートクレーブ",
                "オークレ",
                "チューブオークレ",
                "チューブオートクレーブ",
                "autoclave",
                "ac",
            ],
            "segments": [
                {"label": "prepare", "offset_minutes": 0, "duration_minutes": 30},
                {
                    "label": "collect",
                    "min_offset_minutes": 120,
                    "same_day": True,
                    "duration_minutes": 30,
                },
            ],
        },
        {
            "id": "degas_prepare_collect",
            "list_key": "experiment",
            "keywords": [
                "脱気",
                "degas",
                "degassing",
            ],
            "segments": [
                {"label": "prepare", "offset_minutes": 0, "duration_minutes": 15},
                {
                    "label": "collect",
                    "min_offset_minutes": 60,
                    "same_day": True,
                    "duration_minutes": 15,
                },
            ],
        },
    ]


class ClickUpApiError(RuntimeError):
    """Raised when ClickUp API request fails."""


@dataclass
class HelperConfig:
    lists: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_LISTS))
    timezone: str = "Asia/Tokyo"
    work_hours_start: int = 9
    work_hours_end: int = 21
    planning_days: int = 14
    review_horizon_days: int = 14
    review_near_days: int = 2
    review_max_findings: int = 12
    min_gap_minutes: int = 10
    default_duration_minutes: int = 120
    input_default_duration_minutes: int = 120
    max_candidates: int = 3
    monthly_horizon_days: int = 35
    carryover_days: int = 7
    deadline_warn_days: int = 3
    fixed_list_keys: List[str] = field(default_factory=lambda: ["meeting", "competition"])
    fixed_keyword_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_FIXED_KEYWORDS))
    jog_keyword_patterns: List[str] = field(default_factory=lambda: ["jog", "ジョグ"])
    wait_task_use_jog_windows: bool = True
    wait_task_jog_window_minutes: int = 180
    deadline_keyword_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_DEADLINE_KEYWORDS))
    no_schedule_day_list_keys: List[str] = field(default_factory=lambda: ["competition"])
    no_schedule_day_keyword_patterns: List[str] = field(
        default_factory=lambda: list(DEFAULT_NO_SCHEDULE_DAY_KEYWORDS)
    )
    blocked_write_list_keys: List[str] = field(default_factory=list)
    rearrange_allowed_list_keys: List[str] = field(default_factory=list)
    protect_experiment_structure: bool = True
    category_order: List[str] = field(default_factory=lambda: list(DEFAULT_CATEGORY_ORDER))
    category_keyword_rules: Dict[str, List[str]] = field(default_factory=_default_category_keyword_rules)
    category_priority: Dict[str, str] = field(default_factory=_default_category_priority)
    category_bias_hours: Dict[str, int] = field(default_factory=_default_category_bias_hours)
    category_default_list: Dict[str, str] = field(default_factory=_default_category_list_key)
    priority_bias_hours: Dict[str, int] = field(default_factory=_default_priority_bias_hours)
    workflow_enabled: bool = True
    workflow_groups: Dict[str, List[str]] = field(default_factory=_default_workflow_groups)
    workflow_stage_rules: List[Dict[str, object]] = field(default_factory=_default_workflow_stage_rules)
    workflow_gap_minutes: int = 10
    workflow_review_max_findings: int = 4
    medium_flow_enabled: bool = True
    medium_flow_steps: List[Dict[str, object]] = field(default_factory=_default_medium_flow_steps)
    medium_flow_rules: List[Dict[str, object]] = field(default_factory=_default_medium_flow_rules)
    medium_flow_t0_offset_map: Dict[int, str] = field(default_factory=_default_medium_flow_t0_offset_map)
    medium_flow_review_max_findings: int = 6
    split_task_rules: List[Dict[str, object]] = field(default_factory=_default_split_task_rules)
    memory_md_path: str = "schedule_companion_memory.md"
    memory_auto_append: bool = True
    max_pages_per_list: int = 10
    max_tasks_per_list: int = 500
    auto_discover_lists: bool = True


@dataclass
class TaskWindow:
    task_id: str
    name: str
    list_id: str
    list_key: str
    url: str
    status_name: str
    status_type: str
    priority_name: str
    start_at: Optional[datetime]
    end_at: Optional[datetime]
    due_at: Optional[datetime]
    duration_minutes: Optional[int]
    movable: bool
    reason: str = ""


@dataclass
class AgendaSnapshot:
    start_date: date
    end_date: date
    timezone: str
    fetched_tasks: List[TaskWindow]
    scheduled_tasks: List[TaskWindow]
    unscheduled_tasks: List[TaskWindow]
    warnings: List[str]


@dataclass
class SlotCandidate:
    label: str
    slot_start: datetime
    slot_end: datetime
    list_key: str
    score: float
    slack_minutes: int = 0
    reasons: List[str] = field(default_factory=list)
    moves: List[dict] = field(default_factory=list)
    segments: List[dict] = field(default_factory=list)


@dataclass
class TaskPolicy:
    category: str
    priority_name: str
    schedule_bias_hours: int
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class WorkflowTag:
    group: str
    stage: str
    order: int
    matched_keywords: List[str] = field(default_factory=list)


class ClickUpClient:
    def __init__(self, token: str, base_url: str = "https://api.clickup.com/api/v2", timeout_sec: int = 25):
        if not token:
            raise ClickUpApiError("CLICKUP_API_TOKEN is empty")
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def request(
        self,
        method: str,
        path: str,
        query: Optional[Dict[str, str]] = None,
        payload: Optional[dict] = None,
    ) -> dict:
        path = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{path}"
        if query:
            url += "?" + urllib.parse.urlencode(query)

        body = None
        headers = {"Authorization": self.token, "Content-Type": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, data=body, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = ""
            message = f"HTTP {exc.code} {exc.reason} on {method.upper()} {path}"
            if detail:
                message += f" | body={detail[:500]}"
            raise ClickUpApiError(message) from exc
        except urllib.error.URLError as exc:
            raise ClickUpApiError(f"Network error on {method.upper()} {path}: {exc}") from exc

        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ClickUpApiError(f"Invalid JSON response on {method.upper()} {path}") from exc

    def get_user(self) -> dict:
        return self.request("GET", "/user")

    def get_task(self, task_id: str) -> dict:
        return self.request("GET", f"/task/{task_id}")

    def update_task_schedule(self, task_id: str, start_at: datetime, end_at: datetime) -> dict:
        payload = {
            "start_date": to_unix_ms(start_at),
            "start_date_time": True,
            "due_date": to_unix_ms(end_at),
            "due_date_time": True,
            "time_estimate": int((end_at - start_at).total_seconds() * 1000),
        }
        return self.request("PUT", f"/task/{task_id}", payload=payload)

    def create_task(
        self,
        list_id: str,
        name: str,
        start_at: datetime,
        end_at: datetime,
        description: str = "",
    ) -> dict:
        payload = {
            "name": name,
            "start_date": to_unix_ms(start_at),
            "start_date_time": True,
            "due_date": to_unix_ms(end_at),
            "due_date_time": True,
            "time_estimate": int((end_at - start_at).total_seconds() * 1000),
        }
        if description:
            payload["description"] = description

        return self.request("POST", f"/list/{list_id}/task", payload=payload)

    def list_tasks_paginated(
        self,
        list_id: str,
        include_closed: bool,
        max_pages: int,
        max_tasks: int,
    ) -> List[dict]:
        tasks: List[dict] = []
        page = 0

        while page < max_pages and len(tasks) < max_tasks:
            data = self.request(
                "GET",
                f"/list/{list_id}/task",
                query={
                    "page": str(page),
                    "include_closed": "true" if include_closed else "false",
                    "subtasks": "true",
                },
            )
            page_tasks = data.get("tasks") or []
            if not isinstance(page_tasks, list):
                raise ClickUpApiError(
                    f"Unexpected response format: tasks is {type(page_tasks).__name__}"
                )
            tasks.extend(page_tasks)
            if data.get("last_page") is True or not page_tasks:
                break
            page += 1

        return tasks[:max_tasks]

    def list_teams(self) -> List[dict]:
        data = self.request("GET", "/team")
        teams = data.get("teams") or []
        if not isinstance(teams, list):
            raise ClickUpApiError("Unexpected response format: teams is not list")
        return teams

    def list_team_spaces(self, team_id: str) -> List[dict]:
        data = self.request("GET", f"/team/{team_id}/space")
        spaces = data.get("spaces") or []
        if not isinstance(spaces, list):
            raise ClickUpApiError("Unexpected response format: spaces is not list")
        return spaces

    def list_space_folders(self, space_id: str) -> List[dict]:
        data = self.request("GET", f"/space/{space_id}/folder")
        folders = data.get("folders") or []
        if not isinstance(folders, list):
            raise ClickUpApiError("Unexpected response format: folders is not list")
        return folders

    def list_space_lists(self, space_id: str) -> List[dict]:
        data = self.request("GET", f"/space/{space_id}/list")
        lists = data.get("lists") or []
        if not isinstance(lists, list):
            raise ClickUpApiError("Unexpected response format: lists is not list")
        return lists

    def list_folder_lists(self, folder_id: str) -> List[dict]:
        data = self.request("GET", f"/folder/{folder_id}/list")
        lists = data.get("lists") or []
        if not isinstance(lists, list):
            raise ClickUpApiError("Unexpected response format: lists is not list")
        return lists


def to_unix_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def parse_ms_to_local(value: Optional[str], tz: ZoneInfo) -> Optional[datetime]:
    if value in (None, "", "null"):
        return None
    try:
        ms = int(str(value))
    except ValueError:
        return None
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).astimezone(tz)


def parse_date_yyyy_mm_dd(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def date_range(start: date, end: date) -> Iterable[date]:
    cursor = start
    while cursor <= end:
        yield cursor
        cursor += timedelta(days=1)


def load_json_file(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def default_figure_hub_root() -> Path:
    env_value = os.environ.get("FIGURE_HUB_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return (Path.home() / "Desktop" / "figure-hub").resolve()


def choose_figure_hub_config(root: Path, config_override: str = "") -> Path:
    if config_override.strip():
        return Path(config_override).expanduser().resolve()
    config_json = root / "config.json"
    if config_json.exists():
        return config_json.resolve()
    config_local = root / "config.local.json"
    if config_local.exists():
        return config_local.resolve()
    return config_json.resolve()


def build_figfix_clickup_sync_cmd(
    script_path: Path,
    config_path: Path,
    max_tasks: int,
    due_in_days: int,
    list_key: str = "",
    only_fig_id: str = "",
    apply: bool = False,
) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "clickup-sync",
        "--max-tasks",
        str(max(1, max_tasks)),
        "--due-in-days",
        str(max(0, due_in_days)),
    ]
    if list_key:
        cmd.extend(["--list-key", list_key])
    if only_fig_id:
        cmd.extend(["--only-fig-id", only_fig_id])
    if apply:
        cmd.append("--apply")
    return cmd


def keyword_in_text(keyword: str, normalized_text: str) -> bool:
    k = (keyword or "").strip().lower()
    if not k:
        return False
    if re.search(r"[a-z0-9]", k):
        if len(k) <= 3:
            pattern = rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])"
            return re.search(pattern, normalized_text) is not None
        return k in normalized_text
    return k in normalized_text


def load_runtime_config(path_str: str) -> HelperConfig:
    cfg = HelperConfig()

    default_path = Path(__file__).resolve().parent / "clickup_helper_config.json"
    config_path = Path(path_str).expanduser().resolve() if path_str else default_path

    if not config_path.exists():
        return cfg

    raw = load_json_file(config_path)
    if not isinstance(raw, dict):
        raise RuntimeError(f"invalid config file (dict required): {config_path}")

    if isinstance(raw.get("lists"), dict):
        cfg.lists = {str(k): str(v) for k, v in raw["lists"].items()}
    if isinstance(raw.get("timezone"), str):
        cfg.timezone = raw["timezone"]
    if isinstance(raw.get("work_hours_start"), int):
        cfg.work_hours_start = raw["work_hours_start"]
    if isinstance(raw.get("work_hours_end"), int):
        cfg.work_hours_end = raw["work_hours_end"]
    if isinstance(raw.get("planning_days"), int):
        cfg.planning_days = raw["planning_days"]
    if isinstance(raw.get("review_horizon_days"), int):
        cfg.review_horizon_days = raw["review_horizon_days"]
    if isinstance(raw.get("review_near_days"), int):
        cfg.review_near_days = raw["review_near_days"]
    if isinstance(raw.get("review_max_findings"), int):
        cfg.review_max_findings = raw["review_max_findings"]
    if isinstance(raw.get("min_gap_minutes"), int):
        cfg.min_gap_minutes = raw["min_gap_minutes"]
    if isinstance(raw.get("default_duration_minutes"), int):
        cfg.default_duration_minutes = raw["default_duration_minutes"]
    if isinstance(raw.get("input_default_duration_minutes"), int):
        cfg.input_default_duration_minutes = raw["input_default_duration_minutes"]
    if isinstance(raw.get("max_candidates"), int):
        cfg.max_candidates = raw["max_candidates"]
    if isinstance(raw.get("monthly_horizon_days"), int):
        cfg.monthly_horizon_days = raw["monthly_horizon_days"]
    if isinstance(raw.get("carryover_days"), int):
        cfg.carryover_days = raw["carryover_days"]
    if isinstance(raw.get("deadline_warn_days"), int):
        cfg.deadline_warn_days = raw["deadline_warn_days"]
    if isinstance(raw.get("max_pages_per_list"), int):
        cfg.max_pages_per_list = raw["max_pages_per_list"]
    if isinstance(raw.get("max_tasks_per_list"), int):
        cfg.max_tasks_per_list = raw["max_tasks_per_list"]
    if isinstance(raw.get("auto_discover_lists"), bool):
        cfg.auto_discover_lists = raw["auto_discover_lists"]

    if isinstance(raw.get("fixed_list_keys"), list):
        cfg.fixed_list_keys = [str(x) for x in raw["fixed_list_keys"]]
    if isinstance(raw.get("fixed_keyword_patterns"), list):
        cfg.fixed_keyword_patterns = [str(x) for x in raw["fixed_keyword_patterns"]]
    if isinstance(raw.get("jog_keyword_patterns"), list):
        cfg.jog_keyword_patterns = [str(x) for x in raw["jog_keyword_patterns"]]
    if isinstance(raw.get("wait_task_use_jog_windows"), bool):
        cfg.wait_task_use_jog_windows = raw["wait_task_use_jog_windows"]
    if isinstance(raw.get("wait_task_jog_window_minutes"), int):
        cfg.wait_task_jog_window_minutes = raw["wait_task_jog_window_minutes"]
    if isinstance(raw.get("deadline_keyword_patterns"), list):
        cfg.deadline_keyword_patterns = [str(x) for x in raw["deadline_keyword_patterns"]]
    if isinstance(raw.get("no_schedule_day_list_keys"), list):
        cfg.no_schedule_day_list_keys = [str(x) for x in raw["no_schedule_day_list_keys"]]
    if isinstance(raw.get("no_schedule_day_keyword_patterns"), list):
        cfg.no_schedule_day_keyword_patterns = [str(x) for x in raw["no_schedule_day_keyword_patterns"]]
    if isinstance(raw.get("blocked_write_list_keys"), list):
        cfg.blocked_write_list_keys = [str(x) for x in raw["blocked_write_list_keys"]]
    if isinstance(raw.get("rearrange_allowed_list_keys"), list):
        cfg.rearrange_allowed_list_keys = [str(x) for x in raw["rearrange_allowed_list_keys"]]
    if isinstance(raw.get("protect_experiment_structure"), bool):
        cfg.protect_experiment_structure = raw["protect_experiment_structure"]
    if isinstance(raw.get("category_order"), list):
        cfg.category_order = [str(x) for x in raw["category_order"]]
    if isinstance(raw.get("category_keyword_rules"), dict):
        cfg.category_keyword_rules = {
            str(k): [str(v) for v in vals]
            for k, vals in raw["category_keyword_rules"].items()
            if isinstance(vals, list)
        }
    if isinstance(raw.get("category_priority"), dict):
        cfg.category_priority = {
            str(k): str(v).lower()
            for k, v in raw["category_priority"].items()
        }
    if isinstance(raw.get("category_default_list"), dict):
        cfg.category_default_list = {
            str(k): str(v)
            for k, v in raw["category_default_list"].items()
        }
    if isinstance(raw.get("category_bias_hours"), dict):
        cfg.category_bias_hours = {
            str(k): int(v)
            for k, v in raw["category_bias_hours"].items()
            if isinstance(v, int)
        }
    if isinstance(raw.get("priority_bias_hours"), dict):
        cfg.priority_bias_hours = {
            str(k): int(v)
            for k, v in raw["priority_bias_hours"].items()
            if isinstance(v, int)
        }
    if isinstance(raw.get("workflow_enabled"), bool):
        cfg.workflow_enabled = raw["workflow_enabled"]
    if isinstance(raw.get("workflow_groups"), dict):
        cfg.workflow_groups = {
            str(group): [str(x) for x in lists]
            for group, lists in raw["workflow_groups"].items()
            if isinstance(lists, list)
        }
    if isinstance(raw.get("workflow_stage_rules"), list):
        stage_rules: List[Dict[str, object]] = []
        for item in raw["workflow_stage_rules"]:
            if not isinstance(item, dict):
                continue
            group = str(item.get("group", "")).strip()
            stage = str(item.get("stage", "")).strip()
            if not group or not stage:
                continue
            order_raw = item.get("order", 999)
            try:
                order = int(order_raw)
            except (TypeError, ValueError):
                order = 999
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
            stage_rules.append(
                {
                    "group": group,
                    "stage": stage,
                    "order": order,
                    "keywords": [str(x) for x in keywords],
                }
            )
        if stage_rules:
            cfg.workflow_stage_rules = stage_rules
    if isinstance(raw.get("workflow_gap_minutes"), int):
        cfg.workflow_gap_minutes = raw["workflow_gap_minutes"]
    if isinstance(raw.get("workflow_review_max_findings"), int):
        cfg.workflow_review_max_findings = raw["workflow_review_max_findings"]
    if isinstance(raw.get("medium_flow_enabled"), bool):
        cfg.medium_flow_enabled = raw["medium_flow_enabled"]
    if isinstance(raw.get("medium_flow_steps"), list):
        steps: List[Dict[str, object]] = []
        for item in raw["medium_flow_steps"]:
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("id", "")).strip()
            keywords = item.get("keywords", [])
            if not step_id or not isinstance(keywords, list):
                continue
            steps.append(
                {
                    "id": step_id,
                    "keywords": [str(x) for x in keywords],
                }
            )
        if steps:
            cfg.medium_flow_steps = steps
    if isinstance(raw.get("medium_flow_rules"), list):
        rules: List[Dict[str, object]] = []
        for item in raw["medium_flow_rules"]:
            if not isinstance(item, dict):
                continue
            step = str(item.get("step", "")).strip()
            preds = item.get("predecessors", [])
            if not step or not isinstance(preds, list):
                continue
            rule: Dict[str, object] = {
                "step": step,
                "predecessors": [str(x) for x in preds if str(x).strip()],
            }
            for gap_key in ("gap_days", "gap_hours", "gap_minutes"):
                if gap_key not in item:
                    continue
                try:
                    rule[gap_key] = int(item.get(gap_key, 0))
                except (TypeError, ValueError):
                    continue
            rules.append(rule)
        if rules:
            cfg.medium_flow_rules = rules
    if isinstance(raw.get("medium_flow_t0_offset_map"), dict):
        offset_map: Dict[int, str] = {}
        for key, val in raw["medium_flow_t0_offset_map"].items():
            step_id = str(val).strip()
            if not step_id:
                continue
            try:
                offset = int(str(key).strip())
            except ValueError:
                continue
            if offset < 0:
                continue
            offset_map[offset] = step_id
        if offset_map:
            cfg.medium_flow_t0_offset_map = offset_map
    if isinstance(raw.get("medium_flow_review_max_findings"), int):
        cfg.medium_flow_review_max_findings = raw["medium_flow_review_max_findings"]
    if isinstance(raw.get("split_task_rules"), list):
        split_rules: List[Dict[str, object]] = []
        for item in raw["split_task_rules"]:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("id", "")).strip()
            list_key = str(item.get("list_key", "")).strip()
            keywords = item.get("keywords", [])
            segments = item.get("segments", [])
            if not rule_id or not isinstance(keywords, list) or not isinstance(segments, list):
                continue
            norm_segments: List[Dict[str, object]] = []
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                label = str(seg.get("label", "")).strip()
                try:
                    duration_minutes = int(seg.get("duration_minutes", 0))
                except (TypeError, ValueError):
                    continue
                if not label or duration_minutes <= 0:
                    continue
                same_day = bool(seg.get("same_day", False))
                offset_minutes = seg.get("offset_minutes")
                min_offset_minutes = seg.get("min_offset_minutes")
                max_offset_minutes = seg.get("max_offset_minutes")
                norm_seg: Dict[str, object] = {
                    "label": label,
                    "duration_minutes": duration_minutes,
                    "same_day": same_day,
                }
                try:
                    if offset_minutes is not None:
                        norm_seg["offset_minutes"] = int(offset_minutes)
                except (TypeError, ValueError):
                    pass
                try:
                    if min_offset_minutes is not None:
                        norm_seg["min_offset_minutes"] = int(min_offset_minutes)
                except (TypeError, ValueError):
                    pass
                try:
                    if max_offset_minutes is not None:
                        norm_seg["max_offset_minutes"] = int(max_offset_minutes)
                except (TypeError, ValueError):
                    pass
                norm_segments.append(
                    norm_seg
                )
            if not norm_segments:
                continue
            split_rules.append(
                {
                    "id": rule_id,
                    "list_key": list_key,
                    "keywords": [str(x) for x in keywords],
                    "segments": norm_segments,
                }
            )
        if split_rules:
            cfg.split_task_rules = split_rules
    if isinstance(raw.get("memory_md_path"), str):
        cfg.memory_md_path = raw["memory_md_path"]
    if isinstance(raw.get("memory_auto_append"), bool):
        cfg.memory_auto_append = raw["memory_auto_append"]

    return cfg


def discover_workspace_lists(client: ClickUpClient) -> List[dict]:
    entries: List[dict] = []
    teams = client.list_teams()

    for team in teams:
        team_id = str(team.get("id", ""))
        team_name = str(team.get("name", ""))
        if not team_id:
            continue
        spaces = client.list_team_spaces(team_id)
        for space in spaces:
            space_id = str(space.get("id", ""))
            space_name = str(space.get("name", ""))
            if not space_id:
                continue

            for lst in client.list_space_lists(space_id):
                entries.append(
                    {
                        "team_id": team_id,
                        "team_name": team_name,
                        "space_id": space_id,
                        "space_name": space_name,
                        "folder_id": "",
                        "folder_name": "",
                        "list_id": str(lst.get("id", "")),
                        "list_name": str(lst.get("name", "")),
                    }
                )

            for folder in client.list_space_folders(space_id):
                folder_id = str(folder.get("id", ""))
                folder_name = str(folder.get("name", ""))
                if not folder_id:
                    continue
                for lst in client.list_folder_lists(folder_id):
                    entries.append(
                        {
                            "team_id": team_id,
                            "team_name": team_name,
                            "space_id": space_id,
                            "space_name": space_name,
                            "folder_id": folder_id,
                            "folder_name": folder_name,
                            "list_id": str(lst.get("id", "")),
                            "list_name": str(lst.get("name", "")),
                        }
                    )

    return entries


def _normalize_key_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def infer_known_list_key(list_name: str, folder_name: str) -> Optional[str]:
    name_raw = (list_name or "").lower()
    name_norm = _normalize_key_text(list_name or "")
    folder_raw = (folder_name or "").lower()

    if "input" in name_norm or "2input" in name_norm:
        return "input"
    if "slide" in name_norm or "6slide" in name_norm or "スライド" in name_raw:
        return "slide"
    if "experiment" in name_norm or "3experiment" in name_norm or "実験" in name_raw:
        return "experiment"
    if "4code" in name_norm or "code" in name_norm:
        return "code"
    if "manuscript" in name_norm or "5manuscript" in name_norm:
        return "manuscript"
    if "1plan" in name_norm or "plan" in name_norm:
        return "plan"
    if "学会発表" in name_raw:
        return "meeting"
    if "大会" in name_raw or "大会" in folder_raw:
        return "competition"
    if "日常" in name_raw:
        return "daily"
    if "インテリア" in name_raw:
        return "other"
    # folder fallback only for canonical numbered QPI lists.
    if "qpi" in _normalize_key_text(folder_raw):
        if "1" in name_norm and "plan" in name_norm:
            return "plan"
        if "2" in name_norm and "input" in name_norm:
            return "input"
        if "3" in name_norm and "experiment" in name_norm:
            return "experiment"
        if "4" in name_norm and "code" in name_norm:
            return "code"
        if "5" in name_norm and "manuscript" in name_norm:
            return "manuscript"
        if "6" in name_norm and "slide" in name_norm:
            return "slide"
    return None


def enrich_config_lists_from_workspace(
    client: ClickUpClient,
    cfg: HelperConfig,
    discovered: Optional[List[dict]] = None,
) -> Tuple[Dict[str, str], List[str]]:
    if not cfg.auto_discover_lists:
        return {}, []
    required_keys = {
        "plan",
        "input",
        "experiment",
        "code",
        "manuscript",
        "slide",
        "meeting",
        "competition",
        "other",
        "daily",
    }
    if required_keys.issubset(set(cfg.lists.keys())):
        return {}, []

    discovered = discovered if discovered is not None else discover_workspace_lists(client)
    added: Dict[str, str] = {}
    warnings: List[str] = []
    known_ids = {list_id: list_key for list_key, list_id in cfg.lists.items()}

    for item in discovered:
        list_id = str(item.get("list_id", ""))
        list_name = str(item.get("list_name", ""))
        folder_name = str(item.get("folder_name", ""))
        if not list_id or not list_name:
            continue
        inferred_key = infer_known_list_key(list_name=list_name, folder_name=folder_name)
        if not inferred_key:
            continue

        current_id = cfg.lists.get(inferred_key)
        if current_id:
            if current_id != list_id:
                warnings.append(
                    f"list key '{inferred_key}' conflict: configured={current_id} discovered={list_id} ({list_name})"
                )
            continue

        existing_key = known_ids.get(list_id)
        if existing_key:
            continue

        cfg.lists[inferred_key] = list_id
        known_ids[list_id] = inferred_key
        added[inferred_key] = list_id

    return added, warnings


def safe_enrich_config_lists(client: ClickUpClient, cfg: HelperConfig) -> Tuple[Dict[str, str], List[str]]:
    try:
        return enrich_config_lists_from_workspace(client, cfg)
    except Exception as exc:
        return {}, [f"auto list discovery failed: {exc}"]


def _load_clickup_token() -> str:
    env_token = os.getenv("CLICKUP_API_TOKEN", "").strip()
    if env_token:
        return env_token

    mcp_path = Path(__file__).resolve().parent.parent / ".cursor" / "mcp.json"
    if mcp_path.exists():
        with mcp_path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        token = (
            config.get("mcpServers", {})
            .get("clickup", {})
            .get("env", {})
            .get("CLICKUP_API_TOKEN", "")
            .strip()
        )
        if token:
            return token

    raise RuntimeError(
        "CLICKUP_API_TOKEN not found. Set env var or .cursor/mcp.json > mcpServers.clickup.env.CLICKUP_API_TOKEN"
    )


def to_local_datetime(day: date, hour: int, tz: ZoneInfo) -> datetime:
    return datetime.combine(day, time(hour=hour, minute=0), tzinfo=tz)


def normalize_priority_name(task: dict) -> str:
    p = task.get("priority")
    if isinstance(p, dict):
        text = str(p.get("priority", "")).lower().strip()
        return text if text else "none"
    return "none"


def infer_duration_minutes(task: dict, cfg: HelperConfig) -> Optional[int]:
    est = task.get("time_estimate")
    if est is None:
        return None
    try:
        est_ms = int(est)
    except (TypeError, ValueError):
        return None
    if est_ms <= 0:
        return None
    return max(1, est_ms // 60000)


def looks_fixed(task: dict, list_key: str, cfg: HelperConfig) -> Tuple[bool, str]:
    if list_key in cfg.fixed_list_keys:
        return True, f"list={list_key} is fixed"

    merged = f"{task.get('name', '')}\n{task.get('description', '')}".lower()
    for pat in cfg.fixed_keyword_patterns:
        if keyword_in_text(pat, merged):
            return True, f"matched keyword={pat}"

    status = task.get("status") or {}
    status_type = str(status.get("type", "")).lower()
    if status_type == "closed":
        return True, "closed status"

    return False, ""


def normalize_task_window(task: dict, list_key: str, cfg: HelperConfig, tz: ZoneInfo) -> TaskWindow:
    task_id = str(task.get("id", ""))
    name = str(task.get("name", ""))
    list_id = str((task.get("list") or {}).get("id", ""))
    status = task.get("status") or {}
    status_name = str(status.get("status", ""))
    status_type = str(status.get("type", ""))

    start_at = parse_ms_to_local(task.get("start_date"), tz)
    due_at = parse_ms_to_local(task.get("due_date"), tz)
    duration_minutes = infer_duration_minutes(task, cfg)

    end_at: Optional[datetime] = None
    if start_at and due_at:
        end_at = due_at
    elif start_at and duration_minutes:
        end_at = start_at + timedelta(minutes=duration_minutes)
    elif due_at and duration_minutes:
        start_at = due_at - timedelta(minutes=duration_minutes)
        end_at = due_at

    if start_at and end_at and end_at < start_at:
        start_at, end_at = end_at, start_at

    is_fixed, reason = looks_fixed(task, list_key=list_key, cfg=cfg)

    return TaskWindow(
        task_id=task_id,
        name=name,
        list_id=list_id,
        list_key=list_key,
        url=str(task.get("url", "")),
        status_name=status_name,
        status_type=status_type,
        priority_name=normalize_priority_name(task),
        start_at=start_at,
        end_at=end_at,
        due_at=due_at,
        duration_minutes=duration_minutes,
        movable=not is_fixed,
        reason=reason,
    )


def overlaps_range(
    start_at: Optional[datetime],
    end_at: Optional[datetime],
    due_at: Optional[datetime],
    range_start: datetime,
    range_end: datetime,
) -> bool:
    if start_at and end_at:
        return start_at <= range_end and end_at >= range_start
    if due_at:
        return range_start <= due_at <= range_end
    return False


def validate_snapshot(snapshot: AgendaSnapshot) -> List[str]:
    warnings: List[str] = []

    seen = set()
    for t in snapshot.fetched_tasks:
        if not t.task_id:
            warnings.append("task without id")
            continue
        if t.task_id in seen:
            warnings.append(f"duplicate task id: {t.task_id}")
        seen.add(t.task_id)

        if t.start_at and t.end_at and t.end_at < t.start_at:
            warnings.append(f"invalid time window: {t.task_id} end < start")

    scheduled_without_duration = [
        t.task_id for t in snapshot.scheduled_tasks if t.start_at and not t.end_at
    ]
    if scheduled_without_duration:
        warnings.append(
            "scheduled tasks missing end time: " + ", ".join(scheduled_without_duration[:5])
        )

    return warnings


def fetch_agenda(
    client: ClickUpClient,
    cfg: HelperConfig,
    start_day: date,
    days: int,
    list_keys: Sequence[str],
) -> AgendaSnapshot:
    if days <= 0:
        raise RuntimeError("days must be >= 1")

    try:
        tz = ZoneInfo(cfg.timezone)
    except Exception as exc:
        raise RuntimeError(f"invalid timezone in config: {cfg.timezone}") from exc

    end_day = start_day + timedelta(days=days - 1)
    range_start = datetime.combine(start_day, time.min, tzinfo=tz)
    range_end = datetime.combine(end_day, time.max, tzinfo=tz)

    fetched: List[TaskWindow] = []
    warnings: List[str] = []

    for list_key in list_keys:
        list_id = cfg.lists.get(list_key)
        if not list_id:
            warnings.append(f"list key not configured: {list_key}")
            continue

        raw_tasks = client.list_tasks_paginated(
            list_id=list_id,
            include_closed=False,
            max_pages=cfg.max_pages_per_list,
            max_tasks=cfg.max_tasks_per_list,
        )
        for raw in raw_tasks:
            try:
                fetched.append(normalize_task_window(raw, list_key=list_key, cfg=cfg, tz=tz))
            except Exception as exc:
                warnings.append(f"failed to normalize task in {list_key}: {exc}")

    scheduled: List[TaskWindow] = []
    unscheduled: List[TaskWindow] = []

    for task in fetched:
        if not overlaps_range(task.start_at, task.end_at, task.due_at, range_start, range_end):
            continue

        if task.start_at and task.end_at:
            scheduled.append(task)
        else:
            unscheduled.append(task)

    snapshot = AgendaSnapshot(
        start_date=start_day,
        end_date=end_day,
        timezone=cfg.timezone,
        fetched_tasks=fetched,
        scheduled_tasks=sorted(scheduled, key=lambda x: (x.start_at, x.task_id)),
        unscheduled_tasks=sorted(
            unscheduled, key=lambda x: (x.due_at or datetime.max.replace(tzinfo=tz), x.task_id)
        ),
        warnings=warnings,
    )
    snapshot.warnings.extend(validate_snapshot(snapshot))
    return snapshot


def merge_intervals(intervals: Sequence[Tuple[datetime, datetime]]) -> List[Tuple[datetime, datetime]]:
    if not intervals:
        return []

    ordered = sorted(intervals, key=lambda x: x[0])
    merged: List[Tuple[datetime, datetime]] = [ordered[0]]

    for start_at, end_at in ordered[1:]:
        last_start, last_end = merged[-1]
        if start_at <= last_end:
            merged[-1] = (last_start, max(last_end, end_at))
        else:
            merged.append((start_at, end_at))

    return merged


def subtract_intervals(
    base: Tuple[datetime, datetime], busy: Sequence[Tuple[datetime, datetime]]
) -> List[Tuple[datetime, datetime]]:
    free = [base]
    for busy_start, busy_end in busy:
        next_free: List[Tuple[datetime, datetime]] = []
        for free_start, free_end in free:
            if busy_end <= free_start or busy_start >= free_end:
                next_free.append((free_start, free_end))
                continue
            if busy_start > free_start:
                next_free.append((free_start, busy_start))
            if busy_end < free_end:
                next_free.append((busy_end, free_end))
        free = next_free
    return [slot for slot in free if slot[0] < slot[1]]


def ceil_datetime_to_minutes(dt: datetime, unit_minutes: int) -> datetime:
    unit = max(1, unit_minutes)
    floored = dt.replace(second=0, microsecond=0)
    minute_mod = floored.minute % unit
    if minute_mod == 0 and floored == dt.replace(second=0, microsecond=0):
        return floored
    delta = unit - minute_mod if minute_mod != 0 else unit
    return floored + timedelta(minutes=delta)


def build_busy_intervals(
    scheduled_tasks: Sequence[TaskWindow],
    min_gap_minutes: int,
    exclude_task_ids: Optional[set] = None,
) -> List[Tuple[datetime, datetime]]:
    exclude_task_ids = exclude_task_ids or set()
    intervals: List[Tuple[datetime, datetime]] = []
    gap = timedelta(minutes=max(0, min_gap_minutes))

    for task in scheduled_tasks:
        if task.task_id in exclude_task_ids:
            continue
        if not (task.start_at and task.end_at):
            continue
        intervals.append((task.start_at - gap, task.end_at + gap))

    return merge_intervals(intervals)


def is_no_schedule_day_anchor(task: TaskWindow, cfg: HelperConfig) -> bool:
    if task.list_key in set(cfg.no_schedule_day_list_keys):
        return True
    text = (task.name or "").lower()
    return any(keyword_in_text(kw, text) for kw in cfg.no_schedule_day_keyword_patterns)


def build_no_schedule_days(snapshot: AgendaSnapshot, cfg: HelperConfig) -> set:
    blocked: set = set()
    for task in snapshot.fetched_tasks:
        if not is_no_schedule_day_anchor(task, cfg):
            continue
        if task.start_at:
            blocked.add(task.start_at.date())
        elif task.due_at:
            blocked.add(task.due_at.date())
    return blocked


def infer_split_task_rule(
    text: str,
    list_key: str,
    cfg: HelperConfig,
) -> Optional[Dict[str, object]]:
    normalized = normalize_compact_text(text)
    if not normalized:
        return None
    normalized_plain = (text or "").lower().replace("％", "%")

    best_rule: Optional[Dict[str, object]] = None
    best_len = 0
    for rule in cfg.split_task_rules:
        if not isinstance(rule, dict):
            continue
        rule_list_key = str(rule.get("list_key", "")).strip()
        if rule_list_key and rule_list_key != list_key:
            continue
        keywords = rule.get("keywords", [])
        if not isinstance(keywords, list):
            continue
        for kw in keywords:
            kw_raw = str(kw).strip()
            kw_norm = normalize_compact_text(kw_raw)
            if not kw_norm:
                continue
            # Avoid false positives for very short ASCII aliases (e.g., "ac").
            if re.fullmatch(r"[a-z0-9]{1,3}", kw_raw.lower()):
                matched = keyword_in_text(kw_raw, normalized_plain)
            else:
                matched = kw_norm in normalized
            if matched and len(kw_norm) > best_len:
                best_rule = rule
                best_len = len(kw_norm)
    return best_rule


def split_rule_has_wait_gap(split_rule: Optional[Dict[str, object]]) -> bool:
    if not isinstance(split_rule, dict):
        return False
    segments = split_rule.get("segments", [])
    if not isinstance(segments, list) or len(segments) < 2:
        return False
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        for key in ("offset_minutes", "min_offset_minutes", "max_offset_minutes"):
            if key not in seg:
                continue
            try:
                if int(seg.get(key, 0)) > 0:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def infer_wait_task_like(text: str, split_rule: Optional[Dict[str, object]]) -> bool:
    if split_rule_has_wait_gap(split_rule):
        return True
    normalized = (text or "").lower()
    wait_keywords = [
        "待",
        "wait",
        "待機",
        "回収",
        "after",
        "時間後",
        "h後",
    ]
    return any(keyword_in_text(kw, normalized) for kw in wait_keywords)


def collect_keyword_task_intervals(
    scheduled_tasks: Sequence[TaskWindow],
    keyword_patterns: Sequence[str],
) -> List[Tuple[datetime, datetime]]:
    if not keyword_patterns:
        return []
    intervals: List[Tuple[datetime, datetime]] = []
    for task in scheduled_tasks:
        if not (task.start_at and task.end_at):
            continue
        text = (task.name or "").lower()
        if any(keyword_in_text(kw, text) for kw in keyword_patterns):
            intervals.append((task.start_at, task.end_at))
    return merge_intervals(intervals)


def _slot_segment_windows(slot: SlotCandidate) -> List[Tuple[datetime, datetime]]:
    windows: List[Tuple[datetime, datetime]] = []
    for seg in slot.segments:
        if not isinstance(seg, dict):
            continue
        raw_start = str(seg.get("slot_start", "")).strip()
        raw_end = str(seg.get("slot_end", "")).strip()
        if not raw_start or not raw_end:
            continue
        try:
            seg_start = datetime.fromisoformat(raw_start)
            seg_end = datetime.fromisoformat(raw_end)
        except ValueError:
            continue
        if seg_end <= seg_start:
            continue
        windows.append((seg_start, seg_end))
    windows.sort(key=lambda x: x[0])
    return windows


def compute_wait_task_jog_bonus(
    slot: SlotCandidate,
    jog_intervals: Sequence[Tuple[datetime, datetime]],
    window_minutes: int,
) -> float:
    if not jog_intervals:
        return 0.0
    window_sec = max(1, window_minutes) * 60.0
    bonus = 0.0

    # Prefer placing a wait-like task right before or right after jogging.
    edge_dist = min(
        min(
            abs((slot.slot_end - jog_start).total_seconds()),
            abs((slot.slot_start - jog_end).total_seconds()),
        )
        for jog_start, jog_end in jog_intervals
    )
    if edge_dist < window_sec:
        bonus += (window_sec - edge_dist) * 2.0

    # For split tasks, strongly prefer layouts where jogging fits into the waiting gap.
    seg_windows = _slot_segment_windows(slot)
    if len(seg_windows) < 2:
        return bonus

    for idx in range(len(seg_windows) - 1):
        gap_start = seg_windows[idx][1]
        gap_end = seg_windows[idx + 1][0]
        if gap_end <= gap_start:
            continue
        overlap_found = False
        nearest_gap_dist: Optional[float] = None
        for jog_start, jog_end in jog_intervals:
            overlap_start = max(gap_start, jog_start)
            overlap_end = min(gap_end, jog_end)
            if overlap_end > overlap_start:
                overlap_found = True
                overlap_sec = (overlap_end - overlap_start).total_seconds()
                bonus += 7200.0 + min(overlap_sec, window_sec) * 1.0
            dist = min(
                abs((jog_start - gap_start).total_seconds()),
                abs((jog_end - gap_end).total_seconds()),
                abs((jog_start - gap_end).total_seconds()),
                abs((jog_end - gap_start).total_seconds()),
            )
            if nearest_gap_dist is None or dist < nearest_gap_dist:
                nearest_gap_dist = dist
        if not overlap_found and nearest_gap_dist is not None and nearest_gap_dist < window_sec:
            bonus += (window_sec - nearest_gap_dist) * 1.0

    return bonus


def derive_split_segment_name(base_name: str, label: str) -> str:
    label = (label or "").strip().lower()
    if label == "prepare":
        return base_name
    if label == "collect":
        if "作成" in base_name:
            return base_name.replace("作成", "回収")
        return f"{base_name} 回収"
    return f"{base_name} ({label})"


def find_split_slots(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    list_key: str,
    base_name: str,
    split_rule: Dict[str, object],
    start_after: Optional[datetime],
    exclude_task_ids: Optional[set] = None,
    max_slots: int = 20,
) -> List[SlotCandidate]:
    tz = ZoneInfo(snapshot.timezone)
    exclude_task_ids = exclude_task_ids or set()
    blocked_days = build_no_schedule_days(snapshot, cfg)
    no_schedule_lists = set(cfg.no_schedule_day_list_keys)

    segments_raw = split_rule.get("segments", []) if isinstance(split_rule, dict) else []
    if not isinstance(segments_raw, list):
        return []
    segments_def: List[dict] = []
    for seg in segments_raw:
        if not isinstance(seg, dict):
            continue
        try:
            duration_minutes = int(seg.get("duration_minutes", 0))
        except (TypeError, ValueError):
            continue
        label = str(seg.get("label", "")).strip()
        if not label or duration_minutes <= 0:
            continue
        seg_def: Dict[str, object] = {
            "label": label,
            "duration_minutes": duration_minutes,
            "same_day": bool(seg.get("same_day", False)),
        }
        if "offset_minutes" in seg:
            try:
                seg_def["offset_minutes"] = int(seg.get("offset_minutes", 0))
            except (TypeError, ValueError):
                pass
        if "min_offset_minutes" in seg:
            try:
                seg_def["min_offset_minutes"] = int(seg.get("min_offset_minutes", 0))
            except (TypeError, ValueError):
                pass
        if "max_offset_minutes" in seg:
            try:
                seg_def["max_offset_minutes"] = int(seg.get("max_offset_minutes", 0))
            except (TypeError, ValueError):
                pass
        segments_def.append(
            seg_def
        )
    if not segments_def:
        return []

    busy = build_busy_intervals(
        snapshot.scheduled_tasks,
        min_gap_minutes=cfg.min_gap_minutes,
        exclude_task_ids=exclude_task_ids,
    )

    now_local = datetime.now(tz)
    slots: List[SlotCandidate] = []

    for day in date_range(snapshot.start_date, snapshot.end_date):
        if day in blocked_days and list_key not in no_schedule_lists:
            continue
        day_work_start = to_local_datetime(day, cfg.work_hours_start, tz)
        day_work_end = to_local_datetime(day, cfg.work_hours_end, tz)
        if day_work_end <= day_work_start:
            continue

        cursor = day_work_start
        if day == now_local.date():
            cursor = max(cursor, now_local)
        if start_after:
            cursor = max(cursor, start_after)
        cursor = ceil_datetime_to_minutes(cursor, 10)

        while cursor < day_work_end:
            seg_windows: List[dict] = []
            valid = True
            latest_end = cursor
            base_day = cursor.date()

            for seg in segments_def:
                duration_m = int(seg["duration_minutes"])
                same_day = bool(seg.get("same_day", False))
                candidate_starts: List[datetime] = []

                if "offset_minutes" in seg:
                    fixed = int(seg.get("offset_minutes", 0))
                    candidate_starts = [cursor + timedelta(minutes=fixed)]
                else:
                    min_offset = int(seg.get("min_offset_minutes", 0))
                    max_offset_raw = seg.get("max_offset_minutes")
                    earliest = cursor + timedelta(minutes=min_offset)
                    latest = None
                    if max_offset_raw is not None:
                        latest = cursor + timedelta(minutes=int(max_offset_raw))
                    if same_day:
                        day_limit = to_local_datetime(base_day, cfg.work_hours_end, tz) - timedelta(minutes=duration_m)
                        latest = day_limit if latest is None else min(latest, day_limit)
                    if latest is None:
                        latest = earliest + timedelta(hours=12)
                    start_c = ceil_datetime_to_minutes(earliest, 10)
                    while start_c <= latest:
                        candidate_starts.append(start_c)
                        start_c += timedelta(minutes=10)

                chosen_start: Optional[datetime] = None
                for seg_start in candidate_starts:
                    seg_end = seg_start + timedelta(minutes=duration_m)

                    if same_day and seg_start.date() != base_day:
                        continue
                    if seg_start.date() in blocked_days and list_key not in no_schedule_lists:
                        continue
                    seg_work_start = to_local_datetime(seg_start.date(), cfg.work_hours_start, tz)
                    seg_work_end = to_local_datetime(seg_start.date(), cfg.work_hours_end, tz)
                    if seg_start < seg_work_start or seg_end > seg_work_end:
                        continue
                    overlap_busy = any(not (seg_end <= b0 or seg_start >= b1) for b0, b1 in busy)
                    if overlap_busy:
                        continue
                    overlap_seg = any(
                        not (seg_end <= w["slot_start"] or seg_start >= w["slot_end"])
                        for w in seg_windows
                    )
                    if overlap_seg:
                        continue
                    chosen_start = seg_start
                    break

                if chosen_start is None:
                    valid = False
                    break

                chosen_end = chosen_start + timedelta(minutes=duration_m)
                seg_windows.append(
                    {
                        "label": seg["label"],
                        "slot_start": chosen_start,
                        "slot_end": chosen_end,
                        "task_name": derive_split_segment_name(base_name, str(seg["label"])),
                    }
                )
                if chosen_end > latest_end:
                    latest_end = chosen_end

            if valid:
                rule_id = str(split_rule.get("id", "split"))
                segment_desc = ", ".join(
                    f"{s['label']}:{int((s['slot_end'] - s['slot_start']).total_seconds() // 60)}m"
                    for s in seg_windows
                )
                slots.append(
                    SlotCandidate(
                        label="split",
                        slot_start=cursor,
                        slot_end=latest_end,
                        list_key=list_key,
                        score=cursor.timestamp(),
                        reasons=[f"split rule={rule_id} ({segment_desc})"],
                        segments=[
                            {
                                "label": s["label"],
                                "task_name": s["task_name"],
                                "slot_start": s["slot_start"].isoformat(),
                                "slot_end": s["slot_end"].isoformat(),
                            }
                            for s in seg_windows
                        ],
                    )
                )
                if len(slots) >= max_slots:
                    return slots

            cursor += timedelta(minutes=10)

    return slots


def find_free_slots(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    duration_minutes: int,
    list_key: str,
    exclude_task_ids: Optional[set] = None,
    extra_busy: Optional[List[Tuple[datetime, datetime]]] = None,
    start_after: Optional[datetime] = None,
    max_slots: int = 20,
) -> List[SlotCandidate]:
    tz = ZoneInfo(snapshot.timezone)
    duration = timedelta(minutes=duration_minutes)
    blocked_days = build_no_schedule_days(snapshot, cfg)
    no_schedule_lists = set(cfg.no_schedule_day_list_keys)

    busy = build_busy_intervals(
        snapshot.scheduled_tasks,
        min_gap_minutes=cfg.min_gap_minutes,
        exclude_task_ids=exclude_task_ids,
    )
    if extra_busy:
        busy = merge_intervals(busy + extra_busy)

    slots: List[SlotCandidate] = []
    now_local = datetime.now(tz)

    for day in date_range(snapshot.start_date, snapshot.end_date):
        if day in blocked_days and list_key not in no_schedule_lists:
            continue
        work_start = to_local_datetime(day, cfg.work_hours_start, tz)
        work_end = to_local_datetime(day, cfg.work_hours_end, tz)
        if work_end <= work_start:
            continue

        if day == now_local.date():
            work_start = max(work_start, now_local)
        if start_after:
            work_start = max(work_start, start_after)
        if work_end <= work_start:
            continue

        day_busy = [
            (max(work_start, b0), min(work_end, b1))
            for b0, b1 in busy
            if b1 > work_start and b0 < work_end
        ]
        free_ranges = subtract_intervals((work_start, work_end), day_busy)

        for free_start, free_end in free_ranges:
            if free_end - free_start < duration:
                continue
            slot_start = ceil_datetime_to_minutes(free_start, 10)
            slot_end = slot_start + duration
            if slot_end > free_end:
                continue
            slots.append(
                SlotCandidate(
                    label="direct",
                    slot_start=slot_start,
                    slot_end=slot_end,
                    list_key=list_key,
                    score=slot_start.timestamp(),
                    slack_minutes=max(
                        0,
                        int((free_end - free_start).total_seconds() // 60) - duration_minutes,
                    ),
                    reasons=["fits in free time"],
                )
            )
            if len(slots) >= max_slots:
                return slots

    return slots


def parse_duration_from_text(text: str) -> Optional[int]:
    if not text:
        return None

    s = text.lower()
    total = 0

    hour_matches = re.findall(r"(\d+)\s*(?:h|hr|hrs|hour|hours|時間)", s)
    minute_matches = re.findall(r"(\d+)\s*(?:m|min|mins|minute|minutes|分)", s)

    for h in hour_matches:
        total += int(h) * 60
    for m in minute_matches:
        total += int(m)

    if total > 0:
        return total

    compact = re.search(r"(\d+)(?:h|時間)(\d+)(?:m|分)", s)
    if compact:
        return int(compact.group(1)) * 60 + int(compact.group(2))

    return None


def parse_priority(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"urgent", "high", "normal", "low", "auto"}:
        return v
    if v in {"急ぎ", "最優先"}:
        return "urgent"
    if v in {"高", "高め"}:
        return "high"
    if v in {"低", "低め"}:
        return "low"
    return "normal"


def has_deadline_signal(text: str, cfg: HelperConfig) -> bool:
    normalized = (text or "").lower()
    for kw in cfg.deadline_keyword_patterns:
        if keyword_in_text(kw, normalized):
            return True
    return False


def infer_task_policy(
    text: str,
    explicit_priority: str,
    cfg: HelperConfig,
) -> TaskPolicy:
    parsed = parse_priority(explicit_priority)
    normalized_text = (text or "").lower()

    best_category = "general"
    best_matches: List[str] = []

    # Explicit input marker should override keyword collisions like "paper".
    if re.search(r"(^|\s)(input|inbox)\s*[:：#]", normalized_text):
        best_category = "input"
        best_matches = ["input_marker"]
    else:
        for category in cfg.category_order:
            keywords = cfg.category_keyword_rules.get(category, [])
            matched = [kw for kw in keywords if keyword_in_text(kw, normalized_text)]
            if matched:
                best_category = category
                best_matches = matched
                break

    deadline_signal = has_deadline_signal(text, cfg)

    if parsed != "auto":
        chosen_priority = parsed
    elif deadline_signal:
        chosen_priority = "urgent"
    else:
        chosen_priority = cfg.category_priority.get(best_category, "normal")

    priority_bias = cfg.priority_bias_hours.get(chosen_priority, 0)
    category_bias = cfg.category_bias_hours.get(best_category, 0)

    return TaskPolicy(
        category=best_category,
        priority_name=chosen_priority,
        schedule_bias_hours=priority_bias + category_bias,
        matched_keywords=best_matches,
    )


def infer_workflow_tag(
    text: str,
    list_key: str,
    cfg: HelperConfig,
) -> Optional[WorkflowTag]:
    if not cfg.workflow_enabled:
        return None

    normalized = (text or "").lower()

    group_name = ""
    for g, list_keys in cfg.workflow_groups.items():
        if list_key in list_keys:
            group_name = g
            break
    if not group_name:
        return None

    matched_best: Optional[WorkflowTag] = None
    fallback = WorkflowTag(group=group_name, stage="unspecified", order=999, matched_keywords=[])

    for rule in cfg.workflow_stage_rules:
        if str(rule.get("group", "")) != group_name:
            continue
        keywords = rule.get("keywords", [])
        if not isinstance(keywords, list):
            continue
        matched = [kw for kw in keywords if keyword_in_text(str(kw), normalized)]
        if not matched:
            continue
        try:
            order = int(rule.get("order", 999))
        except (TypeError, ValueError):
            order = 999
        stage = str(rule.get("stage", "unspecified")) or "unspecified"
        candidate = WorkflowTag(group=group_name, stage=stage, order=order, matched_keywords=matched)
        if matched_best is None or candidate.order < matched_best.order:
            matched_best = candidate

    return matched_best or fallback


def normalize_compact_text(text: str) -> str:
    compact = (text or "").lower().replace("％", "%")
    compact = re.sub(r"[\s　_\-/~〜→>]+", "", compact)
    return compact


def parse_t0_offset_hours(text: str) -> Optional[int]:
    normalized = (text or "").lower().replace("＋", "+")
    match = re.search(r"t0\s*\+\s*(\d+)\s*h", normalized)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def infer_medium_flow_step(
    text: str,
    list_key: str,
    cfg: HelperConfig,
) -> Optional[str]:
    if not cfg.medium_flow_enabled:
        return None
    if list_key != "experiment":
        return None
    compact = normalize_compact_text(text)
    if not compact:
        return None

    t0_offset_hours = parse_t0_offset_hours(text)
    if t0_offset_hours is not None:
        mapped = cfg.medium_flow_t0_offset_map.get(t0_offset_hours)
        if mapped:
            return mapped

    best_step: Optional[str] = None
    best_len = 0
    for step in cfg.medium_flow_steps:
        step_id = str(step.get("id", "")).strip()
        keywords = step.get("keywords", [])
        if not step_id or not isinstance(keywords, list):
            continue
        for kw in keywords:
            kw_compact = normalize_compact_text(str(kw))
            if not kw_compact:
                continue
            if kw_compact in compact and len(kw_compact) > best_len:
                best_step = step_id
                best_len = len(kw_compact)

    return best_step


def medium_flow_rule_map(cfg: HelperConfig) -> Dict[str, Dict[str, object]]:
    rules: Dict[str, Dict[str, object]] = {}
    for item in cfg.medium_flow_rules:
        if not isinstance(item, dict):
            continue
        step = str(item.get("step", "")).strip()
        preds = item.get("predecessors", [])
        if not step or not isinstance(preds, list):
            continue
        gap_days = int(item.get("gap_days", 0) or 0)
        gap_hours = int(item.get("gap_hours", 0) or 0)
        gap_minutes = int(item.get("gap_minutes", 0) or 0)
        total_gap_minutes = gap_days * 24 * 60 + gap_hours * 60 + gap_minutes
        rules[step] = {
            "predecessors": [str(x) for x in preds if str(x).strip()],
            "gap_minutes": total_gap_minutes,
            "gap_days": gap_days,
            "gap_hours": gap_hours,
        }
    return rules


def medium_flow_step_label(cfg: HelperConfig, step_id: str) -> str:
    for step in cfg.medium_flow_steps:
        if str(step.get("id", "")) != step_id:
            continue
        keywords = step.get("keywords", [])
        if isinstance(keywords, list) and keywords:
            return str(keywords[0])
    return step_id


def infer_medium_flow_start_after(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    step_id: Optional[str],
    reference_time: Optional[datetime] = None,
) -> Optional[datetime]:
    if not cfg.medium_flow_enabled:
        return None
    if not step_id:
        return None

    rule = medium_flow_rule_map(cfg).get(step_id)
    if not rule:
        return None
    preds = rule.get("predecessors", [])
    if not isinstance(preds, list) or not preds:
        return None
    gap_minutes = int(rule.get("gap_minutes", 0))

    anchors_by_step: Dict[str, List[datetime]] = {}
    for task in snapshot.scheduled_tasks:
        anchor = task.start_at or task.end_at or task.due_at
        if not anchor:
            continue
        step = infer_medium_flow_step(task.name, task.list_key, cfg)
        if not step:
            continue
        anchors_by_step.setdefault(step, []).append(anchor)

    candidates: List[datetime] = []
    for pred in preds:
        times = anchors_by_step.get(pred, [])
        if reference_time is not None:
            times = [x for x in times if x <= reference_time]
        if not times:
            continue
        pred_anchor = max(times)
        candidates.append(pred_anchor + timedelta(minutes=gap_minutes))

    if not candidates:
        return None
    return max(candidates)


def infer_medium_flow_end_before(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    step_id: Optional[str],
    reference_time: Optional[datetime] = None,
) -> Optional[datetime]:
    if not cfg.medium_flow_enabled:
        return None
    if not step_id:
        return None

    detail = infer_medium_flow_end_before_detail(
        snapshot=snapshot,
        cfg=cfg,
        step_id=step_id,
        reference_time=reference_time,
    )
    if not detail:
        return None
    return detail["end_before"]


def infer_medium_flow_end_before_detail(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    step_id: Optional[str],
    reference_time: Optional[datetime] = None,
) -> Optional[Dict[str, object]]:
    if not cfg.medium_flow_enabled:
        return None
    if not step_id:
        return None

    rule_map = medium_flow_rule_map(cfg)
    successor_rules = [
        (succ_step, rule)
        for succ_step, rule in rule_map.items()
        if step_id in (rule.get("predecessors") or [])
    ]
    if not successor_rules:
        return None

    anchors_by_step: Dict[str, List[datetime]] = {}
    for task in snapshot.scheduled_tasks:
        anchor = task.start_at or task.end_at or task.due_at
        if not anchor:
            continue
        step = infer_medium_flow_step(task.name, task.list_key, cfg)
        if not step:
            continue
        anchors_by_step.setdefault(step, []).append(anchor)

    best: Optional[Dict[str, object]] = None
    for succ_step, rule in successor_rules:
        gap_minutes = int(rule.get("gap_minutes", 0))
        succ_times = anchors_by_step.get(succ_step, [])
        if reference_time is not None:
            succ_times = [x for x in succ_times if x >= reference_time]
        if not succ_times:
            continue
        succ_anchor = min(succ_times)
        end_before = succ_anchor - timedelta(minutes=gap_minutes)
        if best is None or end_before < best["end_before"]:
            best = {
                "end_before": end_before,
                "successor_step": succ_step,
                "successor_anchor": succ_anchor,
                "gap_minutes": gap_minutes,
            }

    return best


def infer_workflow_start_after(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    workflow_tag: Optional[WorkflowTag],
) -> Optional[datetime]:
    if workflow_tag is None:
        return None
    if workflow_tag.order >= 999:
        return None

    anchor: Optional[datetime] = None
    for task in snapshot.scheduled_tasks:
        if not task.end_at:
            continue
        other_tag = infer_workflow_tag(task.name, task.list_key, cfg)
        if not other_tag:
            continue
        if other_tag.group != workflow_tag.group:
            continue
        if other_tag.order < workflow_tag.order:
            if anchor is None or task.end_at > anchor:
                anchor = task.end_at

    if anchor is None:
        return None
    return anchor + timedelta(minutes=max(0, cfg.workflow_gap_minutes))


def select_best_slots(
    slots: Sequence[SlotCandidate],
    due_at: Optional[datetime],
    priority_name: str,
    schedule_bias_hours: int,
    max_candidates: int,
    prefer_tight_gap: bool = False,
    prefer_around_jog_for_wait: bool = False,
    jog_intervals: Optional[Sequence[Tuple[datetime, datetime]]] = None,
    jog_window_minutes: int = 0,
) -> List[SlotCandidate]:
    priority_boost = PRIORITY_SCORE.get(priority_name, 2)
    if slots:
        tz = slots[0].slot_start.tzinfo
        now_local = datetime.now(tz) if tz else datetime.now()
    else:
        now_local = datetime.now()
    defer_until = now_local + timedelta(hours=max(0, schedule_bias_hours))

    def score_fn(slot: SlotCandidate) -> float:
        start_score = slot.slot_start.timestamp()
        if prefer_tight_gap:
            # Prefer slots that fit into smaller leftover gaps first.
            start_score += max(0, slot.slack_minutes) * 120
        if schedule_bias_hours > 0 and slot.slot_start < defer_until:
            penalty_factor = 3.0 if (not due_at or due_at >= defer_until) else 0.5
            start_score += (defer_until - slot.slot_start).total_seconds() * penalty_factor
        late_penalty = 0.0
        if due_at and slot.slot_end > due_at:
            late_penalty = (slot.slot_end - due_at).total_seconds() / 60.0 * (10 + priority_boost)
        jog_bonus = 0.0
        if prefer_around_jog_for_wait and jog_intervals:
            jog_bonus = compute_wait_task_jog_bonus(
                slot,
                jog_intervals=jog_intervals,
                window_minutes=jog_window_minutes,
            )
        return start_score + late_penalty - jog_bonus

    ranked = sorted(slots, key=score_fn)
    for item in ranked:
        item.score = score_fn(item)
    return ranked[:max_candidates]


def build_rearrangement_candidates(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    duration_minutes: int,
    list_key: str,
    due_at: Optional[datetime],
    priority_name: str,
    schedule_bias_hours: int,
    start_after: Optional[datetime],
    max_candidates: int,
    prefer_around_jog_for_wait: bool = False,
    jog_intervals: Optional[Sequence[Tuple[datetime, datetime]]] = None,
    jog_window_minutes: int = 0,
) -> List[SlotCandidate]:
    direct_slots = find_free_slots(
        snapshot,
        cfg=cfg,
        duration_minutes=duration_minutes,
        list_key=list_key,
        start_after=start_after,
        max_slots=3,
    )
    direct_ranked = select_best_slots(
        direct_slots,
        due_at=due_at,
        priority_name=priority_name,
        schedule_bias_hours=schedule_bias_hours,
        max_candidates=1,
        prefer_around_jog_for_wait=prefer_around_jog_for_wait,
        jog_intervals=jog_intervals,
        jog_window_minutes=jog_window_minutes,
    )
    direct_best = direct_ranked[0] if direct_ranked else None

    allowed_lists = (
        set(cfg.rearrange_allowed_list_keys)
        if cfg.rearrange_allowed_list_keys
        else set()
    )
    # Keep previous behavior (same-list rearrangement) while allowing extra scopes
    # such as low-priority input deferral.
    allowed_lists.add(list_key)
    if cfg.protect_experiment_structure and list_key != "experiment":
        allowed_lists.discard("experiment")
    if not allowed_lists:
        allowed_lists = {list_key}

    movable_tasks = [
        t
        for t in snapshot.scheduled_tasks
        if t.movable and t.start_at and t.end_at and t.list_key in allowed_lists
    ]

    def movable_sort_key(task: TaskWindow):
        input_first = 0 if task.list_key == "input" else 1
        pr = PRIORITY_SCORE.get(task.priority_name, 0)
        due_ts = task.due_at.timestamp() if task.due_at else float("inf")
        return (input_first, pr, due_ts, task.start_at.timestamp())

    candidates: List[SlotCandidate] = []

    for task in sorted(movable_tasks, key=movable_sort_key):
        issue_slots = find_free_slots(
            snapshot,
            cfg=cfg,
            duration_minutes=duration_minutes,
            list_key=list_key,
            exclude_task_ids={task.task_id},
            start_after=start_after,
            max_slots=3,
        )
        if not issue_slots:
            continue

        issue_slot = issue_slots[0]

        if direct_best:
            improved = issue_slot.slot_start + timedelta(minutes=30) < direct_best.slot_start
            if due_at:
                direct_on_time = direct_best.slot_end <= due_at
                improved = improved or (issue_slot.slot_end <= due_at and not direct_on_time)
            if not improved:
                continue

        moved_duration = max(
            30,
            int((task.end_at - task.start_at).total_seconds() // 60),
        )

        moved_slots = find_free_slots(
            snapshot,
            cfg=cfg,
            duration_minutes=moved_duration,
            list_key=task.list_key,
            exclude_task_ids={task.task_id},
            extra_busy=[(issue_slot.slot_start, issue_slot.slot_end)],
            start_after=task.end_at,
            max_slots=5,
        )
        if not moved_slots:
            continue

        if task.list_key == "experiment" and cfg.protect_experiment_structure:
            siblings = [
                s
                for s in snapshot.scheduled_tasks
                if s.task_id != task.task_id and s.list_key == "experiment" and s.start_at and s.end_at
            ]
            prev_end = max(
                (s.end_at for s in siblings if s.end_at <= task.start_at),
                default=None,
            )
            next_start = min(
                (s.start_at for s in siblings if s.start_at >= task.end_at),
                default=None,
            )
            valid_slots = []
            for slot in moved_slots:
                if prev_end and slot.slot_start < prev_end:
                    continue
                if next_start and slot.slot_end > next_start:
                    continue
                valid_slots.append(slot)
            if not valid_slots:
                continue
            moved_to = valid_slots[0]
        else:
            moved_to = moved_slots[0]
        if moved_to.slot_start == task.start_at and moved_to.slot_end == task.end_at:
            continue

        cand = SlotCandidate(
            label="rearrange",
            slot_start=issue_slot.slot_start,
            slot_end=issue_slot.slot_end,
            list_key=list_key,
            score=issue_slot.slot_start.timestamp(),
            reasons=[f"move task {task.task_id} ({task.name})"],
            moves=[
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "from_start": task.start_at.isoformat(),
                    "from_end": task.end_at.isoformat(),
                    "to_start": moved_to.slot_start.isoformat(),
                    "to_end": moved_to.slot_end.isoformat(),
                    "task_url": task.url,
                    "list_key": task.list_key,
                }
            ],
        )
        candidates.append(cand)
        if len(candidates) >= max_candidates:
            break

    return select_best_slots(
        candidates,
        due_at=due_at,
        priority_name=priority_name,
        schedule_bias_hours=schedule_bias_hours,
        max_candidates=max_candidates,
        prefer_around_jog_for_wait=prefer_around_jog_for_wait,
        jog_intervals=jog_intervals,
        jog_window_minutes=jog_window_minutes,
    )


def local_today_string(tz_name: str) -> str:
    tz = ZoneInfo(tz_name)
    return datetime.now(timezone.utc).astimezone(tz).strftime("%Y-%m-%d")


def resolve_write_list_key(
    requested_list_key: str,
    cfg: HelperConfig,
    fallback_hint: str = "",
) -> Tuple[str, Optional[str]]:
    blocked = set(cfg.blocked_write_list_keys)
    if requested_list_key not in blocked:
        return requested_list_key, None

    candidates = [
        fallback_hint,
        cfg.category_default_list.get("general", ""),
        "other",
        "manuscript",
        "daily",
    ]
    for cand in candidates:
        if cand and cand in cfg.lists and cand not in blocked:
            return (
                cand,
                f"list={requested_list_key} is blocked for new schedules; rerouted to {cand}",
            )

    for cand in sorted(cfg.lists.keys()):
        if cand not in blocked:
            return (
                cand,
                f"list={requested_list_key} is blocked for new schedules; rerouted to {cand}",
            )

    raise RuntimeError("all configured lists are blocked for new schedule writes")


def command_add(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    _, discovery_warnings = safe_enrich_config_lists(client, cfg)

    target_list_key, reroute_warn = resolve_write_list_key(args.list, cfg, fallback_hint="other")
    list_id = cfg.lists.get(target_list_key)
    if not list_id:
        for w in discovery_warnings[:3]:
            print(f"[warn] {w}")
        print(f"[error] unknown list key: {target_list_key}", file=sys.stderr)
        return 2
    if reroute_warn:
        print(f"[warn] {reroute_warn}")

    date_str = args.date or local_today_string(cfg.timezone)

    if args.start is not None and args.duration is not None:
        start_hour = args.start
        end_hour = args.start + args.duration
    elif args.time:
        start_hour, end_hour = TIME_PRESETS[args.time]
    else:
        start_hour, end_hour = TIME_PRESETS["default"]

    if end_hour <= start_hour:
        print("[error] end hour must be greater than start hour", file=sys.stderr)
        return 2

    day = parse_date_yyyy_mm_dd(date_str)
    tz = ZoneInfo(cfg.timezone)
    base = datetime.combine(day, time.min, tzinfo=tz)
    start_at = base + timedelta(hours=start_hour)
    end_at = base + timedelta(hours=end_hour)
    start_ms = to_unix_ms(start_at)
    end_ms = to_unix_ms(end_at)

    payload = {
        "name": args.name,
        "start_date": start_ms,
        "start_date_time": True,
        "due_date": end_ms,
        "due_date_time": True,
        "time_estimate": (end_hour - start_hour) * 3600 * 1000,
    }
    if args.description:
        payload["description"] = args.description

    if args.dry_run:
        print("[dry-run] task payload")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    result = client.request("POST", f"/list/{list_id}/task", payload=payload)

    print(f"Created: {result.get('name', args.name)}")
    print(f"URL: {result.get('url', '')}")
    print(f"Time: {date_str} {start_hour:02d}:00-{end_hour:02d}:00 ({cfg.timezone})")
    return 0


def command_doctor(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)

    failures = 0
    report = {
        "timestamp": datetime.now().isoformat(),
        "token_loaded": bool(token),
        "checks": [],
    }
    discovered_entries: List[dict] = []
    auto_added: Dict[str, str] = {}
    discovery_warnings: List[str] = []

    try:
        user = client.get_user()
        username = (
            (user.get("user") or {}).get("username")
            or (user.get("user") or {}).get("email")
            or "(unknown)"
        )
        report["checks"].append({"name": "auth", "ok": True, "user": username})
        print(f"[ok] auth user={username}")
    except Exception as exc:
        failures += 1
        report["checks"].append({"name": "auth", "ok": False, "error": str(exc)})
        print(f"[fail] auth: {exc}")

    if args.discover or cfg.auto_discover_lists:
        try:
            discovered_entries = discover_workspace_lists(client)
            if cfg.auto_discover_lists:
                auto_added, discovery_warnings = enrich_config_lists_from_workspace(
                    client,
                    cfg,
                    discovered=discovered_entries,
                )
        except Exception as exc:
            discovery_warnings.append(f"workspace list discovery failed: {exc}")

    if auto_added:
        print("[ok] auto-discovered list keys:")
        for key, list_id in sorted(auto_added.items()):
            print(f"  - {key}: {list_id}")
    if discovery_warnings:
        for msg in discovery_warnings:
            print(f"[warn] {msg}")

    if args.discover:
        report["discovery"] = {
            "list_count": len(discovered_entries),
            "auto_added": auto_added,
            "warnings": discovery_warnings,
            "lists": discovered_entries,
        }
        print(f"[discover] workspace lists: {len(discovered_entries)}")
        for item in discovered_entries:
            space_name = str(item.get("space_name", ""))
            folder_name = str(item.get("folder_name", ""))
            list_name = str(item.get("list_name", ""))
            list_id = str(item.get("list_id", ""))
            inferred = infer_known_list_key(list_name, folder_name) or "-"
            print(
                f"  - {space_name} / {folder_name or '(space)'} / {list_name} "
                f"({list_id}) key={inferred}"
            )

    target_lists = args.list if args.list else sorted(cfg.lists.keys())

    for list_key in target_lists:
        list_id = cfg.lists.get(list_key)
        if not list_id:
            failures += 1
            report["checks"].append(
                {
                    "name": f"list:{list_key}",
                    "ok": False,
                    "error": "list key not configured",
                }
            )
            print(f"[fail] list={list_key}: not configured")
            continue

        try:
            tasks = client.list_tasks_paginated(
                list_id=list_id,
                include_closed=False,
                max_pages=1,
                max_tasks=min(
                    cfg.max_tasks_per_list,
                    max(1, args.sample * 2, args.probe_tasks),
                ),
            )
            sample = tasks[: args.sample]
            parsed = []
            tz = ZoneInfo(cfg.timezone)
            for raw in sample:
                parsed.append(normalize_task_window(raw, list_key=list_key, cfg=cfg, tz=tz))

            detail_checked = 0
            detail_fail = 0
            if args.deep and parsed:
                for task in parsed[: args.sample]:
                    try:
                        detail = client.get_task(task.task_id)
                        if str(detail.get("id", "")) != task.task_id:
                            raise RuntimeError("task id mismatch")
                        detail_checked += 1
                    except Exception:
                        detail_fail += 1

            report["checks"].append(
                {
                    "name": f"list:{list_key}",
                    "ok": True,
                    "list_id": list_id,
                    "fetched": len(tasks),
                    "sampled": len(parsed),
                    "deep_checked": detail_checked,
                    "deep_fail": detail_fail,
                }
            )

            if detail_fail > 0:
                failures += detail_fail
                print(
                    f"[warn] list={list_key} fetched={len(tasks)} sampled={len(parsed)} deep_fail={detail_fail}"
                )
            else:
                print(
                    f"[ok] list={list_key} fetched={len(tasks)} sampled={len(parsed)} deep_checked={detail_checked}"
                )

        except Exception as exc:
            failures += 1
            report["checks"].append(
                {
                    "name": f"list:{list_key}",
                    "ok": False,
                    "list_id": list_id,
                    "error": str(exc),
                }
            )
            print(f"[fail] list={list_key}: {exc}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    print(f"[doctor] failures={failures}")
    return 0 if failures == 0 else 1


def snapshot_to_json(snapshot: AgendaSnapshot) -> dict:
    def encode_task(task: TaskWindow) -> dict:
        return {
            "task_id": task.task_id,
            "name": task.name,
            "list_id": task.list_id,
            "list_key": task.list_key,
            "url": task.url,
            "status_name": task.status_name,
            "status_type": task.status_type,
            "priority_name": task.priority_name,
            "start_at": task.start_at.isoformat() if task.start_at else None,
            "end_at": task.end_at.isoformat() if task.end_at else None,
            "due_at": task.due_at.isoformat() if task.due_at else None,
            "duration_minutes": task.duration_minutes,
            "movable": task.movable,
            "reason": task.reason,
        }

    return {
        "start_date": snapshot.start_date.isoformat(),
        "end_date": snapshot.end_date.isoformat(),
        "timezone": snapshot.timezone,
        "counts": {
            "fetched": len(snapshot.fetched_tasks),
            "scheduled": len(snapshot.scheduled_tasks),
            "unscheduled": len(snapshot.unscheduled_tasks),
        },
        "warnings": snapshot.warnings,
        "scheduled_tasks": [encode_task(x) for x in snapshot.scheduled_tasks],
        "unscheduled_tasks": [encode_task(x) for x in snapshot.unscheduled_tasks],
    }


def command_agenda(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    auto_added, discovery_warnings = safe_enrich_config_lists(client, cfg)

    start_day = parse_date_yyyy_mm_dd(args.start) if args.start else date.today()
    list_keys = args.list if args.list else sorted(cfg.lists.keys())

    snapshot = fetch_agenda(
        client=client,
        cfg=cfg,
        start_day=start_day,
        days=args.days,
        list_keys=list_keys,
    )
    if auto_added:
        snapshot.warnings.append(
            "auto discovered list keys: "
            + ", ".join(f"{k}={v}" for k, v in sorted(auto_added.items()))
        )
    snapshot.warnings.extend(discovery_warnings)

    print(
        f"[agenda] {snapshot.start_date.isoformat()} -> {snapshot.end_date.isoformat()} ({snapshot.timezone})"
    )
    print(f"- fetched tasks: {len(snapshot.fetched_tasks)}")
    print(f"- scheduled in range: {len(snapshot.scheduled_tasks)}")
    print(f"- unscheduled in range: {len(snapshot.unscheduled_tasks)}")
    print(f"- warnings: {len(snapshot.warnings)}")

    preview = snapshot.scheduled_tasks[: args.preview]
    if preview:
        print("- scheduled preview:")
        for task in preview:
            print(
                f"  - {task.start_at.strftime('%Y-%m-%d %H:%M')} - {task.end_at.strftime('%H:%M')} "
                f"[{task.list_key}] {task.name} ({task.task_id})"
            )

    if args.verify and snapshot.warnings:
        print("[verify] warnings")
        for w in snapshot.warnings[:20]:
            print(f"  - {w}")

    if args.json:
        print(json.dumps(snapshot_to_json(snapshot), ensure_ascii=False, indent=2))

    return 0


def command_dashboard(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    auto_added, discovery_warnings = safe_enrich_config_lists(client, cfg)

    start_day = parse_date_yyyy_mm_dd(args.start) if args.start else date.today()
    days = args.days if args.days > 0 else cfg.monthly_horizon_days
    list_keys = args.list if args.list else sorted(cfg.lists.keys())

    snapshot = fetch_agenda(
        client=client,
        cfg=cfg,
        start_day=start_day,
        days=days,
        list_keys=list_keys,
    )
    if auto_added:
        snapshot.warnings.append(
            "auto discovered list keys: "
            + ", ".join(f"{k}={v}" for k, v in sorted(auto_added.items()))
        )
    snapshot.warnings.extend(discovery_warnings)
    overdue, unscheduled_due_soon, scheduled_after_due = detect_deadline_risks(
        snapshot,
        warn_days=cfg.deadline_warn_days,
    )
    conflicts = detect_schedule_conflicts(snapshot.scheduled_tasks)
    markdown = build_dashboard_markdown(
        snapshot=snapshot,
        overdue=overdue,
        unscheduled_due_soon=unscheduled_due_soon,
        scheduled_after_due=scheduled_after_due,
        conflicts=conflicts,
    )
    path = build_dashboard_file_path()
    with path.open("w", encoding="utf-8") as handle:
        handle.write(markdown)

    print(
        f"[dashboard] range={snapshot.start_date.isoformat()}->{snapshot.end_date.isoformat()} "
        f"scheduled={len(snapshot.scheduled_tasks)}"
    )
    print(
        f"- overdue={len(overdue)} due_soon_unscheduled={len(unscheduled_due_soon)} "
        f"after_due={len(scheduled_after_due)} overlaps={len(conflicts)}"
    )
    print(f"- output: {path}")

    if args.print:
        print(markdown)

    append_memory_entry(
        cfg,
        title="dashboard check",
        lines=[
            f"range: {snapshot.start_date.isoformat()}->{snapshot.end_date.isoformat()}",
            f"scheduled: {len(snapshot.scheduled_tasks)} / unscheduled: {len(snapshot.unscheduled_tasks)}",
            f"overdue: {len(overdue)} / due_soon_unscheduled: {len(unscheduled_due_soon)} / overlaps: {len(conflicts)}",
            f"dashboard: {path}",
        ],
    )
    return 0


def command_figfix_sync(args: argparse.Namespace, cfg: HelperConfig) -> int:
    hub_root = (
        Path(args.hub_root).expanduser().resolve()
        if args.hub_root
        else default_figure_hub_root()
    )
    script_path = hub_root / "scripts" / "figure_hub.py"
    config_path = choose_figure_hub_config(hub_root, args.figure_hub_config)

    if not script_path.exists():
        print(f"[error] figure_hub script not found: {script_path}", file=sys.stderr)
        return 2
    if not config_path.exists():
        print(f"[error] figure_hub config not found: {config_path}", file=sys.stderr)
        return 2

    print("[figfix-sync] settings")
    print(f"- hub_root: {hub_root}")
    print(f"- figure_hub.py: {script_path}")
    print(f"- config: {config_path}")
    print(f"- max_tasks: {max(1, args.max_tasks)}")
    print(f"- due_in_days: {max(0, args.due_in_days)}")
    print(f"- list_key: {args.list_key or '(config default)'}")
    print(f"- only_fig_id: {args.only_fig_id or '(all)'}")
    print(f"- mode: {'apply' if args.apply else 'preview'}")
    print(f"- run_recommend: {args.recommend_first}")

    if args.recommend_first:
        recommend_cmd = [
            sys.executable,
            str(script_path),
            "--config",
            str(config_path),
            "recommend",
        ]
        proc = subprocess.run(
            recommend_cmd,
            cwd=str(hub_root),
            capture_output=True,
            text=True,
        )
        output = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
        if output:
            print(output)
        if proc.returncode != 0:
            print("[error] recommend failed", file=sys.stderr)
            return 1

    sync_cmd = build_figfix_clickup_sync_cmd(
        script_path=script_path,
        config_path=config_path,
        max_tasks=args.max_tasks,
        due_in_days=args.due_in_days,
        list_key=args.list_key,
        only_fig_id=args.only_fig_id,
        apply=args.apply,
    )
    proc = subprocess.run(
        sync_cmd,
        cwd=str(hub_root),
        capture_output=True,
        text=True,
    )
    output = ((proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")).strip()
    if output:
        print(output)
    if proc.returncode != 0:
        print("[error] clickup-sync failed", file=sys.stderr)
        return 1

    created = 0
    skipped = 0
    failed = 0
    m = re.search(r"- created:\s*(\d+)", output)
    if m:
        created = int(m.group(1))
    m = re.search(r"- skipped:\s*(\d+)", output)
    if m:
        skipped = int(m.group(1))
    m = re.search(r"- failed:\s*(\d+)", output)
    if m:
        failed = int(m.group(1))

    append_memory_entry(
        cfg,
        title="figfix sync",
        lines=[
            f"hub_root: {hub_root}",
            f"mode: {'apply' if args.apply else 'preview'}",
            f"recommend_first: {args.recommend_first}",
            f"list_key: {args.list_key or '(config default)'}",
            f"created={created} skipped={skipped} failed={failed}",
        ],
    )
    return 0


def command_review(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    auto_added, discovery_warnings = safe_enrich_config_lists(client, cfg)

    start_day = parse_date_yyyy_mm_dd(args.start) if args.start else date.today()
    horizon_days = args.days if args.days > 0 else cfg.review_horizon_days
    near_days = args.near_days if args.near_days > 0 else cfg.review_near_days
    max_findings = args.max_findings if args.max_findings > 0 else cfg.review_max_findings
    list_keys = args.list if args.list else sorted(cfg.lists.keys())

    snapshot = fetch_agenda(
        client=client,
        cfg=cfg,
        start_day=start_day,
        days=horizon_days,
        list_keys=list_keys,
    )
    if auto_added:
        snapshot.warnings.append(
            "auto discovered list keys: "
            + ", ".join(f"{k}={v}" for k, v in sorted(auto_added.items()))
        )
    snapshot.warnings.extend(discovery_warnings)
    overdue, unscheduled_due_soon, scheduled_after_due = detect_deadline_risks(
        snapshot,
        warn_days=cfg.deadline_warn_days,
    )
    findings = build_review_findings(
        snapshot=snapshot,
        cfg=cfg,
        near_days=near_days,
        max_findings=max_findings,
    )

    print(
        f"[review] range={snapshot.start_date.isoformat()}->{snapshot.end_date.isoformat()} "
        f"scheduled={len(snapshot.scheduled_tasks)}"
    )
    print(
        f"- pressure: overdue={len(overdue)} due_soon_unscheduled={len(unscheduled_due_soon)} "
        f"after_due={len(scheduled_after_due)}"
    )
    print(f"- findings: {len(findings)}")

    if not findings:
        print("[result] No significant issues found.")
    else:
        print("[result] Review candidates")
        for idx, item in enumerate(findings, start=1):
            print(
                f"{idx}. [{item['severity']}] [{item['type']}] "
                f"[{item['list_key']}] {item['task_name']} ({item['task_id']})"
            )
            print(f"   - reason: {item['reason']}")
            print(f"   - action: {item['action']}")
            if item.get("suggested_slot_start") and item.get("suggested_slot_end"):
                print(
                    f"   - suggestion: {item['suggested_slot_start']} -> {item['suggested_slot_end']}"
                )

    report = {
        "created_at": datetime.now().isoformat(),
        "range": {
            "start": snapshot.start_date.isoformat(),
            "end": snapshot.end_date.isoformat(),
            "near_days": near_days,
        },
        "counts": {
            "fetched": len(snapshot.fetched_tasks),
            "scheduled": len(snapshot.scheduled_tasks),
            "unscheduled": len(snapshot.unscheduled_tasks),
            "overdue": len(overdue),
            "due_soon_unscheduled": len(unscheduled_due_soon),
            "after_due": len(scheduled_after_due),
        },
        "findings": findings,
    }
    path = build_review_file_path()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"- report: {path}")

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))

    append_memory_entry(
        cfg,
        title="review check",
        lines=[
            f"range: {snapshot.start_date.isoformat()}->{snapshot.end_date.isoformat()}",
            f"pressure: overdue={len(overdue)} due_soon_unscheduled={len(unscheduled_due_soon)}",
            f"findings: {len(findings)} report={path}",
        ],
    )
    return 0


def command_carryover(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    auto_added, discovery_warnings = safe_enrich_config_lists(client, cfg)

    tz = ZoneInfo(cfg.timezone)
    today_local = datetime.now(tz).date()
    from_day = parse_date_yyyy_mm_dd(args.from_date) if args.from_date else (today_local - timedelta(days=1))

    list_keys = args.scan_list if args.scan_list else sorted(cfg.lists.keys())

    prev_snapshot = fetch_agenda(
        client=client,
        cfg=cfg,
        start_day=from_day,
        days=1,
        list_keys=list_keys,
    )
    if auto_added:
        prev_snapshot.warnings.append(
            "auto discovered list keys: "
            + ", ".join(f"{k}={v}" for k, v in sorted(auto_added.items()))
        )
    prev_snapshot.warnings.extend(discovery_warnings)
    carryover_targets = [
        t
        for t in prev_snapshot.scheduled_tasks
        if t.status_type.lower() != "closed" and t.movable
    ]
    carryover_targets = sorted(carryover_targets, key=lambda x: (x.end_at or datetime.max.replace(tzinfo=tz)))
    carryover_targets = carryover_targets[: args.max_tasks]

    future_snapshot = fetch_agenda(
        client=client,
        cfg=cfg,
        start_day=today_local,
        days=args.days if args.days > 0 else cfg.carryover_days,
        list_keys=list_keys,
    )
    future_snapshot.warnings.extend(discovery_warnings)

    reserved: List[Tuple[datetime, datetime]] = []
    proposals: List[dict] = []

    for task in carryover_targets:
        duration = task.duration_minutes or cfg.default_duration_minutes
        inferred = infer_task_policy(task.name, "auto", cfg)
        medium_flow_step = infer_medium_flow_step(task.name, task.list_key, cfg)
        medium_flow_start_after = infer_medium_flow_start_after(
            future_snapshot,
            cfg,
            medium_flow_step,
            reference_time=None,
        )
        medium_flow_end_before = infer_medium_flow_end_before(
            future_snapshot,
            cfg,
            medium_flow_step,
            reference_time=datetime.now(tz),
        )
        slots = find_free_slots(
            future_snapshot,
            cfg=cfg,
            duration_minutes=duration,
            list_key=task.list_key,
            extra_busy=reserved,
            start_after=medium_flow_start_after,
            max_slots=8,
        )
        if medium_flow_end_before:
            slots = [s for s in slots if s.slot_end <= medium_flow_end_before]
        if task.list_key == "experiment" and cfg.protect_experiment_structure and task.start_at and task.end_at:
            siblings = [
                s
                for s in future_snapshot.scheduled_tasks
                if s.task_id != task.task_id and s.list_key == "experiment" and s.start_at and s.end_at
            ]
            prev_end = max(
                (s.end_at for s in siblings if s.end_at <= task.start_at),
                default=None,
            )
            next_start = min(
                (s.start_at for s in siblings if s.start_at >= task.start_at),
                default=None,
            )
            constrained_slots = []
            for slot in slots:
                if prev_end and slot.slot_start < prev_end:
                    continue
                if next_start and slot.slot_end > next_start:
                    continue
                constrained_slots.append(slot)
            slots = constrained_slots
        ranked = select_best_slots(
            slots,
            due_at=task.due_at,
            priority_name=inferred.priority_name,
            schedule_bias_hours=inferred.schedule_bias_hours,
            max_candidates=1,
        )
        if not ranked:
            proposals.append(
                {
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "list_key": task.list_key,
                    "status": "no_slot",
                    "from_start": task.start_at.isoformat() if task.start_at else None,
                    "from_end": task.end_at.isoformat() if task.end_at else None,
                }
            )
            continue

        best = ranked[0]
        reserved.append((best.slot_start, best.slot_end))
        proposals.append(
            {
                "task_id": task.task_id,
                "task_name": task.name,
                "list_key": task.list_key,
                "status": "proposed",
                "from_start": task.start_at.isoformat() if task.start_at else None,
                "from_end": task.end_at.isoformat() if task.end_at else None,
                "to_start": best.slot_start.isoformat(),
                "to_end": best.slot_end.isoformat(),
                "task_url": task.url,
            }
        )

    plan = {
        "kind": "carryover",
        "created_at": datetime.now().isoformat(),
        "from_date": from_day.isoformat(),
        "window_start": today_local.isoformat(),
        "window_days": args.days if args.days > 0 else cfg.carryover_days,
        "proposals": proposals,
    }
    path = build_plan_file_path()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    proposed_count = sum(1 for x in proposals if x.get("status") == "proposed")
    missing_count = sum(1 for x in proposals if x.get("status") != "proposed")
    print(
        f"[carryover] from={from_day.isoformat()} targets={len(carryover_targets)} "
        f"proposed={proposed_count} no_slot={missing_count}"
    )
    print(f"- plan: {path}")
    for row in proposals:
        if row.get("status") != "proposed":
            print(f"  - [no_slot] [{row['list_key']}] {row['task_name']} ({row['task_id']})")
            continue
        print(
            "  - [proposed] "
            f"[{row['list_key']}] {row['task_name']} "
            f"{row['from_start']} -> {row['to_start']}"
        )

    moved = 0
    if args.commit:
        for row in proposals:
            if row.get("status") != "proposed":
                continue
            client.update_task_schedule(
                task_id=str(row["task_id"]),
                start_at=datetime.fromisoformat(row["to_start"]),
                end_at=datetime.fromisoformat(row["to_end"]),
            )
            moved += 1
        print(f"[carryover] committed moves: {moved}")
    else:
        print("[dry-run] no schedule updates applied. Add --commit to apply.")

    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    append_memory_entry(
        cfg,
        title="carryover planning",
        lines=[
            f"from: {from_day.isoformat()} targets={len(carryover_targets)} proposed={proposed_count}",
            f"committed: {moved}",
            f"plan: {path}",
        ],
    )
    return 0


def build_plan_file_path() -> Path:
    plan_dir = Path(__file__).resolve().parent / ".clickup_plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return plan_dir / f"plan-{stamp}.json"


def build_dashboard_file_path() -> Path:
    dash_dir = Path(__file__).resolve().parent / ".clickup_dashboards"
    dash_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return dash_dir / f"dashboard-{stamp}.md"


def build_review_file_path() -> Path:
    review_dir = Path(__file__).resolve().parent / ".clickup_reviews"
    review_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return review_dir / f"review-{stamp}.json"


def resolve_memory_path(cfg: HelperConfig) -> Path:
    raw = Path(cfg.memory_md_path).expanduser()
    if raw.is_absolute():
        return raw
    return Path(__file__).resolve().parent / raw


def append_memory_entry(cfg: HelperConfig, title: str, lines: Sequence[str]) -> None:
    if not cfg.memory_auto_append:
        return
    path = resolve_memory_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {ts} - {title}\n")
        for line in lines:
            handle.write(f"- {line}\n")


def suggest_later_slot(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    task: TaskWindow,
    start_after: datetime,
) -> Optional[SlotCandidate]:
    if not task.start_at:
        return None
    duration = task.duration_minutes or cfg.default_duration_minutes
    if duration <= 0:
        duration = cfg.default_duration_minutes

    slots = find_free_slots(
        snapshot=snapshot,
        cfg=cfg,
        duration_minutes=duration,
        list_key=task.list_key,
        exclude_task_ids={task.task_id},
        start_after=start_after,
        max_slots=1,
    )
    return slots[0] if slots else None


def build_workflow_order_findings(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    max_findings: int,
) -> List[dict]:
    if max_findings <= 0 or not cfg.workflow_enabled:
        return []

    grouped: Dict[str, List[Tuple[TaskWindow, WorkflowTag]]] = {}
    for task in snapshot.scheduled_tasks:
        if not (task.start_at and task.end_at):
            continue
        tag = infer_workflow_tag(task.name, task.list_key, cfg)
        if not tag or tag.order >= 999:
            continue
        grouped.setdefault(tag.group, []).append((task, tag))

    findings: List[dict] = []

    for group_name, items in grouped.items():
        ordered = sorted(items, key=lambda x: x[0].start_at)
        for idx, (task, tag) in enumerate(ordered):
            later_prereqs = [
                (other_task, other_tag)
                for other_task, other_tag in ordered[idx + 1 :]
                if other_tag.order < tag.order
            ]
            if not later_prereqs:
                continue

            stage_names = sorted({other_tag.stage for _, other_tag in later_prereqs})
            anchor = max((other_task.end_at for other_task, _ in later_prereqs if other_task.end_at), default=None)
            if anchor:
                anchor = anchor + timedelta(minutes=max(0, cfg.workflow_gap_minutes))
            suggestion_slot = suggest_later_slot(
                snapshot=snapshot,
                cfg=cfg,
                task=task,
                start_after=anchor or task.end_at or task.start_at,
            )
            findings.append(
                {
                    "severity": "medium",
                    "type": "workflow_order",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "list_key": task.list_key,
                    "workflow_group": group_name,
                    "workflow_stage": tag.stage,
                    "reason": (
                        f"Workflow order reversed: stages before stage={tag.stage} "
                        f"({', '.join(stage_names)}) are scheduled after it"
                    ),
                    "action": "Defer to align with workflow order",
                    "suggested_slot_start": suggestion_slot.slot_start.isoformat() if suggestion_slot else None,
                    "suggested_slot_end": suggestion_slot.slot_end.isoformat() if suggestion_slot else None,
                }
            )
            if len(findings) >= max_findings:
                return findings

    return findings


def build_medium_flow_gap_findings(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    max_findings: int,
) -> List[dict]:
    if max_findings <= 0 or not cfg.medium_flow_enabled:
        return []

    findings: List[dict] = []
    rule_map = medium_flow_rule_map(cfg)

    for task in sorted(
        snapshot.scheduled_tasks,
        key=lambda x: (x.start_at or datetime.max.replace(tzinfo=ZoneInfo(snapshot.timezone)), x.task_id),
    ):
        if not task.start_at:
            continue
        step_id = infer_medium_flow_step(task.name, task.list_key, cfg)
        if not step_id:
            continue

        rule = rule_map.get(step_id, {})
        if isinstance(rule, dict):
            min_start = infer_medium_flow_start_after(
                snapshot,
                cfg,
                step_id,
                reference_time=task.start_at,
            )
            preds = [str(x) for x in rule.get("predecessors", [])]
            pred_labels = [medium_flow_step_label(cfg, x) for x in preds]
            gap_minutes = int(rule.get("gap_minutes", 0))
            gap_text = f"{gap_minutes // 60}h" if gap_minutes % 60 == 0 else f"{gap_minutes}m"
        else:
            min_start = None
            preds = []
            pred_labels = []
            gap_minutes = 0
            gap_text = "0m"

        if min_start is not None and task.start_at < min_start:
            suggestion_slot = suggest_later_slot(
                snapshot=snapshot,
                cfg=cfg,
                task=task,
                start_after=min_start,
            )
            findings.append(
                {
                    "severity": "high",
                    "type": "medium_flow_gap",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "list_key": task.list_key,
                    "medium_flow_step": step_id,
                    "reason": (
                        f"{medium_flow_step_label(cfg, step_id)} must be scheduled at least "
                        f"{gap_text} after {' / '.join(pred_labels) if pred_labels else 'predecessor'}"
                    ),
                    "action": "Defer to respect chain order",
                    "suggested_slot_start": suggestion_slot.slot_start.isoformat() if suggestion_slot else None,
                    "suggested_slot_end": suggestion_slot.slot_end.isoformat() if suggestion_slot else None,
                }
            )
            if len(findings) >= max_findings:
                break

        end_before_detail = infer_medium_flow_end_before_detail(
            snapshot,
            cfg,
            step_id,
            reference_time=task.start_at,
        )
        end_before = end_before_detail["end_before"] if end_before_detail else None
        if end_before is not None and task.start_at > end_before:
            succ_step = str(end_before_detail.get("successor_step", "")) if end_before_detail else ""
            succ_label = medium_flow_step_label(cfg, succ_step) if succ_step else "successor task"
            succ_gap_minutes = int(end_before_detail.get("gap_minutes", 0)) if end_before_detail else 0
            succ_gap_text = (
                f"{succ_gap_minutes // 60}h"
                if succ_gap_minutes % 60 == 0
                else f"{succ_gap_minutes}m"
            )
            findings.append(
                {
                    "severity": "high",
                    "type": "medium_flow_before",
                    "task_id": task.task_id,
                    "task_name": task.name,
                    "list_key": task.list_key,
                    "medium_flow_step": step_id,
                    "reason": (
                        f"{medium_flow_step_label(cfg, step_id)} must be scheduled at least "
                        f"{succ_gap_text} before {succ_label}"
                    ),
                    "action": "Move earlier to respect chain order",
                    "suggested_slot_start": None,
                    "suggested_slot_end": None,
                }
            )
            if len(findings) >= max_findings:
                break

    return findings


def build_no_schedule_day_findings(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    max_findings: int,
) -> List[dict]:
    if max_findings <= 0:
        return []

    blocked_days = build_no_schedule_days(snapshot, cfg)
    protected_lists = set(cfg.no_schedule_day_list_keys)
    findings: List[dict] = []

    for task in sorted(
        snapshot.scheduled_tasks,
        key=lambda x: (x.start_at or datetime.max.replace(tzinfo=ZoneInfo(snapshot.timezone)), x.task_id),
    ):
        if not (task.start_at and task.end_at):
            continue
        day = task.start_at.date()
        if day not in blocked_days:
            continue
        if task.list_key in protected_lists:
            continue
        suggestion_slot = suggest_later_slot(
            snapshot=snapshot,
            cfg=cfg,
            task=task,
            start_after=datetime.combine(day, time.max, tzinfo=task.start_at.tzinfo),
        )
        findings.append(
            {
                "severity": "high",
                "type": "no_schedule_day_violation",
                "task_id": task.task_id,
                "task_name": task.name,
                "list_key": task.list_key,
                "reason": f"Task scheduled on competition day ({day.isoformat()})",
                "action": "Move to the day after competition day or later",
                "suggested_slot_start": suggestion_slot.slot_start.isoformat() if suggestion_slot else None,
                "suggested_slot_end": suggestion_slot.slot_end.isoformat() if suggestion_slot else None,
            }
        )
        if len(findings) >= max_findings:
            break

    return findings


def build_review_findings(
    snapshot: AgendaSnapshot,
    cfg: HelperConfig,
    near_days: int,
    max_findings: int,
) -> List[dict]:
    tz = ZoneInfo(snapshot.timezone)
    now_local = datetime.now(tz)
    near_until = now_local + timedelta(days=max(1, near_days))
    overdue, unscheduled_due_soon, scheduled_after_due = detect_deadline_risks(
        snapshot,
        warn_days=cfg.deadline_warn_days,
    )

    findings: List[dict] = []
    seen_task_ids: set = set()

    conflicts = detect_schedule_conflicts(snapshot.scheduled_tasks)
    for left, right in conflicts:
        if len(findings) >= max_findings:
            break
        if right.task_id in seen_task_ids:
            continue
        seen_task_ids.add(right.task_id)
        suggestion_slot = suggest_later_slot(
            snapshot=snapshot,
            cfg=cfg,
            task=right,
            start_after=right.end_at or now_local,
        )
        findings.append(
            {
                "severity": "high",
                "type": "overlap",
                "task_id": right.task_id,
                "task_name": right.name,
                "list_key": right.list_key,
                "reason": (
                    f"Overlaps with {left.name} at {left.start_at.strftime('%m-%d %H:%M')}"
                    if left.start_at
                    else "Overlaps with another task"
                ),
                "action": "Shift later",
                "suggested_slot_start": suggestion_slot.slot_start.isoformat() if suggestion_slot else None,
                "suggested_slot_end": suggestion_slot.slot_end.isoformat() if suggestion_slot else None,
            }
        )

    for task in scheduled_after_due:
        if len(findings) >= max_findings:
            break
        if task.task_id in seen_task_ids:
            continue
        seen_task_ids.add(task.task_id)
        findings.append(
            {
                "severity": "high",
                "type": "after_due",
                "task_id": task.task_id,
                "task_name": task.name,
                "list_key": task.list_key,
                "reason": (
                    f"End time {task.end_at.strftime('%m-%d %H:%M')} exceeds deadline {task.due_at.strftime('%m-%d %H:%M')}"
                    if task.end_at and task.due_at
                    else "Deadline exceeded"
                ),
                "action": "Move earlier or split",
                "suggested_slot_start": None,
                "suggested_slot_end": None,
            }
        )

    medium_flow_findings = build_medium_flow_gap_findings(
        snapshot=snapshot,
        cfg=cfg,
        max_findings=min(cfg.medium_flow_review_max_findings, max_findings),
    )
    for item in medium_flow_findings:
        if len(findings) >= max_findings:
            break
        task_id = str(item.get("task_id", ""))
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        findings.append(item)

    no_schedule_day_findings = build_no_schedule_day_findings(
        snapshot=snapshot,
        cfg=cfg,
        max_findings=max_findings,
    )
    for item in no_schedule_day_findings:
        if len(findings) >= max_findings:
            break
        task_id = str(item.get("task_id", ""))
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        findings.append(item)

    workflow_findings = build_workflow_order_findings(
        snapshot=snapshot,
        cfg=cfg,
        max_findings=min(cfg.workflow_review_max_findings, max_findings),
    )
    for item in workflow_findings:
        if len(findings) >= max_findings:
            break
        task_id = str(item.get("task_id", ""))
        if task_id in seen_task_ids:
            continue
        seen_task_ids.add(task_id)
        findings.append(item)

    backlog_pressure = len(overdue) + len(unscheduled_due_soon)
    low_categories = {"input", "figure"}

    near_tasks = sorted(
        [
            t
            for t in snapshot.scheduled_tasks
            if t.start_at and t.start_at <= near_until and t.start_at >= now_local
        ],
        key=lambda x: x.start_at,
    )
    for task in near_tasks:
        if len(findings) >= max_findings:
            break
        if task.task_id in seen_task_ids:
            continue
        policy = infer_task_policy(task.name, "auto", cfg)
        is_low = policy.priority_name == "low" or policy.category in low_categories
        if not is_low:
            continue
        if backlog_pressure <= 0:
            continue

        suggestion_slot = suggest_later_slot(
            snapshot=snapshot,
            cfg=cfg,
            task=task,
            start_after=near_until,
        )
        seen_task_ids.add(task.task_id)
        findings.append(
            {
                "severity": "medium",
                "type": "defer_low_priority",
                "task_id": task.task_id,
                "task_name": task.name,
                "list_key": task.list_key,
                "reason": (
                    f"Low-priority category ({policy.category}) with {backlog_pressure} deadline-pressure tasks pending"
                ),
                "action": "Consider deferring",
                "suggested_slot_start": suggestion_slot.slot_start.isoformat() if suggestion_slot else None,
                "suggested_slot_end": suggestion_slot.slot_end.isoformat() if suggestion_slot else None,
            }
        )

    for task in near_tasks:
        if len(findings) >= max_findings:
            break
        if task.task_id in seen_task_ids:
            continue
        if task.due_at is not None:
            continue
        policy = infer_task_policy(task.name, "auto", cfg)
        if policy.priority_name not in {"low", "normal"}:
            continue
        suggestion_slot = suggest_later_slot(
            snapshot=snapshot,
            cfg=cfg,
            task=task,
            start_after=now_local + timedelta(days=3),
        )
        seen_task_ids.add(task.task_id)
        findings.append(
            {
                "severity": "low",
                "type": "no_due_soon",
                "task_id": task.task_id,
                "task_name": task.name,
                "list_key": task.list_key,
                "reason": "Task without deadline is consuming a near-term slot",
                "action": "Move later if not urgent",
                "suggested_slot_start": suggestion_slot.slot_start.isoformat() if suggestion_slot else None,
                "suggested_slot_end": suggestion_slot.slot_end.isoformat() if suggestion_slot else None,
            }
        )

    return findings[:max_findings]


def detect_schedule_conflicts(tasks: Sequence[TaskWindow]) -> List[Tuple[TaskWindow, TaskWindow]]:
    conflicts: List[Tuple[TaskWindow, TaskWindow]] = []
    by_day: Dict[date, List[TaskWindow]] = {}

    for t in tasks:
        if not (t.start_at and t.end_at):
            continue
        by_day.setdefault(t.start_at.date(), []).append(t)

    for day_tasks in by_day.values():
        ordered = sorted(day_tasks, key=lambda x: x.start_at)
        for idx in range(len(ordered) - 1):
            left = ordered[idx]
            right = ordered[idx + 1]
            if left.end_at > right.start_at:
                conflicts.append((left, right))

    return conflicts


def detect_deadline_risks(
    snapshot: AgendaSnapshot,
    warn_days: int,
) -> Tuple[List[TaskWindow], List[TaskWindow], List[TaskWindow]]:
    tz = ZoneInfo(snapshot.timezone)
    now_local = datetime.now(tz)
    warn_until = now_local + timedelta(days=max(1, warn_days))

    overdue: List[TaskWindow] = []
    unscheduled_due_soon: List[TaskWindow] = []
    scheduled_after_due: List[TaskWindow] = []

    for task in snapshot.fetched_tasks:
        if not task.due_at:
            continue
        if task.status_type.lower() == "closed":
            continue
        if task.due_at < now_local:
            overdue.append(task)

    for task in snapshot.unscheduled_tasks:
        if not task.due_at:
            continue
        if task.status_type.lower() == "closed":
            continue
        if now_local <= task.due_at <= warn_until:
            unscheduled_due_soon.append(task)

    for task in snapshot.scheduled_tasks:
        if not task.due_at or not task.end_at:
            continue
        if task.status_type.lower() == "closed":
            continue
        if task.end_at > task.due_at:
            scheduled_after_due.append(task)

    return overdue, unscheduled_due_soon, scheduled_after_due


def render_month_table(
    year: int,
    month: int,
    task_count_by_day: Dict[date, int],
    overdue_days: set,
) -> List[str]:
    lines = [f"### {year}-{month:02d}", "| Mon | Tue | Wed | Thu | Fri | Sat | Sun |", "|---|---|---|---|---|---|---|"]
    cal = calendar.Calendar(firstweekday=0)

    for week in cal.monthdatescalendar(year, month):
        cells: List[str] = []
        for d in week:
            if d.month != month:
                cells.append(" ")
                continue
            count = task_count_by_day.get(d, 0)
            badge = f"{d.day}"
            if count > 0:
                badge += f" ({count})"
            if d in overdue_days:
                badge += " !"
            cells.append(badge)
        lines.append("| " + " | ".join(cells) + " |")

    return lines


def render_mermaid_gantt(tasks: Sequence[TaskWindow], max_items: int = 40) -> str:
    lines = [
        "```mermaid",
        "gantt",
        "    dateFormat  YYYY-MM-DD HH:mm",
        "    axisFormat  %m/%d %H:%M",
    ]
    grouped: Dict[str, List[TaskWindow]] = {}
    for task in tasks:
        if not (task.start_at and task.end_at):
            continue
        grouped.setdefault(task.list_key, []).append(task)

    rendered = 0
    for list_key in sorted(grouped.keys()):
        lines.append(f"    section {list_key}")
        for task in sorted(grouped[list_key], key=lambda x: x.start_at):
            if rendered >= max_items:
                break
            label = re.sub(r"[:,#]", " ", task.name).strip() or task.task_id
            label = label[:44]
            start_txt = task.start_at.strftime("%Y-%m-%d %H:%M")
            end_txt = task.end_at.strftime("%Y-%m-%d %H:%M")
            lines.append(f"    {label} : {task.task_id}, {start_txt}, {end_txt}")
            rendered += 1
        if rendered >= max_items:
            break

    lines.append("```")
    return "\n".join(lines)


def build_dashboard_markdown(
    snapshot: AgendaSnapshot,
    overdue: Sequence[TaskWindow],
    unscheduled_due_soon: Sequence[TaskWindow],
    scheduled_after_due: Sequence[TaskWindow],
    conflicts: Sequence[Tuple[TaskWindow, TaskWindow]],
) -> str:
    tz = ZoneInfo(snapshot.timezone)
    today = datetime.now(tz).date()
    task_count_by_day: Dict[date, int] = {}
    overdue_days = {t.due_at.date() for t in overdue if t.due_at}

    for task in snapshot.scheduled_tasks:
        if task.start_at:
            task_count_by_day[task.start_at.date()] = task_count_by_day.get(task.start_at.date(), 0) + 1

    months = []
    cursor = date(snapshot.start_date.year, snapshot.start_date.month, 1)
    end_month = date(snapshot.end_date.year, snapshot.end_date.month, 1)
    while cursor <= end_month:
        months.append((cursor.year, cursor.month))
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)

    today_tasks = [t for t in snapshot.scheduled_tasks if t.start_at and t.start_at.date() == today]

    lines: List[str] = []
    lines.append("# ClickUp Schedule Dashboard")
    lines.append(f"- range: {snapshot.start_date.isoformat()} -> {snapshot.end_date.isoformat()} ({snapshot.timezone})")
    lines.append(f"- fetched: {len(snapshot.fetched_tasks)} / scheduled: {len(snapshot.scheduled_tasks)} / unscheduled: {len(snapshot.unscheduled_tasks)}")
    lines.append(f"- overdue: {len(overdue)} / due_soon_unscheduled: {len(unscheduled_due_soon)} / after_due: {len(scheduled_after_due)} / overlaps: {len(conflicts)}")
    lines.append("")
    lines.append("## Today's Calendar")
    if today_tasks:
        for task in today_tasks:
            lines.append(
                f"- {task.start_at.strftime('%H:%M')} - {task.end_at.strftime('%H:%M')} [{task.list_key}] {task.name} ({task.task_id})"
            )
    else:
        lines.append("- no scheduled tasks today")
    lines.append("")

    lines.append("## Monthly Calendar")
    for year, month in months:
        lines.extend(render_month_table(year, month, task_count_by_day, overdue_days))
        lines.append("")

    lines.append("## Gantt")
    lines.append(render_mermaid_gantt(snapshot.scheduled_tasks))
    lines.append("")

    if overdue:
        lines.append("## Overdue Tasks")
        for task in overdue[:20]:
            due_txt = task.due_at.strftime("%Y-%m-%d %H:%M") if task.due_at else "?"
            lines.append(f"- [{task.list_key}] {task.name} due={due_txt} ({task.task_id})")
        lines.append("")

    if unscheduled_due_soon:
        lines.append("## Due Soon But Unscheduled")
        for task in unscheduled_due_soon[:20]:
            due_txt = task.due_at.strftime("%Y-%m-%d %H:%M") if task.due_at else "?"
            lines.append(f"- [{task.list_key}] {task.name} due={due_txt} ({task.task_id})")
        lines.append("")

    if conflicts:
        lines.append("## Overlap Warnings")
        for left, right in conflicts[:20]:
            lines.append(
                f"- {left.start_at.strftime('%Y-%m-%d %H:%M')} [{left.list_key}] {left.name} "
                f"overlaps [{right.list_key}] {right.name}"
            )
        lines.append("")

    if snapshot.warnings:
        lines.append("## Data Warnings")
        for warning in snapshot.warnings[:30]:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def command_advise(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    auto_added, discovery_warnings = safe_enrich_config_lists(client, cfg)

    start_day = parse_date_yyyy_mm_dd(args.start) if args.start else date.today()
    horizon_days = args.days if args.days > 0 else cfg.planning_days
    list_keys = args.scan_list if args.scan_list else sorted(cfg.lists.keys())

    snapshot = fetch_agenda(
        client=client,
        cfg=cfg,
        start_day=start_day,
        days=horizon_days,
        list_keys=list_keys,
    )
    if auto_added:
        snapshot.warnings.append(
            "auto discovered list keys: "
            + ", ".join(f"{k}={v}" for k, v in sorted(auto_added.items()))
        )
    snapshot.warnings.extend(discovery_warnings)

    policy = infer_task_policy(args.text, args.priority, cfg)
    parsed_duration = args.duration or parse_duration_from_text(args.text) or 0
    if parsed_duration > 0:
        duration = parsed_duration
    elif policy.category == "input":
        duration = cfg.input_default_duration_minutes
    else:
        duration = cfg.default_duration_minutes
    if duration <= 0:
        print("[error] duration must be >= 1 minute", file=sys.stderr)
        return 2

    priority_name = policy.priority_name
    target_list = args.list
    if target_list == "auto":
        target_list = cfg.category_default_list.get(policy.category, "other")
    target_list, reroute_warn = resolve_write_list_key(
        target_list,
        cfg,
        fallback_hint=cfg.category_default_list.get(policy.category, "other"),
    )
    if target_list not in cfg.lists:
        print(f"[error] target list key not configured: {target_list}", file=sys.stderr)
        return 2
    tz = ZoneInfo(snapshot.timezone)
    now_local = datetime.now(tz)
    workflow_tag = infer_workflow_tag(args.text, target_list, cfg)
    workflow_start_after = infer_workflow_start_after(snapshot, cfg, workflow_tag)
    medium_flow_step = infer_medium_flow_step(args.text, target_list, cfg)
    medium_flow_start_after = infer_medium_flow_start_after(
        snapshot,
        cfg,
        medium_flow_step,
        reference_time=None,
    )
    medium_flow_end_before = infer_medium_flow_end_before(
        snapshot,
        cfg,
        medium_flow_step,
        reference_time=now_local,
    )
    combined_start_after = max(
        [x for x in [workflow_start_after, medium_flow_start_after] if x],
        default=None,
    )
    split_rule = infer_split_task_rule(args.text, target_list, cfg)
    wait_task_like = infer_wait_task_like(args.text, split_rule)
    jog_intervals = collect_keyword_task_intervals(
        snapshot.scheduled_tasks,
        cfg.jog_keyword_patterns,
    )
    prefer_jog_windows = cfg.wait_task_use_jog_windows and wait_task_like and bool(jog_intervals)
    split_total_duration = 0
    if split_rule and isinstance(split_rule.get("segments"), list):
        for seg in split_rule.get("segments", []):
            if not isinstance(seg, dict):
                continue
            try:
                split_total_duration += int(seg.get("duration_minutes", 0))
            except (TypeError, ValueError):
                continue
        if parsed_duration <= 0 and split_total_duration > 0:
            duration = split_total_duration

    due_at = None
    if args.due:
        due_day = parse_date_yyyy_mm_dd(args.due)
        due_at = datetime.combine(due_day, time(hour=23, minute=59), tzinfo=tz)

    if split_rule is not None:
        direct_slots = find_split_slots(
            snapshot=snapshot,
            cfg=cfg,
            list_key=target_list,
            base_name=args.text,
            split_rule=split_rule,
            start_after=combined_start_after,
            max_slots=20,
        )
    else:
        direct_slots = find_free_slots(
            snapshot,
            cfg=cfg,
            duration_minutes=duration,
            list_key=target_list,
            start_after=combined_start_after,
            max_slots=20,
        )
    if medium_flow_end_before:
        direct_slots = [s for s in direct_slots if s.slot_end <= medium_flow_end_before]
    direct_best = select_best_slots(
        direct_slots,
        due_at=due_at,
        priority_name=priority_name,
        schedule_bias_hours=policy.schedule_bias_hours,
        max_candidates=cfg.max_candidates,
        prefer_tight_gap=(policy.category == "input"),
        prefer_around_jog_for_wait=prefer_jog_windows,
        jog_intervals=jog_intervals,
        jog_window_minutes=cfg.wait_task_jog_window_minutes,
    )

    if split_rule is not None or policy.category == "input":
        rearranged = []
    else:
        rearranged = build_rearrangement_candidates(
            snapshot,
            cfg=cfg,
            duration_minutes=duration,
            list_key=target_list,
            due_at=due_at,
            priority_name=priority_name,
            schedule_bias_hours=policy.schedule_bias_hours,
            start_after=combined_start_after,
            max_candidates=max(1, cfg.max_candidates - 1),
            prefer_around_jog_for_wait=prefer_jog_windows,
            jog_intervals=jog_intervals,
            jog_window_minutes=cfg.wait_task_jog_window_minutes,
        )
        if medium_flow_end_before:
            rearranged = [s for s in rearranged if s.slot_end <= medium_flow_end_before]

    candidates = direct_best + rearranged
    candidates = sorted(candidates, key=lambda c: c.score)[: cfg.max_candidates]

    print(f"[advise] text={args.text}")
    print(f"- duration: {duration} min")
    print(f"- category: {policy.category} (matched={', '.join(policy.matched_keywords) if policy.matched_keywords else '-'})")
    print(f"- priority: {priority_name} (bias={policy.schedule_bias_hours:+}h)")
    print(f"- target list: {target_list}")
    if reroute_warn:
        print(f"- note: {reroute_warn}")
    if workflow_tag:
        print(
            f"- workflow: group={workflow_tag.group} stage={workflow_tag.stage} "
            f"order={workflow_tag.order}"
        )
        if workflow_tag.matched_keywords:
            print(f"  - workflow matched: {', '.join(workflow_tag.matched_keywords)}")
        if workflow_start_after:
            print(f"  - workflow start_after: {workflow_start_after.isoformat()}")
    if medium_flow_step:
        print(
            f"- medium-flow: step={medium_flow_step} "
            f"({medium_flow_step_label(cfg, medium_flow_step)})"
        )
        if medium_flow_start_after:
            print(f"  - medium-flow start_after: {medium_flow_start_after.isoformat()}")
        if medium_flow_end_before:
            print(f"  - medium-flow end_before: {medium_flow_end_before.isoformat()}")
    if split_rule:
        print(f"- split-task rule: {split_rule.get('id', 'unknown')}")
    if cfg.wait_task_use_jog_windows and wait_task_like:
        if jog_intervals:
            print(
                f"- jog-window optimization: enabled "
                f"(anchors={len(jog_intervals)}, window={cfg.wait_task_jog_window_minutes}m)"
            )
        else:
            print("- jog-window optimization: waiting-task detected, but no jog anchors found")
    print(f"- due: {args.due or '(none)'}")
    if has_deadline_signal(args.text, cfg) and not args.due:
        print("- note: deadline signal detected (feedback/review type). --due recommended")
    blocked_days = build_no_schedule_days(snapshot, cfg)
    if blocked_days:
        print(f"- blocked no-schedule days: {len(blocked_days)}")
    print(f"- scheduled tasks scanned: {len(snapshot.scheduled_tasks)} in {horizon_days} days")
    print(f"- warnings: {len(snapshot.warnings)}")

    if snapshot.warnings and args.verify:
        print("- verify warnings:")
        for w in snapshot.warnings[:20]:
            print(f"  - {w}")

    if not candidates:
        print("[result] No candidates found. Try increasing --days.")
        return 1

    print("[result] Candidates")
    for idx, cand in enumerate(candidates, start=1):
        if cand.label == "direct":
            label = "Add to free slot"
        elif cand.label == "split":
            label = "Split slot"
        else:
            label = "Rearrange"
        print(
            f"{idx}. {label}: {cand.slot_start.strftime('%Y-%m-%d %H:%M')} - "
            f"{cand.slot_end.strftime('%H:%M')} [{cand.list_key}]"
        )
        for r in cand.reasons:
            print(f"   - reason: {r}")
        for seg in cand.segments:
            print(
                "   - segment: "
                f"{seg.get('task_name', '')} "
                f"{seg.get('slot_start', '')} -> {seg.get('slot_end', '')}"
            )
        for move in cand.moves:
            print(
                "   - move: "
                f"{move['task_name']} ({move['task_id']}) "
                f"{move['from_start']} -> {move['to_start']}"
            )

    plan = {
        "created_at": datetime.now().isoformat(),
        "text": args.text,
        "duration_minutes": duration,
        "target_list": target_list,
        "due": args.due,
        "category": policy.category,
        "priority": priority_name,
        "schedule_bias_hours": policy.schedule_bias_hours,
        "matched_keywords": policy.matched_keywords,
        "workflow": (
            {
                "group": workflow_tag.group,
                "stage": workflow_tag.stage,
                "order": workflow_tag.order,
                "matched_keywords": workflow_tag.matched_keywords,
                "start_after": workflow_start_after.isoformat() if workflow_start_after else None,
            }
            if workflow_tag
            else None
        ),
        "medium_flow": (
            {
                "step": medium_flow_step,
                "step_label": medium_flow_step_label(cfg, medium_flow_step) if medium_flow_step else "",
                "start_after": medium_flow_start_after.isoformat() if medium_flow_start_after else None,
                "end_before": medium_flow_end_before.isoformat() if medium_flow_end_before else None,
            }
            if medium_flow_step
            else None
        ),
        "split_task_rule": (
            {
                "id": str(split_rule.get("id", "")),
                "segments": split_rule.get("segments", []),
            }
            if split_rule
            else None
        ),
        "wait_task_like": wait_task_like,
        "jog_window_optimization": {
            "enabled": bool(prefer_jog_windows),
            "anchor_count": len(jog_intervals),
            "window_minutes": cfg.wait_task_jog_window_minutes,
        },
        "timezone": snapshot.timezone,
        "snapshot": {
            "start_date": snapshot.start_date.isoformat(),
            "end_date": snapshot.end_date.isoformat(),
            "warnings": snapshot.warnings,
        },
        "candidates": [
            {
                "label": c.label,
                "slot_start": c.slot_start.isoformat(),
                "slot_end": c.slot_end.isoformat(),
                "list_key": c.list_key,
                "score": c.score,
                "reasons": c.reasons,
                "moves": c.moves,
                "segments": c.segments,
            }
            for c in candidates
        ],
    }

    path = build_plan_file_path()
    with path.open("w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"[plan] {path}")

    if args.json:
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    append_memory_entry(
        cfg,
        title="advise consultation",
        lines=[
            f"text: {args.text}",
            f"category={policy.category} priority={priority_name} bias={policy.schedule_bias_hours:+}h",
            f"target_list={target_list} due={args.due or '(none)'} duration={duration}m",
            (
                "workflow="
                + (
                    f"{workflow_tag.group}/{workflow_tag.stage}/order{workflow_tag.order}"
                    if workflow_tag
                    else "none"
                )
            ),
            (
                "medium_flow="
                + (
                    (
                        f"{medium_flow_step}@{medium_flow_start_after.isoformat()}"
                        if medium_flow_step and medium_flow_start_after
                        else (medium_flow_step or "none")
                    )
                    + (
                        f"<= {medium_flow_end_before.isoformat()}"
                        if medium_flow_step and medium_flow_end_before
                        else ""
                    )
                )
            ),
            f"candidates={len(candidates)} plan={path}",
            "fixed_rules=" + ", ".join(cfg.fixed_list_keys),
            "rearrange_scope="
            + (", ".join(cfg.rearrange_allowed_list_keys) if cfg.rearrange_allowed_list_keys else f"{target_list} only"),
            f"protect_experiment_structure={cfg.protect_experiment_structure}",
        ],
    )

    return 0


def command_book(args: argparse.Namespace, cfg: HelperConfig) -> int:
    token = _load_clickup_token()
    client = ClickUpClient(token)
    _, discovery_warnings = safe_enrich_config_lists(client, cfg)

    path = Path(args.plan).expanduser().resolve()
    if not path.exists():
        print(f"[error] plan not found: {path}", file=sys.stderr)
        return 2

    plan = load_json_file(path)
    candidates = plan.get("candidates") or []
    idx = args.candidate - 1
    if idx < 0 or idx >= len(candidates):
        print(f"[error] invalid candidate index: {args.candidate}", file=sys.stderr)
        return 2

    cand = candidates[idx]
    list_key = str(cand.get("list_key") or plan.get("target_list") or "experiment")
    list_key, reroute_warn = resolve_write_list_key(
        list_key,
        cfg,
        fallback_hint=cfg.category_default_list.get("general", "other"),
    )
    list_id = cfg.lists.get(list_key)
    if not list_id:
        for w in discovery_warnings[:3]:
            print(f"[warn] {w}")
        print(f"[error] list key not configured: {list_key}", file=sys.stderr)
        return 2

    start_at = datetime.fromisoformat(cand["slot_start"])
    end_at = datetime.fromisoformat(cand["slot_end"])
    task_name = args.name or str(plan.get("text") or "New task")
    description = args.description or ""
    segments = cand.get("segments") or []

    print("[book] plan summary")
    print(f"- plan: {path}")
    print(f"- candidate: {args.candidate}")
    print(f"- task: {task_name}")
    print(f"- slot: {start_at.isoformat()} -> {end_at.isoformat()}")
    print(f"- list: {list_key}")
    if reroute_warn:
        print(f"- note: {reroute_warn}")
    if segments:
        print(f"- split segments: {len(segments)}")
        for seg in segments:
            print(
                "  - "
                f"{seg.get('task_name', '')} "
                f"{seg.get('slot_start', '')} -> {seg.get('slot_end', '')}"
            )
    print(f"- apply_moves: {args.apply_moves}")
    print(f"- commit: {args.commit}")

    if not args.commit:
        print("[dry-run] No API write executed. Re-run with --commit to apply.")
        return 0

    if segments:
        for seg in segments:
            seg_start = datetime.fromisoformat(str(seg["slot_start"]))
            seg_end = datetime.fromisoformat(str(seg["slot_end"]))
            seg_name = str(seg.get("task_name", "")).strip() or task_name
            result = client.create_task(
                list_id=list_id,
                name=seg_name,
                start_at=seg_start,
                end_at=seg_end,
                description=description,
            )
            print(f"[created] {result.get('name', seg_name)}")
            print(f"URL: {result.get('url', '')}")
    else:
        result = client.create_task(
            list_id=list_id,
            name=task_name,
            start_at=start_at,
            end_at=end_at,
            description=description,
        )
        print(f"[created] {result.get('name', task_name)}")
        print(f"URL: {result.get('url', '')}")

    if args.apply_moves:
        for move in cand.get("moves") or []:
            move_start = datetime.fromisoformat(move["to_start"])
            move_end = datetime.fromisoformat(move["to_end"])
            task_id = str(move["task_id"])
            client.update_task_schedule(task_id=task_id, start_at=move_start, end_at=move_end)
            print(
                f"[moved] {move.get('task_name', task_id)} "
                f"{move.get('from_start')} -> {move.get('to_start')}"
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ClickUp task scheduler helper")
    parser.add_argument(
        "--config",
        default="",
        help="optional JSON config path (defaults to scripts/clickup_helper_config.json)",
    )

    sub = parser.add_subparsers(dest="command")

    add_parser = sub.add_parser("add", help="add one scheduled task")
    add_parser.add_argument("--name", required=True, help="task name")
    add_parser.add_argument(
        "--list",
        default="experiment",
        help="list key (plan/input/experiment/code/manuscript/slide/meeting/competition/other/daily)",
    )
    add_parser.add_argument("--date", default=None, help="YYYY-MM-DD (JST)")
    add_parser.add_argument(
        "--time", default=None, choices=["morning", "afternoon", "default"], help="time preset"
    )
    add_parser.add_argument("--start", type=int, default=None, help="start hour")
    add_parser.add_argument("--duration", type=int, default=None, help="duration hour")
    add_parser.add_argument("--description", default="", help="description")
    add_parser.add_argument("--dry-run", action="store_true", help="show payload only")
    add_parser.set_defaults(func=command_add)

    doctor_parser = sub.add_parser("doctor", help="verify auth and task retrieval")
    doctor_parser.add_argument(
        "--list",
        action="append",
        default=[],
        help="target list key (repeatable). default: all configured lists",
    )
    doctor_parser.add_argument("--sample", type=int, default=3, help="sample size per list")
    doctor_parser.add_argument(
        "--probe-tasks",
        type=int,
        default=50,
        help="number of tasks to probe per list on first page",
    )
    doctor_parser.add_argument("--deep", action="store_true", help="fetch each sampled task detail")
    doctor_parser.add_argument(
        "--discover",
        action="store_true",
        help="also enumerate workspace folders/lists and show inferred list keys",
    )
    doctor_parser.add_argument("--json", action="store_true", help="print JSON report")
    doctor_parser.set_defaults(func=command_doctor)

    agenda_parser = sub.add_parser("agenda", help="read existing scheduled tasks")
    agenda_parser.add_argument("--start", default="", help="YYYY-MM-DD (default=today)")
    agenda_parser.add_argument("--days", type=int, default=7, help="horizon days")
    agenda_parser.add_argument(
        "--list",
        action="append",
        default=[],
        help="target list key (repeatable). default: all configured lists",
    )
    agenda_parser.add_argument("--preview", type=int, default=12, help="preview count")
    agenda_parser.add_argument("--verify", action="store_true", help="show validation warnings")
    agenda_parser.add_argument("--json", action="store_true", help="print JSON result")
    agenda_parser.set_defaults(func=command_agenda)

    dashboard_parser = sub.add_parser(
        "dashboard",
        help="generate monthly+today+gantt dashboard and conflict/deadline checks",
    )
    dashboard_parser.add_argument("--start", default="", help="YYYY-MM-DD (default=today)")
    dashboard_parser.add_argument("--days", type=int, default=0, help="horizon days (default from config)")
    dashboard_parser.add_argument(
        "--list",
        action="append",
        default=[],
        help="target list key (repeatable). default: all configured lists",
    )
    dashboard_parser.add_argument("--print", action="store_true", help="print markdown body to stdout")
    dashboard_parser.set_defaults(func=command_dashboard)

    figfix_parser = sub.add_parser(
        "figfix-sync",
        help="sync figure-fix tracker recommendations into ClickUp (preview/apply)",
    )
    figfix_parser.add_argument(
        "--hub-root",
        default="",
        help="figure-hub root path (default: FIGURE_HUB_ROOT or ~/Desktop/figure-hub)",
    )
    figfix_parser.add_argument(
        "--figure-hub-config",
        default="",
        help="explicit figure-hub config path (default: <hub-root>/config.json if exists)",
    )
    figfix_parser.add_argument(
        "--list-key",
        default="",
        help="target ClickUp list key override (default from figure-hub config)",
    )
    figfix_parser.add_argument(
        "--due-in-days",
        type=int,
        default=7,
        help="due date offset in days for created tasks",
    )
    figfix_parser.add_argument(
        "--max-tasks",
        type=int,
        default=20,
        help="max figure-fix tasks per run",
    )
    figfix_parser.add_argument(
        "--only-fig-id",
        default="",
        help="limit to one fig_id",
    )
    figfix_parser.add_argument(
        "--no-recommend",
        dest="recommend_first",
        action="store_false",
        help="skip recommend step and use existing recommendations.json",
    )
    figfix_parser.add_argument(
        "--apply",
        action="store_true",
        help="actually create tasks (default: preview only)",
    )
    figfix_parser.set_defaults(func=command_figfix_sync, recommend_first=True)

    review_parser = sub.add_parser(
        "review",
        help="review existing schedule and suggest what looks odd or should be moved later",
    )
    review_parser.add_argument("--start", default="", help="YYYY-MM-DD (default=today)")
    review_parser.add_argument("--days", type=int, default=0, help="horizon days (default from config)")
    review_parser.add_argument("--near-days", type=int, default=0, help="near-term window days (default from config)")
    review_parser.add_argument(
        "--list",
        action="append",
        default=[],
        help="target list key (repeatable). default: all configured lists",
    )
    review_parser.add_argument("--max-findings", type=int, default=0, help="max findings to show")
    review_parser.add_argument("--json", action="store_true", help="print JSON report")
    review_parser.set_defaults(func=command_review)

    advise_parser = sub.add_parser("advise", help="consult best slot for a new issue/task")
    advise_parser.add_argument("--text", required=True, help="issue/task description text")
    advise_parser.add_argument("--list", default="auto", help="target list key for new task (or auto)")
    advise_parser.add_argument("--start", default="", help="YYYY-MM-DD (default=today)")
    advise_parser.add_argument("--days", type=int, default=0, help="horizon days (default from config)")
    advise_parser.add_argument("--due", default="", help="deadline YYYY-MM-DD")
    advise_parser.add_argument("--duration", type=int, default=0, help="required minutes")
    advise_parser.add_argument("--priority", default="auto", help="auto/urgent/high/normal/low")
    advise_parser.add_argument(
        "--scan-list",
        action="append",
        default=[],
        help="which lists to scan as existing schedule (repeatable). default: all",
    )
    advise_parser.add_argument("--verify", action="store_true", help="print validation warnings")
    advise_parser.add_argument("--json", action="store_true", help="print JSON plan")
    advise_parser.set_defaults(func=command_advise)

    carryover_parser = sub.add_parser(
        "carryover",
        help="reschedule unfinished tasks from the previous day (safe dry-run by default)",
    )
    carryover_parser.add_argument("--from-date", default="", help="YYYY-MM-DD (default=yesterday)")
    carryover_parser.add_argument("--days", type=int, default=0, help="planning window days (default from config)")
    carryover_parser.add_argument(
        "--scan-list",
        action="append",
        default=[],
        help="which lists to scan (repeatable). default: all",
    )
    carryover_parser.add_argument("--max-tasks", type=int, default=12, help="max carryover tasks to consider")
    carryover_parser.add_argument("--commit", action="store_true", help="apply schedule updates")
    carryover_parser.add_argument("--json", action="store_true", help="print JSON plan")
    carryover_parser.set_defaults(func=command_carryover)

    book_parser = sub.add_parser(
        "book",
        help="apply one candidate from advise plan (safe by default: dry-run)",
    )
    book_parser.add_argument("--plan", required=True, help="plan JSON path from advise command")
    book_parser.add_argument("--candidate", type=int, default=1, help="candidate index (1-based)")
    book_parser.add_argument("--name", default="", help="override task name")
    book_parser.add_argument("--description", default="", help="task description")
    book_parser.add_argument(
        "--apply-moves",
        action="store_true",
        help="also move tasks suggested by the selected candidate",
    )
    book_parser.add_argument("--commit", action="store_true", help="execute write API calls")
    book_parser.set_defaults(func=command_book)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        cfg = load_runtime_config(args.config)
        return args.func(args, cfg)
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
