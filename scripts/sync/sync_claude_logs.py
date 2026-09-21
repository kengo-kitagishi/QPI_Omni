"""
sync_claude_logs.py
Sync Claude and Codex conversation logs (.jsonl) to Google Drive daily.

Registered as the scheduled task "SyncClaudeLogs" (see register_task.ps1).

Layout on Drive:
  claude_logs/QPI_Omni/*.jsonl                    (this project's Claude sessions)
  codex_logs/YYYY/MM/DD/rollout-*.jsonl           (Codex sessions, shared tree)
  codex_logs/session_index_<HOST>.jsonl           (per-machine, other PCs also write here)
  codex_logs/history_<HOST>.jsonl
"""
import datetime
import os
import shutil
from pathlib import Path

CLAUDE_SRC = Path(r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni")
CLAUDE_DST = Path(r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\claude_logs\QPI_Omni")

CODEX_SRC = Path(r"C:\Users\QPI\.codex")
CODEX_DST = Path(r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\codex_logs")

LOG = Path(r"C:\Users\QPI\.claude\sync_log.txt")
HOST = os.environ.get("COMPUTERNAME", "unknown")


def log_line(text: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def copy_if_newer(src_file: Path, dst_file: Path) -> bool:
    """Copy only when the destination is missing or older. Returns True if copied."""
    if dst_file.exists() and src_file.stat().st_mtime <= dst_file.stat().st_mtime:
        return False
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_file, dst_file)
    return True


def sync_claude() -> str:
    if not CLAUDE_SRC.exists():
        return f"claude[ERROR] source not found: {CLAUDE_SRC}"

    CLAUDE_DST.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0
    for src_file in CLAUDE_SRC.glob("*.jsonl"):
        if copy_if_newer(src_file, CLAUDE_DST / src_file.name):
            copied += 1
        else:
            skipped += 1
    return f"claude[OK] copied={copied} skipped={skipped}"


def sync_codex() -> str:
    if not CODEX_SRC.exists():
        return f"codex[SKIP] source not found: {CODEX_SRC}"

    CODEX_DST.mkdir(parents=True, exist_ok=True)
    copied = skipped = 0

    # Session rollouts keep their YYYY/MM/DD tree. File names carry a UUID,
    # so several machines can share the same tree without colliding.
    sessions = CODEX_SRC / "sessions"
    if sessions.exists():
        for src_file in sessions.rglob("*.jsonl"):
            rel = src_file.relative_to(sessions)
            if copy_if_newer(src_file, CODEX_DST / rel):
                copied += 1
            else:
                skipped += 1

    # These two are per-machine and would clobber another PC's copy, so tag
    # them with the host name.
    for name in ("session_index.jsonl", "history.jsonl"):
        src_file = CODEX_SRC / name
        if not src_file.exists():
            continue
        dst_file = CODEX_DST / f"{src_file.stem}_{HOST}{src_file.suffix}"
        if copy_if_newer(src_file, dst_file):
            copied += 1
        else:
            skipped += 1

    return f"codex[OK] copied={copied} skipped={skipped}"


def main() -> None:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = [sync_claude(), sync_codex()]
    log_line(f"{timestamp} {' | '.join(results)}")


if __name__ == "__main__":
    main()
