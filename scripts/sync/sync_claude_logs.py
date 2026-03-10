"""
sync_claude_logs.py
Claude会話ログ（.jsonl）をGoogle Driveに毎日同期する
"""
import shutil
import datetime
from pathlib import Path

SRC = Path(r"C:\Users\QPI\.claude\projects\C--Users-QPI-Documents-QPI-Omni")
DST = Path(r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\claude_logs\QPI_Omni")
LOG = Path(r"C:\Users\QPI\.claude\sync_log.txt")

def main():
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not SRC.exists():
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} [ERROR] Source not found: {SRC}\n")
        return

    DST.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for src_file in SRC.glob("*.jsonl"):
        dst_file = DST / src_file.name
        # タイムスタンプで差分コピー（新しいものだけ上書き）
        if not dst_file.exists() or src_file.stat().st_mtime > dst_file.stat().st_mtime:
            shutil.copy2(src_file, dst_file)
            copied += 1
        else:
            skipped += 1

    status = f"copied={copied} skipped={skipped}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} [OK] {status} -> {DST}\n")
    print(f"{timestamp} [OK] {status}")

if __name__ == "__main__":
    main()
