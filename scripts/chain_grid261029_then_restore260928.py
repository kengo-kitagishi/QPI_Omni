"""After the 261029 grid recon + calib (+ raw deletion) finishes, restore 260928 from F: to E:/D:.

Waits for run_recon_calib_261029_2per.py to exit. Continues only if its log says the batch is OK
and the raw was deleted (so E: has room); otherwise writes an alert and stops.
The restore plan of 2026-10-29 20:45 was made before the grid used E:, so it is rebuilt from the
current free space (units already copied to E: are kept where they are).
"""
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GRID_LOG = Path(r"E:\261029\recon_calib_2per_grid.log")
PLAN = Path(r"C:\Users\QPI\Documents\QPI_Omni\restore_260928_PLAN.json")
ALERT = Path(r"C:\Users\QPI\Documents\QPI_Omni\chain_261029.ALERT.txt")


def watcher_running():
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
                          "Where-Object { $_.CommandLine -match 'run_recon_calib_261029_2per' } | "
                          "Measure-Object | Select-Object -ExpandProperty Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def alert(msg):
    ALERT.write_text(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n", encoding="utf-8")
    print(msg, flush=True)


def main():
    while watcher_running():
        time.sleep(120)
    text = GRID_LOG.read_text(encoding="utf-8", errors="replace") if GRID_LOG.exists() else ""
    if "batch b1: OK" not in text or "raw deleted" not in text:
        alert("261029 grid recon did not finish cleanly (or raw not deleted); 260928 restore NOT started. "
              f"See {GRID_LOG}")
        sys.exit(1)
    if PLAN.exists():
        plan = json.loads(PLAN.read_text(encoding="utf-8"))
        # keep units that already reached E:, re-plan the rest from today's free space
        done_e = [p for p in plan if p["dst"].startswith("E:") and Path(p["dst"]).exists()]
        PLAN.rename(PLAN.with_suffix(".before_grid.json"))
        print(f"old plan moved aside; {len(done_e)} units already on E:", flush=True)
    rc = subprocess.run([sys.executable, str(SCRIPT_DIR / "restore_260928_from_f_261029.py")]).returncode
    if rc != 0:
        alert(f"260928 restore finished with rc={rc}; see restore_260928_STATUS.txt. F: must NOT be formatted.")
    else:
        alert("260928 restored and verified on E:/D:. F: can now be formatted "
              "(scripts/format_f_exfat_128k.ps1, as administrator).")
    sys.exit(rc)


if __name__ == "__main__":
    main()
