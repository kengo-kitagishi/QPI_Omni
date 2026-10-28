"""Copy every data folder on D: and E: to the new external HDD F:, one robocopy job at a time.

Layout keeps the source path, so nothing from D: and E: lands in the same folder:
    E:\\261024\\...                         -> F:\\from_E\\261024\\...
    D:\\AquisitionData\\Kitagishi\\260908\\... -> F:\\from_D\\AquisitionData\\Kitagishi\\260908\\...

Copy only: nothing is deleted. Re-running skips files robocopy already copied (same size and
time), so an interrupted run resumes where it stopped.

Status of every job (pending / running / done / FAILED) is rewritten to STATUS_PATH after each
job, so "which ones are finished" is one file to open.
"""
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DEST_ROOT = Path("F:/")
LOG_DIR = Path(r"F:\_backup_logs_261028")
STATUS_PATH = Path(r"C:\Users\QPI\Documents\QPI_Omni\backup_d_e_to_f_261028_STATUS.txt")

# E: first: it is full (0 GB free) and the next timelapse writes there.
SOURCES = [
    r"E:\261024",
    r"E:\261004",
    r"E:\260928",
    r"E:\260924",
    r"E:\260927",
    r"E:\260917",
    r"E:\260908_whitelist_realign",
    r"E:\260908_outside_quad",
    r"E:\260908_seg",
    r"E:\Acuisition",
    r"D:\AquisitionData\Kitagishi\261024",
    r"D:\AquisitionData\Kitagishi\261004",
    r"D:\AquisitionData\Kitagishi\260928",
    r"D:\AquisitionData\Kitagishi\260927",
    r"D:\AquisitionData\Kitagishi\260924",
    r"D:\AquisitionData\Kitagishi\260922",
    r"D:\AquisitionData\Kitagishi\260917",
    r"D:\AquisitionData\Kitagishi\260908",
    r"D:\AquisitionData\Kitagishi\260729",
    r"D:\AquisitionData\Kitagishi\basler_image_seq",
]

ROBOCOPY_FLAGS = ["/E", "/COPY:DAT", "/DCOPY:T", "/R:2", "/W:5", "/MT:8",
                  "/NP", "/NFL", "/NDL", "/BYTES"]

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def dest_for(src):
    p = Path(src)
    return DEST_ROOT / f"from_{p.drive[0]}" / p.relative_to(p.anchor)


def summary_from_log(log_path):
    """Pull the 'Files :' and 'Bytes :' totals out of a robocopy log."""
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    out = {}
    for key in ("Files", "Bytes"):
        m = re.findall(rf"^\s*{key} :\s+(.+)$", text, flags=re.M)
        if m:
            out[key] = " ".join(m[-1].split())
    return out


def write_status(rows):
    lines = [f"# backup D:/E: -> F:   updated {datetime.now():%Y-%m-%d %H:%M:%S}",
             "# columns of Files/Bytes: Total Copied Skipped Mismatch FAILED Extras",
             ""]
    for r in rows:
        line = f"{r['state']:<8} {r['src']}  ->  {r['dst']}"
        if r.get("start"):
            line += f"   start {r['start']}"
        if r.get("end"):
            line += f"  end {r['end']}  exit {r['code']}"
        lines.append(line)
        for key in ("Files", "Bytes"):
            if key in r:
                lines.append(f"         {key}: {r[key]}")
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    rows = [{"src": s, "dst": str(dest_for(s)), "state": "pending"} for s in SOURCES]
    for r in rows:
        if not Path(r["src"]).exists():
            r["state"] = "missing"
    write_status(rows)

    for r in rows:
        if r["state"] == "missing":
            continue
        log_path = LOG_DIR / (r["src"].replace(":", "").replace("\\", "_") + ".log")
        r["state"], r["start"] = "running", f"{datetime.now():%m-%d %H:%M}"
        write_status(rows)
        print(f"[{datetime.now():%H:%M:%S}] {r['src']} -> {r['dst']}", flush=True)
        code = subprocess.call(["robocopy", r["src"], r["dst"], *ROBOCOPY_FLAGS,
                                f"/LOG:{log_path}"])
        # robocopy: 0-7 = success (bits for copied/extra/mismatch), >= 8 = at least one failure
        r["code"], r["end"] = code, f"{datetime.now():%m-%d %H:%M}"
        r["state"] = "done" if code < 8 else "FAILED"
        r.update(summary_from_log(log_path))
        write_status(rows)
        print(f"[{datetime.now():%H:%M:%S}]   exit {code}  {r.get('Files', '')}", flush=True)

    failed = [r["src"] for r in rows if r["state"] == "FAILED"]
    print("ALL DONE" if not failed else f"FINISHED WITH FAILURES: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
