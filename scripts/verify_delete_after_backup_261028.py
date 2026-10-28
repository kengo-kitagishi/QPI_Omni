"""Delete each D:/E: source folder once backup_d_e_to_f_261028.py has copied it AND every file is on F:.

Watches the copier's status file. For each job marked done with 0 FAILED files:
  1. walk the source; every file must exist under the destination with the same size
  2. only then delete the source tree
Anything that fails the check is left in place and reported. Folders in KEEP are never deleted.

Status of every folder is rewritten to STATUS_PATH.
"""
import os
import re
import shutil
import stat
import sys
import time
from datetime import datetime
from pathlib import Path

COPY_STATUS = Path(r"C:\Users\QPI\Documents\QPI_Omni\backup_d_e_to_f_261028_STATUS.txt")
STATUS_PATH = Path(r"C:\Users\QPI\Documents\QPI_Omni\backup_d_e_to_f_261028_DELETE_STATUS.txt")
KEEP = {r"E:\261024", r"D:\AquisitionData\Kitagishi\261024"}   # user: keep 261024
POLL_SEC = 120

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

ROW_RE = re.compile(r"^(pending|running|done|FAILED|missing)\s+(\S.*?)\s+->\s+(\S+)")


def read_copy_status():
    """[(state, src, dst, failed_files or None)] in queue order."""
    rows, last = [], None
    for line in COPY_STATUS.read_text(encoding="utf-8").splitlines():
        m = ROW_RE.match(line)
        if m:
            last = [m[1], m[2], m[3].strip(), None]
            rows.append(last)
        elif last and line.strip().startswith("Files:"):
            vals = line.split(":", 1)[1].split()
            last[3] = int(vals[4]) if len(vals) >= 5 else None
    return rows


def verify(src, dst):
    """(ok, n_files, n_bytes, first_problem)."""
    n = size = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(dst, rel, f)
            try:
                ss = os.stat(s).st_size
            except OSError as e:
                return False, n, size, f"cannot stat source {s}: {e}"
            try:
                ds = os.stat(d).st_size
            except OSError:
                return False, n, size, f"missing on F: {d}"
            if ss != ds:
                return False, n, size, f"size differs {s} ({ss}) vs {d} ({ds})"
            n += 1
            size += ss
    return True, n, size, ""


def _on_rm_error(func, path, _exc):
    os.chmod(path, stat.S_IWRITE)      # read-only files block rmtree on Windows
    func(path)


def write_status(done):
    lines = [f"# verify + delete after backup   updated {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"# kept on purpose: {', '.join(sorted(KEEP))}", ""]
    lines += [f"{v['state']:<10} {src}   {v.get('note', '')}" for src, v in done.items()]
    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    done = {}
    while True:
        rows = read_copy_status()
        for state, src, dst, failed in rows:
            if src in done:
                continue
            if src in KEEP:
                done[src] = {"state": "kept", "note": "261024: not deleted (user)"}
            elif state == "missing":
                done[src] = {"state": "missing", "note": "source did not exist"}
            elif state == "FAILED" or (state == "done" and failed):
                done[src] = {"state": "NOT_DELETED", "note": f"copy reported failures ({failed})"}
            elif state == "done":
                t0 = time.time()
                ok, n, size, problem = verify(src, dst)
                if not ok:
                    done[src] = {"state": "NOT_DELETED", "note": f"verify failed: {problem}"}
                else:
                    shutil.rmtree(src, onerror=_on_rm_error)
                    gone = not os.path.exists(src)
                    done[src] = {"state": "deleted" if gone else "PARTIAL",
                                 "note": f"{n} files, {size / 1e9:.1f} GB verified on F: "
                                         f"({(time.time() - t0) / 60:.0f} min)"}
            else:
                continue
            print(f"[{datetime.now():%H:%M:%S}] {done[src]['state']:<10} {src}  {done[src]['note']}",
                  flush=True)
            write_status(done)
        if rows and all(r[1] in done for r in rows):
            break
        time.sleep(POLL_SEC)
    bad = [s for s, v in done.items() if v["state"] in ("NOT_DELETED", "PARTIAL")]
    print("ALL DONE" if not bad else f"LEFT IN PLACE: {bad}", flush=True)


if __name__ == "__main__":
    main()
