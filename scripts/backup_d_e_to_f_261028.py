"""Copy every data folder on D: and E: to the new external HDD F:, one folder at a time.

Layout keeps the source path, so nothing from D: and E: lands in the same folder:
    E:\\261024\\...                         -> F:\\from_E\\261024\\...
    D:\\AquisitionData\\Kitagishi\\260908\\... -> F:\\from_D\\AquisitionData\\Kitagishi\\260908\\...

Copy only: nothing is deleted. Re-running skips files already on F: (same size and mtime),
so an interrupted run resumes where it stopped.

Status of every job (pending / running / done / FAILED) is rewritten to STATUS_PATH after each
job, so "which ones are finished" is one file to open.
"""
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

DEST_ROOT = Path("F:/")
LOG_DIR = Path(r"F:\_backup_logs_261028")
STATUS_PATH = Path(r"C:\Users\QPI\Documents\QPI_Omni\backup_d_e_to_f_261028_STATUS.txt")

# E: first: it is full (0 GB free) and the next timelapse writes there. Within E:, the folders
# that get deleted afterwards come first, largest bytes-per-file first (frees the most GB per
# hour on a disk that tops out at ~12-23 files/s). 261024 is kept, so it goes last on E:.
SOURCES = [
    r"E:\260928",
    r"D:\260928_from_F",          # part of E:\260928 parked on D: while F: was reformatted (2026-10-29)
    r"C:\260928_from_F",          # overflow of the same restore (E: + D: were short)
    r"C:\F_only_misc_261029",     # the two files that existed only on the old F: (audit_f_unique_261029.txt)
    r"C:\261024_grid_raw_parked\2per_grid_1",   # 261024 grid raw Pos100-118 moved off E: for the 261029 grid
    r"E:\261004",
    r"E:\260917",
    r"E:\260908_whitelist_realign",
    r"E:\260908_outside_quad",
    r"E:\260908_seg",
    r"E:\260924",
    r"E:\260927",
    r"E:\Acuisition",
    r"E:\261024",
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

# robocopy reached ~5 MB/s on this USB exFAT HDD (2026-10-28, any /MT, any /COPY); a plain
# per-file copy reached ~13 MB/s, so files are copied here one by one instead.
MTIME_TOLERANCE_S = 2.0   # exFAT keeps 10 ms, but stay tolerant of rounding

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


DEST_OVERRIDE = {   # back to their original places
    r"D:\260928_from_F": Path("F:/") / "from_E" / "260928",
    r"C:\260928_from_F": Path("F:/") / "from_E" / "260928",
    r"C:\F_only_misc_261029": Path("F:/"),
    r"C:\261024_grid_raw_parked\2per_grid_1": Path("F:/") / "from_E" / "261024" / "2per_grid_1",
}


def dest_for(src):
    if src in DEST_OVERRIDE:
        return DEST_OVERRIDE[src]
    p = Path(src)
    return DEST_ROOT / f"from_{p.drive[0]}" / p.relative_to(p.anchor)


def _same(src_stat, dst_path):
    try:
        d = os.stat(dst_path)
    except OSError:
        return False
    return (d.st_size == src_stat.st_size
            and abs(d.st_mtime - src_stat.st_mtime) <= MTIME_TOLERANCE_S)


def copy_tree(src, dst, log):
    """Copy src to dst file by file; skip files already there (size + mtime).

    Each file is written to <name>.part and renamed only when complete, so an interrupted run
    never leaves a short file under the real name. Returns
    {"Files": "total copied skipped 0 failed 0", "Bytes": "..."} in robocopy column order.
    """
    tot = cop = skp = fail = 0
    btot = bcop = 0
    for root, _dirs, files in os.walk(src):
        out_dir = os.path.join(dst, os.path.relpath(root, src))
        os.makedirs(out_dir, exist_ok=True)
        for f in files:
            s = os.path.join(root, f)
            d = os.path.join(out_dir, f)
            try:
                st = os.stat(s)
            except OSError as e:
                fail += 1
                log.write(f"FAILED stat {s}: {e}\n")
                continue
            tot += 1
            btot += st.st_size
            if _same(st, d):
                skp += 1
                continue
            for attempt in (1, 2):
                try:
                    shutil.copyfile(s, d + ".part")
                    os.utime(d + ".part", (st.st_atime, st.st_mtime))
                    os.replace(d + ".part", d)
                    cop += 1
                    bcop += st.st_size
                    break
                except OSError as e:
                    if attempt == 2:
                        fail += 1
                        log.write(f"FAILED copy {s}: {e}\n")
                    time.sleep(5)
            if tot % 5000 == 0:
                log.write(f"{datetime.now():%H:%M:%S} {tot} files seen, {cop} copied, "
                          f"{bcop / 1e9:.1f} GB\n")
                log.flush()
    return {"Files": f"{tot} {cop} {skp} 0 {fail} 0",
            "Bytes": f"{btot} {bcop} {btot - bcop} 0 0 0"}


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


def previous_done():
    """{src: row} for jobs a previous run already finished cleanly (kept, not redone).

    Their sources may already be deleted by verify_delete_after_backup_261028.py, so they must
    not be re-walked or reported as missing.
    """
    if not STATUS_PATH.exists():
        return {}
    done, last = {}, None
    row_re = re.compile(r"^done\s+(\S.*?)\s+->\s+(\S+)\s+start (\S+ \S+)\s+end (\S+ \S+)\s+exit (\d+)")
    for line in STATUS_PATH.read_text(encoding="utf-8").splitlines():
        m = row_re.match(line)
        if m:
            last = {"src": m[1], "dst": m[2], "state": "done", "start": m[3], "end": m[4],
                    "code": int(m[5])}
            done[m[1]] = last
        elif last and line.strip().startswith(("Files:", "Bytes:")):
            key, val = line.strip().split(":", 1)
            last[key] = val.strip()
        else:
            last = None
    return {s: r for s, r in done.items() if r["code"] == 0}


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    carried = previous_done()
    rows = [carried.get(s) or {"src": s, "dst": str(dest_for(s)), "state": "pending"}
            for s in SOURCES]
    for r in rows:
        if r["state"] == "pending" and not Path(r["src"]).exists():
            r["state"] = "missing"
    write_status(rows)

    for r in rows:
        if r["state"] in ("missing", "done"):
            continue
        log_path = LOG_DIR / (r["src"].replace(":", "").replace("\\", "_") + ".log")
        r["state"], r["start"] = "running", f"{datetime.now():%m-%d %H:%M}"
        write_status(rows)
        print(f"[{datetime.now():%H:%M:%S}] {r['src']} -> {r['dst']}", flush=True)
        with open(log_path, "a", encoding="utf-8") as log:
            summary = copy_tree(r["src"], r["dst"], log)
        n_failed = int(summary["Files"].split()[4])
        r["code"], r["end"] = n_failed, f"{datetime.now():%m-%d %H:%M}"
        r["state"] = "done" if n_failed == 0 else "FAILED"
        r.update(summary)
        write_status(rows)
        print(f"[{datetime.now():%H:%M:%S}]   failed {r['code']}  {r.get('Files', '')}",
              flush=True)

    failed = [r["src"] for r in rows if r["state"] == "FAILED"]
    print("ALL DONE" if not failed else f"FINISHED WITH FAILURES: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
