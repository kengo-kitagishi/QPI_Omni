"""Copy 260928 back from F: (its only copy) to E: and D: so F: can be reformatted (2 MB exFAT clusters).

Units = PosN folders under each run folder of F:/from_E/260928 (plus loose files). Units go to
E:/260928/<same path> while E: keeps > E_RESERVE_GB free, the rest to D:/260928_from_F/<same path>.
Copy = backup_d_e_to_f_261028.copy_tree (.part + rename, size+mtime skip, so a re-run resumes).
Then every file is checked (exists, same size) at its destination. Nothing on F: is deleted.
"""
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from backup_d_e_to_f_261028 import copy_tree  # noqa: E402

SRC = Path(r"F:\from_E\260928")
DST_E = Path(r"E:\260928")
DST_D = Path(r"D:\260928_from_F")
DST_C = Path(r"C:\260928_from_F")   # overflow: E: + D: are ~10-20 GB short of 372 GB (2026-10-29)
E_RESERVE_GB = 15
D_RESERVE_GB = 10
PLAN_PATH = Path(r"C:\Users\QPI\Documents\QPI_Omni\restore_260928_PLAN.json")
STATUS_PATH = Path(r"C:\Users\QPI\Documents\QPI_Omni\restore_260928_STATUS.txt")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def tree_size(p):
    n = b = 0
    for root, _d, files in os.walk(p):
        for f in files:
            b += os.stat(os.path.join(root, f)).st_size
            n += 1
    return n, b


def units():
    out = []
    for run in sorted(SRC.iterdir()):
        if run.is_file():
            out.append(run)
            continue
        for u in sorted(run.iterdir()):
            out.append(u)
    return out


def status(lines):
    STATUS_PATH.write_text(f"# restore 260928 F: -> E:/D:   updated {datetime.now():%Y-%m-%d %H:%M:%S}\n"
                           + "\n".join(lines) + "\n", encoding="utf-8")


def main():
    if PLAN_PATH.exists():
        plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    else:
        e_budget = shutil.disk_usage("E:\\").free - E_RESERVE_GB * 1e9
        d_budget = shutil.disk_usage("D:\\").free - D_RESERVE_GB * 1e9
        plan = []
        for u in units():
            n, b = tree_size(u) if u.is_dir() else (1, u.stat().st_size)
            rel = u.relative_to(SRC)
            already = (DST_E / rel).exists()          # started on E: before a re-plan: finish it there
            if already:
                b_e = tree_size(DST_E / rel)[1] if (DST_E / rel).is_dir() else (DST_E / rel).stat().st_size
                dst = DST_E
                e_budget -= max(b - b_e, 0)
            elif b <= e_budget:
                dst = DST_E
                e_budget -= b
            elif b <= d_budget:
                dst = DST_D
                d_budget -= b
            else:
                dst = DST_C
            plan.append({"src": str(u), "dst": str(dst / rel),
                         "files": n, "bytes": b, "is_file": u.is_file()})
        PLAN_PATH.write_text(json.dumps(plan, indent=1), encoding="utf-8")
    tot = sum(p["bytes"] for p in plan)
    on_d = sum(p["bytes"] for p in plan if p["dst"].startswith("D:"))
    on_c = sum(p["bytes"] for p in plan if p["dst"].startswith("C:"))
    print(f"{len(plan)} units, {tot / 1e9:.1f} GB ({on_d / 1e9:.1f} GB to D:, {on_c / 1e9:.1f} GB to C:)", flush=True)

    lines = []
    log_path = Path(r"F:\_backup_logs_261028\restore_260928.log")
    with open(log_path, "a", encoding="utf-8") as log:
        for i, p in enumerate(plan, 1):
            if p["is_file"]:
                Path(p["dst"]).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p["src"], p["dst"])
                summ = {"Files": "1 1 0 0 0 0"}
            else:
                summ = copy_tree(p["src"], p["dst"], log)
            failed = int(summ["Files"].split()[4])
            lines.append(f"{'copied' if failed == 0 else 'FAILED'}  {p['src']} -> {p['dst']}  {summ['Files']}")
            status(lines + [f"... {i}/{len(plan)} units"])
            print(f"[{datetime.now():%H:%M:%S}] {i}/{len(plan)} {Path(p['src']).name} -> {p['dst'][:2]}  {summ['Files']}",
                  flush=True)

    # verify every file at its destination
    bad, n_ok = [], 0
    for p in plan:
        if p["is_file"]:
            pairs = [(p["src"], p["dst"])]
        else:
            pairs = []
            for root, _d, files in os.walk(p["src"]):
                rel = os.path.relpath(root, p["src"])
                pairs += [(os.path.join(root, f), os.path.join(p["dst"], rel, f)) for f in files]
        for s, d in pairs:
            try:
                if os.stat(s).st_size == os.stat(d).st_size:
                    n_ok += 1
                    continue
            except OSError:
                pass
            bad.append(d)
    lines.append(f"VERIFY: {n_ok} files ok, {len(bad)} missing/different" + (f" (first: {bad[0]})" if bad else ""))
    status(lines)
    print(lines[-1], flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
