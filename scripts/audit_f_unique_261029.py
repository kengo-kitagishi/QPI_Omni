"""Before reformatting F:, find what exists only on F: (no same-size copy on D: or E:).

For every file under each F: session folder, look for the same relative path on E:/<session>,
D:/AquisitionData/Kitagishi/<session> (and D:/260928_from_F for 260928). Reported per second-level
folder: files and bytes that have a copy elsewhere vs. only on F:. Read-only on every disk.
"""
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPORT = Path(r"C:\Users\QPI\Documents\QPI_Omni\audit_f_unique_261029.txt")
SESSIONS = ["260908", "260917", "260922", "260928", "261004", "261024"]


def roots():
    for s in SESSIONS:
        yield Path(rf"F:\{s}"), s
        yield Path(rf"F:\from_E\{s}"), s


def candidates(s):
    c = [Path(rf"E:\{s}"), Path(rf"D:\AquisitionData\Kitagishi\{s}")]
    if s == "260928":
        c.append(Path(r"D:\260928_from_F"))
    return c


def walk(top):
    stack = [top]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    else:
                        yield Path(e.path), e.stat().st_size
        except OSError as exc:
            print(f"cannot read {d}: {exc}", flush=True)


def main():
    lines = [f"# F: uniqueness audit  started {datetime.now():%Y-%m-%d %H:%M:%S}",
             "# group | files_elsewhere GB_elsewhere | files_ONLY_F GB_ONLY_F"]
    tot_only = 0
    for top, s in roots():
        if not top.exists():
            continue
        cands = candidates(s)
        groups = defaultdict(lambda: [0, 0, 0, 0])
        for f, size in walk(top):
            rel = f.relative_to(top)
            key = f"{top}\\{rel.parts[0]}" if len(rel.parts) > 1 else f"{top}\\(loose files)"
            found = False
            for c in cands:
                try:
                    if os.stat(c / rel).st_size == size:
                        found = True
                        break
                except OSError:
                    pass
            g = groups[key]
            if found:
                g[0] += 1
                g[1] += size
            else:
                g[2] += 1
                g[3] += size
        for key in sorted(groups):
            g = groups[key]
            tot_only += g[3]
            lines.append(f"{key} | {g[0]} {g[1] / 1e9:.1f} | {g[2]} {g[3] / 1e9:.1f}")
        print(f"[{datetime.now():%H:%M:%S}] {top} done", flush=True)
        REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    lines.append(f"TOTAL only on F: {tot_only / 1e9:.1f} GB   finished {datetime.now():%Y-%m-%d %H:%M:%S}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1], flush=True)


if __name__ == "__main__":
    sys.exit(main())
