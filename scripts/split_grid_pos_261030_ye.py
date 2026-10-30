"""Split the 261029 grid.pos into two halves for the YE grid: C: and D: together have room, neither alone.

Pos0-77 -> grid_YE_A_Pos0-77.pos   (acquire to C:\\261029_YE_grid\\grid_YE_A)
Pos78-118 -> grid_YE_B_Pos78-118.pos (acquire to D:\\AquisitionData\\Kitagishi\\261029\\grid_YE_B)
Same labels and coordinates as grid.pos; only the list is cut.
"""
import json
import re
from pathlib import Path

SRC = Path(r"D:\AquisitionData\Kitagishi\261029\grid.pos")
SPLIT = 78                     # first Pos of the second half
LABEL_RE = re.compile(r"^Pos(\d+)_x[+-]\d+_y[+-]\d+$")


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    positions = data["POSITIONS"]
    a, b = [], []
    for p in positions:
        m = LABEL_RE.match(p["LABEL"])
        if not m:
            raise ValueError(f"unexpected label {p['LABEL']}")
        (a if int(m.group(1)) < SPLIT else b).append(p)
    for name, part in ((f"grid_YE_A_Pos0-{SPLIT - 1}.pos", a), (f"grid_YE_B_Pos{SPLIT}-118.pos", b)):
        out = SRC.with_name(name)
        out.write_text(json.dumps({**data, "POSITIONS": part}, indent=3), encoding="utf-8")
        print(f"{out}: {len(part)} points ({len(part) // 121} Pos)")
    assert len(a) + len(b) == len(positions)


if __name__ == "__main__":
    main()
