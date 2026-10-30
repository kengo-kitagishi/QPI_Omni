"""Split the 261029 grid.pos into rounds for the YE grid: D: holds ~40 Pos of raw at a time.

Each round is acquired to D:\\AquisitionData\\Kitagishi\\261029\\grid_YE_<N>, reconstructed into the shared
recon_YE_grid (run_ye_grid_round_261030.py <N>), and its raw deleted before the next round.
Sizes: raw 2.15 GB/Pos + recon 0.32 GB/Pos must fit in what D: has left after the earlier outputs.
"""
import json
import re
from pathlib import Path

SRC = Path(r"D:\AquisitionData\Kitagishi\261029\grid.pos")
ROUNDS = {1: (0, 37), 2: (38, 69), 3: (70, 97), 4: (98, 118)}
LABEL_RE = re.compile(r"^Pos(\d+)_x[+-]\d+_y[+-]\d+$")


def main():
    data = json.loads(SRC.read_text(encoding="utf-8"))
    positions = data["POSITIONS"]
    n_total = 0
    for r, (a, b) in ROUNDS.items():
        part = []
        for p in positions:
            m = LABEL_RE.match(p["LABEL"])
            if not m:
                raise ValueError(f"unexpected label {p['LABEL']}")
            if a <= int(m.group(1)) <= b:
                part.append(p)
        out = SRC.with_name(f"grid_YE_r{r}_Pos{a}-{b}.pos")
        out.write_text(json.dumps({**data, "POSITIONS": part}, indent=3), encoding="utf-8")
        print(f"{out}: {len(part)} points ({len(part) // 121} Pos)")
        n_total += len(part)
    assert n_total == len(positions)


if __name__ == "__main__":
    main()
