"""Turn the T0 review picks into channel_whitelist.json for the 261030 YE session.

Picks that the drift correction cannot use are reported: a channel is usable only if the latest
drift_log entry measured it (status not tilt_bounds_ng / pass1_failed). They are kept out of the JSON
so the per-Pos count is the number that actually enters the average.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

SESSION = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261030_YE")
PICKS = SESSION / "whitelist_review_t0.selected.txt"
LOG = SESSION / "drift_log_zstack.json"
OUT = SESSION / "channel_whitelist.json"


def main():
    picks = defaultdict(set)
    for line in PICKS.read_text(encoding="utf-8").split("\n"):
        m = re.match(r"^\s*(Pos\d+)\s+ch(\d+)\s*$", line)
        if m:
            picks[m.group(1)].add(int(m.group(2)))

    entries = json.loads(LOG.read_text(encoding="utf-8"))
    last = entries[-1]
    usable = {}   # channels present in channel_details and not failed (absent = never measured)
    for p in last["positions"]:
        usable[p["pos_label"]] = {d["ch"] for d in p.get("channel_details", [])
                                  if d.get("status") not in ("tilt_bounds_ng", "pass1_failed")}

    wl, dropped, singles = {}, [], []
    for pos in sorted(picks, key=lambda s: int(s[3:])):
        bad = picks[pos] - usable.get(pos, set())
        dropped += [f"{pos} ch{c:02d}" for c in sorted(bad)]
        keep = sorted(picks[pos] - bad)
        if keep:
            wl[pos] = keep
        if len(keep) < 3:
            singles.append(f"{pos}:{len(keep)}")

    OUT.write_text(json.dumps({
        "meta": {"source": "User supplied Pos/ch list from T0 HTML review", "purpose": "gaussian2d alignment",
                 "session": "261030_YE", "frame": 0, "drift_log_timepoint": last["timepoint"],
                 "dropped_unmeasured": dropped},
        "whitelist": wl,
    }, indent=2), encoding="utf-8")
    n = sum(len(v) for v in picks.values())
    print(f"{n} picks in {len(picks)} Pos -> {sum(len(v) for v in wl.values())} kept in {len(wl)} Pos")
    print(f"dropped (edge / not measured): {len(dropped)}: {', '.join(dropped)}")
    print(f"Pos with < 3 usable (MAD inactive): {', '.join(singles)}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
