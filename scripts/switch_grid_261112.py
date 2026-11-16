"""switch_grid_261112.py

Switch the reference grid of the RUNNING 261112 run-3 timelapse when the medium is changed.
compute_drift_online.py reads grid_dir from drift_config_zstack.json at every timepoint, so
the switch takes effect from the next timepoint; the drift controller then re-centres each
Pos on the new grid's point-cloud centroid (setpoint grid_centroid) within 2-3 rounds.

Every switch is appended to media_switch_log.csv in the session folder with the wall-clock
time and the last timepoint written to drift_state, so the frame where each grid starts is
on record for the analysis (datasets yaml).

    python scripts/switch_grid_261112.py 2per    # medium changed to 2% EMM
    python scripts/switch_grid_261112.py 0per    # medium changed to 0%
    python scripts/switch_grid_261112.py --show
"""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

SESSION = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261112_3")
GRIDS = {
    "2per": r"D:\AquisitionData\Kitagishi\261112\recon_0p03um_25x25",
    "0per": r"D:\AquisitionData\Kitagishi\261112\recon_0per_0p03um_25x25",
}


def state():
    s = {}
    for ln in open(SESSION / "drift_state_zstack.txt", encoding="utf-8"):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.strip().split("=", 1)
            s[k] = v
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("medium", nargs="?", choices=sorted(GRIDS))
    ap.add_argument("--note", default="")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    cfg_path = SESSION / "drift_config_zstack.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cur = next((k for k, v in GRIDS.items() if Path(v) == Path(cfg["grid_dir"])), "?")
    st = state()
    log = SESSION / "media_switch_log.csv"
    if args.show or args.medium is None:
        print(f"current grid: {cur}  ({cfg['grid_dir']}); drift_state TIMEPOINT={st.get('TIMEPOINT')} at {st.get('TIMESTAMP')}")
        if log.exists():
            print(log.read_text(encoding="utf-8"))
        return
    if args.medium == cur:
        print(f"already on the {cur} grid; nothing changed")
        return
    for k in GRIDS.values():
        if not Path(k).is_dir():
            raise SystemExit(f"grid folder missing: {k}")
    cfg["grid_dir"] = GRIDS[args.medium]
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    new = not log.exists()
    with open(log, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["switched_at", "last_timepoint_in_state", "state_timestamp", "from", "to", "grid_dir", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), st.get("TIMEPOINT"), st.get("TIMESTAMP"),
                    cur, args.medium, GRIDS[args.medium], args.note])
    print(f"grid switched {cur} -> {args.medium}: {GRIDS[args.medium]}")
    print(f"last timepoint in drift_state: {st.get('TIMEPOINT')} ({st.get('TIMESTAMP')}); "
          f"the next timepoint is the first processed with the {args.medium} grid. Logged in {log.name}")


if __name__ == "__main__":
    main()
