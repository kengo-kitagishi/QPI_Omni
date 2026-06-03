"""Show concrete examples of sustained_jump with surrounding frames."""
import json
import numpy as np
from collections import defaultdict

print("Loading...", flush=True)
with open(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json", "r") as f:
    data = json.load(f)

pos = defaultdict(lambda: {"tp": [], "tx": [], "ty": [], "corr": []})
for e in data:
    tp = e["timepoint"]
    for p in e.get("positions", []):
        lb = p.get("pos_label", "")
        pos[lb]["tp"].append(tp)
        pos[lb]["tx"].append(p.get("tx_avg_px", float("nan")))
        pos[lb]["ty"].append(p.get("ty_avg_px", float("nan")))
        pos[lb]["corr"].append(p.get("ecc_correlation", float("nan")))

bf = json.loads(open(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\bad_frames.json", "r").read())

hdr = f"  {'T':>6}  {'tx':>8}  {'ty':>8}  {'corr':>8}"
print()
print("=" * 90)
print("SUSTAINED JUMP EXAMPLES - 5 frames before/after for context")
print("=" * 90)

n = 0
for plabel in ["Pos2", "Pos5", "Pos53", "Pos1", "Pos57", "Pos81", "Pos72", "Pos63"]:
    if plabel not in bf:
        continue
    s = pos[plabel]
    tps = np.array(s["tp"])
    tx = np.array(s["tx"])
    ty = np.array(s["ty"])
    corr = np.array(s["corr"])

    for tp_str, reasons in bf[plabel]["bad_timepoints"].items():
        tp_val = int(tp_str)
        if tp_val < 3:
            continue
        if not any("sustained_jump" in r for r in reasons):
            continue

        idx = np.where(tps == tp_val)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]

        lo = max(0, idx - 5)
        hi = min(len(tps), idx + 6)

        reason_str = "; ".join(reasons)
        print(f"\n--- {plabel} T={tp_val}: {reason_str} ---")
        print(hdr)
        for i in range(lo, hi):
            marker = "  <<<" if i == idx else ""
            print(f"  {int(tps[i]):>6}  {tx[i]:>+8.3f}  {ty[i]:>+8.3f}  {corr[i]:>8.4f}{marker}")

        n += 1
        if n >= 25:
            break
    if n >= 25:
        break

print(f"\n({n} examples shown)")
