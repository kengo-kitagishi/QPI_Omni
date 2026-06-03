"""
Find cases where tx/ty jumps and does NOT return to baseline.
Compare: mean tx before jump vs mean tx after jump (10-frame windows).
If the shift persists, it's likely a wrong-channel lock-on.
"""
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

JUMP_TH = 3.0
WINDOW = 10

print()
print("=" * 90)
print("PERMANENT SHIFT DETECTION")
print(f"Jump > {JUMP_TH}px where mean(tx) 10 frames after differs from")
print(f"mean(tx) 10 frames before by > 1.5px")
print("=" * 90)

permanent = []

for label in sorted(pos, key=lambda x: int(x.replace("Pos", "")) if x.replace("Pos", "").isdigit() else 0):
    s = pos[label]
    tx = np.array(s["tx"])
    ty = np.array(s["ty"])
    tps = np.array(s["tp"])
    corr = np.array(s["corr"])
    dtx = np.diff(tx)
    dty = np.diff(ty)
    mag = np.sqrt(dtx**2 + dty**2)

    for i in range(len(mag)):
        if tps[i + 1] < 3:
            continue
        if mag[i] < JUMP_TH:
            continue

        # Mean tx/ty in window before and after
        before_lo = max(0, i - WINDOW)
        after_hi = min(len(tx), i + 1 + WINDOW)

        if i - before_lo < 3 or after_hi - (i + 1) < 3:
            continue

        tx_before = np.nanmean(tx[before_lo:i])
        ty_before = np.nanmean(ty[before_lo:i])
        tx_after = np.nanmean(tx[i + 4:after_hi])  # skip 3 recovery frames
        ty_after = np.nanmean(ty[i + 4:after_hi])

        shift_tx = tx_after - tx_before
        shift_ty = ty_after - ty_before
        shift_mag = np.sqrt(shift_tx**2 + shift_ty**2)

        if shift_mag > 1.5:
            permanent.append((
                label, int(tps[i + 1]), mag[i],
                tx_before, ty_before, tx_after, ty_after,
                shift_tx, shift_ty, shift_mag
            ))

print(f"\nPermanent shifts found: {len(permanent)}")
print()

if permanent:
    print(f"{'Pos':<8} {'T':>6} {'jump':>7} {'tx_bef':>8} {'tx_aft':>8} {'shift_tx':>9} {'ty_bef':>8} {'ty_aft':>8} {'shift_ty':>9} {'|shift|':>8}")
    print("-" * 95)
    for ev in permanent:
        label, tp, jmag, txb, tyb, txa, tya, stx, sty, smag = ev
        print(f"{label:<8} {tp:>6} {jmag:>6.1f}px {txb:>+8.3f} {txa:>+8.3f} {stx:>+8.3f}   {tyb:>+8.3f} {tya:>+8.3f} {sty:>+8.3f}   {smag:>6.2f}")

    # Show context for first few
    print()
    print("=" * 90)
    print("CONTEXT FOR PERMANENT SHIFTS (20 frames around event)")
    print("=" * 90)
    n = 0
    for ev in permanent[:15]:
        label, tp_val = ev[0], ev[1]
        s = pos[label]
        tps = np.array(s["tp"])
        tx = np.array(s["tx"])
        ty = np.array(s["ty"])
        corr = np.array(s["corr"])

        idx = np.where(tps == tp_val)[0]
        if len(idx) == 0:
            continue
        idx = idx[0]
        lo = max(0, idx - 10)
        hi = min(len(tps), idx + 11)

        print(f"\n--- {label} T={tp_val} (baseline shift: tx{ev[7]:+.2f}, ty{ev[8]:+.2f}) ---")
        hdr = f"  {'T':>6}  {'tx':>8}  {'ty':>8}  {'corr':>8}"
        print(hdr)
        for j in range(lo, hi):
            marker = "  <<<" if j == idx else ""
            print(f"  {int(tps[j]):>6}  {tx[j]:>+8.3f}  {ty[j]:>+8.3f}  {corr[j]:>8.4f}{marker}")
        n += 1
else:
    print("None found! All jumps eventually return to baseline.")
