"""
analyze_drift_outliers_v2.py
----------------------------
Focus on meaningful outliers: large jumps and sustained shifts
(wrong-channel lock-on detection).
"""
import json
import numpy as np
from collections import defaultdict

DRIFT_LOG = r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json"
JUMP_TH = 2.0
SUSTAINED_TH = 3.0
REVERSE_WINDOW = 5


def main():
    print("Loading...", flush=True)
    with open(DRIFT_LOG, "r") as f:
        data = json.load(f)
    print(f"{len(data)} timepoints", flush=True)

    pos_series = defaultdict(lambda: {"tx": [], "ty": [], "corr": [], "tp": []})
    for entry in data:
        tp = entry["timepoint"]
        for p in entry.get("positions", []):
            label = p.get("pos_label", "")
            s = pos_series[label]
            s["tp"].append(tp)
            s["tx"].append(p.get("tx_avg_px", float("nan")))
            s["ty"].append(p.get("ty_avg_px", float("nan")))
            s["corr"].append(p.get("ecc_correlation", float("nan")))

    # ========== JUMP EVENTS ==========
    print()
    print("=" * 80)
    print(f"JUMP EVENTS (frame-to-frame shift > {JUMP_TH} px)")
    print("=" * 80)

    all_jumps = []
    for label in sorted(pos_series, key=lambda x: int(x.replace("Pos", "")) if x.replace("Pos", "").isdigit() else 0):
        s = pos_series[label]
        tx = np.array(s["tx"])
        ty = np.array(s["ty"])
        tps = s["tp"]
        dtx = np.diff(tx)
        dty = np.diff(ty)
        mag = np.sqrt(dtx**2 + dty**2)
        for i in range(len(mag)):
            if mag[i] > JUMP_TH:
                all_jumps.append((label, tps[i + 1], mag[i], dtx[i], dty[i],
                                  tx[i + 1], ty[i + 1], s["corr"][i + 1]))

    print(f"Total jump events: {len(all_jumps)}")
    print()

    jumps_by_pos = defaultdict(list)
    for j in all_jumps:
        jumps_by_pos[j[0]].append(j)

    print(f"Positions with jumps: {len(jumps_by_pos)}")
    print()
    hdr = f"{'Pos':<8} {'#jumps':>6} {'Worst':>8}  First few events"
    print(hdr)
    print("-" * 90)
    for label in sorted(jumps_by_pos, key=lambda x: len(jumps_by_pos[x]), reverse=True)[:25]:
        jlist = jumps_by_pos[label]
        worst = max(j[2] for j in jlist)
        examples = ", ".join(f"T{j[1]}({j[2]:.1f}px)" for j in jlist[:5])
        print(f"{label:<8} {len(jlist):>6} {worst:>7.1f}px  {examples}")

    # ========== SUSTAINED SHIFTS ==========
    print()
    print("=" * 80)
    print(f"SUSTAINED SHIFTS (jump > {SUSTAINED_TH}px, NOT reversed within {REVERSE_WINDOW} frames)")
    print("= likely wrong-channel lock-on =")
    print("=" * 80)
    print()

    sustained = []
    for label in sorted(pos_series, key=lambda x: int(x.replace("Pos", "")) if x.replace("Pos", "").isdigit() else 0):
        s = pos_series[label]
        tx = np.array(s["tx"])
        ty = np.array(s["ty"])
        tps = s["tp"]
        dtx = np.diff(tx)
        dty = np.diff(ty)
        mag = np.sqrt(dtx**2 + dty**2)

        for i in range(len(mag)):
            if mag[i] > SUSTAINED_TH:
                reversed_within = False
                for j in range(i + 1, min(i + 1 + REVERSE_WINDOW, len(mag))):
                    if mag[j] > 2.0:
                        dot = dtx[i] * dtx[j] + dty[i] * dty[j]
                        if dot < 0:
                            reversed_within = True
                            break
                if not reversed_within:
                    sustained.append((label, tps[i + 1], mag[i], dtx[i], dty[i],
                                      tx[i + 1], ty[i + 1], s["corr"][i + 1]))

    print(f"Sustained shift events: {len(sustained)}")
    print()
    print(f"{'Pos':<8} {'T':>6} {'jump':>8} {'dtx':>8} {'dty':>8} {'tx_after':>10} {'ty_after':>10} {'corr':>8}")
    print("-" * 75)
    for ev in sustained[:60]:
        print(f"{ev[0]:<8} {ev[1]:>6} {ev[2]:>7.2f}px {ev[3]:>+7.2f} {ev[4]:>+7.2f}  tx={ev[5]:>+8.3f} ty={ev[6]:>+8.3f} {ev[7]:>.4f}")
    if len(sustained) > 60:
        print(f"  ... +{len(sustained) - 60} more")

    # Group sustained by Pos
    print()
    sus_by_pos = defaultdict(list)
    for ev in sustained:
        sus_by_pos[ev[0]].append(ev)
    print(f"Positions with sustained shifts: {len(sus_by_pos)}")
    print()
    for label in sorted(sus_by_pos, key=lambda x: len(sus_by_pos[x]), reverse=True):
        events = sus_by_pos[label]
        tps_str = ", ".join(f"T{e[1]}({e[2]:.1f}px)" for e in events[:8])
        print(f"  {label:<8} {len(events):>3} events: {tps_str}")

    # ========== LOW CORRELATION (excluding tilt_bounds noise) ==========
    print()
    print("=" * 80)
    print(f"LOW CORRELATION FRAMES (corr < {0.93})")
    print("=" * 80)
    low_corr = []
    for label in sorted(pos_series, key=lambda x: int(x.replace("Pos", "")) if x.replace("Pos", "").isdigit() else 0):
        s = pos_series[label]
        for i, c in enumerate(s["corr"]):
            if c < 0.93:
                low_corr.append((label, s["tp"][i], c, s["tx"][i], s["ty"][i]))

    print(f"Total low-corr frames: {len(low_corr)}")
    if low_corr:
        lc_by_pos = defaultdict(list)
        for ev in low_corr:
            lc_by_pos[ev[0]].append(ev)
        print(f"Positions affected: {len(lc_by_pos)}")
        for label in sorted(lc_by_pos, key=lambda x: len(lc_by_pos[x]), reverse=True)[:15]:
            events = lc_by_pos[label]
            ex = ", ".join(f"T{e[1]}(corr={e[2]:.3f})" for e in events[:5])
            print(f"  {label:<8} {len(events):>4} frames: {ex}")

    # ========== SUMMARY ==========
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total positions: {len(pos_series)}")
    print(f"Total timepoints: {len(data)}")
    print(f"Jump events (>{JUMP_TH}px): {len(all_jumps)} across {len(jumps_by_pos)} Pos")
    print(f"Sustained shifts (>{SUSTAINED_TH}px, no reverse): {len(sustained)} across {len(sus_by_pos)} Pos")
    print(f"Low-corr (<0.93): {len(low_corr)} frames")


if __name__ == "__main__":
    main()
