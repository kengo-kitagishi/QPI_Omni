"""
analyze_drift_outliers.py
-------------------------
Analyze drift session log to find outlier alignments per position.
Detects: large frame-to-frame jumps, low correlation, many outlier channels.
"""
import json
import sys
import numpy as np
from pathlib import Path
from collections import defaultdict

DRIFT_LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")

# Thresholds
JUMP_THRESHOLD_PX = 2.0       # frame-to-frame shift jump
LOW_CORR_THRESHOLD = 0.95     # ecc_correlation below this
OUTLIER_CH_FRACTION = 0.6     # >60% channels marked outlier


def main():
    print(f"Loading {DRIFT_LOG.name} ({DRIFT_LOG.stat().st_size / 1e9:.2f} GB)...", flush=True)
    with open(DRIFT_LOG, "r") as f:
        data = json.load(f)

    n_tp = len(data)
    print(f"Loaded: {n_tp} timepoints", flush=True)

    # Organize per-pos time series
    pos_series = defaultdict(lambda: {
        "tx": [], "ty": [], "corr": [], "tp": [],
        "n_outlier_ch": [], "n_ch": [], "valid": [],
        "jump_detected": [], "statuses": []
    })

    for entry in data:
        tp = entry["timepoint"]
        for p in entry.get("positions", []):
            label = p.get("pos_label", f"Pos{p.get('pos_idx', '?')}")
            s = pos_series[label]
            s["tp"].append(tp)
            s["tx"].append(p.get("tx_avg_px", np.nan))
            s["ty"].append(p.get("ty_avg_px", np.nan))
            s["corr"].append(p.get("ecc_correlation", np.nan))
            s["valid"].append(p.get("correction_valid", True))
            s["jump_detected"].append(p.get("jump_detected", False))

            ch_details = p.get("channel_details", [])
            n_out = sum(1 for c in ch_details if c.get("outlier", False))
            bad_status = [c.get("status", "") for c in ch_details
                         if c.get("status", "") not in ("pass1_ok", "pass2_ok", "pass3_ok", "")]
            s["n_outlier_ch"].append(n_out)
            s["n_ch"].append(len(ch_details))
            s["statuses"].append(bad_status)

    print(f"Positions found: {len(pos_series)}", flush=True)
    print()

    # Analyze each pos
    all_outliers = []

    for label in sorted(pos_series.keys(), key=lambda x: int(x.replace("Pos", "")) if x.replace("Pos", "").isdigit() else 0):
        s = pos_series[label]
        tx = np.array(s["tx"], dtype=float)
        ty = np.array(s["ty"], dtype=float)
        corr = np.array(s["corr"], dtype=float)
        tps = np.array(s["tp"])
        n_out_ch = np.array(s["n_outlier_ch"])
        n_ch = np.array(s["n_ch"])

        outlier_frames = set()
        reasons = defaultdict(list)

        # 1. Frame-to-frame jump detection
        dtx = np.diff(tx)
        dty = np.diff(ty)
        jump_mag = np.sqrt(dtx**2 + dty**2)
        for i in range(len(jump_mag)):
            if jump_mag[i] > JUMP_THRESHOLD_PX:
                outlier_frames.add(i + 1)
                reasons[i + 1].append(f"jump={jump_mag[i]:.2f}px (dtx={dtx[i]:+.2f}, dty={dty[i]:+.2f})")

        # 2. Low correlation
        for i in range(len(corr)):
            if corr[i] < LOW_CORR_THRESHOLD:
                outlier_frames.add(i)
                reasons[i].append(f"low_corr={corr[i]:.4f}")

        # 3. High outlier channel fraction
        for i in range(len(n_out_ch)):
            if n_ch[i] > 0:
                frac = n_out_ch[i] / n_ch[i]
                if frac > OUTLIER_CH_FRACTION:
                    outlier_frames.add(i)
                    reasons[i].append(f"outlier_ch={n_out_ch[i]}/{n_ch[i]} ({frac:.0%})")

        # 4. correction_valid = false
        for i, v in enumerate(s["valid"]):
            if not v:
                outlier_frames.add(i)
                reasons[i].append("correction_invalid")

        # 5. jump_detected by online system
        for i, j in enumerate(s["jump_detected"]):
            if j:
                outlier_frames.add(i)
                reasons[i].append("jump_detected_online")

        # 6. Bad channel statuses (tilt_bounds_ng etc)
        for i, st_list in enumerate(s["statuses"]):
            if st_list:
                outlier_frames.add(i)
                reasons[i].append(f"bad_ch_status={st_list}")

        if outlier_frames:
            all_outliers.append({
                "pos": label,
                "n_outlier": len(outlier_frames),
                "n_total": len(tps),
                "frames": sorted(outlier_frames),
                "reasons": {i: reasons[i] for i in sorted(outlier_frames)},
            })

    # Summary
    print("=" * 80)
    print(f"OUTLIER SUMMARY  (thresholds: jump>{JUMP_THRESHOLD_PX}px, corr<{LOW_CORR_THRESHOLD}, outlier_ch>{OUTLIER_CH_FRACTION:.0%})")
    print("=" * 80)

    total_outlier_frames = 0
    total_frames = 0
    pos_with_outliers = []

    for o in all_outliers:
        total_outlier_frames += o["n_outlier"]
        total_frames += o["n_total"]
        pct = o["n_outlier"] / o["n_total"] * 100
        pos_with_outliers.append(o)

    # Also count pos with zero outliers
    for label in pos_series:
        if not any(o["pos"] == label for o in all_outliers):
            total_frames += len(pos_series[label]["tp"])

    print(f"\nTotal positions: {len(pos_series)}")
    print(f"Positions with outliers: {len(all_outliers)}")
    print(f"Total outlier frames: {total_outlier_frames} / {total_frames} ({total_outlier_frames/max(total_frames,1)*100:.2f}%)")
    print()

    # Sort by outlier count descending
    all_outliers.sort(key=lambda x: x["n_outlier"], reverse=True)

    print(f"{'Pos':<8} {'Outliers':>8} {'Total':>6} {'%':>6}  {'Frame examples (reason)'}")
    print("-" * 80)
    for o in all_outliers[:30]:
        pct = o["n_outlier"] / o["n_total"] * 100
        # Show first 3 outlier frames with reasons
        examples = []
        for fr in o["frames"][:3]:
            r = ", ".join(o["reasons"][fr])
            examples.append(f"T{o['reasons'][fr][0].split('=')[0]}@{fr}")
        ex_str = "; ".join(f"T{fr}: {', '.join(o['reasons'][fr])}" for fr in o["frames"][:3])
        print(f"{o['pos']:<8} {o['n_outlier']:>8} {o['n_total']:>6} {pct:>5.1f}%  {ex_str[:100]}")

    if len(all_outliers) > 30:
        print(f"  ... and {len(all_outliers) - 30} more positions")

    # Detailed breakdown by reason type
    print()
    print("=" * 80)
    print("BREAKDOWN BY REASON")
    print("=" * 80)
    reason_counts = defaultdict(int)
    for o in all_outliers:
        for fr, rlist in o["reasons"].items():
            for r in rlist:
                rtype = r.split("=")[0]
                reason_counts[rtype] += 1
    for rtype, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype:<30s} {cnt:>6} occurrences")

    # Per-pos detail for top offenders
    print()
    print("=" * 80)
    print("TOP 10 WORST POSITIONS - DETAILED")
    print("=" * 80)
    for o in all_outliers[:10]:
        pct = o["n_outlier"] / o["n_total"] * 100
        print(f"\n{o['pos']}: {o['n_outlier']} outliers / {o['n_total']} frames ({pct:.1f}%)")
        for fr in o["frames"][:20]:
            reasons_str = "; ".join(o["reasons"][fr])
            print(f"  T={fr:>5d}: {reasons_str}")
        if len(o["frames"]) > 20:
            print(f"  ... +{len(o['frames'])-20} more")


if __name__ == "__main__":
    main()
