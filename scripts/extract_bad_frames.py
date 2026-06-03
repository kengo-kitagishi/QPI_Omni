"""
extract_bad_frames.py
---------------------
Extract per-Pos bad timepoints from drift_log_zstack.json (T>2 only).
Outputs a JSON file for downstream exclusion.

Bad frame criteria:
  1. ECC total failure: ecc_correlation == 0 or < 0.5
  2. correction_valid == false
  3. All channels failed (all statuses are NG)
  4. Stage-settling pattern: "zure" (large excursion) and "modori" (partial
     recovery) frames.  The "mada" frame that follows is NOT marked bad.
       正常(tx~0) -> [ズレ: BAD] -> [戻り: BAD] -> まだ(tx~3, OK) -> 正常
"""
import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

DRIFT_LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log_zstack.json")
SKIP_TP = 3  # skip T=0,1,2

EXCURSION_TH = 5.0  # px: frame-to-frame jump that marks "zure"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: next to drift log)")
    args = ap.parse_args()

    out_path = Path(args.out) if args.out else DRIFT_LOG.parent / "bad_frames.json"

    print(f"Loading {DRIFT_LOG.name}...", flush=True)
    with open(DRIFT_LOG, "r") as f:
        data = json.load(f)
    n_tp = len(data)
    print(f"Loaded: {n_tp} timepoints", flush=True)

    pos_data = defaultdict(lambda: {
        "tp": [], "tx": [], "ty": [], "corr": [],
        "valid": [], "ch_details": []
    })

    for entry in data:
        tp = entry["timepoint"]
        for p in entry.get("positions", []):
            label = p.get("pos_label", f"Pos{p.get('pos_idx', '?')}")
            s = pos_data[label]
            s["tp"].append(tp)
            s["tx"].append(p.get("tx_avg_px", float("nan")))
            s["ty"].append(p.get("ty_avg_px", float("nan")))
            s["corr"].append(p.get("ecc_correlation", float("nan")))
            s["valid"].append(p.get("correction_valid", True))
            s["ch_details"].append(p.get("channel_details", []))

    del data

    result = {}
    total_bad = 0

    for label in sorted(pos_data, key=lambda x: int(x.replace("Pos", "")) if x.replace("Pos", "").isdigit() else 0):
        s = pos_data[label]
        tps = np.array(s["tp"])
        tx = np.array(s["tx"], dtype=float)
        ty = np.array(s["ty"], dtype=float)
        corr = np.array(s["corr"], dtype=float)

        bad = {}  # tp -> [reasons]

        dtx = np.diff(tx)
        dty = np.diff(ty)
        mag = np.sqrt(dtx**2 + dty**2)

        # Stage-settling: mark "zure" and "modori" frames
        # Pattern: normal -> [zure] -> [modori] -> mada -> normal
        #   diff[i] > TH  => frame i+1 is "zure" (jumped TO wrong position)
        #   frame i+2 is "modori" (partially recovered, usually low corr)
        zure_indices = set()
        modori_indices = set()
        for i in range(len(mag)):
            if mag[i] > EXCURSION_TH:
                zure_idx = i + 1
                modori_idx = i + 2
                if zure_idx < len(tps) and int(tps[zure_idx]) >= SKIP_TP:
                    zure_indices.add(zure_idx)
                if modori_idx < len(tps) and int(tps[modori_idx]) >= SKIP_TP:
                    modori_indices.add(modori_idx)

        for idx in range(len(tps)):
            tp = int(tps[idx])
            if tp < SKIP_TP:
                continue

            reasons = []

            # 1. ECC failure
            c = corr[idx]
            if c == 0.0 or np.isnan(c):
                reasons.append(f"ecc_fail(corr={c})")
            elif c < 0.5:
                reasons.append(f"low_corr({c:.3f})")

            # 2. correction_valid == false
            if not s["valid"][idx]:
                reasons.append("correction_invalid")

            # 3. All channels failed
            ch_list = s["ch_details"][idx]
            if ch_list:
                ok_statuses = {"pass1_ok", "pass2_ok", "pass3_ok"}
                n_ok = sum(1 for ch in ch_list if ch.get("status", "") in ok_statuses)
                if n_ok == 0:
                    reasons.append(f"all_ch_failed({len(ch_list)}ch)")

            # 4. Stage-settling
            if idx in zure_indices:
                jmag = mag[idx - 1]
                reasons.append(f"zure({jmag:.1f}px,tx={tx[idx]:+.1f},ty={ty[idx]:+.1f})")
            if idx in modori_indices:
                reasons.append(f"modori(tx={tx[idx]:+.1f},ty={ty[idx]:+.1f},corr={corr[idx]:.3f})")

            if reasons:
                bad[tp] = reasons

        if bad:
            result[label] = {
                "n_total": int(len(tps)),
                "n_bad": len(bad),
                "bad_timepoints": {int(k): v for k, v in sorted(bad.items())}
            }
            total_bad += len(bad)

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out_path}")

    n_pos_affected = len(result)
    n_pos_total = len(pos_data)
    n_frames_total = sum(len(s["tp"]) for s in pos_data.values())

    print()
    print("=" * 80)
    print("BAD FRAMES SUMMARY (T>2 only)")
    print("=" * 80)
    print(f"Positions: {n_pos_affected}/{n_pos_total} have bad frames")
    print(f"Total bad frames: {total_bad} / {n_frames_total} ({total_bad/n_frames_total*100:.2f}%)")
    print()

    sorted_pos = sorted(result.items(), key=lambda x: x[1]["n_bad"], reverse=True)
    print(f"{'Pos':<8} {'Bad':>5} {'Total':>6} {'%':>6}  Bad timepoints (first 10)")
    print("-" * 90)
    for label, info in sorted_pos[:40]:
        pct = info["n_bad"] / info["n_total"] * 100
        bt = list(info["bad_timepoints"].keys())
        tp_str = ", ".join(str(t) for t in bt[:10])
        if len(bt) > 10:
            tp_str += f" ... +{len(bt) - 10}"
        print(f"{label:<8} {info['n_bad']:>5} {info['n_total']:>6} {pct:>5.1f}%  {tp_str}")

    if len(sorted_pos) > 40:
        remaining = sum(info["n_bad"] for _, info in sorted_pos[40:])
        print(f"  ... +{len(sorted_pos) - 40} more Pos ({remaining} bad frames)")

    print()
    print("=" * 80)
    print("REASON BREAKDOWN")
    print("=" * 80)
    reason_counts = defaultdict(int)
    for label, info in result.items():
        for tp, reasons in info["bad_timepoints"].items():
            for r in reasons:
                rtype = r.split("(")[0]
                reason_counts[rtype] += 1
    for rtype, cnt in sorted(reason_counts.items(), key=lambda x: -x[1]):
        print(f"  {rtype:<25s} {cnt:>6}")


if __name__ == "__main__":
    main()
