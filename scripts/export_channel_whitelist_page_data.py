"""Summarise a drift log per (Pos, channel) so a whitelist can be picked by hand.

Writes one JSON for the review page: per channel, the score history binned over
the run, the pass rate against the current threshold, and how far that channel's
shift sits from the Pos's cell-free consensus. Those three together are what the
call "is this trap empty?" actually rests on.

Usage:
    python scripts/export_channel_whitelist_page_data.py <drift_log.json> <out.json>
                                                        [--config <cfg.json>] [--bins 96]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_channel_selection_stability import build_tables, stable_sets  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("out")
    ap.add_argument("--config", default=None)
    ap.add_argument("--bins", type=int, default=96)
    args = ap.parse_args()

    log_path = Path(args.log)
    cfg_path = Path(args.config) if args.config else log_path.parent / "drift_config_zstack.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    thr = cfg.get("ecc_min_corr", 0.9925)
    px_nm = cfg.get("pixel_scale_um", 0.34567514677103717) * 1000.0

    tab, ntp = build_tables(log_path)
    lab = stable_sets(tab, thr)
    nb = args.bins
    edges = np.linspace(0, ntp, nb + 1).astype(int)

    positions = []
    for p in sorted(tab, key=lambda s: int(s[3:])):
        C, X = tab[p]["C"], tab[p]["X"]
        L = lab[p]
        S = L["inner"] if L["inner"].sum() >= 2 else L["free"]
        fin = np.isfinite(C)
        # consensus = mean tx of the stably cell-free channels, per frame
        ref = np.full(ntp, np.nan)
        if S.sum() >= 2:
            for t in range(ntp):
                s = S & fin[t]
                if s.sum() >= 2:
                    ref[t] = X[t][s].mean()
        chans = []
        for c in np.where(L["ok"])[0]:
            sc = C[:, c]
            obs = np.isfinite(sc)
            # binned median score, -1 where the bin holds no frame
            trace = []
            for i in range(nb):
                w = sc[edges[i]:edges[i + 1]]
                w = w[np.isfinite(w)]
                trace.append(int(round(np.median(w) * 10000)) if len(w) else -1)
            off = (X[:, c] - ref) * px_nm
            off = off[np.isfinite(off)]
            chans.append({
                "ch": int(c),
                "pass": round(float(np.mean(sc[obs] >= thr)), 4),
                "med": round(float(np.median(sc[obs])), 5),
                "p5": round(float(np.percentile(sc[obs], 5)), 5),
                "off": round(float(np.median(np.abs(off))), 1) if len(off) else None,
                "offs": round(float(np.median(off)), 1) if len(off) else None,
                "n": int(obs.sum()),
                "auto": bool(S[c]),
                "end": bool(L["free"][c] and not L["inner"][c] and S is L["inner"]),
                "trace": trace,
            })
        positions.append({"pos": p, "ch": chans})

    payload = {
        "meta": {
            "log": str(log_path),
            "config": str(cfg_path),
            "threshold": thr,
            "estimator": cfg.get("estimator"),
            "cell_bias_nm": cfg.get("cell_bias_nm"),
            "n_tp": ntp,
            "n_pos": len(positions),
            "bins": nb,
            "interval_sec": cfg.get("interval_sec"),
            "px_nm": round(px_nm, 3),
            "save_dir": cfg.get("save_dir"),
        },
        "positions": positions,
    }
    Path(args.out).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    size = Path(args.out).stat().st_size
    print("wrote %s (%.1f KB): %d Pos, %d channels"
          % (args.out, size / 1024, len(positions), sum(len(q["ch"]) for q in positions)))


if __name__ == "__main__":
    main()
