"""Test the patched selection in compute_drift_online against the live log.

Check 1 is the safety gate: with no whitelist the patched helpers must
reproduce every logged tx_avg_px exactly, otherwise the patch changed
production's rule and must be reverted before the next time point.

Check 2 compares the whitelist path to replay_drift_selection.py's independent
implementation, so two separate pieces of code have to agree.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(r"C:\Users\QPI\Documents\QPI_Omni\scripts")
SESSION = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261004_cells")
sys.path.insert(0, str(SCRIPTS))

from analyze_channel_selection_stability import build_tables
import drift_channel_whitelist as wlmod


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / (name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cdo = load("compute_drift_online")

cfg = json.loads((SESSION / "drift_config_zstack.json").read_text(encoding="utf-8"))
thr = cfg.get("ecc_min_corr", 0.9925)
px_nm = cfg.get("pixel_scale_um") * 1000.0
cell_bias_nm = cfg.get("cell_bias_nm", 0.0)
pos_split = cfg.get("pos_split", 57)

tab, ntp = build_tables(SESSION / "drift_log_zstack.json")
wl = {p: set(v) for p, v in json.loads(
    (SESSION / "channel_whitelist.json").read_text(encoding="utf-8"))["whitelist"].items()}

print("config: thr %.4f  cell_bias %.0f nm  pos_split %d" % (thr, cell_bias_nm, pos_split))
print("log   : %d Pos x %d time points" % (len(tab), ntp))


def decide(pos, t, use_wl):
    """Re-run the patched rule for one logged frame. Returns (tx_px, n_used, anchored)."""
    C, X, Y = tab[pos]["C"], tab[pos]["X"], tab[pos]["Y"]
    fin = np.isfinite(C[t])
    if not fin.any():
        return None
    idx = np.where(fin)[0]
    corr = C[t][idx]
    tx = X[t][idx]
    ty = Y[t][idx]
    n = len(idx)
    low = corr < thr
    if n >= 3:
        is_out = (cdo.remove_outliers_mad(list(tx), 5.0)
                  | cdo.remove_outliers_mad(list(ty), 5.0) | low)
    else:
        is_out = low

    mask = None
    if use_wl:
        chs = wl.get(pos)
        if chs:
            m = np.array([c in chs for c in idx], dtype=bool)
            mask = m if m.any() else None

    used, anchored = cdo.whitelist_used_idx(is_out, mask, n)
    v = float(np.mean(tx[used]))
    if (cell_bias_nm and not anchored and len(used) == n and bool(np.all(low))):
        sign = -1.0 if int(pos[3:]) >= pos_split else 1.0
        v += sign * cell_bias_nm / px_nm
    return v, len(used), anchored


# ---- check 1: no whitelist must reproduce the log exactly --------------------
err = []
for pos in tab:
    for t in range(ntp):
        r = decide(pos, t, False)
        if r is None:
            continue
        logged = tab[pos]["txavg"].get(t)
        if logged is not None and np.isfinite(logged):
            err.append((r[0] - logged) * px_nm)
err = np.asarray(err)
worst = float(np.abs(err).max())
print("\n[1] patched rule, whitelist OFF vs the log: max |error| %.6f nm over %d frames -- %s"
      % (worst, len(err), "exact" if worst < 1e-6 else "MISMATCH"))

# ---- check 2: whitelist path must agree with replay_drift_selection ----------
rep = load("replay_drift_selection")
static = {}
for pos, idx in wl.items():
    if pos not in tab:
        continue
    m = np.zeros(tab[pos]["C"].shape[1], dtype=bool)
    for i in idx:
        if 0 <= i < len(m):
            m[i] = True
    static[pos] = m
ref = rep.replay(tab, ntp, thr, px_nm, cell_bias_nm, pos_split, True,
                 wlmod.WINDOW, wlmod.WARMUP, static=static)

d, nmis = [], 0
for pos, t, tx_ref, n_ref, fb, anc_ref, sp in ref:
    r = decide(pos, t, True)
    assert r is not None
    d.append((r[0] - tx_ref) * px_nm)
    if r[1] != n_ref or bool(r[2]) != bool(anc_ref):
        nmis += 1
d = np.asarray(d)
print("[2] patched rule, whitelist ON vs replay_drift_selection: max |diff| %.6f nm "
      "over %d frames, %d frames disagree on (n_used, anchored) -- %s"
      % (float(np.abs(d).max()), len(d), nmis,
         "agree" if (np.abs(d).max() < 1e-6 and nmis == 0) else "MISMATCH"))

anc = np.array([b[5] for b in ref])
print("    anchored frames: %d (%.1f%%)" % (anc.sum(), 100 * anc.mean()))

ok = worst < 1e-6 and float(np.abs(d).max()) < 1e-6 and nmis == 0
print("\nVERDICT:", "PASS - the patch is safe to leave in place" if ok
      else "FAIL - revert scripts/compute_drift_online.py now")
sys.exit(0 if ok else 1)
