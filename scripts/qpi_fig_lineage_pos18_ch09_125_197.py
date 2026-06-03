"""Pos18_ch09 lineage tree restricted to 125-197 h window (relabeled so 197h = 0h).

Source CSVs (synced in figure-hub inbox):
  - clist.csv             (cell-level summary)
  - lineage_data3D.csv    (per-frame data)

Output:
  - vertical lines per cell, horizontal connectors mother→daughters
  - all lines gray (#666666)
  - y-axis: t_rel = t_abs - 197, so 197h → 0, 125h → -72 (inverted, mother at top)
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

INBOX = Path(
    "/Users/kitak/Library/CloudStorage/"
    "GoogleDrive-kengo_kitagishi@cell.c.u-tokyo.ac.jp/共有ドライブ/"
    "wakamotolab_meeting/kitagishi/figure-hub/inbox/2026-05-23/"
    "per_channel_figures/20260522T173317Z_108c71"
)
CLIST   = INBOX / "clist.csv"
LINEAGE = INBOX / "lineage_data3D.csv"

T_START_H = 125.0
T_END_H   = 197.0
T_REF_H   = 125.0   # the time labeled "0 h" (window start)
COLOR_LINE = "#666666"   # gray

clist = pd.read_csv(CLIST)
lineage = pd.read_csv(LINEAGE)

# keep cells whose lifespan overlaps the window
overlap = (clist["death_time_h"] >= T_START_H) & (clist["birth_time_h"] <= T_END_H)
sel = clist[overlap].copy()
print(f"cells overlapping [{T_START_H}, {T_END_H}] h: {len(sel)} / {len(clist)}")

# Build children map (parents come from the same selection: mother_id may not be in window)
sel_ids = set(sel["cell_id"])
children_map: dict[int, list[int]] = {}
for _, row in sel.iterrows():
    p = int(row["mother_id"])
    if p == -1:
        continue
    children_map.setdefault(p, []).append(int(row["cell_id"]))

# Roots = selected cells whose mother is NOT in selection
roots = []
for _, row in sel.iterrows():
    cid = int(row["cell_id"])
    p = int(row["mother_id"])
    if p == -1 or p not in sel_ids:
        roots.append(cid)
# sort roots by birth (oldest first)
roots.sort(key=lambda c: float(sel.loc[sel["cell_id"] == c, "birth_time_h"].iloc[0]))
print(f"roots (treated as 'mothers'): {len(roots)} -> ids {roots[:10]}{'...' if len(roots)>10 else ''}")

# Sort children: Mother Machine physical order — newer daughters get inserted closer
for v in children_map.values():
    v.sort(key=lambda c: float(sel.loc[sel["cell_id"] == c, "birth_time_h"].iloc[0]), reverse=True)

# Assign x positions: each subtree gets a contiguous range
positions: dict[int, float] = {}
def assign(cid: int, x: float) -> float:
    positions[cid] = x
    nxt = x + 1.0
    for ch in children_map.get(cid, []):
        if ch in sel_ids:
            nxt = assign(ch, nxt)
    return nxt

x_cursor = 0.0
for r in roots:
    x_cursor = assign(r, x_cursor)
n_cols = max(1, int(x_cursor))
print(f"n_cols = {n_cols}")


def t_rel(t_h):
    return t_h - T_REF_H  # 197 -> 0, 125 -> -72


plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
MM = 1 / 25.4

width_in = max(4.0, min(9.0, 0.045 * n_cols + 3.0))
fig, ax = plt.subplots(figsize=(width_in, 75 * MM), constrained_layout=True)

# horizontal connectors (parent -> daughter)
for cid in positions:
    row = sel.loc[sel["cell_id"] == cid].iloc[0]
    p = int(row["mother_id"])
    if p not in positions:
        continue
    y = t_rel(float(row["birth_time_h"]))
    ax.plot([positions[p], positions[cid]], [y, y],
            color=COLOR_LINE, lw=0.6)

# vertical cell lines: clip to [T_START_H, T_END_H]
for cid, x in positions.items():
    row = sel.loc[sel["cell_id"] == cid].iloc[0]
    b = max(T_START_H, float(row["birth_time_h"]))
    d = min(T_END_H,   float(row["death_time_h"]))
    if d <= b:
        continue
    ax.plot([x, x], [t_rel(b), t_rel(d)],
            color=COLOR_LINE, lw=0.6, solid_capstyle="butt")

ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(t_rel(T_END_H), t_rel(T_START_H))  # inverted: 0 at top? need to check
# We want mother (= start of window, earliest time) at TOP. earliest = T_START_H (=125h, t_rel=-72).
# matplotlib draws ylim from first to second along data axis but the SCREEN orientation:
# default is ymin at bottom, ymax at top. So invert: pass (T_END_H_rel, T_START_H_rel) =>
# matplotlib auto-inverts when ymin > ymax. Earliest (most negative) at top:
ax.set_ylim(t_rel(T_END_H), t_rel(T_START_H))  # (0, -72) -> auto-inverted, top=-72 (mother), bottom=0
ax.set_xlabel("Lineage (mother → daughters)")
ax.set_ylabel("Time [h]")
ax.set_xticks([])
ax.yaxis.set_major_locator(MultipleLocator(12))
ax.yaxis.set_minor_locator(MultipleLocator(6))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.tick_params(direction="in")

save_figure(
    fig,
    params={
        "channel":      "Pos18_ch09",
        "t_start_h":    T_START_H,
        "t_end_h":      T_END_H,
        "t_ref_h":      T_REF_H,
        "n_cells":      int(len(sel)),
        "n_roots":      len(roots),
        "n_cols":       n_cols,
        "color":        COLOR_LINE,
        "data_source":  str(LINEAGE),
    },
    description=f"Pos18_ch09 lineage tree {T_START_H:g}-{T_END_H:g}h (gray, t={T_REF_H:g}h relabeled as 0)",
    fmt="pdf",
)
save_figure(fig, params={"channel":"Pos18_ch09","fmt":"png"},
            description=f"Pos18_ch09 lineage tree {T_START_H:g}-{T_END_H:g}h gray", fmt="png")
plt.close(fig)
