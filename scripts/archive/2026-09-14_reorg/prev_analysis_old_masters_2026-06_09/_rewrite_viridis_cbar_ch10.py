"""Regenerate the standalone viridis colorbar for Pos51/ch10:
thinner bar, larger label/tick font.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize

OUT_PATH = Path(
    r"f:/260514/Pos51/output_phase/channels/crop_sub_raw_raw/ch10/viridis_pngs/_colorbar.png"
)
VMIN, VMAX = -0.5, 2.0
LABEL = "Phase (rad)"


def main() -> int:
    fig = plt.figure(figsize=(1.8, 4.2))
    # [left, bottom, width, height] in figure fraction — keep the bar narrow
    ax = fig.add_axes([0.18, 0.06, 0.18, 0.88])
    cb = ColorbarBase(
        ax,
        cmap=plt.get_cmap("viridis"),
        norm=Normalize(vmin=VMIN, vmax=VMAX),
        orientation="vertical",
    )
    cb.set_label(LABEL, fontsize=18)
    cb.ax.tick_params(labelsize=16, width=1.2, length=5)
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
