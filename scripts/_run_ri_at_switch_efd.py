"""Generate ONLY the mean-RI-at-media-switch dead/alive histogram, EFD lineage.

Run with QPI_USE_CORRECTED=1 QPI_VOLUME_VARIANT=efd so find_lineage_csv resolves
to corrected_lineage_efd/ (EFD contour-section volume -> mean_ri = phase/volume).

This calls the canonical analyze_starvation_entry_cell_cycle.plot_ri_at_media_switches
(3 stacked histograms at frames 2018/2306/2884, revived=alive vs never_revived=dead),
and also prints the per-frame stats so they can be reported in chat.
"""
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from analyze_starvation_entry_cell_cycle import (  # noqa: E402
    plot_ri_at_media_switches, ri_at_frame, MEDIA_SWITCH_FRAMES,
    list_revived_mothers, list_never_revived_mothers,
)
from scipy.stats import ks_2samp  # noqa: E402


def vals(chs, frame):
    v = [ri_at_frame(p, c, frame) for p, c in chs]
    return np.array([x for x in v if x is not None])


def run():
    rev = list_revived_mothers()
    nr = list_never_revived_mothers()
    print(f"QPI_USE_CORRECTED={os.environ.get('QPI_USE_CORRECTED')} "
          f"QPI_VOLUME_VARIANT={os.environ.get('QPI_VOLUME_VARIANT')}")
    print(f"revived(alive)={len(rev)}, never_revived(dead)={len(nr)}")
    for frame, label in MEDIA_SWITCH_FRAMES:
        r = vals(rev, frame)
        n = vals(nr, frame)
        p = float(ks_2samp(r, n).pvalue) if len(r) and len(n) else float("nan")
        print(f"[frame {frame}] {label}")
        print(f"   alive(revived)   n={len(r):3d}  mean={np.mean(r):.5f}  median={np.median(r):.5f}")
        print(f"   dead(never_rev)  n={len(n):3d}  mean={np.mean(n):.5f}  median={np.median(n):.5f}")
        print(f"   KS p = {p:.4g}")
    plot_ri_at_media_switches(rev, nr)


if __name__ == "__main__":
    run()
