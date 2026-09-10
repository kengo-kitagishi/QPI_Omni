"""
plot_cumdrift_260517.py
-----------------------
Cumulative drift figure from compute_pos_shifts output (pos_shifts_cal.json).

shift_x_avg / shift_y_avg are the per-frame ECC displacement (px) relative to
grid(0,0) = the absolute stage-drift trajectory. We convert px->um and plot the
trajectory over frames for every Pos that has a pos_shifts_cal.json (overlay),
plus the per-Pos total drift magnitude. One figure, saved locally (fast) and via
save_figure.

Usage: python plot_cumdrift_260517.py
"""
import sys, json, glob, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SD = Path(__file__).resolve().parent
sys.path.insert(0, str(_SD))

TL_ROOT = Path(r"E:\260517\2per_0055per_0per_2per")
# px -> um (sensor/mag/dim), same as compute_pos_shifts
PIXEL_SCALE_UM = 3.45e-6 / 40 * 2048 / 511 * 1e6


def load_pos(n):
    p = TL_ROOT / f"Pos{n}" / "z000" / "output_phase" / "channels" / "pos_shifts_cal.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    fr = d.get("frame_results") or d.get("alignment_results") or []
    fr = [e for e in fr if e]
    fr.sort(key=lambda e: e["frame_index"])
    t = np.array([e["frame_index"] for e in fr])
    sx = np.array([e.get("shift_x_avg", e.get("shift_x", 0.0)) for e in fr]) * PIXEL_SCALE_UM
    sy = np.array([e.get("shift_y_avg", e.get("shift_y", 0.0)) for e in fr]) * PIXEL_SCALE_UM
    return t, sx, sy


def main():
    poss = [n for n in range(1, 105) if load_pos(n) is not None]
    print(f"cumdrift: {len(poss)} Pos with pos_shifts_cal.json")
    if not poss:
        print("no data yet"); return

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    data = {}
    for n in poss:
        t, sx, sy = load_pos(n)
        ax[0].plot(t, sx, lw=0.4, alpha=0.5)
        ax[1].plot(t, sy, lw=0.4, alpha=0.5)
        mag = np.hypot(sx, sy)
        ax[2].plot(t, mag, lw=0.4, alpha=0.5)
        data[f"Pos{n}_t"] = t
        data[f"Pos{n}_x_um"] = sx
        data[f"Pos{n}_y_um"] = sy

    ax[0].set_title("Cumulative drift X")
    ax[0].set_xlabel("Frame"); ax[0].set_ylabel("Drift X (μm)")
    ax[1].set_title("Cumulative drift Y")
    ax[1].set_xlabel("Frame"); ax[1].set_ylabel("Drift Y (μm)")
    ax[2].set_title("Drift magnitude")
    ax[2].set_xlabel("Frame"); ax[2].set_ylabel("|drift| (μm)")
    fig.suptitle(f"260517 cumulative drift (all Pos, n={len(poss)}) — float ECC, z=8", fontsize=13)
    fig.tight_layout()

    # local save (fast)
    out_local = _SD.parent / "results" / "figures" / "cumdrift_260517.png"
    out_local.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_local), dpi=130)
    print(f"saved local: {out_local}")

    # also via save_figure (shared Drive)
    try:
        from figure_logger import save_figure
        save_figure(fig,
                    params={"n_pos": len(poss), "pixel_scale_um": PIXEL_SCALE_UM,
                            "z_index": 8, "ecc": "float32"},
                    description="260517 cumulative stage-drift trajectory (um) over frames, all Pos overlay",
                    data=data)
    except Exception as e:
        print(f"[save_figure] skipped: {e}")
    plt.close(fig)


if __name__ == "__main__":
    main()
