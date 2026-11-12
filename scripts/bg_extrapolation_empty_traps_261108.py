"""How far off is the outside_quad background inside the trap? Measured in EMPTY traps (261108 run 3).

outside_quad fits a 2D quadratic on the walls outside the trap and extrapolates it into the trap. In a trap with
cells the interior is never free (traps are packed), so the extrapolation error is measured in the whitelist
(empty) traps instead:
  trap interior template = pixels covered by cells in >= HALF of the segmented traps of the same chip half
                           (Pos < POS_SPLIT / >= POS_SPLIT, the background side flips between them)
  residual(trap, frame)  = median phase of the template pixels in the crop_sub
Per trap: mean over frames (persistent offset) and SD over frames. For scale, the mean phase inside the segmented
cells is reported: a persistent offset d shifts a cell's integrated phase, hence its mass and concentration, by
about d / (mean cell phase).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from figure_logger import save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RAW = Path("D:/AquisitionData/Kitagishi/261108/online_crop_sub_YE_5min_31d_3")
MASK = Path("C:/QPI_analysis/261108/seg")
REL = Path("output_phase/channels/crop_sub_rawraw/z000")
WHITELIST = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session_261108_3\channel_whitelist.json")
SEG_LIST = Path(r"C:\QPI_analysis\261108\inputs\seg_channels.txt")
POS_SPLIT = 60
STEP = 40
HALF = 0.5
FIRST_HALF_ONLY = True    # user, 2026-11-12: Pos >= 60 carry the dirty diffraction grating, use Pos < 60 only


def frames(d):
    return sorted(d.glob("img_*_ph_000.tif"))[2:]


def main():
    seg = [l.split() for l in SEG_LIST.read_text().splitlines() if l.strip()]
    templates, cell_phase = {}, []
    for side in ((0,) if FIRST_HALF_ONLY else (0, 1)):
        unions = []
        for pos, ch in seg:
            if (int(pos[3:]) >= POS_SPLIT) != bool(side):
                continue
            d = MASK / pos / REL / ch / "inference_out"
            ms = [tifffile.imread(str(p)) > 0 for p in sorted(d.glob("*_masks.tif"))[::STEP]]
            if ms:
                unions.append(np.logical_or.reduce(ms))
                raw = RAW / pos / REL / ch
                for p in sorted(d.glob("*_masks.tif"))[::STEP]:
                    img = tifffile.imread(str(raw / p.name.replace("_masks", "")))
                    mk = tifffile.imread(str(p)) > 0
                    cell_phase.append(float(img[mk].mean()))
        templates[side] = np.mean(unions, axis=0) >= HALF
        print(f"side {side}: template from {len(unions)} traps, {templates[side].sum()} px")
    phi_cell = float(np.mean(cell_phase))

    wl = json.loads(WHITELIST.read_text(encoding="utf-8"))["whitelist"]
    rows = []
    for pos, chs in wl.items():
        side = int(int(pos[3:]) >= POS_SPLIT)
        if side not in templates:
            continue
        for c in chs:
            vals = []
            for f in frames(RAW / pos / REL / f"ch{c:02d}")[::STEP]:
                img = tifffile.imread(str(f))
                if img.shape == templates[side].shape:
                    vals.append(float(np.median(img[templates[side]])))
            if vals:
                rows.append((int(pos[3:]), c, np.mean(vals), np.std(vals), len(vals)))
    a = np.array(rows)
    mean_off, sd_t = a[:, 2], a[:, 3]
    res = {"n_empty_traps": int(len(a)), "between_trap_sd_mrad": float(np.std(mean_off) * 1e3),
           "offset_p5_p95_mrad": [float(np.percentile(mean_off, 5) * 1e3), float(np.percentile(mean_off, 95) * 1e3)],
           "median_abs_offset_mrad": float(np.median(np.abs(mean_off)) * 1e3),
           "within_trap_sd_over_time_mrad": float(np.mean(sd_t) * 1e3),
           "mean_cell_phase_rad": phi_cell,
           "relative_mass_error_from_between_trap_sd_pct": float(np.std(mean_off) / phi_cell * 100)}
    for k, v in res.items():
        print(f"{k:45s} {v}")
    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 60 / 25.4), constrained_layout=True)
    axes[0].hist(mean_off * 1e3, bins=60, color="#0072B2")
    axes[0].set_xlabel("persistent offset inside empty trap [mrad]")
    axes[0].set_ylabel("traps")
    axes[0].set_title(f"n = {len(a)} empty traps; between-trap SD {res['between_trap_sd_mrad']:.1f} mrad", fontsize=7)
    sc = axes[1].scatter(a[:, 0], a[:, 1], c=mean_off * 1e3, cmap="coolwarm", s=10,
                         vmin=-np.percentile(np.abs(mean_off), 98) * 1e3, vmax=np.percentile(np.abs(mean_off), 98) * 1e3)
    axes[1].set_xlabel("Pos")
    axes[1].set_ylabel("trap (ch)")
    fig.colorbar(sc, ax=axes[1], label="offset [mrad]")
    save_figure(
        fig, fmt="png", dpi=300, params={"step": STEP, "template_half": HALF, "pos_split": POS_SPLIT},
        description="261108 run 3: extrapolation error of the outside_quad background inside empty traps.",
        data={"pos": a[:, 0], "ch": a[:, 1], "offset": mean_off, "sd_time": sd_t}, extra_meta=res,
        caption=("Background left inside the trap after the outside_quad fit (quadratic fitted on the walls, "
                 "extrapolated into the trap), measured in the whitelist (cell-free) traps of 261108 YE run 3"
                 + (", Pos < 60 only (Pos >= 60 affected by a dirty diffraction grating). " if FIRST_HALF_ONLY else ". ")
                 + "Trap "
                 f"interior = pixels covered by cells in >= {HALF:.0%} of the segmented traps of the same chip half. "
                 f"Offset = median phase of those pixels, every {STEP}th frame, averaged over time (n = traps). "
                 f"Right: offset by Pos and trap. Mean phase inside segmented cells = {phi_cell:.2f} rad for scale. "
                 "No statistical test."),
    )


if __name__ == "__main__":
    main()
