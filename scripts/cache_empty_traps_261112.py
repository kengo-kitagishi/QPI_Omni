"""Read every 261112 cell-free trap crop once into a memmap cache and report which traps exceed the emptiness threshold.

Output (scratch dir given as argv[1]):
  stacks.npy    float32 [n_traps, n_frames, 40, 240]
  index.csv     pos, ch, row, p999_max (99.9th percentile of the trap centre, max over frames), p999_med, min_frame
  flagged_montage.png  frame 10 of up to 48 flagged traps, inferno -0.5..0.5 rad
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

EMPTY_RAW = Path("D:/AquisitionData/Kitagishi/261112/online_crop_sub_2per_5min_31d_1")
REL = Path("output_phase/channels/crop_sub_rawraw/z000")
POS_SPLIT = 60
SKIP_POS = {"Pos124", "Pos0"}
THR = 0.5


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    traps = []
    for pd_ in sorted(EMPTY_RAW.glob("Pos*"), key=lambda p: int(p.name[3:])):
        if pd_.name in SKIP_POS:
            continue
        for cd in sorted((pd_ / REL).glob("ch*")):
            files = sorted(cd.glob("img_*_ph_000.tif"))
            if len(files) >= 10:
                traps.append((pd_.name, cd.name, files))
    n_fr = min(len(f) for _, _, f in traps)
    mm = np.lib.format.open_memmap(out / "stacks.npy", mode="w+", dtype=np.float32, shape=(len(traps), n_fr, 40, 240))
    rows = []
    for i, (pos, ch, files) in enumerate(traps):
        st = np.stack([tifffile.imread(str(p)) for p in files[:n_fr]])
        mm[i] = st
        c = st[:, 8:32, 20:220].reshape(n_fr, -1)
        p999 = np.percentile(c, 99.9, axis=1)
        rows.append({"i": i, "pos": pos, "ch": ch, "row": 0 if int(pos[3:]) < POS_SPLIT else 1,
                     "p999_max": float(p999.max()), "p999_med": float(np.median(p999)),
                     "min": float(c.min()), "p01_min": float(np.percentile(c, 0.1, axis=1).min())})
        if i % 200 == 0:
            print(i, pos, ch, flush=True)
    mm.flush()
    t = pd.DataFrame(rows)
    t.to_csv(out / "index.csv", index=False)
    t["flag"] = t.p999_max > THR
    print("n traps", len(t), "frames", n_fr, "flagged", int(t.flag.sum()))
    print(t.groupby(["row", "ch"]).flag.agg(["sum", "size"]).T.to_string())
    print("p999_max quantiles", t.p999_max.quantile([.1, .25, .5, .75, .9, .95]).round(3).to_dict())
    print("p999_med quantiles", t.p999_med.quantile([.1, .25, .5, .75, .9, .95]).round(3).to_dict())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fl = t[t.flag].sort_values("p999_max", ascending=False)
    sel = pd.concat([fl.iloc[::max(1, len(fl) // 24)].head(24), t[~t.flag].sample(min(24, (~t.flag).sum()), random_state=0)])
    fig, axes = plt.subplots(len(sel), 1, figsize=(8, 0.42 * len(sel)))
    for ax, (_, r) in zip(axes, sel.iterrows()):
        ax.imshow(mm[int(r.i), min(10, n_fr - 1)], cmap="inferno", vmin=-0.5, vmax=0.5, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(f"{r.pos} {r.ch}\n{r.p999_max:.2f}", fontsize=5, rotation=0, ha="right", va="center")
    fig.savefig(out / "flagged_montage.png", dpi=150, bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    main()
