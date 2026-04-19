# %%
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from figure_logger import save_figure

# === Data paths ===
DATA_DIR = r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox\2026-03-05\shift_visualize\shift_visualize_20260305T113225Z_cb7d7c"

FILES = {
    "shift = -0.5 px": os.path.join(DATA_DIR, "Values_m0p5.txt"),
    "shift = -1.0 px": os.path.join(DATA_DIR, "Values_m1.txt"),
    "shift = -1.5 px": os.path.join(DATA_DIR, "Values_m1p5.txt"),
}

# %%
# Load data
datasets = {}
for label, path in FILES.items():
    data = np.loadtxt(path)
    datasets[label] = {"x": data[:, 0], "y": data[:, 1]}

# Calculate common axis range
all_x = np.concatenate([d["x"] for d in datasets.values()])
all_y = np.concatenate([d["y"] for d in datasets.values()])
x_margin = (all_x.max() - all_x.min()) * 0.03
y_margin = (all_y.max() - all_y.min()) * 0.1
xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

colors = ["tab:blue", "tab:orange", "tab:green"]
ylabel = "Mean pixel value (rad)"
xlabel = "Frame index"

params_base = {
    "data_dir": DATA_DIR,
    "files": list(FILES.values()),
    "xlim": xlim,
    "ylim": ylim,
}

# %%
# Figures 1, 2, 3: Output each shift individually at the same scale
for (label, d), color in zip(datasets.items(), colors):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(d["x"], d["y"], color=color, marker="o", markersize=3, linewidth=1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(label)
    fig.tight_layout()
    save_figure(fig, params={**params_base, "target": label}, description=f"Values {label}")
    plt.close(fig)

print("done")
