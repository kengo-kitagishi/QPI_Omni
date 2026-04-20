"""
qpi_fig_03_lineage_analysis.py — Cell lineage / cell cycle analysis for QPI data

Supported CSV formats (auto-detect):
  (A) ImageJ ROI Results: columns = Slice, Area, Mean (=RI)
  (B) Physical units (via batch_volume_trace_overlay -> npz_to_lineage_csv):
      columns = frame_index, volume_um3_rod, mean_ri, mass_pg

Analysis contents:
  1. Individual cell size and RI time series (with division event detection)
  2. Population mean +/- SEM (size, RI)
  3. Birth size vs Added size (size homeostasis: sizer / adder / timer classification)
  4. Birth RI vs Added RI (dry mass homeostasis)
  5. Division interval histogram
  6. Division interval per generation
  7. Cell cycle aligned trajectories (Size / RI / Dry mass)
  8. Density (RI) distribution histogram (Gaussian fit)
  9. Density homeostasis (birth RI vs delta RI)
  10. Growth rate analysis (dSize/dt, dMass/dt variation within cell cycle)

References:
  - Oldewurtel et al. (2021) eLife 10:e64901
    "Robust surface-to-mass coupling and turgor-dependent cell width
     determine bacterial dry-mass density"

Usage:
  python qpi_fig_03_lineage_analysis.py
  python qpi_fig_03_lineage_analysis.py --base-dir <dir> --mode physical
"""

# %%
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from figure_logger import save_figure

# =============================================================================
# === Settings ===
# =============================================================================

# --- Data directory (folder containing CSV files) ---
BASE_DIR = str(
    (
        __import__("pathlib").Path(__file__).resolve().parent
        / "results"
        / "lineage_csv"
        / "test_eroded_volume_overlay__Volume_overlay_8ch_eroded_mask_-1px_thin__20260417T013807Z_6f4157__f002"
    )
)

# --- CSV files to use (if empty list, use all *.csv in BASE_DIR) ---
FILEPATHS = []

# --- Cells to highlight (filename substring) ---
HIGHLIGHT_SERIES = []

# --- Data mode ("auto" / "imagej" / "physical") ---
# "physical": columns=frame_index,volume_um3_rod,mean_ri,mass_pg
# "imagej":   columns=Slice,Area,Mean
# "auto":     Auto-detect from CSV columns
DATA_MODE = "auto"

# --- Physical constants ---
PIXEL_AREA_TO_UM2 = (140 / 648) ** 2   # pixel^2 -> um^2 (magnification from ImageJ ROI analysis)
FRAMES_PER_HOUR = 12                    # 5 min interval = 12 frames/hour
N_MEDIUM = 1.333                        # Refractive index of medium
ALPHA_RI = 0.00018                      # [mL/mg] specific refractive increment
TIME_INTERVAL_H = 1 / FRAMES_PER_HOUR  # Frame interval [h]
N_INTERP = 100                          # Number of interpolation points for cell cycle aligned trajectory

# --- Media switch timing (frame number) ---
# 260405 dataset: [575, 875, 1439] (matches vline_frames in test_eroded_volume_overlay.py)
MEDIA_SWITCH_FRAMES = [575, 875, 1439]

# --- Analysis range (frames beyond this are excluded from all analyses) ---
# Use only the normal growth period before 1st media switch (frame 575). None for all frames.
ANALYSIS_MAX_FRAME: int | None = 575

# --- Division detection parameters ---
DIV_AREA_DROP = 0.6     # Size drops below this ratio -> detected as division
# Use only the normal growth period before starvation. None for all frames.
DIV_TIME_MAX_H = 575 / FRAMES_PER_HOUR

# --- Output ---
OUTPUT_DIR = "results/figures"

# --- ラベル（DATA_MODE に応じて main() で上書きされる） ---
SIZE_COL_LABEL = "Size"
SIZE_COL_UNIT = "a.u."
MASS_UNIT = "a.u."


# =============================================================================
# === Utilities ===
# =============================================================================

def load_filepaths(base_dir: str, filepaths: list[str]) -> list[str]:
    """Return a list of CSV file paths."""
    if filepaths:
        return filepaths
    return sorted(glob.glob(os.path.join(base_dir, "*.csv")))


def detect_data_mode(df: pd.DataFrame) -> str:
    """CSVのカラムからデータモードを判定する。"""
    if "volume_um3_rod" in df.columns and "mean_ri" in df.columns:
        return "physical"
    if "Slice" in df.columns and "Area" in df.columns:
        return "imagej"
    raise ValueError(
        f"Unknown CSV format. Columns: {list(df.columns)}. "
        "Expected either (Slice,Area,Mean) or (frame_index,volume_um3_rod,mean_ri,mass_pg)."
    )


def load_cell_data(filepath: str) -> pd.DataFrame:
    """Load CSV and normalize to common columns (Time, Size, RI, Density, DryMass).

    - "Size" refers to Area [um^2] in imagej mode and Volume [um^3] in physical mode
      ("size" is a dimension-agnostic generic name).
    - "DryMass" is a 2D proxy in imagej mode and direct mass_pg [pg] in physical mode.
    """
    df = pd.read_csv(filepath)
    mode = DATA_MODE if DATA_MODE in ("imagej", "physical") else detect_data_mode(df)

    if mode == "imagej":
        df = df.sort_values(by="Slice").reset_index(drop=True)
        df["Time"] = df["Slice"] / FRAMES_PER_HOUR              # [h]
        df["Size"] = df["Area"] * PIXEL_AREA_TO_UM2             # [um^2]
        df["RI"] = df["Mean"]
        df["Density"] = (df["RI"] - N_MEDIUM) / ALPHA_RI        # [mg/mL]
        df["DryMass"] = df["Density"] * df["Size"]              # 2D proxy
    elif mode == "physical":
        df = df.sort_values(by="frame_index").reset_index(drop=True)
        df["Time"] = df["frame_index"] / FRAMES_PER_HOUR        # [h]
        df["Size"] = df["volume_um3_rod"]                       # [um^3]
        df["RI"] = df["mean_ri"]
        df["Density"] = (df["RI"] - N_MEDIUM) / ALPHA_RI        # [mg/mL]
        df["DryMass"] = df["mass_pg"]                           # [pg] direct
    else:
        raise ValueError(f"Unsupported data mode: {mode}")

    # Backward-compatible alias: Area_um2 = Size
    df["Area_um2"] = df["Size"]
    df["_source"] = filepath
    df["_mode"] = mode
    # ANALYSIS_MAX_FRAME で解析範囲を制限
    if ANALYSIS_MAX_FRAME is not None:
        frame_col = df["frame_index"] if "frame_index" in df.columns else df["Slice"]
        df = df[frame_col <= ANALYSIS_MAX_FRAME].reset_index(drop=True)
    # 無効な行（NaN volume / RI）は落とす
    df = df.dropna(subset=["Size", "RI"]).reset_index(drop=True)
    return df


def detect_divisions(df: pd.DataFrame, drop_ratio: float = DIV_AREA_DROP) -> pd.DataFrame:
    """Detect division events from sudden drops in area.

    Returns:
        Rows of frames where division occurred (first frame after division).
    """
    prev_area = df["Area_um2"].shift(1)
    mask = df["Area_um2"] < drop_ratio * prev_area
    return df[mask].copy()


def extract_cell_cycles(df: pd.DataFrame,
                        drop_ratio: float = DIV_AREA_DROP,
                        max_time_h: float | None = None) -> list[dict]:
    """Extract cell cycle information from a single cell time series.

    Returns:
        list of dict with keys:
            birth_time, div_time, interval,
            birth_area, div_area, added_area,
            birth_ri, div_ri, added_ri
    """
    div_events = detect_divisions(df, drop_ratio)
    if max_time_h is not None:
        div_events = div_events[div_events["Time"] <= max_time_h]

    div_indices = div_events.index.tolist()
    if len(div_indices) < 2:
        return []

    cycles = []
    for i in range(len(div_indices) - 1):
        # birth = immediately after division, div = just before next division
        birth_idx = div_indices[i]
        div_idx = div_indices[i + 1] - 1  # One frame before division occurs
        if div_idx <= birth_idx:
            continue

        birth_row = df.loc[birth_idx]
        div_row = df.loc[div_idx]

        cycles.append({
            "birth_time": birth_row["Time"],
            "div_time": df.loc[div_indices[i + 1], "Time"],
            "interval": df.loc[div_indices[i + 1], "Time"] - birth_row["Time"],
            "birth_area": birth_row["Area_um2"],
            "div_area": div_row["Area_um2"],
            "added_area": div_row["Area_um2"] - birth_row["Area_um2"],
            "birth_ri": birth_row["RI"],
            "div_ri": div_row["RI"],
            "added_ri": div_row["RI"] - birth_row["RI"],
        })

    return cycles


def extract_cycle_traces(df: pd.DataFrame,
                         drop_ratio: float = DIV_AREA_DROP,
                         max_time_h: float | None = None,
                         n_interp: int = N_INTERP,
                         include_div_frame: bool = False) -> list[dict]:
    """Interpolate each cell cycle time series by relative progression (0->1).

    Corresponds to Oldewurtel et al. (2021) Fig 2B-D.

    Args:
        include_div_frame:
            False (default): cycle = [birth, next_div - 1]（分裂フレームを含めない）。
                rel_progress は半開区間 [0, 1) 上に正規化される。
            True: cycle = [birth, next_div]（分裂フレームを含める）。
                rel_progress は閉区間 [0, 1] 上に正規化される。

    Returns:
        list of dict with keys:
            rel_progress (0->1), area_interp, ri_interp, mass_interp,
            birth_time, interval
    """
    div_events = detect_divisions(df, drop_ratio)
    if max_time_h is not None:
        div_events = div_events[div_events["Time"] <= max_time_h]

    div_indices = div_events.index.tolist()
    if len(div_indices) < 2:
        return []

    traces = []
    rel = np.linspace(0, 1, n_interp)

    for i in range(len(div_indices) - 1):
        start = div_indices[i]
        end = div_indices[i + 1]
        if include_div_frame:
            cycle = df.loc[start:end]
        else:
            cycle = df.loc[start:end - 1]  # Exclude the next division frame
        if len(cycle) < 4:
            continue

        # Relative progression
        t = cycle["Time"].values
        if include_div_frame:
            # 分裂フレームが最終点。rel[-1] = 1 となる閉区間正規化。
            denom = t[-1] - t[0]
            interval = denom
        else:
            # 分裂フレームは含めない。+dt で "次の分裂までの時間" を反映。
            denom = t[-1] - t[0] + TIME_INTERVAL_H
            interval = denom
        t_rel = (t - t[0]) / denom

        # Interpolation
        area_i = np.interp(rel, t_rel, cycle["Area_um2"].values)
        ri_i = np.interp(rel, t_rel, cycle["RI"].values)
        mass_i = np.interp(rel, t_rel, cycle["DryMass"].values)

        traces.append({
            "rel_progress": rel,
            "area_interp": area_i,
            "ri_interp": ri_i,
            "mass_interp": mass_i,
            "birth_time": t[0],
            "interval": interval,
            "include_div_frame": include_div_frame,
        })

    return traces


def label_from_path(filepath: str) -> str:
    return os.path.basename(filepath).replace(".csv", "").replace("_Results", "")


def size_label() -> str:
    """E.g. 'Area [µm²]' or 'Volume [µm³]'."""
    return rf"{SIZE_COL_LABEL} [{SIZE_COL_UNIT}]"


def mass_label() -> str:
    return f"Dry mass [{MASS_UNIT}]"


def _apply_mode_labels(mode: str) -> None:
    """モードに応じて module-level ラベルを更新する。"""
    global SIZE_COL_LABEL, SIZE_COL_UNIT, MASS_UNIT
    if mode == "imagej":
        SIZE_COL_LABEL = "Area"
        SIZE_COL_UNIT = r"$\mu$m$^2$"
        MASS_UNIT = "a.u."
    elif mode == "physical":
        SIZE_COL_LABEL = "Volume"
        SIZE_COL_UNIT = r"$\mu$m$^3$"
        MASS_UNIT = "pg"


# =============================================================================
# === Color palette (Okabe-Ito) ===
# =============================================================================

OI = {
    "orange":    "#E69F00",
    "skyblue":   "#56B4E9",
    "green":     "#009E73",
    "yellow":    "#F0E442",
    "blue":      "#0072B2",
    "vermilion": "#D55E00",
    "purple":    "#CC79A7",
    "black":     "#000000",
}

# =============================================================================
# === Figure 1: Individual cell area and RI time series ===
# =============================================================================

def fig1_individual_traces(all_data: list[tuple[pd.DataFrame, str]]):
    """Individual cell area and RI time series. Division events marked with red dots."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(183/25.4, 120/25.4),
                                    sharex=True)

    for df, fp in all_data:
        label = label_from_path(fp)
        is_hl = any(h in fp for h in HIGHLIGHT_SERIES)
        divs = detect_divisions(df)

        # Area
        if is_hl or not HIGHLIGHT_SERIES:
            ax1.plot(df["Time"], df["Area_um2"], linewidth=0.6, label=label)
            ax1.scatter(divs["Time"], divs["Area_um2"],
                        color=OI["vermilion"], s=12, zorder=5)
        else:
            ax1.plot(df["Time"], df["Area_um2"], linewidth=0.15, color="0.75")
            ax1.scatter(divs["Time"], divs["Area_um2"],
                        color=OI["vermilion"], s=6, alpha=0.4)

        # RI
        if is_hl or not HIGHLIGHT_SERIES:
            ax2.plot(df["Time"], df["RI"], linewidth=0.6, label=label)
        else:
            ax2.plot(df["Time"], df["RI"], linewidth=0.15, color="0.75")

    for frame in MEDIA_SWITCH_FRAMES:
        t = frame / FRAMES_PER_HOUR
        ax1.axvline(t, color="0.6", ls="--", lw=0.5)
        ax2.axvline(t, color="0.6", ls="--", lw=0.5)

    ax1.set_ylabel(f"Cell {size_label()}", fontsize=8)
    ax2.set_ylabel("RI", fontsize=8)
    ax2.set_xlabel("Time [h]", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"base_dir": BASE_DIR, "n_cells": len(all_data),
                        "div_drop_ratio": DIV_AREA_DROP,
                        "data_mode": all_data[0][0]["_mode"].iloc[0] if all_data else "?"},
                description=f"Individual cell {SIZE_COL_LABEL.lower()} and RI traces with division events")
    return fig


# =============================================================================
# === Figure 2: Population mean +/- SEM ===
# =============================================================================

def fig2_population_mean(all_data: list[tuple[pd.DataFrame, str]]):
    """Group all cell area/RI by time and plot mean +/- SEM."""
    frames = [df[["Time", "Area_um2", "RI"]] for df, _ in all_data]
    df_all = pd.concat(frames, ignore_index=True)
    grouped = df_all.groupby("Time")

    mean_area = grouped["Area_um2"].mean()
    sem_area = grouped["Area_um2"].sem()
    mean_ri = grouped["RI"].mean()
    sem_ri = grouped["RI"].sem()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(89/25.4, 100/25.4),
                                    sharex=True)

    # Size (Area or Volume)
    ax1.plot(mean_area.index, mean_area.values, color=OI["blue"], lw=0.8,
             label=f"Mean {size_label()}")
    ax1.fill_between(mean_area.index,
                     mean_area - sem_area, mean_area + sem_area,
                     color=OI["blue"], alpha=0.25, label="±1 SEM")
    ax1.set_ylabel(size_label(), fontsize=8)
    ax1.legend(fontsize=6, loc="upper left")

    # RI
    ax2.plot(mean_ri.index, mean_ri.values, color=OI["green"], lw=0.8,
             label="Mean RI")
    ax2.fill_between(mean_ri.index,
                     mean_ri - sem_ri, mean_ri + sem_ri,
                     color=OI["green"], alpha=0.25, label="±1 SEM")
    ax2.set_ylabel("RI", fontsize=8)
    ax2.set_xlabel("Time [h]", fontsize=8)
    ax2.legend(fontsize=6, loc="upper left")

    for ax in (ax1, ax2):
        for frame in MEDIA_SWITCH_FRAMES:
            ax.axvline(frame / FRAMES_PER_HOUR, color="0.6", ls="--", lw=0.5)
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"n_cells": len(all_data)},
                description="Population mean ± SEM for area and RI")
    return fig


# =============================================================================
# === Figure 3: Size homeostasis (Birth size vs Added size / Birth RI vs Added RI) ===
# =============================================================================

def fig3_size_homeostasis(all_cycles: list[dict]):
    """Scatter plot of Birth size vs Added size. Calculate Pearson r."""
    df = pd.DataFrame(all_cycles)
    if len(df) < 5:
        print("  [fig3] Too few division events (<5). Skipping.")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 80/25.4))

    # --- Area homeostasis ---
    r_a, p_a = pearsonr(df["birth_area"], df["added_area"])
    ax1.scatter(df["birth_area"], df["added_area"],
                color=OI["blue"], alpha=0.6, s=15, edgecolors="none")
    # Mean crosshairs
    ax1.axvline(df["birth_area"].mean(), color="0.6", ls="--", lw=0.5)
    ax1.axhline(df["added_area"].mean(), color="0.6", ls="--", lw=0.5)
    ax1.set_xlabel(f"Birth {size_label()}", fontsize=8)
    ax1.set_ylabel(f"Added {size_label()}", fontsize=8)
    ax1.set_title(f"{SIZE_COL_LABEL} homeostasis (r={r_a:.2f}, p={p_a:.2e})", fontsize=8)

    # --- RI homeostasis ---
    r_ri, p_ri = pearsonr(df["birth_ri"], df["added_ri"])
    ax2.scatter(df["birth_ri"], df["added_ri"],
                color=OI["green"], alpha=0.6, s=15, edgecolors="none")
    ax2.axvline(df["birth_ri"].mean(), color="0.6", ls="--", lw=0.5)
    ax2.axhline(df["added_ri"].mean(), color="0.6", ls="--", lw=0.5)
    ax2.set_xlabel("Birth RI", fontsize=8)
    ax2.set_ylabel("Added RI", fontsize=8)
    ax2.set_title(f"RI homeostasis (r={r_ri:.2f}, p={p_ri:.2e})", fontsize=8)

    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"n_cycles": len(df),
                        "r_area": round(r_a, 3), "p_area": float(f"{p_a:.3e}"),
                        "r_ri": round(r_ri, 3), "p_ri": float(f"{p_ri:.3e}")},
                description="Cell size and RI homeostasis (birth vs added)")
    return fig


# =============================================================================
# === Figure 4: Division interval histogram ===
# =============================================================================

def fig4_division_interval_histogram(all_cycles: list[dict]):
    """Histogram of division intervals (cell cycle time)."""
    intervals = [c["interval"] for c in all_cycles]
    if len(intervals) < 3:
        print("  [fig4] Too few division events. Skipping.")
        return None

    intervals = np.array(intervals)
    mean_iv = np.mean(intervals)
    std_iv = np.std(intervals, ddof=1)

    fig, ax = plt.subplots(figsize=(89/25.4, 65/25.4))
    ax.hist(intervals, bins=np.arange(0, max(intervals) + 1, 0.5),
            color=OI["skyblue"], edgecolor="k", linewidth=0.3)
    ax.axvline(mean_iv, color=OI["vermilion"], ls="--", lw=1,
               label=f"Mean={mean_iv:.2f} h")
    ax.set_xlabel("Division interval [h]", fontsize=8)
    ax.set_ylabel("Frequency", fontsize=8)
    ax.set_title(f"Division intervals (n={len(intervals)}, "
                 f"mean={mean_iv:.2f}±{std_iv:.2f} h)", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"n_intervals": len(intervals),
                        "mean_h": round(mean_iv, 2),
                        "std_h": round(std_iv, 2),
                        "max_time_h": DIV_TIME_MAX_H},
                description="Division interval histogram")
    return fig


# =============================================================================
# === Figure 5: Division interval per generation ===
# =============================================================================

def fig5_interval_per_generation(all_data: list[tuple[pd.DataFrame, str]]):
    """Visualize division interval per generation using box plots."""
    max_gen = 12
    gen_intervals = {g: [] for g in range(max_gen)}

    for df, fp in all_data:
        div_events = detect_divisions(df)
        if DIV_TIME_MAX_H:
            div_events = div_events[div_events["Time"] <= DIV_TIME_MAX_H]
        div_times = div_events["Time"].values
        if len(div_times) < 2:
            continue
        intervals = np.diff(div_times)
        for g, iv in enumerate(intervals):
            if g < max_gen:
                gen_intervals[g].append(iv)

    # Exclude empty generations
    gen_data = {g: v for g, v in gen_intervals.items() if len(v) >= 2}
    if len(gen_data) < 2:
        print("  [fig5] Insufficient data. Skipping.")
        return None

    labels = [f"G{g}" for g in sorted(gen_data)]
    data = [gen_data[g] for g in sorted(gen_data)]

    fig, ax = plt.subplots(figsize=(89/25.4, 65/25.4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True,
                    widths=0.5,
                    medianprops=dict(color=OI["vermilion"], lw=1),
                    flierprops=dict(marker="o", markersize=3, alpha=0.5))
    for patch in bp["boxes"]:
        patch.set_facecolor(OI["skyblue"])
        patch.set_alpha(0.6)

    ax.set_xlabel("Generation", fontsize=8)
    ax.set_ylabel("Division interval [h]", fontsize=8)
    ax.set_title("Division interval per generation", fontsize=8)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"max_gen": max_gen,
                        "n_per_gen": {f"G{g}": len(v) for g, v in gen_data.items()},
                        "max_time_h": DIV_TIME_MAX_H},
                description="Division interval boxplot per generation")
    return fig


# =============================================================================
# === Figure 6: Division size vs RI (relationship between size and density) ===
# =============================================================================

def fig6_divsize_vs_ri(all_cycles: list[dict]):
    """Scatter plot of area vs RI at division."""
    df = pd.DataFrame(all_cycles)
    if len(df) < 5:
        print("  [fig6] Insufficient data. Skipping.")
        return None

    r, p = pearsonr(df["div_area"], df["div_ri"])

    fig, ax = plt.subplots(figsize=(89/25.4, 75/25.4))
    ax.scatter(df["div_area"], df["div_ri"],
               color=OI["purple"], alpha=0.6, s=15, edgecolors="none")
    ax.set_xlabel(f"Division {size_label()}", fontsize=8)
    ax.set_ylabel("RI at division", fontsize=8)
    ax.set_title(f"Division size vs RI (r={r:.2f}, p={p:.2e})", fontsize=8)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"n_points": len(df),
                        "r": round(r, 3), "p": float(f"{p:.3e}")},
                description="Division size vs RI scatter")
    return fig


# =============================================================================
# === Figure 7: Cell cycle aligned trajectories (Oldewurtel Fig 2B-D) ===
# =============================================================================

def fig7_aligned_trajectories(all_traces: list[dict]):
    """Overlay all cell cycles by relative progression and display Area / RI / Dry mass."""
    if len(all_traces) < 3:
        print("  [fig7] Insufficient traces (<3). Skipping.")
        return None

    rel = all_traces[0]["rel_progress"]
    areas = np.array([t["area_interp"] for t in all_traces])
    ris = np.array([t["ri_interp"] for t in all_traces])
    masses = np.array([t["mass_interp"] for t in all_traces])

    fig, axes = plt.subplots(3, 1, figsize=(89/25.4, 140/25.4), sharex=True)

    for ax, data, ylabel, color, label in [
        (axes[0], areas, size_label(), OI["blue"], SIZE_COL_LABEL),
        (axes[1], ris, "RI", OI["green"], "RI"),
        (axes[2], masses, mass_label(), OI["orange"], "Dry mass"),
    ]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # Individual traces (faint)
        for row in data:
            ax.plot(rel, row, color=color, alpha=0.08, lw=0.3)

        # Mean +/- SD
        ax.plot(rel, mean, color=color, lw=1.2, label=f"Mean (n={len(data)})")
        ax.fill_between(rel, mean - std, mean + std,
                        color=color, alpha=0.2, label="±1 SD")
        ax.set_ylabel(ylabel, fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    axes[2].set_xlabel("Relative cell cycle progression", fontsize=8)
    axes[0].set_title("Cell cycle aligned trajectories", fontsize=8)

    fig.tight_layout()
    save_figure(fig,
                params={"n_cycles": len(all_traces), "n_interp": N_INTERP,
                        "max_time_h": DIV_TIME_MAX_H},
                description="Cell cycle aligned Area/RI/DryMass trajectories "
                            "(Oldewurtel et al. 2021 style)")
    return fig


def fig7b_ri_cycle_end_compare(traces_excl: list[dict],
                                traces_incl: list[dict]):
    """cycle 末端定義 2 種類で mean RI vs rel_progress を並べて比較。

    左: cycle = [birth, next_div - 1]（分裂フレームを含めない、現行定義）
    右: cycle = [birth, next_div]（分裂フレームを含める）
    """
    if len(traces_excl) < 3 or len(traces_incl) < 3:
        print("  [fig7b] トレース不足。スキップ。")
        return None

    rel = traces_excl[0]["rel_progress"]
    ri_excl = np.array([t["ri_interp"] for t in traces_excl])
    ri_incl = np.array([t["ri_interp"] for t in traces_incl])

    fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 80 / 25.4),
                             sharex=True, sharey=True)

    for ax, data, title in [
        (axes[0], ri_excl,
         f"End = div frame − 1 (exclude)\nn={len(traces_excl)}"),
        (axes[1], ri_incl,
         f"End = div frame (include)\nn={len(traces_incl)}"),
    ]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        for row in data:
            ax.plot(rel, row, color=OI["green"], alpha=0.08, lw=0.3)
        ax.plot(rel, mean, color=OI["green"], lw=1.4, label="Mean")
        ax.fill_between(rel, mean - std, mean + std,
                        color=OI["green"], alpha=0.2, label="±1 SD")
        ax.set_xlabel("Relative cell cycle progression", fontsize=8)
        ax.set_ylabel("RI", fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.legend(fontsize=6, loc="upper left")
        ax.tick_params(labelsize=7)

    fig.suptitle("Mean RI vs relative cell cycle "
                 "(cycle-end definition comparison)", fontsize=9)
    fig.tight_layout()

    # Save both mean curves to data
    save_figure(fig,
                params={
                    "n_cycles_exclude_div": len(traces_excl),
                    "n_cycles_include_div": len(traces_incl),
                    "n_interp": N_INTERP,
                    "ri_mean_end_excl": float(np.mean(ri_excl, axis=0)[-1]),
                    "ri_mean_end_incl": float(np.mean(ri_incl, axis=0)[-1]),
                    "ri_mean_start_excl": float(np.mean(ri_excl, axis=0)[0]),
                    "ri_mean_start_incl": float(np.mean(ri_incl, axis=0)[0]),
                },
                data={
                    "rel_progress": rel,
                    "ri_mean_exclude_div": np.mean(ri_excl, axis=0),
                    "ri_std_exclude_div": np.std(ri_excl, axis=0),
                    "ri_mean_include_div": np.mean(ri_incl, axis=0),
                    "ri_std_include_div": np.std(ri_incl, axis=0),
                },
                description="Mean RI vs relative cell cycle "
                            "(compare cycle-end = div-1 vs cycle-end = div)")
    return fig


# =============================================================================
# === Figure 8: Density (RI) distribution histogram (Oldewurtel Fig 1C) ===
# =============================================================================

def _gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fig8_density_distribution(all_data: list[tuple[pd.DataFrame, str]]):
    """Display the RI distribution across all time points as histogram + Gaussian fit."""
    all_ri = np.concatenate([df["RI"].dropna().values for df, _ in all_data])
    if len(all_ri) < 10:
        print("  [fig8] Insufficient data. Skipping.")
        return None

    mu = np.mean(all_ri)
    sigma = np.std(all_ri)
    cv = sigma / mu

    fig, ax = plt.subplots(figsize=(89/25.4, 70/25.4))

    counts, bin_edges, _ = ax.hist(all_ri, bins=60, density=True,
                                    color=OI["skyblue"], edgecolor="k",
                                    linewidth=0.3, alpha=0.7)

    # Gaussian fit
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    try:
        popt, _ = curve_fit(_gaussian, bin_centers, counts,
                            p0=[counts.max(), mu, sigma])
        x_fit = np.linspace(all_ri.min(), all_ri.max(), 200)
        ax.plot(x_fit, _gaussian(x_fit, *popt), color=OI["vermilion"], lw=1.2,
                label=f"Gauss fit: µ={popt[1]:.4f}, σ={popt[2]:.4f}")
    except RuntimeError:
        pass

    ax.set_xlabel("RI", fontsize=8)
    ax.set_ylabel("Probability density", fontsize=8)
    ax.set_title(f"RI distribution (n={len(all_ri)}, CV={cv:.3f})", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"n_points": len(all_ri),
                        "mean_ri": round(mu, 5), "std_ri": round(sigma, 5),
                        "cv": round(cv, 4)},
                description="RI (density) distribution with Gaussian fit "
                            "(Oldewurtel et al. 2021 Fig 1C style)")
    return fig


# =============================================================================
# === Figure 9: Density homeostasis (Oldewurtel Fig 1D) ===
# =============================================================================

def fig9_density_homeostasis(all_cycles: list[dict]):
    """Verify the inverse correlation between birth RI (initial density) and delta RI (density change)."""
    df = pd.DataFrame(all_cycles)
    if len(df) < 5:
        print("  [fig9] Insufficient data. Skipping.")
        return None

    r, p = pearsonr(df["birth_ri"], df["added_ri"])

    fig, ax = plt.subplots(figsize=(89/25.4, 75/25.4))
    ax.scatter(df["birth_ri"], df["added_ri"],
               color=OI["green"], alpha=0.6, s=15, edgecolors="none")

    # Linear regression line
    z = np.polyfit(df["birth_ri"], df["added_ri"], 1)
    x_line = np.linspace(df["birth_ri"].min(), df["birth_ri"].max(), 50)
    ax.plot(x_line, np.polyval(z, x_line), color=OI["vermilion"], lw=1,
            ls="--", label=f"slope={z[0]:.2f}")

    ax.axhline(0, color="0.6", ls=":", lw=0.5)
    ax.set_xlabel("Birth RI (initial density)", fontsize=8)
    ax.set_ylabel("ΔRI (density change)", fontsize=8)
    ax.set_title(f"Density homeostasis (r={r:.2f}, p={p:.2e})", fontsize=8)
    ax.legend(fontsize=6)
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"n_cycles": len(df),
                        "r": round(r, 3), "p": float(f"{p:.3e}"),
                        "slope": round(z[0], 4)},
                description="Density homeostasis: birth RI vs delta RI "
                            "(Oldewurtel et al. 2021 Fig 1D style)")
    return fig


# =============================================================================
# === Figure 10: Growth rate analysis (dArea/dt, dMass/dt within cell cycle) ===
# =============================================================================

def fig10_growth_rate(all_traces: list[dict]):
    """Compute and plot growth rates within the relative cell cycle.

    By comparing volume (area) growth rate and mass growth rate,
    determine whether density fluctuations arise from changes in
    volume growth rate or biosynthesis rate (core argument of
    Oldewurtel et al.).
    """
    if len(all_traces) < 3:
        print("  [fig10] Insufficient traces. Skipping.")
        return None

    rel = all_traces[0]["rel_progress"]
    dr = rel[1] - rel[0]  # Step size in relative progression

    # Compute growth rate for each cycle (relative growth rate normalized by birth value)
    area_rates = []
    mass_rates = []

    for t in all_traces:
        a = t["area_interp"]
        m = t["mass_interp"]

        # Specific growth rate: (1/X) dX/d(rel_progress)
        # = d(ln X) / d(rel_progress)
        da = np.gradient(np.log(a + 1e-12), dr)
        dm = np.gradient(np.log(m + 1e-12), dr)

        area_rates.append(da)
        mass_rates.append(dm)

    area_rates = np.array(area_rates)
    mass_rates = np.array(mass_rates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(89/25.4, 110/25.4),
                                    sharex=True)

    size_rate_label = r"$\frac{1}{V}\frac{dV}{d\tau}$" if SIZE_COL_LABEL == "Volume" else r"$\frac{1}{A}\frac{dA}{d\tau}$"
    for ax, data, ylabel, color, title in [
        (ax1, area_rates, size_rate_label,
         OI["blue"], f"{SIZE_COL_LABEL} growth rate"),
        (ax2, mass_rates, r"$\frac{1}{M}\frac{dM}{d\tau}$",
         OI["orange"], "Dry mass growth rate"),
    ]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        for row in data:
            ax.plot(rel, row, color=color, alpha=0.06, lw=0.3)
        ax.plot(rel, mean, color=color, lw=1.2)
        ax.fill_between(rel, mean - std, mean + std,
                        color=color, alpha=0.2)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(title, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.axhline(np.mean(mean), color="0.5", ls=":", lw=0.5)

    ax2.set_xlabel("Relative cell cycle progression (τ)", fontsize=8)

    fig.tight_layout()
    save_figure(fig,
                params={"n_cycles": len(all_traces)},
                description="Specific growth rate of area and dry mass over "
                            "relative cell cycle (Oldewurtel et al. 2021 style)")
    return fig


# =============================================================================
# === Figure 11: Generation time homeostasis (interval_n vs interval_{n+1} / delta interval) ===
# =============================================================================

def fig11_interval_homeostasis(per_cell_cycles: list[list[dict]]):
    """Build consecutive-generation interval pairs within the same cell and plot two homeostasis views.

    Panel A: Poincare return map (interval_n vs interval_{n+1})
             slope~1: strong correlation (mother-daughter share similar generation time)
             slope~0: reset (generations are independent)
             slope<0: over-correction (homeostatic correction)

    Panel B: interval_n vs delta_interval (= interval_{n+1} - interval_n)
             Same format as Fig 3/9. Negative correlation indicates homeostasis.
    """
    pairs_n = []
    pairs_np1 = []
    for cycles in per_cell_cycles:
        intervals = [c["interval"] for c in cycles]
        for i in range(len(intervals) - 1):
            pairs_n.append(intervals[i])
            pairs_np1.append(intervals[i + 1])

    if len(pairs_n) < 5:
        print("  [fig11] Too few consecutive generation pairs (<5). Skipping.")
        return None

    pairs_n = np.array(pairs_n)
    pairs_np1 = np.array(pairs_np1)
    d_interval = pairs_np1 - pairs_n

    r_ab, p_ab = pearsonr(pairs_n, pairs_np1)
    r_dd, p_dd = pearsonr(pairs_n, d_interval)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183 / 25.4, 80 / 25.4))

    # --- Panel A: Poincare return map ---
    ax1.scatter(pairs_n, pairs_np1,
                color=OI["blue"], alpha=0.6, s=15, edgecolors="none")
    z = np.polyfit(pairs_n, pairs_np1, 1)
    x_line = np.linspace(pairs_n.min(), pairs_n.max(), 50)
    ax1.plot(x_line, np.polyval(z, x_line),
             color=OI["vermilion"], lw=1, ls="--",
             label=f"slope={z[0]:.2f}")
    # y=x reference
    lim = [min(pairs_n.min(), pairs_np1.min()), max(pairs_n.max(), pairs_np1.max())]
    ax1.plot(lim, lim, color="0.6", ls=":", lw=0.5, label="y=x")
    ax1.set_xlabel(r"Interval$_n$ [h]", fontsize=8)
    ax1.set_ylabel(r"Interval$_{n+1}$ [h]", fontsize=8)
    ax1.set_title(f"Return map (r={r_ab:.2f}, p={p_ab:.2e})", fontsize=8)
    ax1.legend(fontsize=6)
    ax1.tick_params(labelsize=7)

    # --- Panel B: interval_n vs delta_interval ---
    ax2.scatter(pairs_n, d_interval,
                color=OI["purple"], alpha=0.6, s=15, edgecolors="none")
    z2 = np.polyfit(pairs_n, d_interval, 1)
    x_line2 = np.linspace(pairs_n.min(), pairs_n.max(), 50)
    ax2.plot(x_line2, np.polyval(z2, x_line2),
             color=OI["vermilion"], lw=1, ls="--",
             label=f"slope={z2[0]:.2f}")
    ax2.axhline(0, color="0.6", ls=":", lw=0.5)
    ax2.set_xlabel(r"Interval$_n$ [h]", fontsize=8)
    ax2.set_ylabel(r"$\Delta$Interval (= Interval$_{n+1}$ - Interval$_n$) [h]", fontsize=8)
    ax2.set_title(f"Homeostasis (r={r_dd:.2f}, p={p_dd:.2e})", fontsize=8)
    ax2.legend(fontsize=6)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(
        fig,
        params={
            "n_pairs": int(len(pairs_n)),
            "return_slope": round(float(z[0]), 4),
            "return_r": round(float(r_ab), 3),
            "return_p": float(f"{p_ab:.3e}"),
            "homeostasis_slope": round(float(z2[0]), 4),
            "homeostasis_r": round(float(r_dd), 3),
            "homeostasis_p": float(f"{p_dd:.3e}"),
            "mean_interval_h": round(float(np.mean(pairs_n)), 3),
        },
        data={
            "interval_n": pairs_n,
            "interval_np1": pairs_np1,
            "delta_interval": d_interval,
        },
        description="Generation time homeostasis (Poincare return map + birth-vs-delta)",
    )
    return fig


# =============================================================================
# === Main ===
# =============================================================================

def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-dir", type=str, default=None,
                    help="CSVディレクトリ（省略時は BASE_DIR 定数）")
    ap.add_argument("--mode", choices=["auto", "imagej", "physical"], default=None,
                    help="データモード（省略時は DATA_MODE 定数）")
    ap.add_argument("--div-time-max-h", type=float, default=None,
                    help="この時刻以前の分裂のみ使用（省略時は DIV_TIME_MAX_H 定数）")
    ap.add_argument("--max-frame", type=int, default=None,
                    help="このフレーム以降を全解析から除外（省略時は ANALYSIS_MAX_FRAME 定数）")
    ap.add_argument("--div-drop", type=float, default=None,
                    help="分裂検出のサイズ低下比率（省略時は DIV_AREA_DROP 定数）")
    ap.add_argument("--frames-per-hour", type=float, default=None,
                    help="1時間あたりのフレーム数（省略時は FRAMES_PER_HOUR 定数）")
    ap.add_argument("--min-valid-frames", type=int, default=100,
                    help="有効フレーム数がこれ未満のチャネルは除外する")
    return ap.parse_args()


def main():
    args = _parse_args()
    global BASE_DIR, DATA_MODE, DIV_TIME_MAX_H, DIV_AREA_DROP, FRAMES_PER_HOUR, TIME_INTERVAL_H, ANALYSIS_MAX_FRAME
    if args.base_dir:
        BASE_DIR = args.base_dir
    if args.mode:
        DATA_MODE = args.mode
    if args.div_time_max_h is not None:
        DIV_TIME_MAX_H = args.div_time_max_h
    if args.div_drop is not None:
        DIV_AREA_DROP = args.div_drop
    if args.frames_per_hour is not None:
        FRAMES_PER_HOUR = args.frames_per_hour
        TIME_INTERVAL_H = 1 / FRAMES_PER_HOUR
    if args.max_frame is not None:
        ANALYSIS_MAX_FRAME = args.max_frame

    filepaths = load_filepaths(BASE_DIR, FILEPATHS)
    if not filepaths:
        print(f"ERROR: No CSV files found in {BASE_DIR}")
        sys.exit(1)

    print(f"=== Cell Lineage Analysis ===")
    print(f"  Data: {BASE_DIR}")
    print(f"  Files: {len(filepaths)}")

    # --- Load data ---
    all_data = []
    for fp in filepaths:
        df = load_cell_data(fp)
        if len(df) < args.min_valid_frames:
            print(f"  {label_from_path(fp)}: SKIP ({len(df)} valid frames < {args.min_valid_frames})")
            continue
        all_data.append((df, fp))
        print(f"  {label_from_path(fp)}: {len(df)} frames, "
              f"{df['Time'].max():.1f} h  (mode={df['_mode'].iloc[0]})")

    if not all_data:
        print("ERROR: No usable CSV after filtering.")
        sys.exit(1)

    # モードに応じたラベル適用
    detected_mode = all_data[0][0]["_mode"].iloc[0]
    _apply_mode_labels(detected_mode)
    print(f"  Mode: {detected_mode}  ({SIZE_COL_LABEL} [{SIZE_COL_UNIT}], Dry mass [{MASS_UNIT}])")
    if ANALYSIS_MAX_FRAME is not None:
        print(f"  Analysis max frame: {ANALYSIS_MAX_FRAME} "
              f"(= {ANALYSIS_MAX_FRAME / FRAMES_PER_HOUR:.1f} h)")
    print(f"  Div time cutoff: {DIV_TIME_MAX_H:.1f} h")

    # --- Extract cell cycles ---
    all_cycles = []
    per_cell_cycles = []  # Fig 11 用: 細胞ごとの連続サイクル
    for df, fp in all_data:
        cycles = extract_cell_cycles(df, max_time_h=DIV_TIME_MAX_H)
        all_cycles.extend(cycles)
        per_cell_cycles.append(cycles)
        n_div = len(detect_divisions(df))
        print(f"    → {n_div} divisions, {len(cycles)} complete cycles")

    print(f"\n  Total cycles: {len(all_cycles)}")

    # --- Extract cell cycle aligned traces ---
    all_traces = []       # Exclude division frame (current definition: cycle = [birth, div-1])
    all_traces_incl = []  # Include division frame (cycle = [birth, div])
    for df, fp in all_data:
        traces = extract_cycle_traces(df, max_time_h=DIV_TIME_MAX_H,
                                      include_div_frame=False)
        all_traces.extend(traces)
        traces_incl = extract_cycle_traces(df, max_time_h=DIV_TIME_MAX_H,
                                           include_div_frame=True)
        all_traces_incl.extend(traces_incl)
    print(f"  Aligned traces (excl div): {len(all_traces)}")
    print(f"  Aligned traces (incl div): {len(all_traces_incl)}")

    # --- Generate figures ---
    print("\n--- Generating figures ---")

    print("  [1/10] Individual traces...")
    fig1_individual_traces(all_data)

    print("  [2/10] Population mean ± SEM...")
    fig2_population_mean(all_data)

    print("  [3/10] Size homeostasis...")
    fig3_size_homeostasis(all_cycles)

    print("  [4/10] Division interval histogram...")
    fig4_division_interval_histogram(all_cycles)

    print("  [5/10] Interval per generation...")
    fig5_interval_per_generation(all_data)

    print("  [6/10] Division size vs RI...")
    fig6_divsize_vs_ri(all_cycles)

    print("  [7/10] Cell cycle aligned trajectories...")
    fig7_aligned_trajectories(all_traces)

    print("  [7b] Cycle-end definition comparison (RI)...")
    fig7b_ri_cycle_end_compare(all_traces, all_traces_incl)

    print("  [8/10] Density (RI) distribution...")
    fig8_density_distribution(all_data)

    print("  [9/10] Density homeostasis...")
    fig9_density_homeostasis(all_cycles)

    print("  [10/11] Growth rate analysis...")
    fig10_growth_rate(all_traces)

    print("  [11/11] Generation time homeostasis...")
    fig11_interval_homeostasis(per_cell_cycles)

    print("\nDone.")
    plt.show()


if __name__ == "__main__":
    main()
