"""
qpi_fig_03_lineage_analysis.py — Cell lineage / cell cycle analysis for QPI data

ImageJ ROI Results.csv から細胞サイズ恒常性・分裂間隔・RI恒常性を解析する。
250416_kaiseki.pdf + Oldewurtel et al. (eLife 2021) のワークフローを
QPI (RI / dry mass) データに適用。

解析内容:
  1. 個別細胞の面積・RI 時系列（分裂イベント検出付き）
  2. 集団平均 ± SEM（面積・RI）
  3. Birth size vs Added size（サイズ恒常性: sizer / adder / timer 判定）
  4. Birth RI vs Added RI（乾燥質量恒常性）
  5. 分裂間隔ヒストグラム
  6. 世代ごとの分裂間隔推移
  7. Cell cycle aligned trajectories（Area / RI / Dry mass）
  8. Density (RI) 分布ヒストグラム（ガウスフィット）
  9. Density homeostasis（birth RI vs ΔRI）
  10. Growth rate 解析（dArea/dt, dMass/dt の cell cycle 内変動）

References:
  - Oldewurtel et al. (2021) eLife 10:e64901
    "Robust surface-to-mass coupling and turgor-dependent cell width
     determine bacterial dry-mass density"

Usage:
  python qpi_fig_03_lineage_analysis.py
"""

# %%
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
# === 設定 ===
# =============================================================================

# --- データディレクトリ（CSVファイルを含むフォルダ） ---
BASE_DIR = "/Users/kitak/Desktop/251105_QPI_results/0.0055_Results"

# --- 使用するCSVファイル（空リストなら BASE_DIR 内の *.csv を全て使う）---
FILEPATHS = []

# --- ハイライトする細胞（ファイル名の部分文字列） ---
HIGHLIGHT_SERIES = []

# --- 物理定数 ---
PIXEL_AREA_TO_UM2 = (140 / 648) ** 2   # pixel² → µm²（ImageJ ROI 解析時の倍率）
FRAMES_PER_HOUR = 12                    # 5分間隔 = 12フレーム/時
N_MEDIUM = 1.333                        # 培地の屈折率
ALPHA_RI = 0.00018                      # [mL/mg] specific refractive increment
TIME_INTERVAL_H = 1 / FRAMES_PER_HOUR  # フレーム間隔 [h]
N_INTERP = 100                          # cell cycle aligned trajectory の補間点数

# --- 培地切り替えタイミング（フレーム番号） ---
MEDIA_SWITCHES = [
    (0,    "wo_2"),     # 2% glucose
    (1145, "wo_0"),     # 0% glucose（飢餓）
    (1435, "wo_0"),     # 0% glucose（継続）
    (2014, "wo_2"),     # 2% glucose（回復）
]
MEDIA_SWITCH_FRAMES = [s[0] for s in MEDIA_SWITCHES if s[0] > 0]

# --- 分裂検出パラメータ ---
DIV_AREA_DROP = 0.6     # 面積がこの割合以下に低下 → 分裂と判定
DIV_TIME_MAX_H = 100    # この時刻以前の分裂のみ使用（飢餓前の正常増殖期）

# --- 出力 ---
OUTPUT_DIR = "results/figures"


# =============================================================================
# === ユーティリティ ===
# =============================================================================

def load_filepaths(base_dir: str, filepaths: list[str]) -> list[str]:
    """CSVファイルパスのリストを返す。"""
    if filepaths:
        return filepaths
    return sorted(glob.glob(os.path.join(base_dir, "*.csv")))


def load_cell_data(filepath: str) -> pd.DataFrame:
    """Results.csv を読み込み、物理単位に変換する。"""
    df = pd.read_csv(filepath)
    df = df.sort_values(by="Slice").reset_index(drop=True)
    df["Time"] = df["Slice"] / FRAMES_PER_HOUR          # [h]
    df["Area_um2"] = df["Area"] * PIXEL_AREA_TO_UM2      # [µm²]
    df["RI"] = df["Mean"]                                 # RI（補正は別途）
    # Dry mass proxy: concentration × area
    # C [mg/mL] = (RI - n_medium) / alpha_ri
    # dry_mass [pg] ∝ C × Area (2D proxy; 厳密には3D積分が必要)
    df["Density"] = (df["RI"] - N_MEDIUM) / ALPHA_RI     # [mg/mL]
    df["DryMass"] = df["Density"] * df["Area_um2"]        # [pg·µm² / mL] proxy
    df["_source"] = filepath
    return df


def detect_divisions(df: pd.DataFrame, drop_ratio: float = DIV_AREA_DROP) -> pd.DataFrame:
    """面積の急激な低下から分裂イベントを検出する。

    Returns:
        分裂が起きたフレームの行（分裂後の最初のフレーム）。
    """
    prev_area = df["Area_um2"].shift(1)
    mask = df["Area_um2"] < drop_ratio * prev_area
    return df[mask].copy()


def extract_cell_cycles(df: pd.DataFrame,
                        drop_ratio: float = DIV_AREA_DROP,
                        max_time_h: float | None = None) -> list[dict]:
    """1細胞の時系列からセルサイクルごとの情報を抽出する。

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
        # birth = 分裂直後, div = 次の分裂直前
        birth_idx = div_indices[i]
        div_idx = div_indices[i + 1] - 1  # 分裂が起きる1フレーム前
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
                         n_interp: int = N_INTERP) -> list[dict]:
    """各セルサイクルの時系列を相対進行度 (0→1) で補間して返す。

    Oldewurtel et al. (2021) Fig 2B-D に対応。

    Returns:
        list of dict with keys:
            rel_progress (0→1), area_interp, ri_interp, mass_interp,
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
        cycle = df.loc[start:end - 1]  # 次の分裂フレームは含めない
        if len(cycle) < 4:
            continue

        # 相対進行度
        t = cycle["Time"].values
        t_rel = (t - t[0]) / (t[-1] - t[0] + TIME_INTERVAL_H)

        # 補間
        area_i = np.interp(rel, t_rel, cycle["Area_um2"].values)
        ri_i = np.interp(rel, t_rel, cycle["RI"].values)
        mass_i = np.interp(rel, t_rel, cycle["DryMass"].values)

        traces.append({
            "rel_progress": rel,
            "area_interp": area_i,
            "ri_interp": ri_i,
            "mass_interp": mass_i,
            "birth_time": t[0],
            "interval": t[-1] - t[0] + TIME_INTERVAL_H,
        })

    return traces


def label_from_path(filepath: str) -> str:
    return os.path.basename(filepath).replace(".csv", "").replace("_Results", "")


# =============================================================================
# === カラーパレット（Okabe-Ito） ===
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
# === Figure 1: 個別細胞の面積・RI 時系列 ===
# =============================================================================

def fig1_individual_traces(all_data: list[tuple[pd.DataFrame, str]]):
    """個別細胞の面積とRI時系列。分裂イベントを赤点でマーク。"""
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

    ax1.set_ylabel(r"Cell Area [$\mu$m$^2$]", fontsize=8)
    ax2.set_ylabel("RI", fontsize=8)
    ax2.set_xlabel("Time [h]", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax2.tick_params(labelsize=7)

    fig.tight_layout()
    save_figure(fig,
                params={"base_dir": BASE_DIR, "n_cells": len(all_data),
                        "div_drop_ratio": DIV_AREA_DROP},
                description="Individual cell area and RI traces with division events")
    return fig


# =============================================================================
# === Figure 2: 集団平均 ± SEM ===
# =============================================================================

def fig2_population_mean(all_data: list[tuple[pd.DataFrame, str]]):
    """全細胞の面積・RIを時間でグルーピングし、平均±SEMをプロット。"""
    frames = [df[["Time", "Area_um2", "RI"]] for df, _ in all_data]
    df_all = pd.concat(frames, ignore_index=True)
    grouped = df_all.groupby("Time")

    mean_area = grouped["Area_um2"].mean()
    sem_area = grouped["Area_um2"].sem()
    mean_ri = grouped["RI"].mean()
    sem_ri = grouped["RI"].sem()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(89/25.4, 100/25.4),
                                    sharex=True)

    # Area
    ax1.plot(mean_area.index, mean_area.values, color=OI["blue"], lw=0.8,
             label=r"Mean Area [$\mu$m$^2$]")
    ax1.fill_between(mean_area.index,
                     mean_area - sem_area, mean_area + sem_area,
                     color=OI["blue"], alpha=0.25, label="±1 SEM")
    ax1.set_ylabel(r"Area [$\mu$m$^2$]", fontsize=8)
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
# === Figure 3: サイズ恒常性（Birth size vs Added size / Birth RI vs Added RI）===
# =============================================================================

def fig3_size_homeostasis(all_cycles: list[dict]):
    """Birth size vs Added size の散布図。Pearson r を算出。"""
    df = pd.DataFrame(all_cycles)
    if len(df) < 5:
        print("  [fig3] 分裂イベントが少なすぎます（<5）。スキップ。")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(183/25.4, 80/25.4))

    # --- Area homeostasis ---
    r_a, p_a = pearsonr(df["birth_area"], df["added_area"])
    ax1.scatter(df["birth_area"], df["added_area"],
                color=OI["blue"], alpha=0.6, s=15, edgecolors="none")
    # 平均の十字線
    ax1.axvline(df["birth_area"].mean(), color="0.6", ls="--", lw=0.5)
    ax1.axhline(df["added_area"].mean(), color="0.6", ls="--", lw=0.5)
    ax1.set_xlabel(r"Birth size [$\mu$m$^2$]", fontsize=8)
    ax1.set_ylabel(r"Added size [$\mu$m$^2$]", fontsize=8)
    ax1.set_title(f"Area homeostasis (r={r_a:.2f}, p={p_a:.2e})", fontsize=8)

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
# === Figure 4: 分裂間隔ヒストグラム ===
# =============================================================================

def fig4_division_interval_histogram(all_cycles: list[dict]):
    """分裂間隔（cell cycle time）のヒストグラム。"""
    intervals = [c["interval"] for c in all_cycles]
    if len(intervals) < 3:
        print("  [fig4] 分裂イベントが少なすぎます。スキップ。")
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
# === Figure 5: 世代ごとの分裂間隔推移 ===
# =============================================================================

def fig5_interval_per_generation(all_data: list[tuple[pd.DataFrame, str]]):
    """各細胞の世代ごとの分裂間隔をボックスプロットで可視化。"""
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

    # 空の世代を除外
    gen_data = {g: v for g, v in gen_intervals.items() if len(v) >= 2}
    if len(gen_data) < 2:
        print("  [fig5] データ不足。スキップ。")
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
# === Figure 6: 分裂時サイズ vs RI（サイズと密度の関係） ===
# =============================================================================

def fig6_divsize_vs_ri(all_cycles: list[dict]):
    """分裂時の面積 vs RI の散布図。"""
    df = pd.DataFrame(all_cycles)
    if len(df) < 5:
        print("  [fig6] データ不足。スキップ。")
        return None

    r, p = pearsonr(df["div_area"], df["div_ri"])

    fig, ax = plt.subplots(figsize=(89/25.4, 75/25.4))
    ax.scatter(df["div_area"], df["div_ri"],
               color=OI["purple"], alpha=0.6, s=15, edgecolors="none")
    ax.set_xlabel(r"Division size (Area [$\mu$m$^2$])", fontsize=8)
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
# === Figure 7: Cell cycle aligned trajectories（Oldewurtel Fig 2B-D）===
# =============================================================================

def fig7_aligned_trajectories(all_traces: list[dict]):
    """全セルサイクルを相対進行度で重ね合わせ、Area / RI / Dry mass を表示。"""
    if len(all_traces) < 3:
        print("  [fig7] トレース不足（<3）。スキップ。")
        return None

    rel = all_traces[0]["rel_progress"]
    areas = np.array([t["area_interp"] for t in all_traces])
    ris = np.array([t["ri_interp"] for t in all_traces])
    masses = np.array([t["mass_interp"] for t in all_traces])

    fig, axes = plt.subplots(3, 1, figsize=(89/25.4, 140/25.4), sharex=True)

    for ax, data, ylabel, color, label in [
        (axes[0], areas, r"Area [$\mu$m$^2$]", OI["blue"], "Area"),
        (axes[1], ris, "RI", OI["green"], "RI"),
        (axes[2], masses, "Dry mass [a.u.]", OI["orange"], "Dry mass"),
    ]:
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)

        # 個別トレース（薄く）
        for row in data:
            ax.plot(rel, row, color=color, alpha=0.08, lw=0.3)

        # 平均 ± SD
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


# =============================================================================
# === Figure 8: Density (RI) 分布ヒストグラム（Oldewurtel Fig 1C）===
# =============================================================================

def _gaussian(x, a, mu, sigma):
    return a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fig8_density_distribution(all_data: list[tuple[pd.DataFrame, str]]):
    """全時点のRI分布をヒストグラム＋ガウスフィットで表示。"""
    all_ri = np.concatenate([df["RI"].dropna().values for df, _ in all_data])
    if len(all_ri) < 10:
        print("  [fig8] データ不足。スキップ。")
        return None

    mu = np.mean(all_ri)
    sigma = np.std(all_ri)
    cv = sigma / mu

    fig, ax = plt.subplots(figsize=(89/25.4, 70/25.4))

    counts, bin_edges, _ = ax.hist(all_ri, bins=60, density=True,
                                    color=OI["skyblue"], edgecolor="k",
                                    linewidth=0.3, alpha=0.7)

    # ガウスフィット
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
# === Figure 9: Density homeostasis（Oldewurtel Fig 1D）===
# =============================================================================

def fig9_density_homeostasis(all_cycles: list[dict]):
    """Birth RI（初期密度）vs ΔRI（密度変化）の逆相関を検証。"""
    df = pd.DataFrame(all_cycles)
    if len(df) < 5:
        print("  [fig9] データ不足。スキップ。")
        return None

    r, p = pearsonr(df["birth_ri"], df["added_ri"])

    fig, ax = plt.subplots(figsize=(89/25.4, 75/25.4))
    ax.scatter(df["birth_ri"], df["added_ri"],
               color=OI["green"], alpha=0.6, s=15, edgecolors="none")

    # 線形回帰ライン
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
# === Figure 10: Growth rate 解析（cell cycle 内の dArea/dt, dMass/dt）===
# =============================================================================

def fig10_growth_rate(all_traces: list[dict]):
    """相対 cell cycle 内の成長率を算出しプロット。

    Volume (area) growth rate と mass growth rate を比較することで、
    密度変動が体積成長速度の変化に起因するか、物質合成速度の変化に
    起因するかを判別する（Oldewurtel et al. の核心的議論）。
    """
    if len(all_traces) < 3:
        print("  [fig10] トレース不足。スキップ。")
        return None

    rel = all_traces[0]["rel_progress"]
    dr = rel[1] - rel[0]  # 相対進行度のステップ幅

    # 各サイクルの成長率を計算（birth value で正規化した相対成長率）
    area_rates = []
    mass_rates = []

    for t in all_traces:
        a = t["area_interp"]
        m = t["mass_interp"]

        # 相対成長率: (1/X) dX/d(rel_progress)
        # = d(ln X) / d(rel_progress)
        da = np.gradient(np.log(a + 1e-12), dr)
        dm = np.gradient(np.log(m + 1e-12), dr)

        area_rates.append(da)
        mass_rates.append(dm)

    area_rates = np.array(area_rates)
    mass_rates = np.array(mass_rates)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(89/25.4, 110/25.4),
                                    sharex=True)

    for ax, data, ylabel, color, title in [
        (ax1, area_rates, r"$\frac{1}{A}\frac{dA}{d\tau}$",
         OI["blue"], "Area (volume) growth rate"),
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
# === メイン ===
# =============================================================================

def main():
    filepaths = load_filepaths(BASE_DIR, FILEPATHS)
    if not filepaths:
        print(f"ERROR: No CSV files found in {BASE_DIR}")
        sys.exit(1)

    print(f"=== Cell Lineage Analysis ===")
    print(f"  Data: {BASE_DIR}")
    print(f"  Files: {len(filepaths)}")

    # --- データ読み込み ---
    all_data = []
    for fp in filepaths:
        df = load_cell_data(fp)
        all_data.append((df, fp))
        print(f"  {label_from_path(fp)}: {len(df)} frames, "
              f"{df['Time'].max():.1f} h")

    # --- セルサイクル抽出 ---
    all_cycles = []
    for df, fp in all_data:
        cycles = extract_cell_cycles(df, max_time_h=DIV_TIME_MAX_H)
        all_cycles.extend(cycles)
        n_div = len(detect_divisions(df))
        print(f"    → {n_div} divisions, {len(cycles)} complete cycles")

    print(f"\n  Total cycles: {len(all_cycles)}")

    # --- Cell cycle aligned traces 抽出 ---
    all_traces = []
    for df, fp in all_data:
        traces = extract_cycle_traces(df, max_time_h=DIV_TIME_MAX_H)
        all_traces.extend(traces)
    print(f"  Aligned traces: {len(all_traces)}")

    # --- 図生成 ---
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

    print("  [8/10] Density (RI) distribution...")
    fig8_density_distribution(all_data)

    print("  [9/10] Density homeostasis...")
    fig9_density_homeostasis(all_cycles)

    print("  [10/10] Growth rate analysis...")
    fig10_growth_rate(all_traces)

    print("\nDone.")
    plt.show()


if __name__ == "__main__":
    main()
