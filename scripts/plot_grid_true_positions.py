# %%
"""
plot_grid_true_positions.py
----------------------------
visualize_grid_true_positions.py の Direct ECC 結果 (npz) を読み込んで
グリッド真位置を詳細に可視化する。ECC の再計算なし。

【4パネル構成】
  Panel 1: Nominal lattice vs Actual positions (scatter)
  Panel 2: Displacement quiver — nominal → actual の変位ベクトル（QUIVER_SCALE 倍）
  Panel 3: dx error 2D heatmap (actual_dx - nominal_dx) [um]
  Panel 4: dy error 2D heatmap (actual_dy - nominal_dy) [um]
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ============================================================
# 設定: Direct ECC npz のパス（visualize_grid_true_positions.py の出力）
# ============================================================
NPZ_PATH = (
    r"G:\共有ドライブ\wakamotolab_meeting\kitagishi\figure-hub\inbox"
    r"\2026-03-30\visualize_grid_true_positions\20260330T034611Z_1b7af7"
    r"\visualize_grid_true_positions__Grid_true_position_check_Pos1_z_9_direct"
    r"__20260330T034611Z_1b7af7__f001_data.npz"
)

# quiver の矢印スケール（変位を何倍に拡大して描くか）
QUIVER_SCALE = 20   # 20倍: 0.1 um = 2 um 分の矢印長さで表示

POS_PREFIX = "Pos1"
Z_IDX      = 9
# ============================================================


def load_npz(path):
    d = np.load(str(path))
    return {k: d[k] for k in d.files}


def reshape_to_grid(xi, yi, values):
    """
    (xi, yi) → values を 2D grid (xi_axis × yi_axis) に変換。
    xi_axis: 昇順ソート済みのユニーク xi 値
    yi_axis: 昇順ソート済みのユニーク yi 値
    返り値: (grid_2d, xi_axis, yi_axis)
    """
    xi_u = np.sort(np.unique(xi))
    yi_u = np.sort(np.unique(yi))
    grid = np.full((len(xi_u), len(yi_u)), np.nan)
    xi_idx = {v: i for i, v in enumerate(xi_u)}
    yi_idx = {v: i for i, v in enumerate(yi_u)}
    for k in range(len(xi)):
        grid[xi_idx[xi[k]], yi_idx[yi[k]]] = values[k]
    return grid, xi_u, yi_u


def main():
    d = load_npz(NPZ_PATH)
    nom_dx = d["nominal_dx_um"]   # [um]
    nom_dy = d["nominal_dy_um"]
    act_dx = d["actual_dx_um"]
    act_dy = d["actual_dy_um"]
    xi     = d["xi"].astype(int)
    yi     = d["yi"].astype(int)
    corr   = d["corr"]

    err_dx = act_dx - nom_dx   # [um]
    err_dy = act_dy - nom_dy
    residual = np.sqrt(err_dx**2 + err_dy**2)

    print(f"Points: {len(xi)}")
    print(f"err_dx [um]: mean={err_dx.mean():.4f}  std={err_dx.std():.4f}  "
          f"max_abs={np.abs(err_dx).max():.4f}")
    print(f"err_dy [um]: mean={err_dy.mean():.4f}  std={err_dy.std():.4f}  "
          f"max_abs={np.abs(err_dy).max():.4f}")
    print(f"residual [um]: mean={residual.mean():.4f}  max={residual.max():.4f}")

    # ---- 2D grid 変換 ----
    grid_dx,  xi_u, yi_u = reshape_to_grid(xi, yi, err_dx)
    grid_dy,  _,    _    = reshape_to_grid(xi, yi, err_dy)
    grid_res, _,    _    = reshape_to_grid(xi, yi, residual)

    # quiver 用: nominal 位置 = (nom_dx, nom_dy)
    # Panel 2 は xi,yi 軸を使うシンプル quiver
    sym_max = max(np.abs(err_dx).max(), np.abs(err_dy).max(), 0.01)

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Grid true position map — Direct ECC  [{POS_PREFIX}, z={Z_IDX}]\n"
        f"err_dx: mean={err_dx.mean():+.4f} um  std={err_dx.std():.4f} um  |  "
        f"err_dy: mean={err_dy.mean():+.4f} um  std={err_dy.std():.4f} um",
        fontsize=10,
    )

    # ------ Panel 1: Nominal vs Actual scatter ------
    ax = axes[0]
    ax.scatter(nom_dx, nom_dy, s=60, facecolors="none", edgecolors="#aaa",
               linewidths=1.0, zorder=2, label="Nominal")
    sc = ax.scatter(act_dx, act_dy, s=20, c=residual, cmap="hot_r",
                    vmin=0, vmax=0.35, zorder=3, label="Actual")
    plt.colorbar(sc, ax=ax, label="Residual (um)")
    ax.set_xlabel("dx (um)  [image X / stage Y dir.]")
    ax.set_ylabel("dy (um)  [image Y / stage X dir.]")
    ax.set_title("Nominal (○) vs Actual (●)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # ------ Panel 2: Displacement quiver ------
    ax2 = axes[1]
    # 矢印の根本 = nominal position (nom_dx, nom_dy)
    quiv = ax2.quiver(
        nom_dx, nom_dy,
        err_dx * QUIVER_SCALE,
        err_dy * QUIVER_SCALE,
        residual,
        cmap="hot_r", clim=(0, 0.35),
        angles="xy", scale_units="xy", scale=1,
        width=0.004,
    )
    plt.colorbar(quiv, ax=ax2, label="Residual (um)")
    ax2.set_xlabel("nom_dx (um)")
    ax2.set_ylabel("nom_dy (um)")
    ax2.set_title(f"Displacement vectors  (×{QUIVER_SCALE})")
    ax2.set_aspect("equal")
    ax2.grid(True, linewidth=0.3, alpha=0.4)

    # ------ Panel 3: dx error heatmap ------
    ax3 = axes[2]
    im3 = ax3.imshow(
        grid_dx,
        origin="lower",
        extent=[yi_u[0] - 0.5, yi_u[-1] + 0.5,
                xi_u[0] - 0.5, xi_u[-1] + 0.5],
        cmap="RdBu_r",
        vmin=-sym_max, vmax=sym_max,
        aspect="auto",
    )
    plt.colorbar(im3, ax=ax3, label="err_dx = actual_dx − nom_dx (um)")
    ax3.set_xlabel("yi  (stage Y index)")
    ax3.set_ylabel("xi  (stage X index)")
    ax3.set_title("dx error map (image X / stage Y)")
    for i, xv in enumerate(xi_u):
        for j, yv in enumerate(yi_u):
            val = grid_dx[i, j]
            if not np.isnan(val):
                ax3.text(yv, xv, f"{val:.2f}", ha="center", va="center",
                         fontsize=4.5, color="k")

    # ------ Panel 4: dy error heatmap ------
    ax4 = axes[3]
    im4 = ax4.imshow(
        grid_dy,
        origin="lower",
        extent=[yi_u[0] - 0.5, yi_u[-1] + 0.5,
                xi_u[0] - 0.5, xi_u[-1] + 0.5],
        cmap="RdBu_r",
        vmin=-sym_max, vmax=sym_max,
        aspect="auto",
    )
    plt.colorbar(im4, ax=ax4, label="err_dy = actual_dy − nom_dy (um)")
    ax4.set_xlabel("yi  (stage Y index)")
    ax4.set_ylabel("xi  (stage X index)")
    ax4.set_title("dy error map (image Y / stage X)")
    for i, xv in enumerate(xi_u):
        for j, yv in enumerate(yi_u):
            val = grid_dy[i, j]
            if not np.isnan(val):
                ax4.text(yv, xv, f"{val:.2f}", ha="center", va="center",
                         fontsize=4.5, color="k")

    plt.tight_layout()

    save_figure(
        fig,
        params={
            "pos_prefix": POS_PREFIX, "z_idx": Z_IDX,
            "method": "direct_ecc",
            "quiver_scale": QUIVER_SCALE,
            "source_npz": str(Path(NPZ_PATH).name),
        },
        description=(
            f"Grid true position map (Direct ECC): {POS_PREFIX} z={Z_IDX}. "
            f"4-panel: nominal vs actual, quiver, dx/dy error heatmaps. "
            f"err_dx std={err_dx.std():.4f} um, err_dy std={err_dy.std():.4f} um"
        ),
        data={
            "nominal_dx_um": nom_dx, "nominal_dy_um": nom_dy,
            "actual_dx_um": act_dx, "actual_dy_um": act_dy,
            "err_dx_um": err_dx, "err_dy_um": err_dy,
            "xi": xi, "yi": yi, "residual_um": residual, "corr": corr,
        },
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()

# %%
