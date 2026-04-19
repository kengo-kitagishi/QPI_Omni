# %%
"""
plot_grid_calibration.py
------------------------
Read grid_calibration_PosN.json output by calibrate_grid_positions.py and
visualize the true grid positions in um.

[4 panels]
  Panel 1: Nominal vs Actual scatter (stage Y / stage X, um)
  Panel 2: Error quiver -- displacement vectors from nominal to actual position (um)
  Panel 3: dx error heatmap (stage Y/X um axes, um colorbar)
  Panel 4: ECC correlation heatmap (stage Y/X um axes)
"""
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# ============================================================
# Settings
# ============================================================
CALIB_JSON = (
    r"E:\Acuisition\kitagishi\260331\grid_2pergluc_60ms_1"
    r"\grid_calibration_Pos1.json"
)
POS_LABEL  = "Pos1"

# Quiver arrow scale (1=actual scale; larger values make arrows longer)
QUIVER_SCALE = 1
# ============================================================


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_grid(xi_arr, yi_arr, values, xi_u, yi_u):
    """1D arrays -> 2D grid  [xi_idx, yi_idx]  (origin lower = increasing xi = up)"""
    xi_map = {v: i for i, v in enumerate(xi_u)}
    yi_map = {v: i for i, v in enumerate(yi_u)}
    g = np.full((len(xi_u), len(yi_u)), np.nan)
    for xi, yi, v in zip(xi_arr, yi_arr, values):
        g[xi_map[xi], yi_map[yi]] = v
    return g


def main():
    d = load_json(CALIB_JSON)
    psc      = d["pixel_scale_um"]     # μm/px
    x_step   = d["x_step_um"]          # 0.1 μm
    y_step   = d["y_step_um"]          # 0.1 μm
    positions = d["positions"]

    xi_arr   = np.array([p["xi"]           for p in positions])
    yi_arr   = np.array([p["yi"]           for p in positions])
    corr     = np.array([p["mean_correlation"] if p["mean_correlation"] is not None else np.nan
                         for p in positions])

    # Nominal positions (um): stage X = xi*step, stage Y = yi*step
    nom_stageX = xi_arr * x_step
    nom_stageY = yi_arr * y_step

    # Actual positions (um): ECC actual_dx/dy are in image X/Y direction.
    # image X (dx) <-> stage Y, image Y (dy) <-> stage X (90-degree mapping) +
    # Actual sign relationship: stage+X -> content shifts in -image_Y (= SHIFT_SIGN=-1 is correct)
    # Therefore: act_stageX = -actual_dy_px * psc, act_stageY = -actual_dx_px * psc
    act_stageX = -np.array([p["actual_dy_px"] for p in positions]) * psc
    act_stageY = -np.array([p["actual_dx_px"] for p in positions]) * psc

    # Positioning error (um) in stage space
    err_stageX = act_stageX - nom_stageX
    err_stageY = act_stageY - nom_stageY

    # Error magnitude for all points
    res = np.sqrt(err_stageX**2 + err_stageY**2)

    # 2D grid
    xi_u  = np.sort(np.unique(xi_arr))
    yi_u  = np.sort(np.unique(yi_arr))
    g_errX   = to_grid(xi_arr, yi_arr, err_stageX, xi_u, yi_u)
    g_errY   = to_grid(xi_arr, yi_arr, err_stageY, xi_u, yi_u)
    g_corr   = to_grid(xi_arr, yi_arr, corr,       xi_u, yi_u)

    # imshow extent (um)
    yi_min_um  = yi_u[0]  * y_step
    yi_max_um  = yi_u[-1] * y_step
    xi_min_um  = xi_u[0]  * x_step
    xi_max_um  = xi_u[-1] * x_step
    half_y = y_step / 2
    half_x = x_step / 2
    extent = [yi_min_um - half_y, yi_max_um + half_y,
              xi_min_um - half_x, xi_max_um + half_x]

    sym_err = max(np.nanmax(np.abs(g_errX)), np.nanmax(np.abs(g_errY)), 0.01)

    # ----------------------------------------------------------------
    print(f"N={len(xi_arr)}  pixel_scale={psc:.4f} μm/px")
    print(f"err_stageX [μm]: mean={err_stageX.mean():+.4f}  std={err_stageX.std():.4f}  max_abs={np.abs(err_stageX).max():.4f}")
    print(f"err_stageY [μm]: mean={err_stageY.mean():+.4f}  std={err_stageY.std():.4f}  max_abs={np.abs(err_stageY).max():.4f}")
    print(f"residual [μm]: mean={res.mean():.4f}  max={res.max():.4f}")
    print(f"ECC corr: mean={np.nanmean(corr):.4f}  min={np.nanmin(corr):.4f}")
    # ----------------------------------------------------------------

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Grid calibration — {POS_LABEL}  |  pixel_scale={psc:.4f} μm/px\n"
        f"err_stageX: mean={err_stageX.mean():+.4f} μm  std={err_stageX.std():.4f} μm  |  "
        f"err_stageY: mean={err_stageY.mean():+.4f} μm  std={err_stageY.std():.4f} μm",
        fontsize=10,
    )

    # Extent in stage space: X-axis=stage X (xi), Y-axis=stage Y (yi)
    extent_st = [xi_min_um - half_x, xi_max_um + half_x,
                 yi_min_um - half_y, yi_max_um + half_y]

    # ── Panel 1: Nominal vs Actual scatter ──────────────────────────
    ax = axes[0]
    ax.scatter(nom_stageX, nom_stageY, s=50, facecolors="none",
               edgecolors="#999", linewidths=1.0, zorder=2, label="Nominal")
    sc = ax.scatter(act_stageX, act_stageY, s=20, c=res, cmap="hot_r",
                    vmin=0, vmax=np.percentile(res, 95), zorder=3, label="Actual")
    plt.colorbar(sc, ax=ax, label="Residual (μm)")
    ax.set_xlabel("Stage X (μm)")
    ax.set_ylabel("Stage Y (μm)")
    ax.set_title("Nominal (○) vs Actual (●)")
    ax.set_aspect("equal")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, linewidth=0.3, alpha=0.4)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Panel 2: Error quiver ────────────────────────────────────────
    ax2 = axes[1]
    quiv = ax2.quiver(
        nom_stageX, nom_stageY,
        err_stageX * QUIVER_SCALE,
        err_stageY * QUIVER_SCALE,
        res,
        cmap="hot_r", clim=(0, np.percentile(res, 95)),
        angles="xy", scale_units="xy", scale=1,
        width=0.003,
    )
    plt.colorbar(quiv, ax=ax2, label="Residual (μm)")
    ax2.set_xlabel("Nominal stage X (μm)")
    ax2.set_ylabel("Nominal stage Y (μm)")
    ax2.set_title(f"Error vectors  ({'actual scale' if QUIVER_SCALE == 1 else f'×{QUIVER_SCALE}'})")
    ax2.set_aspect("equal")
    ax2.grid(True, linewidth=0.3, alpha=0.4)
    ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Panel 3: stage Y error heatmap ──────────────────────────────
    # g_errY[xi_idx, yi_idx] → .T → [yi_idx, xi_idx]
    # origin="lower": row=yi_idx -> Y-axis=stage Y, col=xi_idx -> X-axis=stage X
    ax3 = axes[2]
    im3 = ax3.imshow(
        g_errY.T, origin="lower", extent=extent_st,
        cmap="RdBu_r", vmin=-sym_err, vmax=sym_err,
        aspect="equal",
    )
    plt.colorbar(im3, ax=ax3, label="Δ stage Y  (μm)")
    ax3.set_xlabel("Stage X (μm)")
    ax3.set_ylabel("Stage Y (μm)")
    ax3.set_title("Stage Y positioning error")
    for i, xv in enumerate(xi_u):
        for j, yv in enumerate(yi_u):
            val = g_errY[i, j]
            if not np.isnan(val):
                ax3.text(xv * x_step, yv * y_step, f"{val:.2f}",
                         ha="center", va="center", fontsize=4.5, color="k")
    ax3.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax3.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax3.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    # ── Panel 4: ECC correlation heatmap ────────────────────────────
    ax4 = axes[3]
    im4 = ax4.imshow(
        g_corr.T, origin="lower", extent=extent_st,
        cmap="viridis", vmin=0.9, vmax=1.0,
        aspect="equal",
    )
    plt.colorbar(im4, ax=ax4, label="ECC correlation")
    ax4.set_xlabel("Stage X (μm)")
    ax4.set_ylabel("Stage Y (μm)")
    ax4.set_title("ECC correlation")
    for i, xv in enumerate(xi_u):
        for j, yv in enumerate(yi_u):
            val = g_corr[i, j]
            if not np.isnan(val):
                ax4.text(xv * x_step, yv * y_step, f"{val:.3f}",
                         ha="center", va="center", fontsize=4.0, color="w")
    ax4.xaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax4.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
    ax4.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    plt.tight_layout()

    save_figure(
        fig,
        params={
            "pos_label": POS_LABEL,
            "calib_json": str(Path(CALIB_JSON).name),
            "pixel_scale_um": psc,
            "quiver_scale": QUIVER_SCALE,
        },
        description=(
            f"Grid calibration {POS_LABEL}: 4-panel (scatter, quiver, stageY-error, ECC-corr). "
            f"All axes in μm (sign-corrected). "
            f"err_stageX mean={err_stageX.mean():+.4f} μm std={err_stageX.std():.4f} μm. "
            f"err_stageY mean={err_stageY.mean():+.4f} μm std={err_stageY.std():.4f} μm."
        ),
        data={
            "xi": xi_arr, "yi": yi_arr,
            "nom_stageX_um": nom_stageX, "nom_stageY_um": nom_stageY,
            "act_stageX_um": act_stageX, "act_stageY_um": act_stageY,
            "err_stageX_um": err_stageX, "err_stageY_um": err_stageY,
            "residual_um": res, "corr": corr,
        },
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()

# %%
