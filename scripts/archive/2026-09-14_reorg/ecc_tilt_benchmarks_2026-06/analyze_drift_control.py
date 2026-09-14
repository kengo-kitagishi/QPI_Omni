"""
analyze_drift_control.py
------------------------
Load drift_log.json and run offline simulation.
Compare EMA(alpha=0.3) vs EMA(alpha=0.10) vs Kalman.

Does not modify files used during measurement (compute_drift_online.py, drift_log.json, etc.).
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure


# ======= Settings =======
DRIFT_LOG      = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
PIXEL_SCALE_UM = 0.34567514677103717
INTERVAL_SEC   = 300      # 1 frame = 5 min
SX_SIGN        = 1        # shift_sign_x (from config)
SY_SIGN        = 1        # shift_sign_y (from config)

# Kalman hyperparameters
#   estimated from true drift ~20nm/frame, ECC measurement noise ~110nm
KF_Q_POS = 400.0   # process noise (position) [nm²]  = (20nm)²
KF_Q_VEL = 4.0     # process noise (velocity) [nm²/frame²] = (2nm/frame)²
KF_R     = 12100.0  # measurement noise [nm²] = (110nm)²

EMA_ALPHAS = [1.0, 0.3, 0.10]
EMA_COLORS = {1.0: "#AAAAAA", 0.3: "#2196F3", 0.10: "#FF9800"}
EMA_LABELS = {1.0: "EMA α=1.0 (raw)", 0.3: "EMA α=0.3 (current)", 0.10: "EMA α=0.10 (proposed)"}
KALMAN_COLOR   = "#F44336"
KF_VEL_COLOR   = "#4CAF50"


# ======= Helper functions =======

def ema_filter(z: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average (causal)"""
    out = np.empty_like(z)
    out[0] = z[0]
    for t in range(1, len(z)):
        out[t] = alpha * z[t] + (1.0 - alpha) * out[t - 1]
    return out


def kf_pos_only(z_nm: np.ndarray, Q_pos: float, R: float):
    """
    Kalman filter (position-only, constant-position model)
    z_nm: measurement values [nm]
    Returns: (filtered_pos [nm], Kalman_gain)
    """
    N  = len(z_nm)
    out = np.zeros(N)
    K_h = np.zeros(N)

    x = z_nm[0]
    P = R  # initial uncertainty = measurement noise
    out[0] = x
    K_h[0] = 1.0

    for t in range(1, N):
        P_pred = P + Q_pos
        K = P_pred / (P_pred + R)
        x = x + K * (z_nm[t] - x)
        P = (1.0 - K) * P_pred
        out[t] = x
        K_h[t] = K

    return out, K_h


def kf_pos_vel(z_nm: np.ndarray, Q_pos: float, Q_vel: float, R: float):
    """
    Kalman filter ([position, velocity] model)
    z_nm: measurement values [nm]
    Returns: (pos [nm], vel [nm/frame], Kalman_gain[N,2])
    """
    N   = len(z_nm)
    pos = np.zeros(N)
    vel = np.zeros(N)
    K_h = np.zeros((N, 2))

    x = np.array([z_nm[0], 0.0])
    P = np.diag([R, Q_vel])  # initial state uncertainty

    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.diag([Q_pos, Q_vel])
    H = np.array([[1.0, 0.0]])

    pos[0] = x[0]
    vel[0] = x[1]

    for t in range(1, N):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        # Update
        S    = float(H @ P @ H.T) + R
        K    = (P @ H.T) / S
        x    = x + K.flatten() * (z_nm[t] - x[0])
        P    = (np.eye(2) - np.outer(K.flatten(), H.flatten())) @ P
        pos[t] = x[0]
        vel[t] = x[1]
        K_h[t] = K.flatten()

    return pos, vel, K_h


def compute_metrics(z_nm: np.ndarray, correction: np.ndarray) -> dict:
    """
    Return quality metrics for the filter.
    correction[t] = correction command issued at frame t [nm]
    -> residual at frame t+1 = z[t+1] - correction[t]
    """
    cmd_diff  = np.diff(correction)
    residual  = z_nm[1:] - correction[:-1]   # residual at next frame

    r1_cmd = pearsonr(correction[:-1], correction[1:])[0] if len(correction) > 2 else 0.0
    r1_res = pearsonr(residual[:-1],   residual[1:]  )[0] if len(residual)   > 2 else 0.0

    return {
        "cmd_diff_std_nm": float(np.std(cmd_diff)),
        "residual_std_nm": float(np.std(residual)),
        "cmd_lag1":        float(r1_cmd),
        "residual_lag1":   float(r1_res),
    }


# ======= Data loading =======
records = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))
N       = len(records)

tp      = np.array([r["timepoint"]              for r in records])
time_h  = tp * INTERVAL_SEC / 3600.0

tx_raw  = np.array([r["tx_avg_px"]               for r in records])  # image X -> stage Y
ty_raw  = np.array([r["ty_avg_px"]               for r in records])  # image Y -> stage X
corr_sx = np.array([r["correction_stage_x_um"]   for r in records])  # actually applied stage X correction
corr_sy = np.array([r["correction_stage_y_um"]   for r in records])  # actually applied stage Y correction

# Closed-loop residual [nm] (ECC raw value = position error at each frame)
z_sy_nm = tx_raw * PIXEL_SCALE_UM * SY_SIGN * 1000.0   # stage Y direction
z_sx_nm = ty_raw * PIXEL_SCALE_UM * SX_SIGN * 1000.0   # stage X direction

print(f"Loaded {N} frames  (TP {int(tp[0])} - {int(tp[-1])})")
print(f"z_sy  mean={np.mean(z_sy_nm):.1f} nm  std={np.std(z_sy_nm):.1f} nm")
print(f"z_sx  mean={np.mean(z_sx_nm):.1f} nm  std={np.std(z_sx_nm):.1f} nm")


# ======= Open-loop reconstruction =======
# open_loop[t] = (cumulative corrections applied before frame t) + (ECC residual at frame t)
#              = true absolute position of sample [nm]
cum_sx_before = np.concatenate([[0.0], np.cumsum(corr_sx[:-1])]) * 1000.0
cum_sy_before = np.concatenate([[0.0], np.cumsum(corr_sy[:-1])]) * 1000.0

ol_sy_nm = cum_sy_before + z_sy_nm   # stage Y open-loop [nm]
ol_sx_nm = cum_sx_before + z_sx_nm   # stage X open-loop [nm]

# Linear trend
t_arr      = np.arange(N, dtype=float)
slope_y, intercept_y = np.polyfit(t_arr, ol_sy_nm, 1)
slope_x, intercept_x = np.polyfit(t_arr, ol_sx_nm, 1)
print(f"\nOpen-loop trend  Y={slope_y:.2f} nm/frame  X={slope_x:.2f} nm/frame")


# ======= Filter computation =======
# -- EMA (applied to closed-loop residual) --
ema_sy = {a: ema_filter(z_sy_nm, a) for a in EMA_ALPHAS}

# -- Kalman (closed-loop residual, position-only) --
kf_cl_sy, kf_cl_K = kf_pos_only(z_sy_nm, KF_Q_POS, KF_R)

# -- Kalman (open-loop, position+velocity) --
kf_ol_sy_pos, kf_ol_sy_vel, _ = kf_pos_vel(ol_sy_nm, KF_Q_POS, KF_Q_VEL, KF_R)
kf_ol_sx_pos, kf_ol_sx_vel, _ = kf_pos_vel(ol_sx_nm, KF_Q_POS, KF_Q_VEL, KF_R)


# ======= Evaluation metrics =======
print("\n=== Method comparison (Stage Y = image X, primary drift axis) ===")
print(f"{'Method':22s} | {'cmd diff std [nm]':17s} | {'resid std [nm]':13s} | {'cmd lag-1':9s} | {'resid lag-1':9s}")
print("-" * 80)

methods_order = []
methods_metrics = {}

for a in EMA_ALPHAS:
    m  = compute_metrics(z_sy_nm, ema_sy[a])
    lbl = f"EMA α={a:.2f}"
    methods_order.append(lbl)
    methods_metrics[lbl] = m
    print(f"{lbl:22s} | {m['cmd_diff_std_nm']:17.1f} | {m['residual_std_nm']:13.1f} | {m['cmd_lag1']:9.3f} | {m['residual_lag1']:9.3f}")

m_kf = compute_metrics(z_sy_nm, kf_cl_sy)
lbl  = "Kalman (pos-only)"
methods_order.append(lbl)
methods_metrics[lbl] = m_kf
print(f"{lbl:22s} | {m_kf['cmd_diff_std_nm']:17.1f} | {m_kf['residual_std_nm']:13.1f} | {m_kf['cmd_lag1']:9.3f} | {m_kf['residual_lag1']:9.3f}")

# Kalman with feedforward (open-loop: correction = pos + vel)
kf_ff_cmd = kf_ol_sy_pos + kf_ol_sy_vel   # predicted position for next frame
m_kf_ff   = compute_metrics(ol_sy_nm, kf_ff_cmd)
lbl_ff    = "Kalman (pos+vel FF)"
methods_order.append(lbl_ff)
methods_metrics[lbl_ff] = m_kf_ff
print(f"{lbl_ff:22s} | {m_kf_ff['cmd_diff_std_nm']:17.1f} | {m_kf_ff['residual_std_nm']:13.1f} | {m_kf_ff['cmd_lag1']:9.3f} | {m_kf_ff['residual_lag1']:9.3f}")

kf_ol_converge_vel = float(np.mean(kf_ol_sy_vel[max(1, N // 2):]))
print(f"\nKF open-loop velocity (second-half mean): {kf_ol_converge_vel:.2f} nm/frame")
print(f"Linear fit slope:                        {slope_y:.2f} nm/frame")


# ======= FIGURE =======
fig = plt.figure(figsize=(13, 11))
gs  = gridspec.GridSpec(3, 3, hspace=0.55, wspace=0.42,
                        left=0.08, right=0.97, top=0.95, bottom=0.06)

ax1   = fig.add_subplot(gs[0, :])     # Panel 1 (wide): open-loop + Kalman
ax2   = fig.add_subplot(gs[1, :])     # Panel 2 (wide): closed-loop filter comparison
ax3l  = fig.add_subplot(gs[2, 0])     # Panel 3L: residual std bar
ax3m  = fig.add_subplot(gs[2, 1])     # Panel 3M: cmd diff std bar
ax3r  = fig.add_subplot(gs[2, 2])     # Panel 3R: lag-1 autocorr bar


# ---- Panel 1: Open-loop + Kalman ----
ax1.set_title("Open-loop drift (reconstructed) + Kalman estimation  [Stage Y]", fontsize=10, fontweight="bold")
ax1.plot(time_h, ol_sy_nm / 1000, color="#BBBBBB", lw=0.8, alpha=0.8, label="Open-loop (raw)")
ax1.plot(time_h, kf_ol_sy_pos / 1000, color=KALMAN_COLOR, lw=1.8, label="Kalman pos estimate")

# linear trend
trend_y = slope_y * t_arr + intercept_y
ax1.plot(time_h, trend_y / 1000, "--", color="orange", lw=1.0,
         label=f"Linear trend ({slope_y:.1f} nm/frame)")

# velocity on twin axis
ax1b = ax1.twinx()
ax1b.plot(time_h, kf_ol_sy_vel, color=KF_VEL_COLOR, lw=1.0, alpha=0.7, ls=":", label="KF velocity")
ax1b.axhline(slope_y, color=KF_VEL_COLOR, lw=0.6, ls="--", alpha=0.5)
ax1b.set_ylabel("Velocity estimate (nm/frame)", color=KF_VEL_COLOR, fontsize=8)
ax1b.tick_params(axis="y", labelcolor=KF_VEL_COLOR, labelsize=7)

ax1.set_xlabel("Time (h)")
ax1.set_ylabel("Stage Y position (μm)")
lines1, lbs1 = ax1.get_legend_handles_labels()
lines2, lbs2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, lbs1 + lbs2, frameon=False, fontsize=7, ncol=2)
ax1.spines["top"].set_visible(False)


# ---- Panel 2: Closed-loop correction command comparison ----
ax2.set_title("Closed-loop correction command  [Stage Y residual: EMA vs Kalman]", fontsize=10, fontweight="bold")
for a in EMA_ALPHAS:
    lw = 1.4 if a < 1.0 else 0.5
    ax2.plot(time_h, ema_sy[a] / 1000, color=EMA_COLORS[a], lw=lw, alpha=0.85, label=EMA_LABELS[a])
ax2.plot(time_h, kf_cl_sy / 1000, color=KALMAN_COLOR, lw=1.8, label="Kalman (pos-only)", zorder=5)
ax2.axhline(0, color="gray", lw=0.5, ls="--")
ax2.set_xlabel("Time (h)")
ax2.set_ylabel("Correction command (μm)")
ax2.legend(frameon=False, fontsize=8, ncol=2)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)


# ---- Panels 3: Bar charts ----
bar_labels  = ["EMA\nα=1.0", "EMA\nα=0.3\n(current)", "EMA\nα=0.10\n(proposed)", "Kalman\n(pos)"]
bar_colors  = [EMA_COLORS[1.0], EMA_COLORS[0.3], EMA_COLORS[0.10], KALMAN_COLOR]

res_stds  = [methods_metrics[f"EMA α={a:.2f}"]["residual_std_nm"]   for a in EMA_ALPHAS] + [m_kf["residual_std_nm"]]
cmd_stds  = [methods_metrics[f"EMA α={a:.2f}"]["cmd_diff_std_nm"]   for a in EMA_ALPHAS] + [m_kf["cmd_diff_std_nm"]]
lag1_vals = [methods_metrics[f"EMA α={a:.2f}"]["cmd_lag1"]           for a in EMA_ALPHAS] + [m_kf["cmd_lag1"]]

for ax_b, vals, ylabel, title in [
    (ax3l, res_stds,  "Residual std (nm)",     "Next-frame residual std"),
    (ax3m, cmd_stds,  "Cmd diff std (nm)",      "Correction command jitter"),
    (ax3r, lag1_vals, "Lag-1 autocorrelation",  "Command lag-1 autocorr"),
]:
    bars = ax_b.bar(bar_labels, vals, color=bar_colors, width=0.6, alpha=0.85)
    if ylabel == "Lag-1 autocorrelation":
        ax_b.axhline(0, color="gray", lw=0.8)
        y_fmt = ".3f"
    else:
        y_fmt = ".0f"
    ax_b.set_title(title, fontsize=9)
    ax_b.set_ylabel(ylabel, fontsize=8)
    ax_b.tick_params(axis="x", labelsize=7)
    ax_b.tick_params(axis="y", labelsize=7)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)

    offset = max(abs(v) for v in vals) * 0.03
    for bar, val in zip(bars, vals):
        y_pos = val + offset if val >= 0 else val - offset * 4
        ax_b.text(bar.get_x() + bar.get_width() / 2, y_pos,
                  format(val, y_fmt), ha="center", va="bottom", fontsize=7)


# ======= Save =======
save_figure(
    fig,
    params={
        "ema_alphas":         EMA_ALPHAS,
        "kf_Q_pos_nm2":       KF_Q_POS,
        "kf_Q_vel_nm2":       KF_Q_VEL,
        "kf_R_nm2":           KF_R,
        "n_frames":           int(N),
        "slope_y_nm_frame":   float(slope_y),
        "slope_x_nm_frame":   float(slope_x),
        "kf_ol_vel_converged": float(kf_ol_converge_vel),
    },
    description=(
        "Drift control method comparison (offline simulation): "
        "EMA(alpha=0.3/0.1) vs Kalman. Open-loop reconstruction, velocity estimation, "
        "residual std, command jitter, and lag-1 autocorrelation comparison."
    ),
    data={
        "time_h":         time_h,
        "z_sy_nm":        z_sy_nm,
        "z_sx_nm":        z_sx_nm,
        "ol_sy_nm":       ol_sy_nm,
        "ol_sx_nm":       ol_sx_nm,
        "kf_ol_sy_pos":   kf_ol_sy_pos,
        "kf_ol_sy_vel":   kf_ol_sy_vel,
        "kf_cl_sy":       kf_cl_sy,
        "ema_03_sy":      ema_sy[0.3],
        "ema_010_sy":     ema_sy[0.10],
    },
)
plt.close(fig)
print("\n[Done] figure saved.")
