"""analyze_kalman_gain.py -- Check if current K=0.80 is optimal.

Actual correction pipeline (use_kalman_filter=True):
  1. ECC measures residual shift (tx_avg_px)
  2. Open-loop position: ol_pos = cumulative + measurement
  3. Kalman filter: pos_new = pos_old + K * (ol_pos - pos_old)
  4. correction = pos_new - cumulative  (EMA is NOT used)
  5. cumulative += correction  => cumulative = pos_new

Analyses:
  1. Innovation autocorrelation (whiteness test)
  2. Closed-loop K-sweep with noise separation (Monte Carlo)
  3. Theoretical optimal K from estimated Q, R
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

# -- Load data -------------------------------------------------------
DRIFT_LOG = Path(r"C:\Users\QPI\Documents\QPI_Omni\drift_session\drift_log.json")
data = json.loads(DRIFT_LOG.read_text(encoding="utf-8"))
pixel_scale = 0.34567514677103717  # um/px
ps_nm = pixel_scale * 1000  # nm/px

tx_raw = np.array([r["tx_avg_px"] for r in data])
ty_raw = np.array([r["ty_avg_px"] for r in data])
innov_x = np.array([r.get("kf_innovation_tx_nm", np.nan) for r in data])
innov_y = np.array([r.get("kf_innovation_ty_nm", np.nan) for r in data])
cum_x = np.array([r["cumulative_dx_um"] for r in data])
cum_y = np.array([r["cumulative_dy_um"] for r in data])
time_h = np.arange(len(data)) * 300 / 3600

# -- Innovation statistics (skip transient) --------------------------
SKIP = 20
ix = innov_x[SKIP:]
iy = innov_y[SKIP:]
N = len(ix)

print(f"=== Innovation Statistics (skip first {SKIP}) ===")
print(f"N = {N}")
print(f"Innovation X: mean={np.mean(ix):.1f} nm, std={np.std(ix):.1f} nm")
print(f"Innovation Y: mean={np.mean(iy):.1f} nm, std={np.std(iy):.1f} nm")


def autocorr(x, maxlag=50):
    x = x - np.mean(x)
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1:]
    acf = acf / acf[0]
    return acf[:maxlag + 1]


acf_x = autocorr(ix, 50)
acf_y = autocorr(iy, 50)
ci_95 = 1.96 / np.sqrt(N)

print(f"\n=== Innovation Autocorrelation ===")
print(f"95%% CI: +/- {ci_95:.4f}")
sig_x = "SIGNIFICANT (K too low)" if acf_x[1] > ci_95 else "OK"
sig_y = "SIGNIFICANT (K too low)" if acf_y[1] > ci_95 else "OK"
print(f"ACF_x(1) = {acf_x[1]:.4f}  -> {sig_x}")
print(f"ACF_y(1) = {acf_y[1]:.4f}  -> {sig_y}")


def optimal_K(Q, R):
    """Steady-state Kalman gain for random-walk model."""
    if Q <= 0 or R <= 0:
        return np.nan
    return (-Q + np.sqrt(Q**2 + 4 * Q * R)) / (2 * R)


# -- Reconstruct observed position -----------------------------------
# obs_pos[t] = cumulative[t-1] + raw_measurement[t] * pixel_scale
# = true_drift[t] + ECC_noise[t]  (inseparable from single run)
obs_x_um = np.zeros(len(tx_raw))
obs_y_um = np.zeros(len(ty_raw))
obs_x_um[0] = tx_raw[0] * pixel_scale
obs_y_um[0] = ty_raw[0] * pixel_scale
for i in range(1, len(tx_raw)):
    obs_x_um[i] = cum_x[i - 1] + tx_raw[i] * pixel_scale
    obs_y_um[i] = cum_y[i - 1] + ty_raw[i] * pixel_scale

# Separate smooth drift (low-pass) and noise (high-pass)
SG_WIN = 61  # ~5 hours
SG_ORD = 3
smooth_x = savgol_filter(obs_x_um, SG_WIN, SG_ORD)
smooth_y = savgol_filter(obs_y_um, SG_WIN, SG_ORD)
noise_x = obs_x_um - smooth_x  # um
noise_y = obs_y_um - smooth_y

noise_std_x_nm = np.std(noise_x[SKIP:]) * 1000
noise_std_y_nm = np.std(noise_y[SKIP:]) * 1000
print(f"\n=== Noise Estimation (SG filter, w={SG_WIN}) ===")
print(f"ECC noise std X: {noise_std_x_nm:.1f} nm")
print(f"ECC noise std Y: {noise_std_y_nm:.1f} nm")

# Process noise: variance of smooth drift increments
drift_step_x = np.diff(smooth_x) * 1000  # nm
drift_step_y = np.diff(smooth_y) * 1000
Q_x_nm2 = np.var(drift_step_x[SKIP:])
Q_y_nm2 = np.var(drift_step_y[SKIP:])
R_x_nm2 = noise_std_x_nm**2
R_y_nm2 = noise_std_y_nm**2

K_opt_x = optimal_K(Q_x_nm2, R_x_nm2)
K_opt_y = optimal_K(Q_y_nm2, R_y_nm2)

print(f"\n=== Estimated Q, R ===")
print(f"X: Q={Q_x_nm2:.0f} nm2 ({np.sqrt(Q_x_nm2):.1f} nm/step), R={R_x_nm2:.0f} nm2 ({noise_std_x_nm:.1f} nm)")
print(f"Y: Q={Q_y_nm2:.0f} nm2 ({np.sqrt(Q_y_nm2):.1f} nm/step), R={R_y_nm2:.0f} nm2 ({noise_std_y_nm:.1f} nm)")
print(f"Q/R ratio: X={Q_x_nm2 / R_x_nm2:.2f}, Y={Q_y_nm2 / R_y_nm2:.2f}")
print(f"Optimal K: X={K_opt_x:.3f}, Y={K_opt_y:.3f}")
print(f"Current K: 0.800")
print(f"Config:    Q_tx=877, R_tx=274, Q_ty=291, R_ty=91")


# -- Closed-loop K-sweep (no EMA, matching actual pipeline) ----------
# Simulate: kf_step_posonly logic with constant gain K.
#   ol_pos_nm = cumulative_nm + measurement_nm  (= true_drift + noise)
#   kf_pos = kf_pos_prev + K * (ol_pos - kf_pos_prev)
#   cumulative = kf_pos  (correction = kf_pos - cumulative_prev)
#   next residual = true_drift[t+1] + noise[t+1] - kf_pos[t]

K_sweep = np.arange(0.3, 1.005, 0.01)
rms_x_arr = np.zeros(len(K_sweep))
rms_y_arr = np.zeros(len(K_sweep))

N_MC = 20
rng = np.random.default_rng(42)

for mc in range(N_MC):
    # Shuffle noise to break any temporal structure
    noise_x_mc = rng.permutation(noise_x)
    noise_y_mc = rng.permutation(noise_y)

    for ki, K_test in enumerate(K_sweep):
        kf_pos_x = 0.0  # nm
        kf_pos_y = 0.0
        resid_x = []
        resid_y = []

        for i in range(len(smooth_x)):
            obs_x_nm = smooth_x[i] * 1000 + noise_x_mc[i] * 1000
            obs_y_nm = smooth_y[i] * 1000 + noise_y_mc[i] * 1000

            # Residual = obs - kf_pos_prev (what ECC sees)
            resid_x.append(obs_x_nm - kf_pos_x)
            resid_y.append(obs_y_nm - kf_pos_y)

            # Kalman update: kf_pos = kf_pos + K * (obs - kf_pos)
            kf_pos_x = kf_pos_x + K_test * (obs_x_nm - kf_pos_x)
            kf_pos_y = kf_pos_y + K_test * (obs_y_nm - kf_pos_y)

        rms_x_arr[ki] += np.std(resid_x[SKIP:])
        rms_y_arr[ki] += np.std(resid_y[SKIP:])

rms_x_arr /= N_MC
rms_y_arr /= N_MC

best_K_x = K_sweep[np.argmin(rms_x_arr)]
best_K_y = K_sweep[np.argmin(rms_y_arr)]

print(f"\n=== K-sweep (Kalman only, no EMA, MC={N_MC}) ===")
for ki, K in enumerate(K_sweep):
    show = (abs(K - 0.80) < 0.005 or abs(K - 0.90) < 0.005 or
            abs(K - 0.95) < 0.005 or abs(K - 1.00) < 0.005 or
            abs(K - best_K_x) < 0.005 or abs(K - best_K_y) < 0.005 or
            K == K_sweep[0] or K == K_sweep[-1])
    if show:
        tag = ""
        if abs(K - 0.80) < 0.005:
            tag = " <-- current"
        if abs(K - best_K_x) < 0.005:
            tag += " <-- best X"
        if abs(K - best_K_y) < 0.005:
            tag += " <-- best Y"
        print(f"  K={K:.2f}: X={rms_x_arr[ki]:.1f} nm, Y={rms_y_arr[ki]:.1f} nm{tag}")

actual_rms_x = np.std(tx_raw[SKIP:]) * ps_nm
actual_rms_y = np.std(ty_raw[SKIP:]) * ps_nm
print(f"\nActual observed residual: X={actual_rms_x:.1f} nm, Y={actual_rms_y:.1f} nm")

idx_08 = np.argmin(np.abs(K_sweep - 0.80))
imp_x = rms_x_arr[idx_08] - rms_x_arr[np.argmin(rms_x_arr)]
imp_y = rms_y_arr[idx_08] - rms_y_arr[np.argmin(rms_y_arr)]
pct_x = imp_x / rms_x_arr[idx_08] * 100
pct_y = imp_y / rms_y_arr[idx_08] * 100
print(f"\nImprovement K=0.80 -> best:")
print(f"  X: {imp_x:.1f} nm ({pct_x:.1f}%%)")
print(f"  Y: {imp_y:.1f} nm ({pct_y:.1f}%%)")


# -- Plot ------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Innovation ACF
lags = np.arange(26)
ax = axes[0, 0]
w = 0.35
ax.bar(lags[1:] - w / 2, acf_x[1:26], width=w, alpha=0.7, label="X", color="tab:blue")
ax.bar(lags[1:] + w / 2, acf_y[1:26], width=w, alpha=0.7, label="Y", color="tab:red")
ax.axhline(ci_95, color="gray", ls="--", lw=1, label=f"95%% CI ({ci_95:.3f})")
ax.axhline(-ci_95, color="gray", ls="--", lw=1)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("Lag")
ax.set_ylabel("Autocorrelation")
ax.set_title(f"Innovation ACF  (K=0.80, N={N})\nlag1: X={acf_x[1]:.3f}, Y={acf_y[1]:.3f}")
ax.legend(fontsize=9)

# Panel 2: Noise separation
ax = axes[0, 1]
ax.plot(time_h, obs_x_um, "b-", alpha=0.3, lw=0.5, label="Observed X")
ax.plot(time_h, smooth_x, "b-", lw=2, label="Smooth drift X")
ax.plot(time_h, obs_y_um, "r-", alpha=0.3, lw=0.5, label="Observed Y")
ax.plot(time_h, smooth_y, "r-", lw=2, label="Smooth drift Y")
ax.set_xlabel("Time (h)")
ax.set_ylabel("Position (um)")
ax.set_title(f"Drift signal separation\nNoise std: X={noise_std_x_nm:.1f} nm, Y={noise_std_y_nm:.1f} nm")
ax.legend(fontsize=8)

# Panel 3: K-sweep (Kalman only, no EMA)
ax = axes[1, 0]
ax.plot(K_sweep, rms_x_arr, "b-", lw=2, label="X residual")
ax.plot(K_sweep, rms_y_arr, "r-", lw=2, label="Y residual")
ax.axvline(0.80, color="gray", ls=":", lw=2, label="Current K=0.80")
ax.axvline(best_K_x, color="b", ls="--", lw=1.5, label=f"Best X: K={best_K_x:.2f}")
ax.axvline(best_K_y, color="r", ls="--", lw=1.5, label=f"Best Y: K={best_K_y:.2f}")
if not np.isnan(K_opt_x):
    ax.axvline(K_opt_x, color="b", ls=":", lw=1, alpha=0.5, label=f"Theory X: K={K_opt_x:.2f}")
if not np.isnan(K_opt_y):
    ax.axvline(K_opt_y, color="r", ls=":", lw=1, alpha=0.5, label=f"Theory Y: K={K_opt_y:.2f}")
ax.scatter([0.80], [rms_x_arr[idx_08]], color="b", s=80, zorder=5)
ax.scatter([0.80], [rms_y_arr[idx_08]], color="r", s=80, zorder=5)
ax.set_xlabel("Kalman Gain K")
ax.set_ylabel("Residual RMS (nm)")
ax.set_title(f"K-sweep (Kalman only, no EMA, MC={N_MC})\n"
             f"Improvement: X={imp_x:.1f}nm ({pct_x:.0f}%%), Y={imp_y:.1f}nm ({pct_y:.0f}%%)")
ax.legend(fontsize=7, loc="upper left")

# Panel 4: Residual distribution at K=0.80 vs best K
ax = axes[1, 1]
for K_test, color, label in [(0.80, "gray", "K=0.80"),
                               (best_K_x, "tab:blue", f"K={best_K_x:.2f} (best X)")]:
    kf_pos = 0.0
    resid = []
    for i in range(len(smooth_x)):
        obs_nm = smooth_x[i] * 1000 + noise_x[i] * 1000
        resid.append(obs_nm - kf_pos)
        kf_pos = kf_pos + K_test * (obs_nm - kf_pos)
    resid = np.array(resid[SKIP:])
    ax.hist(resid, bins=80, alpha=0.5, color=color, density=True,
            label=f"{label}  std={np.std(resid):.1f}nm")

ax.set_xlabel("Residual (nm)")
ax.set_ylabel("Density")
ax.set_title("Residual distribution (X axis)")
ax.legend(fontsize=9)

fig.suptitle("Kalman Gain Optimality (no EMA in correction path)",
             fontsize=14, fontweight="bold")
fig.tight_layout()

save_figure(
    fig,
    params={
        "K_current": 0.80,
        "K_opt_x_theory": round(float(K_opt_x), 3) if not np.isnan(K_opt_x) else None,
        "K_opt_y_theory": round(float(K_opt_y), 3) if not np.isnan(K_opt_y) else None,
        "K_best_x_sim": round(float(best_K_x), 2),
        "K_best_y_sim": round(float(best_K_y), 2),
        "acf1_x": round(float(acf_x[1]), 4),
        "acf1_y": round(float(acf_y[1]), 4),
        "Q_x_nm2": round(float(Q_x_nm2)),
        "R_x_nm2": round(float(R_x_nm2)),
        "Q_y_nm2": round(float(Q_y_nm2)),
        "R_y_nm2": round(float(R_y_nm2)),
        "improvement_x_nm": round(float(imp_x), 1),
        "improvement_y_nm": round(float(imp_y), 1),
        "N_MC": N_MC,
    },
    description="Kalman gain optimality (no EMA): innovation ACF, noise separation, K-sweep MC",
    data={
        "acf_x": acf_x,
        "acf_y": acf_y,
        "K_sweep": K_sweep,
        "rms_x_arr": rms_x_arr,
        "rms_y_arr": rms_y_arr,
        "smooth_x_um": smooth_x,
        "smooth_y_um": smooth_y,
        "noise_x_um": noise_x,
        "noise_y_um": noise_y,
    },
)
plt.close()
print("\nDone.")
