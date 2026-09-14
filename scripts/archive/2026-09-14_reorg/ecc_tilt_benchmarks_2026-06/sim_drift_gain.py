"""sim_drift_gain.py -- closed-loop stage-drift sim: correction gain vs residual

Settles the question "can we drop the Kalman and run a fixed gain of 1.0?" by
simulating the actual compute_drift_online control law with the measured noise
and overshoot, and comparing residual tracking error vs gain.

Model (per axis), matching compute_drift_online:
  D_t  true drift          : random walk, D_t = D_{t-1} + N(0, Q)
  p_t  residual error      : D_t - C_{t-1}      (C = applied stage position)
  z_t  ECC measurement     : p_t + N(0, R)
  f_t  EMA-filtered command : alpha*z_t + (1-alpha)*f_{t-1}   (alpha = "gain")
  C_t  applied (overshoot)  : C_{t-1} + f_t*(1+beta)
Residual RMS = std(p_t). Deadbeat gain = 1/(1+beta).

Measured params (memory project_kalman_noise_params, 2026-04-03):
  Q_x = 8818 nm^2 (93.9 nm/step), Q_y = 2480 nm^2 (49.8 nm/step), beta ~ 0.24
  R(uint8 ECC) ~ 274 nm^2 (16.5 nm);  R(float ECC) ~ 25 nm^2 (5 nm, this bench)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from figure_logger import save_figure

BETA = 0.24                      # stage overshoot
Q = {"X": 8818.0, "Y": 2480.0}   # nm^2 per step (process / real drift)
R = {"uint8": 274.0, "float": 25.0}  # nm^2 measurement noise
N = 4000                         # frames
GAINS = np.round(np.arange(0.5, 1.31, 0.05), 2)
DEADBEAT = 1.0 / (1.0 + BETA)


def simulate(gain, q, r, n, beta, rng, step_at=None, step_nm=0.0):
    D = 0.0; C = 0.0; f = 0.0
    p_hist = np.empty(n)
    for t in range(n):
        D += rng.normal(0.0, np.sqrt(q))
        if step_at is not None and t == step_at:
            D += step_nm                      # sudden drift jump (transient test)
        p = D - C                             # residual error (pre-correction)
        p_hist[t] = p
        z = p + rng.normal(0.0, np.sqrt(r))   # measurement
        f = gain * z + (1.0 - gain) * f       # EMA command
        C += f * (1.0 + beta)                 # apply with overshoot
    return p_hist


def main():
    rng = np.random.default_rng(0)
    burn = 200

    # ---- residual RMS vs gain ----
    res = {}  # (axis, rkind) -> array over GAINS
    for axis in ("X", "Y"):
        for rkind in ("uint8", "float"):
            rms = []
            for g in GAINS:
                p = simulate(g, Q[axis], R[rkind], N, BETA, rng)
                rms.append(np.std(p[burn:]))
            res[(axis, rkind)] = np.array(rms)

    print(f"Deadbeat gain = 1/(1+beta) = {DEADBEAT:.3f}  (beta={BETA})")
    for (axis, rkind), rms in res.items():
        i08 = int(np.argmin(np.abs(GAINS - 0.80)))
        i10 = int(np.argmin(np.abs(GAINS - 1.00)))
        gbest = GAINS[int(np.argmin(rms))]
        print(f"  {axis} R={rkind:5s}: residual RMS  gain0.80={rms[i08]:.1f}nm  "
              f"gain1.00={rms[i10]:.1f}nm  (min at {gbest:.2f}: {rms.min():.1f}nm)  "
              f"  1.0 vs 0.80: {100*(rms[i10]/rms[i08]-1):+.1f}%")

    # ---- transient: response to a 300 nm drift step (float R, X) ----
    trans = {}
    for g in (0.80, 1.00, 1.10):
        p = simulate(g, 1e-6, R["float"], 120, BETA, rng, step_at=20, step_nm=300.0)
        trans[g] = p

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    colors = {("X", "float"): "#1f77b4", ("X", "uint8"): "#aec7e8",
              ("Y", "float"): "#2ca02c", ("Y", "uint8"): "#98df8a"}
    for (axis, rkind), rms in res.items():
        ls = "-" if rkind == "float" else "--"
        ax.plot(GAINS, rms, ls, color=colors[(axis, rkind)], lw=1.8,
                label=f"{axis}, R={rkind}")
    ax.axvline(DEADBEAT, color="k", ls=":", lw=1, label=f"deadbeat {DEADBEAT:.2f}")
    ax.axvline(1.0, color="red", ls=":", lw=1, label="gain 1.0")
    ax.set_xlabel("correction gain (EMA alpha)")
    ax.set_ylabel("residual tracking RMS (nm)")
    ax.set_title(f"Residual vs gain (beta={BETA}); float ECC -> R tiny, drift Q dominates")
    ax.legend(frameon=False, fontsize=7); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    tcol = {0.80: "#2ca02c", 1.00: "red", 1.10: "#d62728"}
    for g, p in trans.items():
        ax.plot(np.arange(len(p)), p, color=tcol[g], lw=1.4, label=f"gain {g:.2f}")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(20, color="gray", lw=0.8, ls=":")
    ax.set_xlim(15, 60)
    ax.set_xlabel("frame"); ax.set_ylabel("residual error (nm)")
    ax.set_title("Transient: response to a 300 nm drift step")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Drop Kalman, fixed gain: gain 1.0 vs deadbeat 0.80 (measured Q/R/beta)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    save_figure(
        fig,
        params={"beta": BETA, "deadbeat_gain": float(DEADBEAT), "n_frames": N,
                **{f"rms_{a}_{r}_gain1.0": float(res[(a, r)][int(np.argmin(np.abs(GAINS-1.0)))])
                   for a in ("X", "Y") for r in ("uint8", "float")},
                **{f"rms_{a}_{r}_gain0.8": float(res[(a, r)][int(np.argmin(np.abs(GAINS-0.8)))])
                   for a in ("X", "Y") for r in ("uint8", "float")}},
        description=("Closed-loop stage-drift simulation: residual tracking error vs "
                     "correction gain, with measured Q/R/overshoot. Tests whether a "
                     "fixed gain of 1.0 (no Kalman) is acceptable vs deadbeat 0.80."),
        data={f"rms_{a}_{r}": res[(a, r)] for a in ("X", "Y") for r in ("uint8", "float")}
             | {"gains": GAINS} | {f"transient_gain{int(g*100)}": p for g, p in trans.items()},
    )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
