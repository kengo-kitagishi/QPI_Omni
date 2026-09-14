"""Theoretical correction for second-moment ellipse fit applied to rod cells.

skimage.measure.regionprops uses second central moments to fit an
equivalent ellipse to the mask. For a true rod (cylinder + 2 semicircular
caps) this is not a perfect fit; both major_axis_length and
minor_axis_length are systematically overestimated, with the bias
depending on the aspect ratio AR = L / (2r) (true length / true width).

This module computes the correction factors so that
    true_long  = corr_long(AR_observed)  * skimage_major
    true_short = corr_short(AR_observed) * skimage_minor

AR_observed = skimage_major / skimage_minor is used as a first-order
proxy. Iterative refinement converges in 2-3 iterations because the
correction is monotone and bounded.
"""
from __future__ import annotations

import numpy as np


def _variance_ratio_y(AR: np.ndarray) -> np.ndarray:
    """variance_y / r² for a rod with aspect ratio AR = L/(2r).
    For AR=1 returns π/(4·π) = 1/4 (disk).
    For AR→∞ returns 1/3 (long cylinder).
    """
    AR = np.asarray(AR, dtype=float)
    num = (4.0 / 3.0) * (AR - 1.0) + np.pi / 4.0
    den = 4.0 * (AR - 1.0) + np.pi
    return num / den


def _variance_ratio_x(AR: np.ndarray) -> np.ndarray:
    """variance_x / r² for a rod with aspect ratio AR.

    Calculation:
      Area / r²       = 4(AR-1) + π
      M_xx / r⁴ = cylinder + cap contributions

    cylinder body  (length L-2r, width 2r, centered at origin):
        I_x_cyl = (2r)(L-2r)³/12 = (2r)(2r(AR-1))³/12
        / r⁴   = (2/12) (2(AR-1))³ = (4/3)(AR-1)³
    each semicircular cap, geometric center at ±(L/2 - r):
        cap centroid offset along x: x_c = ±(L/2 - r + 4r/(3π))
                                         = ±r[(AR-1) + 4/(3π)]
        cap area = π r² / 2
        cap own-centroid 2nd moment:
            I_x_cap_local = (πr⁴/8) - (πr²/2)(4r/(3π))² = πr⁴/8 - 8r⁴/(9π)
        parallel-axis:
            I_x_cap_total = I_x_cap_local + (πr²/2) x_c²
        2 caps contribute: 2 × I_x_cap_total
    """
    AR = np.asarray(AR, dtype=float)
    # cylinder
    I_cyl = (4.0 / 3.0) * (AR - 1.0) ** 3
    # cap local 2nd moment about its own centroid (offset 4r/(3π) from disk
    # center; semicircle moment about disk center is πr⁴/8)
    cap_local = np.pi / 8.0 - 8.0 / (9.0 * np.pi)
    cap_area = np.pi / 2.0
    x_c = (AR - 1.0) + 4.0 / (3.0 * np.pi)
    I_cap_total = cap_local + cap_area * x_c ** 2
    I_xx_total = I_cyl + 2.0 * I_cap_total  # / r⁴
    area = 4.0 * (AR - 1.0) + np.pi  # / r²
    return I_xx_total / area  # = variance_x / r²


# Precompute the inverse lookup table:
#   AR_observed = sk_major / sk_minor = sqrt(rx(AR_true) / ry(AR_true))
# We tabulate AR_true → AR_observed once, then invert by interpolation.
_AR_TRUE = np.linspace(1.0, 200.0, 4000)
_AR_OBS = np.sqrt(
    _variance_ratio_x(_AR_TRUE) / _variance_ratio_y(_AR_TRUE)
)


def observed_to_true_AR(AR_obs: np.ndarray | float) -> np.ndarray:
    """Invert AR_observed → AR_true via the precomputed lookup."""
    AR_obs = np.asarray(AR_obs, dtype=float)
    return np.interp(AR_obs, _AR_OBS, _AR_TRUE)


def correction_factors(skimage_major: np.ndarray | float,
                       skimage_minor: np.ndarray | float
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Recover (true_L, true_2r) from skimage major/minor axis observations.

    The skimage second-moment ellipse fit applied to a rod (cylinder + 2
    semicircular caps) gives
        sk_minor = 4 r sqrt(ry(AR_true))
        sk_major = 4 r sqrt(rx(AR_true))
    so
        AR_obs = sk_major / sk_minor = sqrt(rx(AR_true) / ry(AR_true))
    which we invert via the precomputed table.
    """
    M = np.asarray(skimage_major, dtype=float)
    m = np.asarray(skimage_minor, dtype=float)
    AR_obs = M / np.maximum(m, 1e-9)
    AR_true = observed_to_true_AR(AR_obs)
    ry = _variance_ratio_y(AR_true)
    true_minor = m / (2.0 * np.sqrt(ry))   # = 2r
    true_major = AR_true * true_minor      # = L
    return true_major, true_minor


def true_rod_volume_um3(skimage_major_um: np.ndarray | float,
                        skimage_minor_um: np.ndarray | float
                        ) -> np.ndarray | float:
    """Corrected rod volume from skimage axis observations."""
    L, w = correction_factors(skimage_major_um, skimage_minor_um)
    r = w / 2.0
    h = np.maximum(L - 2.0 * r, 0.0)
    return (4.0 / 3.0) * np.pi * r ** 3 + np.pi * r ** 2 * h


def _self_test():
    print("--- self test ---")
    for AR in [1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0]:
        ry = _variance_ratio_y(np.array([AR]))[0]
        rx = _variance_ratio_x(np.array([AR]))[0]
        sk_minor = 4.0 * np.sqrt(ry)   # / r
        sk_major = 4.0 * np.sqrt(rx)   # / r
        # 2r = 2, L = 2 * AR  in units of r
        true_minor = 2.0
        true_major = 2.0 * AR
        true_L, true_2r = correction_factors(sk_major, sk_minor)
        print(f"AR={AR:5.1f}  sk_minor/true_minor={sk_minor/2.0:.4f}  "
              f"sk_major/true_major={sk_major/(2.0*AR):.4f}  "
              f"recovered_2r/2={true_2r/2.0:.4f}  recovered_L/L={true_L/(2.0*AR):.4f}")


if __name__ == "__main__":
    _self_test()
