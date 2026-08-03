"""Chlorophyll Kalman local-level smoother (M4 in the private project's model
ladder) and the engineered hybrid's length-routed assignment support.

**Known degeneracy, stated up front, not buried in a separate document.** The
private project's own root-cause audit
(`docs/reports/05_chlorophyll_canonical_pipeline_stages/
stage_a_m4_kalman_root_cause_audit.md`) found that the maximum-likelihood fit
of this local-level model on the chlorophyll series is degenerate:
`sigma_r` (the observation-noise standard deviation) converges to
approximately `2.9e-15` -- i.e. essentially zero observation noise. A
random-walk state-space model with zero observation noise has, as a
mathematical property, an RTS-smoothed mean equal to linear interpolation
between the flanking observations. Direct comparison against the released
681-gap pool found this model's predictions bit-identical to linear
interpolation (to floating-point precision) in 632/681 gaps (93%).

**Consequence for how this module is used**: the private project's engineered
hybrid pipeline (`engineered_hybrid.py`) historically assigned this model to
the L=4-29 gap-length segment under the rationale "Kalman smoothing for
medium gaps" -- that rationale is not currently evidenced; the segment
behaves like interpolation. This module reproduces the model exactly as
released (same fitting procedure, same degenerate result) because that is
what `engineered_hybrid.py` needs to reproduce the released reconstruction;
it does not attempt to fix the degeneracy (re-fitting with a constrained
`sigma_r` would be a model-definition change, a scientific decision out of
scope for a publication port of already-released results).
`estimate_kalman_params` and `kalman_degeneracy_report` make the degenerate
`sigma_r` visible to any caller rather than hiding it inside an opaque fit.

GP M1 is not defined here -- it is the reusable half of the same original
private file, published in full as
`coastal_gap_reconstruction.gaussian_process` (target-agnostic: no
chlorophyll-specific column names or event logic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

__all__ = [
    "SIGMA_R_DEGENERATE_THRESHOLD",
    "kalman_filter_smoother_local_level",
    "estimate_kalman_params",
    "run_kalman_on_gap",
    "kalman_degeneracy_report",
]

# Any fitted sigma_r below this is considered degenerate (numerically
# indistinguishable from zero observation noise). The released fit converges
# to ~2.9e-15; this threshold is set two orders of magnitude above that so a
# genuinely non-degenerate future refit would not be misclassified.
SIGMA_R_DEGENERATE_THRESHOLD = 1e-10


def kalman_filter_smoother_local_level(
    y: np.ndarray, sigma_q: float, sigma_r: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Local-level Kalman filter + Rauch-Tung-Striebel (RTS) backward smoother.

    `y` is a 1-D array on the modelling scale (log10 chlorophyll); NaN entries
    are treated as missing observations (the filter predicts through them
    without an update step). Returns `(mu_filter, P_filter, mu_smooth,
    P_smooth)`, all the same length as `y`.
    """
    T = len(y)
    mu_f = np.empty(T)
    P_f = np.empty(T)
    mu_pred = np.empty(T)
    P_pred = np.empty(T)

    first_obs = y[~np.isnan(y)]
    mu_init = float(first_obs[0]) if len(first_obs) > 0 else 0.0
    P_init = sigma_r * 10  # diffuse prior

    mu_f[0] = mu_init
    P_f[0] = P_init
    mu_pred[0] = mu_init
    P_pred[0] = P_init + sigma_q**2

    for t in range(1, T):
        mp = mu_f[t - 1]
        Pp = P_f[t - 1] + sigma_q**2
        mu_pred[t] = mp
        P_pred[t] = Pp
        if not np.isnan(y[t]):
            K = Pp / (Pp + sigma_r**2)
            mu_f[t] = mp + K * (y[t] - mp)
            P_f[t] = (1 - K) * Pp
        else:
            mu_f[t] = mp
            P_f[t] = Pp

    mu_s = mu_f.copy()
    P_s = P_f.copy()
    for t in range(T - 2, -1, -1):
        J = P_f[t] / P_pred[t + 1]
        mu_s[t] = mu_f[t] + J * (mu_s[t + 1] - mu_pred[t + 1])
        P_s[t] = P_f[t] + J**2 * (P_s[t + 1] - P_pred[t + 1])

    return mu_f, P_f, mu_s, P_s


def _neg_log_lik_local_level(log_params: np.ndarray, y: np.ndarray) -> float:
    sigma_q = np.exp(log_params[0])
    sigma_r = np.exp(log_params[1])
    T = len(y)
    mu = y[~np.isnan(y)][0] if np.any(~np.isnan(y)) else 0.0
    P = sigma_r * 10.0
    nll = 0.0
    for t in range(T):
        S = P + sigma_r**2
        if not np.isnan(y[t]):
            e = y[t] - mu
            nll += 0.5 * (np.log(2 * np.pi * S) + e**2 / S)
            K = P / S
            mu = mu + K * e
            P = (1 - K) * P
        P = P + sigma_q**2
    return nll


def estimate_kalman_params(y_obs: np.ndarray) -> tuple[float, float]:
    """Estimate `(sigma_q, sigma_r)` by minimizing negative log marginal
    likelihood (Nelder-Mead, matching the released fitting procedure exactly).

    On the chlorophyll series this reliably converges to a near-zero
    `sigma_r` (see module docstring) -- that is the released, reproduced
    behavior, not a bug in this function.
    """
    diffs = np.diff(y_obs[~np.isnan(y_obs)])
    sigma_r0 = float(np.std(diffs)) / np.sqrt(2) if len(diffs) > 1 else 0.3
    sigma_q0 = sigma_r0 * 0.1
    sigma_r0 = max(sigma_r0, 1e-4)
    sigma_q0 = max(sigma_q0, 1e-5)

    x0 = np.log([sigma_q0, sigma_r0])
    result = minimize(
        _neg_log_lik_local_level, x0, args=(y_obs,), method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
    )
    sigma_q = float(np.exp(result.x[0]))
    sigma_r = float(np.exp(result.x[1]))
    return sigma_q, sigma_r


def run_kalman_on_gap(
    dates: pd.DatetimeIndex,
    values: np.ndarray,
    start_date: pd.Timestamp,
    gap_length: int,
    sigma_q: float,
    sigma_r: float,
) -> dict | None:
    """Run the local-level Kalman smoother on a full series with one gap masked.

    `dates`/`values` describe the full series (already on the modelling
    scale); the gap's own dates are masked to NaN before filtering, so the
    smoother never sees the hidden ground truth (leakage-safe by
    construction). Returns `None` if the gap's dates are not found in `dates`.
    """
    gap_dates = set(pd.date_range(start_date, periods=gap_length, freq="D"))
    y_full = np.asarray(values, dtype=float).copy()
    mask = pd.DatetimeIndex(dates).isin(gap_dates)
    if not mask.any():
        return None
    y_full[mask] = np.nan

    _, _, mu_s, _ = kalman_filter_smoother_local_level(y_full, sigma_q, sigma_r)
    gap_idx = np.where(mask)[0]
    return {
        "dates": pd.DatetimeIndex(dates)[gap_idx],
        "pred": mu_s[gap_idx],
    }


def kalman_degeneracy_report(sigma_q: float, sigma_r: float) -> dict:
    """Report whether a fitted `(sigma_q, sigma_r)` pair is in the degenerate
    (near-zero observation noise, interpolation-equivalent) regime.

    Used by `engineered_hybrid.py` and by
    `tests/test_probabilistic_models.py` to make the degeneracy an assertion
    rather than a claim in prose only.
    """
    is_degenerate = sigma_r < SIGMA_R_DEGENERATE_THRESHOLD
    return {
        "sigma_q": sigma_q,
        "sigma_r": sigma_r,
        "is_degenerate": bool(is_degenerate),
        "note": (
            "sigma_r below threshold: fit is numerically equivalent to linear "
            "interpolation (see module docstring)."
            if is_degenerate
            else "sigma_r above threshold: fit retains non-trivial observation noise."
        ),
    }
