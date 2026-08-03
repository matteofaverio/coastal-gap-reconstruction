"""Local-context Gaussian process reconstruction (GP M1 in the model ladder).

Reusable across targets: this is the reusable half of what was one file
(`probabilistic_sequence_pilot.py`) in the private project. The Kalman
local-level smoother (M4) is target-specific and stays in
`experiments/chlorophyll/probabilistic_models.py` -- it is fit on the raw
target series with no external features, but its degeneracy finding is
chlorophyll-specific evidence (see that module's docstring), so it is not
promoted here as a generically-reusable "M4" the way GP M1 is.

Model: an ARD Matern-3/2 kernel plus a white-noise term, fit on a local
context window around each gap (not the full series -- `pre_days`/`post_days`
control the window), predicting the log-scale target at the hidden dates.
`GaussianProcessRegressor.predict(..., return_std=True)` gives this model's
actual posterior predictive mean and standard deviation under the fitted
kernel (a genuine per-point predictive uncertainty, not a generic "confidence
interval" label) -- `run_gp_on_gap` returns both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel as C,
)
from sklearn.gaussian_process.kernels import (
    Matern,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler

__all__ = [
    "M1_FEATURES",
    "build_gp_kernel",
    "get_gap_dates",
    "get_context_window",
    "compute_t_rel",
    "run_gp_on_gap",
]

# Default GP M1 feature set: relative time plus calendar (day-of-year) position.
# No external predictors -- this model is deliberately "time-only".
M1_FEATURES: list[str] = ["t_rel", "doy_sin", "doy_cos"]


def get_gap_dates(start_date: pd.Timestamp, gap_length: int) -> pd.DatetimeIndex:
    """All hidden dates for a gap of `gap_length` days starting at `start_date`."""
    return pd.date_range(start_date, periods=gap_length, freq="D")


def get_context_window(
    target_df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    pre_days: int,
    post_days: int,
    value_col: str,
    eligible_col: str,
) -> pd.DataFrame:
    """Observed context rows within `pre_days`/`post_days` of the gap.

    `target_df` must be indexed by date and carry a `date` column (or the index
    will be reset into one). LEAKAGE SAFEGUARD: every date strictly inside the
    gap is excluded from context regardless of window width, even if it happens
    to carry a non-NaN value in `target_df` (e.g. a real gap masked upstream).
    """
    df = target_df.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.index.name or "index": "date"})
    gap_dates = set(get_gap_dates(start_date, gap_length))
    gap_end = start_date + pd.Timedelta(days=gap_length - 1)
    window_start = start_date - pd.Timedelta(days=pre_days)
    window_end = gap_end + pd.Timedelta(days=post_days)

    mask = (
        (df["date"] >= window_start)
        & (df["date"] <= window_end)
        & (~df["date"].isin(gap_dates))
        & (df[eligible_col].fillna(False).astype(bool))
        & df[value_col].notna()
    )
    ctx = df[mask].copy()
    assert not any(d in gap_dates for d in ctx["date"]), "LEAKAGE: gap date in GP context"
    return ctx


def compute_t_rel(dates: pd.Series, gap_start: pd.Timestamp, gap_length: int) -> np.ndarray:
    """Relative-time feature centred on the gap, scaled by its half-length."""
    gap_center = gap_start + pd.Timedelta(days=gap_length / 2)
    scale = max(gap_length / 2, 1)
    return ((pd.Series(dates) - gap_center).dt.days / scale).values


def build_gp_kernel(n_features: int, noise_init: float = 0.1):
    """ARD Matern-3/2 kernel plus a white-noise term.

    The first feature (index 0, conventionally `t_rel`) gets a broader initial
    length scale than the rest, matching the released private configuration.
    """
    length_scales = np.ones(n_features)
    length_scales[0] = 10.0
    return (
        C(1.0, constant_value_bounds=(1e-3, 10.0))
        * Matern(length_scale=length_scales, length_scale_bounds=(0.1, 1e3), nu=1.5)
        + WhiteKernel(noise_level=noise_init, noise_level_bounds=(1e-5, 1.0))
    )


def _impute_train_only(X_train: np.ndarray, X_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median-impute NaNs using training-column statistics only (0.0 fallback
    if a training column is entirely NaN)."""
    X_train = X_train.copy()
    X_pred = X_pred.copy()
    col_medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        fill = col_medians[j] if np.isfinite(col_medians[j]) else 0.0
        X_train[np.isnan(X_train[:, j]), j] = fill
        X_pred[np.isnan(X_pred[:, j]), j] = fill
    return X_train, X_pred


def run_gp_on_gap(
    target_df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    value_col: str,
    eligible_col: str,
    feature_cols: list[str] = M1_FEATURES,
    pre_days: int = 30,
    post_days: int = 30,
    n_restarts: int = 3,
    random_state: int | None = None,
) -> dict | None:
    """Fit a local-context GP and predict the hidden dates of one gap.

    `value_col` must already be on the modelling scale (log10 for chlorophyll,
    per `TargetSpec.benchmark_scoring_scale`) -- this function does not
    transform it. Returns `None` if there are fewer than 5 context points or
    the fit fails; otherwise a dict with `dates`, `pred`, `pred_std` (the GP's
    actual posterior predictive standard deviation on the modelling scale),
    and `n_train`.
    """
    if random_state is not None:
        np.random.seed(random_state)

    ctx = get_context_window(
        target_df, start_date, gap_length, pre_days, post_days, value_col, eligible_col
    )
    if len(ctx) < 5:
        return None

    gap_dates = get_gap_dates(start_date, gap_length)
    t_rel_ctx = compute_t_rel(ctx["date"], start_date, gap_length)
    t_rel_gap = compute_t_rel(pd.Series(gap_dates), start_date, gap_length)

    def build_X(df: pd.DataFrame, t_rel: np.ndarray) -> np.ndarray:
        cols = []
        for col in feature_cols:
            if col == "t_rel":
                cols.append(t_rel)
            else:
                cols.append(df[col].values if col in df.columns else np.full(len(df), np.nan))
        return np.column_stack(cols)

    gap_rows = pd.DataFrame({"date": gap_dates})
    for col in feature_cols:
        if col != "t_rel" and col in target_df.columns:
            gap_rows[col] = target_df.reindex(gap_dates)[col].values

    X_train = build_X(ctx, t_rel_ctx)
    y_train = ctx[value_col].to_numpy(dtype=float)
    X_pred = build_X(gap_rows, t_rel_gap)

    X_train, X_pred = _impute_train_only(X_train, X_pred)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_pred_s = scaler.transform(X_pred)

    kernel = build_gp_kernel(X_train_s.shape[1])
    gp = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=n_restarts, normalize_y=True, alpha=1e-6,
        random_state=random_state,
    )
    try:
        gp.fit(X_train_s, y_train)
        y_pred, y_std = gp.predict(X_pred_s, return_std=True)
    except Exception:
        return None

    return {
        "dates": gap_dates,
        "pred": y_pred,
        "pred_std": y_std,
        "n_train": len(ctx),
    }
