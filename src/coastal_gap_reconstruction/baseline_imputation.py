"""Baseline reconstruction methods (Model 0 in the model ladder).

Three baselines, all fit only on visible (non-hidden, eligible) target days
and never using any hidden/masked target value:

A. Monthly climatology -- mean of the target by calendar month, excluding
   hidden days.
B. Persistence -- the last observed value before the gap, held flat across
   all hidden days.
C. Linear interpolation -- linear interpolation between the last pre-gap and
   first post-gap observations. This is a diagnostic reconstruction method
   only; it is not forecast-safe (it requires future data).

If a prediction cannot be computed (e.g. no data on one side of the gap),
these functions return NaN for the affected days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_loading import TARGET_COL, ELIGIBLE_COL


def _visible_eligible(
    masked_target: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
) -> pd.DataFrame:
    """Return eligible, non-hidden rows from a target table with a gap masked."""
    hidden_dates = set(pd.date_range(start_date, periods=gap_length, freq="D"))
    eligible_mask = masked_target[ELIGIBLE_COL].fillna(False).astype(bool)
    visible = masked_target[
        eligible_mask
        & masked_target[TARGET_COL].notna()
        & ~masked_target.index.isin(hidden_dates)
    ]
    return visible


def monthly_climatology(
    masked_target: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
) -> dict[pd.Timestamp, float]:
    """Predict each hidden day using the mean target value for its calendar month.

    Falls back to the global mean of visible eligible days if a month has no data.
    """
    visible = _visible_eligible(masked_target, start_date, gap_length)
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")

    monthly_means: dict[int, float] = {}
    if not visible.empty:
        for month, group in visible.groupby(visible.index.month):
            monthly_means[int(month)] = float(group[TARGET_COL].mean())
    global_mean = float(visible[TARGET_COL].mean()) if not visible.empty else np.nan

    return {d: monthly_means.get(d.month, global_mean) for d in hidden_dates}


def persistence_baseline(
    masked_target: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
) -> dict[pd.Timestamp, float]:
    """Predict every hidden day with the most recent visible value before the gap."""
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")
    visible = _visible_eligible(masked_target, start_date, gap_length)

    pre_gap = visible[visible.index < start_date]
    last_val = float(pre_gap[TARGET_COL].iloc[-1]) if not pre_gap.empty else np.nan

    return {d: last_val for d in hidden_dates}


def linear_interpolation_baseline(
    masked_target: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
) -> dict[pd.Timestamp, float]:
    """Linearly interpolate between the last pre-gap and first post-gap observations.

    Diagnostic reconstruction only -- not forecast-safe, since it requires
    knowledge of a future (post-gap) observation.
    """
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")
    gap_end = start_date + pd.Timedelta(days=gap_length - 1)
    visible = _visible_eligible(masked_target, start_date, gap_length)

    pre_gap = visible[visible.index < start_date]
    post_gap = visible[visible.index > gap_end]

    if pre_gap.empty or post_gap.empty:
        return {d: np.nan for d in hidden_dates}

    t_before = pre_gap.index[-1]
    v_before = float(pre_gap[TARGET_COL].iloc[-1])
    t_after = post_gap.index[0]
    v_after = float(post_gap[TARGET_COL].iloc[0])

    total_days = (t_after - t_before).days

    predictions: dict[pd.Timestamp, float] = {}
    for d in hidden_dates:
        frac = (d - t_before).days / total_days
        predictions[d] = v_before + frac * (v_after - v_before)
    return predictions


def run_all_baselines(
    masked_target: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
) -> dict[str, dict[pd.Timestamp, float]]:
    """Run all three baselines and return predictions keyed by method name."""
    return {
        "clim_monthly": monthly_climatology(masked_target, start_date, gap_length),
        "persistence": persistence_baseline(masked_target, start_date, gap_length),
        "linear_interp": linear_interpolation_baseline(masked_target, start_date, gap_length),
    }
