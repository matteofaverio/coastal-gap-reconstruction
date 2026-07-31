"""Scoring metrics for gap reconstruction methods.

compute_gap_metrics: per-gap, per-method metric rows (MAE, RMSE, bias, r).
aggregate_metrics: group-level summary (mean / std of per-gap metrics),
    e.g. grouped by method and gap length, or by method and season.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except ImportError:  # pragma: no cover - scipy is an optional dependency here
    _scipy_stats = None

from .data_loading import TARGET_COL


def compute_gap_metrics(
    target_df: pd.DataFrame,
    predictions: dict[str, dict[pd.Timestamp, float]],
    start_date: pd.Timestamp,
    gap_length: int,
    gap_id: str,
    gap_info: dict,
    target_col: str = TARGET_COL,
) -> list[dict]:
    """Compute per-gap metrics for each method in `predictions`.

    For each method, computes MAE, RMSE, bias (mean(pred - true), positive
    = overestimate), Pearson r (NaN for gap_length == 1 or constant
    predictions), n_valid (days with both true and predicted values), and
    coverage (fraction of hidden days with a non-NaN prediction).

    `target_col` lets this score against any sensor's daily target table,
    not just the default chlorophyll column.
    """
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")

    rows: list[dict] = []
    for method_name, preds in predictions.items():
        y_true: list[float] = []
        y_pred: list[float] = []
        n_predicted = 0

        for d in hidden_dates:
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            pred_val = preds.get(d, np.nan)
            if not np.isnan(pred_val):
                n_predicted += 1
            if not np.isnan(true_val) and not np.isnan(pred_val):
                y_true.append(true_val)
                y_pred.append(pred_val)

        n_valid = len(y_true)
        coverage = n_predicted / gap_length if gap_length > 0 else np.nan

        if n_valid == 0:
            mae = rmse = bias = r = np.nan
        else:
            arr_true = np.array(y_true, dtype=float)
            arr_pred = np.array(y_pred, dtype=float)
            errors = arr_pred - arr_true
            mae = float(np.mean(np.abs(errors)))
            rmse = float(np.sqrt(np.mean(errors**2)))
            bias = float(np.mean(errors))
            if (
                n_valid >= 3
                and np.std(arr_true) > 0
                and np.std(arr_pred) > 0
                and _scipy_stats is not None
            ):
                r = float(_scipy_stats.pearsonr(arr_true, arr_pred)[0])
            else:
                r = np.nan

        rows.append({
            "gap_id": gap_id,
            "method": method_name,
            "gap_length": gap_length,
            "season": gap_info.get("season", np.nan),
            "year": gap_info.get("year", np.nan),
            "target_mean_true": gap_info.get("target_mean_true", np.nan),
            "n_valid": n_valid,
            "coverage": round(coverage, 4) if not np.isnan(coverage) else np.nan,
            "mae": round(mae, 4) if not np.isnan(mae) else np.nan,
            "rmse": round(rmse, 4) if not np.isnan(rmse) else np.nan,
            "bias": round(bias, 4) if not np.isnan(bias) else np.nan,
            "r": round(r, 4) if not np.isnan(r) else np.nan,
        })

    return rows


def aggregate_metrics(metrics_df: pd.DataFrame, groupby_cols: list[str]) -> pd.DataFrame:
    """Aggregate per-gap metrics by an arbitrary set of grouping columns.

    Returns one row per group with the mean and std of MAE, RMSE, bias, r,
    and coverage, plus the number of gaps contributing to the group.
    """
    if metrics_df.empty:
        return pd.DataFrame()

    agg = metrics_df.groupby(groupby_cols).agg(
        n_gaps=("gap_id", "nunique"),
        mae_mean=("mae", "mean"),
        mae_std=("mae", "std"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        bias_mean=("bias", "mean"),
        r_mean=("r", "mean"),
        coverage_mean=("coverage", "mean"),
    ).reset_index()
    return agg
