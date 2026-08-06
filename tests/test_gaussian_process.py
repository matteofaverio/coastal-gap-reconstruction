"""Tests for the reusable local-context Gaussian process model (GP M1)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import gaussian_process as gp


def _synthetic_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    y = np.sin(np.arange(n) / 10) + rng.normal(0, 0.05, n)
    df = pd.DataFrame({
        "y_log": y,
        "eligible": True,
        "doy_sin": np.sin(2 * np.pi * dates.dayofyear / 365),
        "doy_cos": np.cos(2 * np.pi * dates.dayofyear / 365),
        "date": dates,
    }, index=dates)
    return df


def test_get_gap_dates_length():
    dates = gp.get_gap_dates(pd.Timestamp("2020-01-01"), 5)
    assert len(dates) == 5


def test_context_window_excludes_gap_dates():
    df = _synthetic_df(100)
    start = df.index[50]
    ctx = gp.get_context_window(df, start, 5, pre_days=20, post_days=20, value_col="y_log", eligible_col="eligible")
    gap_dates = set(gp.get_gap_dates(start, 5))
    assert not any(d in gap_dates for d in ctx["date"])


def test_context_window_leakage_safeguard_even_if_masked_row_has_a_value():
    """Even if a row inside the gap window happens to carry a non-NaN value
    (e.g. because an upstream caller forgot to mask it, or it's a synthetic
    fixture), the context window must still exclude it by date."""
    df = _synthetic_df(100)
    start = df.index[50]
    gap_dates = gp.get_gap_dates(start, 5)
    # Deliberately leave the gap dates' values populated (no NaN masking).
    ctx = gp.get_context_window(df, start, 5, pre_days=10, post_days=10, value_col="y_log", eligible_col="eligible")
    assert set(gap_dates).isdisjoint(set(ctx["date"]))


def test_run_gp_on_gap_returns_predictions_and_std():
    df = _synthetic_df(200)
    result = gp.run_gp_on_gap(
        df, df.index[100], 5, value_col="y_log", eligible_col="eligible",
        feature_cols=gp.M1_FEATURES, random_state=42,
    )
    assert result is not None
    assert len(result["dates"]) == 5
    assert len(result["pred"]) == 5
    assert all(np.isfinite(result["pred_std"]))
    assert result["n_train"] > 5


def test_run_gp_on_gap_returns_none_with_too_little_context():
    df = _synthetic_df(200)
    df["eligible"] = False
    df.loc[df.index[95:105], "eligible"] = True
    result = gp.run_gp_on_gap(
        df, df.index[100], 3, value_col="y_log", eligible_col="eligible",
        pre_days=1, post_days=1, random_state=42,
    )
    assert result is None


def test_predictive_std_is_nonnegative():
    df = _synthetic_df(200)
    result = gp.run_gp_on_gap(
        df, df.index[100], 5, value_col="y_log", eligible_col="eligible", random_state=42,
    )
    assert result is not None
    assert np.all(result["pred_std"] >= 0)
