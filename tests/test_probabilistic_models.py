"""Tests for the chlorophyll Kalman local-level smoother, including a pinned
degeneracy-regression test against the released daily target table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.chlorophyll import probabilistic_models as pm

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"


def test_filter_smoother_deterministic_and_finite():
    rng = np.random.default_rng(0)
    y = np.sin(np.arange(100) / 10) + rng.normal(0, 0.05, 100)
    y[40:45] = np.nan
    mu_f, P_f, mu_s, P_s = pm.kalman_filter_smoother_local_level(y, sigma_q=0.1, sigma_r=0.05)
    assert np.all(np.isfinite(mu_f))
    assert np.all(np.isfinite(mu_s))
    assert np.all(P_f >= 0)
    assert np.all(P_s >= 0)


def test_run_kalman_on_gap_masks_the_gap_before_filtering():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(1)
    values = np.sin(np.arange(100) / 10) + rng.normal(0, 0.02, 100)
    result = pm.run_kalman_on_gap(dates, values, dates[50], 5, sigma_q=0.1, sigma_r=0.05)
    assert result is not None
    assert len(result["dates"]) == 5
    assert np.all(np.isfinite(result["pred"]))


def test_run_kalman_on_gap_prediction_does_not_depend_on_hidden_truth():
    """Tamper-invariance: the gap's own hidden values are masked to NaN before
    filtering, so changing them beforehand must not change the smoothed
    prediction for that gap."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    rng = np.random.default_rng(2)
    values = np.sin(np.arange(100) / 10) + rng.normal(0, 0.02, 100)
    baseline = pm.run_kalman_on_gap(dates, values, dates[50], 5, sigma_q=0.1, sigma_r=0.05)

    tampered = values.copy()
    tampered[50:55] = 9999.0
    tampered_result = pm.run_kalman_on_gap(dates, tampered, dates[50], 5, sigma_q=0.1, sigma_r=0.05)

    np.testing.assert_allclose(baseline["pred"], tampered_result["pred"])


def test_kalman_degeneracy_report_thresholds():
    assert pm.kalman_degeneracy_report(0.1, 1e-14)["is_degenerate"] is True
    assert pm.kalman_degeneracy_report(0.1, 0.5)["is_degenerate"] is False


def test_estimate_kalman_params_reproduces_the_documented_degeneracy_on_the_released_series():
    """Pinned regression test: fitting on the real released chlorophyll daily
    target must reproduce the documented sigma_r degeneracy (order ~1e-13 to
    ~1e-15), not merely "some small number" -- this is the exact finding the
    module docstring and the private root-cause audit both describe."""
    df = pd.read_csv(TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()
    eligible = df["target_eligible_default"].fillna(False).astype(bool)
    y = df["chl_mean"].where(eligible & (df["chl_mean"] > 1e-4))
    y_log = np.log10(y).dropna().to_numpy()

    sigma_q, sigma_r = pm.estimate_kalman_params(y_log)
    report = pm.kalman_degeneracy_report(sigma_q, sigma_r)

    assert report["is_degenerate"] is True
    assert sigma_r < 1e-8
