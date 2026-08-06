"""Tests for the engineered hybrid length-routed method assignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import engineered_hybrid as eh


@pytest.mark.parametrize(
    "gap_length,expected_method",
    [(1, "gaussian_process"), (3, "gaussian_process"), (4, "kalman_local_level"),
     (29, "kalman_local_level"), (30, "gap_edge_residual"), (90, "gap_edge_residual")],
)
def test_assign_method_boundaries(gap_length, expected_method):
    assert eh.assign_method(gap_length) == expected_method


def test_assignment_rule_covers_1_through_100_with_no_gaps():
    for L in range(1, 100):
        eh.assign_method(L)  # must not raise


def _synthetic_target(n: int = 500, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    rng = np.random.default_rng(seed)
    values = 1.0 + 0.3 * np.sin(np.arange(n) / 15) + rng.normal(0, 0.05, n)
    values = np.abs(values) + 0.1
    return pd.DataFrame({"chl_mean": values, "target_eligible_default": True}, index=dates)


def test_reconstruct_gap_routes_gp_for_short_gap():
    target_df = _synthetic_target()
    features_df = pd.DataFrame(index=target_df.index)
    sigma_q, sigma_r = 0.1, 0.05
    result = eh.reconstruct_gap(target_df, features_df, target_df.index[100], 2, sigma_q, sigma_r)
    assert result["method"] == "gaussian_process"
    assert "kalman_degeneracy" not in result


def test_reconstruct_gap_routes_kalman_for_medium_gap():
    target_df = _synthetic_target()
    features_df = pd.DataFrame(index=target_df.index)
    obs_log = np.log10(target_df["chl_mean"]).dropna().to_numpy()
    from experiments.chlorophyll import probabilistic_models as pm
    sigma_q, sigma_r = pm.estimate_kalman_params(obs_log)
    result = eh.reconstruct_gap(target_df, features_df, target_df.index[200], 10, sigma_q, sigma_r)
    assert result["method"] == "kalman_local_level"
    assert "kalman_degeneracy" in result
    assert result["kalman_degeneracy"]["is_degenerate"] in (True, False)


def test_run_engineered_hybrid_over_multiple_lengths():
    target_df = _synthetic_target()
    features_df = pd.DataFrame(index=target_df.index)
    candidates = pd.DataFrame([
        {"gap_id": "g1", "gap_length": 2, "start_date": target_df.index[50]},
        {"gap_id": "g2", "gap_length": 10, "start_date": target_df.index[150]},
        {"gap_id": "g3", "gap_length": 35, "start_date": target_df.index[300]},
    ])
    preds, kalman_params = eh.run_engineered_hybrid(candidates, target_df, features_df)
    assert set(preds["assigned_method"]) <= {"gaussian_process", "kalman_local_level", "gap_edge_residual"}
    assert "sigma_q" in kalman_params and "sigma_r" in kalman_params
    methods_by_gap = preds.groupby("gap_id")["assigned_method"].first().to_dict()
    assert methods_by_gap.get("g1") == "gaussian_process"
    assert methods_by_gap.get("g2") == "kalman_local_level"


def test_run_engineered_hybrid_gap_edge_branch_uses_pool_wide_context():
    """Regression test for a real bug found this session: the gap_edge_residual
    branch (L>=30) must draw leave-one-gap-out training rows from the *entire*
    candidate pool, not just its own (empty) context -- a lone L>=30 gap with
    no other gaps to train against will otherwise silently produce zero
    prediction rows for every L>=30 gap (discovered via the full 449-gap
    matched-support run, where every one of the 50 L=30 gaps failed with
    "insufficient_training_rows" before this fix)."""
    target_df = _synthetic_target(n=3000)
    features_df = pd.DataFrame(index=target_df.index)
    starts = target_df.index[60 : 60 + 20 * 120 : 120]
    candidates = pd.DataFrame([
        {
            "gap_id": f"L30_{i}", "gap_length": 35, "start_date": s,
            "end_date": s + pd.Timedelta(days=34),
        }
        for i, s in enumerate(starts)
    ])
    preds, _ = eh.run_engineered_hybrid(candidates, target_df, features_df)
    assert not preds.empty, "gap_edge_residual branch produced no predictions with a realistic context pool"
    assert (preds["assigned_method"] == "gap_edge_residual").all()
    assert set(preds["gap_id"]).issuperset(set(candidates["gap_id"].iloc[:5]))
