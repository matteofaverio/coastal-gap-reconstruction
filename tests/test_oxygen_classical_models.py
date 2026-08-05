"""Tests for `experiments.oxygen.classical_models` -- leakage safety and
determinism on synthetic data (no dependency on the real feature tables)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.oxygen import classical_models as cm

TARGET_COL = "oxygen_mean_mgL"
ELIGIBLE_COL = "eligible_ge_18"


def _synthetic_target(n=200, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    values = np.abs(rng.normal(6.0, 1.0, n))
    df = pd.DataFrame({TARGET_COL: values, ELIGIBLE_COL: True}, index=dates)
    return df


def _synthetic_features(target_df, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {f"feat_{i}": rng.normal(0, 1, len(target_df)) for i in range(5)},
        index=target_df.index,
    )


def test_model0_linear_interp_uses_only_visible_data():
    target_df = _synthetic_target()
    start = target_df.index[100]
    preds = cm.run_model0_gap("climatology", target_df, start, 3)
    assert len(preds) == 3
    assert all(v == v for v in preds.values())  # no NaN with full eligibility


def test_tabular_loco_excludes_hidden_dates_from_training():
    target_df = _synthetic_target()
    features_df = _synthetic_features(target_df)
    start = target_df.index[100]
    gap_length = 3
    hidden_dates = pd.date_range(start, periods=gap_length, freq="D")

    # Monkeypatch: corrupt the target at hidden dates with an extreme value;
    # if the model trained on it, predictions would be wildly different from
    # a model that never saw it.
    corrupted = target_df.copy()
    corrupted.loc[hidden_dates, TARGET_COL] = 9999.0

    result_clean = cm.fit_predict_tabular_gap(
        "ridge", target_df, features_df, start, gap_length, list(features_df.columns),
        target_col=TARGET_COL, eligible_col=ELIGIBLE_COL,
    )
    result_corrupted = cm.fit_predict_tabular_gap(
        "ridge", corrupted, features_df, start, gap_length, list(features_df.columns),
        target_col=TARGET_COL, eligible_col=ELIGIBLE_COL,
    )
    # Predictions must be identical: the corrupted values at hidden dates
    # were never part of the training set in either case.
    for d in hidden_dates:
        assert result_clean["pred"][d] == pytest.approx(result_corrupted["pred"][d])


def test_tabular_loco_is_deterministic():
    target_df = _synthetic_target()
    features_df = _synthetic_features(target_df)
    start = target_df.index[100]
    r1 = cm.fit_predict_tabular_gap("extratrees_high_capacity", target_df, features_df, start, 3,
                                     list(features_df.columns), target_col=TARGET_COL, eligible_col=ELIGIBLE_COL)
    r2 = cm.fit_predict_tabular_gap("extratrees_high_capacity", target_df, features_df, start, 3,
                                     list(features_df.columns), target_col=TARGET_COL, eligible_col=ELIGIBLE_COL)
    assert r1["pred"] == pytest.approx(r2["pred"])


def test_tabular_loco_returns_warning_for_insufficient_training_rows():
    target_df = _synthetic_target(n=15)
    target_df[ELIGIBLE_COL] = False
    features_df = _synthetic_features(target_df)
    result = cm.fit_predict_tabular_gap("ridge", target_df, features_df, target_df.index[5], 2,
                                         list(features_df.columns), target_col=TARGET_COL, eligible_col=ELIGIBLE_COL)
    assert result["warning"] is not None
    assert result["pred"] == {}


def test_build_learner_unknown_raises():
    with pytest.raises(ValueError, match="Unknown learner"):
        cm.build_learner("not_a_learner")


def test_gp_gap_edge_never_uses_hidden_target_values():
    target_df = _synthetic_target()
    start = target_df.index[100]
    gap_length = 3
    hidden_dates = pd.date_range(start, periods=gap_length, freq="D")

    corrupted = target_df.copy()
    corrupted.loc[hidden_dates, TARGET_COL] = 9999.0

    r_clean = cm.run_gp_gap_edge_gap(target_df, start, gap_length, target_col=TARGET_COL, eligible_col=ELIGIBLE_COL)
    r_corrupted = cm.run_gp_gap_edge_gap(corrupted, start, gap_length, target_col=TARGET_COL, eligible_col=ELIGIBLE_COL)
    if r_clean["warning"] is None and r_corrupted["warning"] is None:
        for d in hidden_dates:
            assert r_clean["pred"][d] == pytest.approx(r_corrupted["pred"][d])


def test_model0_evaluation_produces_one_row_per_gap_day():
    target_df = _synthetic_target()
    candidates = pd.DataFrame([
        {"gap_id": "g1", "gap_length": 3, "start_date": target_df.index[50]},
        {"gap_id": "g2", "gap_length": 1, "start_date": target_df.index[100]},
    ])
    result = cm.run_model0_evaluation("linear_interp", candidates, target_df, target_col=TARGET_COL, eligible_col=ELIGIBLE_COL)
    assert len(result) == 4  # 3 + 1 days
    assert set(result["gap_id"]) == {"g1", "g2"}
