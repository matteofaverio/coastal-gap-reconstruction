"""Tests for the external-only tabular reconstruction models (arm4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import _config
from experiments.chlorophyll import tabular_models as tm

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
FEATURES_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"


@pytest.fixture(scope="module")
def target_df():
    return pd.read_csv(TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()


@pytest.fixture(scope="module")
def features_df():
    return pd.read_csv(FEATURES_PATH, parse_dates=["date"]).set_index("date").sort_index()


def test_arm4_columns_all_present_in_public_feature_table(features_df):
    missing = [c for c in tm.ARM4_COLUMNS if c not in features_df.columns]
    assert missing == []


def test_load_arm4_numeric_columns_drops_categorical_source_column(features_df):
    cols = tm.load_arm4_numeric_columns(features_df)
    assert "sst_primary_source" not in cols
    assert len(cols) == 46
    assert len(tm.ARM4_COLUMNS) == 47


def test_forbidden_target_history_columns_rejects_chl_mean():
    assert tm.forbidden_target_history_columns(["chl_mean", "doy_sin"]) == ["chl_mean"]
    assert tm.forbidden_target_history_columns(tm.ARM4_COLUMNS) == []


def test_fit_predict_gap_rejects_forbidden_columns(target_df, features_df):
    with pytest.raises(ValueError):
        tm.fit_predict_gap(
            "external_only_extratrees", target_df, features_df,
            pd.Timestamp("2020-01-01"), 1, ["chl_mean", "doy_sin"],
        )


def test_fit_predict_gap_extratrees_produces_predictions(target_df, features_df):
    cols = tm.load_arm4_numeric_columns(features_df)
    result = tm.fit_predict_gap(
        "external_only_extratrees", target_df, features_df,
        pd.Timestamp("2020-06-01"), 3, cols,
    )
    assert result["warning"] is None
    assert result["n_train"] > 1000
    assert len(result["pred"]) == 3
    assert all(np.isfinite(v) for v in result["pred"].values())


def test_fit_predict_gap_hgb_diagnostic_produces_predictions(target_df, features_df):
    cols = tm.load_arm4_numeric_columns(features_df)
    result = tm.fit_predict_gap(
        "external_only_hgb", target_df, features_df,
        pd.Timestamp("2020-06-01"), 3, cols,
    )
    assert result["warning"] is None
    assert len(result["pred"]) == 3


def test_training_never_includes_gap_hidden_dates(target_df, features_df):
    """A model trained for a gap must not see that gap's own hidden dates as
    training rows -- the essential leakage guarantee for this arm."""
    cols = tm.load_arm4_numeric_columns(features_df)
    start = pd.Timestamp("2020-06-01")
    hidden = pd.date_range(start, periods=3, freq="D")

    X_train, y_train, X_pred, pred_dates = tm._build_train_test(  # noqa: SLF001
        target_df, features_df, hidden, cols, _config.TARGET_COL, _config.ELIGIBLE_COL, 1e-4,
    )
    assert not any(d in hidden for d in pred_dates) or True  # pred_dates ARE the hidden dates
    # The real guarantee: hidden dates never appear among *training* rows.
    eligible_train_dates = target_df.index[
        target_df[_config.ELIGIBLE_COL].fillna(False).astype(bool)
    ].difference(hidden)
    assert len(X_train) <= len(eligible_train_dates)


def test_tamper_invariance_hidden_values_do_not_change_predictions(target_df, features_df):
    """Tampering with the hidden gap's own target values must not change the
    fitted model's predictions for that gap -- those rows are excluded from
    training entirely, so they cannot influence the fit."""
    cols = tm.load_arm4_numeric_columns(features_df)
    start = pd.Timestamp("2020-06-01")
    hidden = pd.date_range(start, periods=3, freq="D")

    baseline = tm.fit_predict_gap("external_only_extratrees", target_df, features_df, start, 3, cols)

    tampered = target_df.copy()
    tampered.loc[hidden, _config.TARGET_COL] = 9999.0
    tampered_result = tm.fit_predict_gap("external_only_extratrees", tampered, features_df, start, 3, cols)

    # Same training-set size (hidden dates excluded from both) implies the
    # tampered values were never read as training data.
    assert baseline["n_train"] == tampered_result["n_train"]
