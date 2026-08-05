"""Tests for `experiments.oxygen.tsicl_models` (no live checkpoint required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.oxygen import benchmark_contract as bc
from experiments.oxygen import tsicl_models as tm


def test_nine_arms_defined_five_audited_four_ablation():
    assert set(tm.AUDITED_ORIGINAL_ARM_COLUMNS) == set(bc.TSICL_AUDITED_ORIGINAL_ARMS)
    assert set(tm.FAMILY_ABLATION_ARM_COLUMNS) == set(bc.TSICL_FAMILY_ABLATION_ARMS)
    assert len(tm.ALL_ARM_COLUMNS) == 9


def test_target_only_has_no_covariate_columns():
    assert tm.AUDITED_ORIGINAL_ARM_COLUMNS["target_only"] is None


def test_no_satellite_chlorophyll_or_forbidden_column_in_any_arm():
    for arm, cols in tm.ALL_ARM_COLUMNS.items():
        if cols is None:
            continue
        for c in cols:
            assert "chl_cons" not in c
            assert "chl_perm" not in c
            assert "chl_anom" not in c
            assert "chl_mean" not in c
            assert "BTGOXD" not in c


def test_assert_covariate_cols_safe_raises_on_satellite_chl():
    with pytest.raises(ValueError, match="Forbidden substring"):
        tm.assert_covariate_cols_safe(["chl_cons_log10"], "test_arm")


def test_build_covariate_block_unknown_arm_raises():
    df = pd.DataFrame({"doy_sin": [0.1, 0.2]}, index=pd.date_range("2020-01-01", periods=2))
    with pytest.raises(ValueError, match="Unknown oxygen TS-ICL arm"):
        tm.build_covariate_block("not_a_real_arm", df)


def test_build_covariate_block_target_only_returns_none():
    df = pd.DataFrame({"doy_sin": [0.1, 0.2]}, index=pd.date_range("2020-01-01", periods=2))
    assert tm.build_covariate_block("target_only", df) is None


def test_window_days_matches_chlorophyll_convention():
    assert tm.WINDOW_DAYS == 730


def test_load_target_series_is_raw_mgL_no_log_transform():
    dates, target = tm.load_target_series()
    valid = target[~np.isnan(target)]
    # Raw oxygen values are O(1-15) mg/L; a log10 series would be O(0-1.2).
    assert valid.min() >= 0
    assert valid.max() > 2.0  # would be impossible under a log10 transform of mg/L this small
