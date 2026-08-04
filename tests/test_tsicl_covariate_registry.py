"""Tests for the TS-ICL covariate-arm registry: exact column membership,
descriptive-name consistency with the released public table, and placebo
transform determinism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import tsicl_covariate_registry as reg

REPO_ROOT_RESULTS = "results_public/chlorophyll/chlorophyll_covariate_mechanism_summary.csv"


def test_registry_uses_only_descriptive_names_not_internal_codes():
    """No arm_id or public_name may be a bare internal short code like
    'C9' or 'arm D' -- the whole point of this registry."""
    internal_code_pattern = __import__("re").compile(r"^C\d+[a-z]?$|^arm [A-Z]$", __import__("re").IGNORECASE)
    for arm_id, spec in reg.COVARIATE_ARMS.items():
        assert not internal_code_pattern.match(arm_id), arm_id
        assert not internal_code_pattern.match(spec.public_name), spec.public_name


def test_registry_public_names_match_released_table_where_present():
    released = pd.read_csv(REPO_ROOT_RESULTS)
    released_names = set(released["covariate_public_name"].dropna())
    registry_names = {spec.public_name for spec in reg.COVARIATE_ARMS.values()}
    # Every combined-arm name in the registry that isn't the standalone
    # satellite-proxy arm (reported in a different frozen table) must match
    # the released covariate-mechanism table exactly, not approximately.
    for name in registry_names - {"No covariate (target-only baseline)", "Satellite chlorophyll proxy"}:
        assert name in released_names, name


def test_no_forbidden_target_history_column_in_any_arm():
    """No arm may include the in-situ target or its lags -- only external/
    satellite covariates are admissible TS-ICL covariate inputs."""
    forbidden = {"chl_mean", "chl_mean_lag1", "chl_mean_lag3", "chl_mean_lag7"}
    for arm_id, spec in reg.COVARIATE_ARMS.items():
        assert forbidden.isdisjoint(spec.columns), arm_id


def test_satellite_proxy_columns_are_the_two_documented_columns():
    assert reg.COVARIATE_ARMS["satellite_proxy"].columns == ["chl_cons_w3x3_mean", "chl_anom_log10_doy"]


def test_current_transport_arms_flagged_as_requiring_extended_table():
    assert reg.COVARIATE_ARMS["current_transport_only"].requires_extended_table is True
    assert reg.COVARIATE_ARMS["proxy_plus_current_transport"].requires_extended_table is True
    assert reg.COVARIATE_ARMS["satellite_proxy"].requires_extended_table is False


def test_target_only_arm_has_no_columns():
    assert reg.COVARIATE_ARMS["target_only"].columns == []


def test_placebo_transform_wrong_lag_is_a_pure_roll():
    block = np.arange(20).reshape(10, 2).astype(np.float32)
    dates = np.array(["2020-01-01"], dtype="datetime64[D]") + np.arange(10)
    out = reg.apply_placebo_transform(block, dates, "wrong_lag")
    np.testing.assert_array_equal(out, np.roll(block, shift=90, axis=0))


def test_placebo_transform_permuted_is_deterministic_under_fixed_seed():
    block = np.arange(20).reshape(10, 2).astype(np.float32)
    dates = np.array(["2020-01-01"], dtype="datetime64[D]") + np.arange(10)
    out1 = reg.apply_placebo_transform(block, dates, "permuted", seed=7)
    out2 = reg.apply_placebo_transform(block, dates, "permuted", seed=7)
    np.testing.assert_array_equal(out1, out2)


def test_placebo_transform_permuted_actually_reorders_rows():
    block = np.arange(40).reshape(20, 2).astype(np.float32)
    dates = np.array(["2020-01-01"], dtype="datetime64[D]") + np.arange(20)
    out = reg.apply_placebo_transform(block, dates, "permuted", seed=1)
    assert not np.array_equal(out, block)
    # Same multiset of rows, different order -- a permutation, not a corruption.
    assert set(map(tuple, out.tolist())) == set(map(tuple, block.tolist()))


def test_placebo_transform_rejects_unknown_transform():
    block = np.zeros((5, 2), dtype=np.float32)
    dates = np.array(["2020-01-01"], dtype="datetime64[D]") + np.arange(5)
    with pytest.raises(ValueError):
        reg.apply_placebo_transform(block, dates, "bogus")


def test_build_engineered_products_computes_expected_values():
    df = pd.DataFrame({
        "plv_solar_wm2": [2.0, 3.0], "plv_upwelling_ms": [4.0, 5.0],
        "mur_sst_cooling_7d_degC": [1.0, 2.0],
    })
    out = reg.build_engineered_products(df)
    np.testing.assert_allclose(out["solar_x_upwelling_product"], [8.0, 15.0])
    np.testing.assert_allclose(out["upwelling_x_cooling_product"], [4.0, 10.0])


def test_placebo_eligible_arms_are_all_registered():
    for arm_id in reg.PLACEBO_ELIGIBLE_ARMS:
        assert arm_id in reg.COVARIATE_ARMS
