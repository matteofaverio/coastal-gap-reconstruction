"""Tests for `experiments.oxygen.feature_registry`."""

from __future__ import annotations

import pytest

from experiments.oxygen import benchmark_contract as bc
from experiments.oxygen import feature_registry as fr


def test_all_four_arms_load_without_error():
    for arm in fr.ARM_NAMES:
        df = fr.get_feature_arm(arm)
        assert len(df) > 0
        assert df.shape[1] > 0


def test_unknown_arm_raises():
    with pytest.raises(ValueError, match="Unknown oxygen feature arm"):
        fr.get_feature_arm("not_a_real_arm")


def test_external_physical_core_excludes_currents_and_chlorophyll_and_local_btg():
    df = fr.get_feature_arm("external_physical_core")
    for col in df.columns:
        assert not col.startswith(fr.CURRENT_COLS_PREFIXES)
        assert col not in fr.SATELLITE_CHL_COLS
        assert "btg_water_temp" not in col
        assert "btg_pressure" not in col


def test_external_physical_plus_currents_includes_currents_but_not_chlorophyll():
    df = fr.get_feature_arm("external_physical_plus_currents")
    assert any(c.startswith(fr.CURRENT_COLS_PREFIXES) for c in df.columns)
    for col in df.columns:
        assert col not in fr.SATELLITE_CHL_COLS


def test_external_all_available_includes_currents_and_satellite_chlorophyll():
    df = fr.get_feature_arm("external_all_available")
    assert any(c.startswith(fr.CURRENT_COLS_PREFIXES) for c in df.columns)
    assert any(c in fr.SATELLITE_CHL_COLS for c in df.columns)


def test_local_btg_arm_includes_diagnostic_columns_and_core_but_not_currents():
    df = fr.get_feature_arm("local_btg_temp_pressure_diagnostic")
    assert "btg_water_temp_daily_mean" in df.columns
    assert "btg_pressure_daily_mean" in df.columns
    assert not any(c.startswith(fr.CURRENT_COLS_PREFIXES) for c in df.columns)


def test_no_forbidden_predictor_substring_in_any_arm():
    for arm in fr.ARM_NAMES:
        df = fr.get_feature_arm(arm)
        # Only chl_mean/BTGOXD*/BTGSAL/BTGTUR/BTGCND are hard-forbidden; satellite
        # chl columns are permitted only in external_all_available (checked above).
        for col in df.columns:
            assert "chl_mean" not in col
            assert "BTGOXD" not in col
            assert "BTGSAL" not in col
            assert "BTGTUR" not in col
            assert "BTGCND" not in col


def test_arm_is_diagnostic_matches_contract():
    assert fr.arm_is_diagnostic("external_all_available") is True
    assert fr.arm_is_diagnostic("local_btg_temp_pressure_diagnostic") is True
    assert fr.arm_is_diagnostic("external_physical_core") is False
    assert fr.arm_is_diagnostic("external_physical_plus_currents") is False


def test_assert_columns_safe_raises_on_forbidden_substring():
    with pytest.raises(ValueError, match="Forbidden substring"):
        fr.assert_columns_safe(["chl_mean_something"], "test_arm")


def test_assert_columns_safe_allows_satellite_chl_columns():
    # Satellite chl columns are not in the classical registry's forbidden list.
    fr.assert_columns_safe(["chl_cons_log10"], "external_all_available")


def test_local_btg_diagnostic_role_matches_contract():
    assert bc.LOCAL_BTG_TEMP_PRESSURE_ROLE == "diagnostic_arm_only"
    assert bc.SATELLITE_CHLOROPHYLL_ROLE == "exploratory_ablation_only"
