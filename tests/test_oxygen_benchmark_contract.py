"""Tests pinning `experiments.oxygen.benchmark_contract` against the real
released public oxygen CSVs -- every count/role in the contract must match
the actual data, not merely be asserted."""

from __future__ import annotations

import pandas as pd

from experiments.oxygen import benchmark_contract as bc


def _pool():
    return pd.read_csv(bc.VALIDATION_GAPS_PATH)


def test_total_gap_count_matches_released_pool():
    pool = _pool()
    assert len(pool) == bc.TOTAL_N_GAPS == 412


def test_primary_and_exploratory_counts_match_released_support_role_column():
    pool = _pool()
    counts = pool["support_role"].value_counts()
    assert counts[bc.SUPPORT_ROLE_PRIMARY] == bc.PRIMARY_N_GAPS == 406
    assert counts[bc.SUPPORT_ROLE_EXPLORATORY] == bc.EXPLORATORY_N_GAPS == 6


def test_per_length_counts_match_released_pool():
    pool = _pool()
    by_length = pool["gap_length"].value_counts().to_dict()
    for length, expected in bc.ALL_GAP_LENGTHS.items():
        assert by_length.get(length) == expected, f"L={length}: expected {expected}, got {by_length.get(length)}"


def test_support_role_function_matches_released_column_for_every_row():
    pool = _pool()
    computed = pool["gap_length"].map(bc.support_role)
    assert (computed == pool["support_role"]).all()


def test_primary_hidden_days_matches_sum_of_length_times_count():
    assert bc.PRIMARY_N_HIDDEN_DAYS == 2912
    assert bc.PRIMARY_N_HIDDEN_DAYS == sum(L * n for L, n in bc.PRIMARY_GAP_LENGTHS.items())


def test_daily_target_has_the_contract_columns():
    daily = pd.read_csv(bc.DAILY_TARGET_PATH)
    assert bc.TARGET_COLUMN in daily.columns
    assert bc.ELIGIBLE_COLUMN in daily.columns
    assert bc.DATE_COLUMN in daily.columns
    assert (daily["source_variable"] == bc.SENSOR_CODE).all()
    assert (daily["unit"] == bc.TARGET_UNIT).all()


def test_eligibility_threshold_matches_18_valid_hours():
    daily = pd.read_csv(bc.DAILY_TARGET_PATH)
    recomputed = daily["valid_hours"] >= bc.MIN_VALID_HOURS
    assert (recomputed == daily[bc.ELIGIBLE_COLUMN]).all()


def test_no_day_night_rule_declared():
    assert bc.DAY_NIGHT_RULE is None


def test_target_transform_is_identity_not_log():
    assert bc.TARGET_TRANSFORM == "identity"


def test_forbidden_predictor_substrings_do_not_include_local_btg():
    for bad in bc.FORBIDDEN_PREDICTOR_SUBSTRINGS:
        assert "BTGTA" not in bad
        assert "BTGPA" not in bad


def test_local_btg_diagnostic_features_file_matches_contract_role():
    assert bc.LOCAL_BTG_TEMP_PRESSURE_ROLE == "diagnostic_arm_only"
    df = pd.read_csv(bc.LOCAL_BTG_DIAGNOSTIC_FEATURES_PATH)
    assert "btg_water_temp_daily_mean" in df.columns
    assert "btg_pressure_daily_mean" in df.columns


def test_all_six_method_statuses_defined():
    expected = {
        "frozen_primary_benchmark", "frozen_exploratory_extended", "executable_bounded_validation",
        "exploratory_ablation", "same_station_diagnostic", "new_consistency_evaluation",
    }
    assert set(bc.METHOD_STATUSES) == expected


def test_tsicl_arm_registry_has_five_audited_original_arms():
    assert len(bc.TSICL_AUDITED_ORIGINAL_ARMS) == 5
    assert bc.TSICL_BEST_ARM in bc.TSICL_AUDITED_ORIGINAL_ARMS
    assert len(bc.TSICL_FAMILY_ABLATION_ARMS) == 4


def test_frozen_result_tables_exist_and_are_readable():
    for path in (bc.BENCHMARK_BY_LENGTH_PATH, bc.PAIRED_DELTAS_VS_TSICL_PATH,
                 bc.TAIL_PERSISTENCE_METRICS_PATH, bc.TAIL_QUANTILE_BAND_METRICS_PATH):
        df = pd.read_csv(path)
        assert len(df) > 0


def test_shared_feature_files_exist():
    assert bc.CHLOROPHYLL_BASE_FEATURES_PATH.exists()
    assert bc.CURRENT_KINEMATIC_EXTENSION_PATH.exists()
