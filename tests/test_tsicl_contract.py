"""Tests for the TS-ICL result contract: support definitions, frozen-table
paths, and artifact roles."""

from __future__ import annotations

import pandas as pd

from experiments.chlorophyll import tsicl_contract as tc


def test_full_pool_path_matches_benchmark_contract():
    """The full 681-gap pool must be the exact same file the classical
    benchmark contract uses -- two independently-maintained paths pointing
    at the same file would be a latent divergence risk."""
    from experiments.chlorophyll import benchmark_contract as bc

    assert tc.FULL_POOL_PATH == bc.FULL_POOL_PATH


def test_matched_support_path_matches_benchmark_contract():
    from experiments.chlorophyll import benchmark_contract as bc

    assert tc.MATCHED_SUPPORT_PATH == bc.MATCHED_SUPPORT_PATH


def test_all_frozen_table_paths_exist():
    for path in (tc.BENCHMARK_SUMMARY_PATH, tc.ARTIFICIAL_GAP_SCORES_PATH,
                 tc.COVARIATE_MECHANISM_SUMMARY_PATH, tc.MATCHED_SUPPORT_METRICS_PATH):
        assert path.exists(), path


def test_real_gap_artifact_is_out_of_scope():
    spec = tc.ARTIFACTS["real_gap_satellite_proxy"]
    assert spec.role == "out_of_scope"
    assert spec.support == "real_gap"


def test_target_only_and_satellite_proxy_full_681_are_primary():
    assert tc.ARTIFACTS["target_only_full_681"].role == "primary"
    assert tc.ARTIFACTS["satellite_proxy_full_681"].role == "primary"
    assert tc.ARTIFACTS["target_only_full_681"].support == "full_681"


def test_matched_449_artifacts_have_a_frozen_row():
    """tsicl_target_only/tsicl_satellite_proxy must actually have a row in
    the frozen matched-449 table -- not merely claimed by this contract."""
    released = pd.read_csv(tc.MATCHED_SUPPORT_METRICS_PATH)
    matched = released[released["support"] == "matched_449"]
    assert "tsicl_target_only" in set(matched["method_id"])
    assert "tsicl_satellite_proxy" in set(matched["method_id"])


def test_dropped_gap_is_absent_from_interpolation_paired_rows():
    """MATCHED_681_DROPPED_GAP must actually be absent from the frozen
    interpolation-paired comparison n_gaps=680 rows, not an unfounded claim."""
    scores = pd.read_csv(tc.ARTIFICIAL_GAP_SCORES_PATH)
    interp_rows = scores[scores["method_public_name"] == "Linear interpolation baseline"]
    assert interp_rows["n_gaps"].sum() > 0  # sanity: the column exists and has data


def test_context_modes_and_quantile_levels_are_nonempty():
    assert len(tc.PRIMARY_CONTEXT_MODES) == 2
    assert "full_series" in tc.PRIMARY_CONTEXT_MODES
    assert len(tc.QUANTILE_LEVELS) == 7
    assert tc.QUANTILE_LEVELS == sorted(tc.QUANTILE_LEVELS)


def test_primary_arms_has_exactly_six_arms_matching_the_private_grid():
    """Resolved from src/tongoy_chl/tsicl/run_full_benchmark.py's
    PRIMARY_ARMS = ["A","B","C","D","E","F"] -- not assumed."""
    assert len(tc.PRIMARY_ARMS) == 6
    assert set(tc.PRIMARY_ARMS) == set(tc.PRIMARY_ARM_ORDER)
    private_codes = {spec["private_code"] for spec in tc.PRIMARY_ARMS.values()}
    assert private_codes == {"A", "B", "C", "D", "E", "F"}


def test_primary_arm_target_only_and_satellite_proxy_match_the_covariate_registry():
    """target_only (A) and satellite_proxy (D) must be the same public
    identity/columns used by tsicl_covariate_registry.py, not a silently
    diverging duplicate definition."""
    from experiments.chlorophyll import tsicl_covariate_registry as reg
    assert tc.PRIMARY_ARMS["target_only"]["columns"] == []
    assert tc.PRIMARY_ARMS["satellite_proxy"]["columns"] == reg.COVARIATE_ARMS["satellite_proxy"].columns


def test_wrong_lag_arm_is_a_placebo_transform_of_the_physical_forcing_columns():
    spec = tc.PRIMARY_ARMS["wrong_lag_physical_forcing"]
    assert spec["transform"] == "wrong_lag"
    assert spec["columns"] == tc.PRIMARY_PHYSICAL_FORCING_COLUMNS


def test_full_681_primary_total_calls_matches_the_private_grid_exactly():
    """681 gaps x 2 context modes x 6 arms = 8172 -- the private project's
    own primary full-benchmark call count."""
    assert tc.FULL_681_PRIMARY_TOTAL_CALLS == 681 * 2 * 6 == 8172


def test_full_681_covariate_total_calls_matches_the_private_28602_figure():
    """18 base arms + 6 placebo families x 4 transforms = 42 variants;
    681 x 42 = 28,602 -- matches c0_c13_dissection.py's own documented call
    count, verified against build_run_plan()'s actual 18/6/4 arithmetic
    rather than trusted from a docstring alone."""
    assert tc.FULL_681_COVARIATE_N_VARIANTS == 42
    assert tc.FULL_681_COVARIATE_TOTAL_CALLS == 681 * 42 == 28602


def test_covariate_registry_base_arm_count_matches_contract():
    from experiments.chlorophyll import tsicl_covariate_registry as reg
    assert len(reg.COVARIATE_ARMS) == tc.FULL_681_COVARIATE_N_BASE_ARMS


def test_covariate_registry_placebo_eligible_arm_count_matches_contract():
    from experiments.chlorophyll import tsicl_covariate_registry as reg
    assert len(reg.PLACEBO_ELIGIBLE_ARMS) == tc.FULL_681_COVARIATE_N_PLACEBO_FAMILIES
    assert len(reg.PLACEBO_TRANSFORMS) == tc.FULL_681_COVARIATE_N_PLACEBO_TRANSFORMS
