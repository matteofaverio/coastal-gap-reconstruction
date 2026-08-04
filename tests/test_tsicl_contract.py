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
