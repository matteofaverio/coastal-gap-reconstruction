"""Tests for the chlorophyll classical-benchmark contract.

Verifies the 449-gap matched support is exactly the intersection this package's
headline non-TS-ICL comparators share, and that the method registry matches the
frozen released result table's labels exactly (not approximately).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.chlorophyll import benchmark_contract as bc

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_matched_support_is_449_gaps():
    ids = bc.load_matched_support_gap_ids()
    assert len(ids) == bc.MATCHED_SUPPORT_N_GAPS == 449
    assert ids["gap_id"].is_unique


def test_matched_support_gap_lengths_match_released_counts():
    ids = bc.load_matched_support_gap_ids()
    counts = ids["gap_length"].value_counts().to_dict()
    assert counts == bc.MATCHED_SUPPORT_COUNTS_BY_LENGTH
    assert sorted(counts) == bc.MATCHED_SUPPORT_GAP_LENGTHS


def test_matched_support_day_rows_total():
    ids = bc.load_matched_support_gap_ids()
    assert int(ids["n_days"].sum()) == bc.MATCHED_SUPPORT_N_DAY_ROWS


def test_matched_support_is_subset_of_full_pool():
    full = bc.load_full_pool()
    matched_ids = set(bc.load_matched_support_gap_ids()["gap_id"])
    assert matched_ids.issubset(set(full["gap_id"]))


def test_load_matched_support_pool_has_all_18_full_pool_columns():
    full_cols = set(bc.load_full_pool().columns)
    matched = bc.load_matched_support_pool()
    assert set(matched.columns) == full_cols
    assert len(matched) == bc.MATCHED_SUPPORT_N_GAPS


def test_matched_support_pool_gap_length_distribution_matches_ids_table():
    pool = bc.load_matched_support_pool()
    ids = bc.load_matched_support_gap_ids()
    assert (
        pool["gap_length"].value_counts().sort_index().to_dict()
        == ids["gap_length"].value_counts().sort_index().to_dict()
    )


def test_gap_lengths_reexports_config():
    from experiments.chlorophyll import _config

    assert bc.GAP_LENGTHS == _config.GAP_LENGTHS


@pytest.mark.parametrize(
    "method_id",
    [
        "canonical_interpolation",
        "gp_m1",
        "ext_tabular_extratrees",
        "ext_tabular_hgb",
        "external_only_extratrees",
        "external_only_hgb",
        "tier_ch_deployed",
        "engineered_hybrid",
    ],
)
def test_classical_benchmark_methods_registered(method_id):
    spec = bc.METHODS[method_id]
    assert spec.role == "classical_benchmark"
    assert spec.method_id == method_id


def test_matched_support_manifest_uses_the_correct_event_flag_name():
    """The matched-support manifest's event column must be named
    `is_high_chl_event` (the actual, 90th-percentile-threshold definition),
    not `event_p85` -- a private overnight script's naming artifact that
    never corresponded to a distinct 85th-percentile computation. See
    `benchmark_contract.py`'s "Event flag" docstring section."""
    ids = bc.load_matched_support_gap_ids()
    assert "is_high_chl_event" in ids.columns
    assert "event_p85" not in ids.columns
    assert ids["is_high_chl_event"].dtype == bool


def test_frozen_vs_new_methods_have_correct_support_status():
    """`ext_tabular_extratrees`/`ext_tabular_hgb`/`tier_ch_deployed`/
    `canonical_interpolation`/`gp_m1` have a row in the frozen released
    matched-449 table; `external_only_extratrees`/`external_only_hgb` (the
    plain external-only protocol) and `engineered_hybrid` (run on matched
    support only as a new consistency check, never released at this support)
    do not."""
    frozen = {"canonical_interpolation", "gp_m1", "ext_tabular_extratrees",
              "ext_tabular_hgb", "tier_ch_deployed"}
    new = {"external_only_extratrees", "external_only_hgb", "engineered_hybrid"}
    for method_id in frozen:
        assert bc.METHODS[method_id].support_status == "frozen_matched_449", method_id
    for method_id in new:
        assert bc.METHODS[method_id].support_status == "new_evaluation_on_matched_449", method_id


def test_new_evaluation_methods_absent_from_frozen_matched_metrics_table():
    """Confirms, against the actual frozen CSV (not just the registry's own
    claim about itself), that `engineered_hybrid` and the plain
    external-only protocol have no row in the released matched-449 table --
    i.e. `support_status="new_evaluation_on_matched_449"` is not an
    unfounded label."""
    released = pd.read_csv(bc.MATCHED_SUPPORT_METRICS_PATH)
    released_matched_ids = set(released[released["support"] == "matched_449"]["method_id"])
    for method_id in ("engineered_hybrid", "external_only_extratrees", "external_only_hgb"):
        assert method_id not in released_matched_ids, method_id


def test_method_public_names_match_frozen_released_table():
    """Every classical_benchmark method's public_name must match the exact label
    used in a frozen released result table -- this is the string a figure/table
    would key on, so drift here is a silent mislabeling risk, not just cosmetic.

    `engineered_hybrid` is not scored on the 449-gap matched support in the
    private project's own released tables (its Kalman-routed L=4-29 segment was
    only ever evaluated against the full 681-gap pool) -- checked against
    `chlorophyll_benchmark_summary.csv` instead, the table it actually appears
    in.
    """
    matched_names = set(
        pd.read_csv(bc.MATCHED_SUPPORT_METRICS_PATH)
        .pipe(lambda df: df[df["support"] == "matched_449"])["method_label"]
    )
    full_pool_names = set(
        pd.read_csv(bc.RESULTS_PUBLIC_DIR / "chlorophyll_benchmark_summary.csv")["method_public_name"]
    )
    for spec in bc.METHODS.values():
        if spec.role != "classical_benchmark":
            continue
        if spec.support_status != "frozen_matched_449" and spec.method_id != "engineered_hybrid":
            # New/diagnostic methods (e.g. the plain external-only protocol)
            # have no frozen row to match by construction -- see
            # MethodSpec.support_status. `engineered_hybrid` is the one
            # exception: it has no frozen *matched-449* row (support_status
            # reflects that), but its label is still checked against the
            # full-681-pool table it does appear in, per this test's docstring.
            continue
        assert spec.public_name in matched_names or spec.public_name in full_pool_names, (
            f"{spec.method_id}: {spec.public_name!r} not found in any frozen "
            f"released result table"
        )


def test_frozen_result_tables_exist():
    assert bc.MATCHED_SUPPORT_METRICS_PATH.exists()
    assert bc.MATCHED_SUPPORT_BY_LENGTH_PATH.exists()
    assert bc.FULL_POOL_PATH.exists()
    assert bc.MATCHED_SUPPORT_PATH.exists()
