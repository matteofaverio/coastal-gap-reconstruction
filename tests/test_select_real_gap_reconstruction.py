"""Tests for `experiments.chlorophyll.select_real_gap_reconstruction` --
exact reproduction of the released `engineered_hybrid_method` column for
all 128 real gaps, including the one context-constrained exclusion."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.chlorophyll import real_gap_contract as rgc
from experiments.chlorophyll import real_gap_inventory as ri
from experiments.chlorophyll import select_real_gap_reconstruction as sr


@pytest.fixture(scope="module")
def routed_inventory():
    target_df = pd.read_csv(rgc.DAILY_TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()
    inv = ri.detect_real_gaps(target_df)
    return sr.route_real_gaps(inv)


@pytest.fixture(scope="module")
def released_candidates():
    return pd.read_csv(rgc.CANDIDATE_OUTPUTS_GAP_LEVEL_PATH)


def test_method_for_length_rule_d_boundaries():
    assert sr.method_for_length(1) == sr.METHOD_GP
    assert sr.method_for_length(3) == sr.METHOD_GP
    assert sr.method_for_length(4) == sr.METHOD_KALMAN
    assert sr.method_for_length(29) == sr.METHOD_KALMAN
    assert sr.method_for_length(30) == sr.METHOD_GAP_EDGE_RESIDUAL
    assert sr.method_for_length(256) == sr.METHOD_GAP_EDGE_RESIDUAL


def test_method_for_length_rule_c_boundaries():
    assert sr.method_for_length(3, rule=sr.RULE_C) == sr.METHOD_GP
    assert sr.method_for_length(4, rule=sr.RULE_C) == sr.METHOD_KALMAN
    assert sr.method_for_length(13, rule=sr.RULE_C) == sr.METHOD_KALMAN
    assert sr.method_for_length(14, rule=sr.RULE_C) == sr.METHOD_GAP_EDGE_RESIDUAL


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="Unknown rule"):
        sr.method_for_length(10, rule="Z")


def test_routing_reproduces_all_127_assigned_methods_exactly(routed_inventory, released_candidates):
    merged = routed_inventory.merge(
        released_candidates[["gap_id", "engineered_hybrid_method"]], on="gap_id", how="left",
    )
    n_checked, n_mismatched = 0, 0
    for _, row in merged.iterrows():
        released_val = row["engineered_hybrid_method"]
        computed_val = row["assigned_method"]
        if pd.isna(released_val):
            assert pd.isna(computed_val), f"{row['gap_id']}: released excluded but computed assigned {computed_val!r}"
        else:
            n_checked += 1
            if released_val != computed_val:
                n_mismatched += 1
    assert n_checked == 127
    assert n_mismatched == 0


def test_exactly_one_gap_excluded_for_missing_post_edge_context(routed_inventory):
    excluded = routed_inventory[routed_inventory["assigned_method"].isna()]
    assert len(excluded) == 1
    assert excluded.iloc[0]["gap_id"] == "REAL_OPEN_20260515"
    assert excluded.iloc[0]["post_edge_available"] == False  # noqa: E712


def test_pre_edge_unavailable_alone_does_not_exclude_a_gap(routed_inventory):
    """The one gap with pre_edge_available=False but post_edge_available=True
    (REAL_L010_20150701, near the start of the series) must still receive a
    routed method -- this is the exact discriminator this module's docstring
    documents, verified structurally here, not just by the full-inventory
    reproduction test above."""
    row = routed_inventory[routed_inventory["gap_id"] == "REAL_L010_20150701"]
    assert len(row) == 1
    assert row.iloc[0]["pre_edge_available"] == False  # noqa: E712
    assert row.iloc[0]["post_edge_available"] == True  # noqa: E712
    assert row.iloc[0]["assigned_method"] == sr.METHOD_KALMAN


def test_256_day_scenario_gets_gap_edge_residual_method(routed_inventory):
    row = routed_inventory[routed_inventory["length_days"] == 256]
    assert row.iloc[0]["assigned_method"] == sr.METHOD_GAP_EDGE_RESIDUAL


def test_no_model_is_fit_route_real_gaps_is_a_pure_lookup():
    """Structural check: route_real_gaps's only inputs are the inventory
    DataFrame and a rule name -- no model object, no feature table, no
    checkpoint path can be passed, so it cannot be fitting anything."""
    import inspect
    sig = inspect.signature(sr.route_real_gaps)
    assert list(sig.parameters) == ["inventory", "rule"]
