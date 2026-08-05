"""Tests for `experiments.oxygen.tail_diagnostics`, pinned against the real
released band edges/eligible-day count (p10=3.776, p25=5.099, p50=6.429,
p75=7.436, p90=8.303, n_eligible=2880)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coastal_gap_reconstruction.data_loading import load_daily_target
from experiments.oxygen import benchmark_contract as bc
from experiments.oxygen import tail_diagnostics as td


@pytest.fixture(scope="module")
def target_df():
    df = load_daily_target(bc.DAILY_TARGET_PATH)
    df.index = pd.to_datetime(df.index)
    return df


def test_quantile_edges_match_the_frozen_released_values(target_df):
    edges = td.compute_quantile_edges(target_df)
    assert edges["p10"] == pytest.approx(3.7758, abs=1e-3)
    assert edges["p25"] == pytest.approx(5.0991, abs=1e-3)
    assert edges["p50"] == pytest.approx(6.4287, abs=1e-3)
    assert edges["p75"] == pytest.approx(7.4356, abs=1e-3)
    assert edges["p90"] == pytest.approx(8.3030, abs=1e-3)


def test_eligible_day_population_matches_frozen_count(target_df):
    n_eligible = int(target_df[bc.ELIGIBLE_COLUMN].sum())
    assert n_eligible == 2880


def test_classify_quantile_bin_boundaries():
    edges = {"p10": 4.0, "p25": 5.0, "p50": 6.0, "p75": 7.0, "p90": 8.0}
    values = np.array([1.0, 4.5, 5.5, 6.5, 7.5, 9.0])
    bins = td.classify_quantile_bin(values, edges)
    assert list(bins) == ["below_p10", "p10_to_p25", "p25_to_p50", "p50_to_p75", "p75_to_p90", "above_p90"]


def test_run_category_isolated_short_sustained_boundaries():
    assert td._run_category(1) == "isolated_tail_day"
    assert td._run_category(2) == "short_tail_run"
    assert td._run_category(6) == "short_tail_run"
    assert td._run_category(7) == "sustained_tail_run"
    assert td._run_category(30) == "sustained_tail_run"


def test_runs_from_boolean_breaks_on_calendar_date_gap():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05"])
    flags = np.array([True, True, True, True])
    runs = td._runs_from_boolean(dates, flags)
    # date gap between 01-02 and 01-04 (missing 01-03) must break the run
    assert len(runs) == 2
    assert runs[0][2] == 2  # 01-01, 01-02
    assert runs[1][2] == 2  # 01-04, 01-05


def test_runs_from_boolean_single_isolated_day():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
    flags = np.array([False, True, False])
    runs = td._runs_from_boolean(dates, flags)
    assert len(runs) == 1
    assert runs[0][2] == 1


def test_tail_persistence_totals_are_plausible_and_stable(target_df):
    runs = td.classify_tail_persistence(target_df)
    counts = runs.groupby(["quantile_bin", "run_category"])["date"].nunique()
    # Values pinned from a live run against the real released daily target --
    # regression guard, not an independently re-derived ground truth.
    assert counts[("below_p10", "isolated_tail_day")] == 22
    assert counts[("above_p90", "isolated_tail_day")] == 40
    assert set(runs["run_category"].unique()) <= set(td.RUN_CATEGORY_ORDER)
    assert set(runs["quantile_bin"].unique()) == {"below_p10", "above_p90"}
