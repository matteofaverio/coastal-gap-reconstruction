"""Tests for the gap-clustered paired bootstrap comparison mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coastal_gap_reconstruction import paired_statistics as ps


def _synthetic_day_level(n_gaps=30, days_per_gap=3, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_gaps):
        gap_id = f"g{i}"
        base_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=10 * i)
        err_a = rng.normal(0.2, 0.05, days_per_gap).clip(min=0)
        err_b = rng.normal(0.3, 0.05, days_per_gap).clip(min=0)  # b is worse
        for d in range(days_per_gap):
            date = base_date + pd.Timedelta(days=d)
            rows.append({"method_id": "method_a", "gap_id": gap_id, "day": d, "date": date,
                         "absolute_error_log10": err_a[d]})
            rows.append({"method_id": "method_b", "gap_id": gap_id, "day": d, "date": date,
                         "absolute_error_log10": err_b[d]})
    return pd.DataFrame(rows)


def test_gap_level_metrics_day_weighted_vs_gap_weighted_differ_for_uneven_gaps():
    day_errors = {"g1": np.array([0.1, 0.1, 0.1, 0.1, 0.1]), "g2": np.array([1.0])}
    day_weighted = ps.gap_level_metrics(day_errors, np.array(["g1", "g2"]))["day_weighted_mae"]
    gap_weighted = ps.gap_level_metrics(day_errors, np.array(["g1", "g2"]))["gap_weighted_mae"]
    assert day_weighted != pytest.approx(gap_weighted)
    # day-weighted: (5*0.1 + 1*1.0)/6 = 0.25; gap-weighted: (0.1+1.0)/2 = 0.55
    assert day_weighted == pytest.approx(0.25)
    assert gap_weighted == pytest.approx(0.55)


def test_gap_cluster_bootstrap_ci_deterministic_under_fixed_seed():
    values = np.array([0.1, 0.2, 0.15, 0.3, 0.25])
    r1 = ps.gap_cluster_bootstrap_ci(values, n_replicates=200, seed=42)
    r2 = ps.gap_cluster_bootstrap_ci(values, n_replicates=200, seed=42)
    assert r1 == r2


def test_gap_cluster_bootstrap_ci_handles_empty_input():
    result = ps.gap_cluster_bootstrap_ci(np.array([]), n_replicates=100, seed=1)
    assert all(v != v for v in result)  # all NaN


def test_bootstrap_compare_returns_none_when_no_shared_gaps():
    df = pd.DataFrame([
        {"method_id": "a", "gap_id": "g1", "date": "2020-01-01", "absolute_error_log10": 0.1},
        {"method_id": "b", "gap_id": "g2", "date": "2020-01-01", "absolute_error_log10": 0.2},
    ])
    assert ps.bootstrap_compare("a", "b", df) is None


def test_bootstrap_compare_sign_convention_a_minus_b():
    """method_a has lower error than method_b by construction -- delta must
    be negative (a - b < 0), and the interpretation must be an improvement,
    not a degradation."""
    df = _synthetic_day_level(n_gaps=50, seed=0)
    result = ps.bootstrap_compare("method_a", "method_b", df, n_replicates=500, seed=1)
    assert result is not None
    assert result.metrics["day_weighted_mae"]["delta"] < 0
    assert result.interpretation == "significant_improvement"


def test_bootstrap_compare_is_deterministic_under_fixed_seed():
    df = _synthetic_day_level(n_gaps=40, seed=2)
    r1 = ps.bootstrap_compare("method_a", "method_b", df, n_replicates=300, seed=99)
    r2 = ps.bootstrap_compare("method_a", "method_b", df, n_replicates=300, seed=99)
    assert r1.metrics["day_weighted_mae"]["ci_lo"] == r2.metrics["day_weighted_mae"]["ci_lo"]
    assert r1.metrics["day_weighted_mae"]["ci_hi"] == r2.metrics["day_weighted_mae"]["ci_hi"]


def test_bootstrap_compare_insufficient_support_below_min_gaps():
    df = _synthetic_day_level(n_gaps=5, seed=3)
    result = ps.bootstrap_compare("method_a", "method_b", df, n_replicates=200, seed=1)
    assert result.interpretation == "insufficient_support"


def test_bootstrap_compare_respects_gap_ids_allowed_filter():
    df = _synthetic_day_level(n_gaps=30, seed=4)
    allowed = {"g0", "g1", "g2"}
    result = ps.bootstrap_compare("method_a", "method_b", df, gap_ids_allowed=allowed, n_replicates=100, seed=1)
    assert result.n_gaps == 3


def test_paired_comparison_result_to_flat_dict_has_expected_columns():
    df = _synthetic_day_level(n_gaps=30, seed=5)
    result = ps.bootstrap_compare("method_a", "method_b", df, n_replicates=100, seed=1)
    flat = result.to_flat_dict()
    for key in ("day_weighted_mae_a", "day_weighted_mae_b", "day_weighted_mae_delta",
                "day_weighted_mae_ci_lo", "day_weighted_mae_ci_hi", "n_gaps", "interpretation"):
        assert key in flat


def test_bootstrap_uses_gap_level_resampling_not_day_level():
    """A within-gap tamper (changing one day's error inside a gap without
    changing the gap-level mean) must not change the bootstrap CI -- proof
    the resampling unit is gap_id, not individual day rows."""
    rows = []
    for i in range(30):
        gap_id = f"g{i}"
        base_date = pd.Timestamp("2020-01-01") + pd.Timedelta(days=10 * i)
        base = 0.2
        for d in range(4):
            date = base_date + pd.Timedelta(days=d)
            rows.append({"method_id": "a", "gap_id": gap_id, "date": date, "absolute_error_log10": base})
            rows.append({"method_id": "b", "gap_id": gap_id, "date": date, "absolute_error_log10": base + 0.1})
    df = pd.DataFrame(rows)
    r1 = ps.bootstrap_compare("a", "b", df, n_replicates=500, seed=1)

    # Redistribute one gap's day errors (same gap-level mean, different
    # day-level values) -- gap-level bootstrap must be unaffected.
    df2 = df.copy()
    mask = (df2["method_id"] == "a") & (df2["gap_id"] == "g0")
    idx = df2[mask].index
    df2.loc[idx, "absolute_error_log10"] = [0.1, 0.1, 0.3, 0.3]  # same mean (0.2), different spread
    r2 = ps.bootstrap_compare("a", "b", df2, n_replicates=500, seed=1)

    assert r1.metrics["day_weighted_mae"]["delta"] == pytest.approx(r2.metrics["day_weighted_mae"]["delta"])


# ── Exact day-level pairing ────────────────────────────────────────────

def _two_gap_frame(dates_a: dict[str, list[str]], dates_b: dict[str, list[str]]) -> pd.DataFrame:
    """Build a minimal two-method day-level frame from {gap_id: [dates]} maps."""
    rows = []
    for gap_id, dates in dates_a.items():
        for i, d in enumerate(dates):
            rows.append({"method_id": "a", "gap_id": gap_id, "date": d, "absolute_error_log10": 0.1 + i})
    for gap_id, dates in dates_b.items():
        for i, d in enumerate(dates):
            rows.append({"method_id": "b", "gap_id": gap_id, "date": d, "absolute_error_log10": 0.2 + i})
    return pd.DataFrame(rows)


def test_strict_pairing_raises_on_different_dates_within_a_shared_gap():
    df = _two_gap_frame(
        {"g0": ["2020-01-01", "2020-01-02"]},
        {"g0": ["2020-01-01", "2020-01-03"]},
    )
    with pytest.raises(ValueError, match="date_support_mismatch"):
        ps.bootstrap_compare("a", "b", df, n_replicates=50, seed=1)


def test_strict_pairing_raises_when_one_method_is_missing_one_day():
    df = _two_gap_frame(
        {"g0": ["2020-01-01", "2020-01-02", "2020-01-03"]},
        {"g0": ["2020-01-01", "2020-01-02"]},
    )
    with pytest.raises(ValueError, match="date_support_mismatch"):
        ps.bootstrap_compare("a", "b", df, n_replicates=50, seed=1)


def test_strict_pairing_raises_on_duplicate_dates():
    df = _two_gap_frame(
        {"g0": ["2020-01-01", "2020-01-01"]},
        {"g0": ["2020-01-01", "2020-01-02"]},
    )
    with pytest.raises(ValueError, match="duplicate_date_rows"):
        ps.bootstrap_compare("a", "b", df, n_replicates=50, seed=1)


def test_strict_pairing_accepts_same_date_support_in_different_row_order():
    df = _two_gap_frame(
        {"g0": ["2020-01-02", "2020-01-01"]},
        {"g0": ["2020-01-01", "2020-01-02"]},
    )
    # Should not raise -- date *sets* match even though row order differs.
    result = ps.bootstrap_compare("a", "b", df, n_replicates=50, seed=1, pairing="intersection")
    assert result is not None
    assert result.paired_gap_count == 1
    assert result.paired_day_count == 2


def test_intersection_pairing_restricts_to_common_dates_and_records_exclusion():
    df = _two_gap_frame(
        {"g0": ["2020-01-01", "2020-01-02", "2020-01-03"]},
        {"g0": ["2020-01-01", "2020-01-02"]},
    )
    result = ps.bootstrap_compare("a", "b", df, n_replicates=50, seed=1, pairing="intersection")
    assert result is not None
    assert result.paired_gap_count == 1
    assert result.paired_day_count == 2  # only the common 2 days, not 3
    assert len(result.excluded_gaps) == 1
    assert result.excluded_gaps[0]["gap_id"] == "g0"
    assert result.excluded_gaps[0]["reason"] == "date_support_mismatch"


def test_paired_gap_and_day_counts_and_original_gap_counts_recorded():
    df = _synthetic_day_level(n_gaps=20, days_per_gap=3, seed=7)
    result = ps.bootstrap_compare("method_a", "method_b", df, n_replicates=50, seed=1)
    assert result.original_gap_count_a == 20
    assert result.original_gap_count_b == 20
    assert result.paired_gap_count == 20
    assert result.paired_day_count == 60
    assert result.excluded_gaps == []
