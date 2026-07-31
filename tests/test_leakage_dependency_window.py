"""Tests for coastal_gap_reconstruction.leakage's dependency-window exclusion."""
from __future__ import annotations

import pandas as pd
import pytest

from coastal_gap_reconstruction.leakage import dependency_window, exclude_dependency_window


def test_dependency_window_extends_lookback_and_lookahead() -> None:
    start = pd.Timestamp("2020-06-15")
    excl_start, excl_end = dependency_window(start, gap_length=5, max_lookback=7, max_lookahead=3)
    assert excl_start == pd.Timestamp("2020-06-08")  # 7 days before start
    assert excl_end == pd.Timestamp("2020-06-22")  # gap end (06-19) + 3 days


def test_dependency_window_rejects_negative_windows() -> None:
    start = pd.Timestamp("2020-06-15")
    with pytest.raises(ValueError):
        dependency_window(start, gap_length=5, max_lookback=-1)


def test_exclude_dependency_window_removes_every_row_inside_the_window() -> None:
    dates = pd.date_range("2020-06-01", "2020-06-30", freq="D")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)

    start = pd.Timestamp("2020-06-15")
    gap_length = 5  # hides 06-15..06-19
    result = exclude_dependency_window(df, start, gap_length, max_lookback=7, max_lookahead=3)

    excl_start, excl_end = dependency_window(start, gap_length, 7, 3)
    excluded_dates = pd.date_range(excl_start, excl_end, freq="D")
    assert not any(d in result.index for d in excluded_dates)
    assert len(result) == len(df) - len(excluded_dates)


def test_exclude_dependency_window_keeps_rows_outside_the_window() -> None:
    dates = pd.date_range("2020-01-01", "2020-01-10", freq="D")
    df = pd.DataFrame({"value": range(len(dates))}, index=dates)

    # Gap far from the edges of this tiny frame -- everything should survive.
    start = pd.Timestamp("2020-06-15")
    result = exclude_dependency_window(df, start, gap_length=1, max_lookback=1)
    assert len(result) == len(df)
