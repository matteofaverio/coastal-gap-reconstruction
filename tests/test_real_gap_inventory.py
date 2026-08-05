"""Tests for `experiments.chlorophyll.real_gap_inventory` -- exact
reproduction of the released 128-row real-gap inventory, and unit tests on
synthetic data for the detection logic itself."""

from __future__ import annotations

import pandas as pd
import pytest

from experiments.chlorophyll import real_gap_contract as rgc
from experiments.chlorophyll import real_gap_inventory as ri


@pytest.fixture(scope="module")
def target_df():
    df = pd.read_csv(rgc.DAILY_TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()
    return df


@pytest.fixture(scope="module")
def released_inventory():
    return pd.read_csv(rgc.REAL_GAP_INVENTORY_PATH)


def test_reproduces_128_real_gaps(target_df):
    computed = ri.detect_real_gaps(target_df)
    assert len(computed) == 128


def test_gap_ids_match_released_exactly(target_df, released_inventory):
    computed = ri.detect_real_gaps(target_df)
    assert set(computed["gap_id"]) == set(released_inventory["gap_id"])


def test_every_column_matches_released_exactly(target_df, released_inventory):
    computed = ri.detect_real_gaps(target_df).sort_values("gap_id").reset_index(drop=True)
    released = released_inventory.sort_values("gap_id").reset_index(drop=True)
    computed["start_date"] = computed["start_date"].astype(str)
    computed["end_date"] = computed["end_date"].astype(str)
    released = released.copy()
    released["start_date"] = released["start_date"].astype(str)
    released["end_date"] = released["end_date"].astype(str)
    released["notes"] = released["notes"].fillna("")
    for col in ri.INVENTORY_COLUMNS:
        assert (computed[col].astype(str) == released[col].astype(str)).all(), f"mismatch in {col}"


def test_256_day_gap_is_detected_with_correct_notes(target_df):
    computed = ri.detect_real_gaps(target_df)
    row = computed[computed["length_days"] == 256]
    assert len(row) == 1
    assert "2020" in row.iloc[0]["notes"]
    # Verified against the actual released inventory row (gap_id
    # REAL_L091_20200211) -- not CLAUDE.md's rounded "2020-02-12" framing.
    assert str(row.iloc[0]["start_date"]) == "2020-02-11"
    assert str(row.iloc[0]["end_date"]) == "2020-10-23"


def test_never_reads_any_candidate_prediction_file():
    """Detection must be a pure function of the daily target table --
    verified structurally: detect_real_gaps takes only target_df, no
    candidate-file path parameter exists in its signature."""
    import inspect
    sig = inspect.signature(ri.detect_real_gaps)
    assert list(sig.parameters) == ["target_df"]


def test_gap_class_boundaries():
    assert ri.gap_class(7) == "short"
    assert ri.gap_class(8) == "medium"
    assert ri.gap_class(30) == "medium"
    assert ri.gap_class(31) == "long"
    assert ri.gap_class(90) == "long"
    assert ri.gap_class(91) == "very_long"


def test_detect_real_gaps_on_synthetic_data_finds_one_gap():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({rgc.ELIGIBLE_COLUMN: True}, index=dates)
    df.loc[dates[5:10], rgc.ELIGIBLE_COLUMN] = False  # a 5-day gap
    computed = ri.detect_real_gaps(df)
    assert len(computed) == 1
    assert computed.iloc[0]["length_days"] == 5
    assert computed.iloc[0]["gap_class"] == "short"
    assert computed.iloc[0]["interpolation_admissible"] == True  # noqa: E712 -- explicit bool check


def test_detect_real_gaps_open_ended_gap_at_series_end():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({rgc.ELIGIBLE_COLUMN: True}, index=dates)
    df.loc[dates[15:], rgc.ELIGIBLE_COLUMN] = False  # runs to the end
    computed = ri.detect_real_gaps(df)
    assert len(computed) == 1
    assert computed.iloc[0]["gap_id"].startswith("REAL_OPEN_")
    assert computed.iloc[0]["nearest_val_lengths"] == "open"
    assert computed.iloc[0]["extrapolation_beyond_validation"] == "open_ended"


def test_no_gaps_detected_when_fully_eligible():
    dates = pd.date_range("2020-01-01", periods=20, freq="D")
    df = pd.DataFrame({rgc.ELIGIBLE_COLUMN: True}, index=dates)
    computed = ri.detect_real_gaps(df)
    assert len(computed) == 0
