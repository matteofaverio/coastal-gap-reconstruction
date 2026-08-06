"""Tests for the oxygen local BTG water-temperature/pressure diagnostic feature
table, including a maintainer-only comparison against the private authoritative
output (skipped in CI, where the private merged hourly files do not exist).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from experiments.oxygen.local_btg_diagnostic_features import (
    build_oxygen_local_btg_diagnostic_features,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RELEASED_TABLE = REPO_ROOT / "data" / "oxygen" / "oxygen_local_btg_diagnostic_features.csv"
OXYGEN_TARGET = REPO_ROOT / "data" / "oxygen" / "oxygen_daily_target.csv"

EXPECTED_COLUMNS = [
    "date",
    "btg_water_temp_daily_mean", "btg_water_temp_valid_hours", "btg_water_temp_available",
    "btg_pressure_daily_mean", "btg_pressure_valid_hours", "btg_pressure_available",
]


def test_released_table_has_expected_schema() -> None:
    df = pd.read_csv(RELEASED_TABLE)
    assert list(df.columns) == EXPECTED_COLUMNS


def test_released_table_does_not_include_unused_btg_variables() -> None:
    """Only water temperature and pressure -- no salinity, conductivity,
    turbidity, or chlorophyll columns, per the diagnostic arm's actual column
    usage (`local_btg_temp_pressure_diagnostic` uses only these two)."""
    df = pd.read_csv(RELEASED_TABLE)
    forbidden_substrings = ("sal", "cond", "turb", "chl", "oxygen", "OXD", "OXSATPC")
    for col in df.columns:
        for bad in forbidden_substrings:
            assert bad.lower() not in col.lower(), f"unexpected column {col!r} in diagnostic table"


def test_released_table_date_coverage_matches_oxygen_target() -> None:
    df = pd.read_csv(RELEASED_TABLE, parse_dates=["date"])
    target = pd.read_csv(OXYGEN_TARGET, parse_dates=["date"])
    assert list(df["date"]) == list(target["date"])


def test_available_flag_matches_valid_hours() -> None:
    df = pd.read_csv(RELEASED_TABLE)
    assert (df["btg_water_temp_available"] == (df["btg_water_temp_valid_hours"] > 0)).all()
    assert (df["btg_pressure_available"] == (df["btg_pressure_valid_hours"] > 0)).all()


def test_daily_mean_is_nan_exactly_when_unavailable() -> None:
    df = pd.read_csv(RELEASED_TABLE)
    assert (df.loc[~df["btg_water_temp_available"], "btg_water_temp_daily_mean"].isna()).all()
    assert (df.loc[~df["btg_pressure_available"], "btg_pressure_daily_mean"].isna()).all()


@pytest.mark.skipif(
    os.environ.get("RUN_PRIVATE_SNAPSHOT_COMPARISON") != "1",
    reason=(
        "Compares against the private merged hourly BTGTA/BTGPA files, which are "
        "not part of this repository. Set RUN_PRIVATE_SNAPSHOT_COMPARISON=1, "
        "PRIVATE_BTG_TA_PATH, and PRIVATE_BTG_PA_PATH to run it."
    ),
)
def test_matches_private_authoritative_output() -> None:
    ta_path = os.environ.get("PRIVATE_BTG_TA_PATH")
    pa_path = os.environ.get("PRIVATE_BTG_PA_PATH")
    assert ta_path and pa_path, "PRIVATE_BTG_TA_PATH and PRIVATE_BTG_PA_PATH must be set"

    regenerated = build_oxygen_local_btg_diagnostic_features(ta_path, pa_path)
    released = pd.read_csv(RELEASED_TABLE, parse_dates=["date"]).set_index("date")

    pd.testing.assert_frame_equal(regenerated, released, check_dtype=False)
