"""Loaders for the public daily target and feature tables.

These functions assume the standard column names used throughout this
repository's public CSVs:

- date column: "date"
- target column: "chl_mean" (daily mean chlorophyll-a)
- eligibility flag: "target_eligible_default" (True if the day has enough
  valid hourly observations to compute a trustworthy daily mean)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET_COL = "chl_mean"
ELIGIBLE_COL = "target_eligible_default"
DATE_COL = "date"


def load_daily_target(path: str | Path) -> pd.DataFrame:
    """Load the daily chlorophyll target table, indexed by date.

    Parameters
    ----------
    path:
        Path to a CSV with at least a "date" column and "chl_mean" /
        "target_eligible_default" columns (see
        docs/data_dictionary.md for the full schema).
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    return df


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load a predictor feature table, indexed by date.

    Works for any of the curated feature CSVs as long as they have a
    "date" column.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df = df.set_index(DATE_COL).sort_index()
    return df


def load_validation_gap_pool(path: str | Path) -> pd.DataFrame:
    """Load the canonical artificial-gap validation pool.

    Returns a DataFrame with one row per artificial gap (gap_id, gap_length,
    start_date, end_date, season, is_high_chl_event, ...).
    """
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    return df


def load_real_gap_inventory(path: str | Path) -> pd.DataFrame:
    """Load the inventory of real (naturally occurring) gaps in the target series."""
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    return df
