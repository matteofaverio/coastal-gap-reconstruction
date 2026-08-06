"""Loaders for the public daily target and feature tables.

`load_daily_target` and `load_feature_table` re-export from `daily_target.py` and
`feature_tables.py` respectively, which hold the canonical implementations -- kept
here too (not just moved) because existing notebooks import them from this module
by name; this is the one authoritative implementation in both places, not a
duplicate. `load_full_feature_table` moved to `feature_tables.py` in full (its
implementation belongs with the other feature-table loader, not here); import it
from there directly, or via the re-export below.

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

from .daily_target import load_daily_target as _load_daily_target
from .feature_tables import load_feature_table as _load_feature_table
from .feature_tables import load_full_feature_table

TARGET_COL = "chl_mean"
ELIGIBLE_COL = "target_eligible_default"
DATE_COL = "date"

__all__ = [
    "load_daily_target",
    "load_feature_table",
    "load_full_feature_table",
    "load_validation_gap_pool",
    "load_real_gap_inventory",
]


def load_daily_target(path: str | Path) -> pd.DataFrame:
    """Load the daily chlorophyll target table, indexed by date.

    Parameters
    ----------
    path:
        Path to a CSV with at least a "date" column and "chl_mean" /
        "target_eligible_default" columns (see
        docs/data_dictionary.md for the full schema).
    """
    return _load_daily_target(path, date_col=DATE_COL)


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load a predictor feature table, indexed by date.

    Works for any of the curated feature CSVs as long as they have a
    "date" column.
    """
    return _load_feature_table(path)


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
