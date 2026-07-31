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


def load_full_feature_table(
    base_path: str | Path,
    incremental_path: str | Path,
) -> pd.DataFrame:
    """Reconstruct the full 265-column feature table exactly from the two
    published pieces.

    `chlorophyll_predictor_features_curated.csv` (126 columns) is the base table;
    `chlorophyll_current_kinematic_features_incremental.csv` (162 columns: date +
    22 override columns + 139 new columns) is published separately rather than as
    a second full copy of the base table, to avoid duplicating the 104 unchanged
    columns.

    22 of the base table's columns (all MUR SST-derived: gradients, fronts,
    anomalies, cooling rates, rolling means) have different values in the private
    265-column snapshot the oxygen and chlorophyll-currents pipelines were
    actually run against, compared to the values in the already-released
    126-column base table. This is not a bug to silently paper over: the
    incremental file's 22 override columns carry the exact values the private
    265-column snapshot used, and this loader replaces the base table's versions
    of those 22 columns with the incremental file's versions -- reproducing the
    private snapshot exactly, not the base table's own (different) values for
    those columns.

    Returns a DataFrame with exactly the union of both files' columns: 126 + 139
    = 265 value columns, indexed by date, with the 22 shared columns taking the
    incremental file's values.
    """
    base = load_feature_table(base_path)
    incremental = load_feature_table(incremental_path)

    override_cols = [c for c in incremental.columns if c in base.columns]
    new_cols = [c for c in incremental.columns if c not in base.columns]

    result = base.copy()
    result[override_cols] = incremental[override_cols]
    result = result.join(incremental[new_cols], how="left")

    return result


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
