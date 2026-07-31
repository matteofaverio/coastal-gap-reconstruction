"""Daily target table loading, shared by every case study.

Split out of `data_loading.py` because `experiments/chlorophyll/target_and_gap_pool.py`
and `experiments/oxygen/target_and_gap_pool.py` both need exactly this loader (and
previously each redefined their own copy of it) -- one canonical implementation,
parametrized by the date column name rather than hardcoding "date" as a chlorophyll
assumption, even though every released table in this repository happens to use it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

DEFAULT_DATE_COL = "date"


def load_daily_target(path: str | Path, date_col: str = DEFAULT_DATE_COL) -> pd.DataFrame:
    """Load a daily target table, indexed by date.

    Works for any of the released daily-target CSVs (chlorophyll or oxygen) --
    both use the same "date" column name, but this is not assumed by baking in a
    default the caller cannot override.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    return df.set_index(date_col).sort_index()


def target_table_checksum(path: str | Path) -> str:
    """SHA-256 of a daily target CSV's raw file bytes.

    Used to stamp `target_table_checksum` columns in gap-pool tables so a released
    pool records exactly which target-table snapshot it was built from.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
