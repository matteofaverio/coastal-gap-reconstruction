"""Predictor feature table loading, including exact reconstruction of the full
external feature snapshot from its two published pieces.

Split out of `data_loading.py` for the same reason as `daily_target.py`: feature
tables are their own concern, shared across both case studies, and
`load_full_feature_table` in particular is substantial enough to warrant its own
module rather than living alongside unrelated gap-pool/inventory loaders.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATE_COL = "date"


def load_feature_table(path: str | Path) -> pd.DataFrame:
    """Load a predictor feature table, indexed by date.

    Works for any of the curated feature CSVs as long as they have a "date" column.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    return df.set_index(DATE_COL).sort_index()


def load_full_feature_table(
    base_path: str | Path,
    extension_path: str | Path,
) -> pd.DataFrame:
    """Reconstruct the full 265-column external feature snapshot exactly from the
    two published pieces.

    `chlorophyll_predictor_features_curated.csv` (126 columns) is the base table;
    `data_public/shared/external_current_kinematic_extension.csv` (162 columns:
    date + 22 override columns + 139 new columns) is published separately, and
    shared between the chlorophyll and oxygen case studies, rather than as a
    second full copy of the base table -- this avoids duplicating the 104
    unchanged columns, and avoids implying the extension is chlorophyll-specific
    when oxygen's own feature construction reads the same 265-column snapshot.

    22 of the base table's columns (all MUR SST-derived: gradients, fronts,
    anomalies, cooling rates, rolling means) have different values in the private
    265-column snapshot the oxygen and chlorophyll-currents pipelines were
    actually run against, compared to the values in the already-released
    126-column base table. This is not a bug to silently paper over: the
    extension file's 22 override columns carry the exact values the private
    265-column snapshot used, and this loader replaces the base table's versions
    of those 22 columns with the extension file's versions -- reproducing the
    private snapshot exactly, not the base table's own (different) values for
    those columns.

    Returns a DataFrame with exactly the union of both files' columns: 126 + 139
    = 265 value columns, indexed by date, with the 22 shared columns taking the
    extension file's values. See `tests/test_feature_table_reconstruction.py` for
    both a DataFrame-level and a serialized-file-hash equality check against the
    private snapshot.
    """
    base = load_feature_table(base_path)
    extension = load_feature_table(extension_path)

    override_cols = [c for c in extension.columns if c in base.columns]
    new_cols = [c for c in extension.columns if c not in base.columns]

    result = base.copy()
    result[override_cols] = extension[override_cols]
    result = result.join(extension[new_cols], how="left")

    return result
