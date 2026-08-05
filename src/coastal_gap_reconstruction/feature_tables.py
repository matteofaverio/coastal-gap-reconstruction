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

    Uses `float_precision="round_trip"` rather than pandas' default C float
    parser. This is not a stylistic choice: the default parser is fast but is not
    guaranteed to be exact for every float64 value -- verified directly on this
    project's own data (one cell, a `~1e-7`-magnitude value in
    `glorys_okubo_weiss_heuristic_roll7`, parses to a different float64 bit
    pattern than the one that was written, under the default parser, and to the
    exact original bit pattern under `"round_trip"`). Exact reconstruction of a
    frozen benchmark input requires the exact parser on every read, not only the
    write side.
    """
    df = pd.read_csv(path, parse_dates=[DATE_COL], float_precision="round_trip")
    return df.set_index(DATE_COL).sort_index()


def load_full_feature_table(
    base_path: str | Path,
    extension_path: str | Path,
) -> pd.DataFrame:
    """Reconstruct the full 265-column external feature snapshot exactly from the
    two published pieces.

    `chlorophyll_predictor_features_curated.csv` (126 columns) is the base table;
    `data/shared/external_current_kinematic_extension.csv` (210 columns:
    date + 70 override columns + 139 new columns) is published separately, and
    shared between the chlorophyll and oxygen case studies, rather than as a
    second full copy of the base table -- this avoids duplicating the 55 columns
    that are genuinely unchanged, and avoids implying the extension is
    chlorophyll-specific when oxygen's own feature construction reads the same
    265-column snapshot.

    70 of the base table's 125 non-date columns (not merely the 22
    MUR-SST-derived ones a tolerance-based `atol=1e-6` comparison first
    suggested -- see below) have different bit patterns in the private
    265-column snapshot the oxygen and chlorophyll-currents pipelines were
    actually run against, compared to the values in the already-released
    126-column base table. Most of the 70 differ only at the level of a handful
    of ULPs (floating-point noise from independent regeneration of the two
    files), but a few differ meaningfully (e.g. `mur_sst_available`, a
    availability flag, differs on 26 dates). All 70 are treated identically:
    this is not a bug to silently paper over or partially fix. The extension
    file's 70 override columns carry the exact values the private 265-column
    snapshot used, and this loader replaces the base table's versions of those
    70 columns with the extension file's versions -- reproducing the private
    snapshot exactly, not the base table's own (different) values for those
    columns.

    An `atol=1e-6` comparison between the two source files only finds 22 of the
    70 truly-differing columns -- the other 48 differ by less than that
    tolerance per cell, but not by zero, and "zero" is the bar a frozen benchmark
    input must clear, not a tolerance chosen for convenience. See
    `tests/test_feature_table_reconstruction.py` for the exhaustive bit-level
    comparison this override list was derived from.

    Returns a DataFrame with exactly the union of both files' columns: 126 + 139
    = 265 value columns, indexed by date, with the 70 shared columns taking the
    extension file's values. Verified (maintainer-only, env-var-gated) to be
    bitwise float64-identical to the private snapshot on every one of the 264
    non-date columns, and to produce a canonical-CSV SHA-256 that matches the
    private snapshot's own file hash exactly -- not merely DataFrame-equal within
    a tolerance. See `tests/test_feature_table_reconstruction.py`.
    """
    base = load_feature_table(base_path)
    extension = load_feature_table(extension_path)

    override_cols = [c for c in extension.columns if c in base.columns]
    new_cols = [c for c in extension.columns if c not in base.columns]

    result = base.copy()
    result[override_cols] = extension[override_cols]
    result = result.join(extension[new_cols], how="left")

    return result
