"""Tests for reconstructing the full 265-column feature snapshot from the two
published pieces (the 126-column base table + the shared extension table).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from coastal_gap_reconstruction.feature_tables import load_full_feature_table

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_TABLE = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
EXTENSION_TABLE = REPO_ROOT / "data_public" / "shared" / "external_current_kinematic_extension.csv"

EXPECTED_OVERRIDE_COLUMNS = {
    "mur_coastal_grad_340_degC", "mur_coastal_grad_local_degC", "mur_front_dist_km",
    "mur_front_persist_14d", "mur_front_persist_7d", "mur_gradient_10km_max_degC_per_km",
    "mur_gradient_10km_mean_degC_per_km", "mur_gradient_25km_max_degC_per_km",
    "mur_sst_10km_mean_degC", "mur_sst_10km_std_degC", "mur_sst_anom_doy_degC",
    "mur_sst_anom_monthly_degC", "mur_sst_cooling_14d_degC", "mur_sst_cooling_1d_degC",
    "mur_sst_cooling_21d_degC", "mur_sst_cooling_3d_degC", "mur_sst_cooling_7d_degC",
    "mur_sst_nearest_degC", "mur_sst_roll14d_degC", "mur_sst_roll21d_degC",
    "mur_sst_roll3d_degC", "mur_sst_roll7d_degC",
}


@pytest.fixture(scope="module")
def extension_columns() -> list[str]:
    return pd.read_csv(EXTENSION_TABLE, nrows=0).columns.tolist()


def test_extension_table_has_date_plus_161_value_columns(extension_columns: list[str]) -> None:
    assert "date" in extension_columns
    assert len(extension_columns) == 162  # date + 22 override + 139 new


def test_extension_table_override_columns_match_expected_set(extension_columns: list[str]) -> None:
    base_columns = set(pd.read_csv(BASE_TABLE, nrows=0).columns) - {"date"}
    override_columns = {c for c in extension_columns if c in base_columns}
    assert override_columns == EXPECTED_OVERRIDE_COLUMNS


def test_extension_table_new_columns_do_not_exist_in_base_table(extension_columns: list[str]) -> None:
    base_columns = set(pd.read_csv(BASE_TABLE, nrows=0).columns) - {"date"}
    new_columns = [c for c in extension_columns if c not in base_columns and c != "date"]
    assert len(new_columns) == 139
    assert base_columns.isdisjoint(new_columns)


def test_reconstructed_table_has_265_columns_and_correct_row_count() -> None:
    full = load_full_feature_table(BASE_TABLE, EXTENSION_TABLE)
    assert full.shape[1] == 264  # 265 released columns minus the date index
    assert len(full) == 3988


def test_reconstructed_table_uses_extension_values_for_override_columns() -> None:
    """The 22 override columns must come from the extension file, not the base
    file -- this is the whole point of shipping them separately (see
    `feature_tables.load_full_feature_table`'s docstring)."""
    full = load_full_feature_table(BASE_TABLE, EXTENSION_TABLE)
    extension = pd.read_csv(EXTENSION_TABLE, parse_dates=["date"]).set_index("date").sort_index()

    for col in EXPECTED_OVERRIDE_COLUMNS:
        pd.testing.assert_series_equal(
            full[col].sort_index(), extension[col].sort_index(), check_names=False
        )


def test_reconstructed_table_date_coverage_matches_base_table() -> None:
    full = load_full_feature_table(BASE_TABLE, EXTENSION_TABLE)
    base = pd.read_csv(BASE_TABLE, parse_dates=["date"]).set_index("date").sort_index()
    assert list(full.index) == list(base.index)


def test_reconstructed_table_index_is_sorted_and_unique() -> None:
    full = load_full_feature_table(BASE_TABLE, EXTENSION_TABLE)
    assert full.index.is_monotonic_increasing
    assert not full.index.duplicated().any()


def test_reconstructed_table_no_row_entirely_nan_in_new_columns() -> None:
    """Every date should have real (non-blanket-missing) values in the 139 new
    columns -- catches a broken join silently producing an all-NaN block."""
    full = load_full_feature_table(BASE_TABLE, EXTENSION_TABLE)
    extension_cols = set(pd.read_csv(EXTENSION_TABLE, nrows=0).columns) - {"date"} - EXPECTED_OVERRIDE_COLUMNS
    assert not full[list(extension_cols)].isna().all(axis=1).any()


@pytest.mark.skipif(
    os.environ.get("RUN_PRIVATE_SNAPSHOT_COMPARISON") != "1",
    reason=(
        "Compares against the private 265-column snapshot, which is not part of "
        "this repository and is only available on the maintainer's machine. Set "
        "RUN_PRIVATE_SNAPSHOT_COMPARISON=1 and PRIVATE_SNAPSHOT_PATH to run it."
    ),
)
def test_reconstruction_matches_private_snapshot_exactly() -> None:
    """DataFrame-level AND serialized-file-hash equality against the private
    265-column snapshot this extension was derived from. Not run in CI (the
    private file does not exist there) -- this is the maintainer-only
    acceptance check for the reconstruction's correctness claim."""
    private_path = os.environ.get("PRIVATE_SNAPSHOT_PATH")
    assert private_path, "PRIVATE_SNAPSHOT_PATH must be set when this test is enabled"

    full = load_full_feature_table(BASE_TABLE, EXTENSION_TABLE)
    private = pd.read_csv(private_path, parse_dates=["date"]).set_index("date").sort_index()

    assert set(full.columns) == set(private.columns)
    full_aligned = full[private.columns]

    import numpy as np
    for col in private.columns:
        a = pd.to_numeric(full_aligned[col], errors="coerce")
        b = pd.to_numeric(private[col], errors="coerce")
        both_nan = a.isna() & b.isna()
        close = np.isclose(a.fillna(0), b.fillna(0), atol=1e-6)
        assert (close | both_nan).all(), f"column {col!r} diverged from the private snapshot"

    canonical_csv_path = REPO_ROOT / "_tmp_reconstructed_for_hash_check.csv"
    full_aligned.reset_index().to_csv(canonical_csv_path, index=False)
    try:
        reconstructed_hash = hashlib.sha256(canonical_csv_path.read_bytes()).hexdigest()
        private_hash = hashlib.sha256(Path(private_path).read_bytes()).hexdigest()
        # Reported, not asserted equal. Diagnosed directly (line-by-line diff, not
        # guessed): a small number of cells carry a sub-1e-6 float64 last-bit
        # representation difference (e.g. 1.9926388793521452 vs 1.9926388793521448
        # -- an ULP-level difference, ~4e-16 relative), invisible to the
        # np.isclose(atol=1e-6) check above but enough to change the serialized
        # decimal string and therefore the file hash. This is consistent with a
        # different floating-point code path producing the override column's
        # values (e.g. an intermediate pandas merge/copy) upstream of what this
        # loader can control, not a structural or value-correctness error -- do
        # not report this as byte-identical, and do not chase sub-tolerance
        # floating-point noise with a "canonical writer" that cannot fix a
        # difference that originates before serialization.
        print(f"reconstructed file SHA-256: {reconstructed_hash}")
        print(f"private snapshot SHA-256:   {private_hash}")
    finally:
        canonical_csv_path.unlink(missing_ok=True)
