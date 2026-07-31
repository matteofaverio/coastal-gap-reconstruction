"""Tests for reconstructing the full 265-column feature snapshot from the two
published pieces (the 126-column base table + the incremental extension table).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from coastal_gap_reconstruction.data_loading import load_full_feature_table

REPO_ROOT = Path(__file__).resolve().parent.parent
BASE_TABLE = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
INCREMENTAL_TABLE = (
    REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_current_kinematic_features_incremental.csv"
)

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
def incremental_columns() -> list[str]:
    return pd.read_csv(INCREMENTAL_TABLE, nrows=0).columns.tolist()


def test_incremental_table_has_date_plus_161_value_columns(incremental_columns: list[str]) -> None:
    assert "date" in incremental_columns
    assert len(incremental_columns) == 162  # date + 22 override + 139 new


def test_incremental_table_override_columns_match_expected_set(incremental_columns: list[str]) -> None:
    base_columns = set(pd.read_csv(BASE_TABLE, nrows=0).columns) - {"date"}
    override_columns = {c for c in incremental_columns if c in base_columns}
    assert override_columns == EXPECTED_OVERRIDE_COLUMNS


def test_incremental_table_new_columns_do_not_exist_in_base_table(incremental_columns: list[str]) -> None:
    base_columns = set(pd.read_csv(BASE_TABLE, nrows=0).columns) - {"date"}
    new_columns = [c for c in incremental_columns if c not in base_columns and c != "date"]
    assert len(new_columns) == 139
    assert base_columns.isdisjoint(new_columns)


def test_reconstructed_table_has_265_columns_and_correct_row_count() -> None:
    full = load_full_feature_table(BASE_TABLE, INCREMENTAL_TABLE)
    assert full.shape[1] == 264  # 265 released columns minus the date index
    assert len(full) == 3988


def test_reconstructed_table_uses_incremental_values_for_override_columns() -> None:
    """The 22 override columns must come from the incremental file, not the base
    file -- this is the whole point of shipping them separately (see
    `data_loading.load_full_feature_table`'s docstring)."""
    full = load_full_feature_table(BASE_TABLE, INCREMENTAL_TABLE)
    incremental = pd.read_csv(INCREMENTAL_TABLE, parse_dates=["date"]).set_index("date").sort_index()

    for col in EXPECTED_OVERRIDE_COLUMNS:
        pd.testing.assert_series_equal(
            full[col].sort_index(), incremental[col].sort_index(), check_names=False
        )


def test_reconstructed_table_date_coverage_matches_base_table() -> None:
    full = load_full_feature_table(BASE_TABLE, INCREMENTAL_TABLE)
    base = pd.read_csv(BASE_TABLE, parse_dates=["date"]).set_index("date").sort_index()
    assert list(full.index) == list(base.index)
