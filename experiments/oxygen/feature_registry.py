"""The oxygen external-predictor feature registry: which columns belong to
each feature arm, and where they come from.

Reuses the exact same 265-column external feature snapshot chlorophyll's
current-transport arms use
(`coastal_gap_reconstruction.feature_tables.load_full_feature_table`) --
oxygen and chlorophyll share the same site (Tongoy Balsa) and the same
external physical/reanalysis products, so no oxygen-specific external
feature table exists or is needed. This mirrors the private project's own
`oxygen_features.py`, which reads chlorophyll's external feature table by
column-family selection rather than re-deriving external products.

Four arms, per `benchmark_contract.py`'s predictor policy:

- `external_physical_core`: calendar + PLV meteorology (incl. solar/radiation)
  + SST + wind + upwelling. No currents, no satellite/in-situ chlorophyll, no
  local BTG.
- `external_physical_plus_currents`: core + GLORYS/MULTIOBS current-transport
  columns.
- `external_all_available`: core + currents + satellite chlorophyll/ocean-
  color columns -- exploratory/ablation only, never the primary arm
  (`SATELLITE_CHLOROPHYLL_ROLE`).
- `local_btg_temp_pressure_diagnostic`: core + the already-published local-BTG
  water-temperature/pressure diagnostic table
  (`data_public/oxygen/oxygen_local_btg_diagnostic_features.csv`) --
  diagnostic-only, never the primary external-only benchmark.

Every column list is checked against `benchmark_contract.FORBIDDEN_PREDICTOR_SUBSTRINGS`
before being returned -- a forbidden column can never silently enter any arm.
"""

from __future__ import annotations

import pandas as pd

from coastal_gap_reconstruction.feature_tables import load_feature_table, load_full_feature_table

from . import benchmark_contract as bc

__all__ = [
    "CALENDAR_COLS", "SST_COLS", "WIND_COLS", "UPWELLING_COLS", "PLV_MET_COLS",
    "CURRENT_COLS_PREFIXES", "SATELLITE_CHL_COLS",
    "ARM_NAMES", "assert_columns_safe", "load_external_features", "get_feature_arm", "arm_is_diagnostic",
]

CALENDAR_COLS: list[str] = ["doy_sin", "doy_cos", "month", "year"]

SST_COLS: list[str] = [
    "mur_sst_degC", "mur_sst_available", "ostia_sst_degC", "ostia_sst_available",
    "sst_primary_degC", "sst_primary_degC_lag1", "sst_primary_degC_lag3",
    "sst_primary_degC_lag7", "sst_primary_degC_roll3", "sst_primary_degC_roll7",
    "mur_sst_nearest_degC", "mur_sst_10km_mean_degC", "mur_sst_10km_std_degC",
    "mur_gradient_10km_max_degC_per_km", "mur_gradient_10km_mean_degC_per_km",
    "mur_gradient_25km_max_degC_per_km", "mur_front_dist_km", "mur_coastal_grad_340_degC",
    "mur_coastal_grad_local_degC", "mur_sst_anom_doy_degC", "mur_sst_anom_monthly_degC",
    "mur_sst_cooling_1d_degC", "mur_sst_cooling_3d_degC", "mur_sst_cooling_7d_degC",
    "mur_sst_cooling_14d_degC", "mur_sst_cooling_21d_degC", "mur_sst_roll3d_degC",
    "mur_sst_roll7d_degC", "mur_sst_roll14d_degC", "mur_sst_roll21d_degC",
    "mur_front_persist_7d", "mur_front_persist_14d",
]

WIND_COLS: list[str] = [
    "wind_u_ms", "wind_v_ms", "wind_spd_ms", "wind_available",
    "wind_u_ms_lag1", "wind_u_ms_lag3", "wind_u_ms_lag7", "wind_u_ms_roll3", "wind_u_ms_roll7",
    "wind_v_ms_lag1", "wind_v_ms_lag3", "wind_v_ms_lag7", "wind_v_ms_roll3", "wind_v_ms_roll7",
    "wind_spd_ms_lag1", "wind_spd_ms_lag3", "wind_spd_ms_lag7",
    "plv_wind_u_ms", "plv_wind_v_ms", "plv_wind_spd_ms", "plv_wind_valid_hours",
    "plv_wind_coverage", "plv_dir_persist_7d",
]

UPWELLING_COLS: list[str] = [
    "plv_upwelling_ms", "plv_upwelling_cumul3d_ms_d", "plv_upwelling_cumul7d_ms_d",
    "plv_upwelling_cumul14d_ms_d", "plv_upwelling_cumul21d_ms_d",
    "cmems_upwelling_ms", "cmems_upwelling_cumul3d_ms_d", "cmems_upwelling_cumul7d_ms_d",
    "cmems_upwelling_cumul14d_ms_d", "cmems_upwelling_cumul21d_ms_d",
    "plv_relaxation_index_14p3r", "cmems_relaxation_index_14p3r",
]

PLV_MET_COLS: list[str] = [
    "plv_temp_degC", "plv_temp_valid_hours",
    "plv_pressure_hPa", "plv_pressure_valid_hours", "plv_pressure_roll3d_hPa",
    "plv_pressure_roll7d_hPa", "plv_pressure_roll14d_hPa", "plv_pressure_lag3d_hPa",
    "plv_pressure_lag7d_hPa",
    "plv_solar_wm2", "plv_solar_roll3d_wm2", "plv_solar_roll7d_wm2", "plv_solar_roll14d_wm2",
    "plv_solar_roll21d_wm2", "plv_solar_lag3d_wm2", "plv_solar_lag7d_wm2",
    "plv_humid_pct", "plv_humid_roll7d_pct", "plv_humid_roll14d_pct",
    "plv_precip_daily_mm", "plv_precip_roll7d_mm",
]

CURRENT_COLS_PREFIXES: tuple[str, ...] = ("glorys_", "multiobs_")

SATELLITE_CHL_COLS: list[str] = [
    "chl_cons_log10", "chl_cons_available", "chl_cons_log10_lag1", "chl_cons_log10_lag3",
    "chl_cons_log10_lag7", "chl_cons_log10_roll3", "chl_cons_log10_roll7",
    "chl_perm_log10", "chl_perm_available", "chl_is_gapfree", "chl_perm_log10_lag1",
    "chl_cons_nearest", "chl_log10_nearest", "chl_perm_nearest",
    "chl_cons_w3x3_mean", "chl_cons_w3x3_std", "chl_patchiness_cv_3x3",
    "chl_valid_frac_3x3", "chl_valid_frac_5x5", "chl_anom_log10_doy", "chl_anom_log10_monthly",
    "chl_cons_roll3d", "chl_cons_roll7d", "chl_cons_roll14d", "chl_cons_roll21d",
    "chl_patch_dist_km", "chl_patch_persist_7d", "chl_patch_persist_14d",
]

ARM_NAMES: list[str] = [
    "external_physical_core",
    "external_physical_plus_currents",
    "external_all_available",
    "local_btg_temp_pressure_diagnostic",
]


def assert_columns_safe(cols: list[str], arm_name: str) -> None:
    """Raise ValueError if any column in `cols` matches a forbidden predictor
    substring for oxygen (`benchmark_contract.FORBIDDEN_PREDICTOR_SUBSTRINGS`)."""
    for c in cols:
        for bad in bc.FORBIDDEN_PREDICTOR_SUBSTRINGS:
            if bad in c:
                raise ValueError(
                    f"Forbidden substring {bad!r} found in column {c!r} for oxygen arm {arm_name!r}."
                )


def load_external_features(
    base_path=None, extension_path=None, use_currents: bool = True,
) -> pd.DataFrame:
    """Load the shared external feature table (date-indexed).

    `use_currents=False` loads only the 126-column base table (no current/
    kinematic columns) -- cheaper when an arm never needs them.
    """
    base_path = base_path or bc.CHLOROPHYLL_BASE_FEATURES_PATH
    if not use_currents:
        return load_feature_table(base_path)
    extension_path = extension_path or bc.CURRENT_KINEMATIC_EXTENSION_PATH
    return load_full_feature_table(base_path, extension_path)


def get_feature_arm(
    arm_name: str,
    external_df: pd.DataFrame | None = None,
    local_btg_path=None,
) -> pd.DataFrame:
    """Return the date-indexed feature matrix for one of `ARM_NAMES`.

    Raises `ValueError` for an unknown arm name or if a forbidden column
    would be included.
    """
    if arm_name not in ARM_NAMES:
        raise ValueError(f"Unknown oxygen feature arm: {arm_name!r}; choose from {ARM_NAMES}")

    needs_currents = arm_name in ("external_physical_plus_currents", "external_all_available")
    if external_df is None:
        external_df = load_external_features(use_currents=needs_currents)

    core_cols = [c for c in (CALENDAR_COLS + PLV_MET_COLS + WIND_COLS + UPWELLING_COLS + SST_COLS)
                 if c in external_df.columns]

    if arm_name == "external_physical_core":
        cols = core_cols
        assert_columns_safe(cols, arm_name)
        return external_df[cols].copy()

    if arm_name == "external_physical_plus_currents":
        current_cols = [c for c in external_df.columns if c.startswith(CURRENT_COLS_PREFIXES)]
        cols = core_cols + current_cols
        assert_columns_safe(cols, arm_name)
        return external_df[cols].copy()

    if arm_name == "external_all_available":
        current_cols = [c for c in external_df.columns if c.startswith(CURRENT_COLS_PREFIXES)]
        chl_cols = [c for c in SATELLITE_CHL_COLS if c in external_df.columns]
        cols = core_cols + current_cols + chl_cols
        assert_columns_safe(cols, arm_name)
        return external_df[cols].copy()

    if arm_name == "local_btg_temp_pressure_diagnostic":
        local_btg_path = local_btg_path or bc.LOCAL_BTG_DIAGNOSTIC_FEATURES_PATH
        local = load_feature_table(local_btg_path)
        cols = core_cols
        assert_columns_safe(cols, arm_name)
        assert_columns_safe(list(local.columns), arm_name)
        return external_df[cols].join(local, how="left")

    raise AssertionError("unreachable")


def arm_is_diagnostic(arm_name: str) -> bool:
    """True if the arm is diagnostic/exploratory-only, never the primary
    external-only comparator (`external_all_available` per
    `SATELLITE_CHLOROPHYLL_ROLE`; `local_btg_temp_pressure_diagnostic` per
    `LOCAL_BTG_TEMP_PRESSURE_ROLE`)."""
    return arm_name in {"external_all_available", "local_btg_temp_pressure_diagnostic"}
