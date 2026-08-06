"""The oxygen TS-ICL arm registry and calling layer.

Reuses `coastal_gap_reconstruction.tsicl_helpers` entirely -- checkpoint
loading, provenance verification, resume state, run manifest, portable
metadata, and `run_gap_inference`'s context-slicing/masking/output-validation
logic are all shared with chlorophyll unchanged. Nothing here duplicates
that; this module only adds oxygen-specific facts:

- **Raw mg/L target, no log10 transform** (unlike chlorophyll). Oxygen
  legitimately approaches zero under hypoxia; log10 is undefined/unstable
  exactly at that physically meaningful low tail. `run_gap_inference`'s
  `target_log10` parameter name is a naming artifact from its chlorophyll
  origin -- the function itself performs no log10 math, it only masks,
  slices, and calls the model, so passing raw mg/L values through it is
  correct, not a misuse.
- **Oxygen's own arm registry**: 5 "audited-original" arms (`target_only`,
  `calendar_seasonal`, `external_physical_core`,
  `external_physical_plus_currents`, `local_btg_temp_pressure_diagnostic`) x
  2 context modes = 10 headline rows, plus 4 exploratory single-physical-
  family ablation arms (`currents_only`, `sst_thermal_only`,
  `wind_upwelling_only`, `radiation_only`). No oxygen analogue of
  chlorophyll's satellite-proxy arm exists -- satellite chlorophyll is
  excluded from every oxygen TS-ICL covariate block outright (see
  `benchmark_contract.TSICL_FORBIDDEN_COVARIATE_SUBSTRINGS`).
- **`window_days=730`** for `edge_balanced` context, matching chlorophyll's
  own default and the private oxygen benchmark's own setting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import tsicl_helpers as th

from . import benchmark_contract as bc

__all__ = [
    "CALENDAR_COV_COLS", "PHYSICAL_FORCING_COV_COLS", "LOCAL_BTG_COV_COLS", "CURRENTS_COV_COLS",
    "SST_THERMAL_ONLY_COV_COLS", "WIND_UPWELLING_ONLY_COV_COLS", "RADIATION_ONLY_COV_COLS",
    "AUDITED_ORIGINAL_ARM_COLUMNS", "FAMILY_ABLATION_ARM_COLUMNS", "ALL_ARM_COLUMNS",
    "assert_covariate_cols_safe", "load_target_series", "build_covariate_block", "WINDOW_DAYS",
]

WINDOW_DAYS = 730

CALENDAR_COV_COLS: list[str] = ["doy_sin", "doy_cos"]
# Identical column names to chlorophyll's own PHYSICAL_FORCING_COLS
# (tsicl_contract.py) -- external/reanalysis names, confirmed present in
# oxygen's own external feature source (feature_registry.py).
PHYSICAL_FORCING_COV_COLS: list[str] = [
    "plv_solar_wm2", "plv_solar_roll7d_wm2", "plv_upwelling_ms", "plv_upwelling_cumul7d_ms_d",
    "sst_primary_degC", "sst_primary_degC_roll7", "wind_spd_ms",
]
LOCAL_BTG_COV_COLS: list[str] = ["btg_water_temp_daily_mean", "btg_pressure_daily_mean"]
CURRENTS_COV_COLS: list[str] = [
    "glorys_speed_surface_ms", "glorys_alongshore_340_ms", "glorys_crossshore_340_ms",
    "glorys_speed_roll7", "multiobs_geos_speed_ms", "multiobs_ekman_speed_ms",
]
SST_THERMAL_ONLY_COV_COLS: list[str] = ["sst_primary_degC", "sst_primary_degC_roll7"]
WIND_UPWELLING_ONLY_COV_COLS: list[str] = ["plv_upwelling_ms", "plv_upwelling_cumul7d_ms_d", "wind_spd_ms"]
RADIATION_ONLY_COV_COLS: list[str] = ["plv_solar_wm2", "plv_solar_roll7d_wm2"]

AUDITED_ORIGINAL_ARM_COLUMNS: dict[str, list[str] | None] = {
    "target_only": None,
    "calendar_seasonal": CALENDAR_COV_COLS,
    "external_physical_core": PHYSICAL_FORCING_COV_COLS,
    "external_physical_plus_currents": PHYSICAL_FORCING_COV_COLS + CURRENTS_COV_COLS,
    "local_btg_temp_pressure_diagnostic": LOCAL_BTG_COV_COLS,
}
FAMILY_ABLATION_ARM_COLUMNS: dict[str, list[str]] = {
    "currents_only": CURRENTS_COV_COLS,
    "sst_thermal_only": SST_THERMAL_ONLY_COV_COLS,
    "wind_upwelling_only": WIND_UPWELLING_ONLY_COV_COLS,
    "radiation_only": RADIATION_ONLY_COV_COLS,
}
ALL_ARM_COLUMNS: dict[str, list[str] | None] = {**AUDITED_ORIGINAL_ARM_COLUMNS, **FAMILY_ABLATION_ARM_COLUMNS}


def assert_covariate_cols_safe(cols: list[str] | None, arm_name: str = "") -> None:
    if cols is None:
        return
    for c in cols:
        for bad in bc.TSICL_FORBIDDEN_COVARIATE_SUBSTRINGS:
            if bad in c:
                raise ValueError(
                    f"Forbidden substring {bad!r} found in oxygen TS-ICL covariate column "
                    f"{c!r} for arm {arm_name!r}."
                )


# Self-check at import time: every configured arm's covariate list must
# already be safe -- mirrors the private module's own import-time check.
for _arm, _cols in ALL_ARM_COLUMNS.items():
    assert_covariate_cols_safe(_cols, _arm)


def load_target_series(target_path=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (dates, target_mgL) -- dates as datetime64[D], target as raw
    mg/L (no transform, no floor -- unlike chlorophyll's log10 + floor)."""
    target_path = target_path or bc.DAILY_TARGET_PATH
    df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()
    eligible = df[bc.ELIGIBLE_COLUMN].fillna(False).astype(bool)
    value = df[bc.TARGET_COLUMN].where(eligible)
    target_mgL = value.to_numpy(dtype=np.float32)
    dates = df.index.values.astype("datetime64[D]")
    return dates, target_mgL


def build_covariate_block(arm: str, features_df: pd.DataFrame) -> np.ndarray | None:
    """Return the `(T, C)` covariate block for one arm, or `None` for
    `target_only`. Raises if the arm is unknown or its columns are unsafe."""
    if arm not in ALL_ARM_COLUMNS:
        raise ValueError(f"Unknown oxygen TS-ICL arm: {arm!r}; choose from {sorted(ALL_ARM_COLUMNS)}")
    cols = ALL_ARM_COLUMNS[arm]
    assert_covariate_cols_safe(cols, arm)
    if cols is None:
        return None
    return features_df[cols].to_numpy(dtype=np.float32)


def run_oxygen_gap_inference(
    model, dates: np.ndarray, target_mgL: np.ndarray, gap: th.GapSpec,
    context_mode: str = "full_series", covariate_array: np.ndarray | None = None,
    quantile_levels: list[float] | None = None, strict: bool = True,
) -> dict:
    """Thin, oxygen-specific call to the shared `run_gap_inference` --
    `window_days=730` (WINDOW_DAYS), raw mg/L target (no transform). The
    returned dict's `pred_log10`/`true_log10` keys are inherited field names
    from the shared function's chlorophyll origin; their *values* here are
    raw mg/L, not log10 -- callers must not apply `10**x` to them."""
    return th.run_gap_inference(
        model, dates, target_mgL, gap, context_mode=context_mode,
        covariate_array=covariate_array, quantile_levels=quantile_levels,
        window_days=WINDOW_DAYS, strict=strict,
    )
