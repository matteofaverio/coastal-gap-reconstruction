"""The TS-ICL covariate-arm registry for chlorophyll: descriptive public
identities mapped to exact column membership, source, and scientific role.

The private project identified these arms by internal short codes (`C0`-
`C13`, plus `arm D` for the satellite-proxy target-only benchmark arm) in
both code and most internal documentation. This registry is the single
place those internal codes are translated to the **descriptive public
names already used in the released
`results_public/chlorophyll/chlorophyll_covariate_mechanism_summary.csv`**
(`covariate_public_name` column) -- reusing that existing public mapping as
the naming source, not inventing new names. Every arm actually run by
`run_tsicl_covariate_analysis.py` is looked up here; nothing downstream
carries the internal C-codes as its primary identity.

Column lists are ported verbatim from the private project's covariate-
dissection module (feature-family definitions only -- the arm-iteration and
placebo-generation code is ported separately into
`coastal_gap_reconstruction.tsicl_helpers`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = [
    "SATELLITE_PROXY_COLUMNS",
    "SOLAR_COLUMNS",
    "WIND_UPWELLING_COLUMNS",
    "SST_THERMAL_COLUMNS",
    "PLV_MET_COLUMNS",
    "CURRENT_TRANSPORT_COLUMNS",
    "AVAILABILITY_COLUMNS",
    "CURATED_PHYSICAL_COLUMNS",
    "CovariateArmSpec",
    "COVARIATE_ARMS",
    "PLACEBO_TRANSFORMS",
    "PLACEBO_ELIGIBLE_ARMS",
]

# ── Feature-family column lists (verbatim from the private project's
# covariate-dissection module) ──────────────────────────────────────────
SATELLITE_PROXY_COLUMNS: list[str] = ["chl_cons_w3x3_mean", "chl_anom_log10_doy"]

SOLAR_COLUMNS: list[str] = [
    "plv_solar_wm2", "plv_solar_roll3d_wm2", "plv_solar_roll7d_wm2",
    "plv_solar_roll14d_wm2", "plv_solar_roll21d_wm2", "plv_solar_lag3d_wm2", "plv_solar_lag7d_wm2",
]
WIND_UPWELLING_COLUMNS: list[str] = [
    "plv_wind_u_ms", "plv_wind_v_ms", "plv_wind_spd_ms",
    "wind_u_ms", "wind_v_ms", "wind_spd_ms",
    "wind_u_ms_lag1", "wind_u_ms_lag3", "wind_u_ms_lag7", "wind_u_ms_roll3", "wind_u_ms_roll7",
    "wind_v_ms_lag1", "wind_v_ms_lag3", "wind_v_ms_lag7", "wind_v_ms_roll3", "wind_v_ms_roll7",
    "wind_spd_ms_lag1", "wind_spd_ms_lag3", "wind_spd_ms_lag7",
    "plv_upwelling_ms", "plv_upwelling_cumul3d_ms_d", "plv_upwelling_cumul7d_ms_d",
    "plv_upwelling_cumul14d_ms_d", "plv_upwelling_cumul21d_ms_d",
    "cmems_upwelling_ms", "cmems_upwelling_cumul3d_ms_d", "cmems_upwelling_cumul7d_ms_d",
    "cmems_upwelling_cumul14d_ms_d", "cmems_upwelling_cumul21d_ms_d",
    "plv_relaxation_index_14p3r", "cmems_relaxation_index_14p3r", "plv_dir_persist_7d",
]
SST_THERMAL_COLUMNS: list[str] = [
    "sst_primary_degC", "sst_primary_degC_lag1", "sst_primary_degC_lag3", "sst_primary_degC_lag7",
    "sst_primary_degC_roll3", "sst_primary_degC_roll7",
    "mur_sst_degC", "mur_sst_nearest_degC", "mur_sst_10km_mean_degC", "mur_sst_10km_std_degC",
    "mur_gradient_10km_max_degC_per_km", "mur_gradient_10km_mean_degC_per_km",
    "mur_gradient_25km_max_degC_per_km", "mur_front_dist_km",
    "mur_coastal_grad_340_degC", "mur_coastal_grad_local_degC",
    "mur_sst_anom_doy_degC", "mur_sst_anom_monthly_degC",
    "mur_sst_cooling_1d_degC", "mur_sst_cooling_3d_degC", "mur_sst_cooling_7d_degC",
    "mur_sst_cooling_14d_degC", "mur_sst_cooling_21d_degC",
    "mur_sst_roll3d_degC", "mur_sst_roll7d_degC", "mur_sst_roll14d_degC", "mur_sst_roll21d_degC",
    "mur_front_persist_7d", "mur_front_persist_14d", "ostia_sst_degC",
]
PLV_MET_COLUMNS: list[str] = [
    "plv_pressure_hPa", "plv_pressure_roll3d_hPa", "plv_pressure_roll7d_hPa",
    "plv_pressure_roll14d_hPa", "plv_pressure_lag3d_hPa", "plv_pressure_lag7d_hPa",
    "plv_humid_pct", "plv_humid_roll7d_pct", "plv_humid_roll14d_pct",
    "plv_precip_daily_mm", "plv_precip_roll7d_mm", "plv_temp_degC",
]
# Current/transport columns require the full 265-column feature snapshot
# (`coastal_gap_reconstruction.feature_tables.load_full_feature_table`,
# joining `data_public/shared/external_current_kinematic_extension.csv` onto
# the base 126-column table -- Phase 2A's shared extension loader), not the
# base table alone -- see `requires_extended_table` below.
CURRENT_TRANSPORT_COLUMNS: list[str] = [
    "glorys_uo_surface_ms", "glorys_vo_surface_ms", "glorys_speed_surface_ms",
    "glorys_alongshore_340_ms", "glorys_crossshore_340_ms",
    "glorys_alongshore_053_ms", "glorys_crossshore_053_ms",
    "glorys_mld_m", "glorys_thetao_surface_degC",
    "glorys_speed_roll3", "glorys_speed_roll7", "glorys_speed_roll14", "glorys_speed_roll21",
    "glorys_weak_current_frac3", "glorys_weak_current_frac7", "glorys_weak_current_frac14", "glorys_weak_current_frac21",
    "glorys_cumul_along340_3d", "glorys_cumul_cross340_3d",
    "glorys_cumul_along340_7d", "glorys_cumul_cross340_7d",
    "glorys_cumul_along340_14d", "glorys_cumul_cross340_14d",
    "glorys_cumul_along340_21d", "glorys_cumul_cross340_21d",
    "glorys_sign_persist_along340_7d", "glorys_sign_persist_cross340_7d",
    "glorys_sign_persist_along340_14d", "glorys_sign_persist_cross340_14d",
    "glorys_reversal_count_along340_7d", "glorys_reversal_count_along340_14d", "glorys_reversal_count_along340_21d",
    "glorys_mld_roll7",
    "glorys_u_mean_3x3", "glorys_v_mean_3x3", "glorys_speed_mean_3x3",
    "glorys_divergence_s_per_km", "glorys_vorticity_s_per_km", "glorys_okubo_weiss_heuristic",
    "multiobs_ugos_ms", "multiobs_vgos_ms", "multiobs_ue_ms", "multiobs_ve_ms",
    "multiobs_geos_speed_ms", "multiobs_ekman_speed_ms",
    "multiobs_component_sum_u_ms", "multiobs_component_sum_v_ms", "multiobs_component_sum_speed_ms",
    "multiobs_along340_geos_ms", "multiobs_cross340_geos_ms",
    "multiobs_along340_ekman_ms", "multiobs_cross340_ekman_ms",
    "multiobs_ekman_geo_ratio",
]
AVAILABILITY_COLUMNS: list[str] = [
    "mur_sst_available", "ostia_sst_available", "chl_cons_available", "chl_perm_available",
    "wind_available", "plv_wind_valid_hours", "plv_wind_coverage",
    "plv_temp_valid_hours", "plv_pressure_valid_hours",
    "mur_mask_rescued", "chl_is_gapfree",
    "multiobs_availability_flag", "multiobs_source_flag",
]
CURATED_PHYSICAL_COLUMNS: list[str] = [
    "plv_solar_roll7d_wm2", "plv_upwelling_cumul7d_ms_d", "wind_spd_ms",
    "sst_primary_degC_roll7", "mur_sst_cooling_7d_degC", "plv_pressure_hPa",
    "plv_relaxation_index_14p3r",
]


@dataclass(frozen=True)
class CovariateArmSpec:
    """One covariate arm: descriptive public name, exact columns, role."""

    arm_id: str
    public_name: str
    columns: list[str] = field(default_factory=list)
    role: str = "exploratory"  # "primary" | "supporting" | "exploratory" | "negative_control"
    requires_extended_table: bool = False
    note: str = ""


COVARIATE_ARMS: dict[str, CovariateArmSpec] = {
    "target_only": CovariateArmSpec(
        "target_only", "No covariate (target-only baseline)", [], "primary",
        note="TS-ICL sees only the masked target series -- no covariate channel at all.",
    ),
    "satellite_proxy": CovariateArmSpec(
        "satellite_proxy", "Satellite chlorophyll proxy", SATELLITE_PROXY_COLUMNS, "primary",
        note="The headline covariate arm (private 'arm D'): a satellite chlorophyll-a proxy, "
             "not the in-situ target itself -- a genuinely independent remote-sensing "
             "observation, never used as ground truth for scoring.",
    ),
    "solar_only": CovariateArmSpec("solar_only", "Solar radiation only", SOLAR_COLUMNS, "supporting"),
    "wind_upwelling_only": CovariateArmSpec(
        "wind_upwelling_only", "Wind/upwelling forcing only", WIND_UPWELLING_COLUMNS, "supporting",
    ),
    "sst_thermal_only": CovariateArmSpec("sst_thermal_only", "SST thermal only", SST_THERMAL_COLUMNS, "supporting"),
    "plv_meteorological": CovariateArmSpec(
        "plv_meteorological", "PLV meteorological (combined)", PLV_MET_COLUMNS, "supporting",
    ),
    "current_transport_only": CovariateArmSpec(
        "current_transport_only", "Ocean current/transport only", CURRENT_TRANSPORT_COLUMNS,
        "supporting", requires_extended_table=True,
    ),
    "availability_proxy_only": CovariateArmSpec(
        "availability_proxy_only", "Availability/weather proxy only", AVAILABILITY_COLUMNS, "supporting",
    ),
    "solar_upwelling_interaction": CovariateArmSpec(
        "solar_upwelling_interaction", "Solar x upwelling interaction",
        ["plv_solar_wm2", "plv_solar_roll7d_wm2", "plv_upwelling_ms",
         "plv_upwelling_cumul7d_ms_d", "solar_x_upwelling_product"],
        "supporting", note="Includes an engineered product column -- see build_engineered_products().",
    ),
    "upwelling_cooling_interaction": CovariateArmSpec(
        "upwelling_cooling_interaction", "Upwelling x SST cooling interaction",
        ["plv_upwelling_ms", "plv_upwelling_cumul7d_ms_d", "mur_sst_cooling_7d_degC",
         "sst_primary_degC_roll7", "upwelling_x_cooling_product"],
        "supporting", note="Includes an engineered product column -- see build_engineered_products().",
    ),
    "curated_physical": CovariateArmSpec(
        "curated_physical", "Curated low-redundancy physical set", CURATED_PHYSICAL_COLUMNS, "primary",
        note="The best-performing non-proxy physical arm -- primary comparator alongside "
             "target-only and satellite-proxy.",
    ),
    "full_physical_redundant": CovariateArmSpec(
        "full_physical_redundant", "Full (redundant) physical set",
        SOLAR_COLUMNS + WIND_UPWELLING_COLUMNS + SST_THERMAL_COLUMNS + PLV_MET_COLUMNS, "supporting",
        note="All base physical families combined, unfiltered -- performs worse than the "
             "curated subset (redundancy, not more information, per the released ranking).",
    ),
    "proxy_plus_solar": CovariateArmSpec(
        "proxy_plus_solar", "Satellite chlorophyll proxy + solar",
        SATELLITE_PROXY_COLUMNS + SOLAR_COLUMNS, "supporting",
    ),
    "proxy_plus_wind_upwelling": CovariateArmSpec(
        "proxy_plus_wind_upwelling", "Satellite chlorophyll proxy + wind/upwelling",
        SATELLITE_PROXY_COLUMNS + WIND_UPWELLING_COLUMNS, "supporting",
    ),
    "proxy_plus_sst": CovariateArmSpec(
        "proxy_plus_sst", "Satellite chlorophyll proxy + SST",
        SATELLITE_PROXY_COLUMNS + SST_THERMAL_COLUMNS, "supporting",
    ),
    "proxy_plus_plv_met": CovariateArmSpec(
        "proxy_plus_plv_met", "Satellite chlorophyll proxy + PLV meteorological",
        SATELLITE_PROXY_COLUMNS + PLV_MET_COLUMNS, "supporting",
    ),
    "proxy_plus_current_transport": CovariateArmSpec(
        "proxy_plus_current_transport", "Satellite chlorophyll proxy + current/transport",
        SATELLITE_PROXY_COLUMNS + CURRENT_TRANSPORT_COLUMNS, "supporting", requires_extended_table=True,
        note="Two internal arm IDs (private C11e/C12) share this exact same column set; the "
             "released chlorophyll_covariate_mechanism_summary.csv accordingly carries a "
             "second row for this configuration labeled '(duplicate config)' -- reproduced "
             "here as one arm, not two, since the columns are identical.",
    ),
    "proxy_plus_availability": CovariateArmSpec(
        "proxy_plus_availability", "Satellite chlorophyll proxy + availability proxy",
        SATELLITE_PROXY_COLUMNS + AVAILABILITY_COLUMNS, "supporting",
    ),
}

# ── Placebo/negative-control transforms (applied to a covariate arm's own
# columns, not a separate arm identity) ─────────────────────────────────
PLACEBO_TRANSFORMS: list[str] = ["wrong_lag", "season_shuffled", "year_shifted", "permuted"]

# Which primary arms the placebo battery is run against (matches the
# released placebo_robustness_test row: curated physical vs. its own
# randomized/shuffled controls).
PLACEBO_ELIGIBLE_ARMS: list[str] = ["curated_physical", "wind_upwelling_only", "solar_only",
                                     "sst_thermal_only", "current_transport_only"]


def build_engineered_products(features_df):
    """Add the two engineered interaction-product columns some arms need
    (`solar_x_upwelling_product`, `upwelling_x_cooling_product`) -- a
    two-line transform, reproduced exactly rather than described in prose."""
    features_df = features_df.copy()
    features_df["solar_x_upwelling_product"] = features_df["plv_solar_wm2"] * features_df["plv_upwelling_ms"]
    features_df["upwelling_x_cooling_product"] = (
        features_df["plv_upwelling_ms"] * features_df["mur_sst_cooling_7d_degC"]
    )
    return features_df


def apply_placebo_transform(covariate_block, dates, transform: str, seed: int = 0):
    """Apply one of `PLACEBO_TRANSFORMS` to a `(T, C)` covariate block,
    exactly reproducing the private project's four placebo definitions.

    - `"wrong_lag"`: shift the whole block 90 days (destroys the correct
      timing while preserving the covariate's own autocorrelation
      structure).
    - `"season_shuffled"`: permute rows within each calendar month
      independently (destroys day-to-day timing, preserves seasonal
      climatology).
    - `"year_shifted"`: shift the whole block ~365 days (destroys the
      specific-year alignment while preserving season).
    - `"permuted"`: a full random permutation of all rows (destroys all
      temporal structure).
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    block = np.asarray(covariate_block).copy()
    if transform == "wrong_lag":
        return np.roll(block, shift=90, axis=0)
    if transform == "season_shuffled":
        out = block.copy()
        months = dates.astype("datetime64[M]").astype(int) % 12 + 1
        for m in range(1, 13):
            idx = np.where(months == m)[0]
            perm = rng.permutation(idx)
            out[idx] = block[perm]
        return out
    if transform == "year_shifted":
        return np.roll(block, shift=365, axis=0)
    if transform == "permuted":
        perm = rng.permutation(len(block))
        return block[perm]
    raise ValueError(f"unknown placebo transform {transform!r}; choose from {PLACEBO_TRANSFORMS}")
