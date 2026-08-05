"""The authoritative oxygen benchmark contract: target/QC definition, gap-pool
support, predictor admissibility policy, and method/status categories --
tested Python constants/dataclasses, not a YAML file nothing else reads.
Companion to `experiments.chlorophyll.benchmark_contract`, kept separate
because the oxygen and chlorophyll case studies genuinely differ in target
scale, support size, and predictor policy (this module states those
differences explicitly rather than silently reusing chlorophyll's).

Every count and support role below is tested against the released public
CSVs (`tests/test_oxygen_benchmark_contract.py`), not merely asserted here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PUBLIC_DIR = REPO_ROOT / "data_public" / "oxygen"
SHARED_DATA_PUBLIC_DIR = REPO_ROOT / "data_public" / "shared"
RESULTS_PUBLIC_DIR = REPO_ROOT / "results_public" / "oxygen"

# ── Data paths ───────────────────────────────────────────────────────────
DAILY_TARGET_PATH = DATA_PUBLIC_DIR / "oxygen_daily_target.csv"
VALIDATION_GAPS_PATH = DATA_PUBLIC_DIR / "oxygen_validation_gaps.csv"
LOCAL_BTG_DIAGNOSTIC_FEATURES_PATH = DATA_PUBLIC_DIR / "oxygen_local_btg_diagnostic_features.csv"
REAL_GAP_INVENTORY_PATH = DATA_PUBLIC_DIR / "oxygen_real_gap_inventory_by_class.csv"

# Shared with chlorophyll -- both case studies read the same 265-column
# external feature snapshot (base table + current/kinematic extension); see
# `coastal_gap_reconstruction.feature_tables.load_full_feature_table`.
CHLOROPHYLL_BASE_FEATURES_PATH = (
    REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
)
CURRENT_KINEMATIC_EXTENSION_PATH = SHARED_DATA_PUBLIC_DIR / "external_current_kinematic_extension.csv"

# ── Frozen result tables (already public) ───────────────────────────────
BENCHMARK_BY_LENGTH_PATH = RESULTS_PUBLIC_DIR / "oxygen_benchmark_by_length.csv"
PAIRED_DELTAS_VS_TSICL_PATH = RESULTS_PUBLIC_DIR / "oxygen_paired_deltas_vs_tsicl_physical_covariates.csv"
TAIL_PERSISTENCE_METRICS_PATH = RESULTS_PUBLIC_DIR / "oxygen_tail_persistence_metrics.csv"
TAIL_QUANTILE_BAND_METRICS_PATH = RESULTS_PUBLIC_DIR / "oxygen_tail_quantile_band_metrics.csv"

# ── Target/QC contract ───────────────────────────────────────────────────
SENSOR_CODE = "BTGOXD2"
TARGET_COLUMN = "oxygen_mean_mgL"
ELIGIBLE_COLUMN = "eligible_ge_18"
DATE_COLUMN = "date"
TARGET_UNIT = "mg/L"
TARGET_TRANSFORM = "identity"  # raw mg/L throughout -- no log10 (oxygen legitimately approaches 0)
MIN_VALID_HOURS = 18
# No day/night eligibility rule is applied -- explicitly not part of this
# contract unless a future arm documents one as diagnostic (none does).
DAY_NIGHT_RULE = None

# ── Gap-pool support contract ────────────────────────────────────────────
PRIMARY_GAP_LENGTHS: dict[int, int] = {1: 100, 3: 100, 7: 78, 10: 55, 14: 36, 21: 22, 30: 15}
EXPLORATORY_GAP_LENGTHS: dict[int, int] = {45: 3, 60: 1, 90: 1, 120: 1}
ALL_GAP_LENGTHS: dict[int, int] = {**PRIMARY_GAP_LENGTHS, **EXPLORATORY_GAP_LENGTHS}

PRIMARY_N_GAPS = sum(PRIMARY_GAP_LENGTHS.values())  # 406
EXPLORATORY_N_GAPS = sum(EXPLORATORY_GAP_LENGTHS.values())  # 6
TOTAL_N_GAPS = PRIMARY_N_GAPS + EXPLORATORY_N_GAPS  # 412
PRIMARY_N_HIDDEN_DAYS = sum(length * count for length, count in PRIMARY_GAP_LENGTHS.items())  # 2912

SUPPORT_ROLE_PRIMARY = "primary"
SUPPORT_ROLE_EXPLORATORY = "exploratory_extended"


def support_role(gap_length: int) -> str:
    """Public support-role label: primary (L<=30, the only range primary
    scientific claims may be drawn from) vs. exploratory_extended (L>=45,
    illustrative only -- never used to support a headline claim)."""
    return SUPPORT_ROLE_PRIMARY if gap_length <= 30 else SUPPORT_ROLE_EXPLORATORY


# ── Metrics contract ──────────────────────────────────────────────────────
PRIMARY_METRIC = "mae_mgL"  # gap-weighted MAE in mg/L, the headline convention for every oxygen sprint
SECONDARY_METRICS = ["rmse_mgL", "bias_mgL", "correlation", "quantile_coverage", "mae_over_iqr", "mae_over_std"]
WEIGHTING_PRIMARY = "gap_weighted"
WEIGHTING_SECONDARY = "day_weighted"

# ── Predictor admissibility policy ────────────────────────────────────────
CURRENTS_ADMISSIBLE = True
SATELLITE_CHLOROPHYLL_ROLE = "exploratory_ablation_only"  # never a primary predictor
IN_SITU_CHLOROPHYLL_ADMISSIBLE = False
LOCAL_BTG_TEMP_PRESSURE_ROLE = "diagnostic_arm_only"  # BTGTA/BTGPA -- same-station co-missingness risk

# Classical/tabular feature-registry ban: only in-situ chlorophyll (the
# chlorophyll case study's own target) and oxygen-sensor-derived/excluded
# variables are forbidden outright. Satellite chlorophyll proxy columns
# (chl_cons_*/chl_perm_*/chl_anom_*) are deliberately NOT in this list --
# they are legitimate, structurally-restricted predictors in the
# `external_all_available` exploratory-ablation arm only (never
# `external_physical_core`/`external_physical_plus_currents`, which are
# built from disjoint column families and never include them). Matches the
# private project's own `oxygen_features.py::FORBIDDEN_SUBSTRINGS` exactly.
FORBIDDEN_PREDICTOR_SUBSTRINGS: tuple[str, ...] = (
    "chl_mean", "BTGOXD", "BTGOXSATPC", "BTGSAL", "BTGTUR", "BTGCND",
)
# BTGTA/BTGPA (water temp/pressure) are excluded from FORBIDDEN_PREDICTOR_SUBSTRINGS
# deliberately -- they are not forbidden outright, only restricted to the
# clearly-labeled diagnostic arm (see LOCAL_BTG_TEMP_PRESSURE_ROLE and
# feature_registry.py's `arm_is_diagnostic`).

# TS-ICL covariate ban is strictly narrower in scope than the classical
# registry's arms: no oxygen TS-ICL arm ever uses satellite chlorophyll (no
# oxygen analogue of chlorophyll's satellite-proxy arm exists), so every
# satellite-chlorophyll column family is forbidden outright for TS-ICL
# covariates specifically, unlike the classical `external_all_available` arm
# above. Matches the private project's own
# `oxygen_pipeline.py::FORBIDDEN_SUBSTRINGS` exactly.
TSICL_FORBIDDEN_COVARIATE_SUBSTRINGS: tuple[str, ...] = (
    "chl_mean", "chl_cons", "chl_perm", "chl_anom", "chl_log10", "chl_is_gapfree",
    "BTGOXD", "BTGOXSATPC", "BTGSAL", "BTGTUR", "BTGCND",
)
SAME_STATION_AVAILABILITY_CAVEAT = (
    "Water temperature (BTGTA) and pressure (BTGPA) are measured on the same physical "
    "buoy/station as the oxygen sensor (BTGOXD2) itself. Their availability may covary with "
    "oxygen sensor outages for reasons unrelated to any genuine physical relationship -- this "
    "is why the local-BTG arm is diagnostic-only and must never be presented as a reliably "
    "available predictor during an oxygen-sensor failure."
)


@dataclass(frozen=True)
class MethodStatus:
    """One of the six method/evidentiary-status categories every oxygen
    method in this package must be labeled with."""

    name: str
    description: str


METHOD_STATUSES: dict[str, MethodStatus] = {
    "frozen_primary_benchmark": MethodStatus(
        "frozen_primary_benchmark",
        "A released, frozen result on the 406-gap primary (L1-L30) support -- the authoritative "
        "number for this method; this package does not overwrite it.",
    ),
    "frozen_exploratory_extended": MethodStatus(
        "frozen_exploratory_extended",
        "A released, frozen result restricted to the 6-gap exploratory-extended (L45-L120) "
        "support -- illustrative only, never a primary scientific claim.",
    ),
    "executable_bounded_validation": MethodStatus(
        "executable_bounded_validation",
        "Live-executable in this package on a bounded/representative subset (under the compute "
        "cap) to validate code correctness, not to produce a new headline number.",
    ),
    "exploratory_ablation": MethodStatus(
        "exploratory_ablation",
        "A single-physical-family or otherwise narrower ablation arm, evidentiary tier below the "
        "audited-original arms (e.g. the 4 Sprint-5 TS-ICL family-ablation arms).",
    ),
    "same_station_diagnostic": MethodStatus(
        "same_station_diagnostic",
        "Uses the local-BTG (water temperature/pressure) diagnostic arm -- restricted by the "
        "same-station co-missingness caveat, never a core predictor result.",
    ),
    "new_consistency_evaluation": MethodStatus(
        "new_consistency_evaluation",
        "A freshly computed check in this package (e.g. a bounded live run's agreement with the "
        "frozen result) -- evidence of code correctness, not a new scientific claim.",
    ),
}

# ── TS-ICL arm registry (headline/audited-original arms) ─────────────────
# 5 audited-original arms x 2 context modes = 10 headline rows, per
# oxygen_tsicl_audit.md F.1 -- exact arm names/columns in tsicl_models.py.
TSICL_AUDITED_ORIGINAL_ARMS = [
    "target_only", "calendar_seasonal", "external_physical_core",
    "external_physical_plus_currents", "local_btg_temp_pressure_diagnostic",
]
TSICL_CONTEXT_MODES = ["edge_balanced", "full_series"]
TSICL_FAMILY_ABLATION_ARMS = ["currents_only", "sst_thermal_only", "wind_upwelling_only", "radiation_only"]
TSICL_BEST_ARM = "external_physical_plus_currents"
TSICL_BEST_ARM_CONTEXT_MODE = "full_series"

__all__ = [
    "DATA_PUBLIC_DIR", "RESULTS_PUBLIC_DIR",
    "DAILY_TARGET_PATH", "VALIDATION_GAPS_PATH", "LOCAL_BTG_DIAGNOSTIC_FEATURES_PATH",
    "REAL_GAP_INVENTORY_PATH", "CHLOROPHYLL_BASE_FEATURES_PATH", "CURRENT_KINEMATIC_EXTENSION_PATH",
    "BENCHMARK_BY_LENGTH_PATH", "PAIRED_DELTAS_VS_TSICL_PATH",
    "TAIL_PERSISTENCE_METRICS_PATH", "TAIL_QUANTILE_BAND_METRICS_PATH",
    "SENSOR_CODE", "TARGET_COLUMN", "ELIGIBLE_COLUMN", "DATE_COLUMN", "TARGET_UNIT",
    "TARGET_TRANSFORM", "MIN_VALID_HOURS", "DAY_NIGHT_RULE",
    "PRIMARY_GAP_LENGTHS", "EXPLORATORY_GAP_LENGTHS", "ALL_GAP_LENGTHS",
    "PRIMARY_N_GAPS", "EXPLORATORY_N_GAPS", "TOTAL_N_GAPS", "PRIMARY_N_HIDDEN_DAYS",
    "SUPPORT_ROLE_PRIMARY", "SUPPORT_ROLE_EXPLORATORY", "support_role",
    "PRIMARY_METRIC", "SECONDARY_METRICS", "WEIGHTING_PRIMARY", "WEIGHTING_SECONDARY",
    "CURRENTS_ADMISSIBLE", "SATELLITE_CHLOROPHYLL_ROLE", "IN_SITU_CHLOROPHYLL_ADMISSIBLE",
    "LOCAL_BTG_TEMP_PRESSURE_ROLE", "FORBIDDEN_PREDICTOR_SUBSTRINGS",
    "TSICL_FORBIDDEN_COVARIATE_SUBSTRINGS", "SAME_STATION_AVAILABILITY_CAVEAT",
    "MethodStatus", "METHOD_STATUSES",
    "TSICL_AUDITED_ORIGINAL_ARMS", "TSICL_CONTEXT_MODES", "TSICL_FAMILY_ABLATION_ARMS",
    "TSICL_BEST_ARM", "TSICL_BEST_ARM_CONTEXT_MODE",
]
