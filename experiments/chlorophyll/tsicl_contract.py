"""The authoritative chlorophyll TS-ICL result contract.

Maps every released TS-ICL artifact this package can reproduce or compare
against to its exact support, scale, and role -- as tested Python constants,
not a documentation-only YAML file nothing else reads. Companion to
`benchmark_contract.py` (the classical/probabilistic/gap-edge contract); kept
separate because **TS-ICL artifacts do not all share one support** the way
the classical benchmark's methods mostly do.

Three supports coexist, deliberately, not by oversight:

- **Full 681-gap support** (`FULL_POOL_PATH`, same file as
  `benchmark_contract.FULL_POOL_PATH`): the primary support for the
  target-only benchmark and the covariate-arm dissection. The released
  `chlorophyll_benchmark_summary.csv`/`chlorophyll_artificial_gap_scores.csv`
  paired comparisons and `chlorophyll_covariate_mechanism_summary.csv`'s arm
  ranking are built on this support (n_gaps reported as 680-681 depending on
  which gaps a given comparator actually has a prediction for -- one L=1 gap
  is dropped from the interpolation-paired comparison specifically, see
  `MATCHED_681_DROPPED_GAP`).
- **Matched 449-gap support** (`benchmark_contract.MATCHED_SUPPORT_PATH`):
  the same support the classical/probabilistic/gap-edge benchmark uses
  (L=1,3,7,14,30 only). `tsicl_target_only`/`tsicl_satellite_proxy` both
  have a released row on this support in
  `results/chlorophyll/chlorophyll_matched_support_method_metrics.csv`,
  enabling a direct, shared-support comparison against the classical methods
  -- this is the support to use for any TS-ICL-vs-classical-ML comparison,
  not the full 681-gap support (which no classical method beyond
  interpolation/GP is scored on in this package).
- **Real-gap support** (128 real missing intervals, including the
  scenario-only 256-day gap): explicitly **out of scope for this package**.
  `chlorophyll_reconstruction_tsicl_satellite_proxy.csv` exists in
  `results/` from earlier work, but this package does not port
  the real-gap deployment driver, and nothing in this module should be read
  as implying otherwise.

Every retained TS-ICL arm is target-only or covariate-conditioned; every
retained one is a validation/artificial-gap-pool result. This module does
not cover real-gap reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = [
    "FULL_POOL_PATH",
    "MATCHED_SUPPORT_PATH",
    "BENCHMARK_SUMMARY_PATH",
    "ARTIFICIAL_GAP_SCORES_PATH",
    "COVARIATE_MECHANISM_SUMMARY_PATH",
    "MATCHED_SUPPORT_METRICS_PATH",
    "FULL_POOL_N_GAPS",
    "MATCHED_681_DROPPED_GAP",
    "TARGET_COLUMN",
    "TARGET_TRANSFORM",
    "LOG10_FLOOR",
    "MAX_CONTEXT_LENGTH",
    "QUANTILE_LEVELS",
    "PRIMARY_CONTEXT_MODES",
    "PRIMARY_CALENDAR_COLUMNS",
    "PRIMARY_PHYSICAL_FORCING_COLUMNS",
    "PRIMARY_BIOLOGICAL_PROXY_COLUMNS",
    "PRIMARY_ARMS",
    "PRIMARY_ARM_ORDER",
    "FULL_681_PRIMARY_TOTAL_CALLS",
    "FULL_681_COVARIATE_N_BASE_ARMS",
    "FULL_681_COVARIATE_N_PLACEBO_FAMILIES",
    "FULL_681_COVARIATE_N_PLACEBO_TRANSFORMS",
    "FULL_681_COVARIATE_N_VARIANTS",
    "FULL_681_COVARIATE_TOTAL_CALLS",
    "ArtifactSpec",
    "ARTIFACTS",
]

DATA_PUBLIC_DIR = REPO_ROOT / "data" / "chlorophyll"
RESULTS_PUBLIC_DIR = REPO_ROOT / "results" / "chlorophyll"

# ── Supports ─────────────────────────────────────────────────────────────
FULL_POOL_PATH = DATA_PUBLIC_DIR / "chlorophyll_validation_gaps.csv"
MATCHED_SUPPORT_PATH = DATA_PUBLIC_DIR / "chlorophyll_matched_support_449.csv"
FULL_POOL_N_GAPS = 681

# The one L=1 gap present in the full pool but absent from the frozen
# interpolation-paired comparison (n_gaps=680, not 681, for every
# TS-ICL-vs-interpolation row in `chlorophyll_benchmark_summary.csv`) --
# interpolation itself has no bracketing observation on one side for this
# gap, so no paired delta can be formed for it. Not a TS-ICL exclusion.
MATCHED_681_DROPPED_GAP = "L01_20150715"

# ── Frozen result tables (already public) ───────────────────────────────
BENCHMARK_SUMMARY_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_benchmark_summary.csv"
ARTIFICIAL_GAP_SCORES_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_artificial_gap_scores.csv"
COVARIATE_MECHANISM_SUMMARY_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_covariate_mechanism_summary.csv"
MATCHED_SUPPORT_METRICS_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_matched_support_method_metrics.csv"

# ── Target/transform/model configuration (from the verified private provenance) ──
TARGET_COLUMN = "chl_mean"
TARGET_TRANSFORM = "log10"
LOG10_FLOOR = 1e-4
MAX_CONTEXT_LENGTH = 4096
QUANTILE_LEVELS: list[float] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]

# The two context-construction modes the released primary target-only/
# covariate benchmark actually used (a third, "local_window", was skipped in
# the original primary run for runtime-budget reasons -- not
# reproduced here either, for the same reason).
PRIMARY_CONTEXT_MODES: list[str] = ["full_series", "edge_balanced"]

# ── The authoritative primary full benchmark grid ────────────────────────
#
# Resolved by direct inspection of the original primary TS-ICL
# benchmark driver (not assumed/inferred from earlier, inconsistent
# documentation): the released primary target-only/covariate
# benchmark is **681 gaps x 2 context modes x 6 primary arms**,
# `target_repr="raw"` only -- 8,172 calls total. This corrects
# an earlier public driver that implemented only 2 of these 6 arms
# (`target_only`=A, `satellite_proxy`=D) under the `target_only`-benchmark
# label; the other 4 (calendar, physical-forcing, physical-forcing+proxy,
# and a `wrong_lag` placebo control) were previously missing from
# `run_tsicl_benchmark.py` entirely.
#
# Column lists are ported verbatim from the original shared
# call-shaping module's own physical-forcing/biological-proxy/calendar
# column lists (the same original source `tsicl_covariate_registry.py`'s
# 18-arm covariate-dissection column lists were ported from).
PRIMARY_CALENDAR_COLUMNS: list[str] = ["doy_sin", "doy_cos"]
PRIMARY_PHYSICAL_FORCING_COLUMNS: list[str] = [
    "plv_solar_wm2", "plv_solar_roll7d_wm2", "plv_upwelling_ms", "plv_upwelling_cumul7d_ms_d",
    "sst_primary_degC", "sst_primary_degC_roll7", "wind_spd_ms",
]
PRIMARY_BIOLOGICAL_PROXY_COLUMNS: list[str] = ["chl_cons_w3x3_mean", "chl_anom_log10_doy"]

# arm_id (public, descriptive) -> (internal code, columns, placebo transform).
# `target_only` (A) and `satellite_proxy` (D) are the two arms the earlier
# driver already implemented; the remaining four complete the actual
# 6-arm primary grid.
PRIMARY_ARMS: dict[str, dict] = {
    "target_only": {"private_code": "A", "columns": [], "transform": "none"},
    "target_plus_calendar": {"private_code": "B", "columns": PRIMARY_CALENDAR_COLUMNS, "transform": "none"},
    "target_plus_physical_forcing": {
        "private_code": "C", "columns": PRIMARY_PHYSICAL_FORCING_COLUMNS, "transform": "none",
    },
    "satellite_proxy": {"private_code": "D", "columns": PRIMARY_BIOLOGICAL_PROXY_COLUMNS, "transform": "none"},
    "target_plus_physical_forcing_plus_proxy": {
        "private_code": "E",
        "columns": PRIMARY_PHYSICAL_FORCING_COLUMNS + PRIMARY_BIOLOGICAL_PROXY_COLUMNS,
        "transform": "none",
    },
    "wrong_lag_physical_forcing": {
        "private_code": "F", "columns": PRIMARY_PHYSICAL_FORCING_COLUMNS, "transform": "wrong_lag",
    },
}
PRIMARY_ARM_ORDER: list[str] = [
    "target_only", "target_plus_calendar", "target_plus_physical_forcing", "satellite_proxy",
    "target_plus_physical_forcing_plus_proxy", "wrong_lag_physical_forcing",
]
FULL_681_PRIMARY_TOTAL_CALLS = FULL_POOL_N_GAPS * len(PRIMARY_CONTEXT_MODES) * len(PRIMARY_ARM_ORDER)  # 8,172

# The authoritative full covariate dissection grid: 18 base arms (this
# module's `covariate_arm_ranking` artifact, exact column membership in
# `tsicl_covariate_registry.COVARIATE_ARMS`) + 4 placebo transforms on each
# of 6 eligible families (`PLACEBO_ELIGIBLE_ARMS`) = 18 + 24 = 42 variants,
# `context_mode="full_series"` only -- resolved from
# the original covariate-dissection run-plan builder (18 arms in
# `minimum_required`+`remaining_c11`+`[C4,C7,C8]`, 6 placebo families x 4
# transforms = 24), matching that module's own "28,602 calls" docstring
# claim exactly (681 x 42 = 28,602) rather than trusting it uninspected.
FULL_681_COVARIATE_N_BASE_ARMS = 18
FULL_681_COVARIATE_N_PLACEBO_FAMILIES = 6
FULL_681_COVARIATE_N_PLACEBO_TRANSFORMS = 4
FULL_681_COVARIATE_N_VARIANTS = (
    FULL_681_COVARIATE_N_BASE_ARMS
    + FULL_681_COVARIATE_N_PLACEBO_FAMILIES * FULL_681_COVARIATE_N_PLACEBO_TRANSFORMS
)  # 42
FULL_681_COVARIATE_TOTAL_CALLS = FULL_POOL_N_GAPS * FULL_681_COVARIATE_N_VARIANTS  # 28,602


@dataclass(frozen=True)
class ArtifactSpec:
    """One row of the TS-ICL result registry.

    `role` distinguishes artifacts this package can regenerate and compare
    against (`primary`), artifacts used only as supporting evidence for a
    primary one (`supporting`), and artifacts explicitly out of scope for
    this package (`out_of_scope`).
    """

    name: str
    support: str  # "full_681" | "matched_449" | "real_gap"
    scale: str
    role: str
    source_path: Path | None
    description: str


ARTIFACTS: dict[str, ArtifactSpec] = {
    "target_only_full_681": ArtifactSpec(
        "target_only_full_681", "full_681", "log10", "primary", ARTIFICIAL_GAP_SCORES_PATH,
        "TS-ICL target-only day-level scores (arm A, full_series + edge_balanced context), "
        "full 681-gap pool -- the target-only comparator in chlorophyll_benchmark_summary.csv.",
    ),
    "satellite_proxy_full_681": ArtifactSpec(
        "satellite_proxy_full_681", "full_681", "log10", "primary", ARTIFICIAL_GAP_SCORES_PATH,
        "TS-ICL + satellite chlorophyll proxy covariate (arm D), full 681-gap pool -- the "
        "headline foundation-model result, CI-backed superiority over interpolation/GP/gap-edge.",
    ),
    "target_only_matched_449": ArtifactSpec(
        "target_only_matched_449", "matched_449", "log10", "primary", MATCHED_SUPPORT_METRICS_PATH,
        "TS-ICL target-only, matched-449 support -- directly comparable to the classical/"
        "probabilistic methods in benchmark_contract.py on the same gap set.",
    ),
    "satellite_proxy_matched_449": ArtifactSpec(
        "satellite_proxy_matched_449", "matched_449", "log10", "primary", MATCHED_SUPPORT_METRICS_PATH,
        "TS-ICL + satellite proxy, matched-449 support.",
    ),
    "covariate_arm_ranking": ArtifactSpec(
        "covariate_arm_ranking", "full_681", "log10", "primary", COVARIATE_MECHANISM_SUMMARY_PATH,
        "18-arm covariate performance ranking (descriptive public names) plus the "
        "placebo-robustness paired-bootstrap row -- see tsicl_covariate_registry.py for the "
        "exact column membership of each arm.",
    ),
    "paired_deltas_full_681": ArtifactSpec(
        "paired_deltas_full_681", "full_681", "log10", "primary", BENCHMARK_SUMMARY_PATH,
        "Paired day-weighted-MAE deltas + gap-cluster bootstrap 95% CIs, TS-ICL arms vs. "
        "interpolation/GP/gap-edge/each other, full pool.",
    ),
    "by_length_full_681": ArtifactSpec(
        "by_length_full_681", "full_681", "log10", "supporting", ARTIFICIAL_GAP_SCORES_PATH,
        "Day-weighted MAE by gap length, per method, full pool.",
    ),
    "real_gap_satellite_proxy": ArtifactSpec(
        "real_gap_satellite_proxy", "real_gap", "log10", "out_of_scope",
        RESULTS_PUBLIC_DIR / "chlorophyll_reconstruction_tsicl_satellite_proxy.csv",
        "Real-gap (128 gaps, including the scenario-only 256-day gap) reconstruction candidate "
        "output. Exists in results/ from earlier work; this package does not port "
        "the real-gap deployment driver and this artifact is not touched or reproduced here.",
    ),
}
