"""The authoritative chlorophyll real-gap publication contract: tested
Python constants/dataclasses distinguishing every kind of "gap" this
project produces evidence about, and every kind of artifact built around
real gaps -- not a YAML file nothing else reads.

**The distinction this module exists to keep unambiguous**: an artificial
validation gap has a withheld, known true value (the whole point of the
benchmark). A real gap does not -- nothing in this project ever observed
what chlorophyll was doing during a real missing interval. Every real-gap
"reconstruction" in this package is therefore a **candidate**, produced by
a method whose *general* skill was measured on artificial gaps, never a
validated value for that specific missing interval. This module's
`ArtifactRole` enum and per-artifact roles exist specifically so no code or
docstring downstream can silently blur that line.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DATA_PUBLIC_DIR = REPO_ROOT / "data" / "chlorophyll"
RESULTS_PUBLIC_DIR = REPO_ROOT / "results" / "chlorophyll"

# ── Data paths ───────────────────────────────────────────────────────────
DAILY_TARGET_PATH = DATA_PUBLIC_DIR / "chlorophyll_daily_target.csv"
REAL_GAP_INVENTORY_PATH = DATA_PUBLIC_DIR / "chlorophyll_real_gap_inventory.csv"
ENGINEERED_HYBRID_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_reconstruction_engineered_hybrid.csv"
TSICL_SATELLITE_PROXY_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_reconstruction_tsicl_satellite_proxy.csv"
CANDIDATE_OUTPUTS_GAP_LEVEL_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_real_gap_candidate_outputs.csv"
CANDIDATE_OUTPUTS_DAILY_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_real_gap_candidate_outputs_daily.csv"

# ── Target/QC (shared with the rest of the chlorophyll case study) ───────
TARGET_COLUMN = "chl_mean"
ELIGIBLE_COLUMN = "target_eligible_default"
DATE_COLUMN = "date"

# ── Evidence-kind vocabulary ───────────────────────────────────────────────
#
# These are deliberately distinct concepts, never conflated:
ARTIFICIAL_VALIDATION_GAP = "artificial_validation_gap"      # withheld, known true value -- validation evidence
OBSERVED_REAL_MISSING_INTERVAL = "observed_real_missing_interval"  # a real gap; no known true value
RECONSTRUCTION_CANDIDATE = "reconstruction_candidate"          # one method's output for a real gap; not truth
METHOD_SELECTED_CANDIDATE = "method_selected_candidate"        # a candidate chosen by a deterministic length-routed rule
SCENARIO_ONLY_OUTPUT = "scenario_only_output"                  # the 256-day gap's output specifically -- outside validated support


@dataclass(frozen=True)
class ArtifactStatus:
    """One of the eight status categories every published real-gap artifact
    or code path in this package must be labeled with."""

    name: str
    description: str


ARTIFACT_STATUSES: dict[str, ArtifactStatus] = {
    "artificial_validation_gap": ArtifactStatus(
        "artificial_validation_gap",
        "A withheld-and-known-true-value gap from the canonical validation pool "
        "(experiments.chlorophyll.benchmark_contract) -- the only evidence in this project scored "
        "against withheld observations. Not what this module is about, but the category everything else here "
        "is explicitly NOT.",
    ),
    "observed_real_missing_interval": ArtifactStatus(
        "observed_real_missing_interval",
        "A contiguous run of non-eligible days in the released daily target -- an actual, "
        "observed data gap. Has no known true value.",
    ),
    "reconstruction_candidate": ArtifactStatus(
        "reconstruction_candidate",
        "One method's output for a real gap's hidden days. Produced by a method whose general "
        "skill was measured on artificial gaps; not itself validated for this specific interval.",
    ),
    "method_selected_candidate": ArtifactStatus(
        "method_selected_candidate",
        "A candidate chosen for a given real gap by a deterministic, length-routed selection "
        "rule (see select_real_gap_reconstruction.py) -- still a candidate, not observed truth, "
        "even though exactly one method's output is picked per gap.",
    ),
    "scenario_only": ArtifactStatus(
        "scenario_only",
        "The 256-day (2020) gap's output specifically. Outside the validated artificial-gap "
        "length envelope (max validated L<=60/L<=90 depending on method) -- illustrative only, "
        "never evidence of validated 256-day reconstruction accuracy.",
    ),
    "frozen_published_result": ArtifactStatus(
        "frozen_published_result",
        "Already released in results/ or data/ -- authoritative, not regenerated "
        "or overwritten by this package's code by default.",
    ),
    "executable_deterministic_assembly": ArtifactStatus(
        "executable_deterministic_assembly",
        "Code in this package that can be run now: pure joining/validation/inventory logic over "
        "already-frozen inputs, no model fitting or inference.",
    ),
    "expensive_model_generation_not_rerun": ArtifactStatus(
        "expensive_model_generation_not_rerun",
        "Would require new TS-ICL inference or model training to regenerate -- explicitly out of "
        "scope for this package's compute boundary; the frozen output remains authoritative.",
    ),
}


@dataclass(frozen=True)
class RealGapArtifactSpec:
    """Everything a caller needs to know about one published real-gap
    artifact before touching it -- especially whether it is a candidate or
    an observation."""

    name: str
    source_path: Path
    n_real_gaps: int | None  # None if the artifact is day-level, not gap-level
    generating_method: str
    target_scale: str  # "physical_mg_m3" | "log10" | "both"
    point_prediction_column: str | None
    quantile_columns: tuple[str, ...]
    is_candidate_not_observation: bool
    used_in_report_or_demo: bool
    source_candidate_tables: tuple[str, ...]  # which raw per-method file(s) this was assembled from
    role: str  # one of ARTIFACT_STATUSES keys
    description: str


REAL_GAP_ARTIFACTS: dict[str, RealGapArtifactSpec] = {
    "real_gap_inventory": RealGapArtifactSpec(
        "real_gap_inventory", REAL_GAP_INVENTORY_PATH, 128, "n/a (inventory only, no prediction)",
        "n/a", None, (), False, True, (),
        "observed_real_missing_interval",
        "128 observed real missing intervals in the daily chl_mean series (including the "
        "scenario-only 256-day gap), with edge availability/admissibility metadata. No "
        "prediction of any kind -- purely descriptive of where and how long the real gaps are.",
    ),
    "engineered_hybrid_reconstruction": RealGapArtifactSpec(
        "engineered_hybrid_reconstruction", ENGINEERED_HYBRID_PATH, 128,
        "length-routed rule: GP M1 (L1-3) -> state-space Kalman (L4-29) -> gap-edge residual "
        "ExtraTrees (L>=30)",
        "physical_mg_m3", "final_chl", (),
        True, True, (),
        "method_selected_candidate",
        "Full daily series (3988 rows) with one candidate reconstruction per real-gap hidden "
        "day, method assigned deterministically by gap length (see "
        "select_real_gap_reconstruction.py). Includes a Rule-C sensitivity-variant column. "
        "Applied to all 128 real gaps including the 256-day scenario.",
    ),
    "tsicl_satellite_proxy_reconstruction": RealGapArtifactSpec(
        "tsicl_satellite_proxy_reconstruction", TSICL_SATELLITE_PROXY_PATH, 128,
        "TS-ICL zero-shot foundation model, satellite chlorophyll proxy covariate",
        "both", "pred_chl_mg_m3",
        ("q05_chl_mg_m3", "q10_chl_mg_m3", "q25_chl_mg_m3", "q50_chl_mg_m3",
         "q75_chl_mg_m3", "q90_chl_mg_m3", "q95_chl_mg_m3"),
        True, True, (),
        "reconstruction_candidate",
        "Hidden-day-only rows (976 = sum of all 128 real gaps' lengths) with point + 7-quantile "
        "predictions in both log10 and physical scale, one independent candidate per real gap "
        "including the 256-day scenario (flagged scenario_only_256day).",
    ),
    "candidate_outputs_gap_level": RealGapArtifactSpec(
        "candidate_outputs_gap_level", CANDIDATE_OUTPUTS_GAP_LEVEL_PATH, 128,
        "assembly of engineered_hybrid_reconstruction + tsicl_satellite_proxy_reconstruction",
        "physical_mg_m3", None, (),
        True, True,
        ("engineered_hybrid_reconstruction", "tsicl_satellite_proxy_reconstruction"),
        "reconstruction_candidate",
        "One row per real gap (128), both candidates' mean prediction side by side, with "
        "explicit real-gap-caveat and artificial-validation-support notes on every row.",
    ),
    "candidate_outputs_daily": RealGapArtifactSpec(
        "candidate_outputs_daily", CANDIDATE_OUTPUTS_DAILY_PATH, None,
        "assembly of engineered_hybrid_reconstruction + tsicl_satellite_proxy_reconstruction",
        "physical_mg_m3", None, (),
        True, True,
        ("engineered_hybrid_reconstruction", "tsicl_satellite_proxy_reconstruction"),
        "reconstruction_candidate",
        "Full daily series (3988 rows) with both candidates joined day-by-day, an "
        "evidence_label column, and an explicit scenario_only_256day_gap flag.",
    ),
}


def is_candidate_never_observation(artifact_name: str) -> bool:
    """True for every real-gap artifact in this registry -- there is no
    artifact in this package that represents an observed real-gap value.
    Exists as a single, testable assertion point rather than a scattered
    convention."""
    return REAL_GAP_ARTIFACTS[artifact_name].is_candidate_not_observation


__all__ = [
    "DATA_PUBLIC_DIR", "RESULTS_PUBLIC_DIR", "DAILY_TARGET_PATH", "REAL_GAP_INVENTORY_PATH",
    "ENGINEERED_HYBRID_PATH", "TSICL_SATELLITE_PROXY_PATH",
    "CANDIDATE_OUTPUTS_GAP_LEVEL_PATH", "CANDIDATE_OUTPUTS_DAILY_PATH",
    "TARGET_COLUMN", "ELIGIBLE_COLUMN", "DATE_COLUMN",
    "ARTIFICIAL_VALIDATION_GAP", "OBSERVED_REAL_MISSING_INTERVAL", "RECONSTRUCTION_CANDIDATE",
    "METHOD_SELECTED_CANDIDATE", "SCENARIO_ONLY_OUTPUT",
    "ArtifactStatus", "ARTIFACT_STATUSES",
    "RealGapArtifactSpec", "REAL_GAP_ARTIFACTS", "is_candidate_never_observation",
]
