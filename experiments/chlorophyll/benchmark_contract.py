"""The authoritative chlorophyll classical/probabilistic/gap-edge benchmark contract.

This module is the single place that states, as tested Python constants, exactly
which gaps every headline chlorophyll comparator is scored on and how. It does not
compute a validation pool itself (that is `target_and_gap_pool.py`, which builds
the full 681-gap pool from the daily target table); it only *selects* and *labels*
the subsets of that pool used by the classical-ML benchmark in this package.

Two supports coexist deliberately, not by oversight:

- **Full pool** (681 gaps, `GAP_LENGTHS` = 1,3,7,10,14,21,30,45,60): used by TS-ICL
  calibration/covariate diagnostics and any other explicitly full-support analysis.
  Not this module's concern beyond re-exporting the path.
- **Matched support** (449 gaps, `MATCHED_SUPPORT_GAP_LENGTHS` = 1,3,7,14,30): the
  exact gap set every headline non-TS-ICL comparator in this package (external
  tabular, gap-edge residual, Gaussian process, engineered hybrid) is evaluated on.
  This is the original `tier_ch_deployed`-exact support (frozen
  2026-07-08, see `docs/methods.md` for the full provenance note) -- it exists
  because the gap-edge residual model's hindcast feature construction only ever
  ran on the five original ("core") gap lengths, so any fair comparison across
  methods must restrict to gaps that every method actually has a prediction for.

The matched-support gap-ID list is frozen (`data/chlorophyll/
chlorophyll_matched_support_449.csv`) rather than re-derived at import time: it is
exactly the intersection described above, computed once against the
original per-gap prediction tables, and re-deriving it here would require
re-running every model first -- a chicken-and-egg the frozen file avoids. What this
module *does* test is that the frozen 449 IDs are all members of the full 681-gap
pool and that they cover exactly the five core lengths with the released per-length
counts (`test_benchmark_contract.py`).

**Event flag: `is_high_chl_event`, 90th-percentile threshold -- not an 85th-percentile
flag.** The matched-support manifest's event column is the same boolean as the full
pool's `is_high_chl_event` (True iff any hidden day's chl_mean exceeds the 90th
percentile of eligible days, recomputed fresh at pool-build time -- see
`docs/methods.md`). An earlier private aggregation step renamed
this same p90-based column to `event_p85` while attaching it to the matched-support
gap-ID table -- a naming artifact in that one step, not a distinct p85-threshold
computation; no second, differently-thresholded event flag exists anywhere in the
pipeline. This
column was **not** used to select the 449 matched-support gaps (selection is purely
the cross-method gap-ID intersection above); it is carried along only for
event/non-event stratified reporting. This package's matched-support file uses the
correct name `is_high_chl_event` throughout (the public CSV header was corrected from
an earlier `event_p85` naming artifact to match).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import _config

# Repository root, resolved the same way as the existing test suite
# (tests/test_gap_pool_regeneration.py) -- there is no shared `_paths` module in
# this package yet, so every driver resolves it locally rather than assuming one.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

__all__ = [
    "FULL_POOL_PATH",
    "MATCHED_SUPPORT_PATH",
    "MATCHED_SUPPORT_METRICS_PATH",
    "MATCHED_SUPPORT_BY_LENGTH_PATH",
    "GAP_LENGTHS",
    "MATCHED_SUPPORT_GAP_LENGTHS",
    "MATCHED_SUPPORT_N_GAPS",
    "MATCHED_SUPPORT_N_DAY_ROWS",
    "RANDOM_SEED",
    "N_BOOTSTRAP_REPLICATES",
    "TARGET_SPEC",
    "MethodSpec",
    "METHODS",
    "load_matched_support_gap_ids",
    "load_full_pool",
    "load_matched_support_pool",
]

DATA_PUBLIC_DIR = REPO_ROOT / "data" / "chlorophyll"
RESULTS_PUBLIC_DIR = REPO_ROOT / "results" / "chlorophyll"

FULL_POOL_PATH = DATA_PUBLIC_DIR / "chlorophyll_validation_gaps.csv"
MATCHED_SUPPORT_PATH = DATA_PUBLIC_DIR / "chlorophyll_matched_support_449.csv"
MATCHED_SUPPORT_METRICS_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_matched_support_method_metrics.csv"
MATCHED_SUPPORT_BY_LENGTH_PATH = RESULTS_PUBLIC_DIR / "chlorophyll_matched_support_by_length.csv"

GAP_LENGTHS: list[int] = _config.GAP_LENGTHS
MATCHED_SUPPORT_GAP_LENGTHS: list[int] = [1, 3, 7, 14, 30]

MATCHED_SUPPORT_N_GAPS = 449
MATCHED_SUPPORT_N_DAY_ROWS = 3999

# Released per-length gap counts on the matched support (frozen, not recomputed).
MATCHED_SUPPORT_COUNTS_BY_LENGTH: dict[int, int] = {1: 99, 3: 100, 7: 100, 14: 100, 30: 50}

RANDOM_SEED = _config.RANDOM_SEED
N_BOOTSTRAP_REPLICATES = 2000

TARGET_SPEC = _config.TARGET_SPEC

# Scoring: every headline metric in this package is day-weighted MAE on the
# log10(chl_mean) scale, matching TARGET_SPEC.benchmark_scoring_scale.
SCORING_SCALE = TARGET_SPEC.benchmark_scoring_scale
PRIMARY_METRIC = "day_weighted_mae"


@dataclass(frozen=True)
class MethodSpec:
    """One row of the benchmark's method registry.

    `method_id` is the internal/programmatic key; `public_name` is the exact
    label used in figures/tables built from this registry. `role`
    distinguishes methods this package actually implements a driver for
    (`classical_benchmark`) from methods scored elsewhere and not ported here
    (`external_reference`, e.g. TS-ICL, out of scope for this task).

    `support_status` states, precisely, this method's relationship to the
    frozen released matched-support table
    (`results/chlorophyll/chlorophyll_matched_support_method_metrics.csv`):

    - `"frozen_matched_449"`: this `method_id` has a row in that table, and
      this package's implementation is expected to reproduce it (within the
      classification `run_classical_benchmark.py --verify` reports).
    - `"new_evaluation_on_matched_449"`: this method is run on the same
      449-gap support for comparability, but **has no row in the frozen
      table** -- there is no released number to reproduce, and
      `--verify` must report it `not_applicable`, never `not_reproduced`.
    - `"external_reference"`: not implemented by this package at all (TS-ICL).
    """

    method_id: str
    public_name: str
    role: str
    canonical_learner: str | None = None
    support_status: str = "frozen_matched_449"


METHODS: dict[str, MethodSpec] = {
    "canonical_interpolation": MethodSpec(
        "canonical_interpolation", "Linear interpolation", "classical_benchmark", None,
        "frozen_matched_449",
    ),
    "gp_m1": MethodSpec(
        "gp_m1", "Gaussian process", "classical_benchmark",
        "GaussianProcessRegressor (Matern 3/2, time-only)", "frozen_matched_449",
    ),
    "ext_tabular_extratrees": MethodSpec(
        "ext_tabular_extratrees",
        "External tabular (ExtraTrees)",
        "classical_benchmark",
        "ExtraTreesRegressor(n_estimators=500); matched-reference protocol -- external "
        "(arm4) + 5 gap-position meta-features, strict pre-only dependency-window LOCO. "
        "See gap_edge_models.run_reference_arm_loco_evaluation. NOT the same protocol as "
        "external_only_extratrees (plain external-only, no gap-position features).",
        "frozen_matched_449",
    ),
    "ext_tabular_hgb": MethodSpec(
        "ext_tabular_hgb",
        "External tabular (HGB)",
        "classical_benchmark",
        "HistGradientBoostingRegressor; matched-reference protocol (diagnostic comparator, "
        "not the canonical learner of this protocol -- ExtraTrees is).",
        "frozen_matched_449",
    ),
    "external_only_extratrees": MethodSpec(
        "external_only_extratrees",
        "External tabular, plain external-only protocol (ExtraTrees)",
        "classical_benchmark",
        "ExtraTreesRegressor(n_estimators=500); external (arm4) only, no gap-position "
        "features -- see tabular_models.run_loco_evaluation",
        "new_evaluation_on_matched_449",
    ),
    "external_only_hgb": MethodSpec(
        "external_only_hgb",
        "External tabular, plain external-only protocol (HGB)",
        "classical_benchmark",
        "HistGradientBoostingRegressor (diagnostic comparator)",
        "new_evaluation_on_matched_449",
    ),
    "tier_ch_deployed": MethodSpec(
        "tier_ch_deployed",
        "Gap-edge residual model",
        "classical_benchmark",
        "ExtraTreesRegressor(n_estimators=500), residual-over-interpolation",
        "frozen_matched_449",
    ),
    "engineered_hybrid": MethodSpec(
        "engineered_hybrid",
        "Engineered hybrid pipeline (validation-aware method assignment)",
        "classical_benchmark",
        "length-routed GP -> Kalman -> gap-edge assignment",
        "new_evaluation_on_matched_449",
    ),
    "tsicl_target_only": MethodSpec(
        "tsicl_target_only", "TS-ICL target-only", "external_reference", None,
        "frozen_matched_449",
    ),
    "tsicl_satellite_proxy": MethodSpec(
        "tsicl_satellite_proxy", "TS-ICL satellite-proxy", "external_reference", None,
        "frozen_matched_449",
    ),
}


def load_matched_support_gap_ids() -> pd.DataFrame:
    """Load the frozen 449-row matched-support gap-ID table.

    Columns: gap_id, gap_length, season, is_high_chl_event, n_days.
    """
    return pd.read_csv(MATCHED_SUPPORT_PATH)


def load_full_pool() -> pd.DataFrame:
    """Load the full 681-row released artificial-gap pool."""
    return pd.read_csv(FULL_POOL_PATH, parse_dates=["start_date", "end_date"])


def load_matched_support_pool() -> pd.DataFrame:
    """Return the full pool filtered to exactly the 449 matched-support gap_ids.

    Row order and all 18 full-pool columns are preserved; this is a row filter,
    not a re-derivation.
    """
    full = load_full_pool()
    matched_ids = set(load_matched_support_gap_ids()["gap_id"])
    out = full[full["gap_id"].isin(matched_ids)].reset_index(drop=True)
    if len(out) != MATCHED_SUPPORT_N_GAPS:
        raise ValueError(
            f"Matched-support filter produced {len(out)} rows, expected "
            f"{MATCHED_SUPPORT_N_GAPS}. The frozen gap-ID file and the full pool "
            f"have diverged -- do not silently proceed."
        )
    return out


def matched_support_path_for(filename: Path | str) -> Path:  # pragma: no cover - trivial
    return DATA_PUBLIC_DIR / filename
