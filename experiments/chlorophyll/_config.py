"""Tested Python constants for the chlorophyll case study.

These values replace what would otherwise be a YAML "contract" file: they are
imported directly by the code that uses them and asserted by tests, rather than
being parsed from a configuration file that nothing else reads.
"""

from __future__ import annotations

from coastal_gap_reconstruction.target_spec import TargetSpec

TARGET_COL = "chl_mean"
ELIGIBLE_COL = "target_eligible_default"
DATE_COL = "date"

# The released validation pool's exact gap-length support. There is no shorter
# "default" list -- every caller must state explicitly which lengths it wants.
GAP_LENGTHS: list[int] = [1, 3, 7, 10, 14, 21, 30, 45, 60]

# GAP_LENGTHS splits into two subsets, built by two different, non-interchangeable
# procedures over the course of this project's development (see target_and_gap_pool.py's
# module docstring for the full story and verification evidence):
#
# - CORE_GAP_LENGTHS: the original lengths, sampled independently per length with
#   coastal_gap_reconstruction.gaps.sample_nonoverlapping (Python `random.Random`,
#   fresh per length).
# - EXTENDED_GAP_LENGTHS: added in a later pass, sampled with
#   coastal_gap_reconstruction.gaps.sample_nonoverlapping_sequential (a single
#   shared numpy.random.Generator, advanced across lengths in order), over a
#   stricter candidate universe (eligible AND target value > LOG10_FLOOR, not
#   merely eligible), with per-length caps that are not uniform.
#
# Both subsets are exactly regenerable; EXACTLY_REGENERABLE_GAP_LENGTHS = GAP_LENGTHS.
CORE_GAP_LENGTHS: list[int] = [1, 3, 7, 14, 30]
EXTENDED_GAP_LENGTHS: list[int] = [10, 21, 45, 60]
EXACTLY_REGENERABLE_GAP_LENGTHS: list[int] = GAP_LENGTHS

RANDOM_SEED = 42
MAX_GAPS_PER_LENGTH = 100  # core lengths only, applied independently per length

# Extended-length candidate universe additionally requires target value > this
# floor (not merely eligible/non-null) -- see find_positions_with_value_floor.
EXTENDED_VALUE_FLOOR = 1e-4

# Extended-length per-length caps, applied while advancing one shared RNG in this
# exact order (10, 21, 45, 60) -- order matters, it is not merely a lookup table.
EXTENDED_MAX_CANDIDATES: dict[int, int] = {10: 100, 21: 102, 45: 37, 60: 25}

# is_sustained_event threshold, mg/m^3. Reused verbatim from the original
# gap-edge feature construction; not re-derived here.
SUSTAINED_MEAN_THRESHOLD = 9.9022

REGIME = "strict_observed_only"


def required_context_days(gap_length: int) -> int:
    """Minimum pre/post context (in eligible days) a gap of this length must have
    on each side before it is flagged `context_constrained`.

    Exact rule recovered from the released pool: 30 days for gap_length <= 30,
    60 days for anything longer. Verified against all 681 released rows -- do not
    replace with an approximate flat threshold.
    """
    return 30 if gap_length <= 30 else 60


CONTEXT_CAP_DAYS = 60  # max(required_context_days(L) for L in GAP_LENGTHS)

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _is_high_chl_event(hidden_values, threshold: float) -> bool:
    """True iff any hidden day's chl_mean strictly exceeds the 90th percentile
    threshold (recomputed fresh per pool -- see target_and_gap_pool.py)."""
    if not hidden_values.notna().any():
        return False
    return bool(hidden_values.max() > threshold)


TARGET_SPEC = TargetSpec(
    name="chlorophyll",
    target_col=TARGET_COL,
    eligible_col=ELIGIBLE_COL,
    date_col=DATE_COL,
    display_unit="mg/m^3",
    benchmark_scoring_scale="log10",
    positive_only=True,
    event_label=_is_high_chl_event,
)
