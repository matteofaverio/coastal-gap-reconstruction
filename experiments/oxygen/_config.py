"""Tested Python constants for the oxygen case study.

Mirrors `experiments.chlorophyll._config`'s role: values a YAML "contract" would
otherwise hold, imported directly and asserted by tests instead.
"""

from __future__ import annotations

TARGET_COL = "oxygen_mean_mgL"
ELIGIBLE_COL = "eligible_ge_18"
DATE_COL = "date"

MANDATORY_GAP_LENGTHS: list[int] = [1, 3, 7, 10, 14, 21, 30, 45, 60]
EXPLORATORY_GAP_LENGTHS: list[int] = [90, 120]

RANDOM_SEED = 42

# Aspirational per-length caps -- mirror chlorophyll's own canonical achieved counts;
# actual achieved counts may be lower, capped by how many context-qualified
# non-overlapping candidates exist for oxygen's own eligible-day structure.
TARGET_ACHIEVED_COUNTS: dict[int, int] = {
    1: 100, 3: 100, 7: 100, 10: 100, 14: 100, 21: 80, 30: 50, 45: 29, 60: 22,
}

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def context_days_required(gap_length: int) -> int:
    """Minimum eligible-and-observed context each candidate must have on both sides,
    within the same contiguous eligible run, to even be considered a candidate
    (unlike chlorophyll, this is a candidacy filter here, not only a post-hoc
    label -- see `target_and_gap_pool.py`'s module docstring)."""
    return 30 if gap_length <= 30 else 60


def support_role(gap_length: int) -> str:
    """Public benchmark-reporting label: which gap lengths form the primary
    reported benchmark vs. which are exploratory extensions.

    This label does not exist anywhere in the private project's own generator --
    it was assigned only when preparing the public release, independently of the
    `is_mandatory` flag the private generator does produce (`is_mandatory` is True
    for every length in MANDATORY_GAP_LENGTHS, including 45 and 60; `support_role`
    draws the primary/exploratory line at a different point, L<=30 vs L>=45).
    """
    return "primary" if gap_length <= 30 else "exploratory_extended"
