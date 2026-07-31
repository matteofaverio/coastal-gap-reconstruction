"""Tested Python constants for the chlorophyll case study.

These values replace what would otherwise be a YAML "contract" file: they are
imported directly by the code that uses them and asserted by tests, rather than
being parsed from a configuration file that nothing else reads.
"""

from __future__ import annotations

TARGET_COL = "chl_mean"
ELIGIBLE_COL = "target_eligible_default"
DATE_COL = "date"

# The released validation pool's exact gap-length support. There is no shorter
# "default" list -- every caller must state explicitly which lengths it wants.
GAP_LENGTHS: list[int] = [1, 3, 7, 10, 14, 21, 30, 45, 60]

# Of GAP_LENGTHS, this subset is exactly regenerable from the current target table
# with the shared sampling algorithm (`coastal_gap_reconstruction.gaps`) and a fixed
# seed -- verified byte-identical against the released pool row-for-row. The
# remaining lengths (10, 21, 45, 60) were assembled from a separate, later
# extension pass in the private project's history and are not reproducible with
# this algorithm alone; see the module docstring of `target_and_gap_pool.py`.
EXACTLY_REGENERABLE_GAP_LENGTHS: list[int] = [1, 3, 7, 14, 30]

RANDOM_SEED = 42
MAX_GAPS_PER_LENGTH = 100

# is_sustained_event threshold, mg/m^3. Reused verbatim from the private project's
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
