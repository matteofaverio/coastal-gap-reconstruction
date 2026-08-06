"""Oxygen target loading and artificial-gap pool construction.

Mirrors `experiments.chlorophyll.target_and_gap_pool`'s role but for dissolved
oxygen, on top of the same target-neutral mechanics in
`coastal_gap_reconstruction.gaps`. The two builders are intentionally not
identical: oxygen's candidate search requires context (pre/post eligible days)
as a hard candidacy filter (`gaps.find_context_qualified_positions`), while
chlorophyll's only labels context availability after selection
(`gaps.find_candidate_positions` + `gaps.count_context_days`) -- this mirrors a
genuine, intentional design difference in the original implementation, not an
inconsistency to "fix" into agreement.

Reproducibility boundary: the released pool
(`data/oxygen/oxygen_validation_gaps.csv`, 412 rows) is exactly
regenerable from the public daily oxygen target table with the algorithm in this
module -- `build_gap_pool` reproduces all 412 rows and their `is_mandatory`
column byte-for-byte, verified against the released CSV. The `support_role`
column does not exist in the original generator at all; it was
assigned only when preparing the public release (primary for L<=30, exploratory
for L>=45 -- a different split from `is_mandatory`, which is True through L=60).
`add_support_role` reproduces it exactly as a documented post-processing step.
"""

from __future__ import annotations

import pandas as pd

from coastal_gap_reconstruction.daily_target import load_daily_target as _load_daily_target
from coastal_gap_reconstruction.daily_target import target_table_checksum
from coastal_gap_reconstruction.gaps import (
    find_context_qualified_positions,
    sample_nonoverlapping,
)

from . import _config

__all__ = ["load_daily_target", "target_table_checksum", "build_gap_pool", "add_support_role", "POOL_COLUMNS"]

POOL_COLUMNS = [
    "gap_id",
    "gap_length",
    "start_date",
    "end_date",
    "n_hidden_days",
    "target_variable",
    "eligibility_column",
    "season",
    "year",
    "context_before_days",
    "context_after_days",
    "is_mandatory",
    "sample_order_index",
    "support_role",
]


def load_daily_target(path) -> pd.DataFrame:
    """Load the daily oxygen target table, indexed by date.

    Thin wrapper over `coastal_gap_reconstruction.daily_target.load_daily_target`
    (the canonical implementation, shared with the chlorophyll builder).
    """
    return _load_daily_target(path, date_col=_config.DATE_COL)


def find_eligible_runs(target_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Oxygen eligible-run detection: a day counts only if `eligible_ge_18` is True
    AND `oxygen_mean_mgL` is finite -- verified separately rather than assumed
    co-extensive (oxygen's own co-extension was checked at private build time, not
    assumed here either)."""
    eligible_mask = (
        target_df[_config.ELIGIBLE_COL].fillna(False).astype(bool)
        & target_df[_config.TARGET_COL].notna()
    )
    eligible_dates = sorted(target_df.index[eligible_mask])
    if not eligible_dates:
        return []

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    run_start = eligible_dates[0]
    prev = eligible_dates[0]
    for d in eligible_dates[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            runs.append((run_start, prev, (prev - run_start).days + 1))
            run_start = d
            prev = d
    runs.append((run_start, prev, (prev - run_start).days + 1))
    return runs


def build_gap_pool(
    target_df: pd.DataFrame,
    gap_lengths: list[int] = _config.MANDATORY_GAP_LENGTHS,
    exploratory_lengths: list[int] = _config.EXPLORATORY_GAP_LENGTHS,
    seed: int = _config.RANDOM_SEED,
    max_per_length: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Build the oxygen artificial-gap pool (base 14 columns, no `support_role` yet
    -- call `add_support_role` for the full 14-column released schema).
    """
    if max_per_length is None:
        max_per_length = _config.TARGET_ACHIEVED_COUNTS

    runs = find_eligible_runs(target_df)

    rows: list[dict] = []
    all_lengths = list(gap_lengths) + list(exploratory_lengths)
    for gap_length in all_lengths:
        is_mandatory = gap_length in gap_lengths
        context_days = _config.context_days_required(gap_length)
        positions = find_context_qualified_positions(runs, gap_length, context_days)
        if not positions:
            continue

        cap = max_per_length.get(gap_length, len(positions))
        selected = sample_nonoverlapping(positions, gap_length, cap, seed)

        for order_idx, start in enumerate(selected):
            end = start + pd.Timedelta(days=gap_length - 1)

            run_start = run_end = None
            for rs, re_, _rl in runs:
                if rs <= start and end <= re_:
                    run_start, run_end = rs, re_
                    break
            context_before = (start - run_start).days if run_start is not None else float("nan")
            context_after = (run_end - end).days if run_end is not None else float("nan")

            rows.append({
                "gap_id": f"OX_L{gap_length:03d}_{start.strftime('%Y%m%d')}",
                "gap_length": gap_length,
                "start_date": start,
                "end_date": end,
                "n_hidden_days": gap_length,
                "target_variable": _config.TARGET_COL,
                "eligibility_column": _config.ELIGIBLE_COL,
                "season": _config.SEASON_MAP[start.month],
                "year": int(start.year),
                "context_before_days": context_before,
                "context_after_days": context_after,
                "is_mandatory": is_mandatory,
                "sample_order_index": order_idx,
            })

    return pd.DataFrame(rows)


def add_support_role(pool: pd.DataFrame) -> pd.DataFrame:
    """Add the public-only `support_role` column (see module docstring)."""
    pool = pool.copy()
    pool["support_role"] = pool["gap_length"].map(_config.support_role)
    return pool[POOL_COLUMNS]
