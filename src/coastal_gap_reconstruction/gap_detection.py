"""Gap detection utilities for a daily eligibility-flagged time series.

Two related but distinct ideas are used throughout this benchmark:

- "eligible runs": maximal consecutive stretches of calendar days that pass
  the eligibility criterion (enough valid hourly observations). These are
  the only days from which artificial gaps may be carved, and the only days
  used to fit baselines.
- "real gaps": stretches of consecutive non-eligible (missing/insufficient)
  days in the observed record. These are naturally occurring missingness,
  not constructed for validation.
"""

from __future__ import annotations

import pandas as pd

from .data_loading import ELIGIBLE_COL


def find_eligible_runs(target_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find maximal consecutive runs of eligible calendar days.

    Returns a list of (start, end, length) tuples. "Consecutive" means
    sequential calendar days with no non-eligible day in between.
    """
    eligible_mask = target_df[ELIGIBLE_COL].fillna(False).astype(bool)
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


def find_real_gaps(target_df: pd.DataFrame) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find maximal consecutive runs of non-eligible (missing) calendar days.

    Returns a list of (start, end, length) tuples for naturally occurring
    gaps in the observed record (not artificial/constructed gaps).
    """
    eligible_mask = target_df[ELIGIBLE_COL].fillna(False).astype(bool)
    missing_dates = sorted(target_df.index[~eligible_mask])

    if not missing_dates:
        return []

    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    run_start = missing_dates[0]
    prev = missing_dates[0]

    for d in missing_dates[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            runs.append((run_start, prev, (prev - run_start).days + 1))
            run_start = d
            prev = d
    runs.append((run_start, prev, (prev - run_start).days + 1))

    return runs


def coverage_summary(target_df: pd.DataFrame) -> dict:
    """Compute basic coverage/missingness summary statistics.

    Returns a dict with total days, eligible days, eligible fraction,
    number of real gaps, and the longest real gap length.
    """
    eligible_mask = target_df[ELIGIBLE_COL].fillna(False).astype(bool)
    real_gaps = find_real_gaps(target_df)
    return {
        "n_days_total": len(target_df),
        "n_days_eligible": int(eligible_mask.sum()),
        "eligible_fraction": float(eligible_mask.mean()),
        "n_real_gaps": len(real_gaps),
        "longest_real_gap_days": max((g[2] for g in real_gaps), default=0),
    }
