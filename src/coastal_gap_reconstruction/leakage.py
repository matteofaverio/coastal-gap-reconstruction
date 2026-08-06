"""Dependency-window exclusion: the leave-one-gap-out (LOCO) leakage-control rule
used throughout this project's artificial-gap validation.

Masking a target value (`gaps.apply_artificial_gap`) is necessary but not
sufficient to prevent leakage. Any feature built from a lagged or rolling window
of the target (e.g. "chlorophyll 7 days ago", "14-day rolling mean") computed at a
date *near* a hidden gap can still smuggle the hidden values back in indirectly,
even though the target column itself is NaN at the hidden dates -- the lag/roll
value at a nearby visible date was computed before masking, from the original
(unmasked) series. Training a model on such rows leaks the answer.

The dependency window of a gap is the span of dates whose own lag/rolling
features could have been computed using data inside the gap. A leakage-safe
training set excludes every row in that window, not merely the gap's own hidden
dates.
"""

from __future__ import annotations

import pandas as pd


def dependency_window(
    start_date: pd.Timestamp,
    gap_length: int,
    max_lookback: int,
    max_lookahead: int = 0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the (inclusive) date range that must be excluded from training when
    evaluating a gap starting at `start_date` with the given `gap_length`.

    `max_lookback` is the longest lag/rolling window (in days) used by any feature
    a candidate model might rely on (e.g. 21 for a "21-day rolling mean" feature);
    `max_lookahead` is the same for any feature that looks forward (0 for
    forecast-safe features, which by construction use no future data).

    The excluded range extends `max_lookback` days before the gap and
    `max_lookahead` days after it, in addition to the gap's own hidden days --
    every date in this range has at least one lag/rolling feature that reaches
    into the hidden interval.
    """
    if max_lookback < 0 or max_lookahead < 0:
        raise ValueError("max_lookback and max_lookahead must be non-negative")
    gap_end = start_date + pd.Timedelta(days=gap_length - 1)
    excl_start = start_date - pd.Timedelta(days=max_lookback)
    excl_end = gap_end + pd.Timedelta(days=max_lookahead)
    return excl_start, excl_end


def exclude_dependency_window(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    max_lookback: int,
    max_lookahead: int = 0,
) -> pd.DataFrame:
    """Return `df` (indexed by date) with every row inside the gap's dependency
    window removed.

    Use this to build the training set for a single leave-one-gap-out evaluation:
    every candidate gap gets its own dependency window excluded from that specific
    gap's training data, while remaining visible for every other gap's evaluation.
    """
    excl_start, excl_end = dependency_window(start_date, gap_length, max_lookback, max_lookahead)
    outside = (df.index < excl_start) | (df.index > excl_end)
    return df.loc[outside]
