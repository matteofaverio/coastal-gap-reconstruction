"""Target-neutral artificial-gap mechanics shared across every reconstruction target.

This module contains only mechanics that do not depend on which sensor/variable is
being reconstructed: eligible-run detection, candidate starting-position search,
seeded non-overlapping sampling, target masking, and pre/post context-day counting.
Two independent, non-interchangeable sampling strategies are provided
(`sample_nonoverlapping`, seeded per call with `random.Random`; and
`sample_nonoverlapping_sequential`, advancing a shared `numpy.random.Generator`
across calls) because the original gap pools were built with both,
at different times, for different length subsets -- see
`experiments/chlorophyll/target_and_gap_pool.py` for which one reproduces which
released rows.

It deliberately does NOT contain:

- any default list of gap lengths (`gap_lengths` is always a required argument here
  and in every function that calls into this module -- there is no such thing as a
  "the" gap-length list, only a target-specific one, chosen by the code in
  ``experiments/<target>/``);
- event/threshold logic (``is_high_chl_event``, sustained/background labels, or any
  oxygen-specific tail label) -- those are scientific decisions made by
  target-specific code that calls this module, not generic mechanics;
- support-role, "primary"/"exploratory" labeling, or any other benchmark-reporting
  concept -- also target-specific.

Masking (``apply_artificial_gap``) sets the hidden days' target value to ``NaN`` and
returns a new DataFrame; it never returns or logs the values it just hid, so a caller
cannot accidentally leak the hidden ground truth through this function's own output.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def find_eligible_runs(
    target_df: pd.DataFrame, eligible_col: str
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Find maximal consecutive runs of eligible calendar days.

    Returns a list of ``(start, end, length)`` tuples. "Consecutive" means
    sequential calendar days with no non-eligible day in between.
    """
    eligible_mask = target_df[eligible_col].fillna(False).astype(bool)
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


def find_candidate_positions(
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    gap_length: int,
) -> list[pd.Timestamp]:
    """Return every valid starting date for a gap of the given length.

    A starting date is valid if the ``gap_length``-day window beginning there lies
    entirely within one eligible run (from ``find_eligible_runs``).
    """
    positions: list[pd.Timestamp] = []
    for run_start, _run_end, run_len in runs:
        if run_len >= gap_length:
            for offset in range(run_len - gap_length + 1):
                positions.append(run_start + pd.Timedelta(days=offset))
    return positions


def find_context_qualified_positions(
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    gap_length: int,
    context_days: int,
) -> list[pd.Timestamp]:
    """Like `find_candidate_positions`, but additionally requires `context_days` of
    eligible-and-observed run on both sides of the gap, within the same contiguous
    run (this excludes gaps sitting at the very edge of an eligible run by
    construction). Some targets require this stricter search as a candidacy
    filter, not merely a post-hoc label; others only label context availability
    after the fact with `find_candidate_positions` + `count_context_days`. Both are
    genuinely target-neutral mechanics -- which one a target's builder calls is a
    target-specific decision.
    """
    positions: list[pd.Timestamp] = []
    for run_start, run_end, run_len in runs:
        if run_len < gap_length:
            continue
        for offset in range(run_len - gap_length + 1):
            start = run_start + pd.Timedelta(days=offset)
            end = start + pd.Timedelta(days=gap_length - 1)
            pre_ok = (start - run_start).days >= context_days
            post_ok = (run_end - end).days >= context_days
            if pre_ok and post_ok:
                positions.append(start)
    return positions


def sample_nonoverlapping(
    positions: list[pd.Timestamp],
    gap_length: int,
    max_n: int,
    seed: int,
) -> list[pd.Timestamp]:
    """Sample up to ``max_n`` non-overlapping starting positions with a fixed seed.

    Shuffles ``positions`` with a ``random.Random(seed)`` instance private to this
    call, then greedily accepts positions whose ``gap_length``-day window does not
    overlap any already-accepted window. Two gaps of length L starting at s1 and s2
    overlap iff ``|s1 - s2| < L``.
    """
    rng = random.Random(seed)
    shuffled = positions.copy()
    rng.shuffle(shuffled)

    selected: list[pd.Timestamp] = []
    for pos in shuffled:
        gap_end = pos + pd.Timedelta(days=gap_length - 1)
        overlap = any(
            not (gap_end < s or pos > s + pd.Timedelta(days=gap_length - 1))
            for s in selected
        )
        if not overlap:
            selected.append(pos)
        if len(selected) >= max_n:
            break

    return sorted(selected)


def find_positions_with_value_floor(
    target_df: pd.DataFrame,
    gap_length: int,
    eligible_col: str,
    value_col: str,
    value_floor: float,
) -> list[pd.Timestamp]:
    """Return starting dates where every hidden day is eligible AND has a value
    strictly greater than `value_floor` (not merely non-null).

    This is a different, stricter admissibility rule than `find_candidate_positions`
    (which only requires membership in an eligible run): some target pools additionally
    exclude candidate windows touching a floored/degenerate value. Genuinely
    target-neutral (parametrized by column name and floor, not any specific target's
    column), but a distinct rule from the plain eligible-run search -- which rule a
    builder uses is a target-specific decision.
    """
    ok = (
        target_df[eligible_col].fillna(False).astype(bool)
        & target_df[value_col].notna()
        & (target_df[value_col] > value_floor)
    )
    valid_dates = target_df.index[ok].sort_values()
    valid_set = set(valid_dates)
    positions = [
        d for d in valid_dates
        if all(s in valid_set for s in pd.date_range(d, periods=gap_length, freq="D"))
    ]
    return positions


def sample_nonoverlapping_sequential(
    positions: list[pd.Timestamp],
    gap_length: int,
    max_n: int,
    rng: np.random.Generator,
) -> list[pd.Timestamp]:
    """Sample up to `max_n` non-overlapping positions using a shared NumPy
    `Generator` instance (`numpy.random.default_rng(seed)`), advancing its state
    rather than reseeding.

    This is a genuinely different (and non-interchangeable) sampling strategy from
    `sample_nonoverlapping`: it uses `numpy.random.Generator.permutation` instead of
    `random.Random.shuffle`, and -- because the caller passes in one `rng` shared
    across multiple calls (e.g. one per gap length, processed in a fixed order) --
    reproducing a specific call's output requires replaying every earlier call
    against the *same* `rng` object in the *same* order first. A fresh
    `default_rng(seed)` only reproduces the *first* call in a sequence.
    """
    starts_arr = np.array(positions)
    if len(starts_arr) == 0:
        return []
    shuffled = starts_arr[rng.permutation(len(starts_arr))]

    chosen: list[pd.Timestamp] = []
    blocked: set[pd.Timestamp] = set()
    for d in shuffled:
        dt = pd.Timestamp(d)
        if dt in blocked:
            continue
        span = pd.date_range(dt, periods=gap_length, freq="D")
        if any(s in blocked for s in span):
            continue
        chosen.append(dt)
        blocked.update(span)
        if len(chosen) >= max_n:
            break
    return chosen


def apply_artificial_gap(
    target_df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    target_col: str,
) -> pd.DataFrame:
    """Return a copy of ``target_df`` with ``target_col`` masked over the gap window.

    Only ``target_col`` is masked; every other column (predictors, other targets,
    metadata) is left untouched. This function returns the masked frame only -- it
    does not return, log, or otherwise expose the hidden values, so a caller that
    only uses this return value cannot leak the withheld ground truth.
    """
    masked = target_df.copy()
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")
    for d in hidden_dates:
        if d in masked.index:
            masked.loc[d, target_col] = np.nan
    return masked


def count_context_days(
    eligible_dates: set[pd.Timestamp],
    edge_date: pd.Timestamp,
    direction: int,
    window: int,
) -> int:
    """Count eligible days within a fixed-size window adjacent to a gap.

    ``direction`` is ``-1`` to look backward from a gap's start (pre-context,
    counting the ``window`` calendar days ending the day before the gap starts) or
    ``+1`` to look forward from a gap's end (post-context, counting the ``window``
    calendar days starting the day after the gap ends). This counts eligible days
    *within the window*, not a run of unbroken consecutive eligible days -- a
    single non-eligible day inside the window does not reset or stop the count.
    """
    count = 0
    d = edge_date
    for _ in range(window):
        if d in eligible_dates:
            count += 1
        d = d + pd.Timedelta(days=direction)
    return count


_SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def majority_season_and_year(start: pd.Timestamp, end: pd.Timestamp) -> tuple[str, int]:
    """Return the (season, year) with the most hidden calendar days in [start, end].

    A gap's season/year label is not simply derived from its start date -- a gap
    that straddles a season or year boundary (e.g. starting in late December) is
    labeled by whichever season/year covers the majority of its hidden days. Ties
    are broken toward the earliest date (stable count, first-seen wins), matching
    the released pool's own labeling exactly.
    """
    dates = pd.date_range(start, end, freq="D")
    seasons = [_SEASON_MAP[d.month] for d in dates]
    years = [d.year for d in dates]

    def _mode_first_seen(values: list) -> object:
        counts: dict = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        best = values[0]
        best_count = 0
        for v in values:
            if counts[v] > best_count:
                best = v
                best_count = counts[v]
        return best

    return _mode_first_seen(seasons), _mode_first_seen(years)


def generate_candidate_gaps(
    target_df: pd.DataFrame,
    gap_lengths: list[int],
    seed: int,
    max_per_length: int,
    eligible_col: str,
) -> pd.DataFrame:
    """Generate a target-neutral pool of non-overlapping candidate gap windows.

    ``gap_lengths`` is required -- there is no default here, by design (see the
    module docstring). Returns one row per candidate with only target-neutral
    scaffold columns: ``gap_id``, ``gap_length``, ``start_date``, ``end_date``,
    ``n_hidden_days``, ``season``, ``year`` (season/year are majority-vote across
    the gap's hidden days, see ``majority_season_and_year``). Target-specific
    columns (event flags, thresholds, context-availability labels, checksums,
    support roles, ...) are added by target-specific code that consumes this
    DataFrame, never by this function.
    """
    runs = find_eligible_runs(target_df, eligible_col=eligible_col)

    rows: list[dict] = []
    for gap_length in gap_lengths:
        positions = find_candidate_positions(runs, gap_length)
        selected = sample_nonoverlapping(positions, gap_length, max_per_length, seed)

        for start in selected:
            end = start + pd.Timedelta(days=gap_length - 1)
            season, year = majority_season_and_year(start, end)
            rows.append({
                "gap_id": f"L{gap_length:02d}_{start.strftime('%Y%m%d')}",
                "gap_length": gap_length,
                "start_date": start,
                "end_date": end,
                "n_hidden_days": gap_length,
                "season": season,
                "year": year,
            })

    return pd.DataFrame(rows)
