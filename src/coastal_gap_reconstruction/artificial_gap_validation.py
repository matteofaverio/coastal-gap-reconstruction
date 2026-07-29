"""Artificial-gap validation framework.

The core idea: to evaluate a reconstruction method without access to
withheld ground truth on real gaps, we carve "artificial" gaps out of
stretches of the record that ARE observed, mask the target values for those
days, run the candidate method as if those days were missing, and score the
predictions against the (secretly retained) true values.

Design rules enforced here:

- Artificial gaps mask the in-situ target column only. External predictors
  (satellite, reanalysis, meteorological covariates) remain available
  through the gap -- only the variable being reconstructed is hidden.
- A gap candidate is only valid if every hidden day was originally eligible
  (i.e. had enough valid hourly data to trust the true daily value used for
  scoring).
- Gaps of a given length are sampled to be non-overlapping, with a fixed
  random seed for reproducibility.
- Hidden target values must never leak into the feature engineering for a
  candidate gap (e.g. lag/rolling-window features must be computed only
  from masked data).

`generate_gap_candidates` and `apply_artificial_gap` accept `target_col` /
`eligible_col` overrides so this framework runs unchanged against a target
table for any sensor/variable -- see `data_loading.TARGET_COL` /
`data_loading.ELIGIBLE_COL` for the chlorophyll defaults.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pandas as pd

from .data_loading import TARGET_COL, ELIGIBLE_COL
from .gap_detection import find_eligible_runs

GAP_LENGTHS = [1, 3, 7, 14, 30, 45, 60]
RANDOM_SEED = 42
MAX_GAPS_PER_LENGTH = 100

SEASON_MAP = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}


def _find_all_candidate_positions(
    runs: list[tuple[pd.Timestamp, pd.Timestamp, int]],
    gap_length: int,
) -> list[pd.Timestamp]:
    """Return all valid starting dates for gaps of a given length."""
    positions: list[pd.Timestamp] = []
    for run_start, _run_end, run_len in runs:
        if run_len >= gap_length:
            for offset in range(run_len - gap_length + 1):
                positions.append(run_start + pd.Timedelta(days=offset))
    return positions


def _sample_nonoverlapping(
    positions: list[pd.Timestamp],
    gap_length: int,
    max_n: int,
    seed: int,
) -> list[pd.Timestamp]:
    """Sample up to max_n non-overlapping starting positions with a fixed seed."""
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


def generate_gap_candidates(
    target_df: pd.DataFrame,
    gap_lengths: list[int] = GAP_LENGTHS,
    seed: int = RANDOM_SEED,
    max_per_length: int = MAX_GAPS_PER_LENGTH,
    target_col: str = TARGET_COL,
    eligible_col: str = ELIGIBLE_COL,
) -> pd.DataFrame:
    """Generate a pool of artificial gap candidates.

    Every hidden day in a candidate gap is required to be eligible. Event
    status (is_high_chl_event) flags gaps whose true values exceed the 90th
    percentile of all eligible target values, useful for stratified scoring.

    `target_col`/`eligible_col` let this run against any sensor's daily
    target table, not just the default chlorophyll columns.
    """
    eligible_mask = target_df[eligible_col].fillna(False).astype(bool)
    eligible_target = target_df.loc[eligible_mask, target_col].dropna()
    high_threshold = eligible_target.quantile(0.90)

    runs = find_eligible_runs(target_df, eligible_col=eligible_col)

    rows: list[dict] = []
    for gap_length in gap_lengths:
        positions = _find_all_candidate_positions(runs, gap_length)
        if not positions:
            warnings.warn(f"No valid candidates for gap_length={gap_length}", stacklevel=2)
            continue

        selected = _sample_nonoverlapping(positions, gap_length, max_per_length, seed)

        for start in selected:
            end = start + pd.Timedelta(days=gap_length - 1)
            hidden_dates = pd.date_range(start, end, freq="D")
            hidden_vals = target_df.loc[target_df.index.isin(hidden_dates), target_col]

            gap_id = f"L{gap_length:02d}_{start.strftime('%Y%m%d')}"
            season = SEASON_MAP[start.month]
            mean_val = float(hidden_vals.mean()) if hidden_vals.notna().any() else np.nan
            max_val = float(hidden_vals.max()) if hidden_vals.notna().any() else np.nan
            is_event = bool(hidden_vals.max() > high_threshold) if hidden_vals.notna().any() else False

            rows.append({
                "gap_id": gap_id,
                "gap_length": gap_length,
                "start_date": start,
                "end_date": end,
                "season": season,
                "year": start.year,
                "n_hidden_days": gap_length,
                "target_mean_true": round(mean_val, 4),
                "target_max_true": round(max_val, 4),
                "is_high_chl_event": is_event,
                "high_value_90th_threshold": round(float(high_threshold), 4),
            })

    return pd.DataFrame(rows)


def apply_artificial_gap(
    target_df: pd.DataFrame,
    start_date: pd.Timestamp,
    gap_length: int,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """Return a copy of target_df with the target column masked over the gap.

    Only the target column is masked; predictor/covariate columns, if present
    in the same table, are left untouched.
    """
    masked = target_df.copy()
    hidden_dates = pd.date_range(start_date, periods=gap_length, freq="D")
    for d in hidden_dates:
        if d in masked.index:
            masked.loc[d, target_col] = np.nan
    return masked
