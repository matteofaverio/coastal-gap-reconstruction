"""Oxygen distribution-tail and sustained-event diagnostics.

Ported from the original authoritative tail-diagnostic modules
(`oxygen_tail_distribution_diagnostic.py` for the quantile-band definitions,
`oxygen_tail_model_performance.py` for the per-day run-persistence
classification that produced the released
`results/oxygen/oxygen_tail_persistence_metrics.csv`/
`oxygen_tail_quantile_band_metrics.csv`). Reproduces both definitions
exactly, not a chlorophyll-derived guess -- oxygen has no event/high-value
label analogous to chlorophyll's `is_high_chl_event`; every threshold here is
an **empirical quantile of the eligible-day population**, not an ecological
threshold -- no locally-extractable ecological threshold for oxygen exists
to use instead.

Quantile bands (6): below_p10, p10_to_p25, p25_to_p50, p50_to_p75, p75_to_p90,
above_p90 -- edges are the p10/p25/p50/p75/p90 percentiles of
`oxygen_mean_mgL` over eligible days only (`eligible_ge_18 == True`).

Run-persistence categories (3), computed separately for the below_p10 and
above_p90 bands only: `isolated_tail_day` (a single calendar day, no
adjacent tail day), `short_tail_run` (2-6 consecutive tail days),
`sustained_tail_run` (>=7 consecutive tail days) -- a run is broken by any
calendar-date gap, matching the original `_runs_from_boolean`
exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import benchmark_contract as bc

__all__ = [
    "QUANTILE_LEVELS_FOR_EDGES", "BAND_ORDER",
    "compute_quantile_edges", "classify_quantile_bin",
    "compute_tail_runs", "classify_tail_persistence",
]

QUANTILE_LEVELS_FOR_EDGES = [0.10, 0.25, 0.50, 0.75, 0.90]
BAND_ORDER = ["below_p10", "p10_to_p25", "p25_to_p50", "p50_to_p75", "p75_to_p90", "above_p90"]
RUN_CATEGORY_ORDER = ["isolated_tail_day", "short_tail_run", "sustained_tail_run"]


def _eligible_series(target_df: pd.DataFrame) -> pd.Series:
    eligible = target_df[bc.ELIGIBLE_COLUMN].fillna(False).astype(bool)
    return target_df.loc[eligible, bc.TARGET_COLUMN].dropna()


def compute_quantile_edges(target_df: pd.DataFrame) -> dict[str, float]:
    """p10/p25/p50/p75/p90 of `oxygen_mean_mgL` over eligible days only."""
    series = _eligible_series(target_df)
    q = series.quantile(QUANTILE_LEVELS_FOR_EDGES)
    return {f"p{int(level * 100)}": float(q[level]) for level in QUANTILE_LEVELS_FOR_EDGES}


def classify_quantile_bin(values: np.ndarray, edges: dict[str, float]) -> np.ndarray:
    """Assign each value in `values` to one of `BAND_ORDER`."""
    bins = [-np.inf, edges["p10"], edges["p25"], edges["p50"], edges["p75"], edges["p90"], np.inf]
    idx = np.digitize(values, bins[1:-1], right=False)
    return np.array(BAND_ORDER)[idx]


def _runs_from_boolean(dates: pd.DatetimeIndex, flags: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Consecutive-calendar-date runs of True in `flags`, broken by any
    date gap (not just any False value) -- exact port of the private
    `_runs_from_boolean`."""
    runs = []
    n = len(flags)
    i = 0
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1] and (dates[j + 1] - dates[j]).days == 1:
            j += 1
        runs.append((dates[i], dates[j], j - i + 1))
        i = j + 1
    return runs


def _run_category(length: int) -> str:
    if length == 1:
        return "isolated_tail_day"
    if length <= 6:
        return "short_tail_run"
    return "sustained_tail_run"


def compute_tail_runs(target_df: pd.DataFrame, edges: dict[str, float]) -> pd.DataFrame:
    """Per-eligible-day run-length classification for the below_p10 and
    above_p90 bands only. Returns one row per (date, quantile_bin) with
    `run_category`/`run_length`."""
    series = _eligible_series(target_df)
    dates = series.index
    values = series.to_numpy(dtype=float)
    low_v, high_v = edges["p10"], edges["p90"]

    rows = []
    for tail_label, flags in [("below_p10", values <= low_v), ("above_p90", values >= high_v)]:
        for start, end, length in _runs_from_boolean(dates, flags):
            category = _run_category(length)
            for d in pd.date_range(start, end, freq="D"):
                rows.append({"date": d, "quantile_bin": tail_label, "run_category": category, "run_length": length})
    return pd.DataFrame(rows)


def classify_tail_persistence(target_df: pd.DataFrame, edges: dict[str, float] | None = None) -> pd.DataFrame:
    """Convenience wrapper: computes `edges` from `target_df` if not given,
    then returns `compute_tail_runs`'s per-day run classification."""
    if edges is None:
        edges = compute_quantile_edges(target_df)
    return compute_tail_runs(target_df, edges)
