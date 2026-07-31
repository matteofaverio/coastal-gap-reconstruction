"""Chlorophyll target loading and artificial-gap pool construction.

Builds the 18-column artificial-gap validation pool schema used throughout this
project's chlorophyll benchmark, on top of the target-neutral mechanics in
`coastal_gap_reconstruction.gaps`.

Reproducibility boundary (state this precisely, do not overclaim):

The released pool (`data_public/chlorophyll/chlorophyll_validation_gaps.csv`, 681
rows) supports nine gap lengths: 1, 3, 7, 10, 14, 21, 30, 45, 60 days. Of these,
five lengths -- 1, 3, 7, 14, 30 (450 gaps) -- are exactly regenerable from the
public daily target table with the algorithm in this module: calling
`build_gap_pool` with `gap_lengths=_config.EXACTLY_REGENERABLE_GAP_LENGTHS`
reproduces those 450 rows byte-for-byte, verified against every one of the 18
released columns (including the SHA-256 target-table checksum).

The remaining four lengths -- 10, 21, 45, 60 (231 gaps) -- were assembled by a
separate, later extension pass in the private project's history (using a
different candidate pool / selection process not recovered by this port) and are
NOT reproducible by calling this module with the full nine-length list: the
column *formulas* below (event/sustained/background labels, context rule,
checksum) are verified exact given the right gap windows, but the *window
selection* itself for these four lengths does not match the released rows.
Treat the released CSV as the authoritative pool definition for the full
nine-length set; use this module to exactly regenerate the five-length core, or
to build new candidate pools of your own on other data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction.gaps import count_context_days, generate_candidate_gaps

from . import _config

POOL_COLUMNS = [
    "gap_id",
    "gap_length",
    "start_date",
    "end_date",
    "n_hidden_days",
    "season",
    "year",
    "target_mean_true",
    "target_max_true",
    "chl_90th_threshold",
    "is_high_chl_event",
    "is_sustained_event",
    "is_background",
    "pre_context_available_days",
    "post_context_available_days",
    "context_constrained",
    "regime",
    "target_table_checksum",
]


def load_daily_target(path: str | Path) -> pd.DataFrame:
    """Load the daily chlorophyll target table, indexed by date."""
    df = pd.read_csv(path, parse_dates=[_config.DATE_COL])
    return df.set_index(_config.DATE_COL).sort_index()


def target_table_checksum(path: str | Path) -> str:
    """SHA-256 of the daily target CSV's raw bytes, as recorded in the released pool."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_gap_pool(
    target_df: pd.DataFrame,
    checksum: str,
    gap_lengths: list[int] = _config.GAP_LENGTHS,
    seed: int = _config.RANDOM_SEED,
    max_per_length: int = _config.MAX_GAPS_PER_LENGTH,
) -> pd.DataFrame:
    """Build the 18-column chlorophyll artificial-gap pool.

    `checksum` must be `target_table_checksum(path_to_the_daily_target_csv)` --
    passed in explicitly (not recomputed here) so the same checksum is guaranteed
    to describe the exact file `target_df` was loaded from.
    """
    candidates = generate_candidate_gaps(
        target_df,
        gap_lengths=gap_lengths,
        seed=seed,
        max_per_length=max_per_length,
        eligible_col=_config.ELIGIBLE_COL,
    )
    if candidates.empty:
        return pd.DataFrame(columns=POOL_COLUMNS)

    eligible_mask = target_df[_config.ELIGIBLE_COL].fillna(False).astype(bool)
    eligible_target = target_df.loc[eligible_mask, _config.TARGET_COL].dropna()
    chl_90th_threshold = float(eligible_target.quantile(0.90))
    eligible_dates = set(target_df.index[eligible_mask])

    rows: list[dict] = []
    for row in candidates.itertuples(index=False):
        start, end, gap_length = row.start_date, row.end_date, row.gap_length
        hidden_dates = pd.date_range(start, end, freq="D")
        hidden_vals = target_df.loc[
            target_df.index.isin(hidden_dates), _config.TARGET_COL
        ]

        mean_val = float(hidden_vals.mean()) if hidden_vals.notna().any() else np.nan
        max_val = float(hidden_vals.max()) if hidden_vals.notna().any() else np.nan
        is_event = bool(max_val > chl_90th_threshold) if not np.isnan(max_val) else False
        is_sustained = (
            bool(mean_val >= _config.SUSTAINED_MEAN_THRESHOLD)
            if not np.isnan(mean_val)
            else False
        )
        is_background = not (is_event or is_sustained)

        required_ctx = _config.required_context_days(gap_length)
        pre_ctx = count_context_days(
            eligible_dates, start - pd.Timedelta(days=1), -1, required_ctx
        )
        post_ctx = count_context_days(
            eligible_dates, end + pd.Timedelta(days=1), 1, required_ctx
        )
        context_constrained = pre_ctx < required_ctx or post_ctx < required_ctx

        rows.append({
            "gap_id": row.gap_id,
            "gap_length": gap_length,
            "start_date": start,
            "end_date": end,
            "n_hidden_days": row.n_hidden_days,
            "season": row.season,
            "year": row.year,
            "target_mean_true": round(mean_val, 4) if not np.isnan(mean_val) else np.nan,
            "target_max_true": round(max_val, 4) if not np.isnan(max_val) else np.nan,
            "chl_90th_threshold": round(chl_90th_threshold, 4),
            "is_high_chl_event": is_event,
            "is_sustained_event": is_sustained,
            "is_background": is_background,
            "pre_context_available_days": pre_ctx,
            "post_context_available_days": post_ctx,
            "context_constrained": context_constrained,
            "regime": _config.REGIME,
            "target_table_checksum": checksum,
        })

    return pd.DataFrame(rows, columns=POOL_COLUMNS)
