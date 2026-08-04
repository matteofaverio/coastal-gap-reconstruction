"""Chlorophyll target loading and artificial-gap pool construction.

Builds the 18-column artificial-gap validation pool schema used throughout this
project's chlorophyll benchmark, on top of the target-neutral mechanics in
`coastal_gap_reconstruction.gaps`.

Reproducibility: full and exact, but via two different procedures.

The released pool (`data_public/chlorophyll/chlorophyll_validation_gaps.csv`, 681
rows) supports nine gap lengths: 1, 3, 7, 10, 14, 21, 30, 45, 60 days. These split
into two subsets, built at different times in the private project's history by two
different, non-interchangeable sampling procedures:

- Core lengths (1, 3, 7, 14, 30 -- 450 rows): each length sampled independently
  with `gaps.sample_nonoverlapping` (Python `random.Random(42)`, freshly seeded per
  length), over the plain eligible-run candidate universe.
- Extended lengths (10, 21, 45, 60 -- 231 rows), added in a later pass: sampled
  with `gaps.sample_nonoverlapping_sequential` (one shared
  `numpy.random.default_rng(42)`, its state advanced across all four lengths in
  this exact order -- reproducing, say, the L=45 rows requires first replaying the
  L=10 and L=21 draws against the same generator, not reseeding), over a stricter
  candidate universe requiring the hidden target value to exceed 1e-4 (not merely
  be eligible/non-null), with non-uniform per-length caps
  (`_config.EXTENDED_MAX_CANDIDATES`).

`build_gap_pool` runs both procedures internally and concatenates the result.
Regenerating from the public daily target table reproduces:

- **Selection**: all 681 released gap_ids, exactly, for both length subsets.
- **Metadata** (season, year, context columns, event/sustained/background labels,
  regime, checksum): exact on all 681 rows.
- **Numeric aggregates** (target_mean_true, target_max_true, chl_90th_threshold):
  exact on 680 of 681 rows. One row, `L10_20160622`, has `target_mean_true` =
  1.3268 here vs. 1.3269 in the released file -- investigated to a proven root
  cause, not merely tolerated:

  The row's 10 hidden `chl_mean` values sum, by every standard method tried
  (`pandas.Series.mean`, `numpy.mean` on the same values as a Python list or a
  NumPy array, left-to-right Python summation), to a float64 value of
  `1.326849999999999862865...` -- i.e. the true sum sits *fractionally below*
  the `x.xxxx5` rounding boundary, so standard rounding gives 1.3268. Summing the
  same 10 values in **reverse** order gives `1.32685000000000...1` -- fractionally
  *above* the boundary, rounding to 1.3269 and matching the released value exactly.
  A `float32`-intermediate mean also happens to land on the 1.3269 side for this
  specific row, but was ruled out as the actual historical procedure: applied
  pool-wide, it reproduces this one row correctly but *breaks 5 other, previously
  bit-exact rows* (verified by direct pool-wide comparison, not assumed). No
  single alternative summation order or precision path reproduces this row without
  regressing others.

  **Conclusion**: this is a genuine floating-point non-determinism artifact --
  the released value depends on summation-order/precision details of the specific
  historical execution environment (library version, BLAS backend, or an
  intermediate step not visible in the recovered source) that cannot be exactly
  reconstructed from the available code, not a different candidate selection, not
  a different formula, and not an error in either the released value or this
  regeneration. The released CSV remains the authoritative benchmark value for
  this row; this module does not special-case it. See
  `tests/test_gap_pool_regeneration.py` for tests that separately report
  selection, metadata, and numeric equality, and for why canonical-serialized
  (whole-file hash) equality does not hold: independently of this one cell, the
  released file's row order is not sorted by `gap_id` (it reflects the original
  per-length generation order), which this module's `pd.concat`-based assembly
  does not reproduce either.

The extended-length procedure was recovered by reading the private project's
extended-gap-edge evaluation script's own gap-generation functions directly
(a script that generates its own extended-length candidate pool inline
rather than through the core `artificial_gaps.py` module) -- not guessed or
reverse-engineered from the released CSV alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coastal_gap_reconstruction.daily_target import load_daily_target as _load_daily_target
from coastal_gap_reconstruction.daily_target import target_table_checksum
from coastal_gap_reconstruction.gaps import (
    count_context_days,
    find_positions_with_value_floor,
    generate_candidate_gaps,
    sample_nonoverlapping_sequential,
)

from . import _config

__all__ = ["load_daily_target", "target_table_checksum", "build_gap_pool", "POOL_COLUMNS"]

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


def load_daily_target(path) -> pd.DataFrame:
    """Load the daily chlorophyll target table, indexed by date.

    Thin wrapper over `coastal_gap_reconstruction.daily_target.load_daily_target`
    (the canonical implementation, shared with the oxygen builder) fixing the date
    column to this target's convention.
    """
    return _load_daily_target(path, date_col=_config.DATE_COL)


def _label_candidates(
    target_df: pd.DataFrame,
    starts_by_length: dict[int, list[pd.Timestamp]],
    checksum: str,
    chl_90th_threshold: float,
    eligible_dates: set[pd.Timestamp],
) -> list[dict]:
    """Shared column-labeling step for both the core and extended candidate sets."""
    from coastal_gap_reconstruction.gaps import majority_season_and_year

    rows: list[dict] = []
    for gap_length, starts in starts_by_length.items():
        required_ctx = _config.required_context_days(gap_length)
        for start in starts:
            end = start + pd.Timedelta(days=gap_length - 1)
            hidden_dates = pd.date_range(start, end, freq="D")
            hidden_vals = target_df.loc[
                target_df.index.isin(hidden_dates), _config.TARGET_COL
            ]

            mean_val = float(hidden_vals.mean()) if hidden_vals.notna().any() else np.nan
            max_val = float(hidden_vals.max()) if hidden_vals.notna().any() else np.nan
            is_event = _config.TARGET_SPEC.event_label(hidden_vals, chl_90th_threshold)
            is_sustained = (
                bool(mean_val >= _config.SUSTAINED_MEAN_THRESHOLD)
                if not np.isnan(mean_val)
                else False
            )
            is_background = not (is_event or is_sustained)

            pre_ctx = count_context_days(
                eligible_dates, start - pd.Timedelta(days=1), -1, required_ctx
            )
            post_ctx = count_context_days(
                eligible_dates, end + pd.Timedelta(days=1), 1, required_ctx
            )
            context_constrained = pre_ctx < required_ctx or post_ctx < required_ctx

            season, year = majority_season_and_year(start, end)

            rows.append({
                "gap_id": f"L{gap_length:02d}_{start.strftime('%Y%m%d')}",
                "gap_length": gap_length,
                "start_date": start,
                "end_date": end,
                "n_hidden_days": gap_length,
                "season": season,
                "year": year,
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
    return rows


def build_gap_pool(
    target_df: pd.DataFrame,
    checksum: str,
    gap_lengths: list[int] = _config.GAP_LENGTHS,
    seed: int = _config.RANDOM_SEED,
) -> pd.DataFrame:
    """Build the 18-column chlorophyll artificial-gap pool for any subset of
    `_config.GAP_LENGTHS`.

    `checksum` must be `target_table_checksum(path_to_the_daily_target_csv)` --
    passed in explicitly (not recomputed here) so the same checksum is guaranteed
    to describe the exact file `target_df` was loaded from.

    Core lengths (`_config.CORE_GAP_LENGTHS`) and extended lengths
    (`_config.EXTENDED_GAP_LENGTHS`) are sampled by two different procedures (see
    the module docstring); requesting a mix of both in `gap_lengths` runs both
    procedures and concatenates the result.
    """
    core_lengths = [length for length in gap_lengths if length in _config.CORE_GAP_LENGTHS]
    extended_lengths = [length for length in gap_lengths if length in _config.EXTENDED_GAP_LENGTHS]
    unknown = set(gap_lengths) - set(core_lengths) - set(extended_lengths)
    if unknown:
        raise ValueError(
            f"gap_lengths {sorted(unknown)} are neither core nor extended lengths; "
            f"the sampling procedure for a new length must be decided explicitly, "
            f"not assumed."
        )

    eligible_mask = target_df[_config.ELIGIBLE_COL].fillna(False).astype(bool)
    eligible_target = target_df.loc[eligible_mask, _config.TARGET_COL].dropna()
    chl_90th_threshold = float(eligible_target.quantile(0.90))
    eligible_dates = set(target_df.index[eligible_mask])

    starts_by_length: dict[int, list[pd.Timestamp]] = {}

    if core_lengths:
        candidates = generate_candidate_gaps(
            target_df,
            gap_lengths=core_lengths,
            seed=seed,
            max_per_length=_config.MAX_GAPS_PER_LENGTH,
            eligible_col=_config.ELIGIBLE_COL,
        )
        for gap_length in core_lengths:
            sub = candidates[candidates["gap_length"] == gap_length]
            starts_by_length[gap_length] = list(sub["start_date"])

    if extended_lengths:
        rng = np.random.default_rng(seed)
        # Order matters: the shared rng's state must advance through every
        # extended length in this exact order, even if only a subset was
        # requested by the caller, to match the released pool's own build order.
        for gap_length in _config.EXTENDED_GAP_LENGTHS:
            positions = find_positions_with_value_floor(
                target_df,
                gap_length,
                eligible_col=_config.ELIGIBLE_COL,
                value_col=_config.TARGET_COL,
                value_floor=_config.EXTENDED_VALUE_FLOOR,
            )
            max_n = _config.EXTENDED_MAX_CANDIDATES[gap_length]
            chosen = sample_nonoverlapping_sequential(positions, gap_length, max_n, rng)
            if gap_length in extended_lengths:
                starts_by_length[gap_length] = chosen

    rows = _label_candidates(target_df, starts_by_length, checksum, chl_90th_threshold, eligible_dates)
    if not rows:
        return pd.DataFrame(columns=POOL_COLUMNS)
    return pd.DataFrame(rows, columns=POOL_COLUMNS)
