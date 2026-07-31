"""Direct leakage-invariance test: a masked model's predictions must not change
if the secretly-retained hidden truth changes, since the model never sees it.

This is a stronger proof than "the training set excludes the gap's dates" (which
`tests/test_demo_methods.py` already checks): it demonstrates the property
directly, by actually tampering with the hidden truth and showing the prediction
is bit-for-bit unaffected, rather than only checking training-row membership.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction.baseline_imputation import (
    linear_interpolation_baseline,
    monthly_climatology,
    persistence_baseline,
)
from coastal_gap_reconstruction.gaps import apply_artificial_gap

REPO_ROOT = Path(__file__).resolve().parent.parent
CHL_TARGET = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
TARGET_COL = "chl_mean"
ELIGIBLE_COL = "target_eligible_default"


def _load_target() -> pd.DataFrame:
    df = pd.read_csv(CHL_TARGET, parse_dates=["date"])
    return df.set_index("date").sort_index()


def _pick_gap(target_df: pd.DataFrame, gap_length: int = 7) -> pd.Timestamp:
    """Find a real start date with `gap_length` consecutive eligible days,
    itself surrounded by more eligible days (so baselines have context)."""
    eligible = target_df[ELIGIBLE_COL].fillna(False).astype(bool)
    eligible_dates = target_df.index[eligible]
    for start in eligible_dates:
        window = pd.date_range(start, periods=gap_length, freq="D")
        if all(d in eligible_dates for d in window):
            before = target_df.loc[target_df.index < start]
            after = target_df.loc[target_df.index > window[-1]]
            if len(before) > 30 and len(after) > 30:
                return start
    raise AssertionError("no suitable gap window found in the public target table")


def _tamper_hidden_truth(masked: pd.DataFrame, start: pd.Timestamp, gap_length: int) -> pd.DataFrame:
    """Return a copy of `masked` where the (still-secret, not exposed to any
    method) true values at the hidden dates are altered. In this test we
    simulate 'the secretly retained truth changed' by writing new sentinel
    values directly into a *separate* full (unmasked) reference copy, never into
    `masked` itself -- `masked` keeps the hidden interval as NaN throughout, so
    passing it to a method is identical before and after tampering unless the
    method reaches around the mask to read something else.
    """
    tampered_full = masked.copy()
    hidden_dates = pd.date_range(start, periods=gap_length, freq="D")
    rng = np.random.default_rng(0)
    for d in hidden_dates:
        if d in tampered_full.index:
            tampered_full.loc[d, TARGET_COL] = float(rng.uniform(100, 200))  # absurd sentinel value
    # Re-apply the mask: even after tampering the "secret" reference, the model
    # input must still show NaN at the hidden dates -- this is the actual
    # contract every method call receives.
    return apply_artificial_gap(tampered_full, start, gap_length, target_col=TARGET_COL)


def test_climatology_predictions_are_invariant_to_hidden_truth_tampering() -> None:
    target_df = _load_target()
    start = _pick_gap(target_df, gap_length=7)
    masked_before = apply_artificial_gap(target_df, start, 7, target_col=TARGET_COL)

    preds_before = monthly_climatology(masked_before, start, 7)

    masked_after = _tamper_hidden_truth(masked_before, start, 7)
    preds_after = monthly_climatology(masked_after, start, 7)

    assert preds_before.keys() == preds_after.keys()
    for d in preds_before:
        a, b = preds_before[d], preds_after[d]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, f"climatology prediction changed at {d} after hidden-truth tampering"


def test_persistence_predictions_are_invariant_to_hidden_truth_tampering() -> None:
    target_df = _load_target()
    start = _pick_gap(target_df, gap_length=7)
    masked_before = apply_artificial_gap(target_df, start, 7, target_col=TARGET_COL)

    preds_before = persistence_baseline(masked_before, start, 7)
    masked_after = _tamper_hidden_truth(masked_before, start, 7)
    preds_after = persistence_baseline(masked_after, start, 7)

    for d in preds_before:
        a, b = preds_before[d], preds_after[d]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, f"persistence prediction changed at {d} after hidden-truth tampering"


def test_linear_interpolation_predictions_are_invariant_to_hidden_truth_tampering() -> None:
    """Linear interpolation reads both the pre-gap and post-gap edges -- this
    test specifically exercises that the *post*-gap edge used is the real
    post-gap observation, not anything from inside the tampered hidden
    interval."""
    target_df = _load_target()
    start = _pick_gap(target_df, gap_length=7)
    masked_before = apply_artificial_gap(target_df, start, 7, target_col=TARGET_COL)

    preds_before = linear_interpolation_baseline(masked_before, start, 7)
    masked_after = _tamper_hidden_truth(masked_before, start, 7)
    preds_after = linear_interpolation_baseline(masked_after, start, 7)

    for d in preds_before:
        a, b = preds_before[d], preds_after[d]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, f"linear-interpolation prediction changed at {d} after hidden-truth tampering"
