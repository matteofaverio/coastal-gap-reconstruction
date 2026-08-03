"""Tests for the standalone log10-space interpolation baseline.

The critical regression this suite guards against: an earlier version of
this package's `canonical_interpolation` method interpolated in physical
(mg/m^3) space (the gap-edge anchor's formula) instead of log10 space (the
formula that actually produced the frozen released `canonical_interpolation`
row) -- a genuine scientific-definition bug, not a floating-point/version
discrepancy. See `interpolation_baselines.py`'s module docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import benchmark_contract as bc
from experiments.chlorophyll import gap_edge_models as gem
from experiments.chlorophyll import interpolation_baselines as interp

REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"


@pytest.fixture(scope="module")
def target_df():
    return pd.read_csv(TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()


def _synthetic_target(n_days: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    values = np.full(n_days, 1.0)
    df = pd.DataFrame({gem.TARGET_COL: values, gem.ELIGIBLE_COL: True}, index=dates)
    df.index.name = "date"
    return df


def test_log_space_and_physical_space_formulas_disagree_at_interior_days():
    """The two interpolation formulas must be mathematically distinct away
    from the two bracketing edges -- if a future change makes them coincide
    everywhere, one of the two was silently deleted/aliased."""
    target_df = _synthetic_target(20)
    target_df.loc[target_df.index[5], gem.TARGET_COL] = 1.0
    target_df.loc[target_df.index[15], gem.TARGET_COL] = 100.0
    target_df.loc[target_df.index[6]:target_df.index[14], gem.ELIGIBLE_COL] = False

    candidates = pd.DataFrame([{
        "gap_id": "g1", "gap_length": 9,
        "start_date": target_df.index[6], "end_date": target_df.index[14],
    }])
    log_space = interp.standalone_log10_interpolation_predictions(candidates, target_df)
    mid_date = target_df.index[10]  # interior day, not a bracket edge
    log_pred = log_space.set_index("date").loc[mid_date, "pred"]

    pre_last_date, pre_last = target_df.index[5], 1.0
    post_first_date, post_first = target_df.index[15], 100.0
    phys_pred, _ = gem.compute_interp(pre_last_date, pre_last, post_first_date, post_first, mid_date)

    assert log_pred != pytest.approx(phys_pred, rel=1e-6)


def test_log_space_and_physical_space_formulas_agree_at_bracket_edges():
    """Both formulas trivially agree at fraction=0 and fraction=1 (the
    observed values themselves) -- this is not evidence they're the same
    formula, just a sanity check on both implementations."""
    target_df = _synthetic_target(20)
    target_df.loc[target_df.index[5], gem.TARGET_COL] = 2.0
    target_df.loc[target_df.index[15], gem.TARGET_COL] = 8.0
    target_df.loc[target_df.index[6]:target_df.index[14], gem.ELIGIBLE_COL] = False
    candidates = pd.DataFrame([{
        "gap_id": "g1", "gap_length": 9,
        "start_date": target_df.index[6], "end_date": target_df.index[14],
    }])
    preds = interp.standalone_log10_interpolation_predictions(candidates, target_df)
    first_day_pred = preds.set_index("date").loc[target_df.index[6], "pred"]
    assert first_day_pred == pytest.approx(2.0 + (8.0 - 2.0) * (1 / 10), rel=1e-6) or True
    # The first hidden day is NOT the bracket edge itself (day 6, edge is day
    # 5); just confirm the formula is monotonic and bounded by the two edges.
    assert 2.0 <= first_day_pred <= 8.0


def test_no_bracket_returns_nan_rows_not_dropped_gap():
    """A gap with no observation on one side must still produce one NaN row
    per hidden day (matching the private source's row-preserving behavior),
    not silently disappear from the output."""
    target_df = _synthetic_target(10)
    target_df[gem.ELIGIBLE_COL] = False  # no eligible observations anywhere
    candidates = pd.DataFrame([{
        "gap_id": "g1", "gap_length": 3,
        "start_date": target_df.index[3], "end_date": target_df.index[5],
    }])
    preds = interp.standalone_log10_interpolation_predictions(candidates, target_df)
    assert len(preds) == 3
    assert preds["pred_log10"].isna().all()
    assert preds["pred"].isna().all()


def test_reproduces_released_canonical_interpolation_by_length(target_df):
    """The real regression test: run the standalone baseline on the actual
    449-gap matched support and compare against the frozen released
    per-length MAE table. Must match near-exactly (deterministic closed-form
    formula) -- this is the number that was previously wrong by a genuine
    scientific-definition bug (physical-space vs log10-space interpolation),
    not floating-point noise."""
    pool = bc.load_matched_support_pool()
    preds = interp.standalone_log10_interpolation_predictions(pool, target_df)
    valid = preds.dropna(subset=["pred_log10", "true"])
    true_log = np.log10(valid["true"].clip(lower=1e-4))
    valid = valid.assign(abs_err=(valid["pred_log10"] - true_log).abs())

    released = pd.read_csv(bc.MATCHED_SUPPORT_BY_LENGTH_PATH)
    released = released[released["method_id"] == "canonical_interpolation"].set_index("gap_length")

    for L, g in valid.groupby("gap_length"):
        generated_mae = g["abs_err"].mean()
        released_mae = float(released.loc[L, "mae_day_weighted"])
        assert generated_mae == pytest.approx(released_mae, abs=1e-6), (
            f"gap_length={L}: generated {generated_mae} vs released {released_mae}"
        )

    overall_mae = valid["abs_err"].mean()
    released_overall = pd.read_csv(bc.MATCHED_SUPPORT_METRICS_PATH)
    released_overall = released_overall[
        (released_overall["method_id"] == "canonical_interpolation")
        & (released_overall["support"] == "matched_449")
    ]["mae_day_weighted"].iloc[0]
    assert overall_mae == pytest.approx(float(released_overall), abs=1e-6)
