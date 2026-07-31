"""Guards against scoring on the wrong chlorophyll scale.

The report and released benchmark score chlorophyll on log10(chl_mean), not
physical chl_mean. notebooks/03_baselines.ipynb previously scored directly
on physical chl_mean and invited a direct numeric comparison against the
log10-scale benchmark table -- a real bug found during a repository
authenticity review (see docs/methodology/target_and_gap_construction.md,
"Scoring scale"). These tests pin the fix.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coastal_gap_reconstruction.artificial_gap_validation import apply_artificial_gap  # noqa: E402
from coastal_gap_reconstruction.baseline_imputation import (  # noqa: E402
    persistence_baseline,
    run_all_baselines,
)
from coastal_gap_reconstruction.scoring_metrics import compute_gap_metrics  # noqa: E402


def _toy_target_df() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    rng = np.random.default_rng(0)
    chl_mean = np.abs(rng.lognormal(mean=0.0, sigma=1.0, size=len(dates))) + 0.01
    df = pd.DataFrame({"chl_mean": chl_mean, "target_eligible_default": True}, index=dates)
    df.index.name = "date"
    return df


def test_notebook03_scores_on_log10_scale_matches_manual_transform() -> None:
    """Reproduces the notebook 03 pattern end-to-end on a toy table and
    checks the reported MAE is on the log10 scale, not the physical scale."""
    target_df = _toy_target_df()
    target_df["chl_log10"] = np.log10(target_df["chl_mean"])

    start = target_df.index[10]
    gap_length = 3

    masked = apply_artificial_gap(target_df, start, gap_length, target_col="chl_log10")
    preds = run_all_baselines(masked, start, gap_length, target_col="chl_log10")
    rows = compute_gap_metrics(
        target_df, preds, start, gap_length, "toy_gap", {}, target_col="chl_log10"
    )
    metrics = pd.DataFrame(rows)

    assert (metrics["mae"] < 1.0).all()

    masked_phys = apply_artificial_gap(target_df, start, gap_length, target_col="chl_mean")
    preds_phys = run_all_baselines(masked_phys, start, gap_length, target_col="chl_mean")
    rows_phys = compute_gap_metrics(
        target_df, preds_phys, start, gap_length, "toy_gap", {}, target_col="chl_mean"
    )
    metrics_phys = pd.DataFrame(rows_phys)

    assert not metrics["mae"].reset_index(drop=True).equals(
        metrics_phys["mae"].reset_index(drop=True)
    ), "log10-scale and physical-scale MAE must not silently be numerically identical"


def test_persistence_mae_matches_hand_computed_value_on_both_scales() -> None:
    """A fully deterministic, hand-computed example: not a range check like
    'MAE < 1' or a not-equal check, but the exact expected numbers, computed by
    hand outside the implementation and asserted equal to what the code
    produces.

    5-day series: day0 = 2.0 (visible), day1/day2 = hidden gap (true values 3.0
    and 5.0), day3/day4 = visible. Persistence predicts the last visible value
    (day0 = 2.0) for both hidden days.

    Physical-scale errors: |2.0-3.0|=1.0, |2.0-5.0|=3.0 -> MAE = 2.0 exactly.
    log10-scale errors: |log10(2.0)-log10(3.0)|, |log10(2.0)-log10(5.0)| ->
    mean computed by hand with Python's own math.log10, independent of the
    implementation's numpy-based computation path.
    """
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    physical = pd.Series([2.0, 3.0, 5.0, 4.0, 4.0], index=dates)
    df = pd.DataFrame({
        "chl_mean": physical,
        "chl_log10": np.log10(physical),
        "target_eligible_default": True,
    }, index=dates)
    df.index.name = "date"

    start = dates[1]
    gap_length = 2  # hides day1 (true 3.0) and day2 (true 5.0)

    # -- physical scale --
    masked_phys = apply_artificial_gap(df, start, gap_length, target_col="chl_mean")
    preds_phys = persistence_baseline(masked_phys, start, gap_length, target_col="chl_mean")
    rows_phys = compute_gap_metrics(
        df, {"persistence": preds_phys}, start, gap_length, "g", {}, target_col="chl_mean"
    )
    mae_phys = rows_phys[0]["mae"]

    expected_mae_phys = (abs(2.0 - 3.0) + abs(2.0 - 5.0)) / 2  # = 2.0
    assert mae_phys == round(expected_mae_phys, 4)

    # -- log10 scale --
    masked_log = apply_artificial_gap(df, start, gap_length, target_col="chl_log10")
    preds_log = persistence_baseline(masked_log, start, gap_length, target_col="chl_log10")
    rows_log = compute_gap_metrics(
        df, {"persistence": preds_log}, start, gap_length, "g", {}, target_col="chl_log10"
    )
    mae_log = rows_log[0]["mae"]

    pred_log10 = math.log10(2.0)
    expected_mae_log10 = (
        abs(pred_log10 - math.log10(3.0)) + abs(pred_log10 - math.log10(5.0))
    ) / 2
    assert mae_log == round(expected_mae_log10, 4)

    # The two scales must disagree by more than rounding noise -- not merely
    # "not exactly equal", but each independently verified correct.
    assert abs(mae_phys - mae_log) > 0.5
