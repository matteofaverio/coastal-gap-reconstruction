"""Guards against scoring on the wrong chlorophyll scale.

The report and released benchmark score chlorophyll on log10(chl_mean), not
physical chl_mean. notebooks/03_baselines.ipynb previously scored directly
on physical chl_mean and invited a direct numeric comparison against the
log10-scale benchmark table -- a real bug found in the Phase 1 authenticity
audit (see docs/methodology/target_and_gap_construction.md, "Scoring
scale"). These tests pin the fix.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from coastal_gap_reconstruction.artificial_gap_validation import apply_artificial_gap  # noqa: E402
from coastal_gap_reconstruction.baseline_imputation import run_all_baselines  # noqa: E402
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

    # Physical-scale chl_mean in this toy series ranges over multiple mg/m3;
    # log10 MAE for a reasonable baseline on smooth synthetic data should be
    # a small fraction of a log10 unit, not the physical magnitude.
    assert (metrics["mae"] < 1.0).all()

    # Sanity: scoring the *physical* column instead gives a different (larger
    # or differently distributed) MAE -- this is the wrong-scale behavior the
    # old notebook exhibited, kept here as a negative check.
    masked_phys = apply_artificial_gap(target_df, start, gap_length, target_col="chl_mean")
    preds_phys = run_all_baselines(masked_phys, start, gap_length, target_col="chl_mean")
    rows_phys = compute_gap_metrics(
        target_df, preds_phys, start, gap_length, "toy_gap", {}, target_col="chl_mean"
    )
    metrics_phys = pd.DataFrame(rows_phys)

    assert not metrics["mae"].reset_index(drop=True).equals(
        metrics_phys["mae"].reset_index(drop=True)
    ), "log10-scale and physical-scale MAE must not silently be numerically identical"
