"""Regression test for the chlorophyll TS-ICL quantile schema.

Guards against a real bug found while building the live demo: an earlier
version of this file had a `pred_chl` column already in physical units
(mg/m3) sitting next to `q05...q95` columns that were still on the log10
scale, with no unit in the column name to disambiguate. This test checks
both that the unit-explicit columns exist and that the two scales are
numerically consistent with each other.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "results_public" / "chlorophyll" / "chlorophyll_reconstruction_tsicl_satellite_proxy.csv"

QUANTILE_LEVELS = ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def test_no_ambiguous_quantile_columns(df: pd.DataFrame) -> None:
    """The old, unit-ambiguous column names must not exist."""
    ambiguous = set(QUANTILE_LEVELS) | {"pred_chl"}
    present = ambiguous & set(df.columns)
    assert not present, (
        f"Found unit-ambiguous column(s) {present} in {CSV_PATH.name} -- "
        "quantile/point-estimate columns must carry an explicit scale suffix "
        "(_log10_chl or _chl_mg_m3)."
    )


def test_expected_unit_explicit_columns_present(df: pd.DataFrame) -> None:
    expected = {"pred_log10_chl", "pred_chl_mg_m3"}
    for q in QUANTILE_LEVELS:
        expected.add(f"{q}_log10_chl")
        expected.add(f"{q}_chl_mg_m3")
    missing = expected - set(df.columns)
    assert not missing, f"Missing expected column(s): {missing}"


def test_physical_scale_quantiles_match_log10_quantiles(df: pd.DataFrame) -> None:
    """10 ** q*_log10_chl must equal q*_chl_mg_m3, within floating-point tolerance."""
    sample = df.dropna(subset=[f"{q}_log10_chl" for q in QUANTILE_LEVELS]).sample(
        n=min(50, len(df)), random_state=0
    )
    for q in QUANTILE_LEVELS:
        log10_col = f"{q}_log10_chl"
        physical_col = f"{q}_chl_mg_m3"
        expected_physical = 10 ** sample[log10_col]
        pd.testing.assert_series_equal(
            expected_physical.rename(physical_col),
            sample[physical_col],
            check_exact=False,
            rtol=1e-6,
        )


def test_physical_scale_quantiles_are_nonnegative(df: pd.DataFrame) -> None:
    """A chlorophyll concentration cannot be negative -- this is exactly the
    failure mode the original bug produced when log10-scale values were used
    directly as physical-unit uncertainty bounds."""
    for q in QUANTILE_LEVELS:
        col = f"{q}_chl_mg_m3"
        assert (df[col].dropna() >= 0).all(), f"{col} contains negative values"


def test_quantiles_are_monotonic_nondecreasing(df: pd.DataFrame) -> None:
    """q05 <= q10 <= ... <= q95 must hold on both scales, row by row."""
    for suffix in ["log10_chl", "chl_mg_m3"]:
        cols = [f"{q}_{suffix}" for q in QUANTILE_LEVELS]
        sub = df[cols].dropna()
        diffs = sub.diff(axis=1).iloc[:, 1:]
        assert (diffs >= -1e-9).all().all(), f"Non-monotonic quantiles found on the {suffix} scale"
