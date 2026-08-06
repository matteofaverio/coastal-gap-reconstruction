"""Regression test for the oxygen artificial-gap pool's support-tier labeling.

The pool contains 6 gaps at lengths (45/60/90/120 days) outside the
documented L=1-30 primary range. These are real, intentionally kept
(not deleted), but must stay clearly separated from the validated primary
support so no published number is read as covering a length it doesn't.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
GAP_POOL = REPO_ROOT / "data" / "oxygen" / "oxygen_validation_gaps.csv"
BENCHMARK_BY_LENGTH = REPO_ROOT / "results" / "oxygen" / "oxygen_benchmark_by_length.csv"

PRIMARY_LENGTHS = {1, 3, 7, 10, 14, 21, 30}
EXTENDED_LENGTHS = {45, 60, 90, 120}


def test_support_role_column_present_and_complete() -> None:
    df = pd.read_csv(GAP_POOL)
    assert "support_role" in df.columns
    assert df["support_role"].notna().all()
    assert set(df["support_role"].unique()) == {"primary", "exploratory_extended"}


def test_support_role_matches_gap_length() -> None:
    df = pd.read_csv(GAP_POOL)
    primary = df[df["support_role"] == "primary"]
    extended = df[df["support_role"] == "exploratory_extended"]
    assert set(primary["gap_length"].unique()) == PRIMARY_LENGTHS
    assert set(extended["gap_length"].unique()) == EXTENDED_LENGTHS
    assert len(primary) == 406
    assert len(extended) == 6


def test_published_benchmark_stays_within_primary_support() -> None:
    """The headline oxygen benchmark table must not silently include the
    exploratory extended lengths -- it should only ever report on lengths
    that are in the primary tier."""
    bench = pd.read_csv(BENCHMARK_BY_LENGTH)
    assert "gap_length" in bench.columns
    lengths_used = set(bench["gap_length"].unique())
    assert lengths_used <= PRIMARY_LENGTHS, (
        f"oxygen_benchmark_by_length.csv reports on length(s) {lengths_used - PRIMARY_LENGTHS} "
        "outside the primary (validated) support -- this would silently present "
        "exploratory-extended-tier gaps as if they were headline evidence."
    )
