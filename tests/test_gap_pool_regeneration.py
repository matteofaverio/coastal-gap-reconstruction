"""Byte-identical regeneration tests for the chlorophyll and oxygen gap pools.

These are the acceptance tests for `experiments/chlorophyll/target_and_gap_pool.py`
and `experiments/oxygen/target_and_gap_pool.py`: regenerating from the public daily
target tables must reproduce the released validation-pool CSVs exactly.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import _config as chl_config
from experiments.chlorophyll import target_and_gap_pool as chl_tgp
from experiments.oxygen import target_and_gap_pool as ox_tgp

REPO_ROOT = Path(__file__).resolve().parent.parent
CHL_TARGET = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
CHL_POOL = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_validation_gaps.csv"
OX_TARGET = REPO_ROOT / "data_public" / "oxygen" / "oxygen_daily_target.csv"
OX_POOL = REPO_ROOT / "data_public" / "oxygen" / "oxygen_validation_gaps.csv"

# One released row (L10_20160622) differs from a from-scratch regeneration by
# 0.0001 on target_mean_true only -- a floating-point summation-order artifact
# (verified: the selected start date and every other column for that row match
# exactly). Documented, not hidden: see the module docstring of
# experiments/chlorophyll/target_and_gap_pool.py for the full diagnosis.
_KNOWN_FLOAT_ROUNDING_TOLERANCE = 2e-4


def _columns_equal(a: pd.Series, b: pd.Series, atol: float = 1e-9) -> bool:
    if a.dtype == float or b.dtype == float:
        return bool(np.isclose(a.astype(float), b.astype(float), atol=atol, equal_nan=True).all())
    return bool((a.astype(str) == b.astype(str)).all())


def test_chlorophyll_full_pool_matches_released_pool_within_documented_tolerance() -> None:
    """All 681 released rows, all 18 columns, regenerated from the public daily
    target table via the two recovered sampling procedures (core lengths via
    coastal_gap_reconstruction.gaps.sample_nonoverlapping; extended lengths via
    sample_nonoverlapping_sequential, recovered from
    src/tongoy_chl/models/tier_c_7c_extended_eval.py in the private repository)."""
    target_df = chl_tgp.load_daily_target(CHL_TARGET)
    checksum = chl_tgp.target_table_checksum(CHL_TARGET)

    regenerated = chl_tgp.build_gap_pool(target_df, checksum)
    frozen = pd.read_csv(CHL_POOL, parse_dates=["start_date", "end_date"])

    assert len(regenerated) == 681
    assert len(frozen) == 681
    assert set(regenerated["gap_id"]) == set(frozen["gap_id"])
    assert list(regenerated.columns) == chl_tgp.POOL_COLUMNS

    merged = (
        regenerated.sort_values("gap_id")
        .reset_index(drop=True)
        .merge(
            frozen.sort_values("gap_id").reset_index(drop=True),
            on="gap_id",
            suffixes=("_new", "_frozen"),
        )
    )
    for col in chl_tgp.POOL_COLUMNS:
        if col == "gap_id":
            continue
        assert _columns_equal(
            merged[f"{col}_new"], merged[f"{col}_frozen"], atol=_KNOWN_FLOAT_ROUNDING_TOLERANCE
        ), f"column {col!r} diverged from the released pool beyond the documented tolerance"


def test_chlorophyll_full_pool_matches_bit_exactly_except_one_documented_row() -> None:
    """Stricter companion to the tolerance-based test above: at float atol=1e-9,
    exactly one row (L10_20160622) may differ, and only on target_mean_true/
    target_max_true. Any other mismatch is a real regression, not known noise."""
    target_df = chl_tgp.load_daily_target(CHL_TARGET)
    checksum = chl_tgp.target_table_checksum(CHL_TARGET)
    regenerated = chl_tgp.build_gap_pool(target_df, checksum)
    frozen = pd.read_csv(CHL_POOL, parse_dates=["start_date", "end_date"])

    merged = (
        regenerated.sort_values("gap_id")
        .reset_index(drop=True)
        .merge(
            frozen.sort_values("gap_id").reset_index(drop=True),
            on="gap_id",
            suffixes=("_new", "_frozen"),
        )
    )
    mismatching_gap_ids: set[str] = set()
    mismatching_columns: set[str] = set()
    for col in chl_tgp.POOL_COLUMNS:
        if col == "gap_id":
            continue
        a, b = merged[f"{col}_new"], merged[f"{col}_frozen"]
        eq = (
            np.isclose(a.astype(float), b.astype(float), atol=1e-9, equal_nan=True)
            if a.dtype == float or b.dtype == float
            else (a.astype(str) == b.astype(str))
        )
        if not eq.all():
            mismatching_columns.add(col)
            mismatching_gap_ids.update(merged.loc[~eq, "gap_id"])

    assert mismatching_gap_ids == {"L10_20160622"}
    assert mismatching_columns == {"target_mean_true"}


def test_chlorophyll_gap_length_support_and_counts() -> None:
    frozen = pd.read_csv(CHL_POOL)
    counts = frozen["gap_length"].value_counts().sort_index().to_dict()
    assert counts == {1: 100, 3: 100, 7: 100, 10: 100, 14: 100, 21: 80, 30: 50, 45: 29, 60: 22}
    assert len(frozen) == 681


@pytest.mark.parametrize("gap_length", chl_config.GAP_LENGTHS)
def test_chlorophyll_context_constrained_rule(gap_length: int) -> None:
    """context_constrained = pre/post < (30 if gap_length<=30 else 60), verified
    against every released row of the given length, not an approximate rule."""
    frozen = pd.read_csv(CHL_POOL)
    sub = frozen[frozen["gap_length"] == gap_length]
    required = chl_config.required_context_days(gap_length)
    expected = (
        (sub["pre_context_available_days"] < required)
        | (sub["post_context_available_days"] < required)
    )
    assert (sub["context_constrained"] == expected).all()


def test_oxygen_pool_byte_identical_to_released_pool() -> None:
    target_df = ox_tgp.load_daily_target(OX_TARGET)
    regenerated = ox_tgp.build_gap_pool(target_df)
    regenerated = ox_tgp.add_support_role(regenerated)

    frozen = pd.read_csv(OX_POOL, parse_dates=["start_date", "end_date"])

    assert len(regenerated) == 412
    assert len(frozen) == 412
    assert set(regenerated["gap_id"]) == set(frozen["gap_id"])
    assert list(regenerated.columns) == ox_tgp.POOL_COLUMNS

    merged = (
        regenerated.sort_values("gap_id")
        .reset_index(drop=True)
        .merge(
            frozen.sort_values("gap_id").reset_index(drop=True),
            on="gap_id",
            suffixes=("_new", "_frozen"),
        )
    )
    for col in ox_tgp.POOL_COLUMNS:
        if col == "gap_id":
            continue
        assert _columns_equal(merged[f"{col}_new"], merged[f"{col}_frozen"]), (
            f"column {col!r} diverged from the released pool"
        )


def test_oxygen_support_role_matches_length_split() -> None:
    frozen = pd.read_csv(OX_POOL)
    primary_lengths = set(frozen.loc[frozen["support_role"] == "primary", "gap_length"])
    exploratory_lengths = set(frozen.loc[frozen["support_role"] == "exploratory_extended", "gap_length"])
    assert primary_lengths == {1, 3, 7, 10, 14, 21, 30}
    assert exploratory_lengths == {45, 60, 90, 120}


def test_oxygen_pool_has_no_chlorophyll_event_columns() -> None:
    """Oxygen has no event/high-value label anywhere in this project -- the
    released pool schema must not carry a chlorophyll-shaped column by accident."""
    frozen = pd.read_csv(OX_POOL)
    forbidden = {"is_high_chl_event", "is_sustained_event", "is_background", "chl_90th_threshold"}
    assert forbidden.isdisjoint(frozen.columns)
