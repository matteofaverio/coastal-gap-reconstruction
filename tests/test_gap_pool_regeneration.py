"""Byte-identical regeneration tests for the chlorophyll and oxygen gap pools.

These are the acceptance tests for `experiments/chlorophyll/target_and_gap_pool.py`
and `experiments/oxygen/target_and_gap_pool.py`: regenerating from the public daily
target tables must reproduce the released validation-pool CSVs exactly, for the
subset of each pool that is actually regenerable (all of oxygen's 412 rows; the
5-length "core" subset of chlorophyll's 681 rows -- see the module docstring of
`experiments/chlorophyll/target_and_gap_pool.py` for why the other 4 lengths are
not currently reproducible from available code).
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


def _columns_equal(a: pd.Series, b: pd.Series) -> bool:
    if a.dtype == float or b.dtype == float:
        return bool(np.isclose(a.astype(float), b.astype(float), atol=1e-9, equal_nan=True).all())
    return bool((a.astype(str) == b.astype(str)).all())


def test_chlorophyll_core_lengths_byte_identical_to_released_pool() -> None:
    target_df = chl_tgp.load_daily_target(CHL_TARGET)
    checksum = chl_tgp.target_table_checksum(CHL_TARGET)

    regenerated = chl_tgp.build_gap_pool(
        target_df, checksum, gap_lengths=chl_config.EXACTLY_REGENERABLE_GAP_LENGTHS
    )
    frozen = pd.read_csv(CHL_POOL, parse_dates=["start_date", "end_date"])
    frozen_core = frozen[frozen["gap_length"].isin(chl_config.EXACTLY_REGENERABLE_GAP_LENGTHS)]

    assert set(regenerated["gap_id"]) == set(frozen_core["gap_id"])
    assert list(regenerated.columns) == chl_tgp.POOL_COLUMNS

    merged = (
        regenerated.sort_values("gap_id")
        .reset_index(drop=True)
        .merge(
            frozen_core.sort_values("gap_id").reset_index(drop=True),
            on="gap_id",
            suffixes=("_new", "_frozen"),
        )
    )
    for col in chl_tgp.POOL_COLUMNS:
        if col == "gap_id":
            continue
        assert _columns_equal(merged[f"{col}_new"], merged[f"{col}_frozen"]), (
            f"column {col!r} diverged from the released pool"
        )


def test_chlorophyll_extended_lengths_are_not_yet_reproducible() -> None:
    """Documents, rather than hides, a known limitation: regenerating the full
    nine-length pool from scratch does not reproduce the released rows for
    L in {10, 21, 45, 60}. If this test starts failing (i.e. it DOES reproduce
    them), that is good news -- update this test and `EXACTLY_REGENERABLE_GAP_LENGTHS`
    together rather than leaving them silently inconsistent."""
    target_df = chl_tgp.load_daily_target(CHL_TARGET)
    checksum = chl_tgp.target_table_checksum(CHL_TARGET)
    extended_lengths = [
        length for length in chl_config.GAP_LENGTHS
        if length not in chl_config.EXACTLY_REGENERABLE_GAP_LENGTHS
    ]
    regenerated = chl_tgp.build_gap_pool(target_df, checksum, gap_lengths=extended_lengths)
    frozen = pd.read_csv(CHL_POOL, parse_dates=["start_date", "end_date"])
    frozen_extended = frozen[frozen["gap_length"].isin(extended_lengths)]

    assert set(regenerated["gap_id"]) != set(frozen_extended["gap_id"]), (
        "extended-length regeneration now matches the released pool -- great, but "
        "EXACTLY_REGENERABLE_GAP_LENGTHS and this test need to be updated together"
    )


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


@pytest.mark.parametrize("gap_length", [1, 3, 7, 14, 30])
def test_chlorophyll_context_constrained_rule(gap_length: int) -> None:
    """context_constrained = pre/post < (30 if gap_length<=30 else 60), verified
    against every released row of the given length, not an approximate rule."""
    frozen = pd.read_csv(CHL_POOL)
    sub = frozen[frozen["gap_length"] == gap_length]
    required = 30 if gap_length <= 30 else 60
    expected = (
        (sub["pre_context_available_days"] < required)
        | (sub["post_context_available_days"] < required)
    )
    assert (sub["context_constrained"] == expected).all()
