"""Tests for `experiments.oxygen.real_gap_inventory` -- exact reproduction
of the released by-class real-gap summary, and confirmation that no
reconstruction-candidate claim is made anywhere in this module."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from experiments.oxygen import real_gap_contract as rgc
from experiments.oxygen import real_gap_inventory as ri


@pytest.fixture(scope="module")
def target_df():
    df = pd.read_csv(rgc.DAILY_TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()
    return df


def test_detects_125_real_gaps(target_df):
    inv = ri.detect_real_gaps(target_df)
    assert len(inv) == 125


def test_by_class_aggregate_matches_released_exactly(target_df):
    inv = ri.detect_real_gaps(target_df)
    computed = ri.aggregate_by_class(inv).reset_index(drop=True)
    released = pd.read_csv(rgc.REAL_GAP_INVENTORY_BY_CLASS_PATH).reset_index(drop=True)
    assert computed.equals(released)


def test_by_class_columns_match_contract():
    assert list(ri.BY_CLASS_COLUMNS) == [
        "gap_class", "length_range_days", "n_gaps", "total_missing_days",
        "median_length_days", "max_length_days",
    ]


def test_never_reads_a_candidate_prediction_file():
    sig = inspect.signature(ri.detect_real_gaps)
    assert list(sig.parameters) == ["target_df"]


def test_no_reconstruction_candidate_claim_anywhere_in_module():
    """Oxygen real-gap inventory must never claim a candidate reconstruction
    exists -- structural check that no TS-ICL/model-fitting/reconstruction
    vocabulary appears in this module's source."""
    source = inspect.getsource(ri)
    for forbidden in ("tsicl_helpers", "import torch", "sklearn", "reconstructed_chl", "pred_chl"):
        assert forbidden not in source


def test_oxygen_real_gap_status_declares_no_candidates_exist():
    assert rgc.OXYGEN_REAL_GAP_RECONSTRUCTION_CANDIDATES_EXIST is False
    assert "MISSING" in rgc.OXYGEN_REAL_GAP_STATUS or "no authoritative" in rgc.OXYGEN_REAL_GAP_STATUS
