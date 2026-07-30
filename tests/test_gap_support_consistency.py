"""Consistency checks between released gap-support CSVs, documentation, and
the illustrative public gap-sampling code.

These guard against the released pool definition silently drifting away
from what the docs/notebooks claim, and against the illustrative
`GAP_LENGTHS` default in `artificial_gap_validation.py` silently disagreeing
with the chlorophyll released pool it is documented to match.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CHL_GAP_POOL = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_validation_gaps.csv"
VALIDATION_PROTOCOL_DOC = REPO_ROOT / "docs" / "methodology" / "validation_protocol.md"

CHL_LENGTHS = {1, 3, 7, 10, 14, 21, 30, 45, 60}
OXYGEN_PRIMARY_LENGTHS = {1, 3, 7, 10, 14, 21, 30}
OXYGEN_EXTENDED_LENGTHS = {45, 60, 90, 120}


def test_chlorophyll_pool_lengths_match_documented_set() -> None:
    df = pd.read_csv(CHL_GAP_POOL)
    assert set(df["gap_length"].unique()) == CHL_LENGTHS


def test_validation_protocol_doc_states_chlorophyll_lengths() -> None:
    text = VALIDATION_PROTOCOL_DOC.read_text()
    assert "1, 3, 7, 10, 14, 21, 30, 45, and 60 days" in text


def test_validation_protocol_doc_states_oxygen_support() -> None:
    text = VALIDATION_PROTOCOL_DOC.read_text()
    assert "1, 3, 7, 10, 14, 21, 30 days" in text
    assert "45, 60, 90, 120 days" in text


def test_illustrative_default_gap_lengths_match_chlorophyll_pool() -> None:
    """`GAP_LENGTHS` in artificial_gap_validation.py is documented as matching
    the chlorophyll released pool's *length set* (not its exact gap
    instances or metadata). If this default silently drifts, the module
    docstring's claim becomes false."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from coastal_gap_reconstruction.artificial_gap_validation import GAP_LENGTHS

    assert set(GAP_LENGTHS) == CHL_LENGTHS
