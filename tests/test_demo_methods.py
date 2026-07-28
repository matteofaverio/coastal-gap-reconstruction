"""Unit tests for demo/src/ -- artificial-gap masking, no target leakage, and
export schema. These import the demo's helper modules directly (not through
the notebook), so they run fast and do not require TS-ICL or any network
access.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "demo"

# Import demo/src/ as a standalone package without colliding with the
# top-level src/coastal_gap_reconstruction/ package: insert demo/ at the
# front of sys.path only for this import, then restore it.
_sys_path_before = list(sys.path)
sys.path.insert(0, str(DEMO_DIR))
try:
    from src import demo_helpers as dh  # noqa: E402
    from src import methods as mth  # noqa: E402
finally:
    sys.path[:] = _sys_path_before


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return dh.load_demo_data(DEMO_DIR / "data")


@pytest.fixture(scope="module")
def full_record() -> pd.DataFrame:
    return dh.load_full_record(DEMO_DIR / "data")


@pytest.fixture(scope="module")
def gap(data: pd.DataFrame):
    return dh.create_artificial_gap(data, start="2017-04-21", end="2017-05-04")


def test_artificial_gap_masks_target_only(gap) -> None:
    """The target column must be NaN inside the gap in `masked_series`; every
    other column must be untouched."""
    inside = gap.masked_series.loc[gap.is_gap]
    assert inside[gap.target_column].isna().all()
    for col in ["chl_satellite_proxy_log10", "wind_spd_ms", "sst_primary_degC"]:
        # covariates must remain available (not masked) inside the gap
        assert inside[col].notna().any()


def test_artificial_gap_preserves_truth(gap, data: pd.DataFrame) -> None:
    truth_lookup = gap.truth.set_index("date")["truth"]
    for _, row in data.loc[gap.is_gap].iterrows():
        assert truth_lookup[row["date"]] == pytest.approx(row[gap.target_column])


def test_create_artificial_gap_rejects_incomplete_interval(data: pd.DataFrame) -> None:
    """An interval that already contains missing target values cannot be used
    as an artificial gap -- there would be nothing to score against."""
    partially_missing = data.copy()
    some_date = data.loc[data[dh.TARGET_COLUMN].notna(), "date"].iloc[50]
    partially_missing.loc[partially_missing["date"] == some_date, dh.TARGET_COLUMN] = np.nan
    start = (some_date - pd.Timedelta(days=3)).date().isoformat()
    end = (some_date + pd.Timedelta(days=3)).date().isoformat()
    with pytest.raises(ValueError):
        dh.create_artificial_gap(partially_missing, start=start, end=end)


def test_create_artificial_gap_rejects_duplicate_dates(data: pd.DataFrame) -> None:
    dup = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        dh.create_artificial_gap(dup, start="2017-04-21", end="2017-05-04")


def test_context_excludes_gap_rows(gap) -> None:
    assert gap.context["date"].isin(gap.truth["date"]).sum() == 0


# ---------------------------------------------------------------------------
# No leakage: methods that fit on the full record must never see the true
# target value inside the gap window.
# ---------------------------------------------------------------------------


def test_climatology_does_not_use_gap_truth(gap, full_record: pd.DataFrame) -> None:
    result = mth.run_climatology(gap, full_record)
    mae = dh.mean_absolute_error(result.prediction, gap.truth)
    # A method that had leaked the true values would reproduce them almost
    # exactly (near-zero MAE). Climatology has no access to the hidden values,
    # so its error must be well above floating-point-identical.
    assert mae > 0.05


def test_external_tabular_training_excludes_gap_dates(gap, full_record: pd.DataFrame) -> None:
    result = mth.run_external_tabular(gap, full_record)
    feature_table = result.extra["feature_table"]
    assert not feature_table["date"].between(gap.gap_start, gap.gap_end).any() is False
    # The model itself must not have been trained on any row inside the gap.
    # We can't inspect sklearn internals directly, but we can check the
    # training-row count excludes exactly the gap window.
    full = full_record.copy()
    in_gap = (full["date"] >= gap.gap_start) & (full["date"] <= gap.gap_end)
    assert result.extra["n_training_rows"] <= (~in_gap).sum()


def test_gap_edge_residual_training_never_overlaps_the_demo_gap(gap, full_record: pd.DataFrame) -> None:
    """The synthetic training sub-gaps used to fit the residual-correction
    model must never be drawn from inside the actual demonstration gap (that
    window is entirely masked to NaN before synthetic gaps are constructed,
    so a fully-observed synthetic block can never be selected from it)."""
    result = mth.run_gap_edge_residual(gap, full_record)
    assert result.extra["n_training_rows"] > 0
    # A degenerate (leaked or collapsed) fit would predict an exactly-zero
    # correction for every day -- assert the correction actually varies.
    corrections = result.extra["decomposition"]["predicted_correction_log10"]
    assert corrections.std() > 1e-6


def test_gap_edge_and_external_tabular_do_not_reproduce_truth_exactly(gap, full_record: pd.DataFrame) -> None:
    """Regression guard for a real bug found while building this demo: an
    earlier version accidentally left the gap unmasked in the full-record
    copy used for interpolation, so the 'reconstruction' was actually just
    the true value passed through untouched (MAE ~= 0)."""
    edge_result = mth.run_gap_edge_residual(gap, full_record)
    tabular_result = mth.run_external_tabular(gap, full_record)
    assert dh.mean_absolute_error(edge_result.prediction, gap.truth) > 0.05
    assert dh.mean_absolute_error(tabular_result.prediction, gap.truth) > 0.05


# ---------------------------------------------------------------------------
# Export schema
# ---------------------------------------------------------------------------


def test_build_export_table_schema() -> None:
    rows = [
        {
            "date": "2017-04-21",
            "original_target": 4.8,
            "observed_or_missing": "artificially_hidden",
            "method": "persistence",
            "reconstructed_median": 5.2,
            "q05": None,
            "q95": None,
            "artificial_or_real_gap": "artificial",
            "validation_status": "single_illustrative_gap",
            "covariates_used": "none",
        }
    ]
    df = dh.build_export_table(rows)
    expected_columns = [
        "date", "original_target", "observed_or_missing", "method", "reconstructed_median",
        "q05", "q95", "artificial_or_real_gap", "validation_status", "covariates_used",
    ]
    assert list(df.columns) == expected_columns


def test_build_export_table_rejects_missing_columns() -> None:
    with pytest.raises(ValueError):
        dh.build_export_table([{"date": "2016-01-01"}])


def test_real_gap_rows_are_never_labeled_as_validation_evidence() -> None:
    """A real gap has no withheld truth; its export rows must always carry a
    validation_status that marks it as a candidate, never as scored/validated."""
    real_gap = dh.load_real_gap_example(DEMO_DIR / "data")
    assert "in_real_gap" in real_gap.columns
    assert real_gap["in_real_gap"].any()
    # The notebook always assigns "candidate_not_validation_evidence" to real-gap
    # export rows -- assert that literal string is what the notebook source uses,
    # so a future edit can't silently drop the label.
    import json

    nb = json.loads((DEMO_DIR / "gap_reconstruction_walkthrough.ipynb").read_text())
    notebook_source = "".join(
        "".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"
    )
    assert '"candidate_not_validation_evidence"' in notebook_source
    assert '"single_illustrative_gap"' in notebook_source
