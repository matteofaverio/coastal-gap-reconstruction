"""Tests for `experiments.chlorophyll.assemble_real_gap_candidates`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import assemble_real_gap_candidates as arc
from experiments.chlorophyll import real_gap_contract as rgc
from experiments.chlorophyll import real_gap_inventory as ri


@pytest.fixture(scope="module")
def inventory():
    target_df = pd.read_csv(rgc.DAILY_TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()
    return ri.detect_real_gaps(target_df)


@pytest.fixture(scope="module")
def day_level(inventory):
    eh = arc.load_engineered_hybrid_candidates()
    ts = arc.load_tsicl_satellite_proxy_candidates()
    return arc.assemble_day_level(eh, ts)


def test_load_engineered_hybrid_only_returns_reconstructed_rows():
    eh = arc.load_engineered_hybrid_candidates()
    assert eh["gap_id"].notna().all()
    assert eh["scale"].eq("physical_mg_m3").all()


def test_load_tsicl_satellite_proxy_preserves_both_scales():
    ts = arc.load_tsicl_satellite_proxy_candidates()
    assert "pred_log10_chl" in ts.columns
    assert "point_pred" in ts.columns  # physical scale, no silent conversion between the two


def test_assembled_day_level_row_count_matches_tsicl_hidden_days(day_level):
    ts = arc.load_tsicl_satellite_proxy_candidates()
    assert len(day_level) == len(ts)  # outer join on the superset (TS-ICL covers every gap)


def test_assembled_gap_level_covers_all_128_real_gaps(inventory, day_level):
    gap_level = arc.assemble_gap_level(day_level, inventory)
    assert len(gap_level) == 128


def test_context_constrained_gap_has_tsicl_only_support_note(inventory, day_level):
    gap_level = arc.assemble_gap_level(day_level, inventory)
    row = gap_level[gap_level["gap_id"] == "REAL_OPEN_20260515"]
    assert len(row) == 1
    assert row.iloc[0]["engineered_hybrid_available"] == False  # noqa: E712
    assert row.iloc[0]["tsicl_satellite_proxy_available"] == True  # noqa: E712
    assert "context-constrained" in row.iloc[0]["support_note"]


def test_256_day_scenario_flag_matches_the_frozen_files_exactly(inventory, day_level):
    """The 256-day gap must be flagged scenario_only_256day=True and be the
    *only* gap so flagged -- verified against both released per-method files
    directly, not merely internally consistent."""
    gap_level = arc.assemble_gap_level(day_level, inventory)
    flagged = gap_level[gap_level["scenario_only_256day"]]
    assert list(flagged["gap_id"]) == ["REAL_L091_20200211"]

    ts_raw = pd.read_csv(rgc.TSICL_SATELLITE_PROXY_PATH)
    ts_flagged = ts_raw[ts_raw["scenario_only_256day"] == True]["gap_id"].unique()  # noqa: E712
    assert list(ts_flagged) == ["REAL_L091_20200211"]

    released_combined = pd.read_csv(rgc.CANDIDATE_OUTPUTS_GAP_LEVEL_PATH)
    released_flagged = released_combined[released_combined["scenario_only_256day"] == True]["gap_id"].unique()  # noqa: E712
    assert list(released_flagged) == ["REAL_L091_20200211"]


def test_256_day_scenario_row_count_and_dates_match_frozen(day_level):
    """The 256-day gap must contribute exactly 256 day-level rows, spanning
    exactly its declared start/end dates -- not silently truncated or padded."""
    rows = day_level[day_level["gap_id"] == "REAL_L091_20200211"]
    assert len(rows) == 256
    assert rows["date"].min() == pd.Timestamp("2020-02-11")
    assert rows["date"].max() == pd.Timestamp("2020-10-23")


def test_validation_passes_on_the_real_released_inputs(inventory, day_level):
    violations = arc.validate_candidate_rows(day_level, inventory)
    assert violations == []


def test_validation_detects_duplicate_rows(inventory, day_level):
    corrupted = pd.concat([day_level, day_level.iloc[[0]]], ignore_index=True)
    violations = arc.validate_candidate_rows(corrupted, inventory)
    assert any("duplicate" in v for v in violations)


def test_validation_detects_non_finite_point_prediction(inventory, day_level):
    corrupted = day_level.copy()
    corrupted.loc[corrupted.index[0], "tsicl_satellite_proxy_point_pred_mg_m3"] = np.nan
    corrupted.loc[corrupted.index[1], "tsicl_satellite_proxy_point_pred_mg_m3"] = np.inf
    violations = arc.validate_candidate_rows(corrupted, inventory)
    assert any("non-finite" in v for v in violations)


def test_validation_detects_date_outside_declared_gap_window(inventory, day_level):
    corrupted = day_level.copy()
    corrupted.loc[corrupted.index[0], "date"] = pd.Timestamp("1999-01-01")
    violations = arc.validate_candidate_rows(corrupted, inventory)
    assert any("outside their declared real-gap window" in v for v in violations)


def test_validation_detects_unknown_gap_id(inventory, day_level):
    corrupted = day_level.copy()
    corrupted.loc[corrupted.index[0], "gap_id"] = "NOT_A_REAL_GAP_ID"
    violations = arc.validate_candidate_rows(corrupted, inventory)
    assert any("absent from the real-gap inventory" in v for v in violations)


def test_run_assembly_writes_expected_files_and_passes(tmp_path):
    rc = arc.run_assembly(tmp_path)
    assert rc == 0
    assert (tmp_path / "real_gap_candidates_daily.csv").exists()
    assert (tmp_path / "real_gap_candidates_gap_level.csv").exists()
    assert (tmp_path / "assembly_manifest.json").exists()
    assert (tmp_path / "VALIDATION_STATUS").read_text().strip() == "PASSED"


def test_no_tsicl_inference_or_model_fitting_functions_imported():
    """Structural check: this module must never import tsicl_helpers or any
    sklearn estimator -- it only reads and joins already-frozen CSVs."""
    import inspect
    source = inspect.getsource(arc)
    assert "tsicl_helpers" not in source
    assert "import torch" not in source
    assert "sklearn" not in source
