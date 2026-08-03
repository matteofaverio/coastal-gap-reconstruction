"""Tests for the benchmark runner's completion semantics, cache validation,
run metadata, and structured verification report (the runner previously
wrote a `COMPLETE` marker even when methods had failed, cached blindly on
file presence alone, and `--verify` compared only a single aggregate metric
against one global 0.005 tolerance).

Uses tiny synthetic inputs (a handful of days, `canonical_interpolation`
only) so this suite runs in CI in well under a second -- the full 449-gap
benchmark is exercised separately, out of CI, via
`run_classical_benchmark.py`'s own CLI.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import run_classical_benchmark as rcb


@pytest.fixture
def tiny_target(tmp_path):
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    values = 1.0 + 0.1 * np.sin(np.arange(40) / 5)
    df = pd.DataFrame({
        "date": dates, "chl_mean": values, "target_eligible_default": True,
    })
    df.loc[15:19, "target_eligible_default"] = False
    path = tmp_path / "target.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def tiny_features(tmp_path):
    dates = pd.date_range("2020-01-01", periods=40, freq="D")
    df = pd.DataFrame({"date": dates, "doy_sin": np.sin(np.arange(40))})
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)
    return path


def _patch_matched_pool(monkeypatch):
    """Point `bc.load_matched_support_pool` at a tiny 1-gap pool so the
    runner exercises real code paths without the full 449-gap cost."""
    from experiments.chlorophyll import benchmark_contract as bc

    pool = pd.DataFrame([{
        "gap_id": "test_gap_1", "gap_length": 3,
        "start_date": pd.Timestamp("2020-01-16"), "end_date": pd.Timestamp("2020-01-18"),
    }])
    monkeypatch.setattr(bc, "load_matched_support_pool", lambda: pool)


def test_completion_marker_is_complete_on_a_clean_run(tmp_path, tiny_target, tiny_features, monkeypatch):
    _patch_matched_pool(monkeypatch)
    out_dir = tmp_path / "out"
    rc = rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    assert rc == 0
    assert (out_dir / "COMPLETE").exists()
    assert not (out_dir / "FAILED").exists()
    assert not (out_dir / "INCOMPLETE").exists()


def test_completion_marker_is_not_written_after_a_failure(tmp_path, tiny_target, tiny_features, monkeypatch):
    _patch_matched_pool(monkeypatch)
    out_dir = tmp_path / "out"

    def _broken_run_method(method_id, candidates, target_df, features_df):
        raise RuntimeError("synthetic failure for the completion-marker test")

    monkeypatch.setattr(rcb, "run_method", _broken_run_method)
    rc = rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    assert rc != 0
    assert not (out_dir / "COMPLETE").exists()
    assert (out_dir / "FAILED").exists()
    failures = json.loads((out_dir / "failures.json").read_text())
    assert len(failures) == 1
    assert failures[0]["method_id"] == "canonical_interpolation"


def test_run_metadata_has_no_local_absolute_paths_and_real_versions(tmp_path, tiny_target, tiny_features, monkeypatch):
    _patch_matched_pool(monkeypatch)
    out_dir = tmp_path / "out"
    rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    meta = json.loads((out_dir / "run_metadata.json").read_text())
    assert meta["versions"]["numpy"] == np.__version__
    assert meta["versions"]["pandas"] == pd.__version__
    assert "platform" in meta["versions"]
    assert len(meta["target_sha256"]) == 64
    assert len(meta["features_sha256"]) == 64
    assert str(tmp_path) not in json.dumps(meta)


def test_cache_is_reused_when_signature_matches(tmp_path, tiny_target, tiny_features, monkeypatch):
    _patch_matched_pool(monkeypatch)
    out_dir = tmp_path / "out"
    rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    first_mtime = (out_dir / "predictions_canonical_interpolation.csv").stat().st_mtime_ns

    rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    second_mtime = (out_dir / "predictions_canonical_interpolation.csv").stat().st_mtime_ns
    assert first_mtime == second_mtime  # not rewritten -- cache was valid


def test_cache_is_invalidated_when_input_file_changes(tmp_path, tiny_target, tiny_features, monkeypatch):
    _patch_matched_pool(monkeypatch)
    out_dir = tmp_path / "out"
    rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    first_mtime = (out_dir / "predictions_canonical_interpolation.csv").stat().st_mtime_ns

    # Mutate the target file's content -- its sha256 in the cached metadata
    # must no longer match, forcing a recompute even though the file path
    # (and gap set / method / seed) are unchanged.
    df = pd.read_csv(tiny_target, parse_dates=["date"])
    df["chl_mean"] = df["chl_mean"] + 0.5
    df.to_csv(tiny_target, index=False)

    rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    second_mtime = (out_dir / "predictions_canonical_interpolation.csv").stat().st_mtime_ns
    assert second_mtime != first_mtime


def test_cache_is_invalidated_when_gap_set_changes(tmp_path, tiny_target, tiny_features, monkeypatch):
    from experiments.chlorophyll import benchmark_contract as bc

    out_dir = tmp_path / "out"
    pool_a = pd.DataFrame([{
        "gap_id": "gap_a", "gap_length": 3,
        "start_date": pd.Timestamp("2020-01-16"), "end_date": pd.Timestamp("2020-01-18"),
    }])
    monkeypatch.setattr(bc, "load_matched_support_pool", lambda: pool_a)
    rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)

    pool_b = pd.DataFrame([{
        "gap_id": "gap_b", "gap_length": 3,
        "start_date": pd.Timestamp("2020-01-16"), "end_date": pd.Timestamp("2020-01-18"),
    }])
    monkeypatch.setattr(bc, "load_matched_support_pool", lambda: pool_b)
    rc = rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    preds = pd.read_csv(out_dir / "predictions_canonical_interpolation.csv")
    assert set(preds["gap_id"]) == {"gap_b"}
    assert rc == 0


def test_duplicate_prediction_rows_are_flagged_as_a_failure(tmp_path, tiny_target, tiny_features, monkeypatch):
    _patch_matched_pool(monkeypatch)
    out_dir = tmp_path / "out"

    def _duplicating_run_method(method_id, candidates, target_df, features_df):
        from experiments.chlorophyll import interpolation_baselines as interp
        preds = interp.standalone_log10_interpolation_predictions(candidates, target_df)
        return pd.concat([preds, preds.iloc[[0]]], ignore_index=True)

    monkeypatch.setattr(rcb, "run_method", _duplicating_run_method)
    rc = rcb.run_benchmark(["canonical_interpolation"], [3], tiny_target, tiny_features, out_dir)
    assert rc != 0
    failures = json.loads((out_dir / "failures.json").read_text())
    assert any("duplicate" in f["error"] for f in failures)


def test_score_predictions_reports_all_documented_metrics():
    preds = pd.DataFrame({
        "gap_id": ["g1", "g1", "g2"],
        "pred_log10": [0.0, 0.1, 0.2],
        "true": [1.0, 1.0, 1.0],
    })
    metrics = rcb.score_predictions(preds)
    for key in ("n_gaps", "n_rows", "mae_day_weighted", "mae_gap_weighted",
                "rmse", "bias_mean", "median_abs_error", "p90_abs_error"):
        assert key in metrics


def test_verification_report_marks_new_methods_not_applicable(tmp_path):
    """A method whose `support_status` is not `frozen_matched_449` (e.g. the
    plain external-only protocol) must never be scored `mismatched` in
    `--verify` -- there is no released row to compare it against."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame([{
        "method_id": "external_only_extratrees", "n_gaps": 1, "n_rows": 1,
        "mae_day_weighted": 0.5, "mae_gap_weighted": 0.5, "rmse": 0.5,
        "bias_mean": 0.0, "median_abs_error": 0.5, "p90_abs_error": 0.5,
    }]).to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(columns=["method_id", "gap_length", "n_gaps", "n_rows",
                           "mae_day_weighted", "mae_gap_weighted"]).to_csv(
        out_dir / "summary_by_length.csv", index=False
    )
    n_bad = rcb.run_verification(out_dir)
    assert n_bad == 0
    report = pd.read_csv(out_dir / "verification_report.csv")
    assert (report["classification"] == "not_applicable").all()


def test_verification_report_flags_a_real_mismatch_for_a_frozen_method(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    pd.DataFrame([{
        "method_id": "canonical_interpolation", "n_gaps": 449, "n_rows": 3999,
        "mae_day_weighted": 999.0,  # deliberately wrong
        "mae_gap_weighted": 999.0, "rmse": 999.0,
        "bias_mean": 999.0, "median_abs_error": 999.0, "p90_abs_error": 999.0,
    }]).to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(columns=["method_id", "gap_length", "n_gaps", "n_rows",
                           "mae_day_weighted", "mae_gap_weighted"]).to_csv(
        out_dir / "summary_by_length.csv", index=False
    )
    n_bad = rcb.run_verification(out_dir)
    assert n_bad > 0
    summary = json.loads((out_dir / "verification_summary.json").read_text())
    assert summary["per_method"]["canonical_interpolation"]["overall"] == "mismatched"
