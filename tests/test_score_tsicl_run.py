"""Tests for `experiments.chlorophyll.score_tsicl_run`."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from experiments.chlorophyll import score_tsicl_run as mod


def _write_predictions(path, records):
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _pred_record(gap_id, arm, pred, true, context_mode="full_series", dates=None):
    n = len(pred)
    dates = dates or [f"2020-01-{i+1:02d}" for i in range(n)]
    return {
        "key": f"{gap_id}|{context_mode}|{arm}", "gap_id": gap_id, "context_mode": context_mode,
        "arm": arm, "gap_length": n, "date": dates, "pred_log10": pred, "true_log10": true,
    }


def test_load_day_level_explodes_multi_day_gaps_correctly(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    _write_predictions(pred_path, [_pred_record("g1", "target_only", [0.1, 0.2], [0.15, 0.25])])
    day_level = mod.load_day_level(pred_path)
    assert len(day_level) == 2
    assert day_level["method_id"].unique().tolist() == ["tsicl_target_only__full_series"]
    assert day_level["absolute_error_log10"].tolist() == pytest.approx([0.05, 0.05])


def test_compute_aggregate_metrics_basic():
    day_level = pd.DataFrame([
        {"method_id": "m1", "gap_id": "g1", "gap_length": 2, "pred_log10": 0.1, "true_log10": 0.2,
         "absolute_error_log10": 0.1},
        {"method_id": "m1", "gap_id": "g1", "gap_length": 2, "pred_log10": 0.3, "true_log10": 0.3,
         "absolute_error_log10": 0.0},
        {"method_id": "m1", "gap_id": "g2", "gap_length": 1, "pred_log10": 0.0, "true_log10": 0.4,
         "absolute_error_log10": 0.4},
    ])
    agg = mod.compute_aggregate_metrics(day_level).set_index("method_id")
    assert agg.loc["m1", "n_gaps"] == 2
    assert agg.loc["m1", "n_rows"] == 3
    assert agg.loc["m1", "mae_day_weighted"] == np.mean([0.1, 0.0, 0.4])
    # gap-weighted: g1 mean=(0.1+0.0)/2=0.05, g2 mean=0.4 -> mean(0.05, 0.4)
    assert agg.loc["m1", "mae_gap_weighted"] == np.mean([0.05, 0.4])


def test_compute_event_background_metrics_splits_by_flag():
    day_level = pd.DataFrame([
        {"method_id": "m1", "gap_id": "g1", "absolute_error_log10": 0.1},
        {"method_id": "m1", "gap_id": "g2", "absolute_error_log10": 0.5},
    ])
    pool = pd.DataFrame([{"gap_id": "g1", "is_high_chl_event": False},
                          {"gap_id": "g2", "is_high_chl_event": True}])
    result = mod.compute_event_background_metrics(day_level, pool).set_index("is_high_chl_event")
    assert result.loc[False, "mae_day_weighted"] == 0.1
    assert result.loc[True, "mae_day_weighted"] == 0.5


def test_classify_within_empirical_band_vs_mismatched():
    cls, diff = mod._classify("mae_day_weighted", 0.20, 0.205, tol=0.01)
    assert cls == "within_empirical_reporting_band"
    cls2, diff2 = mod._classify("mae_day_weighted", 0.20, 0.35, tol=0.01)
    assert cls2 == "mismatched"


def test_run_scoring_end_to_end_with_synthetic_pool_and_target(tmp_path, monkeypatch):
    dates = pd.date_range("2020-01-01", periods=60, freq="D")
    rng = np.random.default_rng(0)
    target_df = pd.DataFrame({
        "date": dates, "chl_mean": np.abs(rng.normal(1.0, 0.2, len(dates))) + 0.1,
        "target_eligible_default": True,
    })
    target_path = tmp_path / "target.csv"
    target_df.to_csv(target_path, index=False)

    pool = pd.DataFrame({
        "gap_id": ["g1", "g2"],
        "start_date": [dates[20], dates[40]],
        "end_date": [dates[22], dates[42]],
        "gap_length": [3, 3],
        "is_high_chl_event": [False, True],
    })
    pool_path = tmp_path / "pool.csv"
    pool.to_csv(pool_path, index=False)

    monkeypatch.setattr(mod.tc, "FULL_POOL_PATH", pool_path)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [
        _pred_record("g1", "target_only", [0.05, 0.06, 0.07], [0.10, 0.10, 0.10],
                     dates=[str(dates[20].date()), str(dates[21].date()), str(dates[22].date())]),
        _pred_record("g2", "target_only", [0.10, 0.10, 0.10], [0.12, 0.11, 0.10],
                     dates=[str(dates[40].date()), str(dates[41].date()), str(dates[42].date())]),
    ]
    _write_predictions(run_dir / "predictions.jsonl", records)
    (run_dir / "run_metadata.json").write_text(json.dumps({"support": "full_681"}))

    rc = mod.run_scoring(run_dir, target_path)
    assert rc in (0, 1)  # not_applicable (full_681, no matched-449 comparison) is acceptable here
    assert (run_dir / "scored_day_level.csv").exists()
    assert (run_dir / "aggregate_metrics.csv").exists()
    assert (run_dir / "by_length_metrics.csv").exists()
    assert (run_dir / "event_background_metrics.csv").exists()
    assert (run_dir / "paired_bootstrap.csv").exists()
    assert (run_dir / "VERIFICATION_STATUS").exists()
    status = (run_dir / "VERIFICATION_STATUS").read_text().strip()
    assert status == "VERIFICATION_NOT_APPLICABLE"  # full_681 support has no matched-449 aggregate row to check
