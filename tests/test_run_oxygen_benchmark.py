"""Tests for `experiments.oxygen.run_oxygen_benchmark`, monkeypatched so no
real `tsicl`/`torch` dependency is required. Confirms the driver correctly
reuses the chlorophyll resume-state/config-manifest infrastructure rather
than duplicating it."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import tsicl_helpers as th
from experiments.oxygen import run_oxygen_benchmark as mod

_POOL = pd.DataFrame({
    "gap_id": [f"OX_L{L:03d}_test" for L in (1, 1, 3, 3, 7, 10, 14, 21, 30)],
    "gap_length": [1, 1, 3, 3, 7, 10, 14, 21, 30],
    "start_date": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-03-01", "2018-04-01",
                                   "2018-05-01", "2018-06-01", "2018-07-01", "2018-08-01", "2018-09-01"]),
    "end_date": pd.to_datetime(["2018-01-01", "2018-02-01", "2018-03-03", "2018-04-03",
                                 "2018-05-07", "2018-06-10", "2018-07-14", "2018-08-21", "2018-09-30"]),
    "support_role": "primary",
})


def _fake_provenance():
    return {"tsicl_package_version": "0.2.1", "torch_version": "2.9.1",
            "checkpoint_revision": "rev1", "checkpoint_sha256": "hash1"}


def _ok_result(gap, mode):
    return {
        "n_context": 10, "dates": [str(pd.Timestamp("2018-01-01").date())] * gap.length,
        "pred_log10": [7.0] * gap.length, "true_log10": [7.1] * gap.length,
        "quantile_levels": [0.5], "quantiles_log10": [[7.0]] * gap.length,
    }


def test_deterministic_subset_selection_is_reproducible():
    s1 = mod.select_deterministic_tsicl_subset(_POOL, n_gaps=7)
    s2 = mod.select_deterministic_tsicl_subset(_POOL, n_gaps=7)
    assert list(s1["gap_id"]) == list(s2["gap_id"])


def test_deterministic_subset_never_exceeds_n_gaps():
    subset = mod.select_deterministic_tsicl_subset(_POOL, n_gaps=5)
    assert len(subset) <= 5


def test_tsicl_bounded_rejects_a_request_exceeding_max_calls(monkeypatch, tmp_path):
    monkeypatch.setattr(th, "load_tsicl_strict", lambda: (object(), _fake_provenance()))
    rc = mod.run_tsicl_bounded(
        _POOL, arms=["target_only", "external_physical_plus_currents"],
        context_modes=["full_series", "edge_balanced"], n_gaps=9, out_dir=tmp_path,
        max_calls=10,  # 9 gaps x 2 arms x 2 modes = 36 > 10
    )
    assert rc == 1


def test_tsicl_bounded_run_reaches_run_complete_with_zero_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(th, "load_tsicl_strict", lambda: (object(), _fake_provenance()))
    monkeypatch.setattr(
        th, "run_gap_inference",
        lambda model, dates, target, gap, context_mode, covariate_array=None, quantile_levels=None, window_days=730, strict=True: _ok_result(gap, context_mode),
    )
    monkeypatch.setattr(
        mod.tm, "load_target_series",
        lambda: (np.arange(np.datetime64("2018-01-01"), np.datetime64("2018-12-31")),
                  np.full(364, 7.0, dtype=np.float32)),
    )
    import experiments.oxygen.feature_registry as fr
    idx = pd.date_range("2018-01-01", periods=364)
    plus_currents_df = pd.DataFrame({"plv_solar_wm2": [0.0] * 364}, index=idx)
    local_btg_df = pd.DataFrame({"btg_water_temp_daily_mean": [0.0] * 364, "btg_pressure_daily_mean": [0.0] * 364}, index=idx)

    def _fake_get_feature_arm(arm, **kw):
        return local_btg_df if arm == "local_btg_temp_pressure_diagnostic" else plus_currents_df

    monkeypatch.setattr(fr, "get_feature_arm", _fake_get_feature_arm)

    rc = mod.run_tsicl_bounded(
        _POOL, arms=["target_only"], context_modes=["full_series"], n_gaps=9, out_dir=tmp_path, max_calls=60,
    )
    assert rc == 0
    status = (tmp_path / "RUN_STATUS").read_text()
    assert status.startswith("RUN_COMPLETE")
    pred_lines = (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
    keys = [json.loads(line)["key"] for line in pred_lines]
    assert sorted(keys) == sorted(set(keys))  # no duplicate successful records


def test_tsicl_bounded_resume_with_different_arms_raises_config_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(th, "load_tsicl_strict", lambda: (object(), _fake_provenance()))
    monkeypatch.setattr(
        th, "run_gap_inference",
        lambda model, dates, target, gap, context_mode, covariate_array=None, quantile_levels=None, window_days=730, strict=True: _ok_result(gap, context_mode),
    )
    monkeypatch.setattr(
        mod.tm, "load_target_series",
        lambda: (np.arange(np.datetime64("2018-01-01"), np.datetime64("2018-12-31")),
                  np.full(364, 7.0, dtype=np.float32)),
    )
    import experiments.oxygen.feature_registry as fr
    idx = pd.date_range("2018-01-01", periods=364)
    plus_currents_df = pd.DataFrame({"plv_solar_wm2": [0.0] * 364}, index=idx)
    local_btg_df = pd.DataFrame({"btg_water_temp_daily_mean": [0.0] * 364, "btg_pressure_daily_mean": [0.0] * 364}, index=idx)

    def _fake_get_feature_arm(arm, **kw):
        return local_btg_df if arm == "local_btg_temp_pressure_diagnostic" else plus_currents_df

    monkeypatch.setattr(fr, "get_feature_arm", _fake_get_feature_arm)

    mod.run_tsicl_bounded(_POOL, arms=["target_only"], context_modes=["full_series"],
                           n_gaps=9, out_dir=tmp_path, max_calls=60)
    rc2 = mod.run_tsicl_bounded(_POOL, arms=["calendar_seasonal"], context_modes=["full_series"],
                                 n_gaps=9, out_dir=tmp_path, max_calls=60)
    assert rc2 == 1  # rejected, not silently mixed
