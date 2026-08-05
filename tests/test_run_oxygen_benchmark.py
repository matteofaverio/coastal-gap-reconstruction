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


# ── Section 1C/1D integrity fixes: exact-input identity binding, accurate
# support labeling (regression tests for the bounded-run manifest bug) ─────

def _distinct_pool(n=9):
    """A pool with genuinely distinct gap_ids (the module-level _POOL fixture
    above has two accidental gap_id collisions from its zero-padded-length
    naming scheme -- irrelevant to those tests, but these tests need real
    per-gap distinctness)."""
    lengths = [1, 3, 7, 10, 14, 21, 30]
    rows = []
    for i, L in enumerate(lengths):
        rows.append({
            "gap_id": f"OX_L{L:03d}_test{i}", "gap_length": L,
            "start_date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=30 * i),
            "end_date": pd.Timestamp("2018-01-01") + pd.Timedelta(days=30 * i + L - 1),
            "support_role": "primary",
        })
    return pd.DataFrame(rows)


def _patch_common_tsicl_bounded(monkeypatch):
    monkeypatch.setattr(th, "load_tsicl_strict", lambda: (object(), _fake_provenance()))
    monkeypatch.setattr(
        th, "run_gap_inference",
        lambda model, dates, target, gap, context_mode, covariate_array=None, quantile_levels=None, window_days=730, strict=True: _ok_result(gap, context_mode),
    )
    monkeypatch.setattr(
        mod.tm, "load_target_series",
        lambda: (np.arange(np.datetime64("2018-01-01"), np.datetime64("2019-06-01")),
                  np.full(516, 7.0, dtype=np.float32)),
    )


def test_no_placeholder_string_in_bounded_run_manifest(monkeypatch, tmp_path):
    _patch_common_tsicl_bounded(monkeypatch)
    import experiments.oxygen.feature_registry as fr
    idx = pd.date_range("2018-01-01", periods=516)
    plus_currents_df = pd.DataFrame({"plv_solar_wm2": [0.0] * 516}, index=idx)
    local_btg_df = pd.DataFrame({"btg_water_temp_daily_mean": [0.0] * 516, "btg_pressure_daily_mean": [0.0] * 516}, index=idx)
    monkeypatch.setattr(fr, "get_feature_arm",
                         lambda arm, **kw: local_btg_df if arm == "local_btg_temp_pressure_diagnostic" else plus_currents_df)

    mod.run_tsicl_bounded(_distinct_pool(), arms=["target_only"], context_modes=["full_series"],
                           n_gaps=7, out_dir=tmp_path, max_calls=60)
    manifest_text = (tmp_path / "run_manifest.json").read_text()
    assert "n/a_multiple_arms" not in manifest_text
    assert "n/a" not in manifest_text.lower()


def test_support_label_uses_actual_selected_count_not_requested_maximum(monkeypatch, tmp_path):
    _patch_common_tsicl_bounded(monkeypatch)
    import experiments.oxygen.feature_registry as fr
    idx = pd.date_range("2018-01-01", periods=516)
    plus_currents_df = pd.DataFrame({"plv_solar_wm2": [0.0] * 516}, index=idx)
    local_btg_df = pd.DataFrame({"btg_water_temp_daily_mean": [0.0] * 516, "btg_pressure_daily_mean": [0.0] * 516}, index=idx)
    monkeypatch.setattr(fr, "get_feature_arm",
                         lambda arm, **kw: local_btg_df if arm == "local_btg_temp_pressure_diagnostic" else plus_currents_df)

    pool = _distinct_pool()  # 7 distinct lengths
    # Request 9 gaps; the deterministic stratified selector can only find 7
    # (one per length) from this 7-length pool -- the actual selected count.
    mod.run_tsicl_bounded(pool, arms=["target_only"], context_modes=["full_series"],
                           n_gaps=9, out_dir=tmp_path, max_calls=60)
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["requested_max_n_gaps"] == 9
    assert metadata["actual_selected_n_gaps"] == 7
    assert metadata["support"] == "bounded_7_gaps"
    assert "bounded_9_gaps" not in (tmp_path / "run_manifest.json").read_text()


def test_resume_with_different_feature_table_content_raises_config_mismatch(monkeypatch, tmp_path):
    _patch_common_tsicl_bounded(monkeypatch)
    import experiments.oxygen.feature_registry as fr
    idx = pd.date_range("2018-01-01", periods=516)
    local_btg_df = pd.DataFrame({"btg_water_temp_daily_mean": [0.0] * 516, "btg_pressure_daily_mean": [0.0] * 516}, index=idx)

    plus_currents_v1 = pd.DataFrame({"plv_solar_wm2": [0.0] * 516}, index=idx)
    monkeypatch.setattr(fr, "get_feature_arm",
                         lambda arm, **kw: local_btg_df if arm == "local_btg_temp_pressure_diagnostic" else plus_currents_v1)
    pool = _distinct_pool()
    rc1 = mod.run_tsicl_bounded(pool, arms=["target_only"], context_modes=["full_series"],
                                 n_gaps=7, out_dir=tmp_path, max_calls=60)
    assert rc1 == 0

    # Second invocation: feature-table *content* differs (simulating a real
    # feature-table edit) even though the requested arms/gaps are identical.
    # The prior bug (features_sha256="n/a_multiple_arms") could not detect
    # this at all -- the corrected identity must reject it.
    monkeypatch.setattr(mod, "_sha256_file", lambda path: "different_hash_" + str(path))
    rc2 = mod.run_tsicl_bounded(pool, arms=["target_only"], context_modes=["full_series"],
                                 n_gaps=7, out_dir=tmp_path, max_calls=60)
    assert rc2 == 1


def test_resume_with_different_selected_gap_ids_raises_config_mismatch(monkeypatch, tmp_path):
    _patch_common_tsicl_bounded(monkeypatch)
    import experiments.oxygen.feature_registry as fr
    idx = pd.date_range("2018-01-01", periods=516)
    plus_currents_df = pd.DataFrame({"plv_solar_wm2": [0.0] * 516}, index=idx)
    local_btg_df = pd.DataFrame({"btg_water_temp_daily_mean": [0.0] * 516, "btg_pressure_daily_mean": [0.0] * 516}, index=idx)
    monkeypatch.setattr(fr, "get_feature_arm",
                         lambda arm, **kw: local_btg_df if arm == "local_btg_temp_pressure_diagnostic" else plus_currents_df)

    pool_a = _distinct_pool()
    rc1 = mod.run_tsicl_bounded(pool_a, arms=["target_only"], context_modes=["full_series"],
                                 n_gaps=7, out_dir=tmp_path, max_calls=60)
    assert rc1 == 0

    # Same arms/context/support size, but a different underlying gap-ID
    # selection (different start dates -> different deterministic subset
    # despite identical shape) -- must still be caught by the gap-ID hash,
    # not silently resumed against mismatched gap identities.
    pool_b = _distinct_pool()
    pool_b["gap_id"] = pool_b["gap_id"] + "_variant"
    rc2 = mod.run_tsicl_bounded(pool_b, arms=["target_only"], context_modes=["full_series"],
                                 n_gaps=7, out_dir=tmp_path, max_calls=60)
    assert rc2 == 1


def test_arm_registry_config_sha256_changes_with_columns():
    h1 = mod._arm_registry_config_sha256(["target_only"])
    h2 = mod._arm_registry_config_sha256(["calendar_seasonal"])
    assert h1 != h2


def test_gap_ids_sha256_is_order_independent_but_content_sensitive():
    assert mod._gap_ids_sha256(["g1", "g2"]) == mod._gap_ids_sha256(["g2", "g1"])
    assert mod._gap_ids_sha256(["g1", "g2"]) != mod._gap_ids_sha256(["g1", "g3"])
