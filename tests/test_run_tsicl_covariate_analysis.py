"""Resume/failure-safety and placebo-grid tests for
`experiments.chlorophyll.run_tsicl_covariate_analysis`, monkeypatched so no
real `tsicl`/`torch` dependency is required."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import tsicl_helpers as th
from experiments.chlorophyll import run_tsicl_covariate_analysis as mod
from experiments.chlorophyll import tsicl_covariate_registry as reg

_DATES = pd.date_range("2020-01-01", periods=200, freq="D")
_POOL = pd.DataFrame({
    "gap_id": ["g1", "g2"],
    "start_date": [_DATES[50], _DATES[100]],
    "end_date": [_DATES[52], _DATES[102]],
    "gap_length": [3, 3],
})


def _fake_provenance():
    return {"tsicl_package_version": "0.2.1", "torch_version": "2.9.1",
            "checkpoint_revision": "rev1", "checkpoint_sha256": "hash1"}


def _ok_result(gap, mode):
    return {
        "n_context": 10, "dates": [str(gap.start_date)] * gap.length,
        "pred_log10": [0.1] * gap.length, "true_log10": [0.1] * gap.length,
        "quantile_levels": [0.5], "quantiles_log10": [[0.1]] * gap.length,
    }


def _always_ok(model, dates, target_log10, gap, context_mode, covariate_array=None, strict=True):
    return _ok_result(gap, context_mode)


def _patch_common(monkeypatch, run_gap_inference_impl):
    monkeypatch.setattr(
        mod, "load_target_series",
        lambda target_path: (_DATES.values.astype("datetime64[D]"), np.zeros(len(_DATES), dtype=np.float32)),
    )
    monkeypatch.setattr(mod, "load_gap_specs", lambda pool: [
        mod.th.GapSpec(gap_id=r["gap_id"], start_date=str(r["start_date"].date()),
                        end_date=str(r["end_date"].date()), length=int(r["gap_length"]))
        for _, r in _POOL.iterrows()
    ])
    fake_features = pd.DataFrame({c: 0.0 for c in reg.CURATED_PHYSICAL_COLUMNS}, index=_DATES)
    monkeypatch.setattr(mod, "load_feature_table", lambda features_path: fake_features)
    monkeypatch.setattr(mod, "_sha256_file", lambda path: "fixedhash" * 8)
    monkeypatch.setattr(mod.pd, "read_csv", lambda *a, **k: _POOL)
    monkeypatch.setattr(th, "load_tsicl_strict", lambda: (object(), _fake_provenance()))
    monkeypatch.setattr(th, "run_gap_inference", run_gap_inference_impl)


def test_placebo_grid_includes_all_six_eligible_families(tmp_path, monkeypatch):
    _patch_common(monkeypatch, _always_ok)
    out_dir = tmp_path / "run1"
    rc = mod.run_analysis(
        arm_ids=["curated_physical"], support="matched_449", include_placebos=True,
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv",
        extension_path=tmp_path / "e.csv", out_dir=out_dir,
    )
    assert rc == 0
    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    # 1 base arm + 4 placebo transforms (curated_physical is placebo-eligible) = 5 variants x 2 gaps
    assert metadata["n_variants"] == 1 + 4
    assert metadata["n_expected_calls"] == (1 + 4) * 2


def test_resume_retries_failed_key_and_reaches_run_complete(tmp_path, monkeypatch):
    g1_attempts = {"n": 0}

    def flaky(model, dates, target_log10, gap, context_mode, covariate_array=None, strict=True):
        if gap.gap_id == "g1":
            g1_attempts["n"] += 1
            if g1_attempts["n"] < 2:
                raise th.TSICLOutputError("simulated failure")
        return _ok_result(gap, context_mode)

    _patch_common(monkeypatch, flaky)
    out_dir = tmp_path / "run2"
    rc1 = mod.run_analysis(
        arm_ids=["target_only"], support="matched_449", include_placebos=False,
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv",
        extension_path=tmp_path / "e.csv", out_dir=out_dir,
    )
    assert rc1 == 1
    rc2 = mod.run_analysis(
        arm_ids=["target_only"], support="matched_449", include_placebos=False,
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv",
        extension_path=tmp_path / "e.csv", out_dir=out_dir,
    )
    assert rc2 == 0
    pred_lines = (out_dir / "predictions.jsonl").read_text().strip().splitlines()
    keys = [json.loads(line)["key"] for line in pred_lines]
    assert sorted(keys) == sorted(set(keys))
