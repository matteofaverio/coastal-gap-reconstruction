"""End-to-end resume/failure-safety tests for
`experiments.chlorophyll.run_tsicl_benchmark`, exercising the full driver
loop (not just the underlying `tsicl_run_state`/`tsicl_run_manifest`
mechanics) with `th.load_tsicl_strict`/`th.run_gap_inference` monkeypatched
so no real `tsicl`/`torch` dependency or checkpoint is required.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import tsicl_helpers as th
from experiments.chlorophyll import run_tsicl_benchmark as mod

_DATES = pd.date_range("2020-01-01", periods=200, freq="D")
_POOL = pd.DataFrame({
    "gap_id": ["g1", "g2", "g3"],
    "start_date": [_DATES[50], _DATES[100], _DATES[150]],
    "end_date": [_DATES[52], _DATES[102], _DATES[152]],
    "gap_length": [3, 3, 3],
})


def _fake_provenance():
    return {"tsicl_package_version": "0.2.1", "torch_version": "2.9.1",
            "checkpoint_revision": "rev1", "checkpoint_sha256": "hash1"}


def _patch_common(monkeypatch, run_gap_inference_impl):
    monkeypatch.setattr(mod, "_load_pool", lambda support: (_POOL, "fake_pool_path"))
    monkeypatch.setattr(
        mod, "load_target_series",
        lambda target_path: (_DATES.values.astype("datetime64[D]"), np.zeros(len(_DATES), dtype=np.float32)),
    )
    monkeypatch.setattr(mod, "load_feature_table", lambda features_path: pd.DataFrame(index=_DATES))
    monkeypatch.setattr(mod, "_sha256_file", lambda path: "fixedhash" * 8)
    monkeypatch.setattr(th, "load_tsicl_strict", lambda: (object(), _fake_provenance()))
    monkeypatch.setattr(th, "run_gap_inference", run_gap_inference_impl)


def _ok_result(gap, mode):
    return {
        "n_context": 10, "dates": [str(gap.start_date)] * gap.length,
        "pred_log10": [0.1] * gap.length, "true_log10": [0.1] * gap.length,
        "quantile_levels": [0.5], "quantiles_log10": [[0.1]] * gap.length,
    }


def test_first_run_records_a_failure_and_resume_retries_it(tmp_path, monkeypatch):
    g2_attempts = {"n": 0}

    def flaky(model, dates, target_log10, gap, context_mode, covariate_array=None, strict=True):
        if gap.gap_id == "g2":
            g2_attempts["n"] += 1
            if g2_attempts["n"] < 2:
                raise th.TSICLOutputError("simulated transient failure")
        return _ok_result(gap, context_mode)

    _patch_common(monkeypatch, flaky)
    out_dir = tmp_path / "run1"
    rc = mod.run_benchmark(
        support="full_681", arms=["target_only"], context_modes=["full_series"],
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv", out_dir=out_dir,
    )
    assert rc == 1  # g2 still unresolved after the first pass
    status_text = (out_dir / "RUN_STATUS").read_text()
    assert status_text.startswith("RUN_PARTIAL")
    fail_lines = (out_dir / "failures.jsonl").read_text().strip().splitlines()
    assert any(json.loads(line)["gap_id"] == "g2" for line in fail_lines)

    # Resume: g2 now succeeds (calls["n"] keeps climbing past the threshold).
    rc2 = mod.run_benchmark(
        support="full_681", arms=["target_only"], context_modes=["full_series"],
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv", out_dir=out_dir,
    )
    assert rc2 == 0
    assert (out_dir / "RUN_STATUS").read_text().startswith("RUN_COMPLETE")

    # The successful predictions.jsonl must contain exactly one record per
    # expected key -- no duplicate row for g1/g3, which never failed.
    pred_lines = (out_dir / "predictions.jsonl").read_text().strip().splitlines()
    keys = [json.loads(line)["key"] for line in pred_lines]
    assert sorted(keys) == sorted(set(keys)), "no duplicate successful records after resume"
    assert len(keys) == 3  # g1, g2, g3 -- g2 appears once despite multiple failed attempts


def test_run_complete_is_never_reported_while_a_key_is_permanently_failed(tmp_path, monkeypatch):
    def always_fails(model, dates, target_log10, gap, context_mode, covariate_array=None, strict=True):
        if gap.gap_id == "g2":
            raise th.TSICLOutputError("permanent failure")
        return _ok_result(gap, context_mode)

    _patch_common(monkeypatch, always_fails)
    out_dir = tmp_path / "run2"
    for _ in range(3):  # multiple resumes, g2 never recovers
        rc = mod.run_benchmark(
            support="full_681", arms=["target_only"], context_modes=["full_series"],
            target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv", out_dir=out_dir,
        )
    assert rc == 1
    assert (out_dir / "RUN_STATUS").read_text().startswith("RUN_PARTIAL")
    # g2's failure count must have accumulated across all 3 invocations, not
    # reset to 0/1 each time (the exact bug this sprint fixed).
    fail_lines = (out_dir / "failures.jsonl").read_text().strip().splitlines()
    g2_failures = [line for line in fail_lines if json.loads(line)["gap_id"] == "g2"]
    assert len(g2_failures) == 3


def test_resume_with_a_different_arm_list_raises_config_mismatch(tmp_path, monkeypatch):
    def always_ok(model, dates, target_log10, gap, context_mode, covariate_array=None, strict=True):
        return _ok_result(gap, context_mode)

    _patch_common(monkeypatch, always_ok)
    out_dir = tmp_path / "run3"
    mod.run_benchmark(
        support="full_681", arms=["target_only"], context_modes=["full_series"],
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv", out_dir=out_dir,
    )
    rc2 = mod.run_benchmark(
        support="full_681", arms=["target_only", "satellite_proxy"], context_modes=["full_series"],
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv", out_dir=out_dir,
    )
    assert rc2 == 1  # rejected, not silently mixed
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["identity"]["arms"] == ["target_only"]  # unchanged by the rejected attempt


def test_run_metadata_contains_no_local_absolute_path(tmp_path, monkeypatch):
    _patch_common(monkeypatch, lambda model, dates, target_log10, gap, context_mode, covariate_array=None, strict=True: _ok_result(gap, context_mode))
    out_dir = tmp_path / "run4"
    mod.run_benchmark(
        support="full_681", arms=["target_only"], context_modes=["full_series"],
        target_path=tmp_path / "t.csv", features_path=tmp_path / "f.csv", out_dir=out_dir,
    )
    metadata = json.loads((out_dir / "run_metadata.json").read_text())
    assert "checkpoint_path" not in metadata["provenance"]
    assert "checkpoint_path" not in json.dumps(metadata["provenance"])
