"""Tests for TS-ICL checkpoint provenance verification -- pinned revision,
filename, hash, and the explicit failure modes when they don't match.

None of these require the real `tsicl` package or checkpoint (they use a
stub model and monkeypatched file paths), except the maintainer-only live
tests gated behind `RUN_TSICL_LIVE_TESTS=1` at the bottom.
"""

from __future__ import annotations

import hashlib
import os

import pytest

from coastal_gap_reconstruction import tsicl_helpers as th


class _FakeModel:
    max_context_length = 4096


def test_checkpoint_provenance_constants_are_pinned_exactly():
    assert th.CHECKPOINT_REPO_ID == "taharnbl/TS-ICL"
    assert th.CHECKPOINT_REVISION == "f01f3869a735694691401cd67a5e19c17e94e220"
    assert th.CHECKPOINT_FILENAME == "tsicl-v1.ckpt"
    assert th.CHECKPOINT_SHA256 == "a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431"
    assert len(th.CHECKPOINT_SHA256) == 64


def test_verify_checkpoint_provenance_raises_when_checkpoint_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(th.Path, "home", lambda: tmp_path)
    with pytest.raises(th.TSICLProvenanceError, match="not found"):
        th.verify_checkpoint_provenance(_FakeModel())


def test_verify_checkpoint_provenance_raises_on_hash_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr(th.Path, "home", lambda: tmp_path)
    cache_dir_name = f"models--{th.CHECKPOINT_REPO_ID.replace('/', '--')}"
    ckpt_dir = tmp_path / ".cache" / "huggingface" / "hub" / cache_dir_name / "snapshots" / th.CHECKPOINT_REVISION
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / th.CHECKPOINT_FILENAME).write_bytes(b"not the real checkpoint")
    with pytest.raises(th.TSICLProvenanceError, match="SHA-256 mismatch"):
        th.verify_checkpoint_provenance(_FakeModel())


def test_verify_checkpoint_provenance_succeeds_on_matching_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(th.Path, "home", lambda: tmp_path)
    cache_dir_name = f"models--{th.CHECKPOINT_REPO_ID.replace('/', '--')}"
    ckpt_dir = tmp_path / ".cache" / "huggingface" / "hub" / cache_dir_name / "snapshots" / th.CHECKPOINT_REVISION
    ckpt_dir.mkdir(parents=True)
    content = b"pretend checkpoint bytes"
    # Monkeypatch the pinned hash to match this fake content's real hash,
    # so the test doesn't depend on the actual 209 MB checkpoint.
    fake_hash = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(th, "CHECKPOINT_SHA256", fake_hash)
    (ckpt_dir / th.CHECKPOINT_FILENAME).write_bytes(content)
    provenance = th.verify_checkpoint_provenance(_FakeModel())
    assert provenance["checkpoint_sha256"] == fake_hash
    assert provenance["checkpoint_revision"] == th.CHECKPOINT_REVISION
    assert provenance["max_context_length"] == 4096


def test_verify_checkpoint_provenance_raises_when_model_lacks_max_context_length(tmp_path, monkeypatch):
    monkeypatch.setattr(th.Path, "home", lambda: tmp_path)
    cache_dir_name = f"models--{th.CHECKPOINT_REPO_ID.replace('/', '--')}"
    ckpt_dir = tmp_path / ".cache" / "huggingface" / "hub" / cache_dir_name / "snapshots" / th.CHECKPOINT_REVISION
    ckpt_dir.mkdir(parents=True)
    content = b"x"
    monkeypatch.setattr(th, "CHECKPOINT_SHA256", hashlib.sha256(content).hexdigest())
    (ckpt_dir / th.CHECKPOINT_FILENAME).write_bytes(content)

    class _NoAttrModel:
        pass

    with pytest.raises(th.TSICLProvenanceError, match="max_context_length"):
        th.verify_checkpoint_provenance(_NoAttrModel())


def test_load_tsicl_strict_raises_dependency_error_when_tsicl_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _blocking_import(name, *args, **kwargs):
        if name == "tsicl":
            raise ImportError("simulated: no module named tsicl")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocking_import)
    with pytest.raises(th.TSICLDependencyError):
        th.load_tsicl_strict()


# ── Maintainer-only live-checkpoint tests ───────────────────────────────
# Gated behind an explicit env var: never run in normal CI (would require
# the ~209 MB checkpoint and the isolated .venv_tsicl environment). Run
# with: RUN_TSICL_LIVE_TESTS=1 .venv_tsicl/bin/python -m pytest
# tests/test_tsicl_provenance.py -v (see docs/methodology/tsicl_usage.md).
requires_live_tsicl = pytest.mark.skipif(
    os.environ.get("RUN_TSICL_LIVE_TESTS") != "1",
    reason="set RUN_TSICL_LIVE_TESTS=1 and run under .venv_tsicl to exercise the real checkpoint",
)


@requires_live_tsicl
def test_live_load_tsicl_strict_verifies_real_checkpoint():
    model, provenance = th.load_tsicl_strict()
    assert provenance["checkpoint_sha256"] == th.CHECKPOINT_SHA256
    assert provenance["checkpoint_revision"] == th.CHECKPOINT_REVISION
    assert provenance["tsicl_package_version"] == th.EXPECTED_TSICL_PACKAGE_VERSION
    assert model.max_context_length == th.MAX_CONTEXT_LENGTH if hasattr(th, "MAX_CONTEXT_LENGTH") else True


@requires_live_tsicl
def test_live_run_gap_inference_never_leaks_hidden_truth():
    import numpy as np
    import pandas as pd

    model, _ = th.load_tsicl_strict()
    target_path = "data_public/chlorophyll/chlorophyll_daily_target.csv"
    df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()
    eligible = df["target_eligible_default"].fillna(False).astype(bool)
    y = df["chl_mean"].where(eligible & (df["chl_mean"] > 1e-4))
    target_log10 = np.log10(y).to_numpy(dtype=np.float32)
    dates = df.index.values.astype("datetime64[D]")

    gap = th.GapSpec("live_test", "2018-02-20", "2018-02-22", 3)
    baseline = th.run_gap_inference(model, dates, target_log10, gap, context_mode="full_series")

    tampered = target_log10.copy()
    start_idx, end_idx = th.get_gap_row_bounds(dates, gap)
    tampered[start_idx:end_idx] = 9999.0
    tampered_result = th.run_gap_inference(model, dates, tampered, gap, context_mode="full_series")

    np.testing.assert_allclose(baseline["pred_log10"], tampered_result["pred_log10"])
