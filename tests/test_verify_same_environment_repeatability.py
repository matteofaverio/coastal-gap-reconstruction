"""Tests for the `_compare` classification logic in
`experiments.chlorophyll.verify_same_environment_repeatability`."""

from __future__ import annotations

import json

from experiments.chlorophyll import verify_same_environment_repeatability as mod


def _write(path, results):
    path.write_text(json.dumps({"provenance": {}, "results": results}))


def test_identical_results_classify_as_bitwise_repeatable(tmp_path):
    results = {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1],
                                   "quantiles_log10": [[0.05, 0.1, 0.15]]}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, results)
    _write(b, results)
    assert mod._compare(a, b) == 0


def test_tiny_float_diff_classifies_as_numerically_repeatable(tmp_path):
    results_a = {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1],
                                     "quantiles_log10": [[0.05, 0.1, 0.15]]}}
    results_b = {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1 + 1e-9],
                                     "quantiles_log10": [[0.05, 0.1, 0.15]]}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, results_a)
    _write(b, results_b)
    assert mod._compare(a, b) == 0


def test_large_diff_classifies_as_not_repeatable(tmp_path, capsys):
    results_a = {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1],
                                     "quantiles_log10": [[0.05, 0.1, 0.15]]}}
    results_b = {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.5],
                                     "quantiles_log10": [[0.05, 0.1, 0.15]]}}
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, results_a)
    _write(b, results_b)
    rc = mod._compare(a, b)
    assert rc == 1
    assert "not_repeatable" in capsys.readouterr().out


def test_mismatched_keys_fails_fast(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1],
                                   "quantiles_log10": [[0.1]]}})
    _write(b, {"g2|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1],
                                   "quantiles_log10": [[0.1]]}})
    assert mod._compare(a, b) == 1


def test_mismatched_dates_for_the_same_key_fails(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    _write(a, {"g1|target_only": {"dates": ["2020-01-01"], "pred_log10": [0.1],
                                   "quantiles_log10": [[0.1]]}})
    _write(b, {"g1|target_only": {"dates": ["2020-01-02"], "pred_log10": [0.1],
                                   "quantiles_log10": [[0.1]]}})
    assert mod._compare(a, b) == 1
