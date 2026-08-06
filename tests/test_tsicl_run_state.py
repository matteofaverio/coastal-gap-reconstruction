"""Tests for the failure-safe TS-ICL run-state accounting
(`experiments.chlorophyll.tsicl_run_state`).

These reproduce a real historical bug: the earlier per-driver `done_keys.json`
design added a call's key to
the done set even when the call failed, so a resume silently skipped a
permanently-failed call and a run could report RUN_COMPLETE despite a real,
never-retried failure.
"""

from __future__ import annotations

import json

import pytest

from experiments.chlorophyll import tsicl_run_state as rs


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _pred_row(key, gap_id="g1", pred=(0.1, 0.2)):
    return {
        "key": key, "gap_id": gap_id, "context_mode": "full_series", "arm": "target_only",
        "gap_length": len(pred), "date": [f"2020-01-0{i+1}" for i in range(len(pred))],
        "pred_log10": list(pred), "true_log10": list(pred),
    }


def test_first_invocation_records_a_failed_key_and_second_resumes_and_retries(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only", "g2|full_series|target_only"]

    # First invocation: g1 succeeds, g2 fails.
    _write_jsonl(pred_path, [_pred_row("g1|full_series|target_only", gap_id="g1")])
    _write_jsonl(fail_path, [{"key": "g2|full_series|target_only", "gap_id": "g2",
                              "error_type": "TSICLOutputError", "error": "boom"}])

    state1 = rs.load_run_state(pred_path, fail_path)
    assert state1.successful_keys == {"g1|full_series|target_only"}
    assert "g2|full_series|target_only" in state1.failed_keys
    outstanding1 = state1.outstanding(expected_keys, max_attempts=None)
    assert outstanding1 == ["g2|full_series|target_only"], (
        "a resume must retry the failed key, not skip it forever"
    )
    status1, _ = rs.classify_run_status(state1, expected_keys)
    assert status1 == "RUN_PARTIAL"

    # Second invocation resumes and retries g2, which now succeeds.
    with open(pred_path, "a") as f:
        f.write(json.dumps(_pred_row("g2|full_series|target_only", gap_id="g2")) + "\n")

    state2 = rs.load_run_state(pred_path, fail_path)
    assert state2.successful_keys == set(expected_keys)
    # The historical failure attempt remains counted, but no longer blocks completion
    # because g2 now has a valid successful record.
    assert "g2|full_series|target_only" not in state2.failed_keys
    status2, detail2 = rs.classify_run_status(state2, expected_keys)
    assert status2 == "RUN_COMPLETE"
    assert detail2["n_unresolved_failed"] == 0


def test_run_complete_is_impossible_while_a_key_remains_failed(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only"]
    _write_jsonl(fail_path, [{"key": "g1|full_series|target_only", "gap_id": "g1",
                              "error_type": "TSICLOutputError", "error": "still broken"}])
    state = rs.load_run_state(pred_path, fail_path)
    status, _ = rs.classify_run_status(state, expected_keys)
    assert status != "RUN_COMPLETE"


def test_historical_failures_cannot_disappear_because_n_fail_resets(tmp_path):
    """A key that failed 3 times across 3 separate process invocations must
    still show attempts=3 on the 4th invocation's state -- the failure
    history lives in failures.jsonl, not in a per-process counter that
    resets to zero on every new invocation."""
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    _write_jsonl(fail_path, [
        {"key": "g1|full_series|target_only", "gap_id": "g1", "error_type": "E", "error": "1"},
        {"key": "g1|full_series|target_only", "gap_id": "g1", "error_type": "E", "error": "2"},
        {"key": "g1|full_series|target_only", "gap_id": "g1", "error_type": "E", "error": "3"},
    ])
    state = rs.load_run_state(pred_path, fail_path)
    assert state.failed_keys["g1|full_series|target_only"]["attempts"] == 3
    assert state.failed_keys["g1|full_series|target_only"]["last_error"] == "3"


def test_max_attempts_stops_retrying_but_the_key_stays_failed(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only"]
    _write_jsonl(fail_path, [
        {"key": "g1|full_series|target_only", "gap_id": "g1", "error_type": "E", "error": "1"},
        {"key": "g1|full_series|target_only", "gap_id": "g1", "error_type": "E", "error": "2"},
    ])
    state = rs.load_run_state(pred_path, fail_path)
    outstanding = state.outstanding(expected_keys, max_attempts=2)
    assert outstanding == [], "attempt cap reached -- must not be retried again this run"
    status, detail = rs.classify_run_status(state, expected_keys)
    assert status != "RUN_COMPLETE", "a capped-out failed key must still block RUN_COMPLETE"
    assert detail["n_unresolved_failed"] == 1


def test_retrying_does_not_produce_a_duplicate_successful_record(tmp_path):
    """A key already present in successful_keys is never re-attempted, so a
    driver loop over `outstanding()` structurally cannot append a second
    successful record for the same key."""
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only"]
    _write_jsonl(pred_path, [_pred_row("g1|full_series|target_only")])
    state = rs.load_run_state(pred_path, fail_path)
    assert state.outstanding(expected_keys, max_attempts=None) == []


def test_duplicate_successful_records_are_detected_and_block_run_complete(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only"]
    _write_jsonl(pred_path, [
        _pred_row("g1|full_series|target_only"), _pred_row("g1|full_series|target_only"),
    ])
    state = rs.load_run_state(pred_path, fail_path)
    assert state.duplicate_successful_keys == ["g1|full_series|target_only"]
    status, detail = rs.classify_run_status(state, expected_keys)
    assert status != "RUN_COMPLETE"
    assert detail["n_duplicate_successful_keys"] == 1


def test_malformed_prediction_record_is_not_counted_as_successful(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only"]
    # gap_length says 3 but only 2 predictions provided -- malformed.
    row = _pred_row("g1|full_series|target_only", pred=(0.1, 0.2))
    row["gap_length"] = 3
    _write_jsonl(pred_path, [row])
    state = rs.load_run_state(pred_path, fail_path)
    assert "g1|full_series|target_only" not in state.successful_keys
    assert "g1|full_series|target_only" in state.malformed_successful_records
    outstanding = state.outstanding(expected_keys, max_attempts=None)
    assert outstanding == ["g1|full_series|target_only"], "a malformed record must be retried, not trusted"


def test_run_failed_when_no_valid_successful_call_exists(tmp_path):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    expected_keys = ["g1|full_series|target_only"]
    _write_jsonl(fail_path, [{"key": "g1|full_series|target_only", "gap_id": "g1",
                              "error_type": "TSICLDependencyError", "error": "fatal"}])
    state = rs.load_run_state(pred_path, fail_path)
    status, _ = rs.classify_run_status(state, expected_keys)
    assert status == "RUN_FAILED"


def test_state_is_recoverable_purely_from_source_files_with_no_compact_cache(tmp_path):
    """There is no separate done_keys.json-style cache in this design --
    state must be fully recoverable from predictions.jsonl/failures.jsonl
    alone, so a killed-mid-write process cannot corrupt resume behavior
    beyond whatever partial line it left in an append-only file."""
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    _write_jsonl(pred_path, [_pred_row("g1|full_series|target_only")])
    assert not (tmp_path / "done_keys.json").exists()
    state = rs.load_run_state(pred_path, fail_path)
    assert state.successful_keys == {"g1|full_series|target_only"}


@pytest.mark.parametrize("missing", ["predictions", "failures", "both"])
def test_load_run_state_handles_missing_files_gracefully(tmp_path, missing):
    pred_path = tmp_path / "predictions.jsonl"
    fail_path = tmp_path / "failures.jsonl"
    if missing in ("predictions",):
        _write_jsonl(fail_path, [])
    elif missing in ("failures",):
        _write_jsonl(pred_path, [])
    state = rs.load_run_state(pred_path, fail_path)
    assert state.successful_keys == set()
    assert state.failed_keys == {}
