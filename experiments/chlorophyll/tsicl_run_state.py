"""Restart-safe run-state accounting for the TS-ICL benchmark drivers.

Both `run_tsicl_benchmark.py` and `run_tsicl_covariate_analysis.py` delegate
all resume/failure/completion bookkeeping to this module instead of each
maintaining its own compact `done_keys.json` cache.

**The bug this module fixes.** The earlier per-driver design added a call's
key to `done_keys` unconditionally, whether the call succeeded or raised a
`TSICLError`. A subsequent resume then treated that key as already handled
and skipped it forever, and `n_fail` was a this-process-only counter, so a
run could report `RUN_COMPLETE` after a resume even though an earlier
process had recorded a real, never-retried failure for one of its keys.

**The fix is structural, not a patched flag.** There is no compact on-disk
cache to go stale: `successful_keys` and `failed_keys` are always
recomputed directly from `predictions.jsonl`/`failures.jsonl` -- the
append-only source-of-truth files a driver writes as it goes -- every time a
driver starts or reports status. This makes the state trivially recoverable
even if a run is killed mid-write (requirement: state must be recoverable
from the prediction/failure files alone), and makes it structurally
impossible for a failed call to be counted as successful: a key only ever
enters `successful_keys` by having an actual valid record in
`predictions.jsonl`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REQUIRED_PREDICTION_FIELDS = ("key", "gap_id", "pred_log10", "true_log10")


@dataclass
class RunState:
    successful_keys: set[str]
    failed_keys: dict[str, dict]  # key -> {"attempts": int, "last_error": str, "last_error_type": str}
    duplicate_successful_keys: list[str]
    malformed_successful_records: list[str]  # keys with a record present but failing schema validation

    def outstanding(self, expected_keys: list[str], max_attempts: int | None) -> list[str]:
        """Keys this run still needs to attempt: not yet successfully
        recorded, and (if `max_attempts` is set) still under the attempt
        cap. A key with unlimited retries (`max_attempts=None`, the default)
        is retried every run until it succeeds -- normal resume always
        retries failed calls; it never silently drops them."""
        out = []
        for key in expected_keys:
            if key in self.successful_keys:
                continue
            attempts = self.failed_keys.get(key, {}).get("attempts", 0)
            if max_attempts is not None and attempts >= max_attempts:
                continue
            out.append(key)
        return out


def _validate_prediction_row(row: dict) -> bool:
    """Minimal schema check for a `predictions.jsonl` row: required fields
    present, point prediction finite-length list matching its own recorded
    `gap_length`, quantiles (if present) non-decreasing across levels."""
    if not all(f in row for f in REQUIRED_PREDICTION_FIELDS):
        return False
    pred = row.get("pred_log10")
    if not isinstance(pred, list) or len(pred) == 0:
        return False
    if any((v is None) for v in pred):
        return False
    gap_length = row.get("gap_length")
    if gap_length is not None and len(pred) != int(gap_length):
        return False
    dates = row.get("date")
    if dates is not None and len(dates) != len(pred):
        return False
    quantiles = row.get("quantiles_log10")
    levels = row.get("quantile_levels")
    if quantiles and levels:
        for day_q in quantiles:
            if any(day_q[i] > day_q[i + 1] + 1e-9 for i in range(len(day_q) - 1)):
                return False
    return True


def load_run_state(pred_path: Path, fail_path: Path) -> RunState:
    """Rebuild `RunState` entirely from the append-only `predictions.jsonl`/
    `failures.jsonl` files -- never trusts a separate compact cache."""
    successful: dict[str, dict] = {}
    duplicates: list[str] = []
    malformed: list[str] = []
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row["key"]
                if key in successful:
                    duplicates.append(key)
                if not _validate_prediction_row(row):
                    malformed.append(key)
                    continue
                successful[key] = row  # last valid record wins if a key recurs

    failed: dict[str, dict] = {}
    if fail_path.exists():
        with open(fail_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row["key"]
                entry = failed.setdefault(key, {"attempts": 0, "last_error": None, "last_error_type": None})
                entry["attempts"] += 1
                entry["last_error"] = row.get("error")
                entry["last_error_type"] = row.get("error_type")

    successful_keys = set(successful)
    # A key that has since produced a valid successful record is no longer
    # "failed" -- its historical failure attempts stay in failures.jsonl for
    # audit, but do not block RUN_COMPLETE once a real success exists.
    failed_keys = {k: v for k, v in failed.items() if k not in successful_keys}
    return RunState(
        successful_keys=successful_keys,
        failed_keys=failed_keys,
        duplicate_successful_keys=sorted(set(duplicates)),
        malformed_successful_records=sorted(set(malformed)),
    )


def classify_run_status(state: RunState, expected_keys: list[str]) -> tuple[str, dict]:
    """Classify the run given its current state and the configuration's
    full expected-key set.

    - `RUN_COMPLETE`: every expected key has a valid successful record, zero
      unresolved failed keys remain, and there are no duplicate/malformed
      successful records.
    - `RUN_PARTIAL`: some expected calls are missing, still failed, or a
      data-quality issue (duplicate/malformed record) was found.
    - `RUN_FAILED`: no expected key has a valid successful record at all.
    """
    expected_set = set(expected_keys)
    successful_in_scope = state.successful_keys & expected_set
    unresolved_failed = set(state.failed_keys) & expected_set
    missing_untried = expected_set - successful_in_scope - unresolved_failed
    dup_in_scope = [k for k in state.duplicate_successful_keys if k in expected_set]
    malformed_in_scope = [k for k in state.malformed_successful_records if k in expected_set]

    detail = {
        "n_expected": len(expected_set),
        "n_successful": len(successful_in_scope),
        "n_unresolved_failed": len(unresolved_failed),
        "n_missing_untried": len(missing_untried),
        "n_duplicate_successful_keys": len(dup_in_scope),
        "n_malformed_successful_records": len(malformed_in_scope),
    }

    if len(successful_in_scope) == 0:
        return "RUN_FAILED", detail
    if not unresolved_failed and not missing_untried and not dup_in_scope and not malformed_in_scope:
        return "RUN_COMPLETE", detail
    return "RUN_PARTIAL", detail
