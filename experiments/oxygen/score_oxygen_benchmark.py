"""Structured scoring for a completed `run_oxygen_benchmark.py` output
directory -- classical or TS-ICL-bounded -- against the frozen released
oxygen tables. Mirrors `experiments.chlorophyll.score_tsicl_run`'s
orthogonal RUN_STATUS/VERIFICATION_STATUS contract.

Because the TS-ICL bounded mode intentionally runs a small, length-biased
subset (never the full 406-gap primary support -- see
`run_oxygen_benchmark.py`'s module docstring), this script never compares a
bounded-subset aggregate MAE against the frozen full-support headline number
as if they were on the same support: doing so would misrepresent a
`executable_bounded_validation` result as a `frozen_primary_benchmark`
reproduction. It reports the bounded subset's own descriptive statistics
(labeled as such) and, separately, the classical mode's full-406-gap
comparison against the frozen Model-0/GP numbers, which *is* a like-for-like
comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["score_classical_run", "score_tsicl_bounded_run"]


def score_classical_run(run_dir: Path) -> int:
    day_path = run_dir / "classical_day_level.csv"
    if not day_path.exists():
        print(f"No classical_day_level.csv at {day_path} -- run --mode classical first.")
        return 1
    day = pd.read_csv(day_path).dropna(subset=["true"])
    day["ae"] = (day["pred"] - day["true"]).abs()
    gap_means = day.groupby(["method_id", "gap_id"])["ae"].mean().reset_index()
    mae_gapweighted = gap_means.groupby("method_id")["ae"].mean()

    frozen = {"climatology": 1.612, "persistence": 1.041, "linear_interp": 0.733,
              "gp_matern_time_only_exploratory": 0.738}
    rows = []
    n_checked, n_within_tol = 0, 0
    tol = 0.02  # empirical reporting band, mg/L
    for method_id, generated in mae_gapweighted.items():
        released = frozen.get(method_id)
        cls = "not_applicable"
        if released is not None:
            n_checked += 1
            diff = abs(generated - released)
            cls = "within_empirical_reporting_band" if diff < tol else "mismatched"
            if cls == "within_empirical_reporting_band":
                n_within_tol += 1
        rows.append({"method_id": method_id, "generated_mae_gapweighted": generated,
                      "released_mae_gapweighted": released, "classification": cls})
    report = pd.DataFrame(rows)
    report.to_csv(run_dir / "classical_verification_report.csv", index=False)

    status = "VERIFICATION_REPRODUCED" if n_checked and n_within_tol == n_checked else (
        "VERIFICATION_NOT_APPLICABLE" if n_checked == 0 else "VERIFICATION_PARTIAL"
    )
    (run_dir / "VERIFICATION_STATUS").write_text(f"{status}\n")
    print(report.to_string(index=False))
    print(f"VERIFICATION_STATUS: {status}")
    return 0 if status in ("VERIFICATION_REPRODUCED", "VERIFICATION_NOT_APPLICABLE") else 1


def score_tsicl_bounded_run(run_dir: Path) -> int:
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"No predictions.jsonl at {pred_path} -- run --mode tsicl-bounded first.")
        return 1
    rows = [json.loads(line) for line in open(pred_path) if line.strip()]
    day_rows = []
    for r in rows:
        for i in range(len(r["pred_log10"])):
            day_rows.append({
                "arm": r["arm"], "context_mode": r["context_mode"], "gap_id": r["gap_id"],
                "gap_length": r["gap_length"],
                "absolute_error_mgL": abs(r["pred_log10"][i] - r["true_log10"][i]),
            })
    day = pd.DataFrame(day_rows)
    agg = day.groupby(["arm", "context_mode"]).agg(
        n_gaps=("gap_id", "nunique"), n_days=("absolute_error_mgL", "size"),
        mae_bounded_subset=("absolute_error_mgL", "mean"),
    ).reset_index()
    agg.insert(0, "caveat", "bounded_length_biased_subset_not_comparable_to_frozen_full_406_gap_headline")
    agg.to_csv(run_dir / "tsicl_bounded_descriptive_stats.csv", index=False)

    # Structural checks only -- no headline claim.
    finite_ok = np.isfinite(day["absolute_error_mgL"]).all()
    quantiles_ordered_ok = all(
        all(q[i] <= q[i + 1] + 1e-9 for q in r["quantiles_log10"] for i in range(len(q) - 1))
        for r in rows
    )
    status = "VERIFICATION_NOT_APPLICABLE"  # bounded subset is never scored against the frozen headline
    (run_dir / "VERIFICATION_STATUS").write_text(f"{status}\n")
    print(agg.to_string(index=False))
    print(f"finite_predictions: {finite_ok}, quantiles_ordered: {quantiles_ordered_ok}")
    print(f"VERIFICATION_STATUS: {status} (bounded subset -- code-correctness evidence, not a headline reproduction)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["classical", "tsicl-bounded"], required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "classical":
        return score_classical_run(args.run_dir)
    return score_tsicl_bounded_run(args.run_dir)


if __name__ == "__main__":
    sys.exit(main())
