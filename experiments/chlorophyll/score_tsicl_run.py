"""Structured scoring and verification for a completed TS-ICL benchmark run
(`run_tsicl_benchmark.py`'s output directory).

Complements `run_classical_benchmark.py`'s scoring: separate `RUN_STATUS`
(execution completeness, written by the driver itself) from
`VERIFICATION_STATUS` (do the results match the frozen release, written by
this script), using the same orthogonal-status contract and tri-state
`VERIFICATION_REPRODUCED`/`VERIFICATION_PARTIAL`/`VERIFICATION_MISMATCH`/
`VERIFICATION_NOT_APPLICABLE`/`VERIFICATION_NOT_REQUESTED` vocabulary.

Writes, into the same run's output directory:

- `scored_day_level.csv`: one row per (method_id, gap_id, date).
- `aggregate_metrics.csv`, `by_length_metrics.csv`, `event_background_metrics.csv`.
- `paired_bootstrap.csv`: gap-clustered paired bootstrap of every TS-ICL
  method in the run against a freshly generated canonical-interpolation
  comparator on the exact same gap pool (day-level pairing enforced, see
  `paired_statistics.bootstrap_compare`).
- `verification_report.csv` / `verification_summary.json` / `VERIFICATION_STATUS`.

Usage:
    python -m experiments.chlorophyll.score_tsicl_run --run-dir build/chlorophyll/tsicl_benchmark_full681_verified
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import paired_statistics as ps

from . import interpolation_baselines as interp
from . import tsicl_contract as tc

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

AGGREGATE_METRICS = ["n_gaps", "n_rows", "mae_day_weighted", "mae_gap_weighted",
                      "rmse", "bias_mean", "median_abs_error", "p90_abs_error"]
BY_LENGTH_METRICS = ["n_gaps", "n_rows", "mae_day_weighted", "mae_gap_weighted"]
EXACT_COUNT_FIELDS = {"n_gaps", "n_rows"}

# Empirical reporting band, not an independently pre-specified reproduction
# threshold -- matches the order of magnitude of the classical benchmark's
# own evidence-based bands (docs/methodology/validation_protocol.md) and
# this project's own prior live-run diffs (0.0025-0.0031, different
# environment from the original run; see docs/methodology/tsicl_usage.md).
TSICL_METRIC_TOLERANCE = 0.01
TSICL_DELTA_CI_TOLERANCE = 0.01

# TS-ICL public-name mapping used in the released comparison tables
# (results_public/chlorophyll/chlorophyll_benchmark_summary.csv).
ARM_PUBLIC_NAME = {
    "target_only": "TS-ICL (target-only, no covariates)",
    "satellite_proxy": "TS-ICL (satellite chlorophyll proxy covariate)",
}
INTERPOLATION_PUBLIC_NAME = "Linear interpolation baseline"


def load_day_level(pred_path: Path) -> pd.DataFrame:
    rows = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            arm = rec["arm"]
            mode = rec.get("context_mode", "full_series")
            method_id = f"tsicl_{arm}__{mode}"
            for i, date in enumerate(rec["date"]):
                pred = rec["pred_log10"][i]
                true = rec["true_log10"][i]
                rows.append({
                    "method_id": method_id, "arm": arm, "context_mode": mode,
                    "gap_id": rec["gap_id"], "date": pd.Timestamp(date), "gap_length": rec["gap_length"],
                    "pred_log10": pred, "true_log10": true,
                    "absolute_error_log10": abs(pred - true) if (pred == pred and true == true) else np.nan,
                })
    return pd.DataFrame(rows)


def _classify(metric: str, released, generated, tol: float) -> tuple[str, float | None]:
    if released is None or (isinstance(released, float) and released != released):
        return "unavailable_in_frozen_artifact", None
    if generated is None or (isinstance(generated, float) and generated != generated):
        return "mismatched", None
    diff = abs(float(released) - float(generated))
    if metric in EXACT_COUNT_FIELDS:
        return ("exact" if diff == 0 else "mismatched"), diff
    if diff == 0:
        return "exact", diff
    if diff < 1e-9:
        return "numerically_equal", diff
    if diff < tol:
        return "within_empirical_reporting_band", diff
    return "mismatched", diff


def compute_aggregate_metrics(day_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_id, g in day_level.groupby("method_id"):
        g_valid = g.dropna(subset=["absolute_error_log10"])
        gap_means = g_valid.groupby("gap_id")["absolute_error_log10"].mean()
        errs = g_valid["absolute_error_log10"].to_numpy()
        rows.append({
            "method_id": method_id, "n_gaps": g["gap_id"].nunique(), "n_rows": len(g),
            "mae_day_weighted": float(errs.mean()) if len(errs) else float("nan"),
            "mae_gap_weighted": float(gap_means.mean()) if len(gap_means) else float("nan"),
            "rmse": float(np.sqrt((errs**2).mean())) if len(errs) else float("nan"),
            "bias_mean": float((g_valid["pred_log10"] - g_valid["true_log10"]).mean()) if len(g_valid) else float("nan"),
            "median_abs_error": float(np.median(errs)) if len(errs) else float("nan"),
            "p90_abs_error": float(np.quantile(errs, 0.9)) if len(errs) else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_by_length_metrics(day_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method_id, length), g in day_level.groupby(["method_id", "gap_length"]):
        g_valid = g.dropna(subset=["absolute_error_log10"])
        gap_means = g_valid.groupby("gap_id")["absolute_error_log10"].mean()
        errs = g_valid["absolute_error_log10"].to_numpy()
        rows.append({
            "method_id": method_id, "gap_length": int(length), "n_gaps": g["gap_id"].nunique(),
            "n_rows": len(g),
            "mae_day_weighted": float(errs.mean()) if len(errs) else float("nan"),
            "mae_gap_weighted": float(gap_means.mean()) if len(gap_means) else float("nan"),
        })
    return pd.DataFrame(rows)


def compute_event_background_metrics(day_level: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    event_flag = pool.set_index("gap_id")["is_high_chl_event"]
    day_level = day_level.copy()
    day_level["is_high_chl_event"] = day_level["gap_id"].map(event_flag)
    rows = []
    for (method_id, is_event), g in day_level.groupby(["method_id", "is_high_chl_event"]):
        g_valid = g.dropna(subset=["absolute_error_log10"])
        errs = g_valid["absolute_error_log10"].to_numpy()
        rows.append({
            "method_id": method_id, "is_high_chl_event": bool(is_event), "n_rows": len(g),
            "mae_day_weighted": float(errs.mean()) if len(errs) else float("nan"),
        })
    return pd.DataFrame(rows)


def _interpolation_day_level(pool: pd.DataFrame, target_path: Path) -> pd.DataFrame:
    target_df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()
    preds = interp.standalone_log10_interpolation_predictions(pool, target_df)
    preds = preds.dropna(subset=["pred_log10", "true"])
    preds["true_log10"] = np.log10(preds["true"].clip(lower=tc.LOG10_FLOOR))
    preds["absolute_error_log10"] = (preds["pred_log10"] - preds["true_log10"]).abs()
    preds["method_id"] = "canonical_interpolation"
    return preds[["method_id", "gap_id", "date", "gap_length", "pred_log10", "true_log10", "absolute_error_log10"]]


def run_scoring(run_dir: Path, target_path: Path) -> int:
    pred_path = run_dir / "predictions.jsonl"
    if not pred_path.exists():
        print(f"No predictions.jsonl at {pred_path} -- run the benchmark first.")
        return 1

    metadata = json.loads((run_dir / "run_metadata.json").read_text())
    support = metadata["support"]
    day_level = load_day_level(pred_path)
    day_level.to_csv(run_dir / "scored_day_level.csv", index=False)

    aggregate = compute_aggregate_metrics(day_level)
    aggregate.to_csv(run_dir / "aggregate_metrics.csv", index=False)
    by_length = compute_by_length_metrics(day_level)
    by_length.to_csv(run_dir / "by_length_metrics.csv", index=False)

    full_pool = pd.read_csv(tc.FULL_POOL_PATH, parse_dates=["start_date", "end_date"])
    pool = full_pool[full_pool["gap_id"].isin(day_level["gap_id"].unique())].reset_index(drop=True)
    event_bg = compute_event_background_metrics(day_level, pool)
    event_bg.to_csv(run_dir / "event_background_metrics.csv", index=False)

    # ── Aggregate-metric verification against the frozen matched-449 table ──
    verification_rows: list[dict] = []
    n_mismatched = 0
    n_checked = 0
    if support == "matched_449":
        released = pd.read_csv(tc.MATCHED_SUPPORT_METRICS_PATH)
        released_matched = released[released["support"] == "matched_449"].set_index("method_id")
        for arm, released_method_id in (("target_only", "tsicl_target_only"), ("satellite_proxy", "tsicl_satellite_proxy")):
            method_id = f"tsicl_{arm}__full_series"
            if method_id not in aggregate["method_id"].values or released_method_id not in released_matched.index:
                continue
            gen_row = aggregate[aggregate["method_id"] == method_id].iloc[0]
            rel_row = released_matched.loc[released_method_id]
            for metric in AGGREGATE_METRICS:
                gen_val = gen_row.get(metric)
                rel_val = rel_row.get(metric)
                cls, diff = _classify(metric, rel_val, gen_val, TSICL_METRIC_TOLERANCE)
                n_checked += 1
                if cls == "mismatched":
                    n_mismatched += 1
                verification_rows.append({
                    "scope": "aggregate", "method_id": method_id, "metric": metric,
                    "released": rel_val, "generated": gen_val, "abs_diff": diff, "classification": cls,
                })

    # ── Paired bootstrap vs a freshly generated canonical-interpolation
    # comparator on the exact same gap pool, compared to the released
    # full-681 paired-delta table where available ──
    interp_day_level = _interpolation_day_level(pool, target_path)
    combined = pd.concat([day_level, interp_day_level], ignore_index=True)
    released_summary = pd.read_csv(tc.BENCHMARK_SUMMARY_PATH)

    paired_rows = []
    for method_id in sorted(day_level["method_id"].unique()):
        arm = method_id.split("__")[0].removeprefix("tsicl_")
        try:
            result = ps.bootstrap_compare(
                method_id, "canonical_interpolation", combined, pairing="intersection",
                error_col="absolute_error_log10",
            )
        except Exception as exc:  # noqa: BLE001 -- record the failure, do not abort the whole run
            paired_rows.append({"method_id": method_id, "error": str(exc)})
            continue
        if result is None:
            continue
        flat = result.to_flat_dict()
        flat["method_id"] = method_id

        if arm in ARM_PUBLIC_NAME and support == "full_681":
            released_row = released_summary[
                (released_summary["method_public_name"] == ARM_PUBLIC_NAME[arm])
                & (released_summary["compared_against_public_name"] == INTERPOLATION_PUBLIC_NAME)
                & (released_summary["stratum"] == "all_gaps")
            ]
            if not released_row.empty:
                rel_delta = float(released_row.iloc[0]["delta"])
                gen_delta = flat.get("day_weighted_mae_delta")
                cls, diff = _classify("delta", rel_delta, gen_delta, TSICL_DELTA_CI_TOLERANCE)
                flat["released_delta"] = rel_delta
                flat["delta_classification"] = cls
                flat["delta_abs_diff"] = diff
        paired_rows.append(flat)
    pd.DataFrame(paired_rows).to_csv(run_dir / "paired_bootstrap.csv", index=False)

    report = pd.DataFrame(verification_rows)
    report.to_csv(run_dir / "verification_report.csv", index=False)

    if n_checked == 0:
        verification_status = "VERIFICATION_NOT_APPLICABLE"
    elif n_mismatched == 0:
        verification_status = "VERIFICATION_REPRODUCED"
    elif n_mismatched == n_checked:
        verification_status = "VERIFICATION_MISMATCH"
    else:
        verification_status = "VERIFICATION_PARTIAL"

    (run_dir / "verification_summary.json").write_text(json.dumps({
        "support": support, "n_checked_metrics": n_checked, "n_mismatched_metrics": n_mismatched,
        "verification_status": verification_status,
    }, indent=2))
    (run_dir / "VERIFICATION_STATUS").write_text(f"{verification_status}\n")
    print(f"VERIFICATION_STATUS: {verification_status} ({n_mismatched}/{n_checked} mismatched)")
    return 0 if verification_status in ("VERIFICATION_REPRODUCED", "VERIFICATION_NOT_APPLICABLE") else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path,
                         default=REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv")
    args = parser.parse_args(argv)
    return run_scoring(args.run_dir, args.target)


if __name__ == "__main__":
    sys.exit(main())
