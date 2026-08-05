"""Direct per-day same-environment repeatability check.

Runs a bounded sample of real gaps through `run_gap_inference` (the shared
call path both benchmark drivers use) and writes point predictions +
quantiles to a JSON file. Intended to be invoked twice, as two independent
process invocations (not two loops inside one process -- that would not
exercise process-level nondeterminism sources like thread-pool scheduling),
so their outputs can be diffed directly:

    python -m experiments.chlorophyll.verify_same_environment_repeatability --out /tmp/run_a.json
    python -m experiments.chlorophyll.verify_same_environment_repeatability --out /tmp/run_b.json
    python -m experiments.chlorophyll.verify_same_environment_repeatability --compare /tmp/run_a.json /tmp/run_b.json

This directly tests per-day determinism, correcting the earlier session's
weaker inference (aggregate MAE equal to 6 decimal places from two
different *drivers* on 449 gaps was treated as proof of per-day
determinism, which it is not -- day-level compensating errors could produce
an identical mean from different per-day predictions).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import tsicl_helpers as th

from . import tsicl_contract as tc
from .run_tsicl_benchmark import load_gap_specs, load_target_series
from .tsicl_covariate_registry import COVARIATE_ARMS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
N_SAMPLE_GAPS = 10


def _run(out_path: Path, target_path: Path, features_path: Path) -> None:
    from coastal_gap_reconstruction.feature_tables import load_feature_table

    model, provenance = th.load_tsicl_strict()
    dates, target_log10 = load_target_series(target_path)
    features_df = load_feature_table(features_path).reindex(pd.to_datetime(dates))

    matched_ids = pd.read_csv(tc.MATCHED_SUPPORT_PATH)["gap_id"].tolist()[:N_SAMPLE_GAPS]
    full_pool = pd.read_csv(tc.FULL_POOL_PATH, parse_dates=["start_date", "end_date"])
    sample_pool = full_pool[full_pool["gap_id"].isin(matched_ids)].reset_index(drop=True)
    specs = load_gap_specs(sample_pool)

    proxy_cols = COVARIATE_ARMS["satellite_proxy"].columns
    proxy_block = features_df[proxy_cols].to_numpy(dtype=np.float32)

    results = {}
    for gap in specs:
        for arm, covar in (("target_only", None), ("satellite_proxy", proxy_block)):
            result = th.run_gap_inference(model, dates, target_log10, gap, context_mode="full_series",
                                           covariate_array=covar, strict=True)
            key = f"{gap.gap_id}|{arm}"
            results[key] = {
                "dates": [str(d) for d in result["dates"]],
                "pred_log10": [float(v) for v in result["pred_log10"]],
                "quantiles_log10": [[float(v) for v in q] for q in result["quantiles_log10"]],
            }

    out_path.write_text(json.dumps({"provenance": provenance, "results": results}, indent=2))
    print(f"wrote {len(results)} gap-arm results to {out_path}")


def _compare(path_a: Path, path_b: Path) -> int:
    a = json.loads(path_a.read_text())
    b = json.loads(path_b.read_text())
    keys_a, keys_b = set(a["results"]), set(b["results"])
    if keys_a != keys_b:
        print(f"FAIL: key sets differ: only in a={keys_a - keys_b}, only in b={keys_b - keys_a}")
        return 1

    max_abs_diff_point = 0.0
    max_abs_diff_quantile = 0.0
    bitwise_identical = True
    for key in sorted(keys_a):
        ra, rb = a["results"][key], b["results"][key]
        if ra["dates"] != rb["dates"]:
            print(f"FAIL: {key} -- dates differ: {ra['dates']} vs {rb['dates']}")
            return 1
        pa, pb = np.array(ra["pred_log10"]), np.array(rb["pred_log10"])
        diff = np.abs(pa - pb)
        max_abs_diff_point = max(max_abs_diff_point, float(diff.max()) if diff.size else 0.0)
        if not np.array_equal(pa, pb):
            bitwise_identical = False
        qa, qb = np.array(ra["quantiles_log10"]), np.array(rb["quantiles_log10"])
        qdiff = np.abs(qa - qb)
        max_abs_diff_quantile = max(max_abs_diff_quantile, float(qdiff.max()) if qdiff.size else 0.0)
        if not np.array_equal(qa, qb):
            bitwise_identical = False

    if bitwise_identical:
        classification = "bitwise_repeatable_in_this_environment"
    elif max_abs_diff_point < 1e-6 and max_abs_diff_quantile < 1e-6:
        classification = "numerically_repeatable_within_1e-6"
    else:
        classification = "not_repeatable"

    print(f"n_gap_arm_keys={len(keys_a)}")
    print(f"max_abs_diff_point_prediction={max_abs_diff_point!r}")
    print(f"max_abs_diff_quantile={max_abs_diff_quantile!r}")
    print(f"bitwise_identical={bitwise_identical}")
    print(f"CLASSIFICATION: {classification}")
    return 0 if classification != "not_repeatable" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--compare", type=Path, nargs=2, default=None)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    args = parser.parse_args(argv)

    if args.compare:
        return _compare(*args.compare)
    if args.out:
        _run(args.out, args.target, args.features)
        return 0
    parser.error("either --out or --compare is required")
    return 2


if __name__ == "__main__":
    sys.exit(main())
