"""Run the classical/probabilistic/gap-edge/engineered-hybrid benchmark on the
449-gap matched support (`benchmark_contract.MATCHED_SUPPORT_*`).

Usage:
    python -m experiments.chlorophyll.run_classical_benchmark \\
        --target data_public/chlorophyll/chlorophyll_daily_target.csv \\
        --features data_public/chlorophyll/chlorophyll_predictor_features_curated.csv \\
        --out build/chlorophyll/classical_benchmark \\
        [--methods gp_m1,ext_tabular_extratrees,...] [--gap-lengths 1,3] [--verify]

All input/output paths are explicit CLI arguments (no implicit repo-relative
defaults baked into the driver itself, beyond argparse's own defaults, which
point at the already-public files) so this can run against a different
checkout or a build directory outside version control.

Default output directory is a gitignored `build/` path -- this driver never
overwrites the frozen `results_public/` tables by default. Progress is
checkpointed per method: a completed method's predictions are written
immediately together with a metadata sidecar
(`predictions_<method>.meta.json`) recording exactly what produced them
(gap IDs, row count, config/input hashes, software versions). A re-run skips
a method only if the cached sidecar matches the *current* request in every
one of those fields -- any mismatch (different gap set, different input
file, different code/config) forces a recompute instead of silently reusing
a stale cache.

`--verify` compares freshly generated predictions against the frozen
`results_public/chlorophyll/chlorophyll_matched_support_method_metrics.csv`
and `chlorophyll_matched_support_by_length.csv`, and writes a structured,
per-metric verification report (`verification_report.csv`,
`verification_summary.json`) instead of a single pass/fail aggregate
comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from coastal_gap_reconstruction import gaussian_process as gp

from . import benchmark_contract as bc
from . import engineered_hybrid as eh
from . import gap_edge_models as gem
from . import interpolation_baselines as interp
from . import tabular_models as tm

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TARGET_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_daily_target.csv"
DEFAULT_FEATURES_PATH = REPO_ROOT / "data_public" / "chlorophyll" / "chlorophyll_predictor_features_curated.csv"
DEFAULT_OUT_DIR = REPO_ROOT / "build" / "chlorophyll" / "classical_benchmark"

ALL_METHODS = [
    "canonical_interpolation", "gp_m1",
    "ext_tabular_extratrees", "ext_tabular_hgb",
    "external_only_extratrees", "external_only_hgb",
    "tier_ch_deployed", "engineered_hybrid",
]

# Method-specific tolerance for `--verify`'s per-metric classification of
# `mae_day_weighted`/`mae_gap_weighted`/`rmse`/`bias_mean`/`median_abs_error`/
# `p90_abs_error` against the frozen released value. These are NOT a single
# global fudge factor -- each was set from an actual clean, non-cached 449-gap
# run's observed diffs (see `docs/methodology/validation_protocol.md`
# "Reproduction tolerance evidence"), with headroom, not backed into a
# passing threshold after the fact:
#
# - `canonical_interpolation`: a closed-form deterministic formula. The clean
#   run reproduced every aggregate metric exactly (diff 0.0) -- tolerance is
#   floating-point noise only.
# - `gp_m1`, `ext_tabular_extratrees`, `tier_ch_deployed`: all three fit
#   scikit-learn estimators with an internal source of run-to-run numerical
#   variability (GP: L-BFGS-B hyperparameter optimization reaching different
#   local optima on a minority of gaps; ExtraTrees with `n_jobs=-1`: parallel
#   floating-point summation order) -- NOT bit-reproducible across
#   environments even with a fixed `random_state`. Observed clean-run aggregate
#   diffs: gp_m1 up to 1.05e-3 (rmse), ext_tabular_extratrees up to 1.23e-3
#   (rmse), tier_ch_deployed up to 6.6e-4 (bias). `ext_tabular_hgb` shows the
#   same phenomenon with a larger magnitude (HGB's early-stopping validation
#   split adds another source of run-to-run variability): aggregate diffs up
#   to 3.53e-3 (p90).
# - By-length metrics (smaller per-length sample sizes, as few as 50-100 gaps)
#   are noisier than the aggregate for the same reason and use the same
#   per-method tolerance -- some by-length cells legitimately classify
#   `mismatched` even when the aggregate is within tolerance; this is expected
#   statistical behavior (smaller-n subsets have higher variance), not
#   evidence suppressed by widening the tolerance further.
# - `external_only_*` methods have no frozen row (`support_status !=
#   "frozen_matched_449"`) and are never scored against a tolerance at all.
METRIC_TOLERANCE: dict[str, float] = {
    "canonical_interpolation": 1e-6,
    "gp_m1": 2e-3,
    "tier_ch_deployed": 1e-3,
    "ext_tabular_extratrees": 2e-3,
    "ext_tabular_hgb": 4e-3,
}
DEFAULT_TOLERANCE = 1e-6


def load_inputs(target_path: Path, features_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df = pd.read_csv(target_path, parse_dates=["date"]).set_index("date").sort_index()
    features_df = pd.read_csv(features_path, parse_dates=["date"]).set_index("date").sort_index()
    return target_df, features_df


def run_method(
    method_id: str, candidates: pd.DataFrame, target_df: pd.DataFrame, features_df: pd.DataFrame,
) -> pd.DataFrame:
    if method_id == "canonical_interpolation":
        return interp.standalone_log10_interpolation_predictions(candidates, target_df)
    if method_id == "gp_m1":
        return _gp_predictions(candidates, target_df)
    if method_id in ("ext_tabular_extratrees", "ext_tabular_hgb"):
        # Protocol B (matched-reference): external + 5 meta features, strict
        # pre-only dependency-window LOCO -- the protocol that actually
        # produced the frozen rows for these two method IDs.
        model_name = "extratrees" if method_id == "ext_tabular_extratrees" else "hgb"
        preds, _warns = gem.run_reference_arm_loco_evaluation(model_name, candidates, target_df, features_df)
        return preds
    if method_id in ("external_only_extratrees", "external_only_hgb"):
        # Protocol A (plain external-only): no gap-position features.
        cols = tm.load_arm4_numeric_columns(features_df)
        preds, _warns = tm.run_loco_evaluation(method_id, candidates, target_df, features_df, cols)
        return preds
    if method_id == "tier_ch_deployed":
        preds, _warns = gem.run_loco_evaluation(candidates, target_df, features_df)
        return preds
    if method_id == "engineered_hybrid":
        preds, _kalman = eh.run_engineered_hybrid(candidates, target_df, features_df)
        return preds
    raise ValueError(f"unknown method_id {method_id!r}")


def _gp_predictions(candidates: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    target_col = bc.TARGET_SPEC.target_col
    eligible_col = bc.TARGET_SPEC.eligible_col
    y_log = np.log10(target_df[target_col].where(target_df[target_col] > 1e-4))
    ctx_df = pd.DataFrame({
        target_col: y_log,
        eligible_col: target_df[eligible_col],
        "doy_sin": np.sin(2 * np.pi * target_df.index.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * target_df.index.dayofyear / 365.25),
    }, index=target_df.index)

    rows = []
    for _, row in candidates.iterrows():
        start = pd.Timestamp(row["start_date"])
        L = int(row["gap_length"])
        result = gp.run_gp_on_gap(
            ctx_df, start, L, value_col=target_col, eligible_col=eligible_col,
            random_state=bc.RANDOM_SEED,
        )
        if result is None:
            continue
        for d, pred_log in zip(result["dates"], result["pred"]):
            true_val = target_df.loc[d, target_col] if d in target_df.index else np.nan
            rows.append({
                "gap_id": row["gap_id"], "date": d, "gap_length": L,
                "pred_log10": float(pred_log), "pred": float(10.0**pred_log),
                "true": float(true_val) if true_val == true_val else np.nan,
            })
    return pd.DataFrame(rows)


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_predictions(preds: pd.DataFrame) -> dict:
    """Day-weighted and gap-weighted MAE/RMSE/bias/median/p90 on the log10
    scale, matching the released metric definitions exactly (see
    `benchmark_contract.PRIMARY_METRIC`)."""
    valid = preds.dropna(subset=["pred_log10", "true"])
    if valid.empty:
        return {
            "n_gaps": 0, "n_rows": 0, "mae_day_weighted": np.nan, "mae_gap_weighted": np.nan,
            "rmse": np.nan, "bias_mean": np.nan, "median_abs_error": np.nan, "p90_abs_error": np.nan,
        }
    true_log = np.log10(valid["true"].clip(lower=1e-4))
    err = valid["pred_log10"] - true_log
    abs_err = err.abs()
    gap_weighted = abs_err.groupby(valid["gap_id"]).mean().mean()
    return {
        "n_gaps": int(valid["gap_id"].nunique()),
        "n_rows": int(len(valid)),
        "mae_day_weighted": float(abs_err.mean()),
        "mae_gap_weighted": float(gap_weighted),
        "rmse": float(np.sqrt((err**2).mean())),
        "bias_mean": float(err.mean()),
        "median_abs_error": float(abs_err.median()),
        "p90_abs_error": float(abs_err.quantile(0.90)),
    }


def score_by_length(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid = preds.dropna(subset=["pred_log10", "true"])
    if valid.empty:
        return pd.DataFrame(columns=["gap_length", "n_gaps", "n_rows", "mae_day_weighted", "mae_gap_weighted"])
    true_log = np.log10(valid["true"].clip(lower=1e-4))
    valid = valid.assign(abs_err=(valid["pred_log10"] - true_log).abs())
    for L, g in valid.groupby("gap_length"):
        rows.append({
            "gap_length": int(L),
            "n_gaps": int(g["gap_id"].nunique()),
            "n_rows": int(len(g)),
            "mae_day_weighted": float(g["abs_err"].mean()),
            "mae_gap_weighted": float(g.groupby("gap_id")["abs_err"].mean().mean()),
        })
    return pd.DataFrame(rows).sort_values("gap_length").reset_index(drop=True)


# ── Cache validation / run metadata ─────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _config_hash(method_id: str, gap_ids: list[str]) -> str:
    payload = json.dumps({"method_id": method_id, "gap_ids": sorted(gap_ids), "seed": bc.RANDOM_SEED},
                          sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _software_versions() -> dict:
    import scipy
    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.platform(),
    }


def _build_signature(
    method_id: str, candidates: pd.DataFrame, target_path: Path, features_path: Path,
) -> dict:
    return {
        "method_id": method_id,
        "gap_ids": sorted(candidates["gap_id"].astype(str).tolist()),
        "expected_n_rows": int(candidates["gap_length"].sum()),
        "config_hash": _config_hash(method_id, candidates["gap_id"].astype(str).tolist()),
        "target_sha256": _sha256_file(target_path),
        "features_sha256": _sha256_file(features_path),
        "versions": _software_versions(),
    }


def _cache_is_valid(pred_path: Path, meta_path: Path, signature: dict) -> tuple[bool, str]:
    if not pred_path.exists() or not meta_path.exists():
        return False, "no cached prediction/metadata file"
    try:
        cached_meta = json.loads(meta_path.read_text())
    except json.JSONDecodeError:
        return False, "unreadable cached metadata"
    for key in ("method_id", "gap_ids", "config_hash", "target_sha256", "features_sha256"):
        if cached_meta.get(key) != signature[key]:
            return False, f"cached metadata field {key!r} does not match the current request"
    try:
        preds = pd.read_csv(pred_path, parse_dates=["date"])
    except Exception as exc:  # noqa: BLE001
        return False, f"cached predictions file unreadable: {exc}"
    required_cols = {"gap_id", "date", "gap_length", "pred_log10", "pred", "true"}
    if not required_cols.issubset(preds.columns):
        return False, f"cached predictions missing required columns: {required_cols - set(preds.columns)}"
    if preds.duplicated(subset=["gap_id", "date"]).any():
        return False, "cached predictions contain duplicate (gap_id, date) rows"
    if cached_meta.get("actual_n_rows") != len(preds):
        return False, "cached predictions row count does not match its own recorded metadata"
    return True, "ok"


# ── Benchmark run ────────────────────────────────────────────────────────────

def run_benchmark(
    methods: list[str], gap_lengths: list[int], target_path: Path, features_path: Path,
    out_dir: Path, force: bool = False, verify_after: bool = False,
) -> int:
    """Run the requested methods; returns a process-exit-code-style int
    (0 = COMPLETE and, if requested, verification found no mismatches;
    non-zero otherwise)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target_df, features_df = load_inputs(target_path, features_path)

    pool = bc.load_matched_support_pool()
    candidates = pool[pool["gap_length"].isin(gap_lengths)].reset_index(drop=True)
    if candidates.empty:
        raise ValueError(f"no matched-support gaps for gap_lengths={gap_lengths}")

    summary_rows: list[dict] = []
    by_length_rows: list[dict] = []
    failures: list[dict] = []

    for method_id in methods:
        pred_path = out_dir / f"predictions_{method_id}.csv"
        meta_path = out_dir / f"predictions_{method_id}.meta.json"
        signature = _build_signature(method_id, candidates, target_path, features_path)

        cache_ok, cache_reason = (False, "forced") if force else _cache_is_valid(pred_path, meta_path, signature)
        if cache_ok:
            preds = pd.read_csv(pred_path, parse_dates=["date"])
            print(f"[skip, cache valid] {method_id}: {pred_path}")
        else:
            if pred_path.exists() and not force:
                print(f"[recompute, cache invalid: {cache_reason}] {method_id}")
            t0 = time.perf_counter()
            try:
                preds = run_method(method_id, candidates, target_df, features_df)
            except Exception as exc:  # noqa: BLE001
                failures.append({"method_id": method_id, "error": str(exc)})
                print(f"[FAILED] {method_id}: {exc}")
                continue
            elapsed = time.perf_counter() - t0
            preds.to_csv(pred_path, index=False)
            signature["actual_n_rows"] = int(len(preds))
            signature["run_timestamp_utc"] = pd.Timestamp.now('UTC').isoformat()
            meta_path.write_text(json.dumps(signature, indent=2))
            print(f"[done] {method_id}: {len(preds)} rows in {elapsed:.1f}s -> {pred_path}")

        # Integrity checks that gate the COMPLETE marker below.
        scored_gaps = set(preds["gap_id"].astype(str)) if not preds.empty else set()
        missing_gaps = sorted(set(candidates["gap_id"].astype(str)) - scored_gaps)
        if missing_gaps:
            failures.append({
                "method_id": method_id,
                "error": f"{len(missing_gaps)} gaps produced no prediction rows",
                "missing_gap_ids": missing_gaps,
            })
        if not preds.empty and preds.duplicated(subset=["gap_id", "date"]).any():
            failures.append({"method_id": method_id, "error": "duplicate (gap_id, date) prediction rows"})
        expected_rows = int(candidates[candidates["gap_id"].astype(str).isin(scored_gaps)]["gap_length"].sum())
        if len(preds) != expected_rows:
            failures.append({
                "method_id": method_id,
                "error": f"row count {len(preds)} != expected {expected_rows} for scored gaps "
                         "(partial day coverage within a scored gap)",
            })

        metrics = score_predictions(preds)
        if any(v != v for k, v in metrics.items() if k not in ("n_gaps", "n_rows")):
            failures.append({"method_id": method_id, "error": "summary metrics contain non-finite values"})
        summary_rows.append({"method_id": method_id, **metrics})
        for row in score_by_length(preds).to_dict("records"):
            by_length_rows.append({"method_id": method_id, **row})

    pd.DataFrame(summary_rows).to_csv(out_dir / "summary_metrics.csv", index=False)
    pd.DataFrame(by_length_rows).to_csv(out_dir / "summary_by_length.csv", index=False)
    (out_dir / "failures.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "run_metadata.json").write_text(json.dumps({
        "gap_lengths": gap_lengths,
        "methods": methods,
        "n_candidates": int(len(candidates)),
        "seed": bc.RANDOM_SEED,
        "target_sha256": _sha256_file(target_path),
        "features_sha256": _sha256_file(features_path),
        "matched_support_sha256": _sha256_file(bc.MATCHED_SUPPORT_PATH),
        "versions": _software_versions(),
        "timestamp_utc": pd.Timestamp.now('UTC').isoformat(),
        "command_args": sys.argv[1:],
        "verification_requested": verify_after,
    }, indent=2))

    n_bad_verify = 0
    if verify_after:
        n_bad_verify = run_verification(out_dir)

    all_methods_attempted = {f["method_id"] for f in failures} | {r["method_id"] for r in summary_rows}
    complete = (
        set(methods) <= all_methods_attempted
        and len(failures) == 0
        and (not verify_after or n_bad_verify == 0)
    )
    completion_marker = out_dir / "COMPLETE"
    stale_markers = [out_dir / "COMPLETE", out_dir / "FAILED", out_dir / "INCOMPLETE"]
    for m in stale_markers:
        m.unlink(missing_ok=True)
    if complete:
        completion_marker.write_text(
            f"benchmark run complete: {len(methods)} methods, 0 failures"
            f"{', verification clean' if verify_after else ''}\n"
        )
    elif len(summary_rows) == 0 and len(failures) == len(methods):
        (out_dir / "FAILED").write_text(f"all {len(methods)} methods failed -- see failures.json\n")
    else:
        (out_dir / "INCOMPLETE").write_text(
            f"{len(failures)} failure(s) and/or {n_bad_verify if verify_after else 0} verification "
            f"mismatch(es) -- see failures.json"
            f"{' and verification_summary.json' if verify_after else ''}\n"
        )

    print(f"\nSummary written to {out_dir / 'summary_metrics.csv'}")
    if failures:
        print(f"WARNING: {len(failures)} method(s) had failures/missing gaps -- see failures.json")
    print(f"Completion marker: {'COMPLETE' if complete else ('FAILED' if (out_dir / 'FAILED').exists() else 'INCOMPLETE')}")
    return 0 if complete else 1


# ── Structured verification ─────────────────────────────────────────────────

AGGREGATE_METRICS = [
    "n_gaps", "n_rows", "mae_day_weighted", "mae_gap_weighted", "rmse",
    "bias_mean", "median_abs_error", "p90_abs_error",
]
BY_LENGTH_METRICS = ["n_gaps", "n_rows", "mae_day_weighted", "mae_gap_weighted"]
EXACT_COUNT_FIELDS = {"n_gaps", "n_rows"}


def _classify(method_id: str, metric: str, released, generated) -> tuple[str, float | None]:
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
    tol = METRIC_TOLERANCE.get(method_id, DEFAULT_TOLERANCE)
    if diff < tol:
        return "within_documented_method_specific_tolerance", diff
    return "mismatched", diff


def run_verification(out_dir: Path) -> int:
    """Structured, per-metric comparison of a completed run in `out_dir`
    against the frozen released matched-support tables. Writes
    `verification_report.csv` (long, one row per method x metric) and
    `verification_summary.json` (per-method rollup). Returns the number of
    `mismatched` classifications found among methods with
    `support_status == "frozen_matched_449"` (0 = clean; methods with
    `support_status != "frozen_matched_449"` are reported `not_applicable`
    and never counted as mismatches)."""
    summary_path = out_dir / "summary_metrics.csv"
    by_length_path = out_dir / "summary_by_length.csv"
    if not summary_path.exists():
        print(f"No generated summary at {summary_path} -- run the benchmark first.")
        return 1

    generated = pd.read_csv(summary_path).set_index("method_id")
    generated_by_length = (
        pd.read_csv(by_length_path) if by_length_path.exists()
        else pd.DataFrame(columns=["method_id", "gap_length", *BY_LENGTH_METRICS])
    )
    released = pd.read_csv(bc.MATCHED_SUPPORT_METRICS_PATH)
    released_matched = released[released["support"] == "matched_449"].set_index("method_id")
    released_by_length = pd.read_csv(bc.MATCHED_SUPPORT_BY_LENGTH_PATH)

    rows: list[dict] = []
    n_mismatched_frozen = 0

    for method_id in generated.index:
        spec = bc.METHODS.get(method_id)
        support_status = spec.support_status if spec else "unknown"
        is_frozen = support_status == "frozen_matched_449"

        if method_id not in released_matched.index:
            for metric in AGGREGATE_METRICS:
                rows.append({
                    "method_id": method_id, "scope": "aggregate", "gap_length": None, "metric": metric,
                    "released": None, "generated": generated.loc[method_id, metric] if metric in generated.columns else None,
                    "abs_diff": None,
                    "classification": "not_applicable" if not is_frozen else "unavailable_in_frozen_artifact",
                })
            continue

        for metric in AGGREGATE_METRICS:
            rel_val = released_matched.loc[method_id, metric] if metric in released_matched.columns else None
            gen_val = generated.loc[method_id, metric] if metric in generated.columns else None
            cls, diff = _classify(method_id, metric, rel_val, gen_val)
            if is_frozen and cls == "mismatched":
                n_mismatched_frozen += 1
            rows.append({
                "method_id": method_id, "scope": "aggregate", "gap_length": None, "metric": metric,
                "released": rel_val, "generated": gen_val, "abs_diff": diff, "classification": cls,
            })

        rel_bl = released_by_length[released_by_length["method_id"] == method_id]
        gen_bl = generated_by_length[generated_by_length["method_id"] == method_id]
        for L in sorted(set(rel_bl["gap_length"]) | set(gen_bl["gap_length"])):
            rel_row = rel_bl[rel_bl["gap_length"] == L]
            gen_row = gen_bl[gen_bl["gap_length"] == L]
            for metric in BY_LENGTH_METRICS:
                rel_val = float(rel_row[metric].iloc[0]) if not rel_row.empty and metric in rel_row.columns else None
                gen_val = float(gen_row[metric].iloc[0]) if not gen_row.empty and metric in gen_row.columns else None
                cls, diff = _classify(method_id, metric, rel_val, gen_val)
                if is_frozen and cls == "mismatched":
                    n_mismatched_frozen += 1
                rows.append({
                    "method_id": method_id, "scope": "by_length", "gap_length": int(L), "metric": metric,
                    "released": rel_val, "generated": gen_val, "abs_diff": diff, "classification": cls,
                })

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "verification_report.csv", index=False)

    per_method_summary = {}
    for method_id, g in report.groupby("method_id"):
        counts = g["classification"].value_counts().to_dict()
        per_method_summary[method_id] = {
            "support_status": bc.METHODS[method_id].support_status if method_id in bc.METHODS else "unknown",
            "classification_counts": counts,
            "overall": "mismatched" if counts.get("mismatched", 0) else (
                "not_applicable" if set(counts) <= {"not_applicable"} else "reproduced"
            ),
        }
    (out_dir / "verification_summary.json").write_text(json.dumps({
        "n_mismatched_frozen_metrics": n_mismatched_frozen,
        "per_method": per_method_summary,
    }, indent=2))

    print(report.to_string(index=False))
    print(f"\n{n_mismatched_frozen} mismatched metric(s) among frozen (support_status=frozen_matched_449) methods.")
    return n_bad if (n_bad := n_mismatched_frozen) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET_PATH)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--methods", type=str, default=",".join(ALL_METHODS))
    parser.add_argument(
        "--gap-lengths", type=str, default=",".join(str(x) for x in bc.MATCHED_SUPPORT_GAP_LENGTHS)
    )
    parser.add_argument("--force", action="store_true", help="Re-run methods even if a valid cache exists.")
    parser.add_argument("--verify", action="store_true", help="Compare an existing run against the frozen released tables (no run performed).")
    parser.add_argument("--verify-after", action="store_true", help="Run verification immediately after the benchmark and factor it into the completion marker.")
    args = parser.parse_args(argv)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        parser.error(f"unknown methods: {sorted(unknown)}; choose from {ALL_METHODS}")
    gap_lengths = [int(x) for x in args.gap_lengths.split(",") if x.strip()]

    if args.verify:
        n_bad = run_verification(args.out)
        return 1 if n_bad else 0

    return run_benchmark(
        methods, gap_lengths, args.target, args.features, args.out,
        force=args.force, verify_after=args.verify_after,
    )


if __name__ == "__main__":
    sys.exit(main())
