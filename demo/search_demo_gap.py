"""Objective search for a pedagogically representative 14-day demonstration
gap, run with the actual demo method implementations (`demo/src/methods.py`),
not only old cached benchmark tables.

Not needed to run the demo -- this is a one-off selection tool. Its output is
a diagnostic audit table, not a notebook artifact. Run from public_export/:

    python3 demo/search_demo_gap.py

Requires the same environment as the live demo (tsicl, torch, scikit-learn) --
run it with the `.venv_tsicl_demo` interpreter, e.g.:

    .venv_tsicl_demo/bin/python3 demo/search_demo_gap.py

Writes demo/outputs/demo_gap_selection_audit.csv (top 10 candidates) and
prints the selected gap.
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import demo_helpers as dh  # noqa: E402
from src import methods as mth  # noqa: E402

GAP_LENGTH = 14
CONTEXT_DAYS = 75  # local window radius around the gap, matching build_demo_data.py
MIN_PRE_CONTEXT_DAYS = 30
MIN_POST_CONTEXT_DAYS = 30
MIN_PROXY_COMPLETENESS = 0.8
MIN_WIND_SST_COMPLETENESS = 0.95
STEP_DAYS = 5  # candidate start-date stride, trades thoroughness for runtime

METHOD_NAMES = [
    "persistence",
    "climatology",
    "linear_interpolation",
    "gaussian_process",
    "external_tabular",
    "gap_edge_residual",
    "tsicl_target_only",
    "tsicl_satellite_proxy",
    "tsicl_physical_bundle",
]


def load_full_chlorophyll_record() -> pd.DataFrame:
    """Same source join as demo/build_demo_data.py, but for the whole record
    (no window slicing) -- date, chl_mean, chl_satellite_proxy_log10,
    wind_spd_ms, sst_primary_degC, target_eligible_default."""
    target = pd.read_csv("data_public/chlorophyll/chlorophyll_daily_target.csv", parse_dates=["date"])
    feat = pd.read_csv("data_public/chlorophyll/chlorophyll_predictor_features_curated.csv", parse_dates=["date"])
    df = target[["date", "chl_mean", "target_eligible_default"]].merge(
        feat[["date", "chl_cons_log10", "wind_spd_ms", "sst_primary_degC"]].rename(
            columns={"chl_cons_log10": "chl_satellite_proxy_log10"}
        ),
        on="date",
        how="left",
    )
    return df.sort_values("date").reset_index(drop=True)


def enumerate_candidates(full: pd.DataFrame) -> list[dict]:
    """Hard-constraint filter: fully observed target inside the window,
    enough pre/post context, and enough covariate completeness. Returns a
    list of candidate dicts (not yet scored by any method)."""
    dates = full["date"].to_numpy()
    n = len(full)
    candidates = []

    for start_idx in range(MIN_PRE_CONTEXT_DAYS, n - GAP_LENGTH - MIN_POST_CONTEXT_DAYS, STEP_DAYS):
        end_idx = start_idx + GAP_LENGTH - 1
        window = full.iloc[start_idx : end_idx + 1]
        if window["chl_mean"].isna().any():
            continue  # gap must be fully observed to serve as an artificial gap

        ctx_lo = max(0, start_idx - CONTEXT_DAYS)
        ctx_hi = min(n, end_idx + 1 + CONTEXT_DAYS)
        pre_ctx = full.iloc[ctx_lo:start_idx]
        post_ctx = full.iloc[end_idx + 1 : ctx_hi]
        if len(pre_ctx) < MIN_PRE_CONTEXT_DAYS or len(post_ctx) < MIN_POST_CONTEXT_DAYS:
            continue

        proxy_completeness = window["chl_satellite_proxy_log10"].notna().mean()
        wind_completeness = window["wind_spd_ms"].notna().mean()
        sst_completeness = window["sst_primary_degC"].notna().mean()
        if proxy_completeness < MIN_PROXY_COMPLETENESS:
            continue
        if wind_completeness < MIN_WIND_SST_COMPLETENESS or sst_completeness < MIN_WIND_SST_COMPLETENESS:
            continue

        candidates.append(
            {
                "start": pd.Timestamp(dates[start_idx]),
                "end": pd.Timestamp(dates[end_idx]),
                "ctx_lo": ctx_lo,
                "ctx_hi": ctx_hi,
                "proxy_completeness": float(proxy_completeness),
                "wind_completeness": float(wind_completeness),
                "sst_completeness": float(sst_completeness),
            }
        )
    return candidates


def evaluate_candidate(full: pd.DataFrame, cand: dict, tsicl_model) -> dict | None:
    """Run every demo method on one candidate gap, from the same local window
    construction the demo notebook itself uses. Returns a result dict, or
    None if any method fails to run (e.g. degenerate context)."""
    local_window = full.iloc[cand["ctx_lo"] : cand["ctx_hi"]].reset_index(drop=True)
    start_str = cand["start"].date().isoformat()
    end_str = cand["end"].date().isoformat()

    try:
        gap = dh.create_artificial_gap(local_window, start=start_str, end=end_str)
    except ValueError:
        return None

    results: dict[str, mth.MethodResult] = {}
    try:
        results.update(mth.run_baselines(gap, full))
        results["gaussian_process"] = mth.run_gaussian_process(gap)
        results["external_tabular"] = mth.run_external_tabular(gap, full)
        results["gap_edge_residual"] = mth.run_gap_edge_residual(gap, full)

        target_log10_masked = np.log10(gap.full_series[gap.target_column].to_numpy())
        target_log10_masked[gap.is_gap.to_numpy()] = np.nan
        results["tsicl_target_only"] = mth.run_tsicl(tsicl_model, gap, None, [], "tsicl_target_only")
        results["tsicl_satellite_proxy"] = mth.run_tsicl(
            tsicl_model, gap,
            gap.full_series[["chl_satellite_proxy_log10"]].to_numpy(dtype=np.float32),
            ["chl_satellite_proxy_log10"], "tsicl_satellite_proxy",
        )
        results["tsicl_physical_bundle"] = mth.run_tsicl(
            tsicl_model, gap,
            gap.full_series[["wind_spd_ms", "sst_primary_degC"]].to_numpy(dtype=np.float32),
            ["wind_spd_ms", "sst_primary_degC"], "tsicl_physical_bundle",
        )
    except Exception as e:  # noqa: BLE001 -- a candidate that fails any method is simply dropped
        print(f"  candidate {start_str}..{end_str} failed: {type(e).__name__}: {e}")
        return None

    mae = {name: dh.mean_absolute_error(r.prediction, gap.truth) for name, r in results.items()}
    ranks = pd.Series(mae).rank().to_dict()

    truth_values = gap.truth["truth"].to_numpy()
    p90 = np.nanpercentile(local_window["chl_mean"].to_numpy(), 90)
    is_event = bool((truth_values > p90).any())

    return {
        "start": start_str,
        "end": end_str,
        **{f"mae_{k}": v for k, v in mae.items()},
        **{f"rank_{k}": v for k, v in ranks.items()},
        "target_p90_local_window": float(p90),
        "is_event_gap": is_event,
        "target_std_in_gap": float(np.std(truth_values)),
        "target_mean_in_gap": float(np.mean(truth_values)),
        "proxy_completeness": cand["proxy_completeness"],
        "wind_completeness": cand["wind_completeness"],
        "sst_completeness": cand["sst_completeness"],
    }


def score_candidate(row: pd.Series) -> float:
    """Explicit selection score -- lower is better. Every term is documented;
    nothing here is a manual/visual preference for one specific interval."""
    penalty = 0.0

    # TS-ICL + satellite chlorophyll should be among the best methods.
    penalty += 2.0 * (row["rank_tsicl_satellite_proxy"] - 1)  # 0 if it's the best method

    # Interpolation must not be catastrophically bad (should rank in the
    # upper half of methods, not dead last).
    n_methods = len(METHOD_NAMES)
    if row["rank_linear_interpolation"] >= n_methods:  # worst method
        penalty += 5.0
    elif row["rank_linear_interpolation"] > n_methods * 0.75:
        penalty += 1.5

    # External tabular must not be artificially dominant (rank 1 or 2).
    if row["rank_external_tabular"] <= 2:
        penalty += 4.0

    # No single-method pathological collapse: any method's MAE far above the
    # median MAE across methods signals a broken/degenerate fit for this
    # candidate, not real method behaviour.
    mae_cols = [f"mae_{m}" for m in METHOD_NAMES]
    mae_values = row[mae_cols].astype(float)
    median_mae = mae_values.median()
    if median_mae > 0:
        worst_ratio = mae_values.max() / median_mae
        if worst_ratio > 4.0:
            penalty += 3.0

    # Prefer a non-event gap for the main illustrative example (a single
    # extreme spike would look cherry-picked either for or against a method).
    if row["is_event_gap"]:
        penalty += 2.0

    # Prefer high covariate completeness.
    penalty += (1.0 - row["proxy_completeness"]) * 2.0

    # Prefer a representative (not tiny, not huge) error scale: TS-ICL+proxy
    # MAE should sit in a plausible range relative to the target's own scale
    # in this window (avoid near-zero MAE, which usually signals a
    # near-constant window with little pedagogical value).
    target_scale = max(row["target_std_in_gap"], 1e-6)
    relative_mae = row["mae_tsicl_satellite_proxy"] / target_scale
    if relative_mae < 0.15:
        penalty += 1.5  # too easy / too flat to be visually informative

    return penalty


def main() -> None:
    full = load_full_chlorophyll_record()
    print(f"Full record: {len(full)} days, {full['date'].min().date()} to {full['date'].max().date()}")

    candidates = enumerate_candidates(full)
    print(f"Candidates passing hard constraints: {len(candidates)}")

    print("Loading TS-ICL live once, reused for every candidate ...")
    tsicl_model, status = mth.load_tsicl()
    if not status.live:
        raise RuntimeError(f"TS-ICL did not load live ({status.error}) -- cannot run an honest search without it.")
    print(f"TS-ICL loaded live: device={status.device}, load_time={status.load_time_s:.2f}s")

    rows = []
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i, cand in enumerate(candidates):
            result = evaluate_candidate(full, cand, tsicl_model)
            if result is not None:
                rows.append(result)
            if (i + 1) % 20 == 0:
                elapsed = time.time() - t0
                print(f"  {i + 1}/{len(candidates)} candidates evaluated ({elapsed:.0f}s elapsed, {len(rows)} succeeded)")

    print(f"Evaluated {len(rows)} candidates successfully in {time.time() - t0:.0f}s")
    results_df = pd.DataFrame(rows)
    results_df["selection_score"] = results_df.apply(score_candidate, axis=1)
    results_df = results_df.sort_values("selection_score").reset_index(drop=True)

    top10 = results_df.head(10).copy()

    def reason(row: pd.Series) -> str:
        bits = []
        if row["rank_tsicl_satellite_proxy"] <= 2:
            bits.append("TS-ICL+proxy near-best")
        if row["rank_linear_interpolation"] <= len(METHOD_NAMES) * 0.6:
            bits.append("interpolation a credible baseline")
        if row["rank_external_tabular"] > 2:
            bits.append("external-tabular not dominant")
        if not row["is_event_gap"]:
            bits.append("non-event/background window")
        return "; ".join(bits) if bits else "did not meet most preferred criteria"

    top10["reason"] = top10.apply(reason, axis=1)

    out_dir = Path("demo/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "demo_gap_selection_audit.csv"
    top10.to_csv(out_path, index=False)
    print(f"\nWrote top-10 audit table to {out_path}")

    best = top10.iloc[0]
    print("\n=== SELECTED GAP ===")
    print(f"start={best['start']}  end={best['end']}  score={best['selection_score']:.3f}")
    for m in METHOD_NAMES:
        print(f"  {m:24s} MAE={best[f'mae_{m}']:.3f}  rank={best[f'rank_{m}']:.0f}")
    print(f"  is_event_gap={best['is_event_gap']}  proxy_completeness={best['proxy_completeness']:.2f}")


if __name__ == "__main__":
    main()
