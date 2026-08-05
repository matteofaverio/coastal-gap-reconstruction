"""Deterministic assembly of chlorophyll real-gap reconstruction candidates.

Consumes the two already-released, frozen per-method candidate files
(`results/chlorophyll/chlorophyll_reconstruction_engineered_hybrid.csv`,
`chlorophyll_reconstruction_tsicl_satellite_proxy.csv`) and the real-gap
inventory (`real_gap_inventory.py`), and joins them into gap-level and
day-level assembled tables -- the deterministic candidate join the earlier
publication audit found missing from the public repository.

**This script never runs TS-ICL, never fits a model, and never regenerates
a benchmark.** It is pure validation + joining over frozen inputs. It never
silently substitutes interpolation (or anything else) for a failed/missing
method -- a gap with no reconstruction from a given method is left absent
for that method, reported explicitly in the assembly manifest, not filled.

Default output: `build/chlorophyll/real_gap_candidates/` (never overwrites
`results/` by default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import real_gap_contract as rgc
from . import real_gap_inventory as ri

__all__ = [
    "load_engineered_hybrid_candidates", "load_tsicl_satellite_proxy_candidates",
    "validate_candidate_rows", "assemble_gap_level", "assemble_day_level", "run_assembly",
]


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_engineered_hybrid_candidates(path: Path | None = None) -> pd.DataFrame:
    """Load the engineered-hybrid per-day candidate file, restricted to
    actually-reconstructed (real-gap) rows -- physical mg/m^3 scale, no
    conversion applied."""
    path = path or rgc.ENGINEERED_HYBRID_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    recon = df[df["is_reconstructed"] == True].copy()  # noqa: E712
    recon["method_id"] = "engineered_hybrid"
    recon["scale"] = "physical_mg_m3"
    return recon[["gap_id", "date", "method_id", "final_chl", "scale", "method",
                  "uncertainty_lower", "uncertainty_upper", "extrapolation_flag"]].rename(
        columns={"final_chl": "point_pred", "method": "component_method"},
    )


def load_tsicl_satellite_proxy_candidates(path: Path | None = None) -> pd.DataFrame:
    """Load the TS-ICL satellite-proxy per-day candidate file -- both log10
    and physical scale preserved as separate columns, no silent conversion."""
    path = path or rgc.TSICL_SATELLITE_PROXY_PATH
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.copy()
    df["method_id"] = "tsicl_satellite_proxy"
    df["scale"] = "physical_mg_m3"
    return df.rename(columns={"pred_chl_mg_m3": "point_pred"})[
        ["gap_id", "date", "method_id", "point_pred", "scale", "pred_log10_chl",
         "q05_chl_mg_m3", "q10_chl_mg_m3", "q25_chl_mg_m3", "q50_chl_mg_m3",
         "q75_chl_mg_m3", "q90_chl_mg_m3", "q95_chl_mg_m3", "scenario_only_256day"]
    ]


def validate_candidate_rows(day_level: pd.DataFrame, inventory: pd.DataFrame) -> list[str]:
    """Validate the assembled (wide-format, one row per gap-day, one column
    group per method) day-level candidate table. Returns a list of
    violation strings (empty = valid). Does not raise -- callers decide
    whether a violation is fatal."""
    violations: list[str] = []

    dup = day_level.duplicated(subset=["gap_id", "date"])
    if dup.any():
        violations.append(f"{int(dup.sum())} duplicate (gap_id, date) rows -- a conflicting/duplicate candidate join")

    point_pred_cols = [c for c in day_level.columns if c.endswith("_point_pred") or c.endswith("_point_pred_mg_m3")]
    for col in point_pred_cols:
        vals = day_level[col].dropna().astype(float)
        non_finite = ~np.isfinite(vals)
        if non_finite.any():
            violations.append(f"{int(non_finite.sum())} non-finite values in {col!r}")

    quantile_cols = [c for c in day_level.columns if c.startswith("q") and "_chl_mg_m3" in c]
    if quantile_cols:
        ordered_cols = sorted(quantile_cols, key=lambda c: float(c.split("_")[0][1:]) / 100)
        q_arr = day_level[ordered_cols].to_numpy(dtype=float)
        valid_rows = np.isfinite(q_arr).all(axis=1)
        if valid_rows.any():
            diffs = np.diff(q_arr[valid_rows], axis=1)
            if (diffs < -1e-9).any():
                violations.append("quantile columns not monotonically non-decreasing in at least one row")

    inv_bounds = inventory.set_index("gap_id")[["start_date", "end_date"]]
    inv_bounds = inv_bounds.assign(
        start_date=pd.to_datetime(inv_bounds["start_date"]), end_date=pd.to_datetime(inv_bounds["end_date"]),
    )
    joined = day_level.merge(inv_bounds, on="gap_id", how="left", suffixes=("", "_inv"))
    out_of_bounds = (joined["date"] < joined["start_date"]) | (joined["date"] > joined["end_date"])
    if out_of_bounds.any():
        violations.append(f"{int(out_of_bounds.sum())} rows with dates outside their declared real-gap window")

    unknown_gap = ~day_level["gap_id"].isin(inventory["gap_id"])
    if unknown_gap.any():
        violations.append(f"{int(unknown_gap.sum())} rows reference a gap_id absent from the real-gap inventory")

    return violations


def assemble_day_level(
    engineered_hybrid: pd.DataFrame, tsicl_satellite_proxy: pd.DataFrame,
) -> pd.DataFrame:
    """Outer-join the two per-method candidate tables on (gap_id, date) --
    a gap-day present for one method but not the other is preserved with
    NaN for the missing method's columns, never silently dropped or
    filled."""
    eh = engineered_hybrid.rename(columns={
        "point_pred": "engineered_hybrid_point_pred", "component_method": "engineered_hybrid_component_method",
        "uncertainty_lower": "engineered_hybrid_uncertainty_lower",
        "uncertainty_upper": "engineered_hybrid_uncertainty_upper",
        "extrapolation_flag": "engineered_hybrid_extrapolation_flag",
    }).drop(columns=["method_id", "scale"])
    ts = tsicl_satellite_proxy.rename(columns={
        "point_pred": "tsicl_satellite_proxy_point_pred_mg_m3", "pred_log10_chl": "tsicl_satellite_proxy_point_pred_log10",
    }).drop(columns=["method_id", "scale"])
    return eh.merge(ts, on=["gap_id", "date"], how="outer")


def assemble_gap_level(day_level: pd.DataFrame, inventory: pd.DataFrame) -> pd.DataFrame:
    """One row per real gap: mean prediction per method, method identity,
    and support-difference reporting (which methods actually cover this
    gap) -- an explicit column, not an undocumented intersection."""
    rows = []
    for gap_id, g in day_level.groupby("gap_id"):
        eh_available = g["engineered_hybrid_point_pred"].notna().any()
        ts_available = g["tsicl_satellite_proxy_point_pred_mg_m3"].notna().any()
        rows.append({
            "gap_id": gap_id,
            "n_days": len(g),
            "engineered_hybrid_available": bool(eh_available),
            "engineered_hybrid_mean_pred_mg_m3": float(g["engineered_hybrid_point_pred"].mean()) if eh_available else None,
            "engineered_hybrid_component_method": g["engineered_hybrid_component_method"].dropna().iloc[0] if eh_available else None,
            "tsicl_satellite_proxy_available": bool(ts_available),
            "tsicl_satellite_proxy_mean_pred_mg_m3": float(g["tsicl_satellite_proxy_point_pred_mg_m3"].mean()) if ts_available else None,
            "support_note": (
                "both methods cover this gap" if (eh_available and ts_available)
                else "engineered_hybrid only (tsicl unavailable)" if eh_available
                else "tsicl_satellite_proxy only (engineered_hybrid unavailable -- context-constrained, see real_gap_inventory)"
                if ts_available else "no candidate available from either method"
            ),
        })
    gap_level = pd.DataFrame(rows)
    assembled = inventory.merge(gap_level, on="gap_id", how="left")
    # The 256-day (2020) gap specifically -- scenario-only, outside the
    # validated artificial-gap length envelope. Flagged explicitly here
    # (not silently grouped with L1-L30/L1-L60 validated evidence) so any
    # downstream table/plot can filter or annotate it without re-deriving
    # the length threshold. Matches the released files' own
    # `scenario_only_256day` flag exactly (verified in
    # tests/test_assemble_real_gap_candidates.py).
    assembled["scenario_only_256day"] = assembled["length_days"] == 256
    return assembled


def run_assembly(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_csv(rgc.DAILY_TARGET_PATH, parse_dates=["date"]).set_index("date").sort_index()
    inventory = ri.detect_real_gaps(target_df)

    eh = load_engineered_hybrid_candidates()
    ts = load_tsicl_satellite_proxy_candidates()
    day_level = assemble_day_level(eh, ts)

    violations = validate_candidate_rows(day_level, inventory)
    if violations:
        (out_dir / "VALIDATION_STATUS").write_text("FAILED:\n" + "\n".join(violations) + "\n")
        print("VALIDATION FAILED:")
        for v in violations:
            print(f"  - {v}")
        return 1
    (out_dir / "VALIDATION_STATUS").write_text("PASSED\n")

    gap_level = assemble_gap_level(day_level, inventory)

    day_level.to_csv(out_dir / "real_gap_candidates_daily.csv", index=False)
    gap_level.to_csv(out_dir / "real_gap_candidates_gap_level.csv", index=False)

    manifest = {
        "inputs": {
            "engineered_hybrid_sha256": _sha256_file(rgc.ENGINEERED_HYBRID_PATH),
            "tsicl_satellite_proxy_sha256": _sha256_file(rgc.TSICL_SATELLITE_PROXY_PATH),
            "daily_target_sha256": _sha256_file(rgc.DAILY_TARGET_PATH),
        },
        "n_real_gaps_inventoried": len(inventory),
        "n_gaps_with_engineered_hybrid": int(gap_level["engineered_hybrid_available"].fillna(False).sum()),
        "n_gaps_with_tsicl_satellite_proxy": int(gap_level["tsicl_satellite_proxy_available"].fillna(False).sum()),
        "n_gaps_with_both": int((gap_level["engineered_hybrid_available"].fillna(False)
                                  & gap_level["tsicl_satellite_proxy_available"].fillna(False)).sum()),
        "n_day_level_rows": len(day_level),
        "validation_status": "PASSED",
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "assembly_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Assembled {len(gap_level)} gaps, {len(day_level)} day-level rows -> {out_dir}")
    print(json.dumps(manifest, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parent.parent.parent
    parser.add_argument("--out", type=Path, default=repo_root / "build" / "chlorophyll" / "real_gap_candidates")
    args = parser.parse_args(argv)
    return run_assembly(args.out)


if __name__ == "__main__":
    sys.exit(main())
