"""Gap-clustered paired bootstrap comparisons, reusable across targets.

Ported from the private project's canonical paired-bootstrap procedure (the
one that produced the headline TS-ICL-vs-baseline confidence intervals cited
throughout this project's results). The design choice this module encodes
and never silently deviates from:

**Resample gap_id with replacement; every day inside a resampled gap comes
along with it. Day-level bootstrap is never used.** Days within one gap are
not independent draws (they share the same context, the same withheld
event/non-event regime, and often the same model failure mode) -- resampling
individual days would understate the true uncertainty. See the module-level
`bootstrap_compare` docstring for the exact resampling unit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "PairedComparisonResult",
    "gap_level_metrics",
    "bootstrap_compare",
    "gap_cluster_bootstrap_ci",
]

DEFAULT_N_REPLICATES = 2000
DEFAULT_SEED = 42
MIN_GAPS_FOR_SUPPORT = 10


@dataclass
class PairedComparisonResult:
    """One paired comparison's point estimate, CI, and interpretation."""

    method_a: str
    method_b: str
    stratum_label: str
    n_gaps: int
    n_replicates: int
    metrics: dict = field(default_factory=dict)  # metric_name -> {a, b, delta, ci_lo, ci_hi}
    interpretation: str = ""
    original_gap_count_a: int = 0
    original_gap_count_b: int = 0
    paired_gap_count: int = 0
    paired_day_count: int = 0
    excluded_gaps: list = field(default_factory=list)  # [{"gap_id", "reason", ...}]

    def to_flat_dict(self) -> dict:
        """Flatten `metrics` into a single-row-friendly dict, matching the
        private project's released `paired_bootstrap_ci_*.csv` column
        naming (`{metric}_a`, `{metric}_b`, `{metric}_delta`,
        `{metric}_ci_lo`, `{metric}_ci_hi`)."""
        row = {
            "method_a": self.method_a, "method_b": self.method_b,
            "stratum_label": self.stratum_label, "n_gaps": self.n_gaps,
            "n_replicates": self.n_replicates, "interpretation": self.interpretation,
            "original_gap_count_a": self.original_gap_count_a,
            "original_gap_count_b": self.original_gap_count_b,
            "paired_gap_count": self.paired_gap_count,
            "paired_day_count": self.paired_day_count,
            "n_excluded_gaps": len(self.excluded_gaps),
        }
        for metric, vals in self.metrics.items():
            for key in ("a", "b", "delta", "ci_lo", "ci_hi"):
                row[f"{metric}_{key}"] = vals[key]
        return row


def gap_level_metrics(day_errors: dict[str, np.ndarray], gap_ids: np.ndarray) -> dict[str, float]:
    """Compute the standard metric set from a `{gap_id: abs_error_array}`
    mapping, restricted to (and possibly repeating, under bootstrap
    resampling) `gap_ids`.

    `day_weighted_mae`: mean over all days across the (possibly repeated)
    gaps. `gap_weighted_mae`: mean of each gap's own mean, then averaged
    across gaps -- a long gap does not dominate this one the way it does the
    day-weighted metric.
    """
    all_days = np.concatenate([day_errors[g] for g in gap_ids])
    gap_means = np.array([day_errors[g].mean() for g in gap_ids])
    return {
        "day_weighted_mae": float(all_days.mean()),
        "gap_weighted_mae": float(gap_means.mean()),
        "rmse": float(np.sqrt((all_days**2).mean())),
        "median_ae": float(np.median(all_days)),
        "p90_ae": float(np.quantile(all_days, 0.9)),
    }


def gap_cluster_bootstrap_ci(
    values: np.ndarray, n_replicates: int = DEFAULT_N_REPLICATES, seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Resample a 1D array of **gap-level** values (one value per gap) with
    replacement; return `(point_mean, ci_lo_2.5pct, ci_hi_97.5pct)`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_means = np.empty(n_replicates)
    for b in range(n_replicates):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    return float(values.mean()), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def bootstrap_compare(
    method_a: str,
    method_b: str,
    day_level: pd.DataFrame,
    gap_ids_allowed: set[str] | None = None,
    stratum_label: str = "all_gaps",
    n_replicates: int = DEFAULT_N_REPLICATES,
    seed: int = DEFAULT_SEED,
    method_col: str = "method_id",
    gap_id_col: str = "gap_id",
    error_col: str = "absolute_error_log10",
    date_col: str = "date",
    pairing: str = "strict",
) -> PairedComparisonResult | None:
    """Gap-clustered paired bootstrap comparison of `method_a` vs `method_b`.

    `day_level` is a long-format DataFrame with one row per (method, gap,
    day) and at minimum `method_col`/`gap_id_col`/`date_col`/`error_col`.
    Only gaps both methods have day rows for are considered a pairing
    candidate; returns `None` if that intersection is empty.

    **Exact day-level pairing.** Sharing a `gap_id` is not sufficient to
    pair two methods' rows -- each method must also have day rows on
    *exactly* the same dates within that gap, with no duplicate
    (method, gap, date) rows. `pairing="strict"` (the default) raises
    `ValueError` if any common gap has a within-gap date-support mismatch
    (different date sets, different row counts, or duplicate dates) for
    either method. `pairing="intersection"` instead silently restricts each
    mismatched gap to its common date intersection and excludes it if that
    intersection is empty -- use only when explicitly reporting an
    intersection-mode comparison, never as an unstated default.

    **Resampling unit**: `gap_id`, with replacement, `n_replicates` times.
    Every day belonging to a resampled gap is carried along with it (a
    day-level bootstrap is never used -- see module docstring). Sign
    convention: `delta = metric_a - metric_b`; negative means `method_a` has
    lower error (better). Significance: the 95% CI excludes zero
    (`ci_lo > 0` or `ci_hi < 0`).
    """
    if pairing not in ("strict", "intersection"):
        raise ValueError(f"pairing must be 'strict' or 'intersection', got {pairing!r}")

    sub_a = day_level[day_level[method_col] == method_a]
    sub_b = day_level[day_level[method_col] == method_b]
    if gap_ids_allowed is not None:
        sub_a = sub_a[sub_a[gap_id_col].isin(gap_ids_allowed)]
        sub_b = sub_b[sub_b[gap_id_col].isin(gap_ids_allowed)]

    groups_a = {gid: g for gid, g in sub_a.groupby(gap_id_col)}
    groups_b = {gid: g for gid, g in sub_b.groupby(gap_id_col)}
    candidate_gids = sorted(set(groups_a) & set(groups_b))

    err_a: dict[str, np.ndarray] = {}
    err_b: dict[str, np.ndarray] = {}
    excluded: list[dict] = []
    for gid in candidate_gids:
        ga, gb = groups_a[gid], groups_b[gid]
        dup_a = ga[date_col].duplicated().any()
        dup_b = gb[date_col].duplicated().any()
        dates_a, dates_b = set(ga[date_col]), set(gb[date_col])
        if dup_a or dup_b or dates_a != dates_b:
            reason = (
                "duplicate_date_rows" if (dup_a or dup_b)
                else "date_support_mismatch"
            )
            if pairing == "strict":
                raise ValueError(
                    f"exact day-level pairing failed for gap {gid!r} ({method_a!r} vs "
                    f"{method_b!r}): {reason} (n_a={len(ga)}, n_b={len(gb)}, "
                    f"dates_a-dates_b={sorted(dates_a - dates_b)}, "
                    f"dates_b-dates_a={sorted(dates_b - dates_a)}). Use pairing='intersection' "
                    f"to explicitly restrict to the common date subset instead of raising."
                )
            common_dates = dates_a & dates_b
            excluded.append({"gap_id": gid, "reason": reason, "n_a": len(ga), "n_b": len(gb),
                              "n_common_dates": len(common_dates)})
            if not common_dates:
                continue
            ga = ga[ga[date_col].isin(common_dates)].drop_duplicates(subset=[date_col])
            gb = gb[gb[date_col].isin(common_dates)].drop_duplicates(subset=[date_col])
        err_a[gid] = ga.sort_values(date_col)[error_col].to_numpy()
        err_b[gid] = gb.sort_values(date_col)[error_col].to_numpy()

    common_gids = np.array(sorted(err_a))
    n_gaps = len(common_gids)
    n_days = int(sum(len(v) for v in err_a.values()))
    if n_gaps == 0:
        return None

    rng = np.random.default_rng(seed)
    point_a = gap_level_metrics(err_a, common_gids)
    point_b = gap_level_metrics(err_b, common_gids)

    deltas: dict[str, list[float]] = {k: [] for k in point_a}
    for _ in range(n_replicates):
        sampled = rng.choice(common_gids, size=n_gaps, replace=True)
        ma = gap_level_metrics(err_a, sampled)
        mb = gap_level_metrics(err_b, sampled)
        for k in point_a:
            deltas[k].append(ma[k] - mb[k])

    metrics = {}
    for k in point_a:
        point_delta = point_a[k] - point_b[k]
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        metrics[k] = {"a": point_a[k], "b": point_b[k], "delta": point_delta, "ci_lo": float(lo), "ci_hi": float(hi)}

    headline = metrics["day_weighted_mae"]
    if n_gaps < MIN_GAPS_FOR_SUPPORT:
        interpretation = "insufficient_support"
    elif headline["ci_lo"] < 0 and headline["ci_hi"] < 0:
        interpretation = "significant_improvement"
    elif headline["ci_lo"] > 0 and headline["ci_hi"] > 0:
        interpretation = "significant_degradation"
    elif abs(headline["delta"]) < 1e-6:
        interpretation = "indistinguishable"
    else:
        interpretation = "directional_not_significant"

    return PairedComparisonResult(
        method_a=method_a, method_b=method_b, stratum_label=stratum_label,
        n_gaps=n_gaps, n_replicates=n_replicates, metrics=metrics, interpretation=interpretation,
        original_gap_count_a=len(groups_a), original_gap_count_b=len(groups_b),
        paired_gap_count=n_gaps, paired_day_count=n_days, excluded_gaps=excluded,
    )
