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

    def to_flat_dict(self) -> dict:
        """Flatten `metrics` into a single-row-friendly dict, matching the
        private project's released `paired_bootstrap_ci_*.csv` column
        naming (`{metric}_a`, `{metric}_b`, `{metric}_delta`,
        `{metric}_ci_lo`, `{metric}_ci_hi`)."""
        row = {
            "method_a": self.method_a, "method_b": self.method_b,
            "stratum_label": self.stratum_label, "n_gaps": self.n_gaps,
            "n_replicates": self.n_replicates, "interpretation": self.interpretation,
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
) -> PairedComparisonResult | None:
    """Gap-clustered paired bootstrap comparison of `method_a` vs `method_b`.

    `day_level` is a long-format DataFrame with one row per (method, gap,
    day) and at minimum `method_col`/`gap_id_col`/`error_col`. Only gaps
    both methods have day rows for are used (paired support); returns
    `None` if that intersection is empty.

    **Resampling unit**: `gap_id`, with replacement, `n_replicates` times.
    Every day belonging to a resampled gap is carried along with it (a
    day-level bootstrap is never used -- see module docstring). Sign
    convention: `delta = metric_a - metric_b`; negative means `method_a` has
    lower error (better). Significance: the 95% CI excludes zero
    (`ci_lo > 0` or `ci_hi < 0`).
    """
    sub_a = day_level[day_level[method_col] == method_a]
    sub_b = day_level[day_level[method_col] == method_b]
    if gap_ids_allowed is not None:
        sub_a = sub_a[sub_a[gap_id_col].isin(gap_ids_allowed)]
        sub_b = sub_b[sub_b[gap_id_col].isin(gap_ids_allowed)]

    err_a = {gid: g[error_col].to_numpy() for gid, g in sub_a.groupby(gap_id_col)}
    err_b = {gid: g[error_col].to_numpy() for gid, g in sub_b.groupby(gap_id_col)}
    common_gids = np.array(sorted(set(err_a) & set(err_b)))
    n_gaps = len(common_gids)
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
    )
