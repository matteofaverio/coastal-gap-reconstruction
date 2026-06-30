"""Small plotting helpers for the figure types used in this benchmark.

These are intentionally minimal -- enough to reproduce the basic shape of
the figures shipped under figures/chlorophyll/, not a full plotting library.
Requires matplotlib (not a hard dependency of the package; import lazily).
"""

from __future__ import annotations

import pandas as pd


def plot_target_with_gaps(target_df: pd.DataFrame, target_col: str = "chl_mean", ax=None):
    """Plot the full daily target series with missing/non-eligible stretches shaded."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 4))

    ax.plot(target_df.index, target_df[target_col], lw=0.8, color="tab:green")
    ax.set_xlabel("Date")
    ax.set_ylabel(target_col)
    ax.set_title("Daily target series")
    return ax


def plot_method_comparison_by_length(
    scores_df: pd.DataFrame,
    method_col: str = "method_public_name",
    stratum_col: str = "stratum_value",
    value_col: str = "day_weighted_mae",
    ax=None,
):
    """Plot MAE by gap length for each method, given a tidy long-format scores table."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for method, group in scores_df.groupby(method_col):
        group_sorted = group.sort_values(stratum_col)
        ax.plot(group_sorted[stratum_col], group_sorted[value_col], marker="o", label=method)

    ax.set_xlabel("Gap length")
    ax.set_ylabel(value_col)
    ax.legend(fontsize=7, loc="best")
    return ax


def plot_event_vs_nonevent(event_df: pd.DataFrame, ax=None):
    """Bar plot comparing event-day vs non-event-day MAE per method."""
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    methods = event_df["method_public_name"]
    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, event_df["mae_event_days"], width, label="Event days")
    ax.bar(x + width / 2, event_df["mae_nonevent_days"], width, label="Non-event days")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAE")
    ax.legend()
    return ax
