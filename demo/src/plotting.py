"""Plotting functions for the gap-reconstruction walkthrough notebook.

Every function here returns `(fig, axes)` and never calls `plt.show()` or
`plt.savefig()` -- the notebook cell that calls a plotting function is
responsible for calling `plt.show()` explicitly. This keeps the figures
inline-only: nothing here writes an image file.

Matplotlib only (no seaborn). Colors below are used consistently across every
figure in the notebook so the same method always has the same color.
"""
from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Semantic color/style palette, shared by every plotting function below.
# --------------------------------------------------------------------------
INK = "#2B2B2B"
FAINT = "#8A9299"

OBSERVED = "#2A7F8E"       # observed target context
TRUTH = "#B24743"          # withheld truth (artificial gap) -- always dashed
ARTIFICIAL_GAP_SHADE = "#B24743"
REAL_GAP_SHADE = "#5F7F8A"

PERSISTENCE = "#8A9299"
CLIMATOLOGY = "#C0791F"
INTERPOLATION = "#5F7F8A"
GAUSSIAN_PROCESS = "#1B9E77"
EXTERNAL_TABULAR = "#3B6FA0"
GAP_EDGE_RESIDUAL = "#D98A2B"
TSICL_TARGET_ONLY = "#5E4B9B"
TSICL_COVARIATE = "#7E4B9B"
UNCERTAINTY_ALPHA = 0.15

METHOD_COLORS = {
    "persistence": PERSISTENCE,
    "climatology": CLIMATOLOGY,
    "linear_interpolation": INTERPOLATION,
    "gaussian_process": GAUSSIAN_PROCESS,
    "external_tabular": EXTERNAL_TABULAR,
    "gap_edge_residual": GAP_EDGE_RESIDUAL,
    "tsicl_target_only": TSICL_TARGET_ONLY,
    "tsicl_satellite_proxy": TSICL_COVARIATE,
    "tsicl_physical_bundle": TSICL_COVARIATE,
}

METHOD_LABELS = {
    "persistence": "Persistence",
    "climatology": "Climatology",
    "linear_interpolation": "Linear interpolation",
    "gaussian_process": "Gaussian process",
    "external_tabular": "External tabular",
    "gap_edge_residual": "Gap-edge residual",
    "tsicl_target_only": "TS-ICL, target-only",
    "tsicl_satellite_proxy": "TS-ICL, + satellite chlorophyll",
    "tsicl_physical_bundle": "TS-ICL, + wind & SST",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": FAINT,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": False,
        "font.size": 10,
    }
)


def _shade_gap(ax, start, end, kind: str = "artificial") -> None:
    color = ARTIFICIAL_GAP_SHADE if kind == "artificial" else REAL_GAP_SHADE
    ax.axvspan(start, end, color=color, alpha=0.08, zorder=0)


def _style_date_axis(ax) -> None:
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_available_data(data: pd.DataFrame, gap_start, gap_end) -> tuple:
    """Four aligned panels: in-situ chlorophyll target, satellite chlorophyll
    proxy, wind speed, sea-surface temperature. Missing days are shown as gaps
    in the line, not interpolated for display."""
    gap_start, gap_end = pd.Timestamp(gap_start), pd.Timestamp(gap_end)
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)

    panels = [
        ("chl_mean", "Chlorophyll-a (mg/m$^3$)", OBSERVED, "A. In-situ target"),
        ("chl_satellite_proxy_log10", "Satellite Chl proxy\n(log10 scale)", EXTERNAL_TABULAR, "B. Satellite chlorophyll proxy"),
        ("wind_spd_ms", "Wind speed (m/s)", CLIMATOLOGY, "C. Wind speed"),
        ("sst_primary_degC", "SST (°C)", GAUSSIAN_PROCESS, "D. Sea-surface temperature"),
    ]
    for ax, (col, ylabel, color, title) in zip(axes, panels):
        ax.plot(data["date"], data[col], color=color, lw=1.1, marker="o", ms=1.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        _shade_gap(ax, gap_start, gap_end, "artificial")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    _style_date_axis(axes[-1])
    fig.tight_layout()
    return fig, axes


def plot_selected_interval(data: pd.DataFrame, gap_start, gap_end) -> tuple:
    """Full observed target with the demonstration interval highlighted; the
    target is still visible inside the highlighted region (not yet hidden)."""
    gap_start, gap_end = pd.Timestamp(gap_start), pd.Timestamp(gap_end)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(data["date"], data["chl_mean"], color=OBSERVED, lw=1.2, marker="o", ms=2)
    _shade_gap(ax, gap_start, gap_end, "artificial")
    ax.text(
        gap_start + (gap_end - gap_start) / 2,
        data["chl_mean"].max() * 1.05,
        "will be hidden temporarily",
        ha="center",
        va="bottom",
        fontsize=9,
        color=TRUTH,
    )
    ax.set_ylabel("Chlorophyll-a (mg/m$^3$)")
    ax.set_ylim(top=data["chl_mean"].max() * 1.2)
    _style_date_axis(ax)
    fig.tight_layout()
    return fig, ax


def plot_gap_creation(gap) -> tuple:
    """Before/after masking, plus covariate availability through the same
    interval: panel A shows the complete target, panel B shows what the models
    actually see (masked), panel C shows the satellite proxy remains available."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)

    ax = axes[0]
    ax.plot(gap.full_series["date"], gap.full_series[gap.target_column], color=OBSERVED, lw=1.2, marker="o", ms=2)
    _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
    ax.set_ylabel("Chlorophyll-a\n(mg/m$^3$)", fontsize=9)
    ax.set_title("A. Before masking -- withheld observations visible", loc="left", fontsize=10)

    ax = axes[1]
    ax.plot(gap.masked_series["date"], gap.masked_series[gap.target_column], color=OBSERVED, lw=1.2, marker="o", ms=2)
    _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
    ax.set_ylabel("Chlorophyll-a\n(mg/m$^3$)", fontsize=9)
    ax.set_title("B. After masking -- input target seen by every model", loc="left", fontsize=10)

    ax = axes[2]
    if "chl_satellite_proxy_log10" in gap.full_series.columns:
        ax.plot(
            gap.full_series["date"], gap.full_series["chl_satellite_proxy_log10"],
            color=EXTERNAL_TABULAR, lw=1.1, marker="o", ms=1.5,
        )
    _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
    ax.set_ylabel("Satellite Chl proxy\n(log10)", fontsize=9)
    ax.set_title("C. Covariates remain available through the gap", loc="left", fontsize=10)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    _style_date_axis(axes[-1])
    fig.tight_layout()
    return fig, axes


def _reconstruction_panel(ax, gap, result, color, label) -> None:
    """Shared drawing routine for a single reconstruction panel: context, gap
    shading, withheld truth (thin dashed), and one method's reconstruction.

    Plots from `masked_series` (NaN inside the gap), not `context` (gap rows
    dropped): matplotlib draws a straight line across a dropped-row gap, which
    would visually imply an observation bridging the hidden interval that does
    not exist. NaN rows produce an honest break in the line instead.
    """
    context = gap.masked_series
    ax.plot(context["date"], context[gap.target_column], color=OBSERVED, lw=1.0, marker="o", ms=1.8, zorder=2)
    ax.plot(gap.truth["date"], gap.truth["truth"], color=TRUTH, lw=1.2, ls="--", marker="o", ms=2.5, zorder=3, label="Withheld truth")
    if {"q05", "q95"}.issubset(result.prediction.columns):
        ax.fill_between(
            result.prediction["date"], result.prediction["q05"], result.prediction["q95"],
            color=color, alpha=UNCERTAINTY_ALPHA, zorder=1,
        )
    ax.plot(result.prediction["date"], result.prediction["value"], color=color, lw=2.0, zorder=4, label=label)
    _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_baseline_reconstructions(gap, baseline_results: dict) -> tuple:
    """Three side-by-side panels: persistence, climatology, linear interpolation.
    Same x/y axis range in every panel, one MAE value per panel title."""
    from .demo_helpers import mean_absolute_error

    order = ["persistence", "climatology", "linear_interpolation"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    y_min = min(gap.context[gap.target_column].min(), gap.truth["truth"].min()) * 0.9
    y_max = max(gap.context[gap.target_column].max(), gap.truth["truth"].max()) * 1.1

    for ax, name in zip(axes, order):
        result = baseline_results[name]
        color = METHOD_COLORS[name]
        _reconstruction_panel(ax, gap, result, color, METHOD_LABELS[name])
        mae = mean_absolute_error(result.prediction, gap.truth)
        ax.set_title(f"{METHOD_LABELS[name]}\nMAE = {mae:.2f} mg/m$^3$", fontsize=10)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(gap.gap_start - pd.Timedelta(days=25), gap.gap_end + pd.Timedelta(days=25))
        _style_date_axis(ax)

    axes[0].set_ylabel("Chlorophyll-a (mg/m$^3$)")
    fig.tight_layout()
    return fig, axes


def plot_gp_reconstruction(gap, gp_result) -> tuple:
    """One large panel: context, withheld truth, GP mean, and q05-q95 band."""
    from .demo_helpers import mean_absolute_error

    fig, ax = plt.subplots(figsize=(11, 4.2))
    _reconstruction_panel(ax, gap, gp_result, GAUSSIAN_PROCESS, METHOD_LABELS["gaussian_process"])
    mae = mean_absolute_error(gp_result.prediction, gap.truth)
    ax.set_xlim(gap.gap_start - pd.Timedelta(days=30), gap.gap_end + pd.Timedelta(days=30))
    ax.set_ylabel("Chlorophyll-a (mg/m$^3$)")
    ax.set_title(f"Gaussian process, target-only -- MAE = {mae:.2f} mg/m$^3$, runtime = {gp_result.runtime_s:.3f}s", loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    _style_date_axis(ax)
    fig.tight_layout()
    return fig, ax


def plot_tabular_inputs(gap, feature_table: pd.DataFrame) -> tuple:
    """Covariates plus calendar terms the tabular model receives, aligned with
    the hidden target interval."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    full = gap.full_series
    cols = [
        ("chl_satellite_proxy_log10", "Satellite Chl proxy", EXTERNAL_TABULAR),
        ("wind_spd_ms", "Wind speed (m/s)", CLIMATOLOGY),
        ("sst_primary_degC", "SST (°C)", GAUSSIAN_PROCESS),
    ]
    for ax, (col, ylabel, color) in zip(axes, cols):
        ax.plot(full["date"], full[col], color=color, lw=1.1, marker="o", ms=1.5)
        _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
        ax.set_ylabel(ylabel, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_title("Covariates available to the external tabular model (calendar terms not shown)", loc="left", fontsize=10)
    _style_date_axis(axes[-1])
    fig.tight_layout()
    return fig, axes


def plot_tabular_reconstruction(gap, tabular_result) -> tuple:
    from .demo_helpers import mean_absolute_error

    fig, ax = plt.subplots(figsize=(11, 4.0))
    _reconstruction_panel(ax, gap, tabular_result, EXTERNAL_TABULAR, METHOD_LABELS["external_tabular"])
    mae = mean_absolute_error(tabular_result.prediction, gap.truth)
    ax.set_xlim(gap.gap_start - pd.Timedelta(days=30), gap.gap_end + pd.Timedelta(days=30))
    ax.set_ylabel("Chlorophyll-a (mg/m$^3$)")
    ax.set_title(f"External tabular -- MAE = {mae:.2f} mg/m$^3$, runtime = {tabular_result.runtime_s:.3f}s", loc="left")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    _style_date_axis(ax)
    fig.tight_layout()
    return fig, ax


def plot_gap_edge_decomposition(gap, edge_result) -> tuple:
    """Three aligned panels: interpolation, predicted correction, corrected
    reconstruction -- so `corrected = interpolation + predicted_correction` is
    visible, not just stated."""
    decomposition = edge_result.extra["decomposition"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=False)

    ax = axes[0]
    context = gap.masked_series
    ax.plot(context["date"], context[gap.target_column], color=OBSERVED, lw=1.0, marker="o", ms=1.8)
    ax.plot(decomposition["date"], decomposition["interpolation"], color=INTERPOLATION, lw=2.0, label="Linear interpolation")
    _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
    ax.set_title("A. Linear interpolation across the gap", loc="left", fontsize=10)
    ax.set_ylabel("Chlorophyll-a\n(mg/m$^3$)", fontsize=9)
    ax.set_xlim(gap.gap_start - pd.Timedelta(days=20), gap.gap_end + pd.Timedelta(days=20))

    ax = axes[1]
    ax.axhline(0, color=FAINT, lw=0.8)
    ax.bar(decomposition["date"], decomposition["predicted_correction_log10"], color=GAP_EDGE_RESIDUAL, width=0.8)
    ax.set_title("B. Predicted residual correction (log10 scale)", loc="left", fontsize=10)
    ax.set_ylabel("Correction\n(log10)", fontsize=9)
    ax.set_xlim(gap.gap_start - pd.Timedelta(days=20), gap.gap_end + pd.Timedelta(days=20))

    ax = axes[2]
    ax.plot(context["date"], context[gap.target_column], color=OBSERVED, lw=1.0, marker="o", ms=1.8)
    ax.plot(gap.truth["date"], gap.truth["truth"], color=TRUTH, lw=1.2, ls="--", marker="o", ms=2.5, label="Withheld truth")
    ax.plot(decomposition["date"], decomposition["corrected"], color=GAP_EDGE_RESIDUAL, lw=2.0, label="Corrected reconstruction")
    _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
    ax.set_title("C. Corrected reconstruction = A + B", loc="left", fontsize=10)
    ax.set_ylabel("Chlorophyll-a\n(mg/m$^3$)", fontsize=9)
    ax.set_xlim(gap.gap_start - pd.Timedelta(days=20), gap.gap_end + pd.Timedelta(days=20))
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        _style_date_axis(ax)
    fig.tight_layout()
    return fig, axes


def plot_tsicl_inputs(gap) -> tuple:
    """Four aligned panels: masked target, satellite proxy, wind, SST -- the
    actual inputs available to TS-ICL for the three configurations run next."""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    masked = gap.masked_series
    panels = [
        (gap.target_column, "Chlorophyll-a\n(mg/m$^3$, masked)", OBSERVED),
        ("chl_satellite_proxy_log10", "Satellite Chl proxy", EXTERNAL_TABULAR),
        ("wind_spd_ms", "Wind speed (m/s)", CLIMATOLOGY),
        ("sst_primary_degC", "SST (°C)", GAUSSIAN_PROCESS),
    ]
    for ax, (col, ylabel, color) in zip(axes, panels):
        ax.plot(masked["date"], masked[col], color=color, lw=1.1, marker="o", ms=1.5)
        _shade_gap(ax, gap.gap_start, gap.gap_end, "artificial")
        ax.set_ylabel(ylabel, fontsize=9)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_title("TS-ICL inputs: target is masked, covariates remain available", loc="left", fontsize=10)
    _style_date_axis(axes[-1])
    fig.tight_layout()
    return fig, axes


def plot_tsicl_reconstructions(gap, tsicl_results: dict) -> tuple:
    """Three side-by-side panels, one per TS-ICL configuration, identical axes."""
    from .demo_helpers import mean_absolute_error

    order = ["tsicl_target_only", "tsicl_satellite_proxy", "tsicl_physical_bundle"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    y_min = min(gap.context[gap.target_column].min(), gap.truth["truth"].min()) * 0.9
    y_max = max(gap.context[gap.target_column].max(), gap.truth["truth"].max()) * 1.1

    for ax, name in zip(axes, order):
        result = tsicl_results[name]
        color = METHOD_COLORS[name]
        _reconstruction_panel(ax, gap, result, color, METHOD_LABELS[name])
        mae = mean_absolute_error(result.prediction, gap.truth)
        ax.set_title(f"{METHOD_LABELS[name]}\nMAE = {mae:.2f}, {result.runtime_s * 1000:.0f} ms", fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(gap.gap_start - pd.Timedelta(days=25), gap.gap_end + pd.Timedelta(days=25))
        _style_date_axis(ax)

    axes[0].set_ylabel("Chlorophyll-a (mg/m$^3$)")
    fig.tight_layout()
    return fig, axes


def plot_method_comparison(gap, results: dict, method_order: list[str]) -> tuple:
    """Small multiples (2x3), one panel per selected method, identical axes --
    deliberately not a single overlay of every method."""
    from .demo_helpers import mean_absolute_error

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    y_min = min(gap.context[gap.target_column].min(), gap.truth["truth"].min()) * 0.9
    y_max = max(gap.context[gap.target_column].max(), gap.truth["truth"].max()) * 1.1

    for ax, name in zip(axes.flat, method_order):
        result = results[name]
        color = METHOD_COLORS.get(name, INK)
        _reconstruction_panel(ax, gap, result, color, METHOD_LABELS.get(name, name))
        mae = mean_absolute_error(result.prediction, gap.truth)
        ax.set_title(f"{METHOD_LABELS.get(name, name)}\nMAE = {mae:.2f}", fontsize=9)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(gap.gap_start - pd.Timedelta(days=25), gap.gap_end + pd.Timedelta(days=25))
        _style_date_axis(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("Chlorophyll-a (mg/m$^3$)")
    fig.suptitle("One illustrative artificial gap -- not the full benchmark", fontsize=10, color=FAINT, y=1.01)
    fig.tight_layout()
    return fig, axes


def plot_mae_bars(mae_by_method: dict, runtime_by_method: dict | None = None) -> tuple:
    """Horizontal MAE bar chart, one bar per method, direct value labels."""
    order = sorted(mae_by_method, key=lambda k: mae_by_method[k])
    labels = [METHOD_LABELS.get(k, k) for k in order]
    values = [mae_by_method[k] for k in order]
    colors = [METHOD_COLORS.get(k, INK) for k in order]

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(order) + 1.2))
    y = np.arange(len(order))
    ax.barh(y, values, color=colors, height=0.6)
    for yi, v in zip(y, values):
        ax.text(v + max(values) * 0.01, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("MAE (mg/m$^3$) -- one illustrative artificial gap, not the full benchmark")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_real_gap_reconstruction(real_gap: pd.DataFrame, real_result) -> tuple:
    """Standalone figure for the real-gap application: main panel shows target
    context before/after plus the TS-ICL candidate reconstruction and its
    q05-q95 band; lower panel shows the satellite proxy through the same
    interval. No MAE is computed -- there is no truth to score against."""
    is_gap = real_gap["in_real_gap"]
    gap_start = real_gap.loc[is_gap, "date"].min()
    gap_end = real_gap.loc[is_gap, "date"].max()

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, height_ratios=[2.2, 1])

    ax = axes[0]
    context = real_gap.loc[~is_gap]
    ax.plot(context["date"], context["chl_mean"], color=OBSERVED, lw=1.2, marker="o", ms=2, label="Observed context")
    ax.fill_between(real_result.prediction["date"], real_result.prediction["q05"], real_result.prediction["q95"], color=TSICL_COVARIATE, alpha=UNCERTAINTY_ALPHA)
    ax.plot(real_result.prediction["date"], real_result.prediction["value"], color=TSICL_COVARIATE, lw=2.0, label="TS-ICL + satellite proxy (median)")
    _shade_gap(ax, gap_start, gap_end, "real")
    ax.set_ylabel("Chlorophyll-a (mg/m$^3$)")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title(
        "Real gap: 2015-07-01 to 2015-07-14 (genuinely missing, never observed)\n"
        "No withheld truth -- candidate reconstruction only",
        loc="left", fontsize=10,
    )

    ax = axes[1]
    ax.plot(real_gap["date"], real_gap["chl_satellite_proxy_log10"], color=EXTERNAL_TABULAR, lw=1.1, marker="o", ms=1.5)
    _shade_gap(ax, gap_start, gap_end, "real")
    ax.set_ylabel("Satellite Chl proxy\n(log10)", fontsize=9)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    _style_date_axis(axes[-1])
    fig.tight_layout()
    return fig, axes
