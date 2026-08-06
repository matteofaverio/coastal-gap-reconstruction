"""Builds the README overview figure (`assets/project_overview.png`) and a
2:1 social-preview variant (`assets/project_overview_social.png`).

One horizontal, three-panel scientific figure, built entirely from this
repository's own frozen data/result tables and the report's own site
photographs/map -- no generated artwork, no decorative icons, no stock
imagery.

Panel A -- study site and observations:
    - map crop: manuscript/poster/figures/fig1_site_map.png (Panel B of
      that source figure: Tongoy Bay coastline, real GlobColour Chl-a
      satellite field, BTG/PLV station markers, cartopy coastline).
    - photo crop: manuscript/report/figures/fig1_site_context.png (the
      "Tongoy buoy (BTG)" field photograph panel of that source figure).
    - record span read from data/chlorophyll/chlorophyll_daily_target.csv
      and data/oxygen/oxygen_daily_target.csv.

Panel B -- validation design:
    - schematic: drawn from the same logic as report Figure 8 (artificial
      gap = hidden + compared against withheld truth = benchmark evidence;
      real gap = never observed = candidate only), not a reuse of the
      report's TikZ source.
    - example: demo/data/chlorophyll_demo_series.csv (observed context +
      withheld truth, physical mg/m^3) and
      demo/outputs/demo_reconstruction_results.csv (linear-interpolation
      and TS-ICL satellite-proxy reconstructions + q05/q95, same units --
      genuine output of `demo/run_demo.sh`, not synthesized here). The
      illustrative gap is 2017-04-21 to 2017-05-04 (14 days).

Panel C -- cross-case benchmark results (two small multiples, separate
units, never a shared y-axis):
    - chlorophyll: results/chlorophyll/chlorophyll_matched_support_by_length.csv
      (canonical_interpolation, ext_tabular_extratrees, tsicl_satellite_proxy;
      gap lengths 1/3/7/14/30 days; log10 chlorophyll MAE).
    - oxygen: results/oxygen/oxygen_benchmark_by_length.csv (Linear
      interpolation, External tabular (ExtraTrees), TS-ICL physical
      covariates; gap lengths 1/3/7/10/14/21/30 days; mg/L MAE).
    - pooled percent-improvement annotations computed from
      results/chlorophyll/chlorophyll_benchmark_summary.csv and
      results/oxygen/oxygen_paired_deltas_vs_tsicl_physical_covariates.csv.

Run from the repository root:
    python experiments/visualization/build_project_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

REPORT_FIGURES = REPO_ROOT / "manuscript/report/figures"
POSTER_FIGURES = REPO_ROOT / "manuscript/poster/figures"

plt.rcParams.update({
    "font.size": 10.5,
    "axes.labelsize": 10,
    "legend.fontsize": 8.7,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "grid.color": "#dddddd",
    "grid.linewidth": 0.5,
})

# One consistent colour per role, reused across every panel.
COLOR_CHL = "#2c6e49"        # chlorophyll identity (green)
COLOR_OX = "#1d6fa5"         # oxygen identity (blue)
COLOR_TSICL = "#c1121f"      # TS-ICL accent, every panel
COLOR_INTERP = "#8c8c8c"     # linear interpolation, every panel
COLOR_TABULAR = "#6b4c9a"    # external tabular model, every panel
COLOR_GAP = "#f2c14e"        # withheld/missing interval shading
COLOR_TEXT = "#222222"


# ---------------------------------------------------------------------------
# Panel A -- study site and observations
# ---------------------------------------------------------------------------

def _load_site_map_crop() -> Image.Image:
    """Panel B of the poster's own site-map figure: Tongoy Bay coastline,
    real GlobColour Chl-a satellite field, BTG/PLV station markers. The
    source figure's own "B" panel-letter badge is painted over with white
    so it cannot be confused with this figure's own A/B/C lettering."""
    path = POSTER_FIGURES / "fig1_site_map.png"
    if not path.exists():
        raise FileNotFoundError(f"Expected site-map source figure at {path}")
    im = Image.open(path)
    crop = im.crop((1560, 130, 3350, 1560)).convert("RGB")
    from PIL import ImageDraw
    ImageDraw.Draw(crop).rectangle([0, 60, 300, 180], fill="white")
    return crop


def _load_buoy_photo_crop() -> Image.Image:
    """The real "Tongoy buoy (BTG)" field photograph from the report's
    Figure 1 source composite."""
    path = REPORT_FIGURES / "fig1_site_context.png"
    if not path.exists():
        raise FileNotFoundError(f"Expected site-context source figure at {path}")
    im = Image.open(path)
    return im.crop((2800, 150, 3670, 745)).convert("RGB")


def build_panel_a(gs_cell) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, height_ratios=[1.35, 1], hspace=0.08)
    ax_map = plt.subplot(inner[0])
    ax_photo = plt.subplot(inner[1])

    ax_map.imshow(_load_site_map_crop())
    ax_map.axis("off")

    ax_photo.imshow(_load_buoy_photo_crop())
    ax_photo.axis("off")

    chl = pd.read_csv(REPO_ROOT / "data/chlorophyll/chlorophyll_daily_target.csv", parse_dates=["date"])
    ox = pd.read_csv(REPO_ROOT / "data/oxygen/oxygen_daily_target.csv", parse_dates=["date"])
    span_start = min(chl["date"].min(), ox["date"].min()).strftime("%Y")
    span_end = max(chl["date"].max(), ox["date"].max()).strftime("%Y")
    ax_photo.text(
        0.5, -0.05, "Tongoy Balsa buoy, north-central Chile",
        transform=ax_photo.transAxes, ha="center", va="top",
        fontsize=9.5, color=COLOR_TEXT,
    )
    ax_photo.text(
        0.5, -0.14,
        f"Measured: chlorophyll-a & dissolved oxygen, {span_start}–{span_end}",
        transform=ax_photo.transAxes, ha="center", va="top",
        fontsize=8.7, color="#555555",
    )


# ---------------------------------------------------------------------------
# Panel B -- validation design
# ---------------------------------------------------------------------------

def _draw_gap_row(ax, y: float, label: str, segments: list[tuple[float, float, str]], outcome: str, outcome_color: str) -> None:
    """One horizontal timeline row: a sequence of (start, width, kind)
    segments ("obs" | "hidden" | "missing"), a row label above the bar,
    and an outcome label below it -- both kept inside the row's own data
    coordinates so nothing can spill into a neighbouring panel."""
    height = 0.30
    for start, width, kind in segments:
        if kind == "obs":
            ax.add_patch(Rectangle((start, y - height / 2), width, height, facecolor=COLOR_CHL, edgecolor="none", alpha=0.85))
        elif kind == "hidden":
            ax.add_patch(Rectangle((start, y - height / 2), width, height, facecolor=COLOR_GAP, edgecolor=COLOR_TSICL, lw=1.1, hatch="////", alpha=0.9))
        else:  # missing, never observed
            ax.add_patch(Rectangle((start, y - height / 2), width, height, facecolor="white", edgecolor="#888888", lw=1.1, linestyle=(0, (2, 1.5))))
    ax.text(0, y + height / 2 + 0.10, label, ha="left", va="bottom", fontsize=9.3, fontweight="bold", color=COLOR_TEXT)
    ax.text(0, y - height / 2 - 0.12, outcome, ha="left", va="top", fontsize=9.0, color=outcome_color)


def build_panel_b_schematic(ax) -> None:
    ax.set_xlim(0, 9)
    ax.set_ylim(-0.55, 2.35)
    _draw_gap_row(
        ax, 1.75, "Artificial gap",
        [(0, 3.2, "obs"), (3.2, 2.6, "hidden"), (5.8, 3.2, "obs")],
        "hidden, then compared with withheld truth → benchmark evidence", COLOR_TSICL,
    )
    _draw_gap_row(
        ax, 0.55, "Real gap",
        [(0, 3.2, "obs"), (3.2, 2.6, "missing"), (5.8, 3.2, "obs")],
        "never observed, nothing to compare → candidate only", "#666666",
    )
    ax.axis("off")
    ax.set_title("Artificial gaps can be scored; real gaps cannot", fontsize=9.8, color=COLOR_TEXT, pad=4)


def build_panel_b_example(ax) -> None:
    series = pd.read_csv(REPO_ROOT / "demo/data/chlorophyll_demo_series.csv", parse_dates=["date"])
    recon = pd.read_csv(REPO_ROOT / "demo/outputs/demo_reconstruction_results.csv", parse_dates=["date"])
    recon = recon[recon["artificial_or_real_gap"] == "artificial"]

    gap_dates = series.loc[series["in_artificial_gap"], "date"]
    gap_start, gap_end = gap_dates.min(), gap_dates.max()

    observed = series[~series["in_artificial_gap"]]
    withheld = series[series["in_artificial_gap"]]

    ax.axvspan(gap_start, gap_end, color=COLOR_GAP, alpha=0.4, lw=0, label="Withheld interval")
    ax.plot(observed["date"], observed["chl_mean"], color=COLOR_CHL, lw=1.4, label="Observed context")
    ax.plot(withheld["date"], withheld["chl_mean"], color=COLOR_TEXT, lw=1.7, label="Withheld truth")

    interp = recon[recon["method"] == "linear_interpolation"].set_index("date")
    tsicl = recon[recon["method"] == "tsicl_satellite_proxy"].set_index("date")

    ax.fill_between(tsicl.index, tsicl["q05"], tsicl["q95"], color=COLOR_TSICL, alpha=0.18, lw=0, label="TS-ICL q05–q95")
    ax.plot(interp.index, interp["reconstructed_median"], color=COLOR_INTERP, lw=1.3, ls="--", label="Linear interpolation")
    ax.plot(tsicl.index, tsicl["reconstructed_median"], color=COLOR_TSICL, lw=1.6, label="TS-ICL (satellite proxy)")

    ax.set_ylabel("Chlorophyll-a (mg m$^{-3}$)")
    ax.legend(loc="upper left", frameon=False, fontsize=8.6, ncol=1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.set_title("One real 14-day artificial-gap example", fontsize=9.8, color=COLOR_TEXT, pad=6)


def build_panel_b(gs_cell) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, height_ratios=[0.62, 1], hspace=0.5)
    build_panel_b_schematic(plt.subplot(inner[0]))
    build_panel_b_example(plt.subplot(inner[1]))


# ---------------------------------------------------------------------------
# Panel C -- cross-case benchmark results
# ---------------------------------------------------------------------------

def _chlorophyll_pooled_pct() -> float:
    chl = pd.read_csv(REPO_ROOT / "results/chlorophyll/chlorophyll_benchmark_summary.csv")
    row = chl[
        (chl["method_public_name"] == "TS-ICL (satellite chlorophyll proxy covariate)")
        & (chl["compared_against_public_name"] == "Linear interpolation baseline")
    ].iloc[0]
    return -row["delta"] / row["comparison_value"] * 100


def _oxygen_pooled_pct() -> float:
    by_length = pd.read_csv(REPO_ROOT / "results/oxygen/oxygen_benchmark_by_length.csv")
    deltas = pd.read_csv(REPO_ROOT / "results/oxygen/oxygen_paired_deltas_vs_tsicl_physical_covariates.csv")
    interp = by_length[by_length["method_label"] == "Linear interpolation"]
    interp_mae = (interp["mae_gapweighted"] * interp["n_gaps"]).sum() / interp["n_gaps"].sum()
    delta = deltas.loc[deltas["comparator_id"] == "linear_interp", "delta_tsicl_minus_comparator_mae"].iloc[0]
    return -delta / interp_mae * 100


def build_panel_c_chlorophyll(ax) -> None:
    df = pd.read_csv(REPO_ROOT / "results/chlorophyll/chlorophyll_matched_support_by_length.csv")
    lengths = [1, 3, 7, 14, 30]
    methods = [
        ("canonical_interpolation", "Linear interpolation", COLOR_INTERP, "--"),
        ("ext_tabular_extratrees", "External tabular", COLOR_TABULAR, "-"),
        ("tsicl_satellite_proxy", "TS-ICL (satellite proxy)", COLOR_TSICL, "-"),
    ]
    for method_id, label, color, ls in methods:
        sub = df[(df["method_id"] == method_id) & (df["gap_length"].isin(lengths))].sort_values("gap_length")
        ax.plot(sub["gap_length"], sub["mae_day_weighted"], color=color, ls=ls, marker="o", ms=3.5, lw=1.5, label=label)

    pct = _chlorophyll_pooled_pct()
    ax.set_xticks(lengths)
    ax.set_xlabel("Gap length (days)")
    ax.set_ylabel("MAE (log$_{10}$ chlorophyll-a)", color=COLOR_CHL)
    ax.tick_params(axis="y", labelcolor=COLOR_CHL)
    ax.legend(loc="upper left", frameon=False, fontsize=8.6)
    ax.text(
        0.98, 0.05, f"{pct:+.1f}% lower pooled MAE (chlorophyll)\nLower MAE is better.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, color=COLOR_TEXT,
    )
    ax.set_title("Chlorophyll: TS-ICL is strongest overall;\nhigh-Chl events remain difficult", fontsize=9.0, color=COLOR_TEXT, pad=4, linespacing=1.3)


def build_panel_c_oxygen(ax) -> None:
    df = pd.read_csv(REPO_ROOT / "results/oxygen/oxygen_benchmark_by_length.csv")
    lengths = [1, 3, 7, 10, 14, 21, 30]
    methods = [
        ("Linear interpolation", COLOR_INTERP, "--"),
        ("External tabular (ExtraTrees)", COLOR_TABULAR, "-"),
        ("TS-ICL physical covariates", COLOR_TSICL, "-"),
    ]
    for label, color, ls in methods:
        sub = df[(df["method_label"] == label) & (df["gap_length"].isin(lengths))].sort_values("gap_length")
        ax.plot(sub["gap_length"], sub["mae_gapweighted"], color=color, ls=ls, marker="o", ms=3.5, lw=1.5, label=label)

    pct = _oxygen_pooled_pct()
    ax.set_xticks(lengths)
    ax.set_xlabel("Gap length (days)")
    ax.set_ylabel("MAE (mg/L)", color=COLOR_OX)
    ax.tick_params(axis="y", labelcolor=COLOR_OX)
    ax.legend(loc="upper left", frameon=False, fontsize=8.6)
    ax.text(
        0.98, 0.05, f"{pct:+.1f}% lower pooled MAE (oxygen)\nLower MAE is better.",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.6, color=COLOR_TEXT,
    )
    ax.set_title("Oxygen: TS-ICL improves the pooled result;\ndistribution tails remain difficult", fontsize=9.0, color=COLOR_TEXT, pad=4, linespacing=1.3)


def build_panel_c(gs_cell) -> None:
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_cell, hspace=0.78)
    build_panel_c_chlorophyll(plt.subplot(inner[0]))
    build_panel_c_oxygen(plt.subplot(inner[1]))


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def _label_column(fig, x_center: float, letter: str, subtitle: str) -> None:
    fig.text(x_center, 0.975, letter, ha="center", va="top", fontsize=15, fontweight="bold", color=COLOR_TEXT)
    fig.text(x_center, 0.935, subtitle, ha="center", va="top", fontsize=9.6, color="#555555")


def build(output_name: str, figsize: tuple[float, float], dpi: int) -> Path:
    fig = plt.figure(figsize=figsize, dpi=dpi)
    outer = gridspec.GridSpec(
        1, 3, figure=fig, width_ratios=[1.0, 1.15, 1.3],
        left=0.045, right=0.985, top=0.84, bottom=0.085, wspace=0.28,
    )

    build_panel_a(outer[0])
    build_panel_b(outer[1])
    build_panel_c(outer[2])

    col_centers = [0.045 + (0.985 - 0.045) * f for f in (0.14, 0.475, 0.85)]
    _label_column(fig, col_centers[0], "A", "Study site and observations")
    _label_column(fig, col_centers[1], "B", "Validation design")
    _label_column(fig, col_centers[2], "C", "What the two case studies show")

    out_path = ASSETS_DIR / output_name
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path} ({fig.get_size_inches()[0]*dpi:.0f}x{fig.get_size_inches()[1]*dpi:.0f}px)")
    return out_path


def main() -> None:
    build("project_overview.png", figsize=(14.2, 6.6), dpi=170)
    build("project_overview_social.png", figsize=(13.6, 6.8), dpi=175)


if __name__ == "__main__":
    main()
