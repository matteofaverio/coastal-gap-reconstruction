"""Builds the README overview figure (`assets/project_overview.png`) and a
2:1 social-preview variant (`assets/project_overview_social.png`).

Two panels, built entirely from this repository's own tracked data/asset
files -- no generated artwork, no decorative icons, no stock imagery, no
private-repository or internet access needed to run this script.

Left panel -- study site:
    - real satellite-derived Chl-a field + real coastline, rebuilt from
      original geospatial sources (not a rasterized crop of a report
      figure): `overview_site_map.npz`, a small pre-processed companion
      asset (smoothed Chl-a grid + a locally clipped coastline polygon)
      derived from the same private-repository sources
      `scripts/manuscript_figures/fig1_site_map.py` itself uses -- the
      real GlobColour Chl-a NetCDF extraction and the real Natural Earth
      10m land-polygon vector data. See "Regenerating overview_site_map.npz"
      below for exactly how, and why that one regeneration step needs the
      private repository rather than this one.
    - photo crop: manuscript/report/figures/fig1_site_context.png (the
      real "Tongoy buoy (BTG)" field photograph from the report's Figure 1
      source composite), shown as a small inset, not the dominant visual.
    - one small marker + label at the real Tongoy Balsa coordinate.

Right panel -- one artificial-gap validation example:
    - observed context + withheld truth: both real, read directly from
      data/chlorophyll/chlorophyll_daily_target.csv (an artificial gap
      hides real data from the *methods*, not from this repository's own
      target table -- the true values stay in that file throughout).
    - interpolation / TS-ICL / external-tabular reconstructions:
      `overview_gap_example.csv`, a small (14-row) frozen extract of this
      one selected gap's already-computed predictions -- not a live model
      run. See "Regenerating overview_gap_example.csv" below for the exact
      source caches and selection rule.
    - gap ID: L14_20240103 (2024-01-03 to 2024-01-16, 14 days), chosen
      because TS-ICL's MAE improvement over interpolation for this gap
      (+21.9%) sits at the 80th percentile among length-14 gaps with
      complete truth/interpolation/TS-ICL/external-tabular support --
      clearly better than a typical gap without being the single most
      flattering case available.

Regenerating overview_site_map.npz (only needed if the underlying
satellite snapshot or coastline source changes -- not needed to just
rebuild the PNGs from the already-committed companion assets):
    Run, from the PRIVATE repository root (needs xarray/scipy and an
    ephemeral pyshp; reads the real GlobColour NetCDF extraction and
    cartopy's local Natural Earth shapefile cache -- none of which are
    available in this public repository's own environment):
        uv run --with pyshp python <path to>/extract_site_map_a4.py
    then copy the small (~110KB) output over this file.

Regenerating overview_gap_example.csv: re-select a gap from a local,
gitignored full-population prediction cache (not part of this public
repository -- see the private repository's benchmark tooling), then write
that gap's date/interpolation/TS-ICL/external-tabular rows in the same
four-column format as the current file.

Run from the repository root:
    python experiments/visualization/build_project_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)
THIS_DIR = Path(__file__).resolve().parent

REPORT_FIGURES = REPO_ROOT / "manuscript/report/figures"
SITE_MAP_DATA = THIS_DIR / "overview_site_map.npz"
GAP_EXAMPLE_DATA = THIS_DIR / "overview_gap_example.csv"
DAILY_TARGET = REPO_ROOT / "data/chlorophyll/chlorophyll_daily_target.csv"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "axes.edgecolor": "#4d4d4d",
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

COLOR_TRUTH = "#111111"       # withheld truth
COLOR_CONTEXT = "#2f5f45"     # observed context
COLOR_INTERP = "#8c8c8c"      # linear interpolation
COLOR_TSICL = "#a3121a"       # TS-ICL
COLOR_EXT_TABULAR = "#2f6fa8"  # external-predictors-only tabular model
COLOR_GAP_SHADE = "#f4dfa0"   # withheld-interval background
TEXT_DARK = "#1a1a1a"
TEXT_MUTED = "#555555"

OCEAN_COLOR = "#cfe3ee"
LAND_COLOR = "#ddd2b0"
COAST_COLOR = "#3a3a3a"


# ---------------------------------------------------------------------------
# Left panel -- study site
# ---------------------------------------------------------------------------

def _load_buoy_photo() -> Image.Image:
    path = REPORT_FIGURES / "fig1_site_context.png"
    if not path.exists():
        raise FileNotFoundError(f"Expected site-context source figure at {path}")
    im = Image.open(path)
    return im.crop((2790, 130, 3675, 748)).convert("RGB")


def _fig_aspect_correct_height(fig, width_fig: float, target_h_over_w: float) -> float:
    fig_w_in, fig_h_in = fig.get_size_inches()
    return width_fig * target_h_over_w * (fig_w_in / fig_h_in)


def build_left_panel(fig, ax) -> tuple:
    if not SITE_MAP_DATA.exists():
        raise FileNotFoundError(
            f"Missing {SITE_MAP_DATA} -- see this module's docstring, "
            "'Regenerating overview_site_map.npz'."
        )
    d = np.load(SITE_MAP_DATA)
    lon_hi, lat_hi, smooth = d["lon_hi"], d["lat_hi"], d["chl_smooth"]
    n_rings = int(d["n_rings"])
    rings = [d[f"ring_{i}"] for i in range(n_rings)]
    lon_min, lon_max, lat_min, lat_max = d["padded_bbox"]
    tlat, tlon = float(d["tongoy_lat"]), float(d["tongoy_lon"])

    ax.add_patch(Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                            facecolor=OCEAN_COLOR, zorder=0))
    ax.pcolormesh(lon_hi, lat_hi, smooth, cmap="viridis", shading="auto",
                   vmin=-0.6, vmax=1.65, zorder=1)
    for ring in rings:
        ax.add_patch(Polygon(ring, closed=True, facecolor=LAND_COLOR,
                              edgecolor=COAST_COLOR, linewidth=1.0, zorder=2))

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)
    # Plain equal aspect on raw lon/lat degrees -- the same PlateCarree
    # convention (no extra cos(latitude) compression) the source figure
    # scripts/manuscript_figures/fig1_site_map.py renders with.
    ax.set_aspect("equal")
    ax.axis("off")

    ax.scatter([tlon], [tlat], s=90, color="white", zorder=4)
    ax.scatter([tlon], [tlat], s=42, color=COLOR_TSICL, edgecolor="white", linewidth=1.1, zorder=5)
    ax.annotate("Tongoy Balsa", (tlon, tlat), xytext=(9, 3), textcoords="offset points",
                fontsize=11, fontweight="bold", color=TEXT_DARK, va="center", zorder=5)

    km_per_deg_lon = 111.32 * np.cos(np.radians(tlat))
    bar_km = 20.0
    bar_deg = bar_km / km_per_deg_lon
    bar_x0, bar_y0 = lon_min + 0.06, lat_min + 0.06
    ax.plot([bar_x0, bar_x0 + bar_deg], [bar_y0, bar_y0], color=TEXT_DARK, linewidth=2.0, zorder=6)
    ax.annotate(f"{bar_km:.0f} km", (bar_x0 + bar_deg / 2, bar_y0), xytext=(0, 4),
                textcoords="offset points", ha="center", fontsize=9, color=TEXT_DARK, zorder=6)

    fig.canvas.draw()
    pos_map = ax.get_position()

    photo = _load_buoy_photo()
    photo_ratio = photo.height / photo.width
    inset_w = pos_map.width * 0.30
    inset_h = _fig_aspect_correct_height(fig, inset_w, photo_ratio)
    margin = pos_map.width * 0.04
    inset_x = pos_map.x1 - inset_w - margin
    inset_y = pos_map.y0 + margin
    ax_photo = fig.add_axes([inset_x, inset_y, inset_w, inset_h])
    ax_photo.imshow(photo)
    ax_photo.axis("off")
    for spine in ax_photo.spines.values():
        spine.set_visible(True)
        spine.set_color("white")
        spine.set_linewidth(2.2)

    return pos_map


# ---------------------------------------------------------------------------
# Right panel -- one artificial-gap validation example
# ---------------------------------------------------------------------------

def load_gap_example() -> dict:
    if not GAP_EXAMPLE_DATA.exists():
        raise FileNotFoundError(
            f"Missing {GAP_EXAMPLE_DATA} -- see this module's docstring, "
            "'Regenerating overview_gap_example.csv'."
        )
    recon = pd.read_csv(GAP_EXAMPLE_DATA, parse_dates=["date"])
    gap_start, gap_end = recon["date"].min(), recon["date"].max()

    target = pd.read_csv(DAILY_TARGET, parse_dates=["date"]).set_index("date").sort_index()
    context_start = gap_start - pd.Timedelta(days=18)
    context_end = gap_end + pd.Timedelta(days=18)
    context = target.loc[context_start:context_end, "chl_mean"]

    return {
        "gap_start": gap_start,
        "gap_end": gap_end,
        "context_dates": context.index,
        "context_values": context.to_numpy(),
        "recon_dates": recon["date"],
        "interp_physical": recon["interpolation_chl_mean"].to_numpy(),
        "tsicl_physical": recon["tsicl_satellite_proxy_chl_mean"].to_numpy(),
        "ext_tabular_physical": recon["ext_tabular_hgb_chl_mean"].to_numpy(),
    }


def build_right_panel(ax, example: dict) -> int:
    ax.axvspan(example["gap_start"], example["gap_end"], color=COLOR_GAP_SHADE, alpha=0.18, lw=0, zorder=0)

    ctx_dates = example["context_dates"]
    ctx_vals = example["context_values"].astype(float).copy()
    in_gap = (ctx_dates >= example["gap_start"]) & (ctx_dates <= example["gap_end"])
    ctx_vals[in_gap] = np.nan
    n_plotted_in_gap = int(np.sum(~np.isnan(ctx_vals[in_gap])))
    assert n_plotted_in_gap == 0, (
        f"Observed-context series must be entirely NaN inside the withheld "
        f"interval; found {n_plotted_in_gap} non-NaN value(s)."
    )
    ax.plot(ctx_dates, ctx_vals, color=COLOR_CONTEXT, lw=1.6)

    # Withheld truth is the same real target series, just restricted to the
    # gap dates.
    truth = pd.read_csv(DAILY_TARGET, parse_dates=["date"]).set_index("date").sort_index()
    truth = truth.loc[example["recon_dates"], "chl_mean"].to_numpy()

    ax.plot(example["recon_dates"], truth, color=COLOR_TRUTH, lw=1.9)
    ax.plot(example["recon_dates"], example["interp_physical"], color=COLOR_INTERP, lw=1.4, ls="--")
    ax.plot(example["recon_dates"], example["ext_tabular_physical"], color=COLOR_EXT_TABULAR, lw=1.5, ls=":")
    ax.plot(example["recon_dates"], example["tsicl_physical"], color=COLOR_TSICL, lw=1.9)

    legend_handles = [
        Line2D([0], [0], color=COLOR_CONTEXT, lw=1.6, label="observed context"),
        Line2D([0], [0], color=COLOR_TRUTH, lw=1.9, label="withheld truth"),
        Line2D([0], [0], color=COLOR_TSICL, lw=1.9, label="TS-ICL"),
        Line2D([0], [0], color=COLOR_INTERP, lw=1.4, ls="--", label="interpolation"),
        Line2D([0], [0], color=COLOR_EXT_TABULAR, lw=1.5, ls=":", label="external tabular"),
    ]
    ax.legend(
        handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 1.02),
        ncol=5, frameon=False, fontsize=9.6, handlelength=1.7, columnspacing=1.15,
        borderaxespad=0,
    )

    ax.set_ylabel("Chlorophyll-a (mg m$^{-3}$)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.margins(x=0.04)

    # Axis range set from the informative lines only, not from an
    # uncertainty band (none is shown here) -- keeps all five series using
    # the full plot height.
    line_values = np.concatenate([
        ctx_vals[~np.isnan(ctx_vals)], truth,
        example["interp_physical"], example["tsicl_physical"], example["ext_tabular_physical"],
    ])
    lo, hi = float(np.min(line_values)), float(np.max(line_values))
    pad = 0.12 * (hi - lo)
    ax.set_ylim(max(0.0, lo - pad), hi + pad)
    return n_plotted_in_gap


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build(output_name: str, figsize: tuple[float, float], dpi: int) -> Path:
    example = load_gap_example()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[0.40, 0.60],
                               left=0.035, right=0.975, top=0.82, bottom=0.16, wspace=0.08)

    ax_map = plt.subplot(outer[0])
    pos_map = build_left_panel(fig, ax_map)

    ax_gap = plt.subplot(outer[1])
    build_right_panel(ax_gap, example)

    fig.canvas.draw()
    pos_gap = ax_gap.get_position()

    title_y = 0.92
    fig.text((pos_map.x0 + pos_map.x1) / 2, title_y, "Study site",
              ha="center", fontsize=14.5, fontweight="bold", color=TEXT_DARK)
    fig.text((pos_gap.x0 + pos_gap.x1) / 2, title_y, "Artificial-gap validation",
              ha="center", fontsize=14.5, fontweight="bold", color=TEXT_DARK)

    caption_y = min(pos_map.y0, pos_gap.y0) - 0.10
    fig.text((pos_map.x0 + pos_map.x1) / 2, caption_y, "Tongoy Balsa buoy · north-central Chile",
              ha="center", fontsize=11.5, color=TEXT_MUTED)
    fig.text((pos_gap.x0 + pos_gap.x1) / 2, caption_y,
              "Observed values are withheld, reconstructed, and then used to score the methods.",
              ha="center", fontsize=10.5, color=TEXT_MUTED)

    out_path = ASSETS_DIR / output_name
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path} ({fig.get_size_inches()[0]*dpi:.0f}x{fig.get_size_inches()[1]*dpi:.0f}px)")
    return out_path


def main() -> None:
    build("project_overview.png", figsize=(16, 6.2), dpi=150)
    build("project_overview_social.png", figsize=(16, 8.0), dpi=150)


if __name__ == "__main__":
    main()
