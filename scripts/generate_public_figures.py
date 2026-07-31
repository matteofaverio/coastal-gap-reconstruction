"""
generate_public_figures.py
==========================
Generates all public-facing figures for the coastal chlorophyll-a gap
reconstruction benchmark (Case Study 1: Tongoy Balsa, Chile).

Run from the repository root:
    python scripts/generate_public_figures.py

Requirements:
    matplotlib >= 3.7
    pandas >= 1.5
    numpy >= 1.23
"""

import os
import warnings

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore", category=FutureWarning)
matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------------------
# Paths (relative to the repository root as cwd)
# ---------------------------------------------------------------------------
DATA_DIR = "data_public/chlorophyll"
RESULTS_DIR = "results_public/chlorophyll"
FIG_DIR = "figures/chlorophyll"

os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# FIGURE 1 — Target record and missingness
# ---------------------------------------------------------------------------

def figure1_target_record():
    print("Generating Figure 1: target record and missingness...")
    df = pd.read_csv(
        os.path.join(DATA_DIR, "chlorophyll_daily_target.csv"),
        parse_dates=["date"],
    )
    gaps = pd.read_csv(
        os.path.join(DATA_DIR, "chlorophyll_real_gap_inventory.csv"),
        parse_dates=["start_date", "end_date"],
    )

    # Use raw chl_mean with a true log scale — NaN where not eligible so the
    # line breaks cleanly at every gap.
    df["chl_plot"] = np.where(
        df["target_eligible_default"] & df["chl_mean"].notna(),
        df["chl_mean"].clip(lower=1e-3),
        np.nan,
    )

    fig, ax = plt.subplots(figsize=(16, 5))

    # --- shade all real gaps in the background ---
    for _, row in gaps.iterrows():
        is_256 = (row["length_days"] >= 200)
        color = "#F4A432" if is_256 else "#DDDDDD"
        ax.axvspan(row["start_date"], row["end_date"], alpha=0.55, color=color, lw=0)

    # --- plot observed time series (full series; NaN breaks the line at gaps) ---
    ax.plot(
        df["date"], df["chl_plot"],
        color="#2E8B57", lw=0.8, alpha=0.85, zorder=3, label="Observed daily mean"
    )

    # True log scale — matplotlib handles tick placement and labels automatically
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    # --- annotate the 256-day gap (text only, no arrow) ---
    gap256 = gaps[gaps["length_days"] >= 200]
    if not gap256.empty:
        g = gap256.iloc[0]
        mid = g["start_date"] + (g["end_date"] - g["start_date"]) / 2
        ylo, yhi = ax.get_ylim()
        ax.text(
            mid, yhi * 0.4,
            "256-day gap\n(scenario-only)",
            fontsize=8.5, ha="center", va="center",
            color="#B85D00",
        )

    # --- legend patches ---
    patch_obs = Line2D([0], [0], color="#2E8B57", lw=1.5, label="Observed daily mean")
    patch_gap = mpatches.Patch(color="#DDDDDD", alpha=0.7, label="Real gap (127 gaps ≤ 71 days)")
    patch_256 = mpatches.Patch(color="#F4A432", alpha=0.7, label="Scenario-only gap (256 days)")
    ax.legend(handles=[patch_obs, patch_gap, patch_256], loc="upper left", framealpha=0.85)

    ax.set_title(
        "Daily mean chlorophyll-a, Tongoy Balsa (2015–2026)\n"
        "128 real gaps identified; 127 within validated range (1–71 days), 1 scenario-only (256 days)",
        pad=8,
    )
    ax.set_ylabel("Chlorophyll-a (mg m⁻³, log scale)")
    ax.set_xlabel("Date")
    ax.set_xlim(df["date"].min(), df["date"].max())

    fig.tight_layout()
    _save(fig, "figure_target_record_and_missingness.png")


# ---------------------------------------------------------------------------
# FIGURE 2 — Method benchmark summary (forest plot) — Method benchmark summary (forest plot)
# ---------------------------------------------------------------------------

def figure2_benchmark_forest():
    print("Generating Figure 2: benchmark forest plot...")
    bm = pd.read_csv(os.path.join(RESULTS_DIR, "chlorophyll_benchmark_summary.csv"))

    # Keep only the TS-ICL satellite-proxy rows (primary comparisons)
    sat_name = "TS-ICL (satellite chlorophyll proxy covariate)"
    sub = bm[bm["method_public_name"] == sat_name].copy()

    # Sort: most negative delta (best improvement) at top
    sub = sub.sort_values("delta")

    # Shorten comparator labels for readability
    label_map = {
        "Linear interpolation baseline": "Linear interpolation\n(baseline)",
        "Gaussian process (time-only)": "Gaussian process\n(time-only baseline)",
        "Engineered hybrid pipeline (validation-aware method assignment)":
            "Engineered hybrid pipeline\n(validation-aware assignment)",
        "TS-ICL (target-only, no covariates)": "TS-ICL\n(target-only, no covariates)",
    }
    sub["label"] = sub["compared_against_public_name"].map(
        lambda x: label_map.get(x, x)
    )

    sig_color = "#1A8C6A"   # teal-green for significant improvement
    insig_color = "#AAAAAA"
    degrad_color = "#CC4444"

    colors = []
    for _, row in sub.iterrows():
        if row["interpretation"] == "significant_improvement":
            colors.append(sig_color)
        elif row["interpretation"] == "significant_degradation":
            colors.append(degrad_color)
        else:
            colors.append(insig_color)

    n = len(sub)
    # Extra width so long y-tick labels don't crowd the plot; legend goes below.
    fig, ax = plt.subplots(figsize=(13, max(4, n * 1.1 + 2.5)))

    y = np.arange(n)
    ax.axvline(0, color="#444444", lw=1.2, zorder=1, ls="--")

    for i, (_, row) in enumerate(sub.iterrows()):
        c = colors[i]
        ax.plot([row["ci_lo"], row["ci_hi"]], [y[i], y[i]],
                color=c, lw=2.5, zorder=3)
        ax.plot(row["delta"], y[i], "o", color=c, ms=9, zorder=4,
                mec="white", mew=1.0)

    ax.set_yticks(y)
    ax.set_yticklabels(sub["label"].values, fontsize=10)

    # Pad x-axis so CI bars are never clipped at the edges.
    all_vals = list(sub["ci_lo"]) + list(sub["ci_hi"]) + [0]
    xpad = (max(all_vals) - min(all_vals)) * 0.18
    ax.set_xlim(min(all_vals) - xpad, max(all_vals) + xpad)

    # Pad y-axis so top/bottom rows aren't clipped.
    ax.set_ylim(-0.7, n - 0.3)

    ax.set_xlabel("Δ mean absolute error (TS-ICL satellite-proxy  −  comparator)\nNegative = TS-ICL satellite-proxy has lower error  ·  Bars = 95% bootstrap CI")
    ax.set_title(
        "TS-ICL satellite-proxy vs. comparators — mean absolute error difference\n"
        "(artificial-gap validation pool, n ≈ 681 gaps, 2 000 bootstrap replicates)",
        pad=8,
    )

    # Legend placed below the plot to avoid any overlap with data.
    leg_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=sig_color, ms=9,
               label="Significant improvement (95% CI entirely below 0)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=insig_color, ms=9,
               label="Directional, not statistically significant"),
    ]
    ax.legend(
        handles=leg_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        framealpha=0.88,
        fontsize=10,
    )

    ax.text(
        0.5, -0.26,
        "Validation-grade evidence: artificial gaps with withheld ground truth",
        transform=ax.transAxes,
        ha="center", va="bottom", fontsize=9, color="#555555",
        style="italic",
    )

    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22)
    _save(fig, "figure_method_benchmark_summary.png")


# ---------------------------------------------------------------------------
# FIGURE 3 — Gap-length performance
# ---------------------------------------------------------------------------

def figure3_gap_length():
    print("Generating Figure 3: gap-length performance...")
    scores = pd.read_csv(os.path.join(RESULTS_DIR, "chlorophyll_artificial_gap_scores.csv"))

    gl = scores[scores["stratum_type"] == "gap_length"].copy()

    # Define the methods to include (in order, with display names)
    method_spec = [
        ("Linear interpolation baseline",
         "Linear interpolation (baseline)", "#555555", "o", "--"),
        ("Gaussian process (time-only)",
         "Gaussian process (time-only)", "#7B3F9E", "s", "-."),
        ("Engineered hybrid pipeline (validation-aware method assignment)",
         "Engineered hybrid pipeline", "#2E6FA8", "^", "-"),
        ("TS-ICL (satellite chlorophyll proxy covariate)",
         "TS-ICL — satellite chlorophyll proxy covariate", "#1A8C6A", "D", "-"),
    ]

    # Parse numeric gap lengths
    def parse_len(s):
        return int(s.replace("L=", ""))

    fig, ax = plt.subplots(figsize=(11, 6))

    for raw_name, display_name, color, marker, ls in method_spec:
        sub = gl[gl["method_public_name"] == raw_name].copy()
        if sub.empty:
            print(f"    WARNING: method not found: {raw_name!r}")
            continue
        # Deduplicate by taking the first occurrence per stratum_value
        sub = sub.drop_duplicates(subset="stratum_value", keep="first")
        sub["len_num"] = sub["stratum_value"].map(parse_len)
        sub = sub.sort_values("len_num")
        ax.plot(
            sub["len_num"], sub["day_weighted_mae"],
            color=color, marker=marker, ls=ls, lw=2.0, ms=7,
            label=display_name, zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xticks([1, 3, 7, 10, 14, 21, 30, 45, 60])
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.set_xlabel("Gap length (days, log scale)")
    ax.set_ylabel("Day-weighted mean absolute error\n(log₁₀ chlorophyll scale)")
    ax.set_title(
        "Reconstruction error by gap length\n"
        "(artificial-gap validation pool; lower is better)",
        pad=8,
    )
    ax.legend(loc="upper left", framealpha=0.88)
    ax.text(
        0.99, 0.97,
        "MAE on withheld true values; lower is better",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9, color="#555555",
    )
    ax.grid(axis="y", ls=":", alpha=0.5)

    fig.tight_layout()
    _save(fig, "figure_gap_length_performance.png")


# ---------------------------------------------------------------------------
# FIGURE 4 — Event/high-chlorophyll limitation (lollipop)
# ---------------------------------------------------------------------------

def figure4_event_limitation():
    print("Generating Figure 4: event limitation lollipop...")
    ev = pd.read_csv(os.path.join(RESULTS_DIR, "chlorophyll_event_performance_summary.csv"))

    # Sort ascending by event penalty
    ev = ev.sort_values("event_penalty_mae_event_minus_nonevent").reset_index(drop=True)

    # Shorten method names for display
    def shorten(name):
        replacements = {
            "TS-ICL (satellite chlorophyll proxy covariate)": "TS-ICL — satellite proxy",
            "TS-ICL (target-only, no covariates)": "TS-ICL — target-only",
            "TS-ICL (wind/upwelling forcing covariate)": "TS-ICL — wind/upwelling",
            "TS-ICL (curated physical-forcing covariate set)": "TS-ICL — curated physical set",
            "TS-ICL (ocean current/transport covariate)": "TS-ICL — ocean transport",
            "TS-ICL (full redundant physical covariate set)": "TS-ICL — full physical set",
            "TS-ICL (satellite proxy + current/transport covariate)":
                "TS-ICL — satellite proxy + transport",
            "Engineered hybrid pipeline (validation-aware method assignment)":
                "Engineered hybrid pipeline",
            "Gap-edge residual model (deployed)": "Gap-edge residual model",
            "Gaussian process (time-only)": "Gaussian process (time-only)",
            "Linear interpolation baseline": "Linear interpolation (baseline)",
            "Monthly climatology baseline": "Monthly climatology (baseline)",
        }
        return replacements.get(name, name)

    ev["label"] = ev["method_public_name"].map(shorten)

    n = len(ev)
    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.8 + 1.5)))

    y = np.arange(n)
    penalty = ev["event_penalty_mae_event_minus_nonevent"].values

    ax.axvline(0, color="#444444", lw=1.2, ls="--", zorder=1)

    for i in range(n):
        c = "#E07020" if penalty[i] > 0 else "#1A8C6A"
        ax.hlines(y[i], 0, penalty[i], colors=c, lw=2.0, zorder=2)
        ax.plot(penalty[i], y[i], "o", color=c, ms=8, zorder=3,
                mec="white", mew=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(ev["label"].values, fontsize=9.5)
    ax.set_xlabel(
        "Event penalty = event-day MAE − non-event-day MAE\n"
        "Positive = higher error on high-chlorophyll days"
    )
    ax.set_title(
        "Event / high-chlorophyll limitation: all methods under-predict on event days\n"
        "(event days = chlorophyll above 90th percentile of observed values)",
        pad=8,
    )
    ax.text(
        0.99, 0.01,
        "Event days: every method shows higher error.\n"
        "Artificial-gap validation pool.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="#555555",
    )

    ax.invert_yaxis()
    ax.grid(axis="x", ls=":", alpha=0.4)
    fig.tight_layout()
    _save(fig, "figure_event_limitation.png")


# ---------------------------------------------------------------------------
# FIGURE 5 — Real-gap candidate example
# ---------------------------------------------------------------------------

def figure5_real_gap_example():
    print("Generating Figure 5: real-gap candidate example...")
    df = pd.read_csv(
        os.path.join(RESULTS_DIR, "chlorophyll_real_gap_candidate_outputs_daily.csv"),
        parse_dates=["date"],
    )

    # Exclude the 256-day scenario-only gap
    df = df[df["scenario_only_256day_gap"] == False].copy()

    # Choose the REAL_L031_20240325 window (43-day gap, good data on both sides)
    # Fallback: find any gap of 20-50 days with sufficient observations and both methods
    target_gap_id = "REAL_L031_20240325"
    gap_rows = df[df["real_gap_id"] == target_gap_id]

    if gap_rows.empty:
        # Fallback: find a good window
        print("    Target gap ID not found; searching for a suitable window...")
        found = None
        for gid in df[df["is_observed"] == False]["real_gap_id"].dropna().unique():
            rows = df[df["real_gap_id"] == gid]
            glen = rows["real_gap_length_days"].iloc[0]
            if 14 <= glen <= 50:
                gstart = rows["date"].min()
                gend = rows["date"].max()
                window = df[
                    (df["date"] >= gstart - pd.Timedelta(days=30))
                    & (df["date"] <= gend + pd.Timedelta(days=45))
                ]
                obs_count = window["is_observed"].sum()
                tsicl_ok = window["tsicl_satellite_proxy_pred_chl"].notna().sum()
                hybrid_ok = window["engineered_hybrid_reconstructed_chl"].notna().sum()
                if obs_count >= 30 and tsicl_ok >= glen * 0.7 and hybrid_ok >= glen * 0.7:
                    found = gid
                    break
        if found is None:
            print("    SKIPPING Figure 5: no suitable real-gap window found.")
            return
        target_gap_id = found
        gap_rows = df[df["real_gap_id"] == target_gap_id]

    gstart = gap_rows["date"].min()
    gend = gap_rows["date"].max()
    glen = int(gap_rows["real_gap_length_days"].iloc[0])

    # Window: 30 days before gap start to 45 days after gap end
    mask = (
        (df["date"] >= gstart - pd.Timedelta(days=30))
        & (df["date"] <= gend + pd.Timedelta(days=45))
    )
    window = df[mask].copy()

    gap = window[window["real_gap_id"] == target_gap_id]

    # Check that reconstruction columns exist and have data
    tsicl_col = "tsicl_satellite_proxy_pred_chl"
    hybrid_col = "engineered_hybrid_reconstructed_chl"
    if gap[tsicl_col].isna().all() and gap[hybrid_col].isna().all():
        print("    SKIPPING Figure 5: reconstruction columns are empty for chosen gap.")
        return

    # Build an observed series that is NaN during the gap, so matplotlib
    # breaks the line at the gap edges instead of connecting across them.
    obs_series = window["observed_chl_mean"].copy()
    obs_series[window["is_observed"] == False] = np.nan

    fig, ax = plt.subplots(figsize=(14, 5))

    # Shade gap period
    ax.axvspan(gstart, gend, alpha=0.12, color="#2E6FA8", lw=0, zorder=1,
               label="_nolegend_")
    ax.axvline(gstart, color="#2E6FA8", lw=0.8, ls=":", alpha=0.7)
    ax.axvline(gend, color="#2E6FA8", lw=0.8, ls=":", alpha=0.7)

    # Observed — full window series with NaN at gap, so the line breaks cleanly
    ax.plot(window["date"], obs_series, color="#222222",
            lw=1.4, zorder=4, label="Observed")

    # TS-ICL candidate (gap days only)
    ts_gap = gap[gap[tsicl_col].notna()]
    if not ts_gap.empty:
        ax.plot(ts_gap["date"], ts_gap[tsicl_col], color="#1A8C6A",
                lw=2.0, ls="-", zorder=3, marker="o", ms=3,
                label="TS-ICL satellite-proxy (candidate)")

    # Engineered hybrid candidate (gap days only)
    hy_gap = gap[gap[hybrid_col].notna()]
    if not hy_gap.empty:
        ax.plot(hy_gap["date"], hy_gap[hybrid_col], color="#2E6FA8",
                lw=2.0, ls="--", zorder=3, marker="s", ms=3,
                label="Engineered hybrid (candidate)")

    # Annotation
    ax.text(
        0.01, 0.97,
        "Candidate values only — real gaps have no withheld ground truth",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=9.5, color="#774400",
        bbox=dict(boxstyle="round,pad=0.3", fc="#FFF3CD", ec="#DDB80080", alpha=0.9),
    )
    ax.text(
        gstart + (gend - gstart) / 2,
        ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else 0.01,
        f"{glen}-day gap",
        ha="center", va="bottom", fontsize=9, color="#2E6FA8",
    )

    ax.set_title(
        "Candidate reconstructions over a real-gap example window\n"
        f"(gap ID: {target_gap_id}; gap length: {glen} days)"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Chlorophyll-a (mg m⁻³)")
    ax.legend(loc="upper right", framealpha=0.88)
    ax.grid(axis="y", ls=":", alpha=0.4)

    fig.tight_layout()

    # Fix the gap annotation y position after ylim is set
    # (replot text at sensible position)
    ymin, ymax = ax.get_ylim()
    ax.texts[-1].set_position(
        (gstart + (gend - gstart) / 2, ymin + (ymax - ymin) * 0.03)
    )

    _save(fig, "figure_real_gap_candidate_example.png")


# ---------------------------------------------------------------------------
# Covariate mechanism figure — decision
# ---------------------------------------------------------------------------
# The covariate arm MAE values span only 0.1981–0.2237 (< 0.04 range over
# 17 arms). A bar chart would show visually indistinguishable differences
# that require domain knowledge of the arm ordering to interpret.  A clean,
# self-explanatory public figure is not achievable from this data without
# misleading the reader.  This figure is intentionally skipped.
# ---------------------------------------------------------------------------

def _skip_covariate_figure():
    print(
        "  Skipping covariate mechanism figure: arm MAE range too narrow "
        "(0.1981–0.2237) for a clear public-facing bar chart."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Generating public figures ===")
    print(f"  Output directory: {os.path.abspath(FIG_DIR)}")
    print()

    figure1_target_record()
    figure2_benchmark_forest()
    figure3_gap_length()
    figure4_event_limitation()
    figure5_real_gap_example()
    _skip_covariate_figure()

    print()
    print("=== Done. ===")
    final = sorted(os.listdir(FIG_DIR))
    print(f"  Files in {FIG_DIR}:")
    for f in final:
        fp = os.path.join(FIG_DIR, f)
        size_kb = os.path.getsize(fp) // 1024
        print(f"    {f}  ({size_kb} KB)")
