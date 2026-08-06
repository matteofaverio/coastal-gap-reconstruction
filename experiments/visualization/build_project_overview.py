"""Builds the README overview figure (`assets/project_overview.png`) and a
cropped social-preview variant (`assets/project_overview_social.png`) from
already-released public data and result tables only.

Three panels:
  1. The observed chlorophyll record, with real (naturally occurring) gaps
     highlighted -- data/chlorophyll/chlorophyll_daily_target.csv,
     chlorophyll_real_gap_inventory.csv.
  2. The artificial-gap validation idea: one real illustrative gap from the
     live demo, showing observed context, the withheld truth, and two
     candidate reconstructions -- demo/outputs/demo_reconstruction_results.csv
     (genuine output from `demo/run_demo.sh`, not synthesized here).
  3. The main benchmark message: percent MAE improvement over linear
     interpolation for the leading TS-ICL configuration in each case study
     -- results/chlorophyll/chlorophyll_benchmark_summary.csv,
     results/oxygen/oxygen_benchmark_by_length.csv,
     oxygen_paired_deltas_vs_tsicl_physical_covariates.csv.

Run from the repository root:
    python experiments/visualization/build_project_overview.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10.5,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 170,
})

COLOR_OBSERVED = "#2c6e49"
COLOR_GAP = "#e0a800"
COLOR_TRUTH = "#1f1f1f"
COLOR_INTERP = "#8c8c8c"
COLOR_TSICL = "#c1121f"


def panel1_target_and_gaps(ax):
    target = pd.read_csv(
        REPO_ROOT / "data/chlorophyll/chlorophyll_daily_target.csv", parse_dates=["date"]
    ).set_index("date").sort_index()
    real_gaps = pd.read_csv(
        REPO_ROOT / "data/chlorophyll/chlorophyll_real_gap_inventory.csv",
        parse_dates=["start_date", "end_date"],
    )

    ax.plot(target.index, target["chl_mean"], lw=0.5, color=COLOR_OBSERVED, alpha=0.9)
    for _, g in real_gaps.iterrows():
        ax.axvspan(g["start_date"], g["end_date"], color=COLOR_GAP, alpha=0.35, lw=0)

    ax.set_ylabel("Chlorophyll-a (mg m$^{-3}$)")
    ax.set_title("Observed chlorophyll record and real gaps")
    ax.set_ylim(0, target["chl_mean"].quantile(0.995))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))


def panel2_artificial_gap_demo(ax):
    demo = pd.read_csv(REPO_ROOT / "demo/outputs/demo_reconstruction_results.csv", parse_dates=["date"])
    demo = demo[demo["artificial_or_real_gap"] == "artificial"]

    truth = demo[demo["method"] == "persistence"][["date", "original_target"]].dropna()
    hidden_dates = demo[demo["original_target"].notna()]["date"].unique()

    interp = demo[demo["method"] == "linear_interpolation"].set_index("date")["reconstructed_median"]
    tsicl = demo[demo["method"] == "tsicl_satellite_proxy"].set_index("date")["reconstructed_median"]

    ax.axvspan(min(hidden_dates), max(hidden_dates), color=COLOR_GAP, alpha=0.18, lw=0, label="Withheld interval")
    ax.plot(truth["date"], truth["original_target"], color=COLOR_TRUTH, lw=1.6, label="Withheld truth")
    ax.plot(interp.index, interp.values, color=COLOR_INTERP, lw=1.3, ls="--", label="Linear interpolation")
    ax.plot(tsicl.index, tsicl.values, color=COLOR_TSICL, lw=1.6, label="TS-ICL (satellite proxy)")

    ax.set_ylabel("log$_{10}$ chlorophyll-a")
    ax.set_title("Artificial-gap validation: one example")
    ax.legend(loc="upper left", frameon=False, fontsize=7.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(4))


def panel3_benchmark_message(ax):
    chl = pd.read_csv(REPO_ROOT / "results/chlorophyll/chlorophyll_benchmark_summary.csv")
    chl_vs_interp = chl[chl["compared_against_public_name"] == "Linear interpolation baseline"]

    def pct_improve(name: str) -> float:
        row = chl_vs_interp[chl_vs_interp["method_public_name"] == name].iloc[0]
        return -row["delta"] / row["comparison_value"] * 100

    chl_tsicl_pct = pct_improve("TS-ICL (satellite chlorophyll proxy covariate)")
    chl_hybrid_pct = pct_improve("Engineered hybrid pipeline (validation-aware method assignment)")

    ox_by_length = pd.read_csv(REPO_ROOT / "results/oxygen/oxygen_benchmark_by_length.csv")
    ox_deltas = pd.read_csv(REPO_ROOT / "results/oxygen/oxygen_paired_deltas_vs_tsicl_physical_covariates.csv")
    interp_rows = ox_by_length[ox_by_length["method_label"] == "Linear interpolation"]
    interp_mae = (interp_rows["mae_gapweighted"] * interp_rows["n_gaps"]).sum() / interp_rows["n_gaps"].sum()
    ox_delta = ox_deltas[ox_deltas["comparator_id"] == "linear_interp"]["delta_tsicl_minus_comparator_mae"].iloc[0]
    ox_tsicl_pct = -ox_delta / interp_mae * 100

    labels = [
        "Chlorophyll:\nTS-ICL (satellite proxy)",
        "Chlorophyll:\nEngineered hybrid",
        "Oxygen:\nTS-ICL (physical covariates)",
    ]
    values = [chl_tsicl_pct, chl_hybrid_pct, ox_tsicl_pct]
    colors = [COLOR_TSICL, "#457b9d", COLOR_TSICL]

    y_pos = range(len(labels))
    ax.barh(list(y_pos), values, color=colors, height=0.55)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("% MAE improvement over linear interpolation")
    ax.set_title("Artificial-gap benchmark")
    for y, v in zip(y_pos, values):
        ax.text(v + (0.3 if v >= 0 else -0.3), y, f"{v:+.1f}%", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    ax.set_xlim(min(0, min(values) - 2), max(values) + 3)


def build(output_name: str, figsize: tuple[float, float]) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    panel1_target_and_gaps(axes[0])
    panel2_artificial_gap_demo(axes[1])
    panel3_benchmark_message(axes[2])
    fig.tight_layout(w_pad=2.2)
    out_path = ASSETS_DIR / output_name
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


if __name__ == "__main__":
    build("project_overview.png", figsize=(13.5, 3.6))
    build("project_overview_social.png", figsize=(12.8, 6.4))
