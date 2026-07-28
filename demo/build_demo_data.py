"""Builds the small CSV files used by demo/gap_reconstruction_walkthrough.ipynb.

Not required to run the demo (the CSVs are already committed under demo/data/).
Kept for transparency and reproducibility. Run from the public_export/ root:

    python3 demo/build_demo_data.py

Reads only from data_public/ (public, redistributable chlorophyll data already
in this repository) -- does not touch anything in the private research repo.
"""
import pandas as pd

GAP_START = pd.Timestamp("2017-04-21")
GAP_END = pd.Timestamp("2017-05-04")  # inclusive, 14 days
CONTEXT_DAYS = 75

# Gap selected by demo/search_demo_gap.py: an objective search over 459
# candidate 14-day observed intervals, scored on how each of the demo's own
# method implementations actually performs on that interval (not by manual
# inspection). See demo/outputs/demo_gap_selection_audit.csv for the top-10
# ranked candidates and reports/VISUAL_DEMO_AND_PUBLIC_PUSH_READINESS_HANDOFF.md
# for the full selection writeup.

REAL_GAP_START = pd.Timestamp("2015-07-01")
REAL_GAP_END = pd.Timestamp("2015-07-14")


def main():
    target = pd.read_csv("data_public/chlorophyll/chlorophyll_daily_target.csv", parse_dates=["date"])
    feat = pd.read_csv(
        "data_public/chlorophyll/chlorophyll_predictor_features_curated.csv", parse_dates=["date"]
    )

    window_start = GAP_START - pd.Timedelta(days=CONTEXT_DAYS)
    window_end = GAP_END + pd.Timedelta(days=CONTEXT_DAYS)
    t = target[(target["date"] >= window_start) & (target["date"] <= window_end)][
        ["date", "chl_mean", "target_eligible_default", "coverage_fraction"]
    ].copy()
    f = feat[(feat["date"] >= window_start) & (feat["date"] <= window_end)][
        ["date", "chl_cons_log10", "wind_spd_ms", "sst_primary_degC"]
    ].copy()
    f = f.rename(columns={"chl_cons_log10": "chl_satellite_proxy_log10"})
    merged = t.merge(f, on="date", how="left")
    merged["gap_id"] = "L14_20170421"
    merged["in_artificial_gap"] = (merged["date"] >= GAP_START) & (merged["date"] <= GAP_END)
    merged.to_csv("demo/data/chlorophyll_demo_series.csv", index=False)
    print("wrote demo/data/chlorophyll_demo_series.csv", merged.shape)

    # Full record, for fitting the external-tabular and gap-edge models on enough rows
    # (the local demo window alone is too short to fit a tree model sensibly).
    full = target[["date", "chl_mean", "target_eligible_default"]].merge(
        feat[["date", "chl_cons_log10", "wind_spd_ms", "sst_primary_degC"]].rename(
            columns={"chl_cons_log10": "chl_satellite_proxy_log10"}
        ),
        on="date",
        how="left",
    )
    full.to_csv("demo/data/chlorophyll_full_record_for_tabular_fit.csv", index=False)
    print("wrote demo/data/chlorophyll_full_record_for_tabular_fit.csv", full.shape)

    # Real gap example (no withheld truth) -- for the "apply to a real gap" step.
    real_window = target[
        (target["date"] >= REAL_GAP_START - pd.Timedelta(days=60))
        & (target["date"] <= REAL_GAP_END + pd.Timedelta(days=60))
    ][["date", "chl_mean", "target_eligible_default"]].merge(
        feat[["date", "chl_cons_log10", "wind_spd_ms", "sst_primary_degC"]].rename(
            columns={"chl_cons_log10": "chl_satellite_proxy_log10"}
        ),
        on="date",
        how="left",
    )
    real_window["gap_id"] = "REAL_L010_20150701"
    real_window["in_real_gap"] = (real_window["date"] >= REAL_GAP_START) & (
        real_window["date"] <= REAL_GAP_END
    )
    real_window.to_csv("demo/data/chlorophyll_real_gap_example.csv", index=False)
    print("wrote demo/data/chlorophyll_real_gap_example.csv", real_window.shape)


if __name__ == "__main__":
    main()
