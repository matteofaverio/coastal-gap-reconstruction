"""Generates gap_reconstruction_walkthrough.ipynb from plain Python/Markdown cell
strings (easier to author and diff than hand-written notebook JSON). Not needed to
run the demo -- only to regenerate the notebook file. Run:

    python3 demo/_build_notebook.py
"""
import json

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})


def code(text):
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.splitlines(keepends=True),
        }
    )


md("""\
# Gap reconstruction walkthrough (live TS-ICL)

This notebook reconstructs a gap in a daily chlorophyll-a time series using several
methods, including a **live, zero-shot run of TS-ICL** (no training on this data,
no fine-tuning) on this machine. It is the hands-on companion to the coastal
gap-reconstruction workflow described in this repository's README.

What this notebook does, in order:

1. Load a daily in-situ chlorophyll series with two covariate channels.
2. Audit missingness in the loaded window.
3. Hide a known 14-day interval on purpose (an *artificial gap*) and keep the true
   values aside, only for scoring at the end.
4. Reconstruct that interval with: persistence, climatology, linear interpolation,
   a Gaussian process, an external-covariates-only tabular model, a gap-edge
   residual model, TS-ICL target-only, and TS-ICL with covariates.
5. Score every method against the withheld truth (mean absolute error).
6. Plot all reconstructions together, including the TS-ICL q05-q95 interval.
7. Apply the same TS-ICL call to one real gap (no withheld truth -- candidate
   output only, explicitly labelled as such).
8. Export a single results table with method, values, uncertainty (where
   available), and provenance/warning columns.

**Live vs. fallback.** TS-ICL runs live in this notebook by default (see the
environment note in `README.md`). If live inference fails in your environment
(no internet on first run to fetch the checkpoint, incompatible hardware, etc.),
the notebook falls back to the cached predictions shipped in
`data/cached_tsicl_predictions.csv` and prints a visible warning when it does --
it never silently substitutes cached numbers for live ones.
""")

code("""\
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe for restart-and-run-all / CI; remove for inline interactive plots
import matplotlib.pyplot as plt

RNG_SEED = 0
np.random.seed(RNG_SEED)

DEMO_DIR = Path(".").resolve()
if not (DEMO_DIR / "data" / "chlorophyll_demo_series.csv").exists():
    DEMO_DIR = Path(__file__).resolve().parent if "__file__" in dir() else DEMO_DIR
DATA = DEMO_DIR / "data"
OUT = DEMO_DIR / "outputs"
OUT.mkdir(exist_ok=True)

GAP_START = pd.Timestamp("2016-03-11")
GAP_END = pd.Timestamp("2016-03-24")
GAP_LENGTH = 14

runtimes = {}
print("Data directory:", DATA)
""")

md("## 1. Load the series")

code("""\
series = pd.read_csv(DATA / "chlorophyll_demo_series.csv", parse_dates=["date"]).sort_values("date").reset_index(drop=True)
series.head()
""")

md("""\
Columns: `chl_mean` (target, mg/m3), `chl_satellite_proxy_log10` (satellite chlorophyll
proxy, log10 scale -- the covariate the final report found strongest for chlorophyll),
`wind_spd_ms` and `sst_primary_degC` (physical covariates), `target_eligible_default`
(daily QC flag from the project's ingestion pipeline).
""")

md("## 2. Audit missingness")

code("""\
n_dup = series["date"].duplicated().sum()
assert n_dup == 0, f"{n_dup} duplicate dates -- fix the input series before continuing"

full_range = pd.date_range(series["date"].min(), series["date"].max(), freq="D")
missing_dates = full_range.difference(series["date"])
n_missing_value = series["chl_mean"].isna().sum()

print(f"Window: {series['date'].min().date()} to {series['date'].max().date()} ({len(series)} calendar days)")
print(f"Missing calendar dates in the index: {len(missing_dates)}")
print(f"Days with missing chlorophyll value: {n_missing_value} ({100*n_missing_value/len(series):.1f}%)")
""")

md("## 3-4. Select the observed interval and create the artificial gap")

code("""\
is_gap = (series["date"] >= GAP_START) & (series["date"] <= GAP_END)
truth = series.loc[is_gap, ["date", "chl_mean"]].rename(columns={"chl_mean": "truth_chl"}).reset_index(drop=True)
assert not truth["truth_chl"].isna().any(), "chosen interval has missing truth -- pick another one"

observed = series.copy()
observed.loc[is_gap, "chl_mean"] = np.nan

context = observed.dropna(subset=["chl_mean"]).copy()
context["t"] = (context["date"] - series["date"].min()).dt.days

print(f"Artificial gap: {GAP_START.date()} to {GAP_END.date()} ({GAP_LENGTH} days), hidden on purpose.")
print(f"Observed context days remaining: {len(context)}")
""")

md("""\
Climatology and the two tabular methods below need more history than this ~5-month
window contains (climatology needs other years at the same day-of-year; the tabular
models need enough rows to fit). We load the full multi-year record for those methods
only, and immediately mask the same artificial gap in it -- the exact same dates are
hidden everywhere in this notebook, there is no separate "easier" copy of the target.
""")

code("""\
full = pd.read_csv(DATA / "chlorophyll_full_record_for_tabular_fit.csv", parse_dates=["date"])
full["log10_chl"] = np.log10(full["chl_mean"])
full.loc[(full["date"] >= GAP_START) & (full["date"] <= GAP_END), "log10_chl"] = np.nan
full["doy_sin"] = np.sin(2 * np.pi * full["date"].dt.dayofyear / 365.25)
full["doy_cos"] = np.cos(2 * np.pi * full["date"].dt.dayofyear / 365.25)
print(f"Full record loaded: {len(full)} days, artificial gap masked the same way as above.")
""")

md("## 5. Reconstruct with several methods")

md("### 5a. Persistence, climatology, linear interpolation (live)")

code("""\
t0 = time.time()
last_value = context.loc[context["date"] < GAP_START, "chl_mean"].iloc[-1]
pred_persistence = truth[["date"]].copy()
pred_persistence["pred_chl"] = last_value
runtimes["persistence"] = time.time() - t0

# Climatology needs other years at the same day-of-year, so it uses the full masked
# record (`full`), not the single-pass local `context` window used by the other methods.
t0 = time.time()
full_doy = full["date"].dt.dayofyear
clim_by_doy = full.groupby(full_doy)["log10_chl"].mean()
pred_climatology = truth[["date"]].copy()
pred_climatology["pred_chl"] = 10 ** truth["date"].dt.dayofyear.map(clim_by_doy)
runtimes["climatology"] = time.time() - t0

t0 = time.time()
interp_series = observed.set_index("date")["chl_mean"].interpolate(method="linear", limit_area="inside")
pred_interp = interp_series.loc[GAP_START:GAP_END].reset_index()
pred_interp.columns = ["date", "pred_chl"]
runtimes["linear_interpolation"] = time.time() - t0

print("done:", list(runtimes.keys()))
""")

md("### 5b. Gaussian process, target-only (live)")

code("""\
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

t0 = time.time()
X = context[["t"]].to_numpy()
y = np.log10(context["chl_mean"].to_numpy())
kernel = Matern(length_scale=10.0, nu=1.5) + WhiteKernel(noise_level=0.05)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, optimizer=None, normalize_y=True)
gp.fit(X, y)

t0_series = series["date"].min()
X_gap = ((truth["date"] - t0_series).dt.days).to_numpy().reshape(-1, 1)
gp_mean_log10, gp_std_log10 = gp.predict(X_gap, return_std=True)
# return_std=True gives the GP's predictive standard deviation (posterior std plus the
# fitted observation-noise term from the WhiteKernel) -- a predictive interval, not a
# generic "confidence interval" on a fixed unknown parameter.
pred_gp = truth[["date"]].copy()
pred_gp["pred_chl"] = 10 ** gp_mean_log10
pred_gp["gp_q05"] = 10 ** (gp_mean_log10 - 1.645 * gp_std_log10)
pred_gp["gp_q95"] = 10 ** (gp_mean_log10 + 1.645 * gp_std_log10)
runtimes["gaussian_process"] = time.time() - t0
print(f"GP fit+predict: {runtimes['gaussian_process']:.3f}s")
""")

md("""\
### 5c. External-covariates-only tabular model (live)

Fit only on covariates (satellite chlorophyll proxy, wind, SST) -- the target's own
recent history is *not* given to this model. Trained on the full record excluding the
artificial gap, since the local demo window alone is too short to fit a tree model
sensibly.
""")

code("""\
from sklearn.ensemble import HistGradientBoostingRegressor

feature_cols = ["chl_satellite_proxy_log10", "wind_spd_ms", "sst_primary_degC", "doy_sin", "doy_cos"]
# HistGradientBoostingRegressor is used (not ExtraTrees) because it natively tolerates
# missing covariate days -- the satellite chlorophyll proxy in particular has real cloud
# cover gaps -- without needing a separate imputation step.
train_mask = full["log10_chl"].notna()
train_mask &= ~((full["date"] >= GAP_START) & (full["date"] <= GAP_END))

t0 = time.time()
ext_model = HistGradientBoostingRegressor(random_state=RNG_SEED)
ext_model.fit(full.loc[train_mask, feature_cols], full.loc[train_mask, "log10_chl"])

gap_rows = full[(full["date"] >= GAP_START) & (full["date"] <= GAP_END)].copy()
gap_rows["doy_sin"] = np.sin(2 * np.pi * gap_rows["date"].dt.dayofyear / 365.25)
gap_rows["doy_cos"] = np.cos(2 * np.pi * gap_rows["date"].dt.dayofyear / 365.25)
pred_ext_log10 = ext_model.predict(gap_rows[feature_cols])
pred_external = pd.DataFrame({"date": gap_rows["date"].to_numpy(), "pred_chl": 10 ** pred_ext_log10})
runtimes["external_tabular"] = time.time() - t0
print(f"External-tabular fit+predict: {runtimes['external_tabular']:.3f}s ({train_mask.sum()} training rows)")
""")

md("""\
### 5d. Gap-edge residual model (live)

Starts from linear interpolation and learns a correction using the pre-gap and
post-gap edge values plus covariates as reconstruction context. Trained on residuals
(true - interpolated) at other gaps carved out of the full record, then applied using
this gap's own edges.
""")

code("""\
from sklearn.ensemble import HistGradientBoostingRegressor

full_sorted = full.sort_values("date").reset_index(drop=True)
full_sorted["interp_log10"] = full_sorted["log10_chl"].interpolate(method="linear", limit_area="inside")
full_sorted["days_from_prev_obs"] = (
    full_sorted["date"] - full_sorted["date"].where(full_sorted["log10_chl"].notna()).ffill()
).dt.days
full_sorted["days_to_next_obs"] = (
    full_sorted["date"].where(full_sorted["log10_chl"].notna()).bfill() - full_sorted["date"]
).dt.days
full_sorted["residual"] = full_sorted["log10_chl"] - full_sorted["interp_log10"]

edge_feature_cols = feature_cols + ["days_from_prev_obs", "days_to_next_obs"]
edge_train_mask = (
    full_sorted["residual"].notna()
    & full_sorted[["days_from_prev_obs", "days_to_next_obs"]].notna().all(axis=1)
    & ~((full_sorted["date"] >= GAP_START) & (full_sorted["date"] <= GAP_END))
)

t0 = time.time()
edge_model = HistGradientBoostingRegressor(random_state=RNG_SEED)
edge_model.fit(full_sorted.loc[edge_train_mask, edge_feature_cols], full_sorted.loc[edge_train_mask, "residual"])

gap_edge_rows = full_sorted[(full_sorted["date"] >= GAP_START) & (full_sorted["date"] <= GAP_END)].copy()
pred_residual = edge_model.predict(gap_edge_rows[edge_feature_cols])
pred_gap_edge_log10 = gap_edge_rows["interp_log10"].to_numpy() + pred_residual
pred_gap_edge = pd.DataFrame({"date": gap_edge_rows["date"].to_numpy(), "pred_chl": 10 ** pred_gap_edge_log10})
runtimes["gap_edge_residual"] = time.time() - t0
print(f"Gap-edge residual fit+predict: {runtimes['gap_edge_residual']:.3f}s ({edge_train_mask.sum()} training rows)")
print("Note: this method uses post-gap observations directly as reconstruction context")
print("(via days_to_next_obs and the post-edge interpolation anchor) -- it is retrospective,")
print("not forecast-safe, and is not directly applicable to an open-ended real gap.")
""")

md("""\
### 5e. TS-ICL, live, zero-shot

TS-ICL (https://github.com/EDF-Lab/ts-icl, PyPI package `tsicl`) is a pretrained
time-series foundation model. It is called here **zero-shot**: no training or
fine-tuning on Tongoy data happens anywhere in this notebook. Covariates are
optional -- the same call works with `covars=None` (target-only) or with an
aligned covariate array.
""")

code("""\
TSICL_LIVE = False
tsicl_load_error = None
try:
    import torch
    from tsicl import TSICL

    t0 = time.time()
    tsicl_model = TSICL()
    runtimes["tsicl_model_load"] = time.time() - t0
    TSICL_LIVE = True
    device_note = "cuda" if torch.cuda.is_available() else "cpu (no CUDA on this machine; MPS is not used by TS-ICL's device selection)"
    print(f"TS-ICL loaded live in {runtimes['tsicl_model_load']:.2f}s (device: {device_note})")
except Exception as e:
    tsicl_load_error = f"{type(e).__name__}: {e}"
    print("TS-ICL could not be loaded live in this environment:")
    print(" ", tsicl_load_error)
    print("Falling back to cached predictions shipped with this demo (data/cached_tsicl_predictions.csv).")
""")

code("""\
def tsicl_impute(target_series_log10, covar_array=None):
    \"\"\"target_series_log10: 1D float array with NaN at the gap. covar_array, if given:
    2D array (T, C) aligned to the same T timesteps as target_series_log10; internally
    reshaped to TS-ICL's required (1, T, C) batch-first covariate shape.\"\"\"
    inputs_t = torch.from_numpy(target_series_log10.astype(np.float32))
    covars_t = None
    if covar_array is not None:
        covars_t = torch.from_numpy(np.ascontiguousarray(covar_array[None, :, :], dtype=np.float32))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, quantiles = tsicl_model.impute(
            inputs=inputs_t,
            covars=covars_t,
            quantile_levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            denormalize=True,
            replace_by_gt=False,
        )
    return mean.numpy(), quantiles.numpy()


gap_mask = is_gap.to_numpy()
target_log10_full = np.log10(series["chl_mean"].to_numpy())
target_log10_masked = target_log10_full.copy()
target_log10_masked[gap_mask] = np.nan
gap_lo = np.argmax(gap_mask)
gap_hi = gap_lo + gap_mask.sum()

if TSICL_LIVE:
    t0 = time.time()
    mean_a, q_a = tsicl_impute(target_log10_masked, covar_array=None)
    runtimes["tsicl_target_only"] = time.time() - t0
    pred_tsicl_target_only = pd.DataFrame({
        "date": series["date"].iloc[gap_lo:gap_hi].to_numpy(),
        "pred_chl": 10 ** mean_a[gap_lo:gap_hi],
        "q05": 10 ** q_a[gap_lo:gap_hi, 0], "q95": 10 ** q_a[gap_lo:gap_hi, 6],
    })

    covar_proxy = series["chl_satellite_proxy_log10"].to_numpy(dtype=np.float32).reshape(-1, 1)
    t0 = time.time()
    mean_b, q_b = tsicl_impute(target_log10_masked, covar_array=covar_proxy)
    runtimes["tsicl_satellite_proxy"] = time.time() - t0
    pred_tsicl_proxy = pd.DataFrame({
        "date": series["date"].iloc[gap_lo:gap_hi].to_numpy(),
        "pred_chl": 10 ** mean_b[gap_lo:gap_hi],
        "q05": 10 ** q_b[gap_lo:gap_hi, 0], "q95": 10 ** q_b[gap_lo:gap_hi, 6],
    })

    covar_bundle = series[["wind_spd_ms", "sst_primary_degC"]].to_numpy(dtype=np.float32)
    t0 = time.time()
    mean_c, q_c = tsicl_impute(target_log10_masked, covar_array=covar_bundle)
    runtimes["tsicl_physical_bundle"] = time.time() - t0
    pred_tsicl_physical = pd.DataFrame({
        "date": series["date"].iloc[gap_lo:gap_hi].to_numpy(),
        "pred_chl": 10 ** mean_c[gap_lo:gap_hi],
        "q05": 10 ** q_c[gap_lo:gap_hi, 0], "q95": 10 ** q_c[gap_lo:gap_hi, 6],
    })
    print("Live TS-ICL runtimes (s):",
          {k: round(v, 3) for k, v in runtimes.items() if k.startswith("tsicl")})
else:
    cached = pd.read_csv(DATA / "cached_tsicl_predictions.csv", parse_dates=["date"])
    pred_tsicl_target_only = cached[cached["tsicl_arm"] == "target_only"][["date", "pred_chl", "q05", "q95"]].reset_index(drop=True)
    pred_tsicl_proxy = cached[cached["tsicl_arm"] == "satellite_proxy"][["date", "pred_chl", "q05", "q95"]].reset_index(drop=True)
    pred_tsicl_physical = cached[cached["tsicl_arm"] == "physical_bundle"][["date", "pred_chl", "q05", "q95"]].reset_index(drop=True)
    print("*** USING CACHED TS-ICL PREDICTIONS (emergency fallback) -- live inference was not available. ***")
""")

md("## 6. Score every method against the withheld truth")

code("""\
def mae_against_truth(pred_df):
    merged = truth.merge(pred_df[["date", "pred_chl"]], on="date")
    return float((merged["pred_chl"] - merged["truth_chl"]).abs().mean())


results_mae = {
    "persistence": mae_against_truth(pred_persistence),
    "climatology": mae_against_truth(pred_climatology),
    "linear_interpolation": mae_against_truth(pred_interp),
    "gaussian_process": mae_against_truth(pred_gp),
    "external_tabular": mae_against_truth(pred_external),
    "gap_edge_residual": mae_against_truth(pred_gap_edge),
    "tsicl_target_only": mae_against_truth(pred_tsicl_target_only),
    "tsicl_satellite_proxy": mae_against_truth(pred_tsicl_proxy),
    "tsicl_physical_bundle": mae_against_truth(pred_tsicl_physical),
}

print(f"{'method':24s}  MAE (mg/m3)   runtime (s)")
for name, mae in sorted(results_mae.items(), key=lambda kv: kv[1]):
    rt = runtimes.get(name, float("nan"))
    print(f"{name:24s}  {mae:7.3f}       {rt:7.3f}")

print()
print("This is a single illustrative gap, not a validation run. The project's actual")
print("validation scores hundreds of artificial gaps across multiple lengths and seasons.")
""")

md("## 7. Plot all reconstructions")

code("""\
fig, ax = plt.subplots(figsize=(11, 5.5))

ctx_plot = series[~is_gap]
ax.plot(ctx_plot["date"], ctx_plot["chl_mean"], color="#2A7F8E", lw=1.2, marker="o", ms=2,
        label="Observed series (context)")
ax.plot(truth["date"], truth["truth_chl"], color="#B24743", lw=2.2, ls="--", marker="o", ms=4,
        label="Withheld truth (scoring only)")

ax.plot(pred_interp["date"], pred_interp["pred_chl"], color="#5F7F8A", lw=1.4, label="Linear interpolation")
ax.plot(pred_persistence["date"], pred_persistence["pred_chl"], color="#5F7F8A", lw=1.2, ls=":", label="Persistence")
ax.plot(pred_gp["date"], pred_gp["pred_chl"], color="#1B9E77", lw=1.6, label="Gaussian process")
ax.fill_between(pred_gp["date"], pred_gp["gp_q05"], pred_gp["gp_q95"], color="#1B9E77", alpha=0.10)
ax.plot(pred_external["date"], pred_external["pred_chl"], color="#3B6FA0", lw=1.4, ls="-.", label="External tabular")
ax.plot(pred_gap_edge["date"], pred_gap_edge["pred_chl"], color="#C0791F", lw=1.4, label="Gap-edge residual")
ax.plot(pred_tsicl_target_only["date"], pred_tsicl_target_only["pred_chl"], color="#5E4B9B", lw=2.0,
        label="TS-ICL target-only" + (" (live)" if TSICL_LIVE else " (cached)"))
ax.plot(pred_tsicl_proxy["date"], pred_tsicl_proxy["pred_chl"], color="#5E4B9B", lw=2.0, ls="--",
        label="TS-ICL + satellite proxy" + (" (live)" if TSICL_LIVE else " (cached)"))
ax.fill_between(pred_tsicl_proxy["date"], pred_tsicl_proxy["q05"], pred_tsicl_proxy["q95"],
                 color="#5E4B9B", alpha=0.12, label="TS-ICL q05-q95")

ax.axvspan(GAP_START, GAP_END, color="grey", alpha=0.08)
ax.set_ylabel("Chlorophyll-a (mg/m$^3$)")
ax.set_title(f"Reconstruction of a {GAP_LENGTH}-day artificial gap" + (" -- TS-ICL run live" if TSICL_LIVE else " -- TS-ICL from cache"))
ax.legend(loc="upper left", fontsize=8, ncol=2)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(OUT / "reconstruction_figure.png", dpi=160)
plt.show()
print("Saved:", OUT / "reconstruction_figure.png")
""")

md("""\
## 8. Apply to one real gap (no withheld truth)

This interval was genuinely never observed. There is nothing to score it against.
The output below is a **candidate reconstruction**, not validation evidence.
""")

code("""\
real_gap = pd.read_csv(DATA / "chlorophyll_real_gap_example.csv", parse_dates=["date"])
real_is_gap = real_gap["in_real_gap"].to_numpy()
real_target_log10 = np.log10(real_gap["chl_mean"].to_numpy())
real_target_log10_masked = real_target_log10.copy()
real_target_log10_masked[real_is_gap] = np.nan
real_lo = np.argmax(real_is_gap)
real_hi = real_lo + real_is_gap.sum()

if TSICL_LIVE:
    covar_proxy_real = real_gap["chl_satellite_proxy_log10"].to_numpy(dtype=np.float32).reshape(-1, 1)
    t0 = time.time()
    mean_r, q_r = tsicl_impute(real_target_log10_masked, covar_array=covar_proxy_real)
    runtimes["tsicl_real_gap"] = time.time() - t0
    real_pred = pd.DataFrame({
        "date": real_gap["date"].iloc[real_lo:real_hi].to_numpy(),
        "pred_chl": 10 ** mean_r[real_lo:real_hi],
        "q05": 10 ** q_r[real_lo:real_hi, 0], "q95": 10 ** q_r[real_lo:real_hi, 6],
    })
    print(f"Live TS-ICL on the real gap: {runtimes['tsicl_real_gap']:.3f}s")
else:
    cached_real = pd.read_csv(DATA / "cached_tsicl_predictions_real_gap.csv", parse_dates=["date"])
    real_pred = cached_real[["date", "pred_chl", "q05", "q95"]]
    print("*** USING CACHED TS-ICL PREDICTIONS for the real gap (emergency fallback). ***")

print("Real gap: 2015-07-01 to 2015-07-14. NO withheld truth exists for these dates.")
real_pred
""")

md("## 9. Export the results table")

code("""\
export_rows = []


def add_rows(pred_df, method_name, covariates_used, gap_type, gap_id, gap_length,
             validation_status, warning_text=""):
    for _, row in pred_df.iterrows():
        date = row["date"]
        t_row = truth.loc[truth["date"] == date, "truth_chl"] if gap_type == "artificial" else pd.Series(dtype=float)
        export_rows.append({
            "date": pd.Timestamp(date).date().isoformat(),
            "original_target": float(t_row.iloc[0]) if len(t_row) else None,
            "observed_or_missing": "artificially_hidden" if gap_type == "artificial" else "really_missing",
            "reconstruction_method": method_name,
            "reconstructed_median": row["pred_chl"],
            "q05": row["q05"] if "q05" in row and pd.notna(row.get("q05")) else None,
            "q95": row["q95"] if "q95" in row and pd.notna(row.get("q95")) else None,
            "artificial_or_real_gap": gap_type,
            "gap_id": gap_id,
            "gap_length": gap_length,
            "covariates_used": covariates_used,
            "validation_status": validation_status,
            "within_validated_length": gap_length in (1, 3, 7, 10, 14, 21, 30),
            "modelling_scale": "log10",
            "tsicl_mode": "live" if TSICL_LIVE else "cached_fallback",
            "warning_text": warning_text,
        })


add_rows(pred_interp, "linear_interpolation", "none", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_persistence, "persistence", "none", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_gp, "gaussian_process", "none", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_external, "external_tabular", "satellite_proxy,wind,sst", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_gap_edge, "gap_edge_residual", "satellite_proxy,wind,sst,edge_distance", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_tsicl_target_only, "tsicl_target_only", "none", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_tsicl_proxy, "tsicl_satellite_proxy", "satellite_proxy", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(pred_tsicl_physical, "tsicl_physical_bundle", "wind,sst", "artificial", "L14_20160311", 14, "single_illustrative_example")
add_rows(real_pred, "tsicl_satellite_proxy", "satellite_proxy", "real", "REAL_L010_20150701", 14,
         "candidate_not_validation_evidence",
         warning_text="Real gap: no withheld truth exists, do not use to compare methods.")

export_df = pd.DataFrame(export_rows)
export_path = OUT / "demo_reconstruction_results.csv"
export_df.to_csv(export_path, index=False)
print(f"Exported {len(export_df)} rows to {export_path}")

with open(OUT / "runtime_summary.json", "w") as fh:
    json.dump({**runtimes, "tsicl_live": TSICL_LIVE, "tsicl_load_error": tsicl_load_error}, fh, indent=2)
print(f"Runtime summary written to {OUT / 'runtime_summary.json'}")
export_df.head()
""")

md("""\
## 10. Adapting this notebook to another sensor or variable

1. Replace `data/chlorophyll_demo_series.csv` with your own daily table: a `date`
   column, your target column, and any covariate columns you want to test.
2. Update the column names used in sections 3-5 (search for `chl_mean` and the
   covariate column names).
3. State explicitly whether you are modelling in the raw unit or a transformed
   scale (this notebook uses `log10`); do not silently mix the two.
4. Pick an artificial gap length and position with enough observed context on
   both sides; the notebook asserts this before proceeding.
5. The four "live" classical methods (persistence, climatology, interpolation,
   Gaussian process) need no code changes beyond the column renames above.
6. TS-ICL is called the same way regardless of target: `covars=None` for
   target-only, or an aligned `(T, C)` array for any number of covariate
   channels. There is no separate "installation-free" mode -- if `tsicl` and its
   checkpoint are available in the active environment (see `README.md`), it runs
   live; if not, the notebook explicitly falls back to cached output rather than
   pretending to run live.
7. Do not evaluate a gap length your own validation has not covered without
   flagging it as unvalidated.
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "TS-ICL demo (venv)", "language": "python", "name": "tsicl-demo"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("demo/gap_reconstruction_walkthrough.ipynb", "w") as fh:
    json.dump(nb, fh, indent=1)

print(f"Wrote demo/gap_reconstruction_walkthrough.ipynb with {len(cells)} cells")
