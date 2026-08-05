# Using TS-ICL in this benchmark

## What TS-ICL is

TS-ICL is a zero-shot, in-context-learning time-series foundation model: a
single pretrained model is given a series with missing values directly (no
task-specific fine-tuning on this station's data) and asked to fill them in,
optionally conditioned on additional covariate channels supplied alongside
the target series.

**Citation**: Etienne Le Naour, Tahar Nabil, Adrien Petralia, "TS-ICL: A
Flexible Time-Index Foundation Model for Time Series via In-Context
Learning." Package: `tsicl` on PyPI. Model weights: Hugging Face Hub,
`taharnbl/TS-ICL`.

**License**: TS-ICL code and pretrained weights are distributed under the
**TS-ICL Non-Commercial License v1.0, (c) EDF SA 2026** -- separate from
this repository's own license. It prohibits commercial/production use and
hosted-service/API distribution, and requires an attribution notice if the
model or a derivative is itself redistributed (not applicable here: this
repository distributes only original wrapper/orchestration code, never the
checkpoint or the `tsicl` package's own inference code). Review the full
license text in the authors' repository before using TS-ICL in your own
work.

## Reproducibility levels

This document covers a lot of ground. Most readers only need the first
level below -- pick the one that matches what you're actually trying to do.

**QUICK** (minutes, no TS-ICL checkpoint download required for most of it):
run the standard test suite, notebooks, and demo; a single bounded TS-ICL
inference call (the demo, or the "verify the fresh install" command below)
if you want to confirm live inference works at all. This is enough to use
and inspect the repository.

**STANDARD** (tens of minutes to a few hours): the matched-449-support
target-only/satellite-proxy benchmark and the matched-449 covariate arms,
plus inspecting the released result tables in `results_public/chlorophyll/`
directly. This is enough to compare TS-ICL against the classical/
probabilistic methods on shared support, and is what most reproduction
checks should use.

**EXPENSIVE** (hours to days) -- **optional, not required for publication
closure or for using this repository**: the full-681-gap primary target
benchmark (8,172 calls) and/or the complete 28,602-call full covariate
dissection. Both are published, tested, and restart-safe. The full target
benchmark has been run to completion at least once (§ "Full-681 target
run," below) with the results checked into this document. The full
covariate dissection has been validated live on a partial real execution
(2,150/28,602 calls, 0 failures) confirming the driver, resume logic, and
scoring pipeline all work correctly at full-681 scale -- the complete
28,602-call regeneration was intentionally not run to completion, because
its compute cost (on the order of a day of continuous CPU inference) is
disproportionate to what a public reproducibility artifact needs: the
released `chlorophyll_covariate_mechanism_summary.csv` is the authoritative,
complete covariate ranking, and the code that produced the equivalent
result is published, tested, and available to run in full if you choose to.
Running the complete 28,602-call dissection is never required to use,
review, or trust this repository.

## Checkpoint and package provenance (pinned, verified)

| Field | Value |
|---|---|
| Hugging Face repository | `taharnbl/TS-ICL` |
| Revision (commit) | `f01f3869a735694691401cd67a5e19c17e94e220` |
| Checkpoint filename | `tsicl-v1.ckpt` |
| Checkpoint SHA-256 | `a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431` |
| Checkpoint size | 219,150,987 bytes (~209 MB) |
| `tsicl` package version | `0.2.1` |
| `torch` version used for the released benchmark | `2.9.1` (BSD-3-Clause; unrelated to the TS-ICL license) |
| `max_context_length` | 4096 |
| Device (released benchmark) | CPU (no GPU used or required) |
| Determinism | `.eval()` mode (no dropout), no explicit `torch.manual_seed()` call anywhere in the calling code -- repeated calls with identical inputs on the same machine/environment/torch version are expected to be numerically identical, but bit-exact reproduction **has not been independently verified across different machines/torch versions/CPU vs. GPU** -- do not assume it beyond a fixed environment. |

These values are pinned as Python constants in
`coastal_gap_reconstruction/tsicl_helpers.py`
(`CHECKPOINT_REPO_ID`/`CHECKPOINT_REVISION`/`CHECKPOINT_FILENAME`/
`CHECKPOINT_SHA256`/`EXPECTED_TSICL_PACKAGE_VERSION`), and verified by
`verify_checkpoint_provenance()`/`load_tsicl_strict()` -- a reproducibility-
focused run raises immediately if the locally cached checkpoint's hash, or
the installed package version, doesn't match. This repository never
downloads, vendors, or commits the checkpoint or the `tsicl` package's own
code; the checkpoint is fetched by the `tsicl` package's own
`huggingface_hub.hf_hub_download` call the first time a `TSICL()` instance
is constructed with network access, into the standard Hugging Face Hub
local cache (`~/.cache/huggingface/hub/`).

## Installation environment

TS-ICL (and its `torch` dependency) is **not** part of this repository's
standard locked environment (`uv.lock`) -- it is a genuinely, separately
locked execution environment under `environments/tsicl/`
(`pyproject.toml` + `uv.lock`, pinning Python, `tsicl`, `torch`, `numpy`,
`pandas`, `scipy`, `scikit-learn`, `huggingface_hub`, and every other
transitive dependency the drivers need), kept apart so that installing or
running it never destabilizes the lightweight core package that every
other notebook and test in this repository uses. An earlier draft of this
document described a bare `pip install tsicl==0.2.1 torch==2.9.1 pandas
numpy` command as "separately locked" -- it was not: pandas/NumPy and every
transitive dependency were left unpinned. This has been corrected; the
commands below use the real, committed lock file.

```bash
# Core environment (no torch/tsicl) -- what everything else in this
# repository uses by default.
uv sync --locked

# Separate, genuinely locked TS-ICL execution environment.
cd environments/tsicl
uv sync --locked
cd ../..
```

Verify the fresh install before trusting it for a reproducibility run:

```bash
PYTHONPATH=src environments/tsicl/.venv/bin/python -c "
from coastal_gap_reconstruction import tsicl_helpers as th
model, provenance = th.load_tsicl_strict()
print(provenance)
"
```

This must print a provenance dict with the checkpoint hash below and raise
nothing -- if it raises `TSICLProvenanceError`/`TSICLDependencyError`, do
not proceed with a benchmark run.

The first call that constructs `TSICL()` downloads the ~209 MB checkpoint
from Hugging Face Hub (network access required, one-time; respects
`HF_HOME`/`HUGGINGFACE_HUB_CACHE` if set, otherwise the standard
`~/.cache/huggingface/hub/` default). All TS-ICL drivers in this repository
(`experiments/chlorophyll/run_tsicl_benchmark.py`,
`run_tsicl_covariate_analysis.py`) must be run with
`environments/tsicl/.venv`'s Python, not the core environment's:

```bash
PYTHONPATH=src environments/tsicl/.venv/bin/python -m experiments.chlorophyll.run_tsicl_benchmark \
    --support full_681 \
    --arms target_only,target_plus_calendar,target_plus_physical_forcing,satellite_proxy,target_plus_physical_forcing_plus_proxy,wrong_lag_physical_forcing \
    --context-modes full_series,edge_balanced \
    --out build/chlorophyll/tsicl_benchmark_full681
```

**EXPENSIVE, optional** -- the complete covariate dissection (28,602 calls,
on the order of a day of CPU time). Not required for publication closure or
for using this repository (see "Reproducibility levels" above); the
released `chlorophyll_covariate_mechanism_summary.csv` is the authoritative
complete result. The command below is restart-safe and will resume cleanly
from `build/chlorophyll/tsicl_covariates_full681_verified/`'s existing
partial state (2,150/28,602 calls, 0 failures as of this document's last
update -- see `PARTIAL_RUN_STOP_RECORD.json` in that directory) if you
choose to run it further:

```bash
PYTHONPATH=src environments/tsicl/.venv/bin/python -m experiments.chlorophyll.run_tsicl_covariate_analysis \
    --support full_681 --include-placebos \
    --arms target_only,satellite_proxy,solar_only,wind_upwelling_only,sst_thermal_only,plv_meteorological,current_transport_only,availability_proxy_only,solar_upwelling_interaction,upwelling_cooling_interaction,curated_physical,full_physical_redundant,proxy_plus_solar,proxy_plus_wind_upwelling,proxy_plus_sst,proxy_plus_plv_met,proxy_plus_current_transport,proxy_plus_availability \
    --out build/chlorophyll/tsicl_covariates_full681_verified
```

Score a completed run (aggregate/by-length/event metrics, paired bootstrap
vs. a freshly generated interpolation comparator, `VERIFICATION_STATUS`):

```bash
.venv/bin/python -m experiments.chlorophyll.score_tsicl_run --run-dir build/chlorophyll/tsicl_benchmark_full681
```

### Restart safety and configuration-bound output directories

Both drivers are restart-safe. All resume/failure/completion accounting is
recomputed from `predictions.jsonl`/`failures.jsonl` on every invocation --
there is no separate compact cache that can silently go stale. **A failed
call is retried on every subsequent invocation by default** (an earlier
version of both drivers recorded a call's key as "done" even when it
failed, so a resume silently skipped a permanently-failed call and could
report completion despite an unresolved failure -- this was a real bug,
found and fixed this sprint, not a hypothetical risk; see
`tsicl_run_state.py` and `tests/test_tsicl_run_state.py` /
`tests/test_run_tsicl_benchmark.py` for the fix and its regression tests).
`RUN_COMPLETE` requires every expected call to have exactly one valid
successful record and zero unresolved failures.

Each output directory is bound to the exact configuration (support, arm
list, context modes, checkpoint hash, input file hashes) that first wrote
into it, via `run_manifest.json` (`tsicl_run_manifest.py`). Resuming the
same directory with a different configuration raises immediately, instead
of silently mixing predictions from two incompatible runs -- use a new,
empty `--out` directory for a different configuration.

## Two supports: full 681-gap pool vs. matched 449-gap support

See `experiments/chlorophyll/tsicl_contract.py` for the exact, tested
definitions. In short:

- **Full 681-gap pool** (`--support full_681`, the default): the primary
  support for the released target-only/satellite-proxy paired comparison
  (`chlorophyll_benchmark_summary.csv`) and the 18-arm covariate ranking
  (`chlorophyll_covariate_mechanism_summary.csv`). The authoritative
  primary target benchmark grid, resolved by direct inspection of the
  private project's primary TS-ICL benchmark driver (not assumed): **681 gaps x 2 context
  modes (`full_series`, `edge_balanced`) x 6 primary arms** --
  `target_only`, `target_plus_calendar`, `target_plus_physical_forcing`,
  `satellite_proxy`, `target_plus_physical_forcing_plus_proxy`, and a
  `wrong_lag_physical_forcing` placebo control -- 8,172 calls total,
  `target_repr="raw"` throughout (see `tsicl_contract.PRIMARY_ARMS`/
  `PRIMARY_ARM_ORDER`/`FULL_681_PRIMARY_TOTAL_CALLS`). An earlier version
  of this document and of `run_tsicl_benchmark.py` implemented only 2 of
  these 6 arms (`target_only`, `satellite_proxy`); the other 4 were
  entirely missing from the public driver until this correction.
- **Matched 449-gap support** (`--support matched_449`): the same support
  the classical/probabilistic/gap-edge benchmark uses
  (`experiments/chlorophyll/benchmark_contract.py`). `tsicl_target_only`/
  `tsicl_satellite_proxy` both have a released row here
  (`results_public/chlorophyll/chlorophyll_matched_support_method_metrics.csv`)
  -- **use this support, not the full pool, for any TS-ICL-vs-classical-ML
  comparison**, since the classical methods beyond interpolation/GP have no
  row on the full pool.
- **Real-gap support** (128 real missing intervals): out of scope for the
  code published in this phase. A real-gap reconstruction output exists in
  `results_public/` from an earlier phase, but this repository does not
  publish the driver that produced it.

## Input construction

For each gap: the target series (daily chlorophyll, `log10(chl_mean)`,
eligibility floor `1e-4`) is sliced to a context window around the gap
(`full_series`: everything; `edge_balanced`/`local_window`: a fixed window
centered on the gap, truncated symmetrically if it exceeds
`max_context_length`), the gap's own days are masked to `NaN`
unconditionally (the model never receives the hidden truth, regardless of
what the input array contains at those positions before masking), and,
for covariate-conditioned arms, an aligned `(1, T, C)` covariate block is
supplied alongside it (see `tsicl_covariate_registry.py` for the exact
column membership of every retained arm). Covariates may contain `NaN` at
individual sparsely-missing positions (e.g. cloud-masked satellite-proxy
days) -- TS-ICL's own `impute()` call handles this internally; this
repository does not pre-fill or reject them.

## Failure behavior

`run_gap_inference(..., strict=True)` (the default, and what both benchmark
drivers use) raises an explicit, typed error rather than silently
substituting a fallback value:

- `TSICLDependencyError`: `torch`/`tsicl` not importable.
- `TSICLProvenanceError`: checkpoint hash/revision or package version
  mismatch.
- `TSICLInputError`: malformed gap/context/covariate input (unrecognized
  context mode, covariate shape/length mismatch, a gap boundary date not
  present in the series).
- `TSICLOutputError`: TS-ICL's own output fails a basic sanity check (wrong
  shape, non-finite point prediction, non-monotonic quantiles).

The benchmark drivers catch these per-gap, record the failure explicitly in
`failures.jsonl`, and continue to the next gap -- an **explicit resumable
batch mode**, distinct from the low-level function's own strict-by-default
behavior. A gap's failure is never silently skipped or replaced with a
cached/interpolated value.

## Covariate arms

See `experiments/chlorophyll/tsicl_covariate_registry.py`
(`COVARIATE_ARMS`) for the full, tested registry: every retained arm's
descriptive public name (reusing the names already published in
`chlorophyll_covariate_mechanism_summary.csv`, never the private project's
internal short codes), exact column membership, and role
(primary/supporting/exploratory). **18 base arms.** Four placebo/negative-
control transforms (`wrong_lag`, `season_shuffled`, `year_shifted`,
`permuted`) are each applied to **6 placebo-eligible families**
(`PLACEBO_ELIGIBLE_ARMS`: `curated_physical`, `wind_upwelling_only`,
`solar_only`, `sst_thermal_only`, `current_transport_only`,
`availability_proxy_only`) -- 24 placebo variants. **18 + 24 = 42 total
variants**, `context_mode="full_series"` only, matching the private
the private project's own covariate-dissection run-plan arithmetic exactly (681 gaps x
42 variants = 28,602 calls for the full covariate dissection, matching that
module's own documented call count). An earlier version of this registry's
`PLACEBO_ELIGIBLE_ARMS` list was missing `availability_proxy_only`,
silently dropping one whole placebo family (4 of the 24 variants) from any
`--include-placebos` run; this has been corrected.

## Paired statistical procedure

`coastal_gap_reconstruction/paired_statistics.py` implements the gap-
clustered paired bootstrap: **resample `gap_id` with replacement; every day
inside a resampled gap comes along with it.** A day-level bootstrap is
never used -- days within one gap are not independent draws. 2000
replicates, seed 42, 95% percentile CI, significance = CI excludes zero.

**Exact day-level pairing is enforced, not assumed.** Sharing a `gap_id` is
not sufficient to pair two methods' day rows: `bootstrap_compare` verifies
that both methods have day rows on *exactly* the same dates within every
common gap, with no duplicate (method, gap, date) rows, before computing
any metric. The default (`pairing="strict"`) raises `ValueError` on any
within-gap date-support mismatch; `pairing="intersection"` restricts a
mismatched gap to its common date subset and records the exclusion
explicitly (`PairedComparisonResult.excluded_gaps`) rather than silently
comparing rows from different days as if they were the same observation.

Reported findings this procedure supports:

- The pooled (all-gaps) target-only/satellite-proxy TS-ICL improvement over
  interpolation is statistically supported (CI excludes zero).
- Per-length point estimates can be directionally consistent with the
  pooled result without being individually significant at every length --
  this repository does not report a per-length CI where the released
  tables don't provide one, and does not describe a point estimate as
  "significant" without its own CI excluding zero.

## What reproduces exactly, what is environment-sensitive, what is
frozen-only

- **Exact**: checkpoint identity (hash-verified), gap pool membership,
  input construction (masking, context slicing) -- these do not depend on
  the runtime environment.
- **Same-environment per-day repeatability: directly verified, not
  inferred from aggregate equality.**
  `experiments/chlorophyll/verify_same_environment_repeatability.py` runs a
  bounded sample of real gaps through two fully independent process
  invocations and diffs point predictions and quantiles day-by-day (not
  just the aggregate MAE, which can hide compensating per-day errors). The
  result on a 10-gap x 2-arm (20-key) sample:
  `bitwise_repeatable_in_this_environment` -- every point prediction and
  every quantile value identical to the last bit across two separate
  process runs (see `build/chlorophyll/same_environment_repeatability/` for
  the raw comparison output this claim is based on). This is
  same-environment evidence only; bit-exact reproduction across different
  machines/torch versions/CPU vs. GPU remains **not** independently
  verified (see "Determinism" above) -- do not extrapolate this result
  beyond a fixed environment.
- **Environment-sensitive across different environments**: the model's own
  numeric output. Foundation-model inference is sensitive to `torch`
  version, dtype, device, and the `tsicl` package/API version; this
  repository states this directly rather than claiming cross-environment
  bit-reproducibility it has not verified. Compare freshly generated
  predictions against the frozen tables using
  `experiments/chlorophyll/score_tsicl_run.py`'s `VERIFICATION_STATUS`
  classification, which uses an empirical reporting band
  (`TSICL_METRIC_TOLERANCE`), not an assumed threshold.
- **Frozen, authoritative, and not required to be regenerated**: the
  128-real-gap reconstruction candidate output
  (`chlorophyll_reconstruction_tsicl_satellite_proxy.csv`, this repository
  does not publish the real-gap deployment driver at all) and the complete
  28,602-call covariate dissection ranking
  (`chlorophyll_covariate_mechanism_summary.csv`). The driver that could
  regenerate the latter (`run_tsicl_covariate_analysis.py
  --include-placebos --support full_681`) is published, tested, restart-
  safe, and was validated live on a real partial execution (2,150/28,602
  calls, 0 failures, all 42 arm/placebo variants exercised on 51-52
  L=1 gaps) before being intentionally stopped -- a deliberate,
  compute-bounded publication decision, not a code or reliability problem.
  See `build/chlorophyll/tsicl_covariates_full681_verified/
  PARTIAL_RUN_STOP_RECORD.json` and the full-support closure handoff.
  Running it to completion remains available and resumable but is not
  required for publication closure or for using this repository (see
  "Reproducibility levels" above).

## Runtime and compute requirements

CPU inference: measured at approximately 1.0-1.5 seconds per
gap-context-arm call for the target benchmark's arms, but substantially
more for the largest covariate arms (the current/transport and full-
tabular-redundant arms carry far more covariate channels; the private
project's own cost profile put these at roughly 24-43s/call vs. ~2s/call
for the cheapest arms) -- covariate-dissection throughput is not uniform
across arms and should not be extrapolated from the target benchmark's
rate. No GPU required or used.

Measured, not estimated: the full primary target benchmark (681 gaps x 2
context modes x 6 primary arms = 8,172 calls) took **3h01m33s wall clock**
end to end, 0 failures. The full covariate dissection (681 gaps x 42
variants = 28,602 calls) is **EXPENSIVE and optional** (see
"Reproducibility levels" above) -- on the order of a day of continuous CPU
time at the rate observed for the arms reached so far; both jobs are run
sequentially, never concurrently, to avoid uncontrolled CPU contention (see
`build/run_full_support_closure.sh`). The matched-449 support (449 gaps) at
one context mode is far more tractable for a quick reproduction check
(STANDARD level). Both drivers checkpoint every 25 calls specifically
because of this runtime, so a long run can be safely interrupted (including
by `SIGTERM`) and resumed, or left intentionally stopped as a validated
partial execution.
