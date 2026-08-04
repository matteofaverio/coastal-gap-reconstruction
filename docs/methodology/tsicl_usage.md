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
standard locked environment (`uv.lock`) -- it is a separately-locked,
optional execution environment, kept apart so that installing or running it
never destabilizes the lightweight core package that every other notebook
and test in this repository uses.

```bash
# Core environment (no torch/tsicl) -- what everything else in this
# repository uses by default.
uv sync --locked

# Separate TS-ICL execution environment.
python -m venv .venv_tsicl
.venv_tsicl/bin/pip install tsicl==0.2.1 torch==2.9.1 pandas numpy
```

The first call that constructs `TSICL()` downloads the ~209 MB checkpoint
from Hugging Face Hub (network access required, one-time). All TS-ICL
drivers in this repository (`experiments/chlorophyll/run_tsicl_benchmark.py`,
`run_tsicl_covariate_analysis.py`) must be run with `.venv_tsicl`'s Python,
not the core environment's:

```bash
PYTHONPATH=src .venv_tsicl/bin/python -m experiments.chlorophyll.run_tsicl_benchmark \
    --support matched_449 --arms target_only,satellite_proxy \
    --context-modes full_series \
    --out build/chlorophyll/tsicl_benchmark

PYTHONPATH=src .venv_tsicl/bin/python -m experiments.chlorophyll.run_tsicl_covariate_analysis \
    --arms curated_physical,satellite_proxy --support matched_449 \
    --out build/chlorophyll/tsicl_covariates
```

Both drivers are restart-safe (checkpoint every 25 calls to
`predictions.jsonl`/`done_keys.json`; re-running the same command resumes
rather than recomputes) and never overwrite `results_public/` by default.

## Two supports: full 681-gap pool vs. matched 449-gap support

See `experiments/chlorophyll/tsicl_contract.py` for the exact, tested
definitions. In short:

- **Full 681-gap pool** (`--support full_681`, the default): the primary
  support for the released target-only/satellite-proxy paired comparison
  (`chlorophyll_benchmark_summary.csv`) and the 18-arm covariate ranking
  (`chlorophyll_covariate_mechanism_summary.csv`). Uses both `full_series`
  and `edge_balanced` context modes for the primary target-only/satellite-
  proxy arms.
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
(primary/supporting/exploratory). Four placebo/negative-control transforms
(`wrong_lag`, `season_shuffled`, `year_shifted`, `permuted`) are available
for the arms the released placebo-robustness check used.

## Paired statistical procedure

`coastal_gap_reconstruction/paired_statistics.py` implements the gap-
clustered paired bootstrap: **resample `gap_id` with replacement; every day
inside a resampled gap comes along with it.** A day-level bootstrap is
never used -- days within one gap are not independent draws. 2000
replicates, seed 42, 95% percentile CI, significance = CI excludes zero.
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
- **Environment-sensitive**: the model's own numeric output. Foundation-
  model inference is sensitive to `torch` version, dtype, device, and the
  `tsicl` package/API version; this repository states this directly rather
  than claiming bit-reproducibility it has not verified (see "Determinism"
  above). Compare freshly generated predictions against the frozen tables
  using the classification your own run's `--verify`-equivalent evidence
  supports, not an assumed tolerance.
- **Frozen-only** (not reproduced or re-run by the code in this
  repository): the 128-real-gap reconstruction candidate output
  (`chlorophyll_reconstruction_tsicl_satellite_proxy.csv`) and the full
  28,602-call C0-C13 covariate dissection's exhaustive sub-variant grid --
  this repository publishes and can re-run the retained primary/supporting
  arms (`tsicl_covariate_registry.py`), not every private sub-variant ever
  explored.

## Runtime and compute requirements

CPU inference: approximately 1.5-3 seconds per gap-context-arm call on a
laptop CPU (no GPU required or used by the released benchmark). The full
primary target-only benchmark (681 gaps x 2 context modes x 2 primary arms)
is on the order of hours, not minutes; the matched-449 support (449 gaps)
at one context mode is more tractable for a quick reproduction check. Both
drivers checkpoint incrementally specifically because of this runtime, so a
long run can be safely interrupted and resumed.
