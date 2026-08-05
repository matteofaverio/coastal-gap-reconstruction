# Reproducibility

Three levels, from "just look at it" to "regenerate everything." Most
readers only need QUICK.

## QUICK -- install, tests, notebooks, frozen results (minutes, no TS-ICL/torch needed)

```bash
git clone https://github.com/matteofaverio/coastal-gap-reconstruction
cd coastal-gap-reconstruction
uv sync --extra notebooks --extra test   # locked core environment (uv.lock)
uv run pytest tests/
uv run jupyter lab notebooks/01_data_and_gap_audit.ipynb
```

This installs pandas/numpy/matplotlib/scipy/scikit-learn/jupyter at the
exact versions in `uv.lock` -- the same environment CI uses. It does not
install TS-ICL or torch. If you don't have [uv](https://docs.astral.sh/uv/),
a plain-pip equivalent works but is not version-pinned:
`python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[notebooks,test]"`.

At this level you can: run the full test suite, execute every notebook
(they degrade gracefully where TS-ICL isn't installed), and inspect every
frozen result table under `results/` and `data/` directly.

```bash
ruff check .        # lint
uv run pytest tests/test_notebooks_smoke.py -v   # every notebook, executed
```

## STANDARD -- bounded live runs (tens of minutes)

Reruns a scientifically representative subset live, on your machine, for
both case studies:

```bash
# Classical benchmark, chlorophyll (449-gap matched support; ~1-2 hours,
# GP is quick, tree-ensemble arms are the slow part)
python -m experiments.chlorophyll.run_classical_benchmark
python -m experiments.chlorophyll.run_classical_benchmark --verify

# Classical + bounded TS-ICL, oxygen (~30s classical, <=60 live TS-ICL calls)
python -m experiments.oxygen.run_oxygen_benchmark --mode classical
python -m experiments.oxygen.run_oxygen_benchmark --mode tsicl-bounded --tsicl-n-gaps 15
```

A single bounded TS-ICL call (needs the separate locked environment, below)
is enough to confirm live inference works end to end:

```bash
cd environments/tsicl && uv sync --locked && cd ../..
PYTHONPATH=src environments/tsicl/.venv/bin/python -c "
from coastal_gap_reconstruction import tsicl_helpers as th
model, provenance = th.load_tsicl_strict()
print(provenance)
"
```

The live, visual, multi-configuration demo (`demo/gap_reconstruction_walkthrough.ipynb`,
`bash demo/run_demo.sh`) also runs at this level and uses the same
`environments/tsicl/` environment.

`--verify` / `score_*.py` compare a fresh run against the frozen
`results/` tables and report a structured, per-metric classification
(exact / within an empirical tolerance band / mismatched / not
applicable), never a single pass/fail. Small residual differences (parts
per thousand) between a fresh run and the frozen tables are expected --
`ExtraTreesRegressor`/`HistGradientBoostingRegressor`/GP-optimizer fits are
not bit-reproducible across independent environments, and this is
documented per method rather than papered over with a looser tolerance.

## EXPENSIVE, optional -- full grids (hours to a day+, never required)

Not required to use, inspect, or trust this repository. The frozen tables
under `results/` are the authoritative complete results; the code below
regenerates the equivalent and is published, tested, and restart-safe, but
was intentionally not run to completion as part of preparing this
repository, because the compute cost is disproportionate to what a public
reproducibility artifact needs.

```bash
# Full chlorophyll TS-ICL target benchmark: 681 gaps x 2 context modes x 6
# arms = 8,172 calls. Measured: ~3h wall clock, 0 failures.
PYTHONPATH=src environments/tsicl/.venv/bin/python -m experiments.chlorophyll.run_tsicl_benchmark \
    --support full_681 \
    --arms target_only,target_plus_calendar,target_plus_physical_forcing,satellite_proxy,target_plus_physical_forcing_plus_proxy,wrong_lag_physical_forcing \
    --context-modes full_series,edge_balanced \
    --out build/chlorophyll/tsicl_benchmark_full681

# Full covariate dissection: 681 gaps x 42 arm/placebo variants = 28,602
# calls, on the order of a day of CPU time. Restart-safe -- the driver
# recomputes resume state from predictions.jsonl/failures.jsonl on every
# invocation, so it is always safe to interrupt (including SIGTERM) and
# resume later.
PYTHONPATH=src environments/tsicl/.venv/bin/python -m experiments.chlorophyll.run_tsicl_covariate_analysis \
    --support full_681 --include-placebos \
    --arms target_only,satellite_proxy,solar_only,wind_upwelling_only,sst_thermal_only,plv_meteorological,current_transport_only,availability_proxy_only,solar_upwelling_interaction,upwelling_cooling_interaction,curated_physical,full_physical_redundant,proxy_plus_solar,proxy_plus_wind_upwelling,proxy_plus_sst,proxy_plus_plv_met,proxy_plus_current_transport,proxy_plus_availability \
    --out build/chlorophyll/tsicl_covariates_full681

# Full oxygen TS-ICL grid: 5 audited-original arms x 2 context modes + 4
# exploratory arms, all 406 primary gaps, several thousand calls.
PYTHONPATH=src environments/tsicl/.venv/bin/python -m experiments.oxygen.run_oxygen_benchmark \
    --mode tsicl-bounded --tsicl-n-gaps 406 --max-calls 100000
```

## TS-ICL: environment, checkpoint, license

TS-ICL (and `torch`) is a genuinely, separately locked environment under
`environments/tsicl/` (`pyproject.toml` + `uv.lock`), kept apart from the
core `uv.lock` so installing or running it never destabilizes the
lightweight core environment every notebook and test otherwise uses. All
TS-ICL-calling code in this repository -- the demo, notebook 04, and both
benchmark drivers -- goes through one calling layer,
`src/coastal_gap_reconstruction/tsicl_helpers.py`, and must be run with
`environments/tsicl/.venv`'s Python, not the core environment's.

| Field | Value |
|---|---|
| Hugging Face repository | `taharnbl/TS-ICL` |
| Revision | `f01f3869a735694691401cd67a5e19c17e94e220` |
| Checkpoint SHA-256 | `a67ae9f694c2a83cfc8e7ec41745ff4f41a4a76ee2b17172ec3430d8d29da431` |
| `tsicl` package version | `0.2.1` |
| `torch` version | `2.9.1` |
| Device | CPU (no GPU required or used) |

Pinned as Python constants in `tsicl_helpers.py` and verified by
`verify_checkpoint_provenance()`/`load_tsicl_strict()`, which raise
immediately on any hash/version mismatch. The checkpoint (~209 MB) is
fetched on first use via the `tsicl` package's own `huggingface_hub` call
into the standard local cache (`~/.cache/huggingface/hub/`, or
`HF_HOME`/`HUGGINGFACE_HUB_CACHE` if set) -- never vendored or committed
here.

**Citation**: Etienne Le Naour, Tahar Nabil, Adrien Petralia, "TS-ICL: A
Flexible Time-Indexed Foundation Model for Time Series via In-Context
Learning," 2026 ([EDF-Lab/ts-icl](https://github.com/EDF-Lab/ts-icl)).
**License**: TS-ICL Non-Commercial License v1.0, (c) EDF SA 2026 -- separate
from this repository's MIT license, prohibits commercial/production use.
Review the license in the authors' repository before using TS-ICL in your
own work.

**Determinism**: same-process, same-environment repeatability is directly
verified (`experiments/chlorophyll/verify_same_environment_repeatability.py`
-- bitwise-identical predictions across two independent process
invocations). Bit-exact reproduction across different machines/torch
versions/CPU vs. GPU is **not** independently verified -- do not assume it
beyond a fixed environment.

## What's frozen vs. regenerable

Every table under `results/` is a frozen, released result -- authoritative
whether or not you rerun anything. Regenerating a table with the STANDARD
or EXPENSIVE commands above will not produce a bit-identical file (ML
estimator fits and TS-ICL inference are environment-sensitive, as noted
above); `--verify`/`score_*.py` report a structured comparison rather than
requiring exact equality. See `docs/evidence_and_limitations.md` for which
results carry validation-grade weight in the first place.

## Documents

Each document under `manuscript/` compiles independently with
[Tectonic](https://tectonic-typesetting.github.io/) (self-contained, no
separate TeX Live install needed):

```bash
cd manuscript/report && tectonic main.tex
cd ../presentation && tectonic main.tex
cd ../presentation_colleagues_es && tectonic main.tex
cd ../poster && tectonic main.tex
```

## CI

`.github/workflows/ci.yml` runs `ruff check .`, the bounded pytest suite,
and notebook execution on a locked install -- no live TS-ICL checkpoint, no
model training, no full-grid execution.
