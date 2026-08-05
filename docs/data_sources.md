# Data sources and licensing

## Code license

All source code in this repository -- `src/coastal_gap_reconstruction/`,
`experiments/`, the code cells in `notebooks/`, `demo/`, and configuration/
build files (`pyproject.toml`) -- is licensed under the MIT License. See
`LICENSE` for the full text.

## Data and results: not MIT licensed

Everything under `data/` and `results/`, and every figure derived from
them, is derived from third-party data sources that carry their own
attribution requirements. Including these files here does not constitute a
claim of ownership over the underlying data and does not relicense it under
MIT terms. Downstream users must independently satisfy each provider's
attribution and use conditions, summarized below.

| Source / provider | Product or station | Variables used | Role in this project | Product ID / DOI | Required acknowledgement |
|---|---|---|---|---|---|
| CEAZAMet / CEAZA | Tongoy Balsa buoy (chlorophyll, sensor BTGOXD2 for oxygen), Punta Lengua de Vaca (PLV) meteorological station | `chl_mean`, `oxygen_mean_mgL`, PLV temperature/pressure/humidity/precipitation/solar radiation/wind | Reconstruction targets (both case studies) and meteorological predictors | -- (station data, www.ceazamet.cl) | Use **one** of: "Datos provistos por CEAZA, obtenidos desde www.ceazamet.cl, 2019" or "Centro de Estudios Avanzados en Zonas Áridas (CEAZA), www.ceazamet.cl, 2019" -- not both |
| NASA / PO.DAAC | GHRSST Level 4 MUR Global Foundation SST Analysis (v4.1) | `mur_sst_*`, `mur_gradient_*`, `mur_front_*` | Sea-surface-temperature predictor features | `10.5067/GHGMR-4FJ04` | Cite the current PO.DAAC dataset page |
| Copernicus Marine Service (CMEMS) | Wind reanalysis/forecast product | `wind_u_ms`, `wind_v_ms`, `wind_spd_ms` and lag/roll features | Wind predictor features | -- | Attribution to Copernicus Marine Service |
| Copernicus / OSTIA | OSTIA SST analysis | `ostia_sst_degC` | SST predictor feature | -- | Attribution to Copernicus Marine Service |
| ESA / Copernicus | SST Climate Change Initiative product | SST-related predictor features | Predictor features | -- | Attribution to Copernicus Marine Service |
| Copernicus / GlobColour | Satellite ocean-colour chlorophyll-a | `chl_cons_*`, `chl_perm_*`, patchiness/anomaly features | Satellite chlorophyll proxy predictor and TS-ICL covariate | -- | Attribution to Copernicus Marine Service |
| EUMETSAT / ASCAT | Advanced Scatterometer wind/current product | Wind and upwelling-related predictor features | Predictor features | -- | Attribution to Copernicus Marine Service |
| Copernicus / GLORYS12, MULTIOBS | Ocean current reanalysis | Current/transport/kinematic features (`data/shared/external_current_kinematic_extension.csv`) | Ocean-current predictor features (chlorophyll current-transport arms, oxygen `external_physical_plus_currents`) | -- | Attribution to Copernicus Marine Service |

CEAZAMet's hourly in-situ observations are publicly accessible at their
source (www.ceazamet.cl) and are not committed to this repository in raw
form -- only the derived daily aggregate tables in `data/` are included,
with the attribution above. Copernicus/NASA products are represented only
as derived, station-level feature values extracted at this project's study
site, never as the underlying gridded products themselves. No ownership
claim is made over any third-party data.

Files under `results/` (benchmark summaries, artificial-gap scores,
covariate mechanism summaries, reconstruction candidate outputs) are
project-derived analytical products, computed from the sources above plus
this project's own modeling and validation pipeline. They carry the same
not-MIT-licensed, attribution-encumbered status as the data they are
computed from.

## TS-ICL

TS-ICL (used in `demo/` and `notebooks/04_tsicl_and_covariates.ipynb`) is a
separate optional dependency, never vendored in this repository, governed
by its own **TS-ICL Non-Commercial License v1.0, (c) EDF SA 2026** --
distinct from this repository's MIT license. See
`docs/reproducibility.md` for citation and installation details.

If in doubt about whether a specific reuse is permitted, consult the
original provider's terms (CEAZAMet, PO.DAAC, Copernicus Marine Service)
directly.
