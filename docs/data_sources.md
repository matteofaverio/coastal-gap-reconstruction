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
| NASA / PO.DAAC (JPL MUR MEaSUREs) | GHRSST Level 4 MUR Global Foundation SST Analysis (v4.1) | `mur_sst_*`, `mur_gradient_*`, `mur_front_*` | Sea-surface-temperature predictor features | DOI `10.5067/GHGMR-4FJ04` | Cite the current PO.DAAC dataset page |
| Met Office / GHRSST (OSTIA) | The Current Configuration of the OSTIA System for Operational Production of Foundation SST and Ice Concentration Analyses (Good et al., *Remote Sensing*, 2020) | `ostia_sst_degC` | SST predictor feature | DOI `10.3390/rs12040720` | Cite Good et al. (2020) |
| Copernicus Marine Service | Global Ocean Hourly Reprocessed Sea Surface Wind and Stress from Scatterometer and Model | `wind_u_ms`, `wind_v_ms`, `wind_spd_ms` and lag/roll features | Wind predictor features | Product `WIND_GLO_PHY_L4_MY_012_006`, DOI `10.48670/moi-00185` | Attribution to Copernicus Marine Service |
| Copernicus Marine Service (ACRI-ST) | Global Ocean Colour (Copernicus-GlobColour), Bio-Geo-Chemical, L4 | `chl_cons_*`, `chl_perm_*`, patchiness/anomaly features | Satellite chlorophyll proxy predictor and TS-ICL covariate | Product `OCEANCOLOUR_GLO_BGC_L4_MY_009_104`, DOI `10.48670/moi-00281` | Attribution to Copernicus Marine Service |
| Copernicus Marine Service | Global Ocean Physics Reanalysis (GLORYS12) | Current/transport/kinematic features (`data/shared/external_current_kinematic_extension.csv`) | Ocean-current predictor features (chlorophyll current-transport arms, oxygen `external_physical_plus_currents`) | Product `GLOBAL_MULTIYEAR_PHY_001_030`, DOI `10.48670/moi-00021` | Attribution to Copernicus Marine Service |
| Copernicus Marine Service | Global Total (COPERNICUS-GLOBCURRENT), Ekman and Geostrophic Currents at the Surface and 15m (MULTIOBS) | Current/transport/kinematic features (`data/shared/external_current_kinematic_extension.csv`) | Ocean-current predictor features (chlorophyll current-transport arms, oxygen `external_physical_plus_currents`) | Product `MULTIOBS_GLO_PHY_MYNRT_015_003`, DOI `10.48670/mds-00327` | Attribution to Copernicus Marine Service |

The CMEMS wind product above is itself a blend (bias-corrected ERA5 fields
using ASCAT Metop-A/B/C and historical QuikSCAT/ERS scatterometer
observations) -- ASCAT is a contributing observational source within that
one product, not a separately extracted dataset in this project.

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
