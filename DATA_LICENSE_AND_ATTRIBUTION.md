# Data license and attribution

This file states, in summary/legal-framing form, what license applies to
each part of this repository. For a more detailed, narrative description of
each data source, see `docs/data_sources_and_attribution.md` -- the two
documents are kept consistent; this one is the short, license-focused
version.

## 1. Code: MIT License

All source code in this repository -- `src/coastal_gap_reconstruction/`,
the code cells in `notebooks/`, and configuration/build files
(`pyproject.toml`, `config/contracts/`) -- is licensed under the MIT
License. See `LICENSE` for the full text. Copyright (c) 2026 Matteo
Faverio.

## 2. Data and derived analytical products: NOT MIT licensed

Everything under `data_public/` and `results_public/`, and the figures
under `figures/`, is **not** covered by the MIT License above. These files
are included in this repository for reproducibility and transparency, but
they are derived from third-party data sources that carry their own
attribution requirements. Including these files here does **not**
constitute a claim of ownership over the underlying data, and does **not**
relicense the underlying data under MIT terms.

Downstream users must independently satisfy the attribution and use
conditions of each original data provider, summarized below.

### 2.1 CEAZAMet / CEAZA (in-situ station data)

`data_public/chlorophyll/chlorophyll_daily_target.csv` is a processed
daily summary derived from hourly in-situ chlorophyll-a sensor data
collected by CEAZAMet, the monitoring network operated by the Centro de
Estudios Avanzados en Zonas Áridas (CEAZA), Chile. Meteorological
predictor features sourced from the nearby Punta Lengua de Vaca (PLV)
CEAZAMet station are also covered by this attribution.

Required attribution (use both phrases):

> Datos provistos por CEAZA, obtenidos desde www.ceazamet.cl, 2019

> Centro de Estudios Avanzados en Zonas Áridas (CEAZA), www.ceazamet.cl, 2019

No raw hourly CEAZAMet data are redistributed here -- only the derived
daily aggregate table.

### 2.2 NASA / PO.DAAC -- MUR Sea Surface Temperature

Predictor features prefixed `mur_sst_*`, `mur_gradient_*`, `mur_front_*`
in `chlorophyll_predictor_features_curated.csv` are derived from:

> GHRSST Level 4 MUR Global Foundation Sea Surface Temperature Analysis (v4.1)

Distributed by NASA's Physical Oceanography Distributed Active Archive
Center (PO.DAAC).

PO.DAAC lists GHRSST Level 4 MUR Global Foundation Sea Surface Temperature
Analysis v4.1 with DOI `10.5067/GHGMR-4FJ04`. Users should cite the
current PO.DAAC dataset page when using the derived MUR SST features.

### 2.3 Copernicus Marine Service (CMEMS) and related Copernicus products

Predictor features draw on several Copernicus Marine Service (CMEMS) and
related Copernicus programme products, identified by source-product
families in the underlying extraction:

| Source family | Product | Used for |
|---|---|---|
| CMEMS_WIND | Copernicus Marine wind reanalysis/forecast product | wind components/speed and derived lag/roll features |
| OSTIA | OSTIA (Operational SST and Sea Ice Analysis), distributed via Copernicus | `ostia_sst_degC` |
| SST_CCI | ESA Sea Surface Temperature Climate Change Initiative product | SST-related predictor features |
| MUR | MUR SST (also distributed via Copernicus mirrors in some pipelines; see section 2.2 for the canonical PO.DAAC attribution) | `mur_sst_*`, frontal/gradient features |
| CHL | Satellite-derived chlorophyll-a product(s) (e.g. GlobColour/CMEMS ocean colour) | satellite chlorophyll proxy covariate, `chl_cons_*`, `chl_perm_*`, `chl_patch*` features |
| ASCAT | Advanced Scatterometer (ASCAT) satellite wind/current product | wind and upwelling-related predictor features |

Copernicus Marine Service products require attribution to the Copernicus
programme and the relevant Copernicus Marine Service per their terms of
use. No ownership claim is made over any Copernicus-sourced data in this
repository -- only derived, station-level feature values extracted at this
project's study site are included, not the underlying gridded products.

### 2.4 Derived results

Files under `results_public/` (benchmark summaries, artificial-gap scores,
covariate mechanism summaries, event performance summaries, reconstruction
candidate outputs, and the joined per-day candidate comparison table) are
project-derived analytical products. They are computed from the source
data described above plus this project's own modeling and validation
pipeline, and are redistributed here with attribution to the upstream
providers. These derived products are **not** MIT licensed and the
upstream datasets they are computed from remain the property of their
respective providers; treat them as research outputs provided for
transparency and reproducibility, not as freely relicensable data.

## 3. Summary

| Path | License / status |
|---|---|
| `src/`, notebook code cells, `pyproject.toml`, `config/` | MIT License (see `LICENSE`) |
| `data_public/`, `results_public/`, `figures/` | Not MIT licensed; attribution-encumbered, see sections 2.1-2.4 above and `docs/data_sources_and_attribution.md` |

If in doubt about whether a specific reuse is permitted, consult the
original provider's terms (CEAZAMet, PO.DAAC, Copernicus Marine Service)
directly.
