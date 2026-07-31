# Data sources and attribution

## 1. CEAZAMet / CEAZA attribution

The daily chlorophyll target table
(`data_public/chlorophyll/chlorophyll_daily_target.csv`) and the daily
oxygen target table (`data_public/oxygen/oxygen_daily_target.csv`, sensor
BTGOXD2) are both processed daily summary tables derived from hourly
in-situ sensor data provided by CEAZAMet, the coastal/meteorological
monitoring network operated by the Centro de Estudios Avanzados en Zonas
Áridas (CEAZA), Chile. CEAZAMet's hourly observations are publicly/
externally accessible at their source (www.ceazamet.cl) and are not
committed to this repository in raw form -- only the derived daily
aggregate tables described in `docs/methodology/target_and_gap_construction.md`
are included here, with attribution.

Required attribution (use one of the two formulations below, not both):

> Datos provistos por CEAZA, obtenidos desde www.ceazamet.cl, 2019

or

> Centro de Estudios Avanzados en Zonas Áridas (CEAZA), www.ceazamet.cl, 2019

Any use of the chlorophyll target data, or of any predictor feature in this
repository sourced from CEAZAMet station data (e.g. the PLV meteorological
station variables), should retain this attribution. This data-attribution
requirement is separate from this repository's code license (MIT, see
`LICENSE`) -- the two are independent.

## 2. NASA / PO.DAAC MUR SST attribution

Several predictor features in
`data_public/chlorophyll/chlorophyll_predictor_features_curated.csv`
(columns prefixed `mur_sst_*`, `mur_gradient_*`, `mur_front_*`) are derived
from the Multi-scale Ultra-high Resolution (MUR) Sea Surface Temperature
analysis:

> GHRSST Level 4 MUR Global Foundation Sea Surface Temperature Analysis (v4.1)

Distributed by NASA's Physical Oceanography Distributed Active Archive
Center (PO.DAAC).

PO.DAAC lists GHRSST Level 4 MUR Global Foundation Sea Surface Temperature
Analysis v4.1 with DOI `10.5067/GHGMR-4FJ04`. Users should cite the
current PO.DAAC dataset page when using the derived MUR SST features.

## 3. Copernicus / CMEMS attribution

Predictor features in this repository draw on several Copernicus Marine
Service (CMEMS) and related European Copernicus programme products,
identified by the following source product families found in the
underlying extraction (column prefixes in the curated feature table follow
the same naming where applicable):

| Source family | Likely product | Used for |
|---|---|---|
| CMEMS_WIND | Copernicus Marine wind reanalysis/forecast product | `wind_u_ms`, `wind_v_ms`, `wind_spd_ms` and related lag/roll features |
| OSTIA | OSTIA (Operational SST and Sea Ice Analysis) SST analysis, distributed via Copernicus | `ostia_sst_degC` |
| SST_CCI | ESA Sea Surface Temperature Climate Change Initiative product | SST-related predictor features |
| MUR | Multi-scale Ultra-high Resolution SST (see section 2 above; also distributed via Copernicus mirrors in some pipelines) | `mur_sst_*`, frontal/gradient features |
| CHL | Satellite-derived chlorophyll-a product(s) (e.g. GlobColour/CMEMS ocean colour), used as the satellite chlorophyll proxy covariate | `chl_cons_*`, `chl_perm_*`, `chl_patch*` features, and the TS-ICL satellite-proxy covariate |
| ASCAT | Advanced Scatterometer (ASCAT) satellite wind/current product | wind and upwelling-related predictor features |

Copernicus / CMEMS products require attribution to the Copernicus
programme and the relevant Copernicus Marine Service when used or
redistributed. This repository makes no ownership claim over any
Copernicus-sourced data; only derived station-level feature values
(extracted at the project's study site) are included here, not the
underlying gridded products themselves.

## 4. Derived results

All files under `results_public/` (benchmark summaries, reconstruction
outputs, covariate mechanism summaries, event performance summaries, and
the joined per-day candidate comparison table) are derived analytical
products created within this project, computed from the source data
described in sections 1-3 above plus this project's own modeling and
validation pipeline.

Data and derived products are provided with attribution to original
providers; users should check upstream provider terms for their intended
use.

## 5. License framing

This document describes data sources narratively. For a short,
license-focused statement of what is and is not MIT licensed in this
repository, see `DATA_LICENSE_AND_ATTRIBUTION.md` at the repository root.
In summary: code is MIT licensed; the data and derived results described
above are not MIT licensed and remain subject to the attribution
obligations listed in sections 1-4.
