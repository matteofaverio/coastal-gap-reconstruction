# Event / high-chlorophyll limitation

## The finding

Every reconstruction method evaluated in this benchmark performs
measurably worse on high-chlorophyll ("event") days than on background
days, and every method under-predicts the amplitude of those events. This
is an unresolved limitation as of this release, not a solved problem.

An "event" day is defined as a day whose true chlorophyll value exceeds the
90th percentile of all eligible observed values (see
`docs/methodology/validation_protocol.md`).

## Numbers (artificial-gap validation pool)

From `results_public/chlorophyll/chlorophyll_event_performance_summary.csv`,
mean absolute error (log10 chlorophyll scale) on event vs. non-event days:

| Method | MAE, event days | MAE, non-event days | Event penalty (event - non-event) | Amplitude bias, event days |
|---|---|---|---|---|
| Linear interpolation baseline | 0.323 | 0.230 | +0.093 | -0.004 |
| Monthly climatology baseline | 0.348 | 0.355 | -0.007 | +0.009 |
| Gaussian process (time-only) | 0.316 | 0.221 | +0.095 | -0.071 |
| Engineered hybrid pipeline | 0.309 | 0.222 | +0.087 | -0.046 |
| Gap-edge residual model | 0.281 | 0.205 | +0.076 | -0.096 |
| TS-ICL, target-only | 0.301 | 0.211 | +0.090 | -0.064 |
| TS-ICL, satellite proxy covariate | 0.292 | 0.205 | +0.086 | -0.052 |
| TS-ICL, wind/upwelling covariate | 0.300 | 0.207 | +0.093 | -0.083 |
| TS-ICL, curated physical set | 0.299 | 0.206 | +0.093 | -0.082 |

(amplitude bias is reported in log10 chlorophyll units; negative values
indicate the method systematically under-predicts the magnitude of events)

Every method shows a positive event penalty (worse error on event days than
background days) and a negative amplitude bias (under-prediction) on event
days, with the partial exception of the climatology baseline, which has a
roughly flat penalty but the worst absolute error of all methods on both
regimes.

The satellite-proxy TS-ICL configuration has the best event-day MAE and
event penalty among the leading candidate methods, but still meaningfully
under-predicts event amplitude (amplitude bias -0.052) and has a
substantially higher error on event days than on background days.

## Why this matters

High-chlorophyll events are the periods of greatest scientific and
practical interest (harmful algal blooms, productivity pulses), so
systematic under-prediction during exactly these periods is a material
limitation for any downstream use of these reconstructions. Until this is
resolved, reconstructed values during flagged or suspected event periods
should be treated with extra caution, and any analysis sensitive to peak
amplitude should not rely on these reconstructions without independent
verification.
