---
number: 19
title: 'Modern Time Series: Prophet, sktime, darts'
phase: Time Series
bundles:
- bundle_prophet
- bundle_sktime_darts
project:
  title: 'Forecasting Bake-off: Classical vs Modern'
  dataset: UCI Electricity Load Diagrams 2011–2014 (or PJM hourly).
  description: Implement SARIMAX vs Prophet vs a sktime feature-based regressor (e.g.,
    HGB with lagged features). Use identical rolling-origin evaluation; visualise
    error profiles around holidays and regime shifts.
  dataset_links:
  - https://archive.ics.uci.edu/ml/datasets/electricityloaddiagrams20112014
  - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
  metrics:
  - MASE/sMAPE on multiple horizons; interval coverage.
  - 'Operational metrics: training/forecast latency.'
  nuances:
  - Changepoint prior scale tuning in Prophet; avoid overfitting trends.
  - Feature leakage when creating calendar/weather features.
code_focus:
- Prophet for trend/seasonality/holiday with automatic changepoints; parameter sensitivity
  and diagnostics.
- sktime pipelines for feature-based forecasting (lagged features + tree ensembles)
  and model comparison.
- darts for quick prototypes (regressors, backtesting utilities); compare to SARIMAX
  on the same rolling windows.
math_stats:
- Additive seasonality and piecewise linear trends; regularisation of changepoints.
- Why feature-based tree models can outperform ARIMA on complex exogenous signals.
- Evaluation with expanding vs sliding windows; leakage-free lag construction.
docs:
- https://facebook.github.io/prophet/
- https://www.sktime.net/en/stable/
- https://unit8co.github.io/darts/
bibliography:
- Hyndman & Athanasopoulos — fpp3 (modern methods and evaluation).
- Taylor & Letham — Forecasting at Scale (Prophet paper).
---

## Summary

You extend beyond ARIMA to the pragmatic tools used in industry for irregular, holiday-driven signals. The emphasis is consistency of evaluation and engineering-grade feature handling.
## Code Focus

- Prophet for trend/seasonality/holiday with automatic changepoints; parameter sensitivity and diagnostics.
- sktime pipelines for feature-based forecasting (lagged features + tree ensembles) and model comparison.
- darts for quick prototypes (regressors, backtesting utilities); compare to SARIMAX on the same rolling windows.
## Math & Stats

- Additive seasonality and piecewise linear trends; regularisation of changepoints.
- Why feature-based tree models can outperform ARIMA on complex exogenous signals.
- Evaluation with expanding vs sliding windows; leakage-free lag construction.
## Docs

- https://facebook.github.io/prophet/
- https://www.sktime.net/en/stable/
- https://unit8co.github.io/darts/
## Bibliography

- Hyndman & Athanasopoulos — fpp3 (modern methods and evaluation).
- Taylor & Letham — Forecasting at Scale (Prophet paper).
