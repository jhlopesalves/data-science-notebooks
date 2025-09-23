---
number: 18
title: ARIMA/SARIMAX and Stationarity
phase: Time Series
bundles:
- bundle_statsmodels_ts
- bundle_fpp3
project:
  title: Energy or Traffic Forecaster
  dataset: PJM Hourly Energy Consumption or Metro Interstate Traffic Volume.
  description: Build SARIMAX with weather/holiday regressors; produce rolling-origin
    forecasts and compare to naive/seasonal naive. Discuss residual autocorrelation
    and refit strategy.
  dataset_links:
  - https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
  - https://www.kaggle.com/datasets/rgupta12/metro-interstate-traffic-volume
  metrics:
  - sMAPE, MASE and RMSE; coverage of 80/95% forecast intervals.
  nuances:
  - Data gaps, timezone issues and daylight-saving transitions.
  - Non-stationarity vs structural breaks; when differencing harms interpretability.
code_focus:
- Seasonal decomposition; stationarity checks (ADF); ARIMA/SARIMAX identification
  with ACF/PACF heuristics.
- External regressors (weather/holidays), differencing and seasonal differencing;
  rolling-origin evaluation; confidence vs prediction intervals.
- 'Baseline comparisons: naive, seasonal naive, ETS (if available) vs ARIMA; holiday
  effects via regressors.'
math_stats:
- AR, MA, ARMA/ARIMA state-space view; invertibility and stationarity constraints.
- Forecast error variance; why backtesting must respect temporal order.
- Information criteria for order selection.
docs:
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
bibliography:
- 'Hyndman & Athanasopoulos — Forecasting: Principles and Practice (fpp3).'
- Box, Jenkins, Reinsel & Ljung — Time Series Analysis (ARIMA bible).
---

## Summary

You learn principled classical forecasting, the baseline that modern methods must beat. You will never again backtest with shuffled folds.
## Code Focus

- Seasonal decomposition; stationarity checks (ADF); ARIMA/SARIMAX identification with ACF/PACF heuristics.
- External regressors (weather/holidays), differencing and seasonal differencing; rolling-origin evaluation; confidence vs prediction intervals.
- Baseline comparisons: naive, seasonal naive, ETS (if available) vs ARIMA; holiday effects via regressors.
## Math & Stats

- AR, MA, ARMA/ARIMA state-space view; invertibility and stationarity constraints.
- Forecast error variance; why backtesting must respect temporal order.
- Information criteria for order selection.
## Docs

- https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html
- https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
## Bibliography

- Hyndman & Athanasopoulos — Forecasting: Principles and Practice (fpp3).
- Box, Jenkins, Reinsel & Ljung — Time Series Analysis (ARIMA bible).
