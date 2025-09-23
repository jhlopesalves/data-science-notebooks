---
number: 17
title: Inference Deep-Dive with statsmodels
phase: Statistical Modelling
bundles:
- bundle_stats_inference
project:
  title: Explainable Linear Model with Proper Inference
  dataset: California Housing (regression) and/or Wage data (any public wage/labour
    dataset).
  dataset_links:
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
  metrics:
  - 'Inference quality: specification tests passed, stable coefficients across perturbations.'
  nuances:
  - Separate predictive goals from inferential goals; be explicit which you pursue.
  - Use cross-validation only for predictive checks; inference uses the full model
    with diagnostics.
code_focus:
- 'OLS/GLM in statsmodels for coefficient inference: robust (HC0–HC3) standard errors;
  hypothesis testing with t/Wald; confidence and prediction intervals.'
- 'Model diagnostics: residual plots, QQ plots, heteroskedasticity tests (Breusch–Pagan,
  White), multicollinearity checks (VIF), outlier influence (Cook’s distance).'
- Information criteria (AIC/BIC) and likelihood-based model comparison; nested vs
  non-nested comparisons.
math_stats:
- Likelihood, score, Fisher information; asymptotic normality of MLE.
- Robust sandwich estimators; when inference survives model misspecification.
- Model selection by ICs; limits of hypothesis testing in observational data.
docs:
- https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
- https://www.statsmodels.org/dev/glm.html
- https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html
bibliography:
- Wooldridge — Introductory Econometrics (robust SEs, diagnostics).
- Greene — Econometric Analysis (likelihood theory).
- Gelman et al. — Regression and Other Stories (diagnostics in practice).
---

## Summary

You step back from predictive accuracy and learn to make defensible statements about effects and uncertainty. This equips you to speak with analysts, economists and scientists in their dialect.
## Project Description

Fit an OLS model with a principled feature set; produce robust SEs, diagnostic plots, and a short narrative explaining coefficient meaning, uncertainty and any specification changes motivated by diagnostics.
## Code Focus

- OLS/GLM in statsmodels for coefficient inference: robust (HC0–HC3) standard errors; hypothesis testing with t/Wald; confidence and prediction intervals.
- Model diagnostics: residual plots, QQ plots, heteroskedasticity tests (Breusch–Pagan, White), multicollinearity checks (VIF), outlier influence (Cook’s distance).
- Information criteria (AIC/BIC) and likelihood-based model comparison; nested vs non-nested comparisons.
## Math & Stats

- Likelihood, score, Fisher information; asymptotic normality of MLE.
- Robust sandwich estimators; when inference survives model misspecification.
- Model selection by ICs; limits of hypothesis testing in observational data.
## Docs

- https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.OLS.html
- https://www.statsmodels.org/dev/glm.html
- https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html
## Bibliography

- Wooldridge — Introductory Econometrics (robust SEs, diagnostics).
- Greene — Econometric Analysis (likelihood theory).
- Gelman et al. — Regression and Other Stories (diagnostics in practice).
