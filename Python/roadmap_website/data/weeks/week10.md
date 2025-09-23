---
number: 10
title: Regularised GLMs for Non-Gaussian Targets
phase: Core ML
bundles:
- bundle_glm
project:
  title: Frequency–Severity Insurance Modelling
  dataset: Insurance claims (frequency and severity) + Bike sharing counts (as a second
    domain).
  dataset_links:
  - https://www.kaggle.com/competitions/allstate-claims-severity
  - https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html
  - https://www.kaggle.com/c/bike-sharing-demand
  metrics:
  - 'Frequency: Poisson deviance on a temporal hold-out.'
  - 'Severity: Gamma/Tweedie deviance and Pinball loss at τ∈{0.5,0.9}.'
  - 'Pure premium: Tweedie deviance; practical lift over baseline rate card.'
  nuances:
  - Stability of estimates with high-cardinality categoricals; target encoding only
    within CV folds.
  - Heavy-tail handling (Winsorising; log-scale modelling) and implications for calibration.
code_focus:
- 'Implement Poisson, Gamma and Tweedie regression in scikit-learn: PoissonRegressor,
  GammaRegressor, TweedieRegressor with power in {1 (Poisson), 1<p<2 (Tweedie compound
  Poisson), 2 (Gamma)} and alpha (L2) regularisation; compare with statsmodels.GLM
  for full inference (SEs, Wald tests).'
- Engineering exposure offsets (log link), canonical links, and variance functions;
  handle zero-inflation via two-part frequency–severity modelling (count model + positive-only
  severity model).
- 'Robust evaluation beyond R^2/MAE: mean deviance, Poisson/Gamma deviance, Tweedie
  deviance; calibration plots for count and claim-size models.'
- 'Pipeline good practice: ColumnTransformer (one-hot for categoricals; target/Box-Cox
  where appropriate), outlier flooring/capping for heavy-tailed severity, and grouped
  CV for temporal leakage control.'
- 'Diagnostics: residuals on the response and Pearson scales; influence of high-leverage
  points; compare scikit-learn (prediction-centric) vs statsmodels (inference-centric).'
math_stats:
- Exponential family; canonical links; mean-variance relationship.
- Convexity of GLM negative log-likelihood; L2 regularisation and bias–variance trade-off.
- Deviance as twice the log-likelihood ratio and its interpretation.
- 'Two-part models: E[Y]=E[N]*E[severity | N>0]; Tweedie as frequency–severity compound.'
docs:
- https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
- https://www.statsmodels.org/dev/glm.html
- https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html
bibliography:
- Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, Ch. 4 & 6.
- Dobson & Barnett — An Introduction to Generalized Linear Models (CRC).
- Kuhn & Silge — Tidy Modeling with R (GLM chapters for conceptual clarity).
- Frees — Regression Modeling with Actuarial and Financial Applications (GLM/Tweedie).
---

## Summary

You move from Gaussian regression to the GLM toolkit that dominates operational ML on counts and costs. The week forces careful thinking about distributions, link functions and evaluation: MSE is the wrong ruler for counts and claims. By building the classical frequency–severity pipeline and comparing it to a Tweedie pure-premium model, you will learn when decomposition aids interpretability and when Tweedie’s parsimony wins. You will also practise deviance-based selection and produce business-grade calibration analyses.
## Project Description

Build a two-part model: (1) frequency via PoissonRegressor with an exposure term; (2) severity via GammaRegressor or TweedieRegressor (p≈1.5–1.9) on strictly positive claims. Compare to a single Tweedie pure-premium model. Produce deviance-based model selection, calibration plots, and an interpretability report explaining which risk factors affect frequency versus severity.
## Code Focus

- Implement Poisson, Gamma and Tweedie regression in scikit-learn: PoissonRegressor, GammaRegressor, TweedieRegressor with power in {1 (Poisson), 1<p<2 (Tweedie compound Poisson), 2 (Gamma)} and alpha (L2) regularisation; compare with statsmodels.GLM for full inference (SEs, Wald tests).
- Engineering exposure offsets (log link), canonical links, and variance functions; handle zero-inflation via two-part frequency–severity modelling (count model + positive-only severity model).
- Robust evaluation beyond R^2/MAE: mean deviance, Poisson/Gamma deviance, Tweedie deviance; calibration plots for count and claim-size models.
- Pipeline good practice: ColumnTransformer (one-hot for categoricals; target/Box-Cox where appropriate), outlier flooring/capping for heavy-tailed severity, and grouped CV for temporal leakage control.
- Diagnostics: residuals on the response and Pearson scales; influence of high-leverage points; compare scikit-learn (prediction-centric) vs statsmodels (inference-centric).
## Math & Stats

- Exponential family; canonical links; mean-variance relationship.
- Convexity of GLM negative log-likelihood; L2 regularisation and bias–variance trade-off.
- Deviance as twice the log-likelihood ratio and its interpretation.
- Two-part models: E[Y]=E[N]*E[severity | N>0]; Tweedie as frequency–severity compound.
## Docs

- https://scikit-learn.org/stable/modules/linear_model.html#generalized-linear-regression
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html
- https://www.statsmodels.org/dev/glm.html
- https://www.statsmodels.org/stable/generated/statsmodels.genmod.generalized_linear_model.GLM.html
## Bibliography

- Hastie, Tibshirani & Friedman — The Elements of Statistical Learning, Ch. 4 & 6.
- Dobson & Barnett — An Introduction to Generalized Linear Models (CRC).
- Kuhn & Silge — Tidy Modeling with R (GLM chapters for conceptual clarity).
- Frees — Regression Modeling with Actuarial and Financial Applications (GLM/Tweedie).
