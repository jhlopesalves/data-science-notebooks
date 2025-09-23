---
number: 5
title: Least Squares, Regularisation Paths, and Baselines
phase: Supervised (Tabular)
bundles:
- bundle_regularisation
project:
  title: Regularised Regression on Housing
  dataset: California Housing (sklearn)
  description: Train OLS, Ridge, Lasso, Elastic Net on California Housing with a robust
    preprocessing Pipeline. Produce coefficient-path plots, validation curves, and
    a stability selection summary across resamples.
  dataset_links:
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
  metrics:
  - Hold-out RMSE/MAE
  - Coefficient stability summary
  - Learning and validation curves
  nuances:
  - Compare raw vs log-target (TransformedTargetRegressor)
  - Discuss interpretability vs performance trade-offs
code_focus:
- OLS baseline; Ridge, Lasso, Elastic Net with *_CV; standardise features; inspect
  coefficient stability under resampling.
- Plot regularisation paths; compare validation curves; nested CV vs single CV for
  fair model selection.
- Add polynomial/features interactions via PolynomialFeatures or patsy-style design;
  caution about multicollinearity.
math_stats:
- 'Derive normal equations x̂ = (XᵀX)⁻¹Xᵀy (geometric view: projection).'
- Bias–variance trade-off under L2; sparsity intuition for L1; Elastic Net as compromise.
- Condition number and ill-conditioning; role of standardisation.
docs:
- '[sklearn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)'
- '[sklearn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)'
- '[sklearn ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)'
- '[sklearn: PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)'
- '[California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)'
bibliography:
- '*ISLR (Python), Ch. 3 & 6* — James et al. — (2023) — https://www.statlearning.com/'
- '*Elements of Statistical Learning, Ch. 3* — Hastie et al. — (2009) — https://hastie.su.domains/ElemStatLearn/'
---

## Summary

You make linear models do real work. The point is not just to fit but to reason about stability and generalisation. You should end the week with judgement about when ‘linear with features’ is enough.
## Code Focus

- OLS baseline; Ridge, Lasso, Elastic Net with *_CV; standardise features; inspect coefficient stability under resampling.
- Plot regularisation paths; compare validation curves; nested CV vs single CV for fair model selection.
- Add polynomial/features interactions via PolynomialFeatures or patsy-style design; caution about multicollinearity.
## Math & Stats

- Derive normal equations x̂ = (XᵀX)⁻¹Xᵀy (geometric view: projection).
- Bias–variance trade-off under L2; sparsity intuition for L1; Elastic Net as compromise.
- Condition number and ill-conditioning; role of standardisation.
## Docs

- [sklearn Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)
- [sklearn Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)
- [sklearn ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- [sklearn: PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)
- [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html)
## Bibliography

- *ISLR (Python), Ch. 3 & 6* — James et al. — (2023) — https://www.statlearning.com/
- *Elements of Statistical Learning, Ch. 3* — Hastie et al. — (2009) — https://hastie.su.domains/ElemStatLearn/
