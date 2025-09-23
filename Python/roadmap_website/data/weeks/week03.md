---
number: 3
title: Pipelines, ColumnTransformer, Honest CV
phase: Foundations
bundles:
- bundle_linear_algebra
- bundle_sklearn_core
project:
  title: Fair Adult Income Baseline
  dataset: UCI Adult (Census Income)
  dataset_links:
  - https://archive.ics.uci.edu/dataset/2/adult
  metrics:
  - ROC AUC, PR AUC on test
  - Report fold-wise variance
  - Baseline better than DummyClassifier by substantial margin
  nuances:
  - Document the decision about handling ‘?’, strip whitespace in categorical labels
  - Keep a pristine test set
code_focus:
- 'Compose preprocessing with ColumnTransformer (numeric: impute+scale; categorical:
  impute+one-hot; dates: custom transformer).'
- 'Build full Pipeline: preprocessing → estimator; ensure no leakage by fitting only
  inside cross-validation.'
- Train/validation/test protocol; StratifiedKFold vs KFold; repeated CV for variance
  estimates.
- Custom sklearn-compatible transformers (fit/transform signatures), including target-aware
  transforms via TransformedTargetRegressor.
math_stats:
- Bias–variance via repeated CV estimates; what high variance in scores tells you.
- 'Linear algebra refresher: projections and rank to prepare for least squares next
  week.'
docs:
- '[sklearn: Pipeline & ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)'
- '[sklearn: Cross-validation strategies](https://scikit-learn.org/stable/modules/cross_validation.html)'
bibliography:
- '*Elements of Statistical Learning (ESL)* — Hastie, Tibshirani, Friedman — (2009)
  — https://hastie.su.domains/ElemStatLearn/'
- '*CS229 Supervised Learning notes (model selection section)* — Stanford — (2019)
  — https://cs229.stanford.edu/'
---

## Summary

You operationalise honest evaluation. Everything—imputation, encoding, scaling—lives inside the Pipeline so cross-validation reflects reality. This is the guardrail that prevents leaky scores that look great and fail in production.
## Project Description

Construct an end-to-end Pipeline for Adult: robust preprocessing, logistic baseline with CV, and a hold-out test set. Produce a short model card: features used, preprocessing steps, CV protocol, and limitations.
## Code Focus

- Compose preprocessing with ColumnTransformer (numeric: impute+scale; categorical: impute+one-hot; dates: custom transformer).
- Build full Pipeline: preprocessing → estimator; ensure no leakage by fitting only inside cross-validation.
- Train/validation/test protocol; StratifiedKFold vs KFold; repeated CV for variance estimates.
- Custom sklearn-compatible transformers (fit/transform signatures), including target-aware transforms via TransformedTargetRegressor.
## Math & Stats

- Bias–variance via repeated CV estimates; what high variance in scores tells you.
- Linear algebra refresher: projections and rank to prepare for least squares next week.
## Docs

- [sklearn: Pipeline & ColumnTransformer](https://scikit-learn.org/stable/modules/compose.html#columntransformer-for-heterogeneous-data)
- [sklearn: Cross-validation strategies](https://scikit-learn.org/stable/modules/cross_validation.html)
## Bibliography

- *Elements of Statistical Learning (ESL)* — Hastie, Tibshirani, Friedman — (2009) — https://hastie.su.domains/ElemStatLearn/
- *CS229 Supervised Learning notes (model selection section)* — Stanford — (2019) — https://cs229.stanford.edu/
