---
number: 9
title: 'Classification Essentials: Logistic Regression, Imbalance, Calibration'
phase: Supervised (Tabular)
bundles:
- bundle_calibration
- bundle_metrics
project:
  title: 'Credit Risk: Calibrated Classifier'
  dataset: Kaggle Give Me Some Credit, Home Credit Default Risk (optional larger)
  dataset_links:
  - https://www.kaggle.com/c/3136
  - https://www.kaggle.com/competitions/home-credit-default-risk
  metrics:
  - ROC AUC and PR AUC on hold-out
  - Calibration (Brier score) improved after calibration
  - Fairness slice table computed and discussed
  nuances:
  - Do not leak resampling across folds; use Pipeline
  - Document how thresholds map to false positive costs
code_focus:
- LogisticRegression (liblinear/saga solvers); feature scaling and regularisation;
  interpret coefficients and odds ratios.
- 'Imbalance handling: class weights vs resampling (SMOTE/SMOTENC) with imbalanced-learn;
  stratified CV; threshold tuning for business metrics.'
- 'Probability calibration: reliability diagrams, CalibratedClassifierCV (sigmoid
  vs isotonic), CalibrationDisplay; integrate into Pipeline.'
- 'Fairness quick pass: basic group metrics (demographic parity difference, equal
  opportunity) using fairlearn; caveats on causal interpretation.'
math_stats:
- Log-likelihood for Bernoulli GLM; link functions; proper scoring rules (log loss,
  Brier).
- Calibration vs discrimination; why AUC can be high but calibration poor.
docs:
- '[sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)'
- '[sklearn Calibration (User Guide)](https://scikit-learn.org/stable/modules/calibration.html)'
- '[CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)'
- '[calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)'
- '[imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/over_sampling.html)'
- '[fairlearn quickstart](https://fairlearn.org/main/about/quickstart.html)'
bibliography:
- '*ISLR (Python), Ch. 4* — James et al. — (2023) — https://www.statlearning.com/'
- '*Model Cards for Model Reporting* — Mitchell et al. — (2019) — https://arxiv.org/abs/1810.03993'
---

## Summary

This is the first serious classifier you could ship. You learn to manage imbalance, scrutinise probabilities, and tie thresholds to costs. A simple logistic model—with care—beats many naive complex models.
## Project Description

Build a logistic baseline with robust preprocessing; address imbalance with class_weight vs SMOTE; evaluate ROC AUC, PR AUC, and calibration. Produce a reliability diagram and apply isotonic/sigmoid calibration; report threshold choices tied to business costs. Include a one-page model card with fairness slice metrics.
## Code Focus

- LogisticRegression (liblinear/saga solvers); feature scaling and regularisation; interpret coefficients and odds ratios.
- Imbalance handling: class weights vs resampling (SMOTE/SMOTENC) with imbalanced-learn; stratified CV; threshold tuning for business metrics.
- Probability calibration: reliability diagrams, CalibratedClassifierCV (sigmoid vs isotonic), CalibrationDisplay; integrate into Pipeline.
- Fairness quick pass: basic group metrics (demographic parity difference, equal opportunity) using fairlearn; caveats on causal interpretation.
## Math & Stats

- Log-likelihood for Bernoulli GLM; link functions; proper scoring rules (log loss, Brier).
- Calibration vs discrimination; why AUC can be high but calibration poor.
## Docs

- [sklearn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [sklearn Calibration (User Guide)](https://scikit-learn.org/stable/modules/calibration.html)
- [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
- [calibration_curve](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.calibration_curve.html)
- [imbalanced-learn SMOTE](https://imbalanced-learn.org/stable/over_sampling.html)
- [fairlearn quickstart](https://fairlearn.org/main/about/quickstart.html)
## Bibliography

- *ISLR (Python), Ch. 4* — James et al. — (2023) — https://www.statlearning.com/
- *Model Cards for Model Reporting* — Mitchell et al. — (2019) — https://arxiv.org/abs/1810.03993
