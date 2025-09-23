---
number: 15
title: Model Selection Without Leakage
phase: Core ML
bundles:
- bundle_model_selection
project:
  title: Honest Model Bake-off
  dataset: Adult Income (again) or Credit Card Default (UCI).
  dataset_links:
  - https://archive.ics.uci.edu/dataset/2/adult
  - https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
  metrics:
  - Outer-fold ROC-AUC, PR-AUC, ECE; fairness deltas (e.g., demographic parity difference).
  nuances:
  - All preprocessing inside the Pipeline; split by groups if repeated IDs exist.
  - Report score distributions, not only means.
code_focus:
- GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV; nested CV for unbiased model
  assessment.
- StratifiedKFold/GroupKFold/TimeSeriesSplit; target leakage audits in Pipelines;
  scorer selection (ROC-AUC vs PR-AUC for imbalance).
- 'Fairness as part of selection: add calibration and subgroup metrics when choosing
  models; report variance across folds.'
math_stats:
- Bias of reusing validation; why nested CV approximates generalisation.
- Multiple comparisons and optimistic bias; variance of CV estimates and CIs.
- Proper vs improper scoring rules.
docs:
- https://scikit-learn.org/stable/modules/grid_search.html
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://scikit-learn.org/stable/modules/compose.html#pipeline
bibliography:
- Kohavi (1995) — A study of cross-validation and bootstrap for accuracy estimation.
- Cawley & Talbot (2010) — On over-fitting in model selection and performance evaluation.
---

## Summary

You learn to avoid the gravest sin in ML—leakage—and to produce trustworthy estimates. The outcome is a ‘bake-off’ template you will reuse for years.
## Project Description

Compare logistic + regularisation, RandomForest and HistGB with nested CV; incorporate calibration and subgroup metrics (e.g., by sex and race where available). Deliver a short model-selection report with justification, not just scores.
## Code Focus

- GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV; nested CV for unbiased model assessment.
- StratifiedKFold/GroupKFold/TimeSeriesSplit; target leakage audits in Pipelines; scorer selection (ROC-AUC vs PR-AUC for imbalance).
- Fairness as part of selection: add calibration and subgroup metrics when choosing models; report variance across folds.
## Math & Stats

- Bias of reusing validation; why nested CV approximates generalisation.
- Multiple comparisons and optimistic bias; variance of CV estimates and CIs.
- Proper vs improper scoring rules.
## Docs

- https://scikit-learn.org/stable/modules/grid_search.html
- https://scikit-learn.org/stable/modules/cross_validation.html
- https://scikit-learn.org/stable/modules/compose.html#pipeline
## Bibliography

- Kohavi (1995) — A study of cross-validation and bootstrap for accuracy estimation.
- Cawley & Talbot (2010) — On over-fitting in model selection and performance evaluation.
