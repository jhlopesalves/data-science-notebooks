---
number: 13
title: Boosting (HGB, LightGBM, XGBoost) and Inspection
phase: Core ML
bundles:
- bundle_boosting
project:
  title: Boosted Baselines for California Housing
  dataset: 'California Housing; optional: Home Credit Default Risk (Kaggle) for classification.'
  dataset_links:
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
  - https://www.kaggle.com/c/home-credit-default-risk
  metrics:
  - 'Regression: RMSE and relative RMSE vs linear baseline, SHAP stability across
    folds.'
  - 'Classification (optional): ROC-AUC, PR-AUC, ECE.'
  nuances:
  - Explain why histogram binning changes split behaviour and explains speed.
  - Show when monotone constraints align with domain knowledge (e.g., income ↑ → price
    ↑).
code_focus:
- HistGradientBoostingClassifier/Regressor with categorical handling and early_stopping;
  compare to sklearn GradientBoosting for understanding.
- 'LightGBM (Python API): histogram-based splits, num_leaves/max_depth, min_data_in_leaf,
  feature_fraction, learning_rate, early stopping; monotonic constraints for domain-safe
  monotone effects.'
- XGBoost as an alternative; GPU awareness; sparse awareness.
- 'Inspection toolbox: SHAP (TreeExplainer) for global and local explanations; partial
  dependence/ICE and interaction plots; permutation importance for stability checks.'
math_stats:
- Boosting as gradient descent in function space; shrinkage and regularisation.
- Additive trees and bias–variance behaviour; when monotone constraints trade variance
  for validity.
- 'Shapley values: additivity, local accuracy; caveats in causal interpretation.'
docs:
- https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- https://lightgbm.readthedocs.io/
- https://xgboost.readthedocs.io/
- https://scikit-learn.org/stable/modules/partial_dependence.html
- https://shap.readthedocs.io/en/latest/
bibliography:
- ESL — Ch. 10 (Boosting).
- 'Chen & Guestrin — XGBoost: A Scalable Tree Boosting System (KDD 2016).'
- Lundberg & Lee — A Unified Approach to Interpreting Model Predictions (NeurIPS 2017).
---

## Summary

You adopt the modern workhorse for tabular ML. The week balances raw predictive performance with transparent, reproducible explanation. You will be able to defend boosted models with principled plots and stability checks rather than hand-waving.
## Project Description

Tune HistGradientBoosting and LightGBM (with early stopping) on California Housing; build a monotone-constrained variant; produce SHAP summary and PDPs. For the optional classification dataset, replicate the workflow with class weighting and calibration.
## Code Focus

- HistGradientBoostingClassifier/Regressor with categorical handling and early_stopping; compare to sklearn GradientBoosting for understanding.
- LightGBM (Python API): histogram-based splits, num_leaves/max_depth, min_data_in_leaf, feature_fraction, learning_rate, early stopping; monotonic constraints for domain-safe monotone effects.
- XGBoost as an alternative; GPU awareness; sparse awareness.
- Inspection toolbox: SHAP (TreeExplainer) for global and local explanations; partial dependence/ICE and interaction plots; permutation importance for stability checks.
## Math & Stats

- Boosting as gradient descent in function space; shrinkage and regularisation.
- Additive trees and bias–variance behaviour; when monotone constraints trade variance for validity.
- Shapley values: additivity, local accuracy; caveats in causal interpretation.
## Docs

- https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting
- https://lightgbm.readthedocs.io/
- https://xgboost.readthedocs.io/
- https://scikit-learn.org/stable/modules/partial_dependence.html
- https://shap.readthedocs.io/en/latest/
## Bibliography

- ESL — Ch. 10 (Boosting).
- Chen & Guestrin — XGBoost: A Scalable Tree Boosting System (KDD 2016).
- Lundberg & Lee — A Unified Approach to Interpreting Model Predictions (NeurIPS 2017).
