---
number: 16
title: Phase Project — Defensible Tabular Model
phase: Core ML
bundles:
- bundle_fairness_calibration
- bundle_model_selection
project:
  title: Defensible Income or Credit Risk Model
  dataset: Adult or Home Credit.
  description: Ship a repo with MLflow runs, a final calibrated classifier, interpretability
    artefacts (SHAP summary, PDPs), and a model card capturing intended use, limits
    and fairness considerations.
  dataset_links:
  - https://archive.ics.uci.edu/dataset/2/adult
  - https://www.kaggle.com/c/home-credit-default-risk
  metrics:
  - As Week 15; plus business-aligned threshold metrics (precision at k, cost curve).
  nuances:
  - 'Reproducibility: seeds, pinned library versions, deterministic data splits.'
  - 'Ethical boundaries: make explicit what the model must not be used for.'
code_focus:
- End-to-end Pipeline with robust preprocessing, nested CV, calibration and interpretability
  (permutation importance, PDP/ICE, SHAP for boosted trees).
- 'Documentation discipline: README with data cards and model cards; MLflow tracking
  for experiments.'
math_stats:
- Consolidation of bias–variance, regularisation, deviance and proper scoring.
- 'Uncertainty communication: confidence intervals for CV differences via bootstrap.'
docs:
- https://mlflow.org/docs/latest/index.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://scikit-learn.org/stable/modules/permutation_importance.html
- https://shap.readthedocs.io/en/latest/
bibliography:
- Mitchell et al. — Model Cards for Model Reporting (FAT* 2019).
- Molnar — Interpretable ML (model reporting chapter).
---

## Summary

You compress four weeks of learning into one professional-grade artefact. By the end, you can defend modelling choices to sceptical reviewers and non-technical stakeholders.
## Code Focus

- End-to-end Pipeline with robust preprocessing, nested CV, calibration and interpretability (permutation importance, PDP/ICE, SHAP for boosted trees).
- Documentation discipline: README with data cards and model cards; MLflow tracking for experiments.
## Math & Stats

- Consolidation of bias–variance, regularisation, deviance and proper scoring.
- Uncertainty communication: confidence intervals for CV differences via bootstrap.
## Docs

- https://mlflow.org/docs/latest/index.html
- https://scikit-learn.org/stable/modules/calibration.html
- https://scikit-learn.org/stable/modules/permutation_importance.html
- https://shap.readthedocs.io/en/latest/
## Bibliography

- Mitchell et al. — Model Cards for Model Reporting (FAT* 2019).
- Molnar — Interpretable ML (model reporting chapter).
