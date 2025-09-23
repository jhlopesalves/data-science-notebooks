---
number: 6
title: 'Engineering Habits: Testing, Docs, and Experiment Tracking'
phase: Foundations
bundles:
- bundle_mlops_hygiene
project:
  title: Experiment-Ready Baselines
  dataset: California Housing, Titanic
  description: Refactor Week 5 housing and Week 1 Titanic code into modules with tests.
    Add MLflow logging (params, metrics, figures) and produce an experiments README
    summarising what each run tested.
  dataset_links:
  - https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
  - https://www.kaggle.com/c/titanic
  metrics:
  - '>=85% test coverage on custom code'
  - Passing CI locally (pre-commit)
  - MLflow runs reproducible by another machine
  nuances:
  - Treat figures as artifacts; store environment (pip freeze) with runs
  - Keep selective tests fast with parametrisation
code_focus:
- 'pytest: parametrised tests for data transforms and metrics; fixtures for small
  toy frames; coverage reports.'
- Docstring standards (NumPy style) and minimal API docs; README checklists; CHANGELOG
  and semantic versioning.
- 'MLflow for experiment tracking: parameters, metrics, artifacts; compare runs; record
  preprocessing and model versions.'
math_stats:
- 'Consolidation of Weeks 1–5: variance of CV estimates; simple power calculations
  to decide fold counts; when to use repeated CV.'
- No new theory—focus on rigour and reproducibility.
docs:
- '[pytest parametrisation](https://docs.pytest.org/en/stable/how-to/parametrize.html)'
- '[coverage.py (devguide pointer)](https://devguide.python.org/testing/coverage/)'
- '[MLflow Tracking](https://www.mlflow.org/docs/latest/ml/tracking)'
bibliography:
- '*NumPy Docstring Guide* — numpydoc — (2024) — https://numpydoc.readthedocs.io/en/latest/format.html'
- '*MLflow Tracking Quickstart* — MLflow — (2025) — https://mlflow.org/docs/latest/ml/tracking/quickstart/'
- '*Keep a Changelog* — keepachangelog.com — (2019) — https://keepachangelog.com/en/1.1.0/'
---

## Summary

You turn prototypes into reliable artefacts. Tests guard against silent regressions; MLflow preserves context so results are trustable. This is what separates hobby projects from professional work.
## Code Focus

- pytest: parametrised tests for data transforms and metrics; fixtures for small toy frames; coverage reports.
- Docstring standards (NumPy style) and minimal API docs; README checklists; CHANGELOG and semantic versioning.
- MLflow for experiment tracking: parameters, metrics, artifacts; compare runs; record preprocessing and model versions.
## Math & Stats

- Consolidation of Weeks 1–5: variance of CV estimates; simple power calculations to decide fold counts; when to use repeated CV.
- No new theory—focus on rigour and reproducibility.
## Docs

- [pytest parametrisation](https://docs.pytest.org/en/stable/how-to/parametrize.html)
- [coverage.py (devguide pointer)](https://devguide.python.org/testing/coverage/)
- [MLflow Tracking](https://www.mlflow.org/docs/latest/ml/tracking)
## Bibliography

- *NumPy Docstring Guide* — numpydoc — (2024) — https://numpydoc.readthedocs.io/en/latest/format.html
- *MLflow Tracking Quickstart* — MLflow — (2025) — https://mlflow.org/docs/latest/ml/tracking/quickstart/
- *Keep a Changelog* — keepachangelog.com — (2019) — https://keepachangelog.com/en/1.1.0/
