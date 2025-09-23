---
number: 24
title: From Notebooks to Flows
phase: MLOps Foundations
bundles:
- bundle_orchestration_tracking
- bundle_monitoring
project:
  title: '“Taxi Forecast Pipeline: Scheduled & Watched”'
  dataset: NYC TLC monthly aggregates built in Week 22.
  dataset_links:
  - TLC sources as before. (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  metrics:
  - End-to-end runtime, task success rate, drift alerts precision (alerts that correspond
    to meaningful performance drops).
  nuances:
  - Scheduling cadence; backfilling; deterministic environments; secrets handling.
code_focus:
- Build production-like pipelines with **Prefect 2** (tasks, flows, retries, caching,
  parameters; local agent).
- Compare to **Airflow** with a minimal DAG to understand scheduler vs orchestrator
  differences.
- Data validation with **Great Expectations** or **Pandera**; model/data drift detection
  with **Evidently**.
- Track experiments with **MLflow**; promote best runs to a simple “model registry”
  folder with metadata.
math_stats:
- 'Drift types: covariate shift vs label shift; PSI/JS divergence; confidence bands
  for monitoring metrics.'
- Expected calibration error and threshold selection under drift.
docs:
- '[Prefect Quickstart](https://docs.prefect.io/latest/get-started/overview/)'
- '[Apache Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html)'
- '[Evidently Documentation](https://docs.evidentlyai.com/)'
- '[Great Expectations Quickstart](https://greatexpectations.io/docs/tutorials/quickstart/)'
- '[MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)'
bibliography:
- Lakshmanan et al. — *Machine Learning Design Patterns* (O’Reilly), patterns on pipeline
  orchestration and monitoring.
- Ville Satopaa et al. “Information Theoretic Measures of Change Detection” (for drift
  intuition).
- Breck et al. (2017) “The ML Test Score.”
---

## Summary

You are turning craft into systems thinking. The product is a reliable pipeline that can run unattended, prove it behaved, and warn you when the world changes underneath your model.
## Project Description

Build a Prefect flow that (1) ingests the latest month, (2) recomputes features, (3) retrains a simple gradient boosting forecaster, (4) validates with Great Expectations, (5) logs to MLflow, (6) posts a drift report with Evidently.
## Code Focus

- Build production-like pipelines with **Prefect 2** (tasks, flows, retries, caching, parameters; local agent).
- Compare to **Airflow** with a minimal DAG to understand scheduler vs orchestrator differences.
- Data validation with **Great Expectations** or **Pandera**; model/data drift detection with **Evidently**.
- Track experiments with **MLflow**; promote best runs to a simple “model registry” folder with metadata.
## Math & Stats

- Drift types: covariate shift vs label shift; PSI/JS divergence; confidence bands for monitoring metrics.
- Expected calibration error and threshold selection under drift.
## Docs

- [Prefect Quickstart](https://docs.prefect.io/latest/get-started/overview/)
- [Apache Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial/index.html)
- [Evidently Documentation](https://docs.evidentlyai.com/)
- [Great Expectations Quickstart](https://greatexpectations.io/docs/tutorials/quickstart/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
## Bibliography

- Lakshmanan et al. — *Machine Learning Design Patterns* (O’Reilly), patterns on pipeline orchestration and monitoring.
- Ville Satopaa et al. “Information Theoretic Measures of Change Detection” (for drift intuition).
- Breck et al. (2017) “The ML Test Score.”
