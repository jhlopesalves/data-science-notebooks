---
number: 8
title: Data Validation and Monitoring on Notebooks
phase: Foundations
bundles:
- bundle_fairness_calibration
project:
  title: Contracts Before Models
  dataset: Ames Housing, Titanic
  description: 'Add data validation to previous pipelines: define expectations/schemas
    for raw and post-processed frames. Fail fast if contracts break; show a small
    drift report between train and test splits.'
  dataset_links:
  - https://www.kaggle.com/c/house-prices-advanced-regression-techniques
  - https://www.kaggle.com/c/titanic
  metrics:
  - Automated validation passes in CI-like run
  - Readable data-docs (if GE)
  - Clear remediation steps for violated expectations
  nuances:
  - Be conservative in early contracts; tighten over time
  - Document why certain checks are excluded to avoid brittleness
code_focus:
- 'Great Expectations or Pandera for data contracts: define expectations/schemas;
  validate raw vs post-transform frames; build a quick ‘data docs’ site (if using
  GE).'
- 'Drift checks in notebooks: population stability index (PSI) sketches; distribution
  comparisons (KS/AD tests) for train–test splits.'
- Refactor Week 1–5 notebooks to assert data contracts before model fitting.
math_stats:
- Two-sample tests (KS, AD) and effect sizes; interpreting practical vs statistical
  significance in large n.
- Basics of monitoring metrics vs model metrics; separation of concerns.
docs:
- '[Great Expectations](https://docs.greatexpectations.io/)'
- '[Pandera](https://pandera.readthedocs.io/)'
bibliography:
- '*Great Expectations docs (concepts)* — GE — (2025) — https://docs.greatexpectations.io/'
- '*Pandera documentation* — Pandera — (2025) — https://pandera.readthedocs.io/'
---

## Summary

You formalise what ‘good data’ means for your projects. By encoding assumptions as checks, you prevent subtle distribution issues from silently corrupting training or evaluation.
## Code Focus

- Great Expectations or Pandera for data contracts: define expectations/schemas; validate raw vs post-transform frames; build a quick ‘data docs’ site (if using GE).
- Drift checks in notebooks: population stability index (PSI) sketches; distribution comparisons (KS/AD tests) for train–test splits.
- Refactor Week 1–5 notebooks to assert data contracts before model fitting.
## Math & Stats

- Two-sample tests (KS, AD) and effect sizes; interpreting practical vs statistical significance in large n.
- Basics of monitoring metrics vs model metrics; separation of concerns.
## Docs

- [Great Expectations](https://docs.greatexpectations.io/)
- [Pandera](https://pandera.readthedocs.io/)
## Bibliography

- *Great Expectations docs (concepts)* — GE — (2025) — https://docs.greatexpectations.io/
- *Pandera documentation* — Pandera — (2025) — https://pandera.readthedocs.io/
