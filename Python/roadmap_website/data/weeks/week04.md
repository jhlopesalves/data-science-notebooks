---
number: 4
title: Inference, Resampling, and A/B Testing
phase: Foundations
bundles:
- bundle_stats_inference
project:
  title: A/B Test Analyst
  dataset: 'Kaggle: AB Test for an E-Commerce Website'
  dataset_links:
  - https://www.kaggle.com/datasets/zhangluyuan/ab-test-for-an-e-commerce-website
  metrics:
  - Report absolute and relative lift with 95% CIs
  - Permutation-test p-value with clear assumptions
  - 'Readable figures: conversion bars with error bars; cumulative conversion over
    time'
  nuances:
  - Check randomisation integrity (pre-experiment covariates balance)
  - Segment by device/region but control FWER or FDR when scanning many segments
code_focus:
- Confidence intervals via bootstrap (percentile, BCa) for means, medians, and model
  metrics; permutation tests for group differences.
- Power and sample-size sketches; practical p-value interpretation; multiple-testing
  control when slicing cohorts.
- 'Implement a minimal A/B analysis toolkit: lift, pooled vs unpooled variance, delta
  method approximation; visualise with uncertainty bands.'
math_stats:
- Central Limit Theorem (operational view); sampling distributions; properties of
  estimators (bias, variance, consistency).
- Randomisation inference logic for permutation tests.
docs:
- '[SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)'
- '[statsmodels user guide (Inference)](https://www.statsmodels.org/stable/user-guide.html)'
bibliography:
- '*All of Statistics* — Wasserman — (2004) — https://www.stat.cmu.edu/~larry/all-of-statistics/'
- '*Stat 110: Confidence intervals & hypothesis testing lectures* — Blitzstein — (2016)
  — https://projects.iq.harvard.edu/stat110/home'
---

## Summary

This week injects statistical muscle: you can now quantify uncertainty and avoid false confidence. You build resampling tools that you will reuse in model comparison, hyperparameter selection, and monitoring.
## Project Description

Analyse an A/B dataset: clean exposures/clicks/conversions, compute uplift with CIs, perform a permutation test for the difference in conversion rates, and provide a practical recommendation memo with a risk assessment.
## Code Focus

- Confidence intervals via bootstrap (percentile, BCa) for means, medians, and model metrics; permutation tests for group differences.
- Power and sample-size sketches; practical p-value interpretation; multiple-testing control when slicing cohorts.
- Implement a minimal A/B analysis toolkit: lift, pooled vs unpooled variance, delta method approximation; visualise with uncertainty bands.
## Math & Stats

- Central Limit Theorem (operational view); sampling distributions; properties of estimators (bias, variance, consistency).
- Randomisation inference logic for permutation tests.
## Docs

- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels user guide (Inference)](https://www.statsmodels.org/stable/user-guide.html)
## Bibliography

- *All of Statistics* — Wasserman — (2004) — https://www.stat.cmu.edu/~larry/all-of-statistics/
- *Stat 110: Confidence intervals & hypothesis testing lectures* — Blitzstein — (2016) — https://projects.iq.harvard.edu/stat110/home
