---
number: 2
title: Data Cleaning, Missingness, and Robust Descriptives
phase: Foundations
bundles:
- bundle_stats_core
- bundle_sklearn_core
project:
  title: Ames Data-Quality Report
  dataset: 'Kaggle House Prices: Advanced Regression Techniques (Ames)'
  dataset_links:
  - https://www.kaggle.com/c/house-prices-advanced-regression-techniques
  metrics:
  - 'Completeness: <5% unknown types'
  - At least 10 diagnostic plots with captions
  - Comparison table of imputation strategies on a Ridge baseline (RMSE)
  nuances:
  - Be explicit about leakage risks when imputing using target-derived information
  - Document categorical levels consolidation decisions
code_focus:
- 'Imputation: SimpleImputer, KNNImputer; categorical encoding strategies; ordinal
  vs one-hot; date feature extraction.'
- 'Outliers and influence: robust scalers (quantile, robust), Windsorising; compare
  mean/median-trimmed.'
- Non-parametric summaries with Pingouin (e.g., robust correlation) and SciPy hypothesis
  tests.
- 'Plot repertoire: distribution diagnostics (QQ plots), pairplots with kernel density,
  violin/box with swarm overlay; bivariate regression plots with CIs.'
math_stats:
- Quantiles, order statistics, IQR; robust vs efficient estimators.
- 'Hypothesis testing grammar: null, alternative, test statistic, sampling distribution;
  multiple comparisons control (Bonferroni) when scanning many relations.'
- Conceptualise missingness MCAR/MAR/MNAR and the bias they induce; practical sensitivity
  analysis via pattern plots.
docs:
- '[SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)'
- '[statsmodels user guide](https://www.statsmodels.org/stable/user-guide.html)'
- '[Pingouin docs](https://pingouin-stats.org/)'
- '[seaborn tutorial](https://seaborn.pydata.org/tutorial.html)'
bibliography:
- '*All of Statistics* — Larry Wasserman — (2004) — https://www.stat.cmu.edu/~larry/all-of-statistics/'
- '*ISLR (Python)* — James et al. — (2023) — https://www.statlearning.com/'
- '*Harvard Stat 110 (selected: hypothesis testing, CLT)* — Joe Blitzstein — (2016)
  — https://projects.iq.harvard.edu/stat110/home'
---

## Summary

You learn to treat cleaning, imputation, and robust descriptive statistics as first-class modelling steps. The report you write must justify your choices with plots and simple tests. That discipline pays off when regularised models arrive next week.
## Project Description

Produce a data-quality and exploratory report for Ames: schema table, missingness patterns per feature group, robust univariate summaries, pairwise correlations (rank-based), and a draft feature dictionary. Implement at least two imputation strategies and compare their effect on downstream baseline error via a tiny Ridge baseline.
## Code Focus

- Imputation: SimpleImputer, KNNImputer; categorical encoding strategies; ordinal vs one-hot; date feature extraction.
- Outliers and influence: robust scalers (quantile, robust), Windsorising; compare mean/median-trimmed.
- Non-parametric summaries with Pingouin (e.g., robust correlation) and SciPy hypothesis tests.
- Plot repertoire: distribution diagnostics (QQ plots), pairplots with kernel density, violin/box with swarm overlay; bivariate regression plots with CIs.
## Math & Stats

- Quantiles, order statistics, IQR; robust vs efficient estimators.
- Hypothesis testing grammar: null, alternative, test statistic, sampling distribution; multiple comparisons control (Bonferroni) when scanning many relations.
- Conceptualise missingness MCAR/MAR/MNAR and the bias they induce; practical sensitivity analysis via pattern plots.
## Docs

- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [statsmodels user guide](https://www.statsmodels.org/stable/user-guide.html)
- [Pingouin docs](https://pingouin-stats.org/)
- [seaborn tutorial](https://seaborn.pydata.org/tutorial.html)
## Bibliography

- *All of Statistics* — Larry Wasserman — (2004) — https://www.stat.cmu.edu/~larry/all-of-statistics/
- *ISLR (Python)* — James et al. — (2023) — https://www.statlearning.com/
- *Harvard Stat 110 (selected: hypothesis testing, CLT)* — Joe Blitzstein — (2016) — https://projects.iq.harvard.edu/stat110/home
