---
number: 20
title: Treatment Effects with DoWhy/EconML
phase: Causality (Foundations)
bundles:
- bundle_causal
project:
  title: 'Job-Training Effect: Re-examining LaLonde/NSW'
  dataset: NSW/LaLonde experimental + observational replicas (IHDP optional).
  dataset_links:
  - https://users.nber.org/~rdehejia/data/.nswdata2.html
  - https://www.pywhy.org/dowhy/v0.9/example_notebooks/dowhy_ihdp_data_example.html
  metrics:
  - Estimated ATE with CIs; uplift curves; refuter stability; overlap diagnostics.
  nuances:
  - Separate experimental from observational samples; demonstrate how estimates drift.
  - Discuss untestable assumptions explicitly in your report.
code_focus:
- 'Formulate a causal graph; identify assumptions (back-door, front-door); run a DoWhy
  pipeline: identify → estimate → refute with placebo and bootstrap refuters.'
- Estimate ATE/CATE with EconML (e.g., Doubly Robust, T-learner, Causal Forests);
  include uplift plots and policy curves.
- 'Sensitivity analysis: how robust is your effect to unobserved confounding?'
math_stats:
- Potential outcomes, ignorability and overlap; propensity scores and balancing.
- Doubly robust estimation; orthogonalisation; heterogeneous treatment effects.
- Refutation logic and why observational claims must be humble.
docs:
- https://microsoft.github.io/dowhy/
- https://www.pywhy.org/dowhy/v0.9/example_notebooks/dowhy_ihdp_data_example.html
- https://econml.azurewebsites.net/
bibliography:
- Angrist & Pischke — Mostly Harmless Econometrics.
- Imbens & Rubin — Causal Inference for Statistics, Social, and Biomedical Sciences.
- 'Hernán & Robins — Causal Inference: What If.'
---

## Summary

You learn to ask and answer ‘what if’ responsibly. Instead of treating correlation as causation, you will articulate assumptions, quantify sensitivity, and present results with the humility that observational data demands.
## Project Description

Estimate the causal effect of a training programme on earnings. Start with naive OLS; then apply propensity score methods and a doubly robust estimator. Run DoWhy refuters (placebo, bootstrap) and present a sensitivity analysis.
## Code Focus

- Formulate a causal graph; identify assumptions (back-door, front-door); run a DoWhy pipeline: identify → estimate → refute with placebo and bootstrap refuters.
- Estimate ATE/CATE with EconML (e.g., Doubly Robust, T-learner, Causal Forests); include uplift plots and policy curves.
- Sensitivity analysis: how robust is your effect to unobserved confounding?
## Math & Stats

- Potential outcomes, ignorability and overlap; propensity scores and balancing.
- Doubly robust estimation; orthogonalisation; heterogeneous treatment effects.
- Refutation logic and why observational claims must be humble.
## Docs

- https://microsoft.github.io/dowhy/
- https://www.pywhy.org/dowhy/v0.9/example_notebooks/dowhy_ihdp_data_example.html
- https://econml.azurewebsites.net/
## Bibliography

- Angrist & Pischke — Mostly Harmless Econometrics.
- Imbens & Rubin — Causal Inference for Statistics, Social, and Biomedical Sciences.
- Hernán & Robins — Causal Inference: What If.
