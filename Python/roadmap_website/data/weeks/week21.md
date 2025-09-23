---
number: 21
title: Fairness, Explainability, and Model Risk
phase: Statistical Modelling → Governance
bundles:
- bundle_fairness_calibration
- bundle_responsible_ai
project:
  title: “Fairness & Explainability Dossier”
  dataset: UCI Adult Income; German Credit; COMPAS re-implementation (for methodological
    critique).
  dataset_links:
  - UCI Adult from multiple mirrors; German Credit (UCI); for a large sliceable alternative
    use NYC TLC trips for subgroup analysis by borough (proxy fairness only as a didactic
    exercise). TLC official repository and AWS/Open Data mirrors. (https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  metrics:
  - AUC/PR-AUC; Brier score; calibration error; disparity measures (DP gap, EO TPR/FPR
    gaps).
  nuances:
  - Discuss impossibility results and justify which fairness target you optimise under
    your domain assumptions; include uncertainty bands for subgroup metrics.
code_focus:
- 'Assess group fairness on tabular classifiers with `fairlearn` (metrics: demographic
  parity, equalised odds; dashboards and report artefacts).'
- 'Local and global explanations: SHAP (TreeExplainer, KernelExplainer) and LIME for
  tabular and text models; compare stability of attributions under resampling.'
- Build a “model card” template (Markdown) auto-filled from training metadata and
  evaluation slices.
- 'Add privacy-aware preprocessing: simple k-anonymity exploration, suppression, and
  PII detection heuristics in Pandas; document limitations.'
math_stats:
- 'Fairness definitions: demographic parity, equalised odds, predictive parity; incompatibility
  trade-offs.'
- 'Game-theoretic attribution (Shapley values): additivity, symmetry, dummy, efficiency;
  local surrogate models for LIME; variance and faithfulness issues.'
- 'Risk framing: harm vectors, uncertainty communication, and error decomposition
  across subpopulations.'
docs:
- Fairlearn quickstart and mitigation user guide. (https://fairlearn.org/main/quickstart.html)
- SHAP documentation. (https://shap.readthedocs.io/)
- LIME documentation (Python). (https://lime-ml.readthedocs.io/en/latest/lime.html)
- NIST AI Risk Management Framework 1.0 overview. (https://www.nist.gov/itl/ai-risk-management-framework)
- EU AI Act overview and timeline. (https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
bibliography:
- Barocas, Hardt, Narayanan — *Fairness and Machine Learning* (open textbook).
- Mitchell et al. (2019) “Model Cards for Model Reporting” (FAT\*’19).
- Molnar — *Interpretable Machine Learning* (2e).
- Ribeiro, Singh, Guestrin (2016) “Why Should I Trust You?” (LIME, KDD).
- Kleinberg, Mullainathan, Raghavan (2016) “Inherent Trade-Offs in the Fair Determination
  of Risk Scores.”
---

## Summary

This week formalises your responsibility as a modeller. You will learn to measure disparity, articulate unavoidable trade-offs, and produce transparent, reproducible explanations. The output is not just numbers but a governance artefact (a model card) that surfaces performance on relevant subgroups and makes explicit the intended use, limits, and ethical posture of your model. Expect to feel tension between accuracy and fairness goals—that tension is real and must be managed, not hand-waved.
## Project Description

Train two classifiers (regularised logistic regression and gradient boosting). Produce: (1) calibrated ROC/PR; (2) fairness metrics across sensitive attributes; (3) SHAP global feature importance and per-instance force plots; (4) LIME explanations for failure cases; (5) a Model Card in Markdown referencing evaluation on slices.
## Code Focus

- Assess group fairness on tabular classifiers with `fairlearn` (metrics: demographic parity, equalised odds; dashboards and report artefacts).
- Local and global explanations: SHAP (TreeExplainer, KernelExplainer) and LIME for tabular and text models; compare stability of attributions under resampling.
- Build a “model card” template (Markdown) auto-filled from training metadata and evaluation slices.
- Add privacy-aware preprocessing: simple k-anonymity exploration, suppression, and PII detection heuristics in Pandas; document limitations.
## Math & Stats

- Fairness definitions: demographic parity, equalised odds, predictive parity; incompatibility trade-offs.
- Game-theoretic attribution (Shapley values): additivity, symmetry, dummy, efficiency; local surrogate models for LIME; variance and faithfulness issues.
- Risk framing: harm vectors, uncertainty communication, and error decomposition across subpopulations.
## Docs

- Fairlearn quickstart and mitigation user guide. (https://fairlearn.org/main/quickstart.html)
- SHAP documentation. (https://shap.readthedocs.io/)
- LIME documentation (Python). (https://lime-ml.readthedocs.io/en/latest/lime.html)
- NIST AI Risk Management Framework 1.0 overview. (https://www.nist.gov/itl/ai-risk-management-framework)
- EU AI Act overview and timeline. (https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai)
## Bibliography

- Barocas, Hardt, Narayanan — *Fairness and Machine Learning* (open textbook).
- Mitchell et al. (2019) “Model Cards for Model Reporting” (FAT\*’19).
- Molnar — *Interpretable Machine Learning* (2e).
- Ribeiro, Singh, Guestrin (2016) “Why Should I Trust You?” (LIME, KDD).
- Kleinberg, Mullainathan, Raghavan (2016) “Inherent Trade-Offs in the Fair Determination of Risk Scores.”
