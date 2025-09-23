---
number: 12
title: Trees, Bagging and Forests
phase: Core ML
bundles:
- bundle_trees_forests
project:
  title: Tabular Risk Modelling with Forests
  dataset: Adult Census Income; Titanic (for pedagogy).
  dataset_links:
  - https://archive.ics.uci.edu/dataset/2/adult
  - https://www.kaggle.com/c/titanic
  metrics:
  - ROC-AUC (macro & weighted), expected calibration error (ECE), OOB accuracy where
    applicable.
  nuances:
  - Leakage-safe preprocessing inside Pipeline/ColumnTransformer.
  - 'Handling high-cardinality categoricals: hashing vs target encoding (fold-safe).'
code_focus:
- 'DecisionTreeClassifier/Regressor: cost-complexity pruning (ccp_alpha), max_depth/min_samples_leaf
  tuning, class_weight handling, monotone constraints (conceptual).'
- BaggingClassifier/Regressor; RandomForest* and ExtraTrees*; OOB estimates; permutation_importance
  for robust importance vs impurity bias.
- 'Model inspection: PartialDependenceDisplay (PDP/ICE) on tabular features; global
  vs local importance; pitfalls with correlated features.'
math_stats:
- Greedy recursive partitioning; impurity (Gini/entropy/MSE) and variance reduction.
- Bagging variance reduction and bias trade-off; why ExtraTrees increases randomness.
- Why impurity-based importance is biased; permutation importance as a model-agnostic
  alternative.
docs:
- https://scikit-learn.org/stable/modules/tree.html
- https://scikit-learn.org/stable/modules/ensemble.html
- https://scikit-learn.org/stable/modules/permutation_importance.html
- https://scikit-learn.org/stable/modules/partial_dependence.html
bibliography:
- Breiman — Random Forests (2001).
- ESL — Ch. 9 (Additive Models, Trees, and Related Methods).
- Molnar — Interpretable Machine Learning (tree interpretation and PDP/ICE).
---

## Summary

You learn to make trees honest. Rather than worship variable importance, you will quantify it with permutation tests and confront interpretation pitfalls in correlated feature sets. Forests become your go-to strong baseline for mixed-type tabular data.
## Project Description

Train pruned trees, bagging and random forests. Use OOB error where appropriate, permutation importance, and PDP/ICE to explain two most influential features. Compare against a regularised logistic baseline.
## Code Focus

- DecisionTreeClassifier/Regressor: cost-complexity pruning (ccp_alpha), max_depth/min_samples_leaf tuning, class_weight handling, monotone constraints (conceptual).
- BaggingClassifier/Regressor; RandomForest* and ExtraTrees*; OOB estimates; permutation_importance for robust importance vs impurity bias.
- Model inspection: PartialDependenceDisplay (PDP/ICE) on tabular features; global vs local importance; pitfalls with correlated features.
## Math & Stats

- Greedy recursive partitioning; impurity (Gini/entropy/MSE) and variance reduction.
- Bagging variance reduction and bias trade-off; why ExtraTrees increases randomness.
- Why impurity-based importance is biased; permutation importance as a model-agnostic alternative.
## Docs

- https://scikit-learn.org/stable/modules/tree.html
- https://scikit-learn.org/stable/modules/ensemble.html
- https://scikit-learn.org/stable/modules/permutation_importance.html
- https://scikit-learn.org/stable/modules/partial_dependence.html
## Bibliography

- Breiman — Random Forests (2001).
- ESL — Ch. 9 (Additive Models, Trees, and Related Methods).
- Molnar — Interpretable Machine Learning (tree interpretation and PDP/ICE).
