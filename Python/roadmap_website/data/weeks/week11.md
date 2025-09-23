---
number: 11
title: 'SVMs and k-NN: Margins vs Neighbourhoods'
phase: Core ML
bundles:
- bundle_svm_knn
project:
  title: Text Topic SVM vs k-NN
  dataset: 20 Newsgroups
  dataset_links:
  - https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
  metrics:
  - Macro-F1, macro ROC-AUC, Brier score; reliability diagrams.
  nuances:
  - Sparse vs dense representations; memory and fit time profiling.
  - Effect of stop-words and n-grams on margin and nearest-neighbour geometry.
code_focus:
- SVC with RBF and linear kernels; tuning C and gamma via log-spaced grids; class_weight
  for imbalance; probability calibration (Platt sigmoid) for decision-thresholding.
- LinearSVC vs SVC trade-offs on high-dimensional sparse text; scaling with StandardScaler
  and pipeline safety.
- k-NN for classification and regression; metric choice (Minkowski, cosine for text
  embeddings), k selection via CV, and curse-of-dimensionality mitigation by PCA/TruncatedSVD.
- Reliability curves and Brier score to compare calibrated SVM vs k-NN; per-class
  ROC and PR curves for imbalanced data.
math_stats:
- Maximum-margin classifiers; primal/dual, role of C and gamma in margin width and
  effective radius.
- Cover–Hart intuition for k-NN; bias–variance as k varies; metric geometry effects.
- 'Calibration theory: proper scoring rules (log loss, Brier).'
docs:
- https://scikit-learn.org/stable/modules/svm.html
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/calibration.html
bibliography:
- Vapnik — Statistical Learning Theory (selected sections on margins).
- Müller & Guido — Introduction to Machine Learning with Python (SVM & k-NN chapters).
- Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities with Supervised
  Learning (calibration).
---

## Summary

This week juxtaposes global margin decision rules with intensely local rules. You will get a working intuition for C and gamma, learn when a linear SVM on sparse TF-IDF already saturates performance, and see how calibration changes downstream decision quality. The k-NN experiments force you to confront metrics, dimensionality and compute constraints.
## Project Description

Vectorise text via TfidfVectorizer, compare LinearSVC (one-vs-rest) to k-NN (cosine). Add probability calibration and report accuracy, macro-F1, macro-AUC and calibration curves. Explore how truncation with TruncatedSVD (LSA) affects both models.
## Code Focus

- SVC with RBF and linear kernels; tuning C and gamma via log-spaced grids; class_weight for imbalance; probability calibration (Platt sigmoid) for decision-thresholding.
- LinearSVC vs SVC trade-offs on high-dimensional sparse text; scaling with StandardScaler and pipeline safety.
- k-NN for classification and regression; metric choice (Minkowski, cosine for text embeddings), k selection via CV, and curse-of-dimensionality mitigation by PCA/TruncatedSVD.
- Reliability curves and Brier score to compare calibrated SVM vs k-NN; per-class ROC and PR curves for imbalanced data.
## Math & Stats

- Maximum-margin classifiers; primal/dual, role of C and gamma in margin width and effective radius.
- Cover–Hart intuition for k-NN; bias–variance as k varies; metric geometry effects.
- Calibration theory: proper scoring rules (log loss, Brier).
## Docs

- https://scikit-learn.org/stable/modules/svm.html
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/calibration.html
## Bibliography

- Vapnik — Statistical Learning Theory (selected sections on margins).
- Müller & Guido — Introduction to Machine Learning with Python (SVM & k-NN chapters).
- Niculescu-Mizil & Caruana (2005) — Predicting Good Probabilities with Supervised Learning (calibration).
