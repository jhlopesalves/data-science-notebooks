---
number: 14
title: Dimensionality Reduction and Clustering as Tools
phase: Unsupervised & Feature Learning
bundles:
- bundle_unsupervised_repr
project:
  title: Text Topics and Customer Segments
  dataset: 20 Newsgroups (topics) + Mall Customers (toy segmentation).
  dataset_links:
  - https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
  - https://www.kaggle.com/datasets/shwetabh123/mall-customers
  metrics:
  - NMI/ARI for labelled text; silhouette/CH/DB for Mall Customers; downstream classifier
    F1 after dimensionality reduction.
  nuances:
  - Hyperparameter sensitivity of t-SNE; stability across random seeds.
  - When to prefer NMF over LSA for interpretability.
code_focus:
- PCA/IncrementalPCA and TruncatedSVD (for sparse TF-IDF); compare explained variance,
  reconstruction error, and downstream classifier performance.
- NMF (Euclidean vs Kullback–Leibler) for parts-based topic-like factors on text.
- 'Clustering: KMeans/MiniBatchKMeans; DBSCAN (eps, min_samples); hierarchical Agglomerative
  (linkage); silhouette, Calinski–Harabasz, Davies–Bouldin indices.'
- 'Visualisation: t-SNE on low-dimensional embeddings; articulate perplexity/learning
  rate effects; warn against using t-SNE distances quantitatively.'
math_stats:
- SVD, eigen-decomposition; low-rank approximation and Eckart–Young–Mirsky theorem
  (intuition).
- Matrix factorisation objectives (PCA vs NMF).
- Density vs centroid notions of cluster; why internal indices are imperfect.
docs:
- https://scikit-learn.org/stable/modules/decomposition.html
- https://scikit-learn.org/stable/modules/clustering.html
- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
bibliography:
- Murphy — Probabilistic Machine Learning (Ch. on Latent Variable Models).
- Aggarwal & Reddy — Data Clustering (Springer).
- Bishop — Pattern Recognition and Machine Learning (mixtures & dimensionality).
---

## Summary

You transform unsupervised learning from a novelty into a pipeline tool. The point is not to ‘discover truth’ but to engineer useful representations and segments that downstream models and stakeholders can use.
## Project Description

Build TF-IDF → TruncatedSVD (LSA) → KMeans for topic discovery and evaluation via NMI with labels (as a sanity check). For Mall Customers, compare KMeans with DBSCAN and discuss cluster validity indices and business interpretability.
## Code Focus

- PCA/IncrementalPCA and TruncatedSVD (for sparse TF-IDF); compare explained variance, reconstruction error, and downstream classifier performance.
- NMF (Euclidean vs Kullback–Leibler) for parts-based topic-like factors on text.
- Clustering: KMeans/MiniBatchKMeans; DBSCAN (eps, min_samples); hierarchical Agglomerative (linkage); silhouette, Calinski–Harabasz, Davies–Bouldin indices.
- Visualisation: t-SNE on low-dimensional embeddings; articulate perplexity/learning rate effects; warn against using t-SNE distances quantitatively.
## Math & Stats

- SVD, eigen-decomposition; low-rank approximation and Eckart–Young–Mirsky theorem (intuition).
- Matrix factorisation objectives (PCA vs NMF).
- Density vs centroid notions of cluster; why internal indices are imperfect.
## Docs

- https://scikit-learn.org/stable/modules/decomposition.html
- https://scikit-learn.org/stable/modules/clustering.html
- https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
## Bibliography

- Murphy — Probabilistic Machine Learning (Ch. on Latent Variable Models).
- Aggarwal & Reddy — Data Clustering (Springer).
- Bishop — Pattern Recognition and Machine Learning (mixtures & dimensionality).
