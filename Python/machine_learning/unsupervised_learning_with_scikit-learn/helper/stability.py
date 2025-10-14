from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def silhouette_stability(
    X: np.ndarray, k: int, seeds: Iterable[int] = (0, 7, 42, 123, 999)
) -> pd.Series:
    """
    Compute silhouette scores across multiple KMeans random seeds to assess clustering stability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
            Feature matrix in the same space used for clustering (preprocessing such as scaling
            should be applied before calling this function).
    k : int
            Number of clusters to fit (k >= 2).
    seeds : iterable of int, optional
            Sequence of random seeds used for different KMeans initialisations.
            Default is (0, 7, 42, 123, 999).

    Returns
    -------
    pandas.Series
            Silhouette scores for each seed, indexed by the seed values. The Series name is
            set to ``f"silhouette_k={k}"``.

    Raises
    ------
    ValueError
            If k < 2.

    Notes
    -----
    Each seed runs an independent KMeans fit with ``n_init="auto"`` and the silhouette score
    is computed on the fitted labels. This is a quick stability diagnostic and does not replace
    more thorough resampling or consensus clustering approaches.
    """
    if int(k) < 2:
        raise ValueError("k must be >= 2.")

    X = np.asarray(X)
    vals = []
    seeds_list = list(seeds)
    for s in seeds_list:
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=int(s))
        labels = km.fit_predict(X)
        vals.append(silhouette_score(X, labels))
    return pd.Series(vals, index=seeds_list, name=f"silhouette_k={k}")
