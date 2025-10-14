from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def scan_k_metrics(
    X: np.ndarray,
    ks: Iterable[int] = range(2, 11),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute internal clustering metrics for KMeans over a range of k.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_features)
            Feature matrix in the SAME space you intend to cluster in (preprocessed:
            imputed, scaled, etc.).
    ks : iterable of int, optional
            Candidate numbers of clusters (k >= 2). Default is range(2, 11).
    random_state : int, optional
            Seed for KMeans to improve reproducibility (default is 42).

    Returns
    -------
    pandas.DataFrame
            DataFrame with one row per candidate k and the following columns:
            - k : int
            - inertia : float (within-cluster sum of squares; lower is better)
            - silhouette : float (mean silhouette score; higher is better)
            - ch : float (Calinski–Harabasz index; higher is better)
            - db : float (Davies–Bouldin index; lower is better)

    Raises
    ------
    ValueError
            If `ks` is empty or any supplied k < 2.

    Notes
    -----
    The function fits KMeans for each candidate k and records standard internal
    validation metrics. It expects the caller to handle any preprocessing
    (scaling, imputation) prior to calling.

    Examples
    --------
    >>> df = scan_k_metrics(X_std, ks=range(2, 7), random_state=0)
    >>> display(df)
    """
    X = np.asarray(X)
    ks_list = list(ks)

    if len(ks_list) == 0:
        raise ValueError("`ks` must be a non-empty iterable of integers (k >= 2).")
    if any(int(k) < 2 for k in ks_list):
        raise ValueError("All values in `ks` must be integers >= 2.")

    records = []
    for k in ks_list:
        km = KMeans(n_clusters=int(k), n_init="auto", random_state=random_state)
        labels = km.fit_predict(X)
        records.append(
            {
                "k": int(k),
                "inertia": float(km.inertia_),
                "silhouette": float(silhouette_score(X, labels)),
                "ch": float(calinski_harabasz_score(X, labels)),
                "db": float(davies_bouldin_score(X, labels)),
            }
        )
    return pd.DataFrame(records)


def knee_from_inertia(inertia: np.ndarray, ks: np.ndarray) -> int:
    """
    Compute the "knee" (elbow) from inertia vs. k using the distance-to-chord method.

    Parameters
    ----------
    inertia : ndarray
            1-D array of within-cluster sum-of-squares (inertia) for each candidate k.
    ks : ndarray
            1-D array of cluster counts corresponding to `inertia`. Must have the same
            shape as `inertia` and typically be increasing (e.g. [2, 3, 4, ...]).

    Returns
    -------
    int
            The value from `ks` corresponding to the maximum perpendicular distance
            to the chord joining the first and last points (the estimated knee).

    Raises
    ------
    ValueError
            If `inertia` and `ks` have mismatched shapes or are empty.

    Notes
    -----
    The method normalises both axes to [0, 1], computes the perpendicular distance
    of each (k, inertia) point to the straight line connecting the two endpoints,
    and returns the k with maximum distance. A small epsilon is added to the
    denominator for numerical stability.

    Examples
    --------
    >>> inertia = np.array([100.0, 60.0, 40.0, 30.0, 25.0])
    >>> ks = np.arange(2, 7)
    >>> knee_from_inertia(inertia, ks)
    3
    """
    ks = np.asarray(ks, dtype=float)
    y = np.asarray(inertia, dtype=float)

    if ks.size == 0 or y.size == 0:
        raise ValueError("`inertia` and `ks` must be non-empty.")
    if ks.shape != y.shape:
        raise ValueError("`inertia` and `ks` must have the same shape.")

    # Normalise to [0, 1]
    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (y - y.min()) / (y.max() - y.min())

    x0, y0 = x[0], y[0]
    x1, y1 = x[-1], y[-1]

    # Perpendicular distance from each point to the line through endpoints
    num = np.abs((y1 - y0) * x - (x1 - x0) * y + x1 * y0 - y1 * x0)
    den = np.hypot(y1 - y0, x1 - x0)
    dist = num / (den + 1e-12)

    idx = int(np.argmax(dist))
    return int(ks[idx])


def recommend_k(df: pd.DataFrame, sil_margin: float = 0.03) -> Dict[str, int]:
    """
    Combine knee-from-inertia with silhouette to produce a transparent k recommendation.

    Parameters
    ----------
    df : pandas.DataFrame
            DataFrame containing columns 'k', 'inertia', and 'silhouette' for candidate k values.
    sil_margin : float, optional
            Tolerance below the maximum silhouette to form the admissible set A.
            Candidates with silhouette >= max_silhouette - sil_margin are included.
            Default is 0.03.

    Returns
    -------
    dict
            Dictionary with keys:
            - 'knee' (int): k suggested by the knee (elbow) from inertia.
            - 'silhouette_max' (int): k with maximum silhouette score.
            - 'final' (int): final recommended k chosen from the admissible set A as the member
              closest to the knee (ties broken by choosing the smaller k).

    Raises
    ------
    ValueError
            If the input DataFrame does not contain the required columns 'k', 'inertia', and 'silhouette'.

    Notes
    -----
    Steps:
      1. Compute global elbow k_knee using knee_from_inertia on inertia vs k.
      2. Find k_silmax with maximum silhouette.
      3. Form admissible set A = {k: silhouette >= sil_max - sil_margin}.
      4. Choose k_final from A minimizing (|k - k_knee|, k) to prefer proximity to the knee and
             smaller k in case of ties.
    """
    if not {"k", "inertia", "silhouette"} <= set(df.columns):
        raise ValueError("DataFrame must contain 'k', 'inertia', and 'silhouette'.")

    ks = df["k"].to_numpy()
    k_knee = knee_from_inertia(df["inertia"].to_numpy(), ks)
    k_silmax = int(df.loc[df["silhouette"].idxmax(), "k"])
    sil_star = df["silhouette"].max()
    admissible = df.loc[df["silhouette"] >= sil_star - sil_margin, "k"].to_numpy()
    if admissible.size == 0:
        admissible = np.array([k_silmax])
    k_final = int(sorted(admissible, key=lambda k: (abs(k - k_knee), k))[0])
    return {"knee": k_knee, "silhouette_max": k_silmax, "final": k_final}