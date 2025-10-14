from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)


def k_report(X_std, k, seed=42, true_labels=None):
    """
    Compute KMeans clustering and common cluster-validation metrics for a given
    standardized feature matrix and number of clusters.

    Parameters
    ----------
    X_std : array-like of shape (n_samples, n_features)
        Standardized feature matrix (e.g., output of StandardScaler). Must be
        suitable for clustering; no additional scaling is performed by this
        function.
    k : int
        Number of clusters to fit with KMeans. Must be at least 2 and less than
        the number of samples.
    seed : int, optional
        Random seed passed to the KMeans `random_state` parameter. Default is 42.
    true_labels : array-like of shape (n_samples,), optional
        Ground truth labels. If provided, external validation metrics (ARI, AMI, NMI)
        will be computed and included in the report. Default is None.

    Returns
    -------
    report : dict
        A dictionary containing the following keys:
        - 'k' (int): the number of clusters used.
        - 'inertia' (float): KMeans inertia (sum of squared distances of samples
          to their nearest cluster center).
        - 'silhouette' (float): mean silhouette score for all samples (higher is
          better; requires at least 2 clusters).
        - 'CH' (float): Calinski–Harabasz index (higher is better).
        - 'DB' (float): Davies–Bouldin index (lower is better).
        - 'ARI' (float): Adjusted Rand Index (only if true_labels is provided).
        - 'AMI' (float): Adjusted Mutual Information (only if true_labels is provided).
        - 'NMI' (float): Normalized Mutual Information (only if true_labels is provided).
        - 'model' (sklearn.cluster.KMeans): the fitted KMeans estimator instance.

    Raises
    ------
    ValueError
        If `k` is invalid for the provided data (e.g., k < 2 or k >= n_samples),
        or if one of the underlying scikit-learn scoring functions raises an
        error for the given input.

    Notes
    -----
    This convenience function fits a KMeans model with n_clusters=k and
    n_init="auto" and then computes several clustering validation metrics from
    scikit-learn: silhouette_score, calinski_harabasz_score, and
    davies_bouldin_score. The input X_std is expected to be already preprocessed
    (e.g., standardized). The silhouette score is only defined for k >= 2.

    Examples
    --------
    >>> # X_std is a (n_samples, n_features) array-like of scaled features
    >>> report = k_report(X_std, k=3, seed=0)
    >>> report['k']
    3
    >>> report['inertia']  # doctest: +ELLIPSIS
    >>> report['silhouette']  # doctest: +ELLIPSIS
    """
    km = KMeans(n_clusters=k, n_init="auto", random_state=seed).fit(X_std)
    labels = km.labels_

    report = {
        "k": k,
        "inertia": km.inertia_,
        "silhouette": silhouette_score(X_std, labels),
        "CH": calinski_harabasz_score(X_std, labels),  # ↑ better
        "DB": davies_bouldin_score(X_std, labels),  # ↓ better
        "model": km,
    }

    if true_labels is not None:
        report["ARI"] = adjusted_rand_score(true_labels, labels)  # ↑ better
        report["AMI"] = adjusted_mutual_info_score(true_labels, labels)  # ↑ better
        report["NMI"] = normalized_mutual_info_score(true_labels, labels)  # ↑ better

    return report
