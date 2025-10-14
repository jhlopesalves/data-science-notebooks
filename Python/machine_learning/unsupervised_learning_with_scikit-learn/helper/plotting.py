import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def plot_elbow_silhouette(df: pd.DataFrame, title_suffix: str = "") -> None:
    """
    Plot elbow (inertia) and silhouette scores for candidate k values.

    Parameters
    ----------
    df : pandas.DataFrame
            DataFrame produced by `scan_k_metrics` containing at least the columns
            ``'k'``, ``'inertia'`` and ``'silhouette'``. Rows should correspond to
            candidate cluster counts (k).
    title_suffix : str, optional
            Suffix appended to each subplot title (default is ``""``).

    Returns
    -------
    None
            Displays a Matplotlib figure with two subplots (inertia and silhouette).

    Raises
    ------
    ValueError
            If ``df`` does not contain the required columns ``'k'``, ``'inertia'`` and
            ``'silhouette'``.

    Notes
    -----
    The left subplot shows within-cluster sum-of-squares (inertia) vs k (useful
    for the elbow method). The right subplot shows mean silhouette score vs k.
    Both plots include markers and light grid lines for readability.

    Examples
    --------
    >>> plot_elbow_silhouette(df_iris, title_suffix=" – Iris")
    """
    if not {"k", "inertia", "silhouette"} <= set(df.columns):
        raise ValueError("DataFrame must contain 'k', 'inertia', and 'silhouette'.")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)

    sns.lineplot(data=df, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title(f"Elbow (Inertia){title_suffix}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Within-cluster sum of squares (inertia)")
    axes[0].grid(True, alpha=0.3)

    sns.lineplot(
        data=df, x="k", y="silhouette", marker="o", color="tab:green", ax=axes[1]
    )
    axes[1].set_title(f"Silhouette{title_suffix}")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Mean silhouette")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_kmeans_feature_pair(
    X_in: np.ndarray,
    kmeans: KMeans,
    jx: int,
    jy: int,
    *,
    title: str = "",
    annotate_centroids: bool = True,
    x_name: str | None = None,
    y_name: str | None = None,
) -> plt.Axes:
    """
    Plot a 2-D scatter for a selected feature pair from the space used to fit KMeans.

    Parameters
    ----------
    X_in : ndarray, shape (n_samples, n_features)
            Feature matrix in the same space used to fit `kmeans` (e.g. scaled data).
    kmeans : sklearn.cluster.KMeans
            A fitted KMeans instance. Must expose ``cluster_centers_`` and support
            ``predict`` for assigning labels.
    jx : int
            Index of the feature to use for the x-axis.
    jy : int
            Index of the feature to use for the y-axis.
    title : str, optional
            Plot title. Default is an empty string.
    annotate_centroids : bool, optional
            If True, annotate centroid positions with their cluster index. Default is True.
    x_name : str or None, optional
            Axis label for x. If None, a generic "Feature {jx}" label is used.
    y_name : str or None, optional
            Axis label for y. If None, a generic "Feature {jy}" label is used.

    Returns
    -------
    matplotlib.axes.Axes
            The Axes object containing the scatter plot.

    Raises
    ------
    ValueError
            If the provided KMeans instance does not appear fitted (missing ``cluster_centers_``).
    IndexError
            If ``jx`` or ``jy`` are out of bounds for the columns of ``X_in`` or the cluster centers.

    Notes
    -----
    This function:
      - Predicts cluster labels for the rows of ``X_in`` using the supplied KMeans.
      - Plots the specified original features (columns ``jx`` and ``jy``) coloured by label.
      - Overlays the cluster-centroid projections in the same feature coordinates.

    Examples
    --------
    >>> ax = plot_kmeans_feature_pair(X_std, km, jx=2, jy=3,
    ...                               title="petal length vs width",
    ...                               x_name="petal length (cm)",
    ...                               y_name="petal width (cm)")
    >>> plt.show()
    """
    if not hasattr(kmeans, "cluster_centers_"):
        raise ValueError(
            "KMeans instance does not appear to be fitted (missing 'cluster_centers_')."
        )

    X_in = np.asarray(X_in)
    n_features = X_in.shape[1]
    if not (0 <= jx < n_features) or not (0 <= jy < n_features):
        raise IndexError(
            f"jx and jy must be in [0, {n_features-1}]. Got jx={jx}, jy={jy}."
        )

    labels = kmeans.predict(X_in)
    centers = kmeans.cluster_centers_

    # Validate centroid shape against requested indices
    if centers.shape[1] <= max(jx, jy):
        raise IndexError(
            "KMeans cluster_centers_ do not have requested feature indices."
        )

    x = X_in[:, jx]
    y = X_in[:, jy]
    cx = centers[:, jx]
    cy = centers[:, jy]

    # Create a DataFrame for Seaborn plotting
    df_plot = pd.DataFrame({"x": x, "y": y, "cluster": labels})

    # Set Seaborn style for polished look
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=130)

    # Use Seaborn scatterplot with hue for clusters
    sns.scatterplot(
        data=df_plot,
        x="x",
        y="y",
        hue="cluster",
        palette="tab10",
        edgecolor="black",
        s=50,
        ax=ax,
        alpha=0.8,
    )

    # Overlay centroids with Seaborn-compatible styling
    ax.scatter(
        cx,
        cy,
        s=160,
        c="red",
        marker="X",
        edgecolor="black",
        label="centroid",
        zorder=5,
    )

    if annotate_centroids:
        for idx, (u, v) in enumerate(zip(cx, cy)):
            ax.annotate(
                str(idx),
                (u, v),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=10,
                color="red",
            )

    ax.set_xlabel(x_name or f"Feature {jx}")
    ax.set_ylabel(y_name or f"Feature {jy}")
    ax.set_title(title or f"k-means clusters on features ({jx}, {jy})")
    ax.legend(title="Cluster", loc="best", frameon=True)
    return ax


def plot_kmeans_pca2(
    X_in: np.ndarray,
    kmeans: KMeans,
    *,
    title: str = "",
    annotate_centroids: bool = True,
) -> plt.Axes:
    """
    Plot a PCA(2) projection of X_in with predicted labels and centroid projections.

    Parameters
    ----------
    X_in : ndarray of shape (n_samples, n_features)
            Feature matrix in the same space used to fit KMeans (e.g. scaled data).
    kmeans : sklearn.cluster.KMeans
            Fitted KMeans instance providing .cluster_centers_ and .predict.
    title : str, optional
            Title for the plot. Default is an empty string.
    annotate_centroids : bool, optional
            If True, annotate centroid positions with their cluster index. Default is True.

    Returns
    -------
    matplotlib.axes.Axes
            The Axes object containing the scatter plot.

    Raises
    ------
    ValueError
            If the provided KMeans instance is not fitted (missing cluster_centers_).

    Notes
    -----
    This function fits a PCA with 2 components on X_in, projects both the input points
    and the cluster centers into the 2-D PCA space, then draws a scatter plot coloured by
    predicted cluster labels. The PCA is fit on X_in so the projection reflects the original
    feature-space geometry used for clustering.

    Examples
    --------
    >>> ax = plot_kmeans_pca2(X_std, km, title="k-means in PCA(2)")
    >>> plt.show()
    """
    if not hasattr(kmeans, "cluster_centers_"):
        raise ValueError("KMeans is not fitted.")
    labels = kmeans.predict(X_in)
    centers = kmeans.cluster_centers_
    pca2 = PCA(n_components=2).fit(X_in)
    Z = pca2.transform(X_in)
    C = pca2.transform(centers)

    # Set Seaborn style for polished look
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=130)

    # Create a DataFrame for Seaborn plotting
    df_plot = pd.DataFrame({"PC1": Z[:, 0], "PC2": Z[:, 1], "cluster": labels})

    # Use Seaborn scatterplot with hue for clusters
    sns.scatterplot(
        data=df_plot,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        edgecolor="black",
        s=50,
        ax=ax,
        alpha=0.8,
    )

    # Overlay centroids with Seaborn-compatible styling
    ax.scatter(
        C[:, 0],
        C[:, 1],
        s=160,
        c="red",
        marker="X",
        edgecolor="black",
        label="centroid",
        zorder=5,
    )

    if annotate_centroids:
        for idx, (u, v) in enumerate(C):
            ax.annotate(
                str(idx),
                (u, v),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=10,
                color="red",
            )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title or "k-means clusters in PCA(2) projection")
    ax.legend(title="Cluster", loc="best", frameon=True)
    return ax
