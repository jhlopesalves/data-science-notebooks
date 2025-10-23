import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_labeled_decision_regions(X, y, models):
    """
    Function producing a scatter plot of the instances contained
    in the 2D dataset (X,y) along with the decision
    regions of two trained classification models contained in the
    list 'models'.

    Parameters
    ----------
    X: pandas DataFrame corresponding to two numerical features
    y: pandas Series corresponding the class labels
    models: list containing two trained classifiers
    """
    # Assuming X has two features
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, clf in enumerate(models):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        axes[i].contourf(xx, yy, Z, alpha=0.3)
        sns.scatterplot(
            x=X.iloc[:, 0], y=X.iloc[:, 1], hue=y, ax=axes[i], edgecolor="k", s=20
        )
        axes[i].set_xlabel(X.columns[0])
        axes[i].set_ylabel(X.columns[1])
        axes[i].set_title(f"Decision Regions for {type(clf).__name__}")

    plt.tight_layout()
    plt.show()
