import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_confusion_matrices(plot_data, class_labels):
    """
    Plot and compare multiple confusion matrices with annotated performance metrics.

    This function visualizes a list of 2x2 confusion matrices as annotated heatmaps,
    allowing for side-by-side comparison of classification performance across different models
    or configurations. For each confusion matrix, key metrics—Accuracy, Precision, and Recall—
    are computed and displayed in the subplot title. Each cell of the heatmap is annotated with
    the corresponding confusion matrix entry, its label (e.g., True Positive), and the percentage
    relative to the total number of samples.

    Args:
        plot_data (list of tuples):
            A list where each element is a tuple containing:
                - confusion_matrix (np.ndarray): A 2x2 confusion matrix.
                - title_prefix (str): A string to prefix the subplot title (e.g., model name).
        class_labels (list of str):
            A list of two strings representing the class labels, in the order [Negative, Positive].

    Example:
        >>> cm1 = np.array([[50, 10], [5, 100]])
        >>> cm2 = np.array([[45, 15], [10, 90]])
        >>> plot_data = [(cm1, "Model A"), (cm2, "Model B")]
        >>> plot_confusion_matrices(plot_data, ["No Churn", "Churn"])

    Notes:
        - The function is designed for binary classification confusion matrices (2x2).
        - The heatmaps are displayed in a single row for easy visual comparison.
        - Metrics are calculated as follows:
            Accuracy = (TP + TN) / Total
            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
        - Each cell annotation includes the label, count, and percentage of total samples.

    Raises:
        ValueError: If any confusion matrix is not 2x2.
    """
    n_plots = len(plot_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(10 * n_plots, 10))

    # Ensure 'axes' is always iterable, even for a single plot
    if n_plots == 1:
        axes = [axes]

    group_names = ["True Negative", "False Positive", "False Negative", "True Positive"]

    for ax, (cm, title_prefix) in zip(axes, plot_data):
        if cm.shape != (2, 2):
            raise ValueError(
                "Each confusion matrix must be 2x2 for binary classification."
            )

        tn, fp, fn, tp = cm.flatten()
        total = np.sum(cm)
        accuracy = (tp + tn) / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        group_counts = [f"{value}" for value in [tn, fp, fn, tp]]
        group_percentages = [f"{value/total:.2%}" for value in [tn, fp, fn, tp]]
        box_labels = [
            f"{name}\n\n{count}\n({percent})"
            for name, count, percent in zip(
                group_names, group_counts, group_percentages
            )
        ]
        box_labels = np.asarray(box_labels).reshape(2, 2)

        title = (
            f"{title_prefix}\n\n"
            f"Accuracy: {accuracy:.2%} | Recall: {recall:.2%} | Precision: {precision:.2%}"
        )

        sns.heatmap(
            cm,
            annot=box_labels,
            fmt="s",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={"size": 16, "va": "center"},
        )
        ax.set(xlabel="Predicted Label", ylabel="True Label", title=title)
        ax.title.set_fontsize(18)
        ax.set_xlabel("Predicted Label", fontsize=15)
        ax.set_ylabel("True Label", fontsize=15)
        ax.xaxis.label.set_size(16)
        ax.yaxis.label.set_size(16)
        ax.tick_params(axis="both", labelsize=15)

    plt.tight_layout(pad=3.0)
    plt.show()
