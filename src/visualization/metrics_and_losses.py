"""

Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Plot the output from the metric computation.
"""

# Import required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, List, Dict
import src.visualization.vis_utils as vis_utils

# Define functions
def _plot_single_multiclass_metric(metric_values, class_names: List, name: str = "", ax=None):
    """
    Save multiclass metric values as a scatter plot.

    Args:
        - metric_values (np.ndarray): the metric values.
        - class_names (List): the names of the classes.
        - name (str): the name of the metric.
        - ax (None or matplotlib.pyplot.Axes): the axis to be used for plotting. If None, a new figure is created.

    Returns:
        - ax (matplotlib.pyplot.Axes): the axis used for plotting.
    """

    # Prepare plots
    if ax is None:
        ax = plt.gca()

    # Plot and Decorate
    print("Metric Name", name)
    print(class_names)
    print(metric_values)
    ax.scatter(np.arange(len(class_names)), metric_values, s=100, label=name)

    return ax


def plot_multiclass_metrics(metrics_dict: Dict, class_names: Union[List, None]):
    """
    Plot all multiclass metrics (the standard ones related to Confusion Matrix, as well as the ones related to areas under the curve). Includes Lachiche if available.

    Args:
        - metrics_dict (Dict): the dictionary containing all metrics.
        - class_names (Union[List, None]): the names of the classes. Defaults to None.

    Returns:
        - Tuple of matplotlib.pyplot.Axes: the axes used for plotting if lachiche, else a single axis.
    """

    STANDARD_METRICS = ["precision", "recall", "macro_f1_score", "ovr_auroc", "ovr_auprc"]

    # Prepare plots
    fig, ax = plt.subplots(figsize=(20, 10))
    if class_names is None:
        class_names = [f"C{i}" for i in range(metrics_dict["precision"].shape[0])]


    # Plot standard metrics
    for metric_name in STANDARD_METRICS:
        ax = _plot_single_multiclass_metric(metrics_dict[metric_name], class_names, metric_name, ax)
    
    # Decorate axes
    ax = vis_utils.decorate_ax(ax=ax, title="Multiclass Metrics",
                            xlabel="Class", ylabel="Metric Value",
                            xticklabels=class_names, xticks=list(range(1, 1 + len(class_names))),
                            yticks=list(np.arange(0, 1.1, 0.1)),
                            legend=True)

    if "lachiche_metrics" in metrics_dict.keys():
        lachiche_ax = plot_multiclass_metrics(metrics_dict["lachiche_metrics"], class_names)
    else:
        lachiche_ax = None

    return ax, lachiche_ax


def plot_multiclass_confusion_matrix(confusion_matrix: np.ndarray, class_names: Union[List, None], ax=None):
    """
    Given a confusion matrix, plot the resulting values as a heatmap.

    Args:
        - confusion_matrix (np.ndarray): the confusion matrix.
        - class_names (Union[List, None]): the names of the classes.
        - ax (None or matplotlib.pyplot.Axes): the axis to be used for plotting. If None, a new figure is created.
    """

    # Prepare plots
    if ax is None:
        ax = plt.gca()

    if class_names is None:
        class_names = [f"C{i}" for i in range(1, confusion_matrix.shape[0] + 1)]

    # Plot heatmap
    ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", ax=ax, cmap="Blues")

    # Decorate Axes
    ax = vis_utils.decorate_ax(ax=ax, title="Confusion Matrix", 
                            xlabel="Predicted", ylabel="True", 
                            xticklabels=class_names, yticklabels=class_names
                        )

    return ax


def plot_losses(train_losses: Dict, val_losses: Union[Dict, None], test_loss: Union[Dict, None], ax=None) -> plt.Axes:
    """
    Given dictionary of losses (potentially including validation losses, as well as the loss on the test set), plot the losses over epochs.

    Args:
        - train_losses (Dict): each key is a loss name, and each value is a list of the loss values over epochs.
        - val_losses (Dict): each key is a loss name, and each value is a list of the loss values over epochs.
        - test_loss (Dict): each key is a loss name, and each value is a list of the loss values over epochs.

    Returns:
        - ax (matplotlib.pyplot.Axes): the axis used for plotting.
    """
    
    # Prepare plots
    if ax is None:
        ax = plt.gca()
    axes = ax.reshape(-1)

    for ax_idx, (loss_name, loss_values) in enumerate(train_losses.items()):
        
        # Plot loss
        epochs = np.arange(len(loss_values)) + 1
        axes[ax_idx].plot(epochs, loss_values, label="Train {}".format(loss_name), linestyle="-", color="b")

        # Plot validation loss if available
        if val_losses is not None:
            axes[ax_idx].plot(epochs, val_losses[loss_name], label=f"Val {loss_name}", linestyle="-", color="r")

        # Plot test loss if available
        if test_loss is not None:
            axes[ax_idx].scatter(epochs[-1], test_loss[loss_name], label=f"Test {loss_name}", color="g")

        # Decorate
        axes[ax_idx] = vis_utils.decorate_ax(
            ax=axes[ax_idx], title=loss_name, 
            xlabel="Epochs", ylabel="Loss", 
            legend=True if ax_idx == 0 else False)
    
    return axes

