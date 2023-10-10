"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

General Plotter Tools for plotting Trajectory variables over time.
"""

# Import required packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from typing import Tuple, Union, List
import src.visualization.vis_utils as vis_utils

# Define functions
def plot_clus_trajectory_mean_sterror(clus_means: np.ndarray, clus_sterr: np.ndarray, feat_names: Union[List, None], time_idxs: Union[np.ndarray, List, None], ax=None):
    """
    Given the cluster means and standard errors, plot the trajectory of each cluster over time.
    
    Args:
        - clus_means (np.ndarray): of shape (K, T, F) the cluster means.
        - clus_sterr (np.ndarray): of shape (K, T, F) the cluster standard errors.
        - feat_names (List): of length D the names of the features.
        - time_idxs (Union[np.ndarray, List, None]): the time indices to be plotted. If None, all time indices are plotted.
        - ax (None or matplotlib.pyplot.Axes): the axis to be used for plotting. If None, a new figure is created.

    Returns:
        - ax (matplotlib.pyplot.Axes): the axis used for plotting.
    """

    # Unpack
    K, T, F = clus_means.shape

    # Prepare plots
    if ax is None:
        ax = plt.gca()
    axes = ax.reshape(-1)

    # Set time indices
    if time_idxs is None:
        time_idxs = np.arange(T)
    
    # Set Features
    if feat_names is None:
        feat_names = [f"Feat {idx}" for idx in range(F)]

    # Set colors
    colors = cm.get_cmap("tab10")(np.linspace(0, 1, K))

    # Iterate through features and clusters
    for feat_idx in range(F):
        for clus_idx, clus_label in enumerate(range(1, K+1)):

            # Plot mean and standard error
            axes[feat_idx].plot(time_idxs, clus_means[clus_idx, :, feat_idx], label=f"C{clus_label}", linestyle="-", color=colors[clus_idx])
            axes[feat_idx].plot(time_idxs, clus_means[clus_idx, :, feat_idx] - clus_sterr[clus_idx, :, feat_idx], linestyle="--", color=colors[clus_idx], alpha=0.2)
            axes[feat_idx].plot(time_idxs, clus_means[clus_idx, :, feat_idx] + clus_sterr[clus_idx, :, feat_idx], linestyle="--", color=colors[clus_idx], alpha=0.2)
        
        # Set title and legend
        axes[feat_idx].set_title(feat_names[feat_idx])
        axes[feat_idx].legend()

    return ax
