"""

Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Utility function for visualization functions.
"""

# Import required packages
import numpy as np
import matplotlib.pyplot as plt

from typing import Tuple, Union, List

# Define functions
def get_nrows_ncols(n_plots: int) -> Tuple[int, int]:
    """
    Get the number of rows and columns for a given number of plots to be displayed in the same figure.

    Args:
        - n_plots (int): the number of plots to be displayed.

    Returns:
        - nrows (int): the number of rows.
        - ncols (int): the number of columns.
    """

    # Get number of rows and columns
    nrows = int(np.ceil(np.sqrt(n_plots)))
    ncols = int(np.ceil(n_plots / nrows))

    return nrows, ncols

def decorate_ax(ax: plt.Axes, title: Union[str, None] = None, xlabel: Union[str, None] = None, ylabel: Union[str, None] = None, 
                xticks: Union[List, None] = None, yticks: Union[List, None] = None,
                xticklabels: Union[List, None] = None, yticklabels: Union[List, None] = None, legend: bool = False) -> plt.Axes:
    """
    Decorate a given axis.

    Args:
        - ax (plt.Axes): the axis to be decorated.
        - title (Union[str, None]): the title of the axis.
        - xlabel (Union[str, None]): the label of the x-axis.
        - ylabel (Union[str, None]): the label of the y-axis.
        - xticks (Union[str, None]): the ticks of the x-axis.
        - yticks (Union[str, None]): the ticks of the y-axis.
        - xticklabels (Union[str, None]): the tick labels of the x-axis.
        - yticklabels (Union[str, None]): the tick labels of the y-axis.
        - legend (bool): whether to display the legend.

    Returns:
        - ax (plt.Axes): the decorated axis.
    """

    # Decorate Axes
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if legend:
        ax.legend()

    return ax
