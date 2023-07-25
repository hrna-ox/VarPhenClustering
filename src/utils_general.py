"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Define general utility functions for all folders.
"""

# ============= Import Libraries =============
from typing import Union, List

import torch
import numpy as np

ARRAY_LIKE = Union[List[float], np.ndarray, torch.Tensor]


# ============= Define Utility Functions =============
def convert_to_npy(*args):
    """
    Convert a sequence of arguments iteratively to numpy arrays.

    Raises:
        ValueError: if any args has type different from np.ndarray or torch.Tensor.

    Returns:
        sequence of npy arrays 
    """
    for arg in args:
        if isinstance(arg, np.ndarray):
            yield arg
        elif isinstance(arg, torch.Tensor):
            yield arg.detach().numpy()
        else:
            try:
                yield np.array(arg)
            except ValueError:
                raise ValueError("Argument has to be either np.ndarray or torch.Tensor.")


def _weighted_averaging(inputs: ARRAY_LIKE, weights: Union[None, ARRAY_LIKE] = None, dim: Union[int, None] = None) -> ARRAY_LIKE:
    """
    Compute weighted average of scores. 

    Args:
        inputs (List): list of input values to normalize
        weights (Union[None, List], optional): weights for each score. Defaults to None, in which case applies uniformly.
        dim (Union[int, None], optional): dimension to apply weighted average. Defaults to None, in which case applies to all dimensions.

    Returns:
        ARRAY_LIKE objected of weighted averages across the corresponding dimension.
    """
    
    # Compute weighted average if numpy array
    if isinstance(inputs, np.ndarray) or isinstance(inputs, List):
        weighted_avg = np.average(inputs, weights=weights, axis=dim, keepdims=False)

    # Compute weighted average if torch tensor
    elif isinstance(inputs, torch.Tensor):
        if weights is None:
            weighted_avg = torch.mean(inputs, dim=dim, keepdim=False)
        
        else:
            assert torch.sum(weights) != 0, "Weights cannot be all zero."   # type: ignore
            weighted_avg = torch.sum(inputs * weights, dim=dim, keepdim=False) / torch.sum(weights)  # type: ignore

    return weighted_avg
