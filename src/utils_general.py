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

eps = 1e-8

# ============= Define Utility Functions =============
def convert_to_npy(*args):
    """
    Convert a sequence of arguments iteratively to numpy arrays.

    Raises:
        ValueError: if any args has type different from np.ndarray or torch.Tensor.

    Returns:
        sequence of npy arrays or 
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


def convert_to_torch(*args):
    """
    Convert a sequence of arguments iteratively to torch tensors.

    Raises:
        ValueError: if any args has type different from np.ndarray or torch.Tensor.

    Returns:
        sequence of torch tensors.
    """
    for arg in args:
        if isinstance(arg, torch.Tensor):
            yield arg
        elif isinstance(arg, np.ndarray):
            yield torch.from_numpy(arg)
        else:
            try:
                yield torch.tensor(arg)
            except ValueError:
                raise ValueError("Argument has to be either np.ndarray or torch.Tensor.")
            

def _convert_to_labels_if_score(x: np.ndarray) -> np.ndarray:
    """
    Convert predictions/one-hot encoded outcomes to labels.
    """

    if x.ndim == 1:           # Argument already is converted to labels
        return x.astype(int)
    elif x.ndim >= 2:
        return np.argmax(x, axis=-1).astype(int)
    else:
        raise ValueError("Input array has to be 1D or 2D. Got {}D.".format(x.ndim))


def _convert_prob_to_score(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert probability predictions to score values using the x / 1 - x conversion.

    Args: 
        - x (np.ndarray or torch.Tensor): array of probability values between 0 and 1.
    """
    if isinstance(x, np.ndarray):
        return np.divide(x, 1 - x + eps)

    elif isinstance(x, torch.Tensor):
        return torch.divide(x, 1 - x + eps)



def _convert_scores_to_prob(x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert score predictions to probability values using the x / 1 + x conversion and then normalizing (weighting scores might not lead necessarily to summing probability values).

    Args:
        - x (np.ndarray or torch.Tensor): array of score values > 0
    """

    if isinstance(x, np.ndarray):
        return np.divide(x, 1 + x + eps)
    elif isinstance(x, torch.Tensor):
        return torch.divide(x, 1 + x + eps)



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
