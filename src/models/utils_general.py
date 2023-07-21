"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Defines general utility functions that are relevant for all models.
"""

# ============= Import Libraries =============
import torch
import numpy as np

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

