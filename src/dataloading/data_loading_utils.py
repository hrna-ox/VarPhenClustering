"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file defines utility functions for the Data Loading process.
"""

# Import required packages
from typing import Iterable
import numpy as np

from typing import Union

def _numpy_forward_fill(array: np.ndarray) -> np.ndarray:
    """
    Forward Fill a numpy array. 

    Args:
        array: Numpy array to be forward filled.
    
    Returns:
        array_out: Forward filled array. Each (a, _, b) sample is forward filled along the time dimension (assumed to be axis=1).
    """

    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Add time indices where not masked, and propagate forward
    inter_array = np.where(
        ~array_mask, 
        np.arange(array_mask.shape[1]).reshape(1, -1, 1), 
        0
    )
    np.maximum.accumulate(
        inter_array, axis=1, out=inter_array
    )  # For each (n, t, d) missing value, get the previously accessible mask value

    # Index matching for output. For n, d sample as previously, use inter_array for previous time id
    array_out = array_out[
        np.arange(array_out.shape[0])[:, None, None], # type: ignore
        inter_array,
        np.arange(array_out.shape[-1])[None, None, :], # type: ignore
    ]

    return array_out

def _numpy_backward_fill(array: np.ndarray) -> np.ndarray:
    """
    Backwards Fill a numpy array. Useful for estimating values prior to the first observation.

    Args:
        array: Numpy array to be forward filled.
    
    Returns:
        array_out: Backward filled array. Each (a, _, b) sample is Backward filled along the time dimension (assumed to be axis=1).
    """

    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Add time indices where not masked, and propagate backward
    inter_array = np.where(
        ~array_mask,
        np.arange(array_mask.shape[1]).reshape(1, -1, 1),
        array_mask.shape[1] - 1,
    )
    inter_array = np.minimum.accumulate(inter_array[:, ::-1], axis=1)[:, ::-1]
    array_out = array_out[
        np.arange(array_out.shape[0])[:, None, None],
        inter_array,
        np.arange(array_out.shape[-1])[None, None, :],
    ]

    return array_out

def _median_fill(array: np.ndarray) -> np.ndarray:
    """
    Median Fill an array. 

    Args:
        array to be median filled.
    
    Returns:
        array_out: Median filled array. 
    """

    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Compute median and impute
    array_med = np.nanmedian(
        np.nanmedian(array, axis=0, keepdims=True), 
        axis=1, 
        keepdims=True
    )
    array_out = np.where(
        array_mask, 
        array_med, 
        array_out
    )

    return array_out

def impute(array):
    return _median_fill(_numpy_backward_fill(_numpy_forward_fill(array)))

def update_dictionary(dic: dict, values: Iterable):
    """
    Update dictionary given iterable of tuples.

    arg:
        dic: Dictionary to be updated.
        values: Iterable of tuples to update dictionary.

    Returns:
        dic: Updated dictionary.
    """

    for key, value in values:
        if key in dic.keys():
            raise ValueError(f"Key {key} already in dictionary.")
        
        else:
            dic[key] = value

def min_max_normalize(array: np.ndarray, _min: Union[np.ndarray, None] = None, _max: Union[np.ndarray, None] = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-Max normalize array.

    Args:
        array: Array to be normalized.
        min: Minimum value to normalize array. (defaults to None, in which case learn from array values)
        max: Maximum value to normalize array. (defaults to None, in which case learn from array values)

    Returns:
        array_out: Normalized array.
    """
    if _min is None:
        _min = np.min(array, axis=0, keepdims=True)

    if _max is None:
        _max = np.max(array, axis=0, keepdims=True)
    assert isinstance(_min, np.ndarray) and isinstance(_max, np.ndarray)

    # Compute normalized array while checking for potential divisions by 0
    array_out = np.where(
        _max == _min,
        0,
        (array - _min) / (_max - _min)
    )

    return array_out, _min, _max
