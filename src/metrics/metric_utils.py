"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Implements utility functions related to metrics.
"""

# Import required packages
import numpy as np
from typing import Literal

eps = 1e-8


# Utility functions
def convert_probabilities_to_odd_scores(probabilities: np.ndarray) -> np.ndarray:
    """
    Convert probabilities to scores using the odd-ratio transformation.

    Args:
        probabilities (np.ndarray): Probabilities. 

    Returns:
        np.ndarray: Scores. Expected shape (N, ).
    """

    # Assert that probabilities are in the correct range
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    # Apply odd-ratio transformation
    scores = np.divide(probabilities, 1 - probabilities + eps)

    return scores

def convert_odd_scores_to_0_1_values(scores: np.ndarray) -> np.ndarray:
    """
    Convert scores to 0-1 values using the inverse of the odd-ratio.

    Args:
        scores (np.ndarray): Scores. Expected shape (N, ).

    Returns:
        np.ndarray: 0-1 values. Expected shape (N, ).
    """

    # Assert that scores are in the correct range
    assert np.all(scores >= 0)

    # Apply inverse of odd-ratio transformation
    values = np.divide(scores, 1 + scores)

    return values

def convert_odd_scores_to_probabilities(scores : np.ndarray) -> np.ndarray:
    """
    Convert scores to probabilities using the inverse of the odd-ratio normalized to sum to 1

    Args:
        scores (np.ndarray): Scores. Expected shape (N, ....). Normalizes over the last axis.

    Returns:
        np.ndarray: Probabilities. Expected shape (N, ...).
    """

    # Assert that scores are in the correct range
    assert np.all(scores >= 0)

    # Apply inverse of odd-ratio transformation and normalize
    values = np.divide(scores, 1 + scores)
    probabilities = values / np.sum(values, axis=-1, keepdims=True)

    return probabilities

def is_target_binary_or_multiclass(target_labels: np.ndarray) -> Literal["binary", "multiclass"]:
    """
    Check if the target_labels is binary or multiclass.
    """

    if len(np.unique(target_labels)) > 2:
        return 'multiclass'
    else:
        return 'binary'
    