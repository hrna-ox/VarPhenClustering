"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file defines useful utility functions for computing and estimating phenotypes over the outcome dimension.
"""

# Import required packages
import numpy as np

from typing import Union, List


# Define functions
def combine_assignments_and_pat_data_to_arr(pis_assign: np.ndarray, pat_data: np.ndarray) -> np.ndarray:
    """
    Given two np ndarrays corresponding to cluster assignment probabilities/one-hot encodings, and patient data (potentially temporal), combine them into a single array.
    
    Args:
        - pis_assign (np.ndarray): of shape (N, K) or (T, N, K) the cluster assignment probability for each patient.
        - pat_data (np.ndarray): of shape (N, D) or (T, N, D) the patient data.

    Returns:
        - combined array of shape (K, N, D) or (T, K, N, D). Note that in case of temporal data, we match the temporal axis of both inputs. NaN values
        are used to disregard clusters not associated with a particular sample.
    """

    # Check inputs
    assert pis_assign.shape[-2] == pat_data.shape[-2]
    assert pis_assign.ndim in [2, 3] 
    assert pat_data.ndim in [2, 3]

    # Convert probabilities to one-hot encodings of the most likely cluster
    pis_assign_one_hot = np.eye(pis_assign.shape[-1])[np.argmax(pis_assign, axis=-1)]

    # Augment dimensions to allow for broadcasting
    if pis_assign.ndim == pat_data.ndim:
        clus_augment = pis_assign_one_hot[..., np.newaxis]
        pat_augment = pat_data[:, np.newaxis, :]
    
    elif pis_assign.ndim == 2 and pat_data.ndim == 3:
        clus_augment = pis_assign_one_hot[np.newaxis, ..., np.newaxis]
        pat_augment = pat_data[:, np.newaxis, :]

    elif pis_assign.ndim == 3 and pat_data.ndim == 2:
        clus_augment = pis_assign_one_hot[..., np.newaxis, :]
        pat_augment = pat_data[np.newaxis, :, np.newaxis, :]

    else:
        raise ValueError("Wrong array dimensions specified.")
    

    # Combine cluster and patient data by multiplication the broadcasted arrays
    combined_arr = clus_augment * pat_augment              # Shape (N, K, D) or (T, N, K, D)

    # Set values to NaN whenever the cluster assignment is zero
    combined_arr[clus_augment == 0, :] = np.nan

    return combined_arr

def estimate_empirical_phen_from_assignment(pis_assign: np.ndarray, pat_outcomes: np.ndarray, weighted: bool = False):
    """
    Estimate the empirically-derived phenotypes given assignment probabilities/one-hot encodings, and the data indicating patient outcomes. Weighted parameter
    determines whether to use cluster likelihood of assignment, or simply computing the most likely cluster.

    Args:
        - pis_assign (np.ndarray): of shape (N, K) or shape (T, N, K) the cluster assignment probability for each patient.
        - pat_outcomes (np.ndarray): of shape (N, O) the one-hot encoding of patient outcomes.
        - weighted (bool): whether to use cluster likelihood of assignment, or simply computing the most likely cluster for each patient.

    Returns:
        - phen_empirical (np.ndarray): of shape (K, O) or (T, K, O) the empirical phenotype matrix.
    """

    # Separate into weighted and unweighted cases
    if weighted:

        # Compute empirical phenotype via matrix multiplication by swapping N and K axes
        phen_empirical = np.swapaxes(pis_assign, axis1=-2, axis2=-1) @ pat_outcomes

    else:
        
        # Convert probabilities to one-hot encodings of the most likely cluster
        pis_assign_one_hot = np.eye(pis_assign.shape[-1])[np.argmax(pis_assign, axis=-1)]
        
        # Compute empirical phenotype
        phen_empirical = np.swapaxes(pis_assign_one_hot, axis1=-2, axis2=-1) @ pat_outcomes

    return phen_empirical

def get_cluster_feature_quantiles(pis_assign: np.ndarray, pat_features: np.ndarray, quantiles: Union[np.ndarray, List, float]) -> np.ndarray:
    """
    Given cluster assignment probabilities/one-hot assignments, and patient feature data, compute the quantiles of each feature for each cluster.

    Args:
        - pis_assign (np.ndarray): of shape (N, K) or (T, N, K) the cluster assignment probability for each patient.
        - pat_features (np.ndarray): of shape (N, D) or (T, N, D) the patient feature data.
        - quantiles (np.ndarray or List): the quantiles to compute for each feature.

    Returns:
        - cluster_feature_quantiles (np.ndarray): of shape (K, D, Q) or (T, K, D, Q) the quantiles of each feature for each cluster.
    """

    # Combine cluster assignment and patient data
    combined_arr = combine_assignments_and_pat_data_to_arr(pis_assign, pat_features)     

    # Compute quantiles
    cluster_feature_quantiles = np.nanquantile(combined_arr, quantiles, axis=-2)     # Shape (K, D, Q) or (T, K, D, Q)

    return cluster_feature_quantiles

def get_cluster_feature_mean_trajectories(pis_assign, pat_trajectories):
    "Special Case of Cluster Feature Quantiles. Assumes pat_trajectories are 3 dimensional."
    return get_cluster_feature_quantiles(pis_assign, pat_trajectories, quantiles=0.5)

def get_cluster_feature_trajectories_sterror(pis_assign, pat_trajectories):
    "Compute standard error of the mean of each cluster feature trajectory."

    # Combine cluster assignment and patient data
    combined_arr = combine_assignments_and_pat_data_to_arr(pis_assign, pat_trajectories)

    # Compute standard error of the mean divided by the size of the cluster groups, i.e. the non NaN values
    cluster_feature_sterror = np.nanstd(combined_arr, axis=-2) / np.sqrt(np.sum(~np.isnan(combined_arr), axis=-2))

    return cluster_feature_sterror
