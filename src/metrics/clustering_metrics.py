"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Defines metrics for clustering. This is largely just a wrapper for sklearn.metrics.
"""

# Import required packages
import numpy as np
import sklearn.metrics as sklearn_metrics
from sklearn.metrics import adjusted_mutual_info_score


def cluster_adjusted_rand_score(labels_true: np.ndarray, clus_pred: np.ndarray) -> np.float32:
    """
    Compute the adjusted Rand score of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        clus_pred (np.ndarray): Predicted cluster labels.

    Returns:
        float: Adjusted Rand score.
    """
    rand_score =  sklearn_metrics.adjusted_rand_score(labels_true, clus_pred)

    return np.float32(rand_score)

def cluster_normalized_mutual_info_score(labels_true: np.ndarray, clus_pred: np.ndarray) -> np.float32:
    """
    Compute the normalized mutual information score of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        clus_pred (np.ndarray): Predicted cluster labels.

    Returns:
        float: Normalized mutual information score.
    """
    nmi_score =  sklearn_metrics.normalized_mutual_info_score(labels_true, clus_pred)

    return np.float32(nmi_score)

def cluster_adjusted_normalized_mutual_information(labels_true: np.ndarray, clus_pred: np.ndarray) -> np.float32:
    """
    Compute the adjusted normalized mutual information of the predictions. This is the normalized mutual information 
    divided by the maximum possible normalized mutual information.

    Args:
        labels_true (np.ndarray): True labels.
        clus_pred (np.ndarray): Predicted cluster labels.

    Returns:
        float: adjusted normalized mutual information.
    """
    return np.float32(adjusted_mutual_info_score(labels_true, clus_pred))


def cluster_fowlkes_mallows_score(labels_true: np.ndarray, clus_pred: np.ndarray) -> np.float32:
    """
    Compute the Fowlkes-Mallows score of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        clus_pred (np.ndarray): Predicted cluster labels.

    Returns:
        float: Fowlkes-Mallows score.
    """
    fm_score =  sklearn_metrics.fowlkes_mallows_score(labels_true, clus_pred)

    return np.float32(fm_score)

def cluster_silhouette_score(X: np.ndarray, clus_pred: np.ndarray, seed: int = 1313) -> np.float32:
    """
    Compute the silhouette score of the predictions.

    Args:
        X (np.ndarray): Data.
        clus_pred (np.ndarray): Predicted cluster labels.
        seed (int, optional): Random seed. Defaults to 1313.

    Returns:
        float: Silhouette score.
    """
    sil_score =  sklearn_metrics.silhouette_score(X, clus_pred, random_state=seed)

    return np.float32(sil_score)

def cluster_calinski_harabasz_score(X: np.ndarray, clus_pred: np.ndarray) -> np.float32:
    """
    Compute the Calinski-Harabasz score of the predictions.

    Args:
        X (np.ndarray): Data.
        clus_pred (np.ndarray): Predicted cluster labels.

    Returns:
        float: Calinski-Harabasz score.
    """
    ch_score =  sklearn_metrics.calinski_harabasz_score(X, clus_pred)

    return np.float32(ch_score)

def cluster_davies_bouldin_score(X: np.ndarray, clus_pred: np.ndarray) -> np.float32:
    """
    Compute the Davies-Bouldin score of the predictions.

    Args:
        X (np.ndarray): Data.
        clus_pred (np.ndarray): Predicted cluster labels.

    Returns:
        float: Davies-Bouldin score.
    """
    db_score =  sklearn_metrics.davies_bouldin_score(X, clus_pred)

    return np.float32(db_score)
