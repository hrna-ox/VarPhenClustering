"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Define Metric computations for the various models
"""

# ============= Import Libraries =============
from typing import Union

import torch
from torchmetrics import AveragePrecision

import numpy as np
import sklearn.metrics as metrics

eps = 1e-8


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute accuracy score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.

    Returns:
        acc (_type_): scalar accuracy score.
    """

    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Compare pairwise and sum across tensor
    correct_pred = np.sum(labels_pred == labels_true)
    num_preds = np.size(labels_pred)

    # Compute accuracy
    acc = correct_pred / num_preds

    return acc

def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute Macro Multi-Class F1 score between true outcomes y_true, and predicted values y_pred.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - y_pred: (N, O) array of outcome probability predictions.

    Returns:
    - f1_score: scalar F1 score.
    """

    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Initialize tracker of f1 score
    f1_scores = []

    # Loop through each class
    for class_idx in range(y_true.shape[1]):

        # Compute True Positives, False Positives and False Negatives
        tp = np.sum((labels_pred == class_idx) & (labels_true == class_idx))
        fp = np.sum((labels_pred == class_idx) & (labels_true != class_idx))
        fn = np.sum((labels_pred != class_idx) & (labels_true == class_idx))

        # Compute precision and recall while disregarding any potential issues with division by zero
        with np.errstate(divide='ignore', invalid='ignore'):

            precision = np.divide(tp, tp + fp)
            recall = np.divide(tp, tp + fn)

            # Compute F1 score and append
            f1 = 2 * np.divide(precision * recall, precision + recall + eps)
            f1_scores.append(f1)

    return np.mean(f1_scores)

def micro_f1_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute Micro Multi-Class F1 score between true outcomes y_true, and predicted values y_pred.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - y_pred: (N, O) array of outcome probability predictions.

    Returns:
    - f1_score: scalar F1 score.
    """

    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Initialize trackers
    total_tp, total_fp, total_fn = 0, 0, 0

    # Loop through each class
    for class_idx in range(y_true.shape[1]):

        # Compute True Positives, False Positives and False Negatives
        tp = np.sum((labels_pred == class_idx) & (labels_true == class_idx))
        fp = np.sum((labels_pred == class_idx) & (labels_true != class_idx))
        fn = np.sum((labels_pred != class_idx) & (labels_true == class_idx))

        # Add to global values
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute f1 score
    f1 = 2 * np.divide(total_tp, 2 * total_tp + total_fp + total_fn)

    return f1

def recall_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute recall score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.

    Returns:
        recall (_type_): scalar recall score.
    """
    
    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Compute TP, FP, FN
    TP = np.sum((labels_pred == labels_true) & (labels_true == 1))
    FN = np.sum((labels_pred != labels_true) & (labels_true == 1))

    # Compute recall
    recall = TP / (TP + FN)

    return recall

def precision_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute precision score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (np.ndarray): (N, O) array of one-hot encoded outcomes.
        y_pred (np.ndarray): (N, O) array of outcome probability predictions.

    Returns:
        precision (float): scalar precision score.
    """
    
    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Compute TP, FP, FN
    TP = np.sum((labels_pred == labels_true) & (labels_true == 1))
    FP = np.sum((labels_pred != labels_true) & (labels_true == 0))

    # Compute precision
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = TP / (TP + FP)

        return precision

def get_roc_auc_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute ROC AUC score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (np.ndarray): (N, O) array of one-hot encoded outcomes.
        y_pred (np.ndarray): (N, O) array of outcome probability predictions.

    Returns:
        roc_auc (float): scalar ROC AUC score.
    """
    
    # Convert to labels
    labels_true = np.argmax(y_true, axis=-1).astype(int)

    # Compute Roc per class
    ovr_roc_scores = metrics.roc_auc_score(labels_true, y_score=y_pred, average=None,
                            multi_class="ovr")
    
    return ovr_roc_scores

def get_torch_pr_auc_score(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    Compute PR AUC score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (torch.Tensor): (N, O) array of one-hot encoded outcomes.
        y_pred (torch.Tensor): (N, O) array of outcome probability predictions.

    Returns:
        pr_auc (torch.Tensor): (O,) array of PR AUC scores per class.
    """
    
    # Convert to labels
    labels_true = torch.argmax(y_true, dim=-1)

    # Compute Area under the Curve for the Precision Recall Curve for each class
    _metric_wrapper  = AveragePrecision(task="multiclass", num_classes = y_true.shape[1], average=None)
    pr_scores = _metric_wrapper(y_pred, labels_true)

    return pr_scores


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute confusion matrix between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (np.ndarray): (N, O) array of one-hot encoded outcomes.
        y_pred (np.ndarray): (N, O) array of outcome probability predictions.

    Returns:
        confusion_matrix (np.ndarray): (O, O) confusion matrix.
    """
    
    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Initialize confusion matrix to zeros
    confusion_matrix = np.zeros((y_pred.shape[1], y_true.shape[1]))

    # Loop through each true class and then predicted class
    for pred_idx in range(y_pred.shape[1]):
        for true_idx in range(y_true.shape[1]):

            # Compute the number of elements with a given predicted class and a given true class
            confusion_matrix[true_idx, pred_idx] = np.sum(
                                            (labels_pred == pred_idx) &
                                            (labels_true == true_idx)
                                        )

    return confusion_matrix

    
def get_sup_scores(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor]):
    """
    Compute all supervised scores between true outcomes y_true, and predicted values y_pred.    
    """

    # Convert arrays, if needed
    if isinstance(y_true, torch.Tensor) and isinstance(y_pred, torch.Tensor):
        y_true_npy, y_pred_npy = y_true.detach().numpy(), y_pred.detach().numpy()
        y_true_torch, y_pred_torch = y_true, y_pred
    
    elif isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray):
        y_true_npy, y_pred_npy = y_true, y_pred
        y_true_torch, y_pred_torch = torch.from_numpy(y_true), torch.from_numpy(y_pred)

    else:
        raise ValueError("y_true and y_pred must both be either np.ndarray or torch.Tensor")


    # Compute scores for confusion matrix
    acc = accuracy_score(y_true_npy, y_pred_npy)
    macro_f1 = macro_f1_score(y_true_npy, y_pred_npy)
    micro_f1 = micro_f1_score(y_true_npy, y_pred_npy)
    recall = recall_score(y_true_npy, y_pred_npy)
    precision = precision_score(y_true_npy, y_pred_npy)

    # Compute Roc and PR AUC scores
    roc_auc = get_roc_auc_score(y_true_npy, y_pred_npy)
    pr_auc = get_torch_pr_auc_score(y_true_torch, y_pred_torch).numpy()

    # Compute Confusion Matrix
    conf_matrix = get_confusion_matrix(y_true_npy, y_pred_npy)

    # Return dictionary of scores
    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "conf_matrix": conf_matrix
    }


def get_clustering_label_metrics(y_true: np.ndarray, clus_pred: np.ndarray):
    """
    Compute various clustering metrics comparing the predicted clustering with the true labels. 

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - clus_pred: (N, T, K) array of cluster probability predictions over time

    Returns:
    - dict of metrics
    """
    
    # If no time step, then edit to 1.
    if len(clus_pred.shape) == 2:
        clus_pred = clus_pred[:, None, :]

    # Convert to labels
    labels_clus = np.argmax(clus_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Initialize outputs
    rand_scores, nmi_scores = [], []

    for t in range(labels_clus.shape[1]):

        # Compute metrics
        rand = metrics.adjusted_rand_score(labels_true, labels_clus[:, t])                  # Closer to 1 the better
        nmi = metrics.normalized_mutual_info_score(labels_true, labels_clus[:, t])   # Between 0 and 1, closer to 1 the better

        # Add to list
        rand_scores.append(rand)
        nmi_scores.append(nmi)

    return {"rand": rand_scores, "nmi": nmi_scores}


def compute_unsupervised_metrics(X: np.ndarray, clus_pred: np.ndarray, seed: int = 0):
    """
    Compute various unsupervised metrics comparing the predicted clustering with the predicted cluster assignments.
    For each time-step we compute the clustering metrics for the predicted clustering, but also for the data only up until the corresponding time step.

    The following metrics are computed: Silhouette Score, Davies-Bouldin Index, and the Variance Ratio Criterion.

    Args:
        X (np.ndarray): Data matrix of shape (N, T, D)
        clus_pred (np.ndarray): Cluster probability predictions of shape (N, T, K)
        seed (int): Random State parameter for Silhouette Coefficient Computation.

    Returns:
    - dict of metrics for each time step.
    """

    # If no time step, then edit to 1.
    if len(clus_pred.shape) == 2:
        clus_pred = clus_pred[:, None, :]

    # Get information
    N, T, K = clus_pred.shape

    # Convert to labels and right format
    labels_clus = np.argmax(clus_pred, axis=-1)

    # Initialize outputs
    sil_scores, vri_scores, dbi_scores = [], [], []

    # Iterate over time
    for t in range(T):
            
        # Reshape input data into right format
        X_t = X[:, :t+1, :].reshape(N, -1)

        # Compute metrics
        sil = metrics.silhouette_score(X_t, labels_clus[:, t], metric="euclidean", random_state=seed)      # Between -1 and 1, closer to 1 the better
        dbi = metrics.davies_bouldin_score(X_t, labels_clus[:, t])              # The higher the better
        vri = metrics.calinski_harabasz_score(X_t, labels_clus[:, t])   # >= 0, the lower the better

        # Add to list
        sil_scores.append(sil)
        dbi_scores.append(dbi)
        vri_scores.append(vri)

    return {"sil": sil_scores, "dbi": dbi_scores, "vri": vri_scores}

def get_clust_scores(y_true: Union[], clus_pred: np.ndarray, X: np.ndarray, seed: int = 0):
    """
    Compute various metrics related to evaluating clustering performance.

    Args:
        y_true (np.ndarray): True labels of shape (N, O)
        clus_pred (np.ndarray): Cluster probability predictions of shape (N, T, K)
        X (np.ndarray): Data matrix of shape (N, T, D)
        seed (int): Random State parameter for Silhouette Coefficient Computation.
    """
    # Convert arrays, if needed
    if isinstance(y_true, torch.Tensor) and isinstance(clus_pred, torch.Tensor):
        y_true_npy, clus_pred_npy = y_true.detach().numpy(), clus_pred.detach().numpy()
        y_true_torch, y_pred_torch = y_true, clus_pred
    
    elif isinstance(y_true, np.ndarray) and isinstance(clus_pred, np.ndarray):
        y_true_npy, clus_pred_npy = y_true, clus_pred
        y_true_torch, clus_pred_torch = torch.from_numpy(y_true), torch.from_numpy(clus_pred)

    else:
        raise ValueError("y_true and y_pred must both be either np.ndarray or torch.Tensor")
    