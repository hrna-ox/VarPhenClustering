"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Defines Loss Functions for the various models.
"""

# ============= Import Libraries =============
from typing import List
from matplotlib import pyplot as plt
from matplotlib.cm import get_cmap
import torch
import torch.nn as nn

import numpy as np
import sklearn.metrics as metrics

eps = 1e-8

# region =============== Loss Functions ===============

def torch_log_gaussian_lik(x, mu, var, device=None):
    """
    Log Likelihood for values x given multivariate normal with mean mu, and variances var.
    
    Params:
    - x: of shape (batch_size, input_size)
    - mu, var: of shape (input_size)
    - device: torch.device object (default None)

    Outputs:
    single values log likelihood for each sample in batch (batch_size)
    """

    # Get parameters
    _ , inp_size = x.size()

    # Compute exponential term
    exp_term =  0.5 * torch.sum(((x - mu) / var) * (x - mu), dim=-1)   # type: ignore # (batch_size)
    lin_term = torch.sum(torch.log(var), dim=-1, keepdim=False)   # (batch_size)
    cons_term = 0.5 * inp_size * torch.log(2 * torch.acos(torch.zeros(1)) * 2).to(device=device) # constant

    # Compute log likelihood
    log_lik = - cons_term - lin_term - exp_term

    return log_lik


def dir_kl_div(a1, a2):
    """
    Computes KL divergence of dirichlet distributions with parameters a1 and a2

    Inputs: a1, a2 array-like of shape (batch_size, K)

    Outputs: array of shape (batch_size) with corresponding KL divergence.
    """

    # Useful pre-computations
    a1_sum = a1.sum(dim=-1, keepdim=True)
    a2_sum = a2.sum(dim=-1, keepdim=True)

    # Compute log of gamma functions
    lgamma1, lgamma2 = torch.lgamma(a1), torch.lgamma(a2)
    lgamma1_sum, lgamma2_sum = torch.lgamma(a1_sum), torch.lgamma(a2_sum)

    # Compute digamma function for a1 and a1_sum
    digamma_1, digamma1_sum = torch.digamma(a1), torch.digamma(a1_sum)

    # Compute individual terms of lemma in paper
    term1 = lgamma1_sum - lgamma2_sum
    term2 = torch.sum(lgamma1 - lgamma2, dim=-1, keepdim=False)
    term3 = torch.sum((a1 - a2) * (digamma_1 - digamma1_sum), dim=-1, keepdim=False)
    
    # Combine all terms
    kl_div = torch.squeeze(term1) + term2 + term3

    return kl_div


def cat_cross_entropy(y_true, y_pred):
    """
    Compute categorical cross-entropy between y_true and y_pred. Assumes y_true is one-hot encoded.

    Parameters:
        - y_true: one-hot encoded of shape (batch_size, num_outcomes)
        - y_pred: of shape (batch_size, num_outcomes)

    Outputs:
        - categorical distribution loss for each sample in batch (batch_size)
    """

    cat_ce = - torch.sum(y_true * torch.log(y_pred + 1e-8), dim=-1)

    return cat_ce

# endregion

# region ============== Metrics ==============

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

        # Compute precision and recall
        precision = np.divide(tp, tp + fp)
        recall = np.divide(tp, tp + fn)

        # Compute F1 score and append
        f1 = 2 * np.divide(precision * recall, precision + recall)
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
    TP = torch.sum((labels_pred == labels_true) & (labels_true == 1))
    FN = torch.sum((labels_pred != labels_true) & (labels_true == 1))

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
    TP = torch.sum((labels_pred == labels_true) & (labels_true == 1))
    FP = torch.sum((labels_pred != labels_true) & (labels_true == 0))

    # Compute precision
    precision = TP / (TP + FP)

    return precision

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
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Compute Roc per class
    roc_scores = metrics.roc_auc_score(labels_true, labels_pred, average=None,
                            multi_class="ovr")

    return roc_scores

def get_pr_auc_score(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Compute PR AUC score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (np.ndarray): (N, O) array of one-hot encoded outcomes.
        y_pred (np.ndarray): (N, O) array of outcome probability predictions.

    Returns:
        pr_auc (float): scalar PR AUC score.
    """
    
    # Convert to labels
    labels_pred = np.argmax(y_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    # Compute Roc per class
    pr_scores = metrics.average_precision_score(labels_true, labels_pred, average=None)

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

def get_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = []):
    """
    Compute ROC curve between outcomes y_true, and predicted outcomes y_pred.

    Returns:
    - fig, ax pair of plt objects with the roc curves. Includes roc curve per each class.
    """

    # If class_names are empty then generate some placeholder names
    num_classes = y_true.shape[1]
    if class_names == []:
        class_names = [f"Class {i}" for i in range(num_classes)]
    roc_scores = get_roc_auc_score(y_true, y_pred)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = get_cmap("tab10").colors # type: ignore

    # Iterate through each class
    for class_idx in range(num_classes):

        # Class true and predicted values
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]

        # roc, auc values for this class
        auc = roc_scores[class_idx]# type: ignore

        # Compute ROC Curve
        metrics.RocCurveDisplay.from_predictions(
            y_true=y_true_class, y_pred=y_pred_class, 
            color=colors[class_idx], plot_chance_level=(class_idx==0),
            ax=ax, name=f"Class {class_idx} (AUC {auc:.2f}))", 
        )
        
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    fig.legend()

    return fig, ax


def get_pr_curve(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = []):
    """
    Compute PR curve between outcomes y_true, and predicted outcomes y_pred.

    Returns:
    - fig, ax pair of plt objects with the roc curves. Includes roc curve per each class.
    """

    # If class_names are empty then generate some placeholder names
    num_classes = y_true.shape[1]
    if class_names == []:
        class_names = [f"Class {i}" for i in range(num_classes)]
    pr_scores = get_pr_auc_score(y_true, y_pred)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = get_cmap("tab10").colors # type: ignore

    # Iterate through each class
    for class_idx in range(num_classes):

        # Class true and predicted values
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]

        # roc, auc values for this class
        prc = pr_scores[class_idx] # type: ignore

        # Compute ROC Curve
        metrics.PrecisionRecallDisplay.from_predictions(
            y_true=y_true_class, y_pred=y_pred_class, 
            color=colors[class_idx], plot_chance_level=True,
            ax=ax, name=f"Class {class_idx} (PR {prc:.2f}))", 
        )
        
    ax.set_title("PR Curves")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")

    fig.legend()

    return fig, ax



# endregion
