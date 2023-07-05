"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Defines Loss Functions for the various models.
"""

# ============= Import Libraries =============
from typing import List
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import RocCurveDisplay

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
    exp_term =  0.5 * torch.sum(((x - mu) / var) * (x - mu), dim=-1)   # (batch_size)
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

        # Compute True Positives, False Postiives and False Negatives
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

        # Compute True Positives, False Postiives and False Negatives
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
    - fig, ax pair of plt objects with the roc curves. Includes roc curve per each class, and also macro and micro averages.
    """

    # If class_names are empty then generate some placeholder names
    num_classes = y_true.shape[1]
    if class_names == []:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Iterate through each class
    for class_idx in range(num_classes):

        # Class true and predicted values
        y_true_class = y_true[:, class_idx]
        y_pred_class = y_pred[:, class_idx]

        # Compute 

# endregion
