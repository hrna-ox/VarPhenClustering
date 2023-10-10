"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Define metrics for binary classification.
"""

# Import required packages
import numpy as np
import sklearn.metrics as sklearn_metrics

from typing import Literal

eps = 1e-8         # Small number to avoid division by zero


def binary_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute the accuracy of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy.
    """
    return np.mean(labels_true == labels_pred)


def binary_true_false_pos_neg(labels_true: np.ndarray, labels_pred: np.ndarray) -> tuple:
    """
    Compute the number of true positives, true negatives, false positives and false negatives.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        tuple: True positives, true negatives, false positives and false negatives.
    """
    true_pos = np.sum((labels_true == 1) & (labels_pred == 1))
    true_neg = np.sum((labels_true == 0) & (labels_pred == 0))
    false_pos = np.sum((labels_true == 0) & (labels_pred == 1))
    false_neg = np.sum((labels_true == 1) & (labels_pred == 0))

    return true_pos, true_neg, false_pos, false_neg

def binary_f1_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute the F1 score of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: F1 score.
    """
    true_pos, _, false_pos, false_neg = binary_true_false_pos_neg(labels_true, labels_pred)

    precision = true_pos / (true_pos + false_pos + eps)
    recall = true_pos / (true_pos + false_neg + eps)

    return 2 * precision * recall / (precision + recall + eps)

def binary_precision(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute the precision of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: Precision.
    """
    true_pos, _, false_pos, _ = binary_true_false_pos_neg(labels_true, labels_pred)

    return true_pos / (true_pos + false_pos + eps)

def binary_recall(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute the recall of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: Recall.
    """
    true_pos, _, _, false_neg = binary_true_false_pos_neg(labels_true, labels_pred)

    return true_pos / (true_pos + false_neg + eps)

def binary_auroc(labels_true: np.ndarray, scores_pred: np.ndarray, *args, **kwargs) -> np.float32:
    """
    Compute the AUROC of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        scores_pred (np.ndarray): Predicted scores.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        float: AUROC.
    """

    # Call scikit-learn function
    auroc = sklearn_metrics.roc_auc_score(labels_true, scores_pred, *args, **kwargs)

    return np.float32(auroc)

def binary_auprc(labels_true: np.ndarray, scores_pred: np.ndarray, *args, **kwargs) -> np.float32:
    """
    Compute the AUPRC of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        scores_pred (np.ndarray): Predicted scores.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        float: AUPRC.
    """

    # Call scikit-learn function
    auprc = sklearn_metrics.average_precision_score(labels_true, scores_pred, *args, **kwargs)

    return np.float32(auprc)

def binary_confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: Confusion matrix.
    """
    true_pos, true_neg, false_pos, false_neg = binary_true_false_pos_neg(labels_true, labels_pred)

    return np.array([[true_pos, false_pos], [false_neg, true_neg]], dtype=np.int32)

def find_best_threshold(labels_true: np.ndarray, scores_pred: np.ndarray, 
        target = Literal["accuracy", "f1"]) -> np.float32:
    """
    Compute the optimal threshold given an array of scores to obtain best value of metric.
    """

    # Define metric to optimise
    if target == "accuracy":
        metric = binary_accuracy
    elif target == "f1":
        metric = binary_f1_score

    # Compute possible threshold values in descending order
    _sort_indices = np.argsort(scores_pred)[::-1] # Sort in descending order
    sorted_scores = scores_pred[_sort_indices]
    sorted_labels = labels_true[_sort_indices]

    # Initalize targets 
    best_threshold = 0
    best_value = metric(  # type: ignore
        labels_true = sorted_labels,
        labels_pred = (sorted_scores >= best_threshold).astype(np.int8)
    )

    # Iterate over possible thresholds
    for threshold in sorted_scores:
        new_value = metric(  # type: ignore
            labels_true = sorted_labels,
            labels_pred = (sorted_scores >= threshold).astype(np.int8)
        )

        if new_value > best_value:
            best_value = new_value
            best_threshold = threshold

    return np.float32(best_threshold)
