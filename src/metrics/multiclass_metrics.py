"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Define metrics for multiclass classification. Assumes that labels are integers from 0 to n_classes - 1, and scores are arrays of dimensions (N, n_classes).
"""

# Import required packages
import numpy as np
import src.metrics.binary_metrics as binary_metrics

from typing import List, Literal

eps = 1e-8         # Small number to avoid division by zero


def multiclass_accuracy(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute the accuracy of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: np.ndarray of accuracies for each class, computed as the number of correct predictions divided by the total number of predictions for each class.
    """
    classes = np.unique(labels_true)

    # Compute the accuracy for each class
    accuracy = np.zeros(len(classes), dtype=np.float32)

    # Loop over classes - note sum != 1 because of size of division.
    for idx, c in enumerate(classes):

        is_true_class_c = labels_true == c
        is_pred_class_c = labels_pred == c
        accuracy[idx] = np.mean(is_true_class_c == is_pred_class_c)
    
    return accuracy

def multiclass_true_false_pos_neg(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute, for each class, the true positives, true negatives, false positives and false negatives.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        tuple: np.ndarray of true positives, true negatives, false positives and false negatives for each class.
    """
    classes = np.unique(labels_true)
    multiclass_tf_pn = np.zeros((len(classes), 4), dtype=np.int8)

    # Loop over classes
    for idx, c in enumerate(classes):

        is_true_class_c = labels_true == c
        is_pred_class_c = labels_pred == c

        # Compute binary predictions
        tp_c, tn_c, fp_c, fn_c = binary_metrics.binary_true_false_pos_neg(labels_true=is_true_class_c, labels_pred=is_pred_class_c)

        # Store results
        multiclass_tf_pn[idx, :] = np.array([tp_c, tn_c, fp_c, fn_c])

    return multiclass_tf_pn
    

def multiclass_macro_f1_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute the macro F1 score of the predictions. This iterates over each class independently and appending the values.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: np.ndarray of macro F1 scores for each class, computed as the harmonic mean of precision and recall for each class.
    """
    classes = np.unique(labels_true)

    # Compute the F1 score for each class
    f1_score = np.zeros(len(classes), dtype=np.float32)

    # Loop over classes
    for idx, c in enumerate(classes):

        is_true_class_c = labels_true == c
        is_pred_class_c = labels_pred == c

        # Compute binary predictions
        f1_score[idx] = binary_metrics.binary_f1_score(labels_true=is_true_class_c, labels_pred=is_pred_class_c)
    
    return f1_score

def multiclass_micro_f1_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.float32:
    """
    Compute the micro F1 score of the predictions. This computes the F1 score globally by counting the total true positives, false negatives and false positives.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: micro f1 score, computed as the harmonic mean of precision and recall, or equivalently 2 * (total_tp) / (2 * total_tp + total_fp + total_fn).
    """
    classes = np.unique(labels_true)
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Loop over classes
    for c in classes:
            
        is_true_class_c = labels_true == c
        is_pred_class_c = labels_pred == c

        # Compute binary predictions
        tp, _, fp, fn = binary_metrics.binary_true_false_pos_neg(labels_true=is_true_class_c, labels_pred=is_pred_class_c)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute micro F1 score
    micro_f1_score = 2 * total_tp / (2 * total_tp + total_fp + total_fn)

    return np.float32(micro_f1_score)

def multiclass_precision(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute the precision of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: np.ndarray of precisions for each class, computed as the number of true positives divided by the total number of predictions for each class.
    """
    classes = np.unique(labels_true)

    # Compute the precision for each class
    precision = np.zeros(len(classes), dtype=np.float32)

    # Loop over classes
    for idx, c in enumerate(classes):

        is_true_class_c = labels_true == c
        is_pred_class_c = labels_pred == c

        # Compute binary predictions
        precision[idx] = binary_metrics.binary_precision(labels_true=is_true_class_c, labels_pred=is_pred_class_c)
    
    return precision

def multiclass_recall(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute the recall of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        float: np.ndarray of recalls for each class, computed as the number of true positives divided by the total number of true labels for each class.
    """
    classes = np.unique(labels_true)

    # Compute the recall for each class
    recall = np.zeros(len(classes), dtype=np.float32)

    # Loop over classes
    for idx, c in enumerate(classes):

        is_true_class_c = labels_true == c
        is_pred_class_c = labels_pred == c

        # Compute binary predictions
        recall[idx] = binary_metrics.binary_recall(labels_true=is_true_class_c, labels_pred=is_pred_class_c)
    
    return recall

def multiclass_confusion_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """
    Compute the confusion matrix of the predictions.

    Args:
        labels_true (np.ndarray): True labels.
        labels_pred (np.ndarray): Predicted labels.

    Returns:
        np.ndarray: confusion matrix of shape (n_classes, n_classes).
    """
    classes = np.unique(labels_true)
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype=np.int32)

    # Loop over classes
    for idx_pred, pred_class in enumerate(classes):
        for idx_true, true_class in enumerate(classes):

            # Compute those that are true class i and predicted class j
            confusion_matrix[idx_pred, idx_true] = np.sum((labels_true == true_class) & (labels_pred == pred_class))

    return confusion_matrix

def multiclass_ovr_auroc(labels_true: np.ndarray, scores_pred: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Compute one-versus-rest multiclass AUROC. This computes the AUROC for each class against all other classes by treating the class as the positive class and 
    all other classes as the negative class.

    Args:
        labels_true (np.ndarray): True labels.
        scores_pred (np.ndarray): Predicted scores.

    Returns:
        np.ndarray: AUROC for each class.
    """
    sorted_classes = np.sort(np.unique(labels_true))
    assert sorted_classes.size == scores_pred.shape[-1]
    
    # Initialize output
    auroc = np.zeros(len(sorted_classes), dtype=np.float32)

    # Loop over classes
    for idx, c in enumerate(sorted_classes):

        is_true_class_c = labels_true == c
        scores_pred_class_c = scores_pred[:, idx]

        # Compute binary auroc
        auroc[idx] = binary_metrics.binary_auroc(labels_true=is_true_class_c, scores_pred=scores_pred_class_c, *args, **kwargs)
    
    return auroc

def multiclass_ovo_auroc(labels_true: np.ndarray, scores_pred: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Compute the one versus one multiclass AUROC. This computes the AUROC for each pair of classes one against each other by subsetting the data to the samples
    included in this pair of classes.

    Args:
        labels_true (np.ndarray): True labels.
        scores_pred (np.ndarray): Predicted scores.

    Returns:
        np.ndarray: AUROC for each class.
    """
    sorted_classes = np.sort(np.unique(labels_true))
    assert sorted_classes.size == scores_pred.shape[-1]

    # Initialize output of shape (class vs class)
    auroc = np.zeros((len(sorted_classes), len(sorted_classes)), dtype=np.float32)

    # Loop over classes
    for idx_1, c_1 in enumerate(sorted_classes):
        for idx_2, c_2 in enumerate(sorted_classes):

            # Skip if same class
            if idx_1 == idx_2:
                continue

            # Subset data to samples in class 1 or class 2
            is_true_class_1_or_2 = (labels_true == c_1) | (labels_true == c_2)
            scores_pred_1_or_2 = scores_pred[is_true_class_1_or_2, :]
            subset_is_true_class_1 = labels_true[is_true_class_1_or_2]
            subset_scores_pred_1 = scores_pred_1_or_2[:, idx_1]

            # Compute binary auroc
            auroc[idx_1, idx_2] = binary_metrics.binary_auroc(labels_true=subset_is_true_class_1, scores_pred=subset_scores_pred_1, *args, **kwargs)
    
    return auroc

def multiclass_ovr_auprc(labels_true: np.ndarray, scores_pred: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Compute one-versus-rest multiclass AUPRC. This computes the AUPRC for each class against all other classes by treating the class as the positive class and 
    all other classes as the negative class.

    Args:
        labels_true (np.ndarray): True labels.
        scores_pred (np.ndarray): Predicted scores.

    Returns:
        np.ndarray: AUPRC for each class.
    """
    sorted_classes = np.sort(np.unique(labels_true))
    assert sorted_classes.size == scores_pred.shape[-1]
    
    # Initialize output
    auprc = np.zeros(len(sorted_classes), dtype=np.float32)

    # Loop over classes
    for idx, c in enumerate(sorted_classes):

        is_true_class_c = labels_true == c
        scores_pred_class_c = scores_pred[:, idx]

        # Compute binary auprc
        auprc[idx] = binary_metrics.binary_auprc(labels_true=is_true_class_c, scores_pred=scores_pred_class_c, *args, **kwargs)
    
    return auprc

def multiclass_ovo_auprc(labels_true: np.ndarray, scores_pred: np.ndarray, *args, **kwargs) -> np.ndarray:
    """
    Compute the one versus one multiclass AUPRC. This computes the AUPRC for each pair of classes one against each other by subsetting the data to the samples
    included in this pair of classes.

    Args:
        labels_true (np.ndarray): True labels.
        scores_pred (np.ndarray): Predicted scores.

    Returns:
        np.ndarray: AUPRC for each class.
    """
    sorted_classes = np.sort(np.unique(labels_true))
    assert sorted_classes.size == scores_pred.shape[-1]

    # Initialize output of shape (class vs class)
    auprc = np.zeros((len(sorted_classes), len(sorted_classes)), dtype=np.float32)

    # Loop over classes
    for idx_1, c_1 in enumerate(sorted_classes):
        for idx_2, c_2 in enumerate(sorted_classes):

            # Skip if same class
            if idx_1 == idx_2:
                continue

            # Subset data to samples in class 1 or class 2
            is_true_class_1_or_2 = (labels_true == c_1) | (labels_true == c_2)
            scores_pred_1_or_2 = scores_pred[is_true_class_1_or_2, :]
            subset_is_true_class_1_or_2 = labels_true[is_true_class_1_or_2] == idx_1
            subset_scores_pred_1_or_2 = scores_pred_1_or_2[:, idx_1]

            # Compute binary auprc
            auprc[idx_1, idx_2] = binary_metrics.binary_auprc(labels_true=subset_is_true_class_1_or_2, scores_pred=subset_scores_pred_1_or_2, *args, **kwargs)
    
    return auprc


def lachiche_algorithm(labels_true: np.ndarray, scores_pred: np.ndarray) -> np.ndarray:
    """
    Computes the weights for each class that maximizes the F1 score. This is the Lachiche algorithm.

    Args:
        labels_true (np.ndarray): True labels of shape (N, )
        scores_pred (np.ndarray): Predicted scores of shape (N, num_classes)

    Returns:
        np.ndarray: Weights for each class of shape (num_classes, )
    """

    # Unpack variables
    classes, counts = np.unique(labels_true, return_counts=True)
    assert classes.size == scores_pred.shape[-1]

    # Sort values from largest to smallest
    _sort_ids = np.argsort(counts)[::-1]
    sorted_classes = classes[_sort_ids]
    sorted_scores = scores_pred[:, _sort_ids]

    # Initialize output
    weights_ordered = np.ones(len(sorted_classes), dtype=np.float32)


    # Iteratively compute weights for each class based on classes previously seen
    for _sort_idx, c in enumerate(weights_ordered[1:], start=1):
        
        # Get max score for previously seen classes and compute ratio
        seen_classes_max_estimate = np.max(weights_ordered[:_sort_idx] * sorted_scores[:, :_sort_idx], axis=1)
        cur_class_ratio = sorted_scores[:, _sort_idx] / seen_classes_max_estimate

        # Compute weight for current class
        is_class_c = labels_true==c
        weights_ordered[_sort_idx] = binary_metrics.find_best_threshold(labels_true=is_class_c, scores_pred=cur_class_ratio, target=Literal["f1"])


    # Convert weights to original ordering of classes - note that argsorting the ordered classes gives the original ordering
    weights = weights_ordered[np.argsort(sorted_classes)]

    # Return weights
    return weights
