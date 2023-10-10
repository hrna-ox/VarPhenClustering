"""

Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Define functions for logging all metrics jointly. Implements single and temporal version.
"""

# Import required packages
import numpy as np
from typing import Literal

# Utility functions
import src.metrics.binary_metrics as binary
import src.metrics.multiclass_metrics as multiclass
import src.metrics.clustering_metrics as clustering

    
def summary_binary_metrics(labels_true: np.ndarray, scores_pred: np.ndarray, with_lachiche: bool = False) -> dict:
    """
    Compute all binary metrics and save them to a dictionary.

    Args:
        labels_true (np.ndarray): True labels. Expected shape (N, )
        scores_pred (np.ndarray): Predicted scores, must be positive. Expected shape (N, )
        with_lachiche (bool, optional): Whether to compute the metrics from Lachiche et al. Defaults to False.

    Returns:
        dict: Dictionary with all metrics.
    """
    assert len(labels_true.shape) == 1 and len(scores_pred.shape) == 1
    assert labels_true.shape[0] == scores_pred.shape[0]

    labels_pred = np.where(scores_pred >= 0.5, 1, 0)

    # Compute metrics
    accuracy = binary.binary_accuracy(labels_true, labels_pred)
    f1_score = binary.binary_f1_score(labels_true, labels_pred)
    precision = binary.binary_precision(labels_true, labels_pred)
    recall = binary.binary_recall(labels_true, labels_pred)
    auroc = binary.binary_auroc(labels_true, scores_pred)
    auprc = binary.binary_auprc(labels_true, scores_pred)
    confusion_matrix = binary.binary_confusion_matrix(labels_true, labels_pred)
    true_false_pos_neg = binary.binary_true_false_pos_neg(labels_true, labels_pred)

    # Save metrics to dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'f1_score': f1_score,
        'precision': precision,
        'recall': recall,
        'auroc': auroc,
        'auprc': auprc,
        'confusion_matrix': confusion_matrix,
        'true_false_pos_neg': true_false_pos_neg
    }
    lachiche_metrics_dict = {}

    # Check if lachiche algorithm is required to estimate new scores
    if with_lachiche:
        
        # Estimate new scores
        best_threshold = binary.find_best_threshold(labels_true=labels_true, scores_pred=scores_pred, target = Literal["f1"])
        scores_pred_binarized = np.where(scores_pred >= best_threshold, 1, 0)
        lachiche_metrics_dict = summary_binary_metrics(labels_true, scores_pred=scores_pred_binarized, with_lachiche=False)

    return {
        **metrics_dict,
        "lachiche": lachiche_metrics_dict,
        "lachiche_threshold": best_threshold
    }
    
def summary_multiclass_metrics(labels_true: np.ndarray, scores_pred: np.ndarray, with_lachiche: bool = False) -> dict:
    """
    Compute all multiclass metrics and save them to a dictionary.

    Args:
        labels_true (np.ndarray): True labels. Expected shape (N, )
        scores_pred (np.ndarray): Predicted scores, must be positive. Expected shape (N, num_classes)
        with_lachiche (bool, optional): Whether to compute the metrics from Lachiche et al. Defaults to False.
    
    Returns:
        dict: Dictionary with all metrics.
    """
    assert len(labels_true.shape) == 1 and len(scores_pred.shape) == 2
    assert labels_true.shape[0] == scores_pred.shape[0]

    labels_pred = np.argmax(scores_pred, axis=-1)
    
    # Compute metrics
    accuracy = multiclass.multiclass_accuracy(labels_true, labels_pred)
    macro_f1_score = multiclass.multiclass_macro_f1_score(labels_true, labels_pred)
    micro_f1_score = multiclass.multiclass_micro_f1_score(labels_true, labels_pred)
    precision = multiclass.multiclass_precision(labels_true, labels_pred)
    recall = multiclass.multiclass_recall(labels_true, labels_pred)
    ovr_auroc = multiclass.multiclass_ovr_auroc(labels_true, scores_pred)
    ovo_auroc = multiclass.multiclass_ovo_auroc(labels_true, scores_pred)
    ovr_auprc = multiclass.multiclass_ovr_auprc(labels_true, scores_pred)
    ovo_auprc = multiclass.multiclass_ovo_auprc(labels_true, scores_pred)
    confusion_matrix = multiclass.multiclass_confusion_matrix(labels_true, labels_pred)
    true_false_pos_neg = multiclass.multiclass_true_false_pos_neg(labels_true, labels_pred)

    # Save metrics to dictionary
    metrics_dict = {
        'accuracy': accuracy,
        'macro_f1_score': macro_f1_score,
        'micro_f1_score': micro_f1_score,
        'precision': precision,
        'recall': recall,
        'ovr_auroc': ovr_auroc,
        'ovo_auroc': ovo_auroc,
        'ovr_auprc': ovr_auprc,
        'ovo_auprc': ovo_auprc,
        'confusion_matrix': confusion_matrix,
        'true_false_pos_neg': true_false_pos_neg,
    }
    lachiche_metrics_dict = {}

    # Check if lachiche algorithm is required to estimate new scores
    if with_lachiche:

        # Estimate new scores
        weights_lachiche = multiclass.lachiche_algorithm(labels_true=labels_true, scores_pred=scores_pred)
        scores_pred_lachiche = np.dot(scores_pred, weights_lachiche)
        lachiche_metrics_dict = summary_multiclass_metrics(labels_true, scores_pred=scores_pred_lachiche, with_lachiche=False)

    return {
        **metrics_dict,
        "lachiche_metrics": lachiche_metrics_dict,
        "lachiche_weights": weights_lachiche
    }


def summary_clustering_metrics(X: np.ndarray, labels_true: np.ndarray, clus_pred: np.ndarray, **kwargs) -> dict:
    """
    Compute all clustering metrics and save them to a dictionary.
    
    Args:
        X (np.ndarray): Data matrix. Expected shape (N, D)
        labels_true (np.ndarray): True labels. Expected shape (N, )
        clus_pred (np.ndarray): Predicted labels. Expected shape (N, )
        **kwargs: Additional arguments to be passed to the clustering metrics (sil score)
        
    Returns:
        dict: Dictionary with all metrics.
    """

    assert len(X.shape) == 2 and len(labels_true.shape) == 1 and len(clus_pred.shape) == 1
    assert X.shape[0] == labels_true.shape[0] and X.shape[0] == clus_pred.shape[0]

    # Compute metrics
    ars = clustering.cluster_adjusted_rand_score(labels_true, clus_pred)
    nmi = clustering.cluster_normalized_mutual_info_score(labels_true, clus_pred)
    anm = clustering.cluster_adjusted_normalized_mutual_information(labels_true, clus_pred)
    fms = clustering.cluster_fowlkes_mallows_score(labels_true, clus_pred)
    sil = clustering.cluster_silhouette_score(X, clus_pred, **kwargs)
    chs = clustering.cluster_calinski_harabasz_score(X, clus_pred)
    dbi = clustering.cluster_davies_bouldin_score(X, clus_pred)

    # Save metrics to dictionary
    metrics_dict = {
        'adjusted_rand': ars,
        'normalized_mutual_info': nmi,
        "adjusted_normalized_mutual_info": anm,
        'fowlkes_mallows': fms,
        'silhouette': sil,
        'calinski_harabasz': chs,
        'davies_bouldin': dbi,
    }

    return metrics_dict

def print_avg_metrics_paper(metrics_dict: dict):
    """
    Print the average F1 scores, Precision, Recall and AUROC (if available), as well as clustering metrics SIL, DBI and VRI.

    Params:
    - metrics_dict (dict): Dictionary with all metrics.
    """
    output_dic = {}
    metric_rename_dic = {"macro_f1_score": "F1", 
                        "precision": "Precision",
                        "recall": "Recall",
                        "ovr_auroc": "Auroc",
                        "silhouette": "SIL",
                        "davies_bouldin": "DBI",
                        "calinski_harabasz": "VRI"
                    }


    for metric in ["macro_f1_score", "precision", "recall", "ovr_auroc", "silhouette", "davies_bouldin", "calinski_harabasz"]:
        if metric in metrics_dict.keys():
            print(f"{metric}: {np.mean(metrics_dict[metric]):.3f}")
            output_dic[metric_rename_dic[metric]] = np.mean(metrics_dict[metric])

        else:
            print(f"{metric}: N/A")
            output_dic[metric_rename_dic[metric]] = "N/A"

    return output_dic
