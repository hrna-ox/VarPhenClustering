"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Define Metric computations given true outcomes and predicted outcomes, and data and predicted cluster assignments.
"""

# ============= Import Libraries =============
from typing import Dict, Union, List, Tuple

import torch
from torchmetrics import AveragePrecision

import numpy as np
import sklearn.metrics as metrics

import utils_general as utils

eps = 1e-8


def _convert_to_labels_if_score(*args):
    """
    Iteratively convert predictions/one-hot encoded outcomes to labels.
    """

    for arg in args:

        if arg.ndim == 1:           # Argument already is converted to labels
            yield arg.astype(int)
        elif arg.ndim >= 2:
            yield np.argmax(arg, axis=-1).astype(int)


def _convert_prob_to_score(*args):
    """
    Iteratively convert probability predictions to score values using the x / 1 - x conversion.
    """

    for arg in args:
        if isinstance(arg, np.ndarray):
            yield np.divide(arg, 1 - arg + eps)

        elif isinstance(arg, torch.Tensor):
            yield torch.divide(arg, 1 - arg + eps)


# =========================== Supervised Metrics ===========================

# region
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.

    Returns:
        acc (_type_): scalar accuracy score.
    """

    # Convert to labels
    labels_pred, labels_true = _convert_to_labels_if_score(y_pred, y_true)

    # Compare pairwise and sum across tensor
    correct_pred = np.sum(labels_pred == labels_true)
    num_preds = np.size(labels_pred)

    # Compute accuracy
    acc = correct_pred / num_preds

    return acc


def get_true_false_pos_neg(label_true: np.ndarray, label_pred: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN between predicted true label label_true, and predicted label label_pred.
    
    Args:
        label_true (_type_): (N, ) array of true labels in [0, 1]
        label_pred (_type_): (N, ) array of predicted labels in [0,1]

    Returns:
        TP, FP, FN, TN (_type_): scalar recall score.
    """
    
    # Compute TP, FP, FN
    TP = int(np.sum((label_pred == label_true) & (label_true == 1)))
    FP = int(np.sum((label_pred != label_true) & (label_true == 0)))
    FN = int(np.sum((label_pred != label_true) & (label_true == 1)))
    TN = int(np.sum((label_pred == label_true) & (label_true == 0)))

    return TP, FP, FN, TN


def recall_binary(label_true: np.ndarray, label_pred: np.ndarray) -> float:
    """
    Compute Recall score between predicted true label label_true, and predicted label label_pred.
    
    Args:
        label_true (_type_): (N, ) array of true labels in [0, 1]
        label_pred (_type_): (N, ) array of predicted labels in [0,1]

    Returns:
        recall (_type_): scalar recall score.
    """
    
    # Compute TP, FP, FN
    TP, _, FN, _ = get_true_false_pos_neg(label_true, label_pred)

    # Compute recall
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / (TP + FN)

    return recall


def recall_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """
    Compute recall score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.

    Returns:
        recall (_type_): scalar recall score.
    """

    # Convert to labels
    labels_pred, labels_true = _convert_to_labels_if_score(y_pred, y_true)

    # Initialize tracker of recall scores
    recall_scores = []

    # Loop through each class
    for class_idx in range(y_true.shape[1]):

        # make estimates for each class
        class_pred = labels_pred == class_idx
        class_true = labels_true == class_idx

        # Get score
        recall = recall_binary(class_true, class_pred)
        recall_scores.append(recall)

    return recall_scores


def precision_binary(label_true: np.ndarray, label_pred: np.ndarray) -> float:
    """
    Compute Precision score between predicted true label label_true, and predicted label label_pred.
    
    Args:
        label_true (_type_): (N, ) array of true labels in [0, 1]
        label_pred (_type_): (N, ) array of predicted labels in [0,1]

    Returns:
        precision (_type_): scalar precision score.
    """
    
    # Get TP, FP, FN
    TP, FP, _, _ = get_true_false_pos_neg(label_true, label_pred)

    # Compute precision
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    return precision


def precision_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
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

    # Initialize tracker of precision scores
    precision_scores = []

    # Loop through each class
    for class_idx in range(y_true.shape[1]):

        # make estimates for each class
        class_pred = labels_pred == class_idx
        class_true = labels_true == class_idx

        # Get score
        precision = precision_binary(class_true, class_pred)
        precision_scores.append(precision)

    return precision_scores


def macro_f1_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> List[float]:
    """
    Compute Macro Multi-Class F1 score between true outcomes y_true, and predicted values y_pred.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - y_pred: (N, O) array of outcome probability predictions.

    Returns:
    - f1_score: macro F1 score (F1 score per class)
    """

    # Convert to labels
    labels_pred, labels_true = _convert_to_labels_if_score(y_pred, y_true)

    # Initialize tracker of f1 score
    f1_scores = []

    # Loop through each class
    for class_idx in range(y_true.shape[1]):

        # Compute prediction for the class
        class_pred = labels_pred == class_idx
        class_true = labels_true == class_idx

        # Get Recall and Precision
        recall = recall_binary(class_true, class_pred)
        precision = precision_binary(class_true, class_pred)

        # Compute F1 score and append
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * np.divide(precision * recall, precision + recall)
        
        f1_scores.append(f1)


    return f1_scores


def micro_f1_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> Union[np.floating, float]:
    """
    Compute Micro Multi-Class F1 score between true outcomes y_true, and predicted values y_pred.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - y_pred: (N, O) array of outcome probability predictions.

    Returns:
    - f1_score: scalar F1 score.
    """

    # Convert to labels
    labels_pred, labels_true = _convert_to_labels_if_score(y_pred, y_true)

    # Initialize trackers
    total_tp, total_fp, total_fn = 0, 0, 0

    # Loop through each class
    for class_idx in range(y_true.shape[1]):

        # Get class specific predictions
        class_pred = labels_pred == class_idx
        class_true = labels_true == class_idx

        # Compute True Positives, False Positives and False Negatives
        tp, fp, fn, _ = get_true_false_pos_neg(class_true, class_pred)

        # Add to global values
        total_tp += tp
        total_fp += fp
        total_fn += fn

    # Compute f1 score - alternative formula
    f1 = 2 * np.divide(total_tp, 2 * total_tp + total_fp + total_fn)

    return f1


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> Union[List, np.floating, float]:
    """
    Compute ROC AUC score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (np.ndarray): (N, O) array of one-hot encoded outcomes.
        y_pred (np.ndarray): (N, O) array of outcome probability predictions.
        average (str): average parameter for sklearn.metrics.roc_auc_score. Defaults to "macro".

    Returns:
        roc_auc: list of ROC AUC scores per class.
    """
    
    # Convert to labels
    labels_true = np.argmax(y_true, axis=-1).astype(int)

    # Compute Roc per class
    ovr_roc_scores = metrics.roc_auc_score(labels_true, y_score=y_pred, average="macro",
                            multi_class="ovr")
    
    return ovr_roc_scores


def get_torch_pr_auc_score(y_true: torch.Tensor, y_pred: torch.Tensor) -> np.ndarray:
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
    pr_scores = _metric_wrapper(y_pred, labels_true).detach().numpy()

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
        y_true_torch, y_pred_torch = utils.convert_to_torch(y_true, y_pred)

    else:
        raise ValueError("y_true and y_pred must both be either np.ndarray or torch.Tensor")


    # Compute scores for confusion matrix
    acc = accuracy(y_true_npy, y_pred_npy)
    recall = recall_multiclass(y_true_npy, y_pred_npy)
    precision = precision_multiclass(y_true_npy, y_pred_npy)
    macro_f1 = macro_f1_multiclass(y_true_npy, y_pred_npy)
    micro_f1 = micro_f1_multiclass(y_true_npy, y_pred_npy)

    # Compute Roc and PR AUC scores
    roc_auc = auroc(y_true_npy, y_pred_npy)
    pr_auc = get_torch_pr_auc_score(y_true_torch, y_pred_torch)

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
# endregion 

# =========================== Clustering Performance Metrics ===========================

# region

def adjusted_rand_score(y_true: np.ndarray, clus_pred: np.ndarray) -> Union[np.floating, float]:
    """
    Compute single adjusted Rand Score Clustering Performance metric given the true labels and the predicted cluster assignments for a single time step.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - clus_pred: (N, K) array of cluster probability predictions.
    """

    # Compute labels given extended arrays
    labels_clus = np.argmax(clus_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    return metrics.adjusted_rand_score(labels_true, labels_clus)

def normalized_mutual_info_score(y_true: np.ndarray, clus_pred: np.ndarray) -> Union[np.floating, float]:
    """
    Compute the normalized mutual information score Clustering Performance metric given the true labels and the predicted cluster assignments for a single time step.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - clus_pred: (N, K) array of cluster probability predictions.
    
    Returns:
    - nmi (float): normalized mutual information score.
    """

    # Compute labels given extended arrays
    labels_clus = np.argmax(clus_pred, axis=-1)
    labels_true = np.argmax(y_true, axis=-1)

    return metrics.normalized_mutual_info_score(labels_true, labels_clus)

def silhouette_score(X: np.ndarray, clus_pred: np.ndarray, seed: int = 0, **kwargs):
    """
    Compute Silhouette Score for the predicted clustering given data X for a single time step.

    Params:
    - X: (N, D) array of data.
    - clus_pred: (N, K) array of cluster probability predictions.

    Returns:
    - sil (float): silhouette score.
    """

    # Compute labels given extended arrays
    labels_clus = np.argmax(clus_pred, axis=-1)

    return metrics.silhouette_score(X, labels_clus, metric="euclidean", random_state=seed, **kwargs)      # Between -1 and 1, closer to 1 the better

def davies_bouldin_score(X: np.ndarray, clus_pred: np.ndarray):
    """
    Compute Davies-Bouldin Index for the predicted clustering given data X for a single time step.

    Params:
    - X: (N, D) array of data.
    - clus_pred: (N, K) array of cluster probability predictions.

    Returns:
    - dbi (float): Davies-Bouldin Index score.
    """

    # Compute labels given extended arrays
    labels_clus = np.argmax(clus_pred, axis=-1)

    return metrics.davies_bouldin_score(X, labels_clus)              # The higher the better

def calinski_harabasz_score(X: np.ndarray, clus_pred: np.ndarray):
    """
    Compute Calinski-Harabasz Index for the predicted clustering given data X for a single time step.

    Params:
    - X: (N, D) array of data.
    - clus_pred: (N, K) array of cluster probability predictions.

    Returns:
    - vri (float): Calinski-Harabasz Index score.
    """

    # Compute labels given extended arrays
    labels_clus = np.argmax(clus_pred, axis=-1)

    return metrics.calinski_harabasz_score(X, labels_clus)   # >= 0, the lower the better

def get_clus_label_match_scores(y_true: Union[np.ndarray, torch.Tensor], clus_pred: Union[np.ndarray, torch.Tensor]) -> Dict:
    """
    Compute the Adjusted Rand score and Normalized Mutual Information Score between the true labels and the predicted cluster assignments.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - clus_pred: (N, T, K) array of cluster probability predictions over time

    Returns:
    - dict of scores for each time step.
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

    # Initialize outputs
    rand_scores, nmi_scores = [], []

    # Iterate over time
    for t in range(clus_pred.shape[1]):
        rand = adjusted_rand_score(y_true_npy, clus_pred_npy[:, t])                  # Closer to 1 the better
        nmi = normalized_mutual_info_score(y_true_npy, clus_pred_npy[:, t])   # Between 0 and 1, closer to 1 the better

        # Add to list
        rand_scores.append(rand)
        nmi_scores.append(nmi)

    return {"rand": rand_scores, "nmi": nmi_scores}

def get_unsup_scores(X: np.ndarray, clus_pred: np.ndarray, seed: int = 0, **kwargs) -> Dict:
    """
    Compute the Silhouette Score, Davies-Bouldin Index, and the Variance Ratio Criterion between the data and the predicted cluster assignments.

    Params:
    - X: (N, T, D) array of data.
    - clus_pred: (N, T, K) array of cluster probability predictions over time

    Returns:
    - dict of scores for each time step.
    """

    # If no time step, then edit to 1.
    if len(clus_pred.shape) == 2:
        clus_pred = clus_pred[:, None, :]

    # Get information
    N, T, K = clus_pred.shape

    # Initialize outputs
    sil_scores, vri_scores, dbi_scores = [], [], []

    # Iterate over time
    for t in range(T):
            
        # Reshape input data into right format
        X_t = X[:, t, :]

        # Compute metrics
        sil = silhouette_score(X_t, clus_pred[:, t], seed=seed, **kwargs)      # Between -1 and 1, closer to 1 the better
        dbi = davies_bouldin_score(X_t, clus_pred[:, t])              # The higher the better
        vri = calinski_harabasz_score(X_t, clus_pred[:, t])   # >= 0, the lower the better

        # Add to list
        sil_scores.append(sil)
        dbi_scores.append(dbi)
        vri_scores.append(vri)

    return {"sil": sil_scores, "dbi": dbi_scores, "vri": vri_scores}


# endregion

# def get_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = []):
#     """
#     Compute ROC curve between outcomes y_true, and predicted outcomes y_pred.s

#     Returns:
#     - fig, ax pair of plt objects with the roc curves. Includes roc curve per each class.
#     """

#     # If class_names are empty then generate some placeholder names
#     num_classes = y_true.shape[1]
#     if class_names == []:
#         class_names = [f"Class {i}" for i in range(num_classes)]
#     roc_scores = get_roc_auc_score(y_true, y_pred)

#     # Initialize figure
#     fig, ax = plt.subplots(figsize=(10, 10))
#     colors = get_cmap("tab10").colors # type: ignore

#     # Iterate through each class
#     for class_idx in range(num_classes):

#         # Class true and predicted values
#         y_true_class = y_true[:, class_idx]
#         y_pred_class = y_pred[:, class_idx]

#         # roc, auc values for this class
#         auc = roc_scores[class_idx]# type: ignore

#         # Compute ROC Curve
#         metrics.RocCurveDisplay.from_predictions(
#             y_true=y_true_class, y_pred=y_pred_class, 
#             color=colors[class_idx], plot_chance_level=(class_idx==0),
#             ax=ax, name=f"Class {class_idx} (AUC {auc:.2f}))", 
#         )
        
#     ax.set_title("ROC Curves")
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate")

#     fig.legend()

#     # Close all open figures
#     plt.close()

#     return fig, ax


# def get_torch_pr_curve(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: List[str] = []):
#     """
#     Compute PR curve between outcomes y_true, and predicted outcomes y_pred.

#     Returns:
#     - fig, ax pair of plt objects with the roc curves. Includes roc curve per each class.
#     """

#     # If class_names are empty then generate some placeholder names
#     num_classes = y_true.shape[1]
#     if class_names == []:
#         class_names = [f"Class {i}" for i in range(num_classes)]
#     pr_scores = get_torch_pr_auc_score(y_true, y_pred).detach().numpy()

#     # Initialize figure 
#     fig, ax = plt.subplots(figsize=(10, 10))
#     colors = get_cmap("tab10").colors # type: ignore

#     # Iterate through each class
#     for class_idx in range(num_classes):

#         # Class true and predicted values
#         y_true_class = y_true[:, class_idx]
#         y_pred_class = y_pred[:, class_idx]

#         # roc, auc values for this class
#         prc = pr_scores[class_idx] # type: ignore

#         # Compute ROC Curve
#         metrics.PrecisionRecallDisplay.from_predictions(
#             y_true=y_true_class, y_pred=y_pred_class, 
#             color=colors[class_idx], plot_chance_level=True,
#             ax=ax, name=f"Class {class_idx} (PR {prc:.2f}))", 
#         )
        
#     ax.set_title("PR Curves")
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision")

#     fig.legend()
#     # Close all open figures
#     plt.close()

#     return fig, ax
