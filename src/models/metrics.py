"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Define Metric computations for the various models
"""

# ============= Import Libraries =============
from typing import Union, List, Tuple

import torch
from torchmetrics import AveragePrecision

import numpy as np
import sklearn.metrics as metrics

eps = 1e-8


def _convert_to_labels(*args):
    """
    Iteratively convert predictions/one-hot encoded outcomes to labels.
    """

    for arg in args:

        if arg.ndim == 1:           # Argument already is converted to labels
            yield arg.astype(int)
        elif arg.ndim >= 2:
            yield np.argmax(arg, axis=-1).astype(int)


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
    labels_pred, labels_true = _convert_to_labels(y_pred, y_true)

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
    labels_pred, labels_true = _convert_to_labels(y_pred, y_true)

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


def precision_multiclass(y_true: np.ndarray, y_pred: np.ndarray) -> List:
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


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> List:
    """
    Compute Macro Multi-Class F1 score between true outcomes y_true, and predicted values y_pred.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - y_pred: (N, O) array of outcome probability predictions.

    Returns:
    - f1_score: macro F1 score (F1 score per class)
    """

    # Convert to labels
    labels_pred, labels_true = _convert_to_labels(y_pred, y_true)

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


def micro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Union[np.floating, float]:
    """
    Compute Micro Multi-Class F1 score between true outcomes y_true, and predicted values y_pred.

    Params:
    - y_true: (N, O) array of one-hot encoded outcomes.
    - y_pred: (N, O) array of outcome probability predictions.

    Returns:
    - f1_score: scalar F1 score.
    """

    # Convert to labels
    labels_pred, labels_true = _convert_to_labels(y_pred, y_true)

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


def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


def get_torch_pr_auc_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
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


def get_roc_curve(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = []):
    """
    Compute ROC curve between outcomes y_true, and predicted outcomes y_pred.s

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

    # Close all open figures
    plt.close()

    return fig, ax


def get_torch_pr_curve(y_true: torch.Tensor, y_pred: torch.Tensor, class_names: List[str] = []):
    """
    Compute PR curve between outcomes y_true, and predicted outcomes y_pred.

    Returns:
    - fig, ax pair of plt objects with the roc curves. Includes roc curve per each class.
    """

    # If class_names are empty then generate some placeholder names
    num_classes = y_true.shape[1]
    if class_names == []:
        class_names = [f"Class {i}" for i in range(num_classes)]
    pr_scores = get_torch_pr_auc_score(y_true, y_pred).detach().numpy()

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
    # Close all open figures
    plt.close()

    return fig, ax
