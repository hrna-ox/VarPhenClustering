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

import numpy as np
import sklearn.metrics as metrics

EPS = 1e-8


# ============= Define Loss Functions =============

def torch_log_GAUSS(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor, device=None) -> torch.Tensor:
    """
    Compute the Log Likelihood of a Gaussian distribution with parameters mu and var, given values x. We assume diagonal 
    convariance matrix with var given by var parameter.

    Args:
        x (torch.Tensor): input values x of shape (batch_size, input_size)
        mu (torch.Tensor): mean values of gaussian distribution for each input of shape (batch_size, input_size)
        var (torch.Tensor): variance values of gaussian distribution for each input of shape (batch_size, input_size)
        device (torch.device): defaults to None.

    Returns:
        torch.Tensor: with log likelihood values for each input of shape (batch_size)
    """

    # Check shape of inputs
    assert x.shape == mu.shape == var.shape, "Inputs have to have same shape."
    batch_size, input_dims = x.shape

    # Compute individual terms
    log_const = - 0.5 * input_dims * torch.log(2 * torch.acos(torch.zeros(1)) * 2).to(device=device)    
    log_det = - 0.5 * torch.sum(torch.log(var), dim=-1, keepdim=False)
    log_exp = - 0.5 * torch.sum(((x - mu) / var) * (x - mu), dim=-1, keepdim=False)

    # Compute log likelihood
    log_lik = log_const + log_det + log_exp           # Shape (batch_size)

    return log_lik 


def torch_dir_kl_div(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
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
    kl_div = torch.squeeze(term1) + term2 + term3                # Shape (batch_size)

    return kl_div


def torch_CatCE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute categorical cross-entropy between y_true and y_pred. Assumes y_true is one-hot encoded.

    Parameters:
        - y_true: one-hot encoded of shape (batch_size, num_outcomes)
        - y_pred: of shape (batch_size, num_outcomes)

    Outputs:
        - categorical distribution loss for each sample in batch (batch_size)
    """

    cat_ce = - torch.sum(y_true * torch.log(y_pred + EPS), dim=-1)         # Shape (batch_size)

    return cat_ce






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
