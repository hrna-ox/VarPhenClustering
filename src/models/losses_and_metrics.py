"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Defines Loss Functions for the various models.
"""

# ============= Import Libraries =============
import torch
import torch.nn as nn

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

def accuracy_score(y_true, y_pred):
    """
    Compute accuracy score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.

    Returns:
        acc (_type_): scalar accuracy score.
    """

    # Convert to labels
    labels_pred = torch.argmax(y_pred, dim=-1)
    labels_true = torch.argmax(y_true, dim=-1)

    # Compare pairwise and sum across tensor
    correct_pred = torch.sum(labels_pred == labels_true)
    num_preds = labels_pred.size()[0]

    # Compute accuracy
    acc = correct_pred / num_preds

    return acc

def f1_multiclass(y_true, y_pred, mode='macro'):
    """
    Compute Multi-class F1 score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.
        mode (_type_): 'macro' or 'micro' F1 score.

    Returns:
        Tuple of macro, micro F1 multi-class scores.
    """

    try:
        assert mode in ['macro', 'micro']
    except AssertionError:
        raise ValueError("mode must be either 'macro' or 'micro'")
    
    # Get params and initialise output tensor
    _, O = y_true.size()
    f1_scores = torch.zeros(O, device=y_pred.device)
    total_fp, total_fn, total_tp = 0, 0, 0

    # Convert to labels
    labels_pred = torch.argmax(y_pred, dim=-1)
    labels_true = torch.argmax(y_true, dim=-1)

    # Iterate over classes
    for class_idx in range(O):

        # Compute True Positives, False Positives, False Negatives
        TP = torch.sum((labels_pred == class_idx) & (labels_true == class_idx))
        FP = torch.sum((labels_pred == class_idx) & (labels_true != class_idx))
        FN = torch.sum((labels_pred != class_idx) & (labels_true == class_idx))

        # Compute precision and recall
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)

        # Compute F1 score
        f1_scores[class_idx] = 2 * precision * recall / (precision + recall)

        # Add to global trackers
        total_fp += FP
        total_fn += FN
        total_tp += TP

    # Compute macro and micro F1 scores
    macro_f1 = torch.mean(f1_scores)
    micro_f1 = total_tp / (total_tp + 0.5 * (total_fp + total_fn))

    return macro_f1, micro_f1

def recall_score(y_true, y_pred):
    """
    Compute recall score between outcomes y_true, and predicted outcomes y_pred.

    Args:
        y_true (_type_): (N, O) array of one-hot encoded outcomes.
        y_pred (_type_): (N, O) array of outcome probability predictions.

    Returns:
        recall (_type_): scalar recall score.
    """
    
    # Convert to labels
    labels_pred = torch.argmax(y_pred, dim=-1)
    labels_true = torch.argmax(y_true, dim=-1)

    # Compute TP, FP, FN
    TP = torch.sum((labels_pred == labels_true) & (labels_true == 1))
    FN = torch.sum((labels_pred != labels_true) & (labels_true == 1))

    # Compute recall
    recall = TP / (TP + FN)

    return recall

# endregion