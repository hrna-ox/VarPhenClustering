"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Defines Loss Functions for the various models.
"""

# ============= Import Libraries =============
import torch
import torch.nn as nn


def log_gaussian_lik(x, mu, var):
    """
    Log Likelihood for values x given multivariate normal with mean mu, and variances var.
    
    Params:
    - x: of shape (batch_size, input_size)
    - mu, var: of shape (input_size)

    Outputs:
    single values log likelihood for each sample in batch (batch_size)
    """

    # Get parameters
    _ , inp_size = x.size()

    # Compute exponential term
    exp_term =  0.5 * torch.sum(((x - mu) / var) * (x - mu), dim=-1)   # (batch_size)
    lin_term = torch.sum(torch.log(var), dim=-1, keepdim=False)   # (batch_size)
    cons_term = 0.5 * inp_size * torch.log(2 * torch.acos(torch.zeros(1)) * 2) # constant

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
    dgamma_1, dgamma1_sum = torch.digamma(a1), torch.digamma(a1_sum)

    # Compute indiviual terms of lemma in paper
    term1 = lgamma1_sum - lgamma2_sum
    term2 = torch.sum(lgamma1 - lgamma2, dim=-1, keepdim=False)
    term3 = torch.sum((a1 - a2) * (dgamma_1 - dgamma1_sum), dim=-1, keepdim=False)
    
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
