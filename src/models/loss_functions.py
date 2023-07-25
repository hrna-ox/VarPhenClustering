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

def torch_log_Gauss_likelihood(x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, device=None) -> torch.Tensor:
    """
    Compute the Log Likelihood of a Gaussian distribution with mean parameter mu and log variance var, given values x. We assume diagonal 
    convariance matrix with var given by torch.exp(logvar) parameter.

    Args:
        x (torch.Tensor): input values x of shape (batch_size, input_size)
        mu (torch.Tensor): mean values of gaussian distribution for each input of shape (batch_size, input_size)
        logvar (torch.Tensor): log of variance values of gaussian distribution for each input of shape (batch_size, input_size)
        device (torch.device): defaults to None.

    Returns:
        torch.Tensor: with average log likelihood value across batch.
    """

    # Check shape of inputs
    assert x.shape == mu.shape == logvar.shape, "Inputs have to have same shape."
    _, input_dims = x.shape

    # Compute individual terms
    log_const = - 0.5 * input_dims * torch.log(2 * torch.acos(torch.zeros(1)) * 2).to(device=device)    
    log_det = - 0.5 * torch.sum(logvar, dim=-1, keepdim=False)
    log_exp = - 0.5 * torch.sum(((x - mu) / (torch.exp(logvar) + EPS)) * (x - mu), dim=-1, keepdim=False)

    # Compute log likelihood
    log_lik = log_const + log_det + log_exp           # Shape (batch_size)

    return torch.mean(log_lik, dim=0) 


def torch_dir_kl_div(a1: torch.Tensor, a2: torch.Tensor) -> torch.Tensor:
    """
    Computes KL divergence of dirichlet distributions with parameters a1 and a2

    Inputs: a1, a2 array-like of shape (batch_size, K)

    Outputs: KL divergence value averaged across batch.
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

    return torch.mean(kl_div, dim=0)


def torch_CatCE(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Compute categorical cross-entropy between y_true and y_pred. Assumes y_true is one-hot encoded.

    Parameters:
        - y_true: one-hot encoded of shape (batch_size, num_outcomes)
        - y_pred: of shape (batch_size, num_outcomes)

    Outputs:
        - categorical distribution loss averaged over batch
    """
    assert torch.all(torch.greater_equal(y_pred, 0)), "y_pred has to be non-negative."

    cat_ce = - torch.sum(y_true * torch.log(y_pred + EPS), dim=-1)         # Shape (batch_size)

    return torch.mean(cat_ce, dim=0)
