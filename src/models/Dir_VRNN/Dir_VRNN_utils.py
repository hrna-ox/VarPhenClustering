"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Auxiliary Function definition for Dirichlet Variational Neural Network Definition.
"""

# ============= Import Libraries =============

import torch
import numpy as np

from datetime import datetime
import os


# ============= Auxiliary Functions =============

def sample_normal(means, logvars):
    """
    How to sample from a Normal distribution given a set of cluster means and logvars so that it is back-propagable.

    Inputs:
        - means: vector of shape (K, _dim) corresponding to cluster means
        - logvars: vector of shape (K, ) corresponding to cluster log of variances

    Outputs:
        - samples from normal distributions of shape (K, )
    """
    K, _dim = means.shape

    # Generate samples form standard normal distribution
    eps = torch.randn(size=[K, _dim], device=means.device)

    # Apply normal reparameterisation trick
    samples = means + torch.reshape(torch.exp(0.5 * logvars), [-1, 1]) * eps

    return samples


def sample_dirichlet(alpha):
    """
    How to sample from a dirichlet distribution with parameter alpha so that it is back-propagable.

    Inputs:
        - alpha: vector of shape (bs, K) or (K) with corresponding alpha parameters.

    Outputs:
        - samples from Dirichlet distribution according to alpha parameters. We first sample gammas and then compute
        an approximated sample.
    """

    # Make 2D if alpha is only 1D
    alphas = torch.reshape(alpha, shape=(-1, list(alpha.shape)[-1]))

    # Generate gamma - X_ik is sampled according to gamma (alpha_ik)
    X = _sample_differentiable_gamma_dist(alphas)

    # Compute sum across rows
    X_sum = torch.sum(X, dim=1, keepdim=True)

    # Generate Dirichlet samples by normalising
    dirichlet_samples = torch.divide(X, X_sum + 1e-8)

    return dirichlet_samples


def _sample_differentiable_gamma_dist(alphas):
    """
    Compute a gamma (rate, 1) sample according to vector of shape parameters. We use inverse transform sampling
    so that we can back propagate through generated sample.
    """

    # Generate uniform distribution with the same shape as alphas
    uniform_samples = torch.rand(size=alphas.shape)

    # Approximate with inverse transform sampling
    gamma_samples = torch.pow(
        (
                alphas *
                uniform_samples *
                torch.exp(torch.lgamma(alphas))
        ),
        1 / alphas
    )

    # Output generated samples
    return gamma_samples


def compute_gaussian_log_lik(x, mu, var):
    """
    Compute log likelihood function for inputs x given multivariate normal distribution with
    mean mu and variance var.

    Parameters:
        - x: of shape (batch_size, input_size)
        - mu, var: of shape (input_size)

    Output:
        - log likelihood of gaussian distribution with shape (batch_size)
    """

    # Compute exponential term
    exp_term = 0.5 * torch.sum(
        torch.mul(
            torch.mul(
                x - mu,  # (batch_size, input_size)
                1 / var  # (batch_size, input_size)
            ),  # (batch_size, input_size)
            x - mu
        ),
        dim=-1,
        keepdim=False
    )  # (batch_size)

    # Compute log likelihood
    _input_size = x.size(dim=-1)
    pi = torch.acos(torch.zeros(1)) * 2
    log_lik = - 0.5 * _input_size * torch.log(2 * pi) - 0.5 * torch.prod(var, dim=-1, keepdim=False) - exp_term

    return torch.mean(log_lik)


def compute_dirichlet_kl_div(alpha_1, alpha_2):
    """
    Computes KL divergence between Dirichlet distribution with parameter alpha 1 and dirichlet distribution of
    parameter alpha 2.

    Inputs: alpha_1, alpha_2 array-like of shape (batch_size, K)

    Outputs: array of shape (batch_size) with corresponding KL divergence.
    """

    # Compute gamma functions for alpha_i
    log_gamma_1, log_gamma_2 = torch.lgamma(alpha_1), torch.lgamma(alpha_2)

    # Sum of alphas
    alpha_1_sum, alpha_2_sum = torch.sum(alpha_1, dim=-1, keepdim=True), torch.sum(alpha_2, dim=-1, keepdim=True)

    # Compute gamma of sum of alphas
    log_gamma_1_sum, log_gamma_2_sum = torch.lgamma(alpha_1_sum), torch.lgamma(alpha_2_sum)

    # Compute digamma for each alpha_1 term and alpha_1 sum
    digamma_1, digamma_1_sum = torch.digamma(alpha_1), torch.digamma(alpha_1_sum)

    # Compute terms in Lemma 3.3.
    first_term = log_gamma_1_sum - log_gamma_2_sum
    second_term = torch.sum(log_gamma_1 - log_gamma_2, dim=-1, keepdim=False)
    third_term = torch.sum(torch.mul(
        alpha_1 - alpha_2,
        digamma_1 - digamma_1_sum
    ),
        dim=-1,
        keepdim=False
    )

    # Combine all terms
    kl_div = torch.squeeze(first_term) + second_term + third_term

    return torch.mean(kl_div)


def compute_outcome_loss(y_true, y_pred):
    """
    Compute outcome loss given true outcome y (one-hot encoded) and predicted categorical distribution parameter y_pred.
    This is the log of the categorical distribution loss.

    Parameters:
        - y_true: one-hot encoded of shape (batch_size, num_outcomes)
        - y_pred: of shape (batch_size, num_outcomes)

    Outputs:
        - categorical distribution loss
    """

    # Compute log term
    log_loss = torch.sum(
        y_true * torch.log(y_pred + 1e-8),
        dim=-1,
        keepdim=False
    )

    return torch.mean(log_loss)


def estimate_new_clus(pis, zs):
    """
    Estimate new cluster representations given probability assignments and estimated samples.

    Parameters:
        - pis: probability of cluster assignment (sampled from Dirichlet) of shape (batch_size, K)
        - zs: current estimation for observation representation of shape (batch_size, latent_dim)

    Output:
        - new cluster estimates of shape (K, latent_dim)
    """

    # This is equivalent to matrix multiplication
    new_clus = torch.matmul(torch.transpose(pis, dim0=1, dim1=0), zs)

    return new_clus


def get_exp_run_path(exp_fd):
    """
    Estimate save path for current experiment run, so that it doesn't overwrite previous runs.

    Parameters:
        - exp_fd: Folder directory for saving experiments

    Outputs:
        - save_fd: Folder directory for saving current experiment run.
    """
    # Add separator if it does not exist
    if exp_fd[-1] != "/":
        exp_fd += "/"

    # If no current run exists
    if not os.path.exists(exp_fd):

        # Make Save_fd as run 1
        save_fd = exp_fd + f"run1-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"

        # Make folder and add logs
        os.makedirs(save_fd)
        os.makedirs(save_fd + "logs/")

        # return experiment directory
        return save_fd

    # Else find first run that has not been previously computed - get list of runs and add 1 to max run
    list_run_dirs = [fd.name for fd in os.scandir(exp_fd) if fd.is_dir()]
    list_runs = [int(run.split("-")[0][3:]) for run in list_run_dirs]
    new_run_num = max(list_runs) + 1

    # Save as new run
    save_fd = exp_fd + f"run{new_run_num}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}/"
    assert not os.path.exists(save_fd)

    os.makedirs(save_fd)
    os.makedirs(save_fd + "logs/")

    # Return experiment directory
    return save_fd
