"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk
Name: auxiliary_functions.py

Auxiliary functions for Dirichlet-VRNN Model. Includes model inner calculations, phenotype computation and other 
intermediate object analysis.
"""

# region =============== IMPORT LIBRARIES ===============
import torch
from torch.nn.functional import one_hot

from tsnecuda import TSNE

# Visualization tools
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

# endregion


# region ========== Functions related to sampling and generation of samples ==========

eps = 1e-6  # small constant to avoid numerical issues

def sample_dir(alpha):
    """
    Approximation to Dirichlet sampling given parameter alpha, that allows for back-propagation.
    First, we sample gamma variables based on inverse transform sampling. Then we combine them
    to generate a dirichlet vector.

    Inputs:
        - alpha: vector of shape (bs, K) with corresponding alpha parameters.

    Outputs:
        - generated samples of size (bs).
    """

    # Use inverse transform sampling from uniform samples to generate gamma samples
    u_samples = torch.rand(size=alpha.shape, device=alpha.device)
    gamma_samples = (alpha * u_samples * torch.exp(torch.lgamma(alpha))) ** (1 / alpha)

    # Divide gamma samples across rows to normalize (which gives a dirichlet sample)
    row_sum = torch.sum(gamma_samples, dim=1, keepdim=True)
    dir_samples = gamma_samples / (row_sum + eps)

    return dir_samples


def gen_samples_from_assign(pi_assign, c_means, log_c_vars):
    """
    Generate samples from latent variable cluster assignment. We re-parameterize normal sampling
    in order to ensure back-propagation.

    Args:
        pi_assign: tensor of shape (bs, K) with probabilities of cluster assignment. Rows sum to 1.
        c_means: tensor of shape (K, l_dim) with cluster mean vectors.
        log_c_vars: tensor of shape (K, ) with cluster log of variance parameters.

    Outputs:
        tensor of shape (bs, l_dim) with generated samples.
    """
    # Get parameters from vectors
    K, l_dim = c_means.shape
    
    # Generate standard normal samples and apply transformation to obtain multivariate normal
    stn_samples = torch.randn(size=[K, l_dim], device=c_means.device)
    mvn_samples = c_means + torch.exp(0.5 * log_c_vars).reshape(-1, 1) * stn_samples

    # Combine multivariate normal samples with cluster assignment probabilities
    samples = torch.matmul(pi_assign, mvn_samples)

    return samples

def gen_diagonal_mvn(mu_g: torch.Tensor, log_var_g: torch.Tensor) -> torch.Tensor:
    """
    Generate samples from multivariate normal distribution with diagonal covariance matrix. 

    Params:
    - mu_g: tensor of shape (N, dim) with the mean vector of the distribution for each sample.
    - log_var_g: tensor of shape (N, dim) with the log of the variance vector of the distribution for each sample.

    Outputs:
    - tensor of shape (N, dim) with generated samples.
    """

    # Get parameters
    N, T, dim = mu_g.shape

    # Generate standard normal samples and apply transformation to obtain multivariate normal
    stn_samples = torch.randn(size=[N, T, dim], device=mu_g.device)
    mvn_samples = mu_g + torch.exp(0.5 * log_var_g) * stn_samples

    return mvn_samples
# endregion


# region ========= Functions related to cluster properties =========
def torch_get_clus_memb_dist(pis_assign: torch.Tensor):
    """
    Given probabilities of cluster assignment, compute the estimated distribution of cluster assignments, using the same ordering.

    Params:
    - pis_assign: torch.Tensor object of shape (bs, K), where rows sum to 1 indicating cluster assignment probability.

    Returns:
    - clus_assign_dist: torch.Tensor of shape (K) with the number of patients assigned to each cluster using a 'most likely assignment' approach.
    """

    # Compute most likely cluster assignment
    clus_assign = torch.argmax(pis_assign, dim=1)

    # Initialize output and iterate through each cluster
    K = pis_assign.shape[1]
    clus_assign_dist = torch.zeros(K)

    for k in range(K):

        # Get number of patients assigned to cluster k
        clus_assign_dist[k] = torch.sum(clus_assign == k)

    return clus_assign_dist


def torch_get_temp_clus_memb_dist(temp_pis_assign: torch.Tensor):
    """
    Given probabilities of cluster assignment, compute the estimated distribution of cluster assignments, using the same ordering. This is 
    a temporal variant of the function 'torch_get_clus_memb_dist'.

    Params:
    - temp_pis_assign: torch.Tensor object of shape (bs, T, K), where the last dimension sums to 1 indicating cluster assignment probability.

    Returns:
    - clus_assign_dist: torch.Tensor of shape (T, K) with the number of patients assigned to each cluster using a 'most likely assignment' approach.
    """

    # Initialize output
    _, T, K = temp_pis_assign.shape
    clus_assign_dist = torch.zeros(T, K)

    # Iterate through each time step
    for t in range(T):

        # Get cluster assignment distribution at time step t
        clus_assign_dist[t, :] = torch_get_clus_memb_dist(temp_pis_assign[:, t, :])

    return clus_assign_dist

def torch_clus_means_separability(clus_means: torch.Tensor) -> torch.Tensor:
    """Compute latent space separability between cluster means.

    Args:
        clus_means (torch.Tensor): of shape (K, dim) with cluster means.

    Returns:
        torch.Tensor: with average L2 distance between cluster means.
    """

    # Get params
    K, dim = clus_means.shape

    # Compute L2 distance between cluster means
    pairwise_diff = clus_means.reshape(K, 1, dim) - clus_means.reshape(1, K, dim)
    pairwise_dist = torch.square(pairwise_diff).sum(dim=2)

    # Get average across cluster pairs
    avg_dist = pairwise_dist.sum() / (K * (K - 1))

    return avg_dist

def torch_clus_mean_2D_tsneproj(clus_means: torch.Tensor, seed: int) -> torch.Tensor:
    """Get 2D TSNE plot of cluster means.

    Args:
        clus_means (torch.Tensor): of shape (K, dim) with cluster means.
        seed (int): random seed for reproducibility.

    Outputs:
        clus_reps: of shape (K, 2) with 2D TSNE plot of cluster means.
    """ 

    # Get TSNE object
    tsne = TSNE(n_components=2, random_seed=seed, verbose=0)

    # Apply tsne transform
    clus_reps = tsne.fit_transform(clus_means)

    # Return figure
    return torch.Tensor(clus_reps)

# endregion
# region ================== Functions related to phenotype computation ==================
def torch_phens_from_prob(pis_assign: torch.Tensor, y_true: torch.Tensor):
    """
    Computes the phenotype for each cluster given cluster assignment probabilities, and the true labels. The contribution of each patient is equivalent 
    to the probability of the patient belonging to the cluster. The phenotype is the sum of the contributions of each patient. 

    Args:
        pis_assign (torch.Tensor): of shape (N, K) with cluster assignment probabilities. Sums to 1 over dimension 1.
        y_true (torch.Tensor): of shape (N, O) with one-hot encoded true labels.

    Returns:
        torch.Tensor: of shape (K, O) with phenotype for each cluster. The sum over dimension 1 is 1.
    """

    # Compute number of patients per cluster with a given outcome through matrix multiplication
    unnorm_phens = torch.matmul(
        torch.transpose(pis_assign, 0, 1),
        y_true
    )

    # Normalize phenotypes per cluster
    phens = unnorm_phens / (torch.sum(unnorm_phens, dim=1, keepdim=True) + eps)

    return phens

def torch_phens_from_clus(pis_assign: torch.Tensor, y_true: torch.Tensor):
    """
    Computes the phenotype for each cluster given cluster assignment probabilities, and the true labels. Each patient contributes only
    to the most likely cluster it is assigned to (e.g. if pi[0] = [0.2, 0.6, 0.2]), then outcome[0] contributes only to Cluster 2 phenotype with a weight of 1. 

    Args:
        pis_assign (torch.Tensor): of shape (N, K) with cluster assignment probabilities. Sums to 1 over dimension 1.
        y_true (torch.Tensor): of shape (N, O) with one-hot encoded true labels.

    Returns:
        torch.Tensor: of shape (K, O) with phenotype for each cluster. The sum over dimension 1 is 1.
    """

    # Get params
    K = pis_assign.size(1)

    # Assign patients to most likely cluster and convert to one-hot encoded vector.
    clus_assign = one_hot(torch.argmax(pis_assign, dim=-1), num_classes=K).float()  # (N, K) of indicator vectors

    # Apply previous function to estimate phenotypes
    phens = torch_phens_from_prob(clus_assign, y_true)

    return phens

def torch_get_temp_phens(pis_assign: torch.Tensor, y_true: torch.Tensor, mode: str = "prob"):
    """
    Compute the estimated phenotype for each cluster over time. This function implements the two different ways of computing the phenotype:
    a) patient contribution is weighted by the probability of belonging to the cluster (mode = "prob")
    b) patient contribution is only valid for the cluster it is assigned to (mode = "one-hot")

    Args:
        pis_assign (torch.Tensor): of shape (N, T, K) with cluster assignment probabilities. Sums to 1 over dimension 2. 
        y_true (torch.Tensor): of shape (N, O) with one-hot encoded true outcome labels.
        mode (str, optional, default="prob"): mode of computing the phenotype. Either "prob" (implements weighted pat contribution), or
            "one-hot" (implements one-hot pat contribution).
    """

    # Check parameter is correct
    try:
        assert mode in ["prob", "one-hot"]
    except AssertionError:
        raise ValueError("mode must be either 'prob' or 'one-hot'.")
    
    # Print message
    print(f"Computing phenotypes using mode '{mode}'.")

    # Get params
    N, T, K = pis_assign.shape
    _, O = y_true.shape

    # Initialise output 
    phens = torch.zeros((K, T, O), device=pis_assign.device)

    # Iterate over time
    for t in range(T):

        # Update using the appropriate function
        if mode == "prob":
            phens[:, t, :] = torch_phens_from_prob(pis_assign[:, t, :], y_true)
        elif mode == "one-hot":
            phens[:, t, :] = torch_phens_from_clus(pis_assign[:, t, :], y_true)

    return phens

# endregion

# region ====================  FUNCTIONS FOR PLOTTING ===============================

def plot_clus_memb_evol(temp_clus_memb: torch.Tensor):
    """
    Given cluster membership assignments, visualize cluster membership evolution over time.

    Params:
    - clus_assign_dist: torch.Tensor of shape (T, K) with the number of patients assigned to each cluster using a 'most likely assignment' approach.
    """

    # Get dimensions
    T, K = temp_clus_memb.shape

    # Initialize figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = cm.get_cmap("tab10")
    colors = cmap.colors # type: ignore

    # Iterate through each cluster
    for k in range(K):
        
        # Plot cluster membership evolution
        ax.plot(range(1, T + 1), temp_clus_memb[:, k], label=f"Cluster {k}", linewidth=2, linestyle='--', color=colors[k])

    # Decorate axes
    ax.set_xticklabels(range(1, T+1))
    ax.set_xlabel("Time")
    ax.set_ylabel("Num")

    ax.set_title("Plot of cluster membership evolution over time")

    return fig, ax


def torch_plot_clus_prob_assign_time(temp_pis_assign: torch.Tensor):
    """
    Make Box plots of cluster assignment probabilities over time.

    Params:
    - temp_pis_assign: torch.Tensor object of shape (bs, T, K), where the last dimension sums to 1 indicating cluster assignment probability.
    
    Returns:
    - fig, ax: matplotlib figure and axis objects. For each plot, we plot the temporal evolution of cluster assignment probabilities.
    """

    # Get params and base information
    _, T, K = temp_pis_assign.shape
    clus_memb_num = torch_get_temp_clus_memb_dist(temp_pis_assign=temp_pis_assign)

    # Initialize figure and axis objects
    nrows, ncols = torch.ceil(K / torch.Tensor(2)).int(), 2
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,5))
    axs = ax.flatten()

    # Iterate over clusters
    for clus_idx in range(K):

        # Iterate over time
        for t in range(T):

            sns.boxplot(temp_pis_assign[:, :, clus_idx], orient="v",
                        ax=axs[clus_idx])

        axs[clus_idx].set_xlabel("Time")
        axs[clus_idx].set_ylabel("Prob")
        axs[clus_idx].set_title(f"Cluster {clus_idx}")
        
    return fig, axs

# endregion
