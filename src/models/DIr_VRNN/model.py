"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Main model file for proposed model for Dirichlet Variational Recurrent Neural Network method.
Main model is represented as a class, and then a wrapper class is added at the end to add to the central pipeline.
"""

# ============= IMPORT LIBRARIES ==============
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from src.models.model_utils import MLP


# ============= CONFIGURATION OR FILE SPECIFIC =============


# ============= UTILITY FUNCTIONS =============


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
    X_sum = torch.sum(X, dim=0, keepdim=True)

    # Generate Dirichlet samples by normalising
    dirichlet_samples = torch.divide(X, X_sum + 1e-8)

    return dirichlet_samples


def _sample_differentiable_gamma_dist(alphas):
    """
    Compute a gamma (rate, 1) sample according to vector of shape parameters. We use inverse transform sampling
    so that we can backpropagate through generated sample.
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
    _input_size = list(x.size)[-1]
    log_lik = - 0.5 * _input_size * torch.log(2 * np.pi) - 0.5 * torch.prod(var, dim=-1, keepdim=False) - exp_term

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


# ============= MAIN MODEL DEFINITION =============


class DirVRNN(nn.Module):

    def __init__(self, input_size, outcome_size, latent_size, gate_layers, gate_nodes,
                 feat_extr_layers, feat_extr_nodes, num_clus, **kwargs):
        super().__init__()

        # Define the dimensions of the input, hidden, and latent variables
        self.input_size = input_size
        self.outcome_size = outcome_size
        self.latent_dim = latent_size
        self.gate_layers = gate_layers
        self.gate_nodes = gate_nodes
        self.feat_extr_layers = feat_extr_layers
        self.feat_extr_nodes = feat_extr_nodes
        self.K = num_clus

        # Define the encoder network - Computes alpha of Dirichlet distribution given h and z
        self.encoder = MLP(input_size=self.latent_dim + self.latent_dim,
                           output_size=self.K,
                           act_fn="relu",
                           hidden_layers=self.gate_layers,
                           hidden_nodes=self.gate_nodes,
                           output_fn="relu")

        # Define the decoder network - computes mean, var of Observation data
        self.decoder = MLP(input_size=self.latent_dim + self.latent_dim,  # input is z and h
                           output_size=self.input_size + self.input_size,  # output is mean/var of observation
                           hidden_layers=gate_layers,
                           hidden_nodes=gate_nodes,
                           act_fn="relu",
                           output_fn="tanh")

        # Define the prior network
        self.prior = MLP(input_size=self.latent_dim,
                         output_size=self.K,
                         act_fn="relu",
                         hidden_layers=self.gate_layers,
                         hidden_nodes=self.gate_nodes,
                         output_fn="relu")

        # Define the output network
        self.predictor = MLP(input_size=self.latent_dim,
                             output_size=self.outcome_size,
                             hidden_layers=gate_layers,
                             hidden_nodes=gate_nodes,
                             act_fn="relu",
                             output_fn="tanh")

        # Define feature extractors
        self.phi_x = MLP(input_size=self.input_size,
                         output_size=self.latent_dim,
                         hidden_layers=feat_extr_layers,
                         hidden_nodes=self.feat_extr_nodes,
                         act_fn="relu",
                         output_fn="tanh")

        self.phi_z = MLP(input_size=self.latent_dim,
                         output_size=self.latent_dim,
                         hidden_layers=feat_extr_layers,
                         hidden_nodes=feat_extr_nodes,
                         act_fn="relu",
                         output_fn="tanh")

        # Recurrency
        self.cell_state_update = MLP(input_size=self.latent_dim + self.latent_dim + self.latent_dim,
                                     output_size=self.latent_dim,
                                     hidden_layers=self.gate_layers,
                                     hidden_nodes=self.gate_nodes,
                                     act_fn="relu",
                                     output_fn="tanh")

    # What to do given input
    def forward(self, x, y):
        """
        Computations and loss update of the model

        """

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()

        # Initialize the current hidden state and cluster means
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        cluster_means = torch.rand(size=[self.K, self.latent_dim]).to(x.device)
        loss = 0

        # Iterate over the sequence
        for t in range(seq_len):

            # Observation data
            x_t = x[:, t, :]

            # Compute encoder alpha by extracting features from input and concatenating with hidden state
            joint_enc_input = torch.cat([
                self.phi_x(x_t),  # Extract feature from x
                h],
                dim=-1  # Concatenate with cell state in last dimension
            )
            alpha_enc = self.encoder(joint_enc_input)

            # Sample mechanism
            pi_samples = sample_dirichlet(alpha=alpha_enc)

            # Average over clusters to estimate latent variable
            z_samples = torch.matmul(pi_samples, cluster_means)

            # Compute posterior parameters by extracting features from latent and concatenating with hidden state
            joint_dec_input = torch.cat([
                self.phi_z(z_samples),
                h],
                dim=-1
            )
            mu_g, logvar_g = torch.chunk(
                self.decoder(joint_dec_input),
                chunks=2,
                dim=-1
            )
            var_g = torch.exp(logvar_g)

            # Compute prior parameters
            alpha_prior = self.prior(h)

            # Update Cluster means
            cluster_means = estimate_new_clus(pi_samples, z_samples)

            # -------- COMPUTE LOSS FOR TIME t -----------

            # Data Likelihood component
            log_lik = compute_gaussian_log_lik(x_t, mu_g, var_g)

            # KL divergence
            kl_div = compute_dirichlet_kl_div(alpha_enc, alpha_prior)

            loss += log_lik - kl_div

            # Last Time Step
            if t == seq_len - 1:
                # Predict outcome
                y_pred = self.predictor(z_samples)

                # Compute loss
                pred_loss = compute_outcome_loss(y_true=y, y_pred=y_pred)

                # Add to total loss
                loss += pred_loss

            # ----- UPDATE CELL STATE
            state_update_input = torch.cat([
                self.phi_z(z_samples),
                self.phi_x(x_t),
                h],
                dim=-1
            )
            h = self.cell_state_update(state_update_input)

        return loss


# =============== MODEL WRAPPER WITH TRAIN/TEST METHODS ==================

class Model:
    """
    Wrapper class to implement Dirichlet VRNN model.
    """

    def __init__(self, data_info, model_config):
        """Initialise model configuration parameters."""
        self.data_info = data_info
        self.model_config = model_config

        # Empty attribute which will be updated later.
        self.model = None
        self.optimizer = None
        self.train_params = None

    def train(self, train_params):
        """
        Train method for training CAMELOT model.

        Params:
        - train_params: dictionary containing training parameter information:
            - "lr": learning rate for training
            - "epochs_init": number of epochs to train initialisation
            - "epochs": number of epochs for main training
            - "bs": batch size
            - "cbck_str": callback_string indicating which callbacks to print during training
        """

        # Unpack relevant data information
        X_train, X_val, X_test = self.data_info["X"]
        output_dim = self.data_info["output_dim"]

        # Unpack training parameters
        self.train_params = train_params
        num_epochs, lr = train_params["epochs"], train_params["lr"]

        # Initialise model
        self.model = DirVRNN(input_size=X_train.shape[-1],
                             **self.model_config)

        # Useful for model training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ============ TRAINING LOOP ==============

        # Iterate through epochs
        for epoch in range(1, num_epochs + 1):
            self.evaluate(epoch)
            self.validate(epoch)

    def evaluate(self, epoch):
        """
        Train model over 1 epoch.
        """
        train_loss = 0
        optimizer = self.optimizer

        # Load training parameters
        batch_size = self.train_params["bs"]

        # Load data
        X_train, X_val, X_test = self.data_info["X"]
        y_train, y_val, y_test = self.data_info["y"]

        # Prepare train data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        for batch_id, (X_batch, y_batch) in enumerate(train_loader):
            # Apply optimiser
            optimizer.zero_grad()
            loss = self.model(X_batch)

            # Back_propagate
            loss.backward()
            optimizer.step()

            # Add to train loss
            train_loss += loss.item()

            print("Train epoch: {}   [{:.5f} - {:.0f}%]".format(
                epoch, loss.item(), 100. * batch_id / len(train_loader)),
                end="\r")

    def validate(self, epoch):
        """Evaluate model performance on test data."""

        # Load data
        X_train, X_val, X_test = self.data_info["X"]
        y_train, y_val, y_test = self.data_info["y"]

        # Prepare train data loader
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

        for batch_id, (X_batch, y_batch) in enumerate(val_loader):

            # Compute Loss
            loss = self.model(X_batch).item()

            print("Train epoch: {}   [{:.5f} - {:.0f}%]".format(
                epoch, loss.item(), 100. * batch_id / len(val_loader)),
                end="\r")
