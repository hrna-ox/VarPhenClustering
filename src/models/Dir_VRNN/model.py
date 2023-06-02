"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Model file to define GC-DaPh class.
"""

# ============= IMPORT LIBRARIES ==============
import torch
import torch.nn as nn

from torch.nn import LSTM
from src.models.model_utils import MLP
import src.models.Dir_VRNN.Dir_VRNN_utils as utils


# ============= MAIN MODEL DEFINITION =============

class BaseModel(nn.Module):

    def __init__(self, input_size: int, outcome_size: int, num_clus: int, latent_size: int = 10,
                 gate_layers: int = 2, gate_nodes: int = 2,
                 feat_extr_layers: int = 2, feat_extr_nodes: int = 2, **kwargs):
        """
        Object initialization.

        Params:
            - input_size: dimensionality of observation data (number of time_series).
            - outcome_size: dimensionality of outcome data (number of outcomes).
            - num_clus: number of clusters to consider.
            - latent_size: dimensionality of latent space (default = 10).
            - gate_layers: number of hidden_layers for prior, encoder, representation layers (default = 2).
            - gate_nodes: number of nodes per hidden_layer for gate_layers layers (default = 30).
            - feat_extr_layers: number of hidden_layers for feature extraction functions (default = 2).
            - feat_extr_nodes: number of nodes per layer for feat_extr_layers layers (default = 20).
            - kwargs: additional arguments (not used) - for compatibility with other models.
        """
        super().__init__()

        # ======== Load Parameters as Attributes ==========

        # Fundamental variables
        self.input_size = input_size
        self.outcome_size = outcome_size
        self.K = num_clus
        self.latent_dim = latent_size

        # Variables related to neural network parameters
        self.gate_layers = gate_layers
        self.gate_nodes = gate_nodes
        self.feat_extr_layers = feat_extr_layers
        self.feat_extr_nodes = feat_extr_nodes

        # Parameters for clusters
        self.clus_means = torch.rand(int(self.K), int(self.latent_dim))
        self.log_clus_vars = torch.rand(int(self.K), )

        # ====== Define Model Blocks ========
        # Define the encoder network - Computes alpha of Dirichlet distribution given h and z
        self.encoder = MLP(input_size=self.latent_dim + self.latent_dim,  # input is z_extr_features and h
                           output_size=self.K,  # output is alpha vector of size K
                           act_fn="relu",
                           hidden_layers=self.gate_layers,
                           hidden_nodes=self.gate_nodes,
                           output_fn="relu")

        # Define the decoder network - computes mean, var of Observation data
        self.decoder = MLP(input_size=self.latent_dim + self.latent_dim,  # input is z_extr_features and h
                           output_size=self.input_size + self.input_size,  # output is mean/var of observation
                           hidden_layers=self.gate_layers,
                           hidden_nodes=self.gate_nodes,
                           act_fn="relu",
                           output_fn="tanh")

        # Define the prior network - computes Dirichlet alpha prior given previous cell state
        self.prior = MLP(input_size=self.latent_dim,  # input is h at previous time step
                         output_size=self.K,  # output is alpha dirichlet for prior distribution
                         act_fn="relu",
                         hidden_layers=self.gate_layers,
                         hidden_nodes=self.gate_nodes,
                         output_fn="relu")

        # Define the output network - computes outcome prediction given predicted cluster assignments
        self.predictor = MLP(input_size=self.latent_dim,  # input is generated samples at last time step
                             output_size=self.outcome_size,  # output is categorical parameter p
                             hidden_layers=self.gate_layers,
                             hidden_nodes=self.gate_nodes,
                             act_fn="relu",
                             output_fn="softmax")

        # Define feature extractors - feature representations for input observation and latent variables
        self.phi_x = 

        self.phi_z = MLP(input_size=self.latent_dim,  # input is average representation vector.
                        output_size=self.latent_dim,  # output is extracted feature version of input.
                        hidden_layers=self.feat_extr_layers,
                        hidden_nodes=self.feat_extr_nodes,
                        act_fn="relu",
                        output_fn="tanh")

        # Recurrence - how to update Cell State
        self.cell_state_update = MLP(input_size=self.latent_dim + self.latent_dim + self.latent_dim,
                                     output_size=self.latent_dim,
                                     hidden_layers=self.gate_layers,
                                     hidden_nodes=self.gate_nodes,
                                     act_fn="relu",
                                     output_fn="tanh")

    # Define training process for this class.
    def forward(self, x, y):
        """
        Model loss computation given batch objects during training.

        Params:
            - x: pytorch Tensor object of input data with shape (batch_size, max_seq_len, input_size);
            - y: pytorch Tensor object of corresponding outcomes with shape (batch_size, outcome_size).

        Output:
            - loss: pytorch Tensor object of loss value.
            - history_objects: dictionary of relevant objects during training:
                a) alpha parameters of Dirichlet distribution for each time step;
                b) cluster assignment probabilities for each time step;
                c) predicted outcomes for the last time step;
                d) estimated cluster means at each time step;
                e) estimated generated data at each time step;
                f) cell state vector at each time step.
        """

        # ========= Define relevant variables and initialise variables ===========

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()

        # Initialize the current hidden state as the null vector and cluster means as random objects in latent space
        h = torch.zeros(batch_size, self.latent_dim).to(x.device)

        # Initialise history objects
        history_objects = {
            "alpha_prior": torch.zeros(size=[batch_size, seq_len, self.K]).to(x.device),
            "alpha_enc": torch.zeros(size=[batch_size, seq_len, self.K]).to(x.device),
            "est_pi": torch.zeros(size=[batch_size, seq_len, self.K]).to(x.device),
            "est_cluster_means": torch.zeros(size=[seq_len, self.K, self.latent_dim]).to(x.device),
            "est_outcomes": torch.zeros(size=[batch_size, self.outcome_size]).to(x.device),
            "est_gen_mean": torch.zeros(size=[batch_size, seq_len, self.input_size]).to(x.device),
            "est_gen_data": torch.zeros(size=[batch_size, seq_len, self.input_size]).to(x.device),
            "cell_state": torch.zeros(size=[batch_size, seq_len, self.latent_dim]).to(x.device),
            "pred_loss": torch.zeros(size=[batch_size, seq_len]).to(x.device)
        }

        # Initialise loss value
        loss = 0

        # ================== Iteration through time-steps ==============
        for t in range(seq_len):

            # Subset observation to time t
            x_t = x[:, t, :]

            # Compute encoder alpha by extracting features from input and concatenating with hidden state
            joint_enc_input = torch.cat([
                self.phi_x(x_t),  # Extract feature from x
                h],
                dim=-1  # Concatenate with cell state in last dimension
            )
            alpha_enc = self.encoder(joint_enc_input) + 1e-6  # Add small value to avoid numerical instability

            # Sample cluster distribution from Dirichlet with parameter alpha_enc
            pi_samples = utils.sample_dirichlet(alpha=alpha_enc)
            # pi_samples = torch.divide(alpha_enc, torch.sum(alpha_enc, dim=-1, keepdim=True))

            # Average over clusters to estimate latent variable - we do this via sampling from the clusters
            clus_samples = utils.sample_normal(self.clus_means, self.log_clus_vars)
            z_samples = torch.matmul(pi_samples, clus_samples)

            # Compute decoder mean/var parameters by extracting features from latent and concatenating with hidden state
            joint_dec_input = torch.cat([
                self.phi_z(z_samples),
                h],
                dim=-1
            )
            mu_g, logvar_g = torch.chunk(  # split output vector of decoder to 2 chunks
                self.decoder(joint_dec_input),
                chunks=2,
                dim=-1
            )
            var_g = torch.exp(logvar_g)
            try:
                assert torch.all(var_g > 0)
            except AssertionError as e:
                print("\nvarg_g:\n", var_g)
                print("\nlogvar_g:\n", logvar_g)
                print("\njoint dec input:\n", joint_dec_input)
                print("\nh:\n", h)
                print("\nz samples:\n", z_samples)
                print("\nClus samples:\n", clus_samples)
                print("\nPi samples:\n", pi_samples)
                print("\nAlpha enc:\n", alpha_enc)
                print("\njoint enc input:\n", joint_enc_input)
                print("\nx_t:\n", x_t)

                raise ValueError("Variance is not positive definite.")

            # Compute prior parameters
            alpha_prior = self.prior(h)

            # Update Cluster means
            cluster_means = utils.estimate_new_clus(pi_samples, z_samples)

            # -------- COMPUTE LOSS FOR TIME t -----------

            # Data Likelihood component
            log_lik = utils.compute_gaussian_log_lik(x_t, mu_g, var_g)

            # KL divergence
            kl_div = utils.compute_dirichlet_kl_div(alpha_enc+1e-8, alpha_prior+1e-8)
            # quot = torch.log(torch.divide(alpha_enc, alpha_prior + 1e-8) + 1e-8)
            # kl_div = torch.mean(torch.sum(alpha_enc * quot, dim=-1))

            # Add to loss tracker
            loss += log_lik - kl_div

            # If last time step, add outcome prediction term
            if t == seq_len - 1:

                # Predict outcome
                y_pred = self.predictor(z_samples)

                # Compute log loss of outcome
                pred_loss = utils.compute_outcome_loss(y_true=y, y_pred=y_pred)

                # Add to total loss
                loss += pred_loss

            # ----- UPDATE CELL STATE ------
            state_update_input = torch.cat([
                self.phi_z(z_samples),
                self.phi_x(x_t),
                h],
                dim=-1
            )
            h = self.cell_state_update(state_update_input)

            # --------- ADD OBJECTS TO HISTORY TRACKER ---------
            history_objects["alpha_prior"][:, t, :] = alpha_prior
            history_objects["alpha_enc"][:, t] = alpha_enc
            history_objects["est_pi"][:, t, :] = pi_samples
            history_objects["est_cluster_means"][t, :, :] = cluster_means
            history_objects["est_gen_mean"][:, t, :] = mu_g
            history_objects["est_gen_data"][:, t, :] = torch.normal(mu_g, torch.sqrt(var_g))
            history_objects["cell_state"][:, t, :] = h
            

        # Add outcome
        history_objects["y_pred"] = y_pred
        history_objects["pred_loss"][:, t] = pred_loss

        return - loss, history_objects      # want to maximize loss, so return negative loss
