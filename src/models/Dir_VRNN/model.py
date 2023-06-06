"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Model file to define GC-DaPh class.
"""

# ============= IMPORT LIBRARIES ==============
import torch
import torch.nn as nn

from torchvision.ops import MLP
from torch.nn import LSTM
import src.models.Dir_VRNN.Dir_VRNN_utils as utils


# Define Neural Networks

# region Define LSTM Decoder v1 (use output at time t as input at time t+1)
class LSTM_Dec_v1(nn.Module):
    """
    Implements LSTM Decoder architecture. A context vector of fixed dimension is given as input to the hidden/cell state of the decoder, and the first input is a vector
    of zeros. At each time-step the output of the LSTM is used as input for the next time-step.
    """
    def __init__(self,
        seq_len: int,     # Number of observations to generate
        output_dim: int,   # Dimensionality of output vectors
        hidden_dim: int,  # Dimensionality of hidden state
        num_layers: int = 1,  # Number of layers (default = 1)
        dropout: float = 0.0, # Dropout rate (default = 0.0, i.e. no dropout)
        **kwargs): 
        """
        Object initialization given input parameters.
        """

        # Call parent class constructor
        super().__init__()

        # Initialise parameters
        self.seq_len, self.h_dim = seq_len, hidden_dim
        self.o_dim, self.i_dim = output_dim, output_dim      # Input and output dimensions are the same for this LSTMCell
        self.n_layers, self.dropout = num_layers, dropout

        # Define main LSTM layer and output layer
        self.lstm_cell = LSTM(input_size=self.i_dim, hidden_dim=self.h_dim, num_layers=self.n_layers, dropout=self.dropout, **kwargs)
        self.fc_out = nn.Linear(self.h_dim, self.i_dim)

    # Forward pass
    def forward(self, c_vector: torch.Tensor):
        """
        Forward pass of LSTM Decoder. Given a context vector, initialise cell and hidden states of LSTM decoder. Pass zero as first input vector of sequence.

        Params:
        - c_vector: context vector of fixed dimensionality, of shape (N, D), where N is batch size and D is latent dimensionality.
        """
        
        #  Get parameters
        bs, T = c_vector.shape[0], self.seq_len

        # Define trackers for cell and hidden states.
        c_states = torch.zeros(bs, T, self.h_dim).to(c_vector.device)
        h_states = torch.zeros(bs, T, self.h_dim).to(c_vector.device)

        # Set values at time 0
        c_states[:, 0, :] = c_vector
        h_states[:, 0, :] = c_vector

        # Initialise iterates
        i_t = torch.zeros(bs, self.i_dim).to(c_vector.device)
        h_t, c_t = c_vector, c_vector
        
        # Apply LSTM Cell to generate sequence
        for t in range(T):

            # Pass through LSTM cell and update input-cell-hidden states
            h_t, c_t = self.lstm_cell(i_t, (h_t, c_t))
            i_t = self.fc_out(c_t)

            # Save states
            c_states[:, t, :] = c_t
            h_states[:, t, :] = h_t

        # Return cell and hidden states
        return h_states, c_states
# endregion


# region Define LSTM Decoder v2 (use context vector as input for multiple time steps)
class LSTM_Dec_v2(nn.Module):
    """
    Implements LSTM Decoder architecture. A context vector of fixed dimension is given, and used as the sequence of input vectors for multiple time steps for the LSTM.
    """
    def __init__(self, 
        time_steps: int, # Number of observations to generate
        input_dim: int,  # Dimensionality of input vectors
        output_dim: int, # Dimensionality of output vector
        hidden_dim: int, # Dimensionality of hidden state
        num_layers: int = 1,  # Number of layers (default = 1)
        dropout: float = 0.0, # Dropout rate (default = 0.0, i.e. no dropout)
        **kwargs):

        # Call parent class constructor
        super().__init__()

        # Initialise parameters
        self.num_steps = time_steps
        self.i_dim, self.o_dim, self.h_dim = input_dim, output_dim, hidden_dim
        self.n_layers, self.dropout = num_layers, dropout

        # Define main LSTM layer and output layer
        self.lstm = LSTM(input_size=self.i_dim, hidden_dim=self.h_dim, num_layers=self.n_layers, dropout=self.dropout, **kwargs)
        self.fc_out = nn.Linear(self.h_dim, self.o_dim)  

    # Forward pass
    def forward(self, context_vector: torch.Tensor):
        """
        Forward pass of v2 LSTM Decoder. Given context vector, use it as input sequence for LSTM.
        
        Params:
            - context_vector: context vector of fixed dimensionality, of shape (N, D), where N is batch size and D is latent dimensionality.
        """

        # Define input sequence for LSTM
        input_seq = context_vector.unsqueeze(1).expand(-1, self.num_steps, -1) # (N, T, D)
        batch_size = input_seq.shape[0]

        # Pass through LSTM
        h0 = torch.zeros(self.num_layers, self.batch_size, batch_size, self.h_dim).to(context_vector.device)
        c0 = torch.zeros(self.num_layers, self.batch_size, batch_size, self.h_dim).to(context_vector.device)
        output, _ = self.lstm(input_seq, (h0, c0)) # (N, T, D)

        return output
# endregion



# region DirVRNN
class BaseModel(nn.Module):
    """ 
    Implements DirVRNN model as described in the paper.

    Given a multi-dimensional time-series of observations, we model cluster assignments over time using a window approach. For each window:
    a) estimate data generation,
    b) estimate cluster assignments,
    c) estimate cell state,
    d) update cluster representations.
    """
    def __init__(self, 
                i_size: int, 
                o_size: int, 
                w_size: int,
                K: int, 
                l_size: int = 10,
                gate_hidden_l: int = 2,
                gate_hidden_n: int = 20,
                **kwargs):
        """
        Object initialization.

        Params:
            - i_size: dimensionality of observation data (number of time_series).
            - o_size: dimensionality of outcome data (number of outcomes).
            - w_size: window size over which to update cell state and generate data.
            - K: number of clusters to consider.
            - l_size: dimensionality of latent space (default = 10).
            - gate_hidden_l: number of hidden layers for gate networks (default = 2).
            - gate_hidden_n: number of nodes of hidden layers for gate networks (default = 20).
            - kwargs: additional arguments for compatibility.
        """
        super().__init__()

        # Initialise input parameters
        self.i_size, self.o_size, self.w_size = i_size, o_size, w_size
        self.K, self.l_size = K, l_size
        self.gate_l, self.gate_n = gate_hidden_l, gate_hidden_n

        # Parameters for Clusters
        self.c_means = torch.rand(int(self.K), int(self.l_size))
        self.log_c_vars = torch.rand(int(self.K), )


        # Initialise encoder block - estimate alpha parameter of Dirichlet distribution given h and extracted input data features.
        self.encoder = MLP(
            in_channels= self.l_size + self.l_size,      # input: concat(h, extr(x))
            hidden_channels=[self.gate_n] * self.gate_l, # gate_l hidden_layers with gate_n nodes each
            norm_layer=None,
            activation_layer=nn.ReLU,             # default activation function is ReLU
            inplace=True,
            bias=True,
            dropout=0.0                           # default dropout is 0 (no dropout)
        )

        # Initialise Prior Block - estimate alpha parameter of Dirichlet distribution given h.
        self.prior = MLP(
            in_channels=self.l_size,               # input: h
            hidden_channels=[self.gate_n] * self.gate_l, # gate_l hidden_layers with gate_n nodes each
            norm_layer=None,
            activation_layer=nn.ReLU,             # default activation function is ReLU
            inplace=True,
            bias=True,
            dropout=0.0                           # default dropout is 0 (no dropout)
        )
        
        # Initialise decoder block - given z, h, we generate a w_size sequence of observations, x.
        self.decoder = LSTM_Dec_v1(
            seq_len=self.w_size,
            output_dim=self.input_size + self.input_size, # output is concatenation of mean-log var of observation
            hidden_dim=self.hidden_size,
            num_layers=1,
            dropout=0.0
        )

        # 

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
        self.phi_x = MLP(input_size=self.input_size,  # input is set of observations.
                         output_size=self.latent_dim,  # output is extracted features.
                         hidden_layers=self.feat_extr_layers,
                         hidden_nodes=self.feat_extr_nodes,
                         act_fn="relu",
                         output_fn="tanh")

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
# endregion
        
        return None
