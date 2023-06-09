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

import src.models.loss_functions as losses


# region ========== DEFINE AUXILIARY FUNCTIONS ==========

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

    # Divide gamma samples across rows to normalise (which gives a dirichlet sample)
    row_sum = torch.sum(gamma_samples, dim=1, keepdim=True)
    dir_samples = gamma_samples / (row_sum + eps)

    return dir_samples


def gen_samples_from_assign(pi_assign, clus_means, log_clus_vars):
    """
    Generate samples from latent variable cluster assignment. We re-parameterize normal sampling
    in order to ensure back-propagation.

    Args:
        pi_assign: tensor of shape (bs, K) with probabilities of cluster assignment. Rows sum to 1.
        clus_means: tensor of shape (K, l_dim) with cluster mean vectors.
        log_clus_vars: tensor of shape (K, ) with cluster log of variance parameters.

    Outputs:
        tensor of shape (bs, l_dim) with generated samples.
    """
    # Get parameters from vectors
    K, l_dim = clus_means.shape
    
    # Generate standard normal samples and apply transformation to obtain multivariate normal
    stn_samples = torch.randn(size=[K, l_dim], device=clus_means.device)
    mvn_samples = clus_means + torch.exp(0.5 * log_clus_vars).reshape(-1, 1) * stn_samples

    # Combine multivariate normal samples with cluster assignment probabilities
    samples = torch.matmul(pi_assign, mvn_samples)

    return samples
# endregion


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
        self.lstm_cell = LSTM(input_size=self.i_dim, hidden_size=self.h_dim, num_layers=self.n_layers, dropout=self.dropout, **kwargs)
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
        outputs = torch.zeros(bs, T, self.o_dim).to(c_vector.device)

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
            o_t = self.fc_out(c_t)

            # Save states
            c_states[:, t, :] = c_t
            h_states[:, t, :] = h_t
            outputs[:, t, :] = o_t

        # Return cell and hidden states
        return h_states, c_states, outputs
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
        #  Get parameters
        bs, T = context_vector.shape[0], self.seq_len

        # Define input sequence for LSTM
        input_seq = context_vector.unsqueeze(1).expand(-1, self.num_steps, -1) # (N, T, D)
        batch_size = input_seq.shape[0]

        # Pass through LSTM
        h0 = torch.zeros(bs, T, self.h_dim).to(context_vector.device)
        c0 = torch.zeros(bs, T, self.h_dim).to(context_vector.device)
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
        self.enc_out = nn.Sequential(
            nn.Linear(self.gate_n, self.K),
            nn.Softmax(dim=-1)
        )                                # output is K-dimensional vector of cluster probabilities

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
        self.prior_out = nn.Sequential(
            nn.Linear(self.gate_n, self.K),
            nn.Softmax(dim=-1)
        )
        
        # Initialise decoder block - given z, h, we generate a w_size sequence of observations, x.
        self.decoder = LSTM_Dec_v1(
            seq_len=self.w_size,
            output_dim=self.i_size + self.i_size, # output is concatenation of mean-log var of observation
            hidden_dim=self.l_size,
            num_layers=1,
            dropout=0.0
        )

        # Define the output network - computes outcome prediction given predicted cluster assignments
        self.predictor = MLP(
            in_channels=self.l_size,               # input: h
            hidden_channels=[self.gate_n] * self.gate_l, # gate_l hidden_layers with gate_n nodes each
            norm_layer=None,
            activation_layer=nn.ReLU,             # default activation function is ReLU
            inplace=True,
            bias=True,
            dropout=0.0                           # default dropout is 0 (no dropout)
        )
        self.pred_out = nn.Sequential(
            nn.Linear(self.gate_n, self.o_size),
            nn.Softmax(dim=-1)
        )                                     # output is o_size-dimensional vector of outcome probabilities
        

        # Define feature transformation functions
        self.phi_x = MLP(
            in_channels=self.i_size,               # input: x
            hidden_channels=[self.gate_n] * self.gate_l, # gate_l hidden_layers with gate_n nodes each
            norm_layer=None,
            activation_layer=nn.ReLU,             # default activation function is ReLU
            inplace=True,
            bias=True,
            dropout=0.0                           # default dropout is 0 (no dropout)
        )
        self.phi_x_out = nn.Sequential(
            nn.Linear(self.gate_n, self.l_size),
            nn.Tanh()
        )                                  # output is l_size-dimensional vector of latent space features

        self.phi_z = MLP(
            in_channels=self.l_size,               # input: z
            hidden_channels=[self.gate_n] * self.gate_l, # gate_l hidden_layers with gate_n nodes each
            norm_layer=None,
            activation_layer=nn.ReLU,             # default activation function is ReLU
            inplace=True,
            bias=True,
            dropout=0.0                           # default dropout is 0 (no dropout)
        )
        self.phi_z_out = nn.Sequential(
            nn.Linear(self.gate_n, self.l_size),
            nn.Tanh()
        )                                    # output is l_size-dimensional vector of latent space features

        # Define Cell Update Gate Functions
        self.state_update = MLP(
            in_channels=self.l_size + self.l_size + self.l_size, # input: concat(h, phi_x, phi_z)
            hidden_channels=[self.gate_n] * self.gate_l, # gate_l hidden_layers with gate_n nodes each
            norm_layer=None,
            activation_layer=nn.ReLU,             # default activation function is ReLU
            inplace=True,
            bias=True,
            dropout=0.0                           # default dropout is 0 (no dropout)
        )
        self.state_out = nn.Sequential(
            nn.Linear(self.gate_n, self.l_size),
            nn.Tanh()
        )
    # Define training process for this class.
    def forward(self, x, y):
        """
        Model loss computation given batch objects during training. Note that we implement 
        a window size within our model itself. Time alignment is checked through pre-processing.

        Params:
            - x: pytorch Tensor object of input data with shape (batch_size, max_seq_len, input_size);
            - y: pytorch Tensor object of corresponding outcomes with shape (batch_size, outcome_size).

        Output:
            - loss: pytorch Tensor object of loss value.
        """

        # ========= Define relevant variables and initialise variables ===========

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()

        # Initialize the current hidden state as the null vector
        h = torch.zeros(batch_size, self.l_size).to(x.device)

        # Initialise probability of cluster assignment as uniform
        est_pi = torch.ones(batch_size, self.K) / self.K
        est_z = gen_samples_from_assign(est_pi, self.clus_means, self.clus_log_vars)
                

        # Initialise loss value
        loss = 0
        num_time_steps = int(seq_len / self.w_size)

        # ================== Iteration through time-steps ==============
        # Can also edit this to use a sliding window approach, where t goes from 0 to seq_len - w_size
        for window_id in range(num_time_steps):
            "Iterate through each window block"
            
            # Bottom and high indices
            lower_t, higher_t = window_id * self.w_size, (window_id + 1) * self.w_size

            # First we estimate the observations for the incoming window given current cell state
            h_dec, c_dec, output_pred = self.decoder(
                torch.cat([
                    self.phi_z(est_z),
                    h
                ], dim=-1)
            )

            # Decompose obvs_pred into mean and log-variance - shape is (bs, T, 2 * input_size)
            mu_g, logvar_g = torch.chunk(output_pred, chunks=2, dim=-1)
            var_g = torch.exp(logvar_g) + eps

            for _w_id, t in enumerate(range(lower_t, higher_t)):
                # Estimate alphas for each time step within the window. 

                # Subset observation to time t
                x_t = x[:, t, :]

                # Compute alpha based on prior gate
                alpha_prior = self.prior_out(self.prior(h)) + eps 

                # Compute alpha of encoder network
                alpha_enc = self.enc_out(
                    self.encoder(
                        torch.cat([
                            self.phi_x(x_t),  # Extract feature from x
                            h
                        ], dim=-1)  # Concatenate with cell state in last dimension
                    ) 
                ) + eps  # Add small value to avoid numerical instability


                # Sample cluster distribution from alpha_enc, and estimate samples from clusters
                est_pi = sample_dir(alpha=alpha_enc)
                est_z = gen_samples_from_assign(
                    pi_assign=est_pi, 
                    clus_means=self.clus_means, 
                    clus_log_vars=self.log_clus_vars
                )
                # pi = torch.divide(alpha_enc, torch.sum(alpha_enc, dim=-1, keepdim=True))

                # -------- COMPUTE LOSS FOR TIME t -----------

                # Data Likelihood component
                log_lik = losses.log_gaussian_lik(x_t, mu_g[:, _w_id, :], var_g[:, _w_id, :])

                # Posterior KL-divergence component
                kl_div = losses.dir_kl_div(a1=alpha_enc, a2=alpha_prior)
                # quot = torch.log(torch.divide(alpha_enc, alpha_prior + 1e-8) + 1e-8)
                # kl_div = torch.mean(torch.sum(alpha_enc * quot, dim=-1))

                # Add to loss tracker
                loss += log_lik - kl_div


                # ----- UPDATE CELL STATE ------
                h = self.state_out(
                        self.state_update(
                            torch.cat([
                                self.phi_z(est_z),
                                self.phi_x(x_t),
                                h
                            ], dim=-1)
                        )
                    )

        # Once all times have been computed, make predictions on outcome
        y_pred = self.pred_out(self.predictor(est_z))

        # Compute log loss of outcome
        pred_loss = losses.cat_cross_entropy(y_true=y, y_pred=y_pred)

        # Add to total loss
        loss += pred_loss

        return (-1) * loss      # want to maximize loss, so return negative loss
# endregion
        