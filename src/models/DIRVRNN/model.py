"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file defines the main proposed model, and how to train the model.
"""

# ============= IMPORT LIBRARIES ==============
import pickle
import torch
import torch.nn as nn
import wandb
import os

from typing import Dict, List, Union
from torch.utils.data import DataLoader, TensorDataset

import src.models.loss_functions as LM_utils
from src.models.deep_learning_base_classes import MLP, LSTM_Dec_v1

import src.models.model_utils as model_utils
import src.models.logging_utils as log_utils


# region DirVRNN
class DirVRNN(nn.Module):
    """
    Implements DirVRNN model as described in the paper.

    Given a multi-dimensional time-series of observations, we model cluster assignments over time using a window approach. For each window:
    a) estimate data generation,
    b) estimate cluster assignments,
    c) estimate cell state,
    d) update cluster representations.
    """

    def __init__(
        self,
        input_dims: int,
        num_classes: int,
        window_num_obvs: int,
        K: int,
        latent_dim: int = 10,
        n_fwd_blocks: int = 1,
        gate_num_hidden_layers: int = 2,
        gate_num_hidden_nodes: int = 20,
        bias: bool = True,
        dropout: float = 0.0,
        device: str = "cpu",
        seed: int = 42,
        K_fold_idx: int = 1,
        **kwargs,
    ):
        """
        Object initialization.

        Params:
            - input_dims: dimensionality of observation data (number of time_series).
            - num_classes: dimensionality of outcome data (number of outcomes).
            - window_num_obvs: number of observations to consider in each window.
            - K: number of clusters to consider.
            - latent_dim: dimensionality of latent space (default = 10).
            - n_fwd_blocks: number of forward blocks to predict (int, default=1).
            - gate_num_hidden_layers: number of hidden layers for gate networks (default = 2).
            - gate_num_hidden_nodes: number of nodes of hidden layers for gate networks (default = 20).
            - bias: whether to include bias terms in MLPs (default = True).
            - dropout: dropout rate for MLPs (default = 0.0, i.e. no dropout).
            - device: device to use for computations (default = 'cpu').
            - seed: random seed for reproducibility (default = 42).
            - K_fold_idx: index of the K-fold cross-validation (default = 1).
            - kwargs: additional arguments for compatibility.
        """
        super().__init__()

        # Initialise input parameters
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.window_num_obvs = window_num_obvs
        self.K = K
        self.latent_dim = latent_dim
        self.n_fwd_blocks = n_fwd_blocks
        self.gate_l = gate_num_hidden_layers
        self.gate_n = gate_num_hidden_nodes
        self.device = device
        self.seed = seed
        self.K_fold_idx = K_fold_idx

        # Initialize Cluster Mean and Variance Parameters
        self.c_means = nn.Parameter(
            torch.rand(
                (int(self.K), int(self.latent_dim)),
                requires_grad=True,
                device=self.device,
            )
        )
        self.log_c_vars = nn.Parameter(
            torch.rand(
                (int(self.K),), 
                requires_grad=True, 
                device=self.device)
        )


        # Initialise encoder block - estimate alpha parameter of Dirichlet distribution given h and extracted input data features.
        self.encoder = MLP(
            input_size=self.latent_dim + self.latent_dim,  # input: concat(h, extr(x))
            output_size=self.K,  # output: K-dimensional vector of cluster probabilities
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,
            dropout=dropout,
        )
        self.encoder_output_fn = nn.Softmax(dim=-1)  

        # Initialise Prior Block - estimate alpha parameter of Dirichlet distribution given h.
        self.prior = MLP(
            input_size=self.latent_dim,  # input: h
            output_size=self.K,  # output: K-dimensional vector of cluster probabilities
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,
            dropout=dropout,
        )
        self.prior_output_fn = nn.Softmax(dim=-1)

        # Initialise decoder block - given z, h, we generate a window_num_obvs sequence of observations, x.
        self.decoder = LSTM_Dec_v1(
            seq_len=self.window_num_obvs,
            output_dim=self.input_dims + self.input_dims,  # output is concatenation of mean-log var of observation
            hidden_dim=self.latent_dim + self.latent_dim,  # state cell has same size as context vector (feat_extr(z) and h)
            dropout=dropout,
        )

        # Define the output network - computes outcome prediction given predicted cluster assignments
        self.predictor = MLP(
            input_size=self.latent_dim,  # input: h
            output_size=self.num_classes,  # output: num_classes-dimensional vector of outcome probabilities
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,
            dropout=dropout,
        )
        self.predictor_output_fn = nn.Softmax(dim=-1)

        # Define feature transformation gate networks
        self.phi_x = MLP(
            input_size=self.input_dims,  # input: x
            output_size=self.latent_dim,  # output: latent_dim-dimensional vector of latent space features
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,  # default bias is True
            dropout=dropout,
        )
        self.phi_x_output_fn = nn.ReLU()

        self.phi_z = MLP(
            input_size=self.latent_dim,  # input: z
            output_size=self.latent_dim,  # output: latent_dim-dimensional vector of latent space features
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,  # default bias is True
            dropout=dropout,
        )
        self.phi_z_output_fn = nn.ReLU()

        # Define Cell Update Gate Functions
        self.cell_update = MLP(
            input_size=self.latent_dim + self.latent_dim + self.latent_dim,  # input: concat(h, phi_x, phi_z)
            output_size=self.latent_dim,  # output: latent_dim-dimensional vector of cell state updates
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,  # default bias is True
            dropout=dropout,
        )
        self.cell_update_output_fn = nn.ReLU()

        # Placeholder for loss function names
        self.__loss_names = ["neg ELBO", "Log Lik", "KL", "Outcome"]


    
    "Define methods to simplify passes through network blocks"
    def _apply_encoder(self, x):
        return self.encoder_output_fn(self.encoder(x))

    def _apply_prior(self, h):
        return self.prior_output_fn(self.prior(h))

    def _apply_decoder(self, c):
        return self.decoder(c)

    def _apply_predictor(self, z):
        return self.predictor_output_fn(self.predictor(z))
    
    def _apply_cell_update(self, x):
        return self.cell_update_output_fn(self.cell_update(x))
    
    def _transform_x(self, x):
        return self.phi_x_output_fn(self.phi_x(x))
    
    def _transform_z(self, z):
        return self.phi_z_output_fn(self.phi_z(z))
    
    def _encoder_pass(self, h, x):
        "Single pass of the encoder to obtain alpha inference param."
        return self._apply_encoder(torch.cat([self._transform_x(x), h], dim=-1))
    
    def _prior_pass(self, h):
        "Single pass of the prior to obtain alpha prior param."
        return self._apply_prior(h)
    
    def _cell_update_pass(self, h, x, z):
        "Single pass of the cell update to obtain updated cell state."
        return self._apply_cell_update(torch.cat([h, self._transform_x(x), self._transform_z(z)], dim=-1))
    
    def _predictor_pass(self, z):
        "Single pass of the predictor to obtain outcome prediction."
        return self._apply_predictor(z)
    
    def _decoder_pass(self, h, z):
        "Single pass of the decoder to obtain generated data."
        return self._apply_decoder(torch.cat([self._transform_z(z), h], dim=-1))



    # Define forward pass over a single batch
    def forward(self, x, y, mode: str = "train"):
        """
        Forward Pass computation for a single pair of batch objects x and y.

        Params:
            - x: pytorch Tensor object of input data with shape (batch_size, max_seq_len, input_size);
            - y: pytorch Tensor object of corresponding outcomes with shape (batch_size, outcome_size).
            - mode: string indicating whether we are training or evaluating (default = 'train'). This parameter controls whether we track objects or not (i.e. just losses).

        Output:
            - loss: pytorch Tensor object of loss value.
            - history: dictionary of relevant training history.

        We iterate over consecutive window time blocks. For the current window block, we use the hidden state obtained at
        the last time window, and we generate mean and variance for generating data at the current window block. We also use the true data 
        to generate latent representations.

        Values are initialized at 0. At the last observed time-step, we compute self.n_fwd_blocks windows of prediction forward. At the last 
        forward block, we use the estimated representations to predict the outcome.

        Loss is computed by summing (all losses averaged over batch):
            a) Log Lik loss (how good we are at generating data), 
            b) KL loss (how close the updated alpha approximates the posterior, and 
            c) outcome loss, taken at the last time step, which indicates whether we are able to predict the outcome correctly. 
        """

        # ========= Define relevant variables ===========
        x, y = x.to(self.device), y.to(self.device)  # move data to device
        assert mode in ["train", "eval"], "Mode has to be either 'train' or 'eval'."

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()


        # Pre-fill the data tensor with zeros if we have a sequence length that is not a multiple of window_num_obvs
        remainder = seq_len % self.window_num_obvs
        if remainder != 0:

            # Pre-pend the input data with zeros to make it a multiple of window_num_obvs
            zeros_append = torch.zeros(batch_size, self.window_num_obvs - remainder, input_size, device=self.device)
            x = torch.cat((zeros_append, x), dim=1)

            # Update seq_len
            seq_len = seq_len + self.window_num_obvs - remainder
            assert (seq_len % self.window_num_obvs == 0)

        # Get number of iterative windows
        num_windows = int(seq_len / self.window_num_obvs)


        # ========== Initialise relevant variables ==========
        past_h = torch.zeros(batch_size, self.latent_dim, device=self.device)
        past_pi = torch.ones(batch_size, self.K, device=self.device) / self.K
        past_z = model_utils.compute_repr_from_clus_assign_prob(past_pi, self.c_means, self.log_c_vars)


        # Initialize ELBO, history tracker and future tracker
        ELBO, Loss_loglik, Loss_kl, Loss_outl = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        history = {
            "loss_loglik": torch.zeros(seq_len, device=self.device),
            "loss_kl": torch.zeros(seq_len, device=self.device),
            "loss_out": 0,
            "pis": torch.zeros(batch_size, seq_len, self.K, device=self.device),
            "zs": torch.zeros(batch_size, seq_len, self.latent_dim, device=self.device),
            "alpha_encs": torch.zeros(batch_size, seq_len, self.K, device=self.device),
            "gen_means": torch.zeros(batch_size, seq_len, self.input_dims, device=self.device),
            "gen_log_vars": torch.zeros(batch_size, seq_len, self.input_dims, device=self.device),
            "y_pred": torch.zeros(batch_size, self.num_classes, device=self.device)
        }
        future = {
            "loss_loglik": torch.zeros(self.n_fwd_blocks * self.window_num_obvs, device=self.device),
            "loss_kl": torch.zeros(self.n_fwd_blocks * self.window_num_obvs, device=self.device),
            "loss_out": 0,
            "pis": torch.zeros(batch_size, self.n_fwd_blocks * self.window_num_obvs, self.K, device=self.device),
            "zs": torch.zeros(batch_size, self.n_fwd_blocks * self.window_num_obvs, self.latent_dim, device=self.device),
            "alpha_encs": torch.zeros(batch_size, self.n_fwd_blocks * self.window_num_obvs, self.K, device=self.device),
            "gen_means": torch.zeros(batch_size, self.n_fwd_blocks * self.window_num_obvs, self.input_dims, device=self.device),
            "gen_log_vars": torch.zeros(batch_size, self.n_fwd_blocks * self.window_num_obvs, self.input_dims, device=self.device),
            "y_pred": torch.zeros(batch_size, self.num_classes, device=self.device)
        }


        # ================== Iterate through time ==================
        """
        Note we iterate over window blocks, and within each block we iterate over individual time steps to compute loss functions.
        There is a past component (data was pre-filled), and a future component (we predict forward self.n_fwd_blocks windows).
        """

        # Iterate OVER PAST
        windows_obvs = range(num_windows)
        for window_id in windows_obvs:

            # Bottom and high indices
            _lower_t_idx = window_id * self.window_num_obvs     # INCLUSIVE 
            _upper_t_idx = (window_id + 1) * self.window_num_obvs    # EXCLUSIVE

            # Generate data for current window block given the existing estimates - decompose into mean and log-variance
            _, _, data_gen_params = self._decoder_pass(h=past_h, z=past_z)
            gen_mean, gen_logvar = torch.chunk(data_gen_params, chunks=2, dim=-1)
        
            try:
                assert torch.isnan(gen_mean).sum() == 0, "ERROR: NaNs in estimated representations"
            except AssertionError as err:
                print(err)
                print("ERROR: NaNs in estimated representations")
                import pdb; pdb.set_trace()

            # Iterate through each time step to estimate updated representations and compute loss terms
            for inner_t_idx, outer_t_idx in enumerate(range(_lower_t_idx, _upper_t_idx)):

                # Access input data at time t
                x_t = x[:, outer_t_idx, :]
                _est_mean, _est_logvar = gen_mean[:, inner_t_idx, :], gen_logvar[:, inner_t_idx, :]

                # Compute alphas of prior and encoder networks given previous cell estimate and current input data
                alpha_prior = self._prior_pass(h=past_h)
                alpha_enc = self._encoder_pass(h=past_h, x=x_t)

                # Sample cluster probs given Dirichlet parameter, and estimate representations based on mixture of Gaussian model
                past_pi = model_utils.sample_dir(alpha=alpha_enc)
                past_z = model_utils.compute_repr_from_clus_assign_prob(pi_assign=past_pi, c_means=self.c_means, log_c_vars=self.log_c_vars)

                # Update Cell State
                past_h = self._cell_update_pass(h=past_h, x=x_t, z=past_z)

                # Compute Loss Terms
                log_lik = LM_utils.torch_log_Gauss_likelihood(x=x_t, mu=_est_mean, logvar=_est_logvar, device=self.device)
                kl_div = LM_utils.torch_dir_kl_div(a1=alpha_enc, a2=alpha_prior)
                ELBO += log_lik - kl_div
                Loss_kl += kl_div
                Loss_loglik += log_lik

                # Append objects to history tracker if we are in evaluation mode
                if mode == "eval":
                    history["pis"][:, outer_t_idx, :] = past_pi
                    history["zs"][:, outer_t_idx, :] = past_z
                    history["alpha_encs"][:, outer_t_idx, :] = alpha_enc

                    # Append generate mean, generate var
                    history["gen_means"][:, outer_t_idx, :] = _est_mean
                    history["gen_log_vars"][:, outer_t_idx, :] = _est_logvar

                    # Add to loss tracker
                    history["loss_kl"][outer_t_idx] += kl_div
                    history["loss_loglik"][outer_t_idx] += log_lik

        # Iterate OVER FUTURE windows
        fut_h, fut_pi, fut_z = past_h, past_pi, past_z

        windows_ftr = range(self.n_fwd_blocks)
        for window_id in windows_ftr:

            # Bottom and high indices
            _lower_t_idx = window_id * self.window_num_obvs     # INCLUSIVE 
            _upper_t_idx = (window_id + 1) * self.window_num_obvs    # EXCLUSIVE

            # Generate data for current window block given the existing estimates - decompose into mean and log-variance
            _, _, data_gen_params = self._decoder_pass(h=fut_h, z=fut_z)
            gen_mean, gen_logvar = torch.chunk(data_gen_params, chunks=2, dim=-1)


            # Iterate through each time step to estimate updated representations and compute loss terms
            for inner_t_idx, outer_t_idx in enumerate(range(_lower_t_idx, _upper_t_idx)):

                # Generate data at time t based on generate parameters mean and log-variance
                _est_mean, _est_logvar = gen_mean[:, inner_t_idx, :], gen_logvar[:, inner_t_idx, :]
                gen_x_t = model_utils.generate_diagonal_multivariate_normal_samples(
                    mu=_est_mean, 
                    logvar=_est_logvar
                )

                # Compute alphas of prior and encoder networks
                alpha_prior = self._prior_pass(h=fut_h)
                alpha_enc = self._encoder_pass(h=fut_h, x=gen_x_t)

                # Sample cluster distribution from alpha_enc, and estimate samples from clusters based on mixture of Gaussian model
                fut_pi = model_utils.sample_dir(alpha=alpha_enc)
                fut_z = model_utils.compute_repr_from_clus_assign_prob(pi_assign=fut_pi, c_means=self.c_means, log_c_vars=self.log_c_vars)

                # Update Cell State
                fut_h = self._cell_update_pass(h=fut_h, x=gen_x_t, z=fut_z)

                # Compute Loss Terms
                future_loglik = LM_utils.torch_log_Gauss_likelihood(x=gen_x_t, mu=_est_mean, logvar=_est_logvar, device=self.device)
                future_kl = LM_utils.torch_dir_kl_div(a1=alpha_enc, a2=alpha_prior)

                # Append objects to future tracker if we are in evaluation mode
                if mode == "eval":
                    future["pis"][:, outer_t_idx, :] = fut_pi
                    future["zs"][:, outer_t_idx, :] = fut_z
                    future["alpha_encs"][:, outer_t_idx, :] = alpha_enc

                    # Append generate mean, generate var
                    future["gen_means"][:, outer_t_idx, :] = _est_mean
                    future["gen_log_vars"][:, outer_t_idx, :] = _est_logvar


                    # Add to loss tracker
                    future["loss_kl"][outer_t_idx] += future_kl
                    future["loss_loglik"][outer_t_idx] += future_loglik


        # Once everything is computed, make predictions on outcome and compute loss
        y_pred = self._predictor_pass(z=fut_z)
        pred_loss = LM_utils.torch_CatCE(y_true=y, y_pred=y_pred)
        ELBO += pred_loss
        Loss_outl += pred_loss

        # Append objects to trackers if we are in evaluation mode
        if mode == "eval":
            future["y_pred"] = y_pred
            history["y_pred"] = y_pred

            # Append Loss to tracker
            history["loss_out"] = Loss_outl
            future["loss_out"] = Loss_outl

        # Convert ELBO to -1 * ELBO since we want to maximize ELBO
        loss = (-1) * ELBO

        return loss, Loss_loglik, Loss_kl, Loss_outl, history, future  # want to maximize ELBO, so return - ELBO


    # Define method to train model on given data
    def fit(self,
        train_data, val_data,
        lr: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 100
    ):
        """
        Method to train model given train and validation data, as well as training parameters.

        Params:
            - train_data: Tuple (X, y) of training data, with shape (N, T, D) and (N, O), respectively.
            - val_data: Tuple (X, y) of validation data. If None or (None, None), then no validation is performed.
            - lr: learning rate for optimizer.
            - batch_size: batch size for training.
            - num_epochs: number of epochs to train for.

        Outputs:
            - loss: final loss value.
            - history: dictionary with training history, including each loss component.
        """

        # ================== INITIALIZATION ==================
        epoch_range = range(1, num_epochs + 1)
        self._train_loss_tracker = {epoch: [] for epoch in epoch_range}
        self._val_loss_tracker = {epoch: [] for epoch in epoch_range}
        self._val_exp_obj_tracker = {epoch: {} for epoch in epoch_range}
        self._val_supervised_scores = {epoch: {} for epoch in epoch_range}
        self._val_supervised_scores_lachiche_algo = {epoch: {} for epoch in epoch_range}
        self._val_clus_quality_scores = {epoch: {} for epoch in epoch_range}
        self._val_clus_assign_scores = {epoch: {} for epoch in epoch_range}

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs


        # ================== DATA PREPARATION ==================

        # Unpack data and make data loaders
        X_train, y_train = train_data

        # Prepare data for training
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer and Logging of weights
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        wandb.watch(models=self, log="all", log_freq=1, idx=self.K_fold_idx)           

        # Printing Message
        print("Printing Losses loss, Log Lik, KL, Outl")

        # ================== TRAINING-VALIDATION LOOP ==================
        for epoch in range(1, num_epochs + 1):

            # Set model to train mode 
            self.train()

            # Initialize loss trackers
            tr_loss = torch.zeros(1, device=self.device)
            tr_loglik = torch.zeros(1, device=self.device)
            tr_kl = torch.zeros(1, device=self.device)
            tr_outl = torch.zeros(1, device=self.device)

            # Iterate through batches
            for batch_id, (x, y) in enumerate(train_loader):

                # Zero out gradients for each batch
                optimizer.zero_grad()

                # Compute Loss for single model pass
                batch_loss, batch_loglik, batch_kl, batch_outl, train_history, train_future = self.forward(x, y, mode="train")

                # Back-propagate loss and update weights
                batch_loss.backward()
                optimizer.step()

                # Print message of loss per batch, which is re-setted at the end of each epoch
                _perc_completed = 100.0 * batch_id / len(train_loader)
                print(f"Train epoch: {epoch}   [{batch_loss.item():.2f} - {_perc_completed}]", end="\r")

                # Add to training losses - we can add mean of batches and take mean over training loss dividing by number of batches
                tr_loss += batch_loss.item()
                tr_loglik += batch_loglik.item()
                tr_kl += batch_kl.item()
                tr_outl += batch_outl.item()  

            # Take average over all batches in the train data
            ep_loss, ep_loglik, ep_kl, ep_outl = tr_loss / len(train_loader), tr_loglik / len(train_loader), tr_kl / len(train_loader), tr_outl / len(train_loader)


            # Print Message at the end of each epoch with the main loss and all auxiliary loss functions
            print(
                "Epoch {} ({}) :  {:.2f} - {:.2f} - {:.2f} - {:.2f}".format(
                    epoch, self.num_epochs, ep_loss.item(), ep_loglik.item(), ep_kl.item(), ep_outl.item()
                ),
                end="     ",
            )            

            # ============== LOGGING ==============
            self._train_loss_tracker[epoch] = [ep_loss, ep_loglik, ep_kl, ep_outl]


            # ============= VALIDATION =============
            if val_data is not None:

                # Unpack and prepare data
                X_val, y_val = val_data
                val_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                    batch_size=X_val.shape[0],
                    shuffle=False,
                )

                # Set model to evaluation mode
                self.eval()
                with torch.inference_mode():
                    for X, y in val_loader:

                        # Run model once with a single forward pass
                        val_loss, val_loglik, val_kl, val_outl, val_history, val_future = self.forward(X, y, mode="eval")

                        # print message
                        print(
                            "Val {} ({}): {:.2f} - {:.2f} - {:.2f} - {:.2f}".format(
                                epoch, self.num_epochs, val_loss.item(), val_loglik.item(), val_kl.item(), val_outl.item()
                            ),
                            end="\n",
                        )

                        # ============= SCORE COMPUTATION =============

                        # Unpack outputs
                        val_hist_y_pred = val_history["y_pred"]
                        val_hist_pis = val_history["pis"]
                        val_hist_zs = val_history["zs"]

                        # # Compute performance scores
                        # history_sup_scores = metrics.get_multiclass_sup_scores(y_true=y, y_pred=val_hist_y_pred, run_weight_algo=False)
                        # history_sup_scores_lachiche_algo = metrics.get_multiclass_sup_scores(y_true=y, y_pred=val_hist_y_pred, run_weight_algo=True)
                        # history_clus_assign_scores = metrics.get_clus_label_match_scores(y_true=y, clus_pred=val_hist_pis)
                        # history_clus_qual_scores = metrics.get_unsup_scores(X=val_hist_zs, clus_pred=val_hist_pis, seed=self.seed)
                        


                        # Do the same for future computations
                        val_fut_y_pred = val_future["y_pred"]
                        val_fut_pis = val_future["pis"]
                        val_fut_zs = val_future["zs"]

                        # # Compute performance scores
                        # future_sup_scores = metrics.get_multiclass_sup_scores(y_true=y, y_pred=val_fut_y_pred, run_weight_algo=False)
                        # future_sup_scores_lachiche_algo = metrics.get_multiclass_sup_scores(y_true=y, y_pred=val_fut_y_pred, run_weight_algo=True)
                        # future_clus_label_scores = metrics.get_clus_label_match_scores(y_true=y, clus_pred=val_fut_pis)
                        # future_clus_qual_scores = metrics.get_unsup_scores(X=val_fut_zs, clus_pred=val_fut_pis, seed=self.seed)

                        # ================== LOGGING ==================

                        # Log history related components
                        self._val_loss_tracker[epoch] = [val_loss, val_loglik, val_kl, val_outl]
                        # self._val_supervised_scores[epoch]["history"] = history_sup_scores
                        # self._val_supervised_scores_lachiche_algo[epoch]["history"] = history_sup_scores_lachiche_algo
                        # self._val_clus_quality_scores[epoch]["history"] = history_clus_qual_scores
                        # self._val_clus_assign_scores[epoch]["history"] = history_clus_assign_scores
                        self._val_exp_obj_tracker[epoch]["history"] = {
                            "y_pred": val_hist_y_pred,
                            "pis": val_hist_pis,
                            "zs": val_hist_zs,
                        }

                        # Log future related components
                        # self._val_supervised_scores[epoch]["future"] = future_sup_scores
                        # self._val_supervised_scores_lachiche_algo[epoch]["future"] = future_sup_scores_lachiche_algo
                        # self._val_clus_quality_scores[epoch]["future"] = future_clus_qual_scores
                        # self._val_clus_assign_scores[epoch]["future"] = future_clus_label_scores
                        self._val_exp_obj_tracker[epoch]["future"] = {
                            "y_pred": val_fut_y_pred,
                            "pis": val_fut_pis,
                            "zs": val_fut_zs,
                        }


        # Log Training and Validation Objects
        self._val_exp_obj_tracker["train_data"] = train_data # type: ignore
        self._val_exp_obj_tracker["val_data"] = val_data # type: ignore
        self._val_exp_obj_tracker["fit_params"] = { # type: ignore
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "lr": lr
        }
        self._val_exp_obj_tracker["model_params"] = self.state_dict() # type: ignore


    def predict(self, X, y):
        """Similar to forward method, but focus on inner computations and tracking objects for the model."""

        # ================== INITIALIZATION ==================
        self._test_loss = {"test": []}
        self._test_exp_obj = {"test": {}}
        self._test_supervised_scores = {"test": {}}
        self._test_supervised_scores_lachiche_algo = {"test": {}}
        self._test_clus_quality_scores = {"test": {}}
        self._test_clus_assign_scores = {"test": {}}


        # ================== DATA PREPARATION ==================

        # Prepare Data
        test_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        test_loader = DataLoader(test_data, batch_size=X.shape[0], shuffle=False)

        # # Initialize logger objects
        # wandb.watch(models=self, log="all", log_freq=1, idx=self.K_fold_idx)           

        # Set model to evaluation mode
        self.eval()

        # Apply forward prediction
        output_dir = {}
        with torch.inference_mode():
            for X, y in test_loader:

                # Pass data through model
                test_loss, test_loglik, test_kl, test_outl, test_history, test_future = self.forward(X, y, mode="eval")

                # ============= SCORE COMPUTATION =============

                # Unpack outputs
                test_hist_y_pred = test_history["y_pred"]
                test_hist_pis = test_history["pis"]
                test_hist_zs = test_history["zs"]

                # Compute performance scores
                # history_sup_scores = metrics.get_multiclass_sup_scores(y_true=y, y_pred=test_hist_y_pred, run_weight_algo=False)
                # history_sup_scores_lachiche_algo = metrics.get_multiclass_sup_scores(y_true=y, y_pred=test_hist_y_pred, run_weight_algo=True)
                # history_clus_label_scores = metrics.get_clus_label_match_scores(y_true=y, clus_pred=test_hist_pis)
                # history_clus_qual_scores = metrics.get_unsup_scores(X=test_hist_zs, clus_pred=test_hist_pis, seed=self.seed)

                
                # Do the same for future computations
                test_fut_y_pred = test_future["y_pred"]
                test_fut_pis = test_future["pis"]
                test_fut_zs = test_future["zs"]

                # Compute performance scores
                # future_sup_scores = metrics.get_multiclass_sup_scores(y_true=y, y_pred=test_fut_y_pred, run_weight_algo=False)
                # future_sup_scores_lachiche_algo = metrics.get_multiclass_sup_scores(y_true=y, y_pred=test_fut_y_pred, run_weight_algo=True)
                # future_clus_label_scores = metrics.get_clus_label_match_scores(y_true=y, clus_pred=test_fut_pis)
                # future_clus_qual_scores = metrics.get_unsup_scores(X=test_fut_zs, clus_pred=test_fut_pis, seed=self.seed)


                # ================== LOGGING ==================
                # Log losses
                self._test_loss["test"] = [test_loss, test_loglik, test_kl, test_outl]

                # Log past objects
                # self._test_supervised_scores["test"]["history"] = history_sup_scores
                # self._test_supervised_scores_lachiche_algo["test"]["history"] = history_sup_scores_lachiche_algo
                # self._test_clus_quality_scores["test"]["history"] = history_clus_qual_scores
                # self._test_clus_assign_scores["test"]["history"] = history_clus_label_scores
                self._test_exp_obj["test"]["history"] = { # type: ignore
                    "y_pred": test_hist_y_pred,
                    "pis": test_hist_pis,
                    "zs": test_hist_zs
                }

                # Log future objects
                # self._test_supervised_scores["test"]["future"]= future_sup_scores
                # self._test_supervised_scores_lachiche_algo["test"]["future"] = future_sup_scores_lachiche_algo
                # self._test_clus_quality_scores["test"]["future"] = future_clus_qual_scores
                # self._test_clus_assign_scores["test"]["future"] = future_clus_label_scores
                self._test_exp_obj["test"]["future"] = { # type: ignore
                    "y_pred": test_fut_y_pred,
                    "pis": test_fut_pis,
                    "zs": test_fut_zs
                }


        # Log model params, model and other objects
        self._test_exp_obj["test"]["test_data"] = (X, y) # type: ignore
        self._test_exp_obj["test"]["clus_params"] = { # type: ignore
            "c_means": self.c_means,
            "log_c_vars": self.log_c_vars
        }
        self._test_exp_obj["test"]["model_params"] = self.state_dict() # type: ignore
        self._test_exp_obj["test"]["outputs_history"] = test_history  # type: ignore
        self._test_exp_obj["test"]["outputs_future"] = test_future # type: ignore
        
        # Get output
        output_dir = self._test_exp_obj["test"] 

        return output_dir
    
    def log_training_files(self, class_names: List = []):
        """Log training and validation losses to csv file."""

        # Initialize directory for saving
        main_dir = "exps/DirVRNN/"
        exp_save_dir = log_utils.log_new_run_dir(save_dir=main_dir, K_fold_idx=self.K_fold_idx)

        # ================== LOG LOSSES ==================

        # Get Loss CSV Header
        train_loss_names = [f"train_{loss}" for loss in self.__loss_names]
        val_loss_names = [f"val_{loss}" for loss in self.__loss_names]
        loss_csv_header = ["epoch"] + train_loss_names + val_loss_names

        # Create CSV file if it doesn't exist
        loss_csv_path = f"{exp_save_dir}/losses.csv"
        log_utils.make_csv_if_not_exist(save_path=loss_csv_path, header=loss_csv_header)

        # Log to CSV iterating over epochs
        for epoch in range(1, self.num_epochs + 1):

            # Access items
            train_loss_values = [loss.item() for loss in self._train_loss_tracker[epoch]]
            val_loss_values = [loss.item() for loss in self._val_loss_tracker[epoch]]

            # Log to csv
            row = [epoch] + train_loss_values + val_loss_values
            log_utils.write_to_csv_if_exists(save_path=loss_csv_path, data=row)

        
        # ================== LOG Supervised Scores ==================
        if class_names == []:
            outcome_names = ["C_{i}" for i in range(self.num_classes)]
        else:
            outcome_names = class_names

        # Get Score CSV Headers
        val_acc_csv_header = ["epoch"] + ["val_acc"]
        val_multiclass_csv_header = ["epoch"] + outcome_names
        val_cm_csv_header = ["epoch"] + [f"Tr_{i}_Pr_{j}" for j in outcome_names for i in outcome_names]

        # Save scores for each of "history" and "future" objects
        for time_period in ["history", "future"]:
                
            # Create CSV files if they don't exist
            val_acc_csv_path = f"{exp_save_dir}/val_acc_{time_period}.csv"
            val_acc_lachiche_algo_csv_path = f"{exp_save_dir}/val_acc_lachiche_algo_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=val_acc_csv_path, header=val_acc_csv_header)
            log_utils.make_csv_if_not_exist(save_path=val_acc_lachiche_algo_csv_path, header=val_acc_csv_header)
            
            val_cm_csv_path = f"{exp_save_dir}/val_cm_{time_period}.csv"
            val_cm_lachiche_algo_csv_path = f"{exp_save_dir}/val_cm_lachiche_algo_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=val_cm_csv_path, header=val_cm_csv_header)
            log_utils.make_csv_if_not_exist(save_path=val_cm_lachiche_algo_csv_path, header=val_cm_csv_header)

            # Iterate over target metrics
            for metric in log_utils.CLASS_METRICS:
                val_metric_csv_path = f"{exp_save_dir}/val_{metric}_{time_period}.csv"
                val_metric_lachiche_algo_csv_path = f"{exp_save_dir}/val_{metric}_lachiche_algo_{time_period}.csv"
                log_utils.make_csv_if_not_exist(save_path=val_metric_csv_path, header=val_multiclass_csv_header)
                log_utils.make_csv_if_not_exist(save_path=val_metric_lachiche_algo_csv_path, header=val_multiclass_csv_header)

            # Log to CSV iterating over epochs
            for epoch in range(1, self.num_epochs + 1):
                
                # Access items
                val_acc_row_to_write = [epoch] + [self._val_supervised_scores[epoch][time_period]["accuracy"]]
                log_utils.write_to_csv_if_exists(save_path=val_acc_csv_path, data=val_acc_row_to_write)

                val_cm_row_to_write = [epoch] + self._val_supervised_scores[epoch][time_period]["confusion_matrix"].flatten().tolist()
                log_utils.write_to_csv_if_exists(save_path=val_cm_csv_path, data=val_cm_row_to_write)

                # Iterate over target multiclass metrics
                for metric in log_utils.CLASS_METRICS:
                    val_metric_row_to_write = [epoch] + self._val_supervised_scores[epoch][time_period][metric]
                    log_utils.write_to_csv_if_exists(save_path=f"{exp_save_dir}/val_{metric}_{time_period}.csv", data=val_metric_row_to_write)

                # Do the same for Lachiche Algo scores
                val_acc_lachiche_algo_row_to_write = [epoch] + [self._val_supervised_scores_lachiche_algo[epoch][time_period]["accuracy"]]
                log_utils.write_to_csv_if_exists(save_path=val_acc_lachiche_algo_csv_path, data=val_acc_lachiche_algo_row_to_write)

                val_cm_lachiche_algo_row_to_write = [epoch] + self._val_supervised_scores_lachiche_algo[epoch][time_period]["confusion_matrix"].flatten().tolist()
                log_utils.write_to_csv_if_exists(save_path=val_cm_lachiche_algo_csv_path, data=val_cm_lachiche_algo_row_to_write)

                # Iterate over target multiclass metrics
                for metric in log_utils.CLASS_METRICS:
                    val_metric_lachiche_algo_row_to_write = [epoch] + self._val_supervised_scores_lachiche_algo[epoch][time_period][metric]
                    log_utils.write_to_csv_if_exists(save_path=f"{exp_save_dir}/val_{metric}_lachiche_algo_{time_period}.csv", data=val_metric_lachiche_algo_row_to_write)


        # ================== LOG Clustering Scores ==================

        # Get Score CSV Headers
        val_clus_qual_csv_header = ["epoch"] + log_utils.CLUS_QUALITY_METRICS
        val_clus_assign_csv_header = ["epoch"] + log_utils.CLUS_ASSIGN_METRICS

        # Save scores for each of "history" and "future" objects
        for time_period in ["history", "future"]:

            # Create CSV files if they don't exist
            val_clus_qual_csv_path = f"{exp_save_dir}/val_clus_qual_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=val_clus_qual_csv_path, header=val_clus_qual_csv_header)

            val_clus_assign_csv_path = f"{exp_save_dir}/val_clus_assign_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=val_clus_assign_csv_path, header=val_clus_assign_csv_header)

            # Log to CSV iterating over epochs
            for epoch in range(1, self.num_epochs + 1):
                
                # Access items
                val_clus_qual_row_to_write = [epoch] + list(self._val_clus_quality_scores[epoch][time_period].values())
                log_utils.write_to_csv_if_exists(save_path=val_clus_qual_csv_path, data=val_clus_qual_row_to_write)

                val_clus_assign_row_to_write = [epoch] + list(self._val_clus_assign_scores[epoch][time_period].values())
                log_utils.write_to_csv_if_exists(save_path=val_clus_assign_csv_path, data=val_clus_assign_row_to_write)

        # Print Message
        print(f"Saved experiment training logs to {exp_save_dir}")

        return exp_save_dir, outcome_names
    
    def log_prediction_to_files(self, exp_save_dir: str, outcome_names: List = []):
        """
        Logs the predictions of the model to csv files

        Args:
            exp_save_dir (str): The directory to save the experiment to
            outcome_names (List, optional): The names of the outcomes. Defaults to [].
        """

        if outcome_names == []:
            outcome_names = ["C_{i}" for i in range(self.num_classes)]


        # ================== LOG Test Losses ==================

        # Get Loss CSV Headers
        test_loss_csv_header = ["mode"] + self.__loss_names
        test_loss_csv_path = f"{exp_save_dir}/test_losses.csv"
        log_utils.make_csv_if_not_exist(test_loss_csv_path, test_loss_csv_header)

        # Log results to CSV
        test_loss_to_write = ["test"] + [loss.item() for loss in self._test_loss["test"]]
        log_utils.write_to_csv_if_exists(save_path=test_loss_csv_path, data=test_loss_to_write)

        # ================== LOG Supervised Scores ==================

        # Get Score CSV Headers
        test_acc_csv_header = ["mode"] + ["test_acc"]
        test_multiclass_csv_header = ["mode"] + outcome_names
        test_cm_csv_header = ["mode"] + [f"Tr_{i}_Pr_{j}" for j in outcome_names for i in outcome_names]
        
        # Iterate over time period
        for time_period in ["history", "future"]:
            
            # Create CSV files if they don't exist
            test_acc_csv_path = f"{exp_save_dir}/test_acc_{time_period}.csv"
            test_acc_lachiche_algo_csv_path = f"{exp_save_dir}/test_acc_lachiche_algo_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=test_acc_csv_path, header=test_acc_csv_header)
            log_utils.make_csv_if_not_exist(save_path=test_acc_lachiche_algo_csv_path, header=test_acc_csv_header)

            test_cm_csv_path = f"{exp_save_dir}/test_cm_{time_period}.csv"
            test_cm_lachiche_algo_csv_path = f"{exp_save_dir}/test_cm_lachiche_algo_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=test_cm_csv_path, header=test_cm_csv_header)
            log_utils.make_csv_if_not_exist(save_path=test_cm_lachiche_algo_csv_path, header=test_cm_csv_header)

            # Iterate over target metrics
            for metric in log_utils.CLASS_METRICS:
                test_metric_csv_path = f"{exp_save_dir}/test_{metric}_{time_period}.csv"
                test_metric_lachiche_algo_csv_path = f"{exp_save_dir}/test_{metric}_lachiche_algo_{time_period}.csv"
                log_utils.make_csv_if_not_exist(save_path=test_metric_csv_path, header=test_multiclass_csv_header)
                log_utils.make_csv_if_not_exist(save_path=test_metric_lachiche_algo_csv_path, header=test_multiclass_csv_header)

            # Log to CSV
            test_acc_row_to_write = ["test"] + [self._test_supervised_scores["test"][time_period]["accuracy"]]
            log_utils.write_to_csv_if_exists(save_path=test_acc_csv_path, data=test_acc_row_to_write)

            test_cm_row_to_write = ["test"] + self._test_supervised_scores["test"][time_period]["confusion_matrix"].flatten().tolist()
            log_utils.write_to_csv_if_exists(save_path=test_cm_csv_path, data=test_cm_row_to_write)

            # Iterate over target multiclass metrics
            for metric in log_utils.CLASS_METRICS:
                test_metric_row_to_write = ["test"] + self._test_supervised_scores["test"][time_period][metric]
                log_utils.write_to_csv_if_exists(save_path=f"{exp_save_dir}/test_{metric}_{time_period}.csv", data=test_metric_row_to_write)

            # Do the same for Lachiche Algorithm
            test_acc_lachiche_algo_row_to_write = ["test"] + [self._test_supervised_scores_lachiche_algo["test"][time_period]["accuracy"]]
            log_utils.write_to_csv_if_exists(save_path=test_acc_lachiche_algo_csv_path, data=test_acc_lachiche_algo_row_to_write)

            test_cm_lachiche_algo_row_to_write = ["test"] + self._test_supervised_scores_lachiche_algo["test"][time_period]["confusion_matrix"].flatten().tolist()
            log_utils.write_to_csv_if_exists(save_path=test_cm_lachiche_algo_csv_path, data=test_cm_lachiche_algo_row_to_write)

            # Iterate over target multiclass metrics
            for metric in log_utils.CLASS_METRICS:
                test_metric_lachiche_algo_row_to_write = ["test"] + self._test_supervised_scores_lachiche_algo["test"][time_period][metric]
                log_utils.write_to_csv_if_exists(save_path=f"{exp_save_dir}/test_{metric}_lachiche_algo_{time_period}.csv", data=test_metric_lachiche_algo_row_to_write)

        # ================== LOG Clustering Scores ==================

        # Get Score CSV Headers
        test_clus_qual_csv_header = ["mode"] + log_utils.CLUS_QUALITY_METRICS
        test_clus_assign_csv_header = ["mode"] + log_utils.CLUS_ASSIGN_METRICS

        # Save scores for each of "history" and "future" objects
        for time_period in ["history", "future"]:

            # Create CSV files if they don't exist
            test_clus_qual_csv_path = f"{exp_save_dir}/test_clus_qual_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=test_clus_qual_csv_path, header=test_clus_qual_csv_header)

            test_clus_assign_csv_path = f"{exp_save_dir}/test_clus_assign_{time_period}.csv"
            log_utils.make_csv_if_not_exist(save_path=test_clus_assign_csv_path, header=test_clus_assign_csv_header)

            # Log to CSV
            test_clus_qual_row_to_write = ["test"] + list(self._test_clus_quality_scores["test"][time_period].values())
            log_utils.write_to_csv_if_exists(save_path=test_clus_qual_csv_path, data=test_clus_qual_row_to_write)

            test_clus_assign_row_to_write = ["test"] + list(self._test_clus_assign_scores["test"][time_period].values())
            log_utils.write_to_csv_if_exists(save_path=test_clus_assign_csv_path, data=test_clus_assign_row_to_write)#

        
        # ================== LOG OBJECTS ==================
        
        # Save test results to pickle file
        test_results = self._test_exp_obj
        test_results_path = f"{exp_save_dir}/test_results.pkl"

        with open(test_results_path, "wb") as f:
            pickle.dump(test_results, f)

        
        # Print Message
        print(f"Saved experiment test logs to {exp_save_dir}")
            
    
    def log_model(self, save_dir: str, objects_to_log: dict = {}, run_info: dict = {}):
        """
        Log model objects. Useful for reproducibility.

        Args:
            save_dir (str): Directory to save the objects.
            objects_to_log (dict): Dictionary containing additional objects to log.
            run_info (dict): Dictionary containing information about the run.
        
        Returns:
            None, saves the objects.
        """

        # Make directory if not exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Then save model configuration and parameters
        models = {
            "input_shape": self.input_shape,
            "output_dim": self.output_dim,
            "model_name": self.model_name,
            "num_feats": self.num_feats,
            "model_params": self.state_dict()
        }
        
        with open(save_dir + "models.pkl", "wb") as f:
            pickle.dump(models, f)

        # Save additional objects
        if objects_to_log != {}:
            with open(save_dir + "data_objects.pkl", "wb") as f:
                pickle.dump(objects_to_log, f)
    
        # Save run information
        if run_info != {}:
            with open(save_dir + "run_info.pkl", "wb") as f:
                pickle.dump(run_info, f)
# endregion