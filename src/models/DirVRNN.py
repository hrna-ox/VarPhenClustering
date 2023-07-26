"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file defines the main proposed model, and how to train the model
"""

# ============= IMPORT LIBRARIES ==============
import csv
import os
import pickle
import torch
import torch.nn as nn
import wandb

from typing import Tuple, Dict, Union, List
from torch.utils.data import DataLoader, TensorDataset

import src.models.loss_functions as LM_utils
from src.models.deep_learning_base_classes import MLP, LSTM_Dec_v1, LSTM_Dec_v2

import src.models.model_utils as model_utils
from src.models.logging_utils import DLLogger as logger
import src.models.metrics as metrics


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
        h = torch.zeros(batch_size, self.latent_dim, device=self.device)
        est_pi = torch.ones(batch_size, self.K, device=self.device) / self.K
        est_z = model_utils.compute_repr_from_clus_assign_prob(est_pi, self.c_means, self.log_c_vars)


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
            _, _, data_gen_params = self._decoder_pass(h=h, z=est_z)
            gen_mean, gen_logvar = torch.chunk(data_gen_params, chunks=2, dim=-1)
        
        
            # Iterate through each time step to estimate updated representations and compute loss terms
            for inner_t_idx, outer_t_idx in enumerate(range(_lower_t_idx, _upper_t_idx)):

                # Access input data at time t
                x_t = x[:, outer_t_idx, :]
                _est_mean, _est_logvar = gen_mean[:, inner_t_idx, :], gen_logvar[:, inner_t_idx, :]

                # Compute alphas of prior and encoder networks given previous cell estimate and current input data
                alpha_prior = self._prior_pass(h=h)
                alpha_enc = self._encoder_pass(h=h, x=x_t)

                # Sample cluster probs given Dirichlet parameter, and estimate representations based on mixture of Gaussian model
                est_pi = model_utils.sample_dir(alpha=alpha_enc)
                est_z = model_utils.compute_repr_from_clus_assign_prob(pi_assign=est_pi, c_means=self.c_means, log_c_vars=self.log_c_vars)

                # Update Cell State
                h = self._cell_update_pass(h=h, x=x_t, z=est_z)

                # Compute Loss Terms
                log_lik = LM_utils.torch_log_Gauss_likelihood(x=x_t, mu=_est_mean, logvar=_est_logvar, device=self.device)
                kl_div = LM_utils.torch_dir_kl_div(a1=alpha_enc, a2=alpha_prior)
                ELBO += log_lik - kl_div
                Loss_kl += kl_div
                Loss_loglik += log_lik

                # Append objects to history tracker if we are in evaluation mode
                if mode == "eval":
                    history["pis"][:, outer_t_idx, :] = est_pi
                    history["zs"][:, outer_t_idx, :] = est_z
                    history["alpha_encs"][:, outer_t_idx, :] = alpha_enc

                    # Append generate mean, generate var
                    history["gen_means"][:, outer_t_idx, :] = _est_mean
                    history["gen_log_vars"][:, outer_t_idx, :] = _est_logvar

                    # Add to loss tracker
                    history["loss_kl"][outer_t_idx] += kl_div
                    history["loss_loglik"][outer_t_idx] += log_lik



        # Iterate OVER FUTURE windows
        windows_ftr = range(self.n_fwd_blocks)
        for window_id in windows_ftr:

            # Bottom and high indices
            _lower_t_idx = window_id * self.window_num_obvs     # INCLUSIVE 
            _upper_t_idx = (window_id + 1) * self.window_num_obvs    # EXCLUSIVE

            # Generate data for current window block given the existing estimates - decompose into mean and log-variance
            _, _, data_gen_params = self._decoder_pass(h=h, z=est_z)
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
                alpha_prior = self._prior_pass(h=h)
                alpha_enc = self._encoder_pass(h=h, x=gen_x_t)

                # Sample cluster distribution from alpha_enc, and estimate samples from clusters based on mixture of Gaussian model
                est_pi = model_utils.sample_dir(alpha=alpha_enc)
                est_z = model_utils.compute_repr_from_clus_assign_prob(pi_assign=est_pi, c_means=self.c_means, log_c_vars=self.log_c_vars)

                # Update Cell State
                h = self._cell_update_pass(h=h, x=gen_x_t, z=est_z)

                # Compute Loss Terms
                future_loglik = LM_utils.torch_log_Gauss_likelihood(x=gen_x_t, mu=_est_mean, logvar=_est_logvar, device=self.device)
                future_kl = LM_utils.torch_dir_kl_div(a1=alpha_enc, a2=alpha_prior)

                # Append objects to future tracker if we are in evaluation mode
                if mode == "eval":
                    future["pis"][:, outer_t_idx, :] = est_pi
                    future["zs"][:, outer_t_idx, :] = est_z
                    future["alpha_encs"][:, outer_t_idx, :] = alpha_enc

                    # Append generate mean, generate var
                    future["gen_means"][:, outer_t_idx, :] = _est_mean
                    future["gen_log_vars"][:, outer_t_idx, :] = _est_logvar


                    # Add to loss tracker
                    future["loss_kl"][outer_t_idx] += future_kl
                    future["loss_loglik"][outer_t_idx] += future_loglik


        # Once everything is computed, make predictions on outcome and compute loss
        y_pred = self._predictor_pass(z=est_z)
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

        return (-1) * ELBO, Loss_loglik, Loss_kl, Loss_outl, history, future  # want to maximize ELBO, so return - ELBO


    # Define method to train model on given data
    def fit(self,
        train_data, val_data,
        lr: float = 0.001,
        batch_size: int = 32,
        num_epochs: int = 100,
        save_params: Union[Dict, None] = None
    ):
        """
        Method to train model given train and validation data, as well as training parameters.

        Params:
            - train_data: Tuple (X, y) of training data, with shape (N, T, D) and (N, O), respectively.
            - val_data: Tuple (X, y) of validation data. If None or (None, None), then no validation is performed.
            - lr: learning rate for optimizer.
            - batch_size: batch size for training.
            - num_epochs: number of epochs to train for.
            - save_params: dictionary of parameters for saving. See src.models.logging_utils for more details. Pertains data saving. If None, then no saving is performed. Includes:
                - K_fold_idx: index of K-fold cross-validation.
                - class_names: list of class names.

        Outputs:
            - loss: final loss value.
            - history: dictionary with training history, including each loss component.
        """

        # ================== DATA PREPARATION ==================

        # Unpack Save Parameters
        if save_params is not None:
            class_names = save_params["class_names"]
            K_fold_idx = save_params["K_fold_idx"]
        
        else:
            class_names = [f"class_{i}" for i in range(self.num_classes)]
            K_fold_idx = 1


        # Unpack data and make data loaders
        X_train, y_train = train_data

        # Prepare data for training
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer and Logging of weights
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        wandb.watch(models=self, log="all", log_freq=1, idx=K_fold_idx)

        # dictionary to save objects for logging
        _exp_objects_saver = {}                

        # Initialize logger object
        exp_save_dir = f"exps/DirVRNN"
        log = logger(save_dir=exp_save_dir, class_names=class_names, K_fold_idx=K_fold_idx, loss_headers=["loss", "loss_kl", "loss_loglik", "loss_out"])
        log._init_exp_fds(mode="train")

        if val_data is not None:
            log._init_exp_fds(mode="val")



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
                batch_loss, batch_loglik, batch_kl, batch_outl, train_history, train_forward = self.forward(x, y, mode="train")

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
                "Train {} ({:.0f}%):  [loss {:.2f} - loglik {:.2f} - kl {:.2f} - outl {:.2f}]".format(
                    epoch, 100, ep_loss, ep_loglik, ep_kl, ep_outl
                ),
                end="     ",
            )


            # ============== LOGGING ==============
            log.log_dirvrnn_losses(losses=[ep_loss, ep_kl, ep_loglik, ep_outl], epoch=epoch, mode="train")


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
                            "Val {} ({:.0f}%):  [loss {:.2f} - loglik {:.2f} - kl {:.2f} - outl {:.2f}]".format(
                                epoch, 100, val_loss.item(), val_loglik.item(), val_kl.item(), val_outl.item()
                            ),
                            end="     ",
                        )

                        # ============= SCORE COMPUTATION =============

                        # Unpack outputs
                        y_pred = val_history["y_pred"]
                        clus_pred = val_history["pis"]
                        z_est = val_history["zs"]

                        # Compute performance scores
                        history_sup_scores = metrics.get_sup_scores(y_true=y, y_pred=y_pred, run_weight_algo=False)
                        add_append_path_to_different_objects

                        # ================== LOGGING ==================
                        log.log_dirvrnn_losses(losses=[val_loss, val_kl, val_loglik, val_outl], epoch=epoch, mode="val")


                        
                # logger(model_params=model_params, X=X, y=y,
                #    log=history_objects,
                #    epoch=epoch, mode="val",
                #    outcomes=outcomes, features=features, save_dir=save_dir
                # )

        # At the end of training, save the data and the final outputs
        history_objects["X_train"] = X_train
        history_objects["y_train"] = y_train

        # Save to pickle
        with open(f"{train_fd}/train_outputs.pkl", "wb") as f:
            pickle.dump(history_objects, f)

        # Do the same for validation data
        outputs_val["X_val"] = X_val
        outputs_val["y_val"] = y_val

        # Save to pickle
        with open(f"{val_fd}/val_outputs.pkl", "wb") as f:
            pickle.dump(outputs_val, f)

    def _eval(self, X, y, epoch: int, viz_params: Dict):
        """
        Compute Performance on Val Dataset.

        Params:
            - X: input data of shape (bs, T, input_size)
            - y: outcome data of shape (bs, output_size)
            - epoch: int indicating epoch number
            - viz_params: dictionary containing visualization parameters for better saving. If None, then standard placeholders are used.
                - save_dir: directory to save results
                - fold: fold number
                - features: list of feature names
                - outcomes: list of outcome names
        """
        if X is None or y is None:
            return None, {}

        # Unpack viz params
        val_fd = f"{viz_params['save_dir']}/val"
        features = viz_params["features"]
        outcomes = viz_params["outcomes"]

        # Set model to evaluation mode
        self.eval()

        # Prepare Data
        val_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        val_loader = DataLoader(val_data, batch_size=X.shape[0], shuffle=False)

        # Apply forward prediction
        with torch.inference_mode():
            for X, y in val_loader:
                # Run model once through the
                val_loss, history_objects = self.forward(X, y, mode="eval")

                # Load individual Losses from tracker
                log_lik = torch.sum(history_objects["loss_loglik"])
                kl_div = torch.sum(history_objects["loss_kl"])
                out_l = history_objects["loss_out"]

                # Log Losses
                wandb.log(
                    {
                        "val/epoch": epoch + 1,
                        "val/loss": val_loss,
                        "val/loglik": log_lik,
                        "val/kldiv": kl_div,
                        "val/out_l": out_l,
                    },
                    step=epoch + 1,
                )

                # Print message
                print(
                    "Val epoch {} ({:.0f}%):  [loss {:.2f} - loglik {:.2f} - kl {:.2f} - out {:.2f}]".format(
                        epoch,
                        100,
                        val_loss.item(),
                        log_lik.item(),
                        kl_div.item(),
                        out_l.item(),
                    )
                )

                # Save to CSV file
                with open(f"{val_fd}/val_losses.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            epoch,
                            val_loss.item(),
                            kl_div.item(),
                            log_lik.item(),
                            out_l.item(),
                        ]
                    )

                # Log performance evaluation scores
                logger.logger_sup_scores(
                    y_true=y,
                    y_pred=history_objects["y_pred"],
                    save_dir=val_fd,
                    epoch=epoch,
                    class_names=outcomes,
                )

                # Log clustering evaluation scores

                # Log more complex results
                model_params = {
                    "c_means": self.c_means,
                    "log_c_vars": self.log_c_vars,
                    "seed": self.seed,
                }

                # logger(model_params=model_params, X=X, y=y,
                #    log=history_objects,
                #    epoch=epoch, mode="val",
                #    outcomes=outcomes, features=features, save_dir=save_dir
                # )

                return val_loss, history_objects

    def predict(
        self,
        X,
        y,
        run_config: Union[Dict, None] = None,
        class_names: List = [],
        feat_names: List = [],
        save_dir: Union[str, None] = None,
    ):
        """Similar to forward method, but focus on inner computations and tracking objects for the model."""

        # Set model to evaluation mode
        self.eval()

        # Prepare Data
        test_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        test_loader = DataLoader(test_data, batch_size=X.shape[0], shuffle=False)

        # Apply forward prediction
        with torch.inference_mode():
            for X, y in test_loader:
                # Pass data through model
                _, history_objects = self.forward(X, y)  # Run forward pass

                # Log results
                if save_dir is not None:
                    model_params = {
                        "c_means": self.c_means,
                        "log_c_vars": self.log_c_vars,
                        "seed": self.seed,
                    }

                    logger(
                        model_params=model_params,
                        X=X,
                        y=y,
                        log=history_objects,
                        epoch=0,
                        mode="test",
                        class_names=class_names,
                        feat_names=feat_names,
                        save_dir=save_dir,
                    )

                # Append Test data
                history_objects["X_test"] = X
                history_objects["y_test"] = y
                history_objects["run_config"] = run_config

                return history_objects
# endregion
