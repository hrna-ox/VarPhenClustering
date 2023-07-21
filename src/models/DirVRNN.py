"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Model file to define GC-DaPh class.
"""

# ============= IMPORT LIBRARIES ==============
import csv
import os
import pickle
import torch
import torch.nn as nn
import wandb

import time

from typing import Tuple, Dict, Union, List
from torch.utils.data import DataLoader, TensorDataset

import src.models.loss_functions as LM_utils
from src.models.deep_learning_base_classes import MLP, LSTM_Dec_v1, LSTM_Dec_v2

import src.models.DirVRNN.auxiliary_functions as model_utils
from src.models.DirVRNN.auxiliary_functions import eps
import src.models.DirVRNN.logger as logger



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
    def __init__(self, 
                i_size: int, 
                o_size: int, 
                w_size: int,
                K: int, 
                l_size: int = 10,
                n_fwd_blocks: int = 1,
                gate_hidden_l: int = 2,
                gate_hidden_n: int = 20,
                bias: bool = True,
                dropout: float = 0.0,
                device: str = 'cpu',
                seed: int = 42,
                **kwargs):
        """
        Object initialization.

        Params:
            - i_size: dimensionality of observation data (number of time_series).
            - o_size: dimensionality of outcome data (number of outcomes).
            - w_size: window size over which to update cell state and generate data.
            - K: number of clusters to consider.
            - l_size: dimensionality of latent space (default = 10).
            - n_fwd_blocks: number of forward blocks to predict (int, default=1).
            - gate_hidden_l: number of hidden layers for gate networks (default = 2).
            - gate_hidden_n: number of nodes of hidden layers for gate networks (default = 20).
            - bias: whether to include bias terms in MLPs (default = True).
            - dropout: dropout rate for MLPs (default = 0.0, i.e. no dropout).
            - device: device to use for computations (default = 'cpu').
            - seed: random seed for reproducibility (default = 42).
            - kwargs: additional arguments for compatibility.
        """
        super().__init__()

        # Initialise input parameters
        self.i_size, self.o_size, self.w_size = i_size, o_size, w_size
        self.K, self.l_size, self.n_fwd_blocks = K, l_size, n_fwd_blocks
        self.gate_l, self.gate_n = gate_hidden_l, gate_hidden_n
        self.device, self.seed = device, seed

        # Parameters for Clusters
        self.c_means = nn.Parameter(
                        torch.rand(
                            (int(self.K), 
                            int(self.l_size)
                            ),
                            requires_grad=True,
                            device=self.device
                        )
                    )
        self.log_c_vars = nn.Parameter(
                            torch.rand(
                                (int(self.K), ),
                                requires_grad=True,
                                device=self.device
                            )
                        )
        
        # Ensure parameters are updatable
        self.c_means.requires_grad = True
        self.log_c_vars.requires_grad = True


        # Initialise encoder block - estimate alpha parameter of Dirichlet distribution given h and extracted input data features.
        self.encoder = MLP(
            input_size = self.l_size + self.l_size,      # input: concat(h, extr(x)) 
            output_size = self.K,                        # output: K-dimensional vector of cluster probabilities
            hidden_layers = self.gate_l,                 # gate_l hidden_layers with gate_n nodes each
            hidden_nodes = self.gate_n,                  # gate_n nodes per hidden layer
            act_fn = nn.ReLU(),                          # default activation function is ReLU
            bias = bias,
            dropout = dropout
        )
        self.enc_out = nn.Softmax(dim=-1)               # output is a probability distribution over clusters

        # Initialise Prior Block - estimate alpha parameter of Dirichlet distribution given h.
        self.prior = MLP(
            input_size = self.l_size,                    # input: h
            output_size = self.K,                        # output: K-dimensional vector of cluster probabilities
            hidden_layers = self.gate_l,                 # gate_l hidden_layers with gate_n nodes each
            hidden_nodes = self.gate_n,                  # gate_n nodes per hidden layer
            act_fn = nn.ReLU(),                          # default activation function is ReLU
            bias = bias,
            dropout = dropout
        )
        self.prior_out = nn.Softmax(dim=-1)             # output is a probability distribution over clusters
        
        # Initialise decoder block - given z, h, we generate a w_size sequence of observations, x.
        self.decoder = LSTM_Dec_v1(
            seq_len=self.w_size,
            output_dim=self.i_size + self.i_size, # output is concatenation of mean-log var of observation
            hidden_dim=self.l_size + self.l_size,  # state cell has same size as context vector (feat_extr(z) and h)
            num_layers=1,
            dropout=dropout
        )

        # Define the output network - computes outcome prediction given predicted cluster assignments
        self.predictor = MLP(
            input_size = self.l_size,                    # input: h
            output_size = self.o_size,                   # output: o_size-dimensional vector of outcome probabilities
            hidden_layers = self.gate_l,                 # gate_l hidden_layers with gate_n nodes each
            hidden_nodes = self.gate_n,                  # gate_n nodes per hidden layer
            act_fn = nn.ReLU(),                          # default activation function is ReLU
            bias = bias,
            dropout = dropout
        )
        self.predictor_out = nn.Softmax(dim=-1)          # output is o_size-dimensional vector of outcome probabilities
        

        # Define feature transformation functions
        self.phi_x = MLP(
            input_size = self.i_size,                    # input: x
            output_size = self.l_size,                   # output: l_size-dimensional vector of latent space features
            hidden_layers = self.gate_l,                 # gate_l hidden_layers with gate_n nodes each
            hidden_nodes = self.gate_n,                  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),                            # default activation function is ReLU
            bias=bias,                                   # default bias is True
            dropout=dropout
        )
        self.phi_x_out = nn.Tanh()                       # output is l_size-dimensional vector of latent space features

        self.phi_z = MLP(
            input_size = self.l_size,                    # input: z
            output_size = self.l_size,                   # output: l_size-dimensional vector of latent space features
            hidden_layers = self.gate_l,                 # gate_l hidden_layers with gate_n nodes each
            hidden_nodes = self.gate_n,                  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),                            # default activation function is ReLU
            bias=bias,                                   # default bias is True
            dropout=dropout
        )
        self.phi_z_out = nn.Tanh()                       # output is l_size-dimensional vector of latent space features

        # Define Cell Update Gate Functions
        self.cell_update = MLP(
            input_size=self.l_size + self.l_size + self.l_size, # input: concat(h, phi_x, phi_z)
            output_size=self.l_size,                            # output: l_size-dimensional vector of cell state updates
            hidden_layers=self.gate_l,                          # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,                           # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),                                   # default activation function is ReLU
            bias=bias,                                          # default bias is True
            dropout=dropout
        )
        self.cell_update_out = nn.Tanh()


    # Define training process for this class.
    def forward(self, x, y) -> Tuple[torch.Tensor, Dict]:
        """
        Forward Pass computation for a single pair of batch objects x and y.

        Params:
            - x: pytorch Tensor object of input data with shape (batch_size, max_seq_len, input_size);
            - y: pytorch Tensor object of corresponding outcomes with shape (batch_size, outcome_size).

        Output:
            - loss: pytorch Tensor object of loss value.
            - history: dictionary of relevant training history.

        We iterate over consecutive window time blocks. Within each window, we use the hidden state obtained at
        the last time window, and we generate mean and variance values for the subsequent window. Within the following
        window, we also use the true data and last hidden state to generate alpha and estimate zs values for
        each time step.

        Loss is computed by summing a) Log Lik loss (how good we are at predicting data), b) KL loss (how close
        the updated alpha approximates the posterior, and c) outcome loss, taken at the last time step, which 
        indicates whether we are able to predict the outcome correctly. Losses are computed at each time step
        for all windows except the latter, as it denotes a 'look' into the future.

        Losses are saved as:
        - - ELBO: Negative Evidence Lower Bound (Log Lik - KL_loss + outcome_loss), averaged over batch.
        - Log Likelihood: Log Likelihood of data, averaged over batch, and saved for each time step.
        - KL Loss: KL Loss between prior and posterior, averaged over batch, and saved for each time step.
        - Outcome Loss: Loss of outcome prediction, averaged over batch.
        """

        # ========= Define relevant variables and initialise variables ===========
        x, y = x.to(self.device), y.to(self.device) # move data to device

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()

        # Pre-fill the data tensor with zeros if we have a sequence length that is not a multiple of w_size
        remainder = seq_len % self.w_size
        if remainder != 0:

            # Calculate number of zeros to pre-append and the minimum index from which real data exists
            timestp_to_append = self.w_size - remainder

            # Pre-pend the input data with zeros to make it a multiple of w_size
            zeros_append = torch.zeros(batch_size, timestp_to_append, input_size, device=self.device)
            x = torch.cat((zeros_append, x), dim=1)

            # Update seq_len
            seq_len = x.size(1)
            assert (seq_len % self.w_size == 0) # Sequence length is not a multiple of w_size

        
        # Get number of time steps that we have
        num_time_steps = int(seq_len / self.w_size)


        # Initialization of pi, z, and h assignments
        h = torch.zeros(batch_size, self.l_size, device=self.device)
        est_pi = torch.ones(batch_size, self.K, device=self.device) / self.K
        est_z = model_utils.gen_samples_from_assign(est_pi, self.c_means, self.log_c_vars)
                

        # Initialise of Loss and History Tracker Objects - note time length includes forward prediction window
        ELBO = 0
        history = {
            "loss_loglik": torch.zeros(seq_len, device=x.device),
            "loss_kl": torch.zeros(seq_len, device=x.device),
            "loss_out": 0,
            "pis": torch.zeros(batch_size, seq_len + self.w_size, self.K, device=self.device),
            "zs": torch.zeros(batch_size, seq_len + self.w_size, self.l_size, device=self.device),
            "alpha_encs": torch.zeros(batch_size, seq_len + self.w_size, self.K, device=self.device),
            "mugs": torch.zeros(batch_size, seq_len + self.w_size, self.i_size, device=self.device),
            "log_vargs": torch.zeros(batch_size, seq_len + self.w_size, self.i_size, device=self.device)
        }




        # ================== Iteration through time-steps  ==============

        # We iterate over all windows of our analysis, this includes the last window where we look into the future and do not compute losses.
        for window_id in range(num_time_steps + self.n_fwd_blocks):
            "Iterate through each window block"

            # Bottom and high indices
            lower_t, higher_t = window_id * self.w_size, (window_id + 1) * self.w_size

            # First we estimate the observations for the incoming window given current estimates. This is of shape (bs, w_size, 2*input_size)
            _, _, data_gen = self.decoder_pass(h=h, z=est_z)

            # Decompose obvs_pred into mean and log-variance - shape is (bs, w_size, 2 * input_size)
            mu_g, logvar_g = torch.chunk(data_gen, chunks=2, dim=-1)
            var_g = torch.exp(logvar_g) + eps

            for _w_id, t in enumerate(range(lower_t, higher_t)):
                # Estimate alphas for each time step within the window. 

                # Use true data if within the original data size, else use generated data
                if t < seq_len:
                    x_t = x[:, t, :]

                else:
                    gen_samples = model_utils.gen_diagonal_mvn(mu_g=mu_g, log_var_g = logvar_g)
                    x_t = gen_samples[:, _w_id, :]

                # Compute alphas of prior and encoder networks
                alpha_prior = self.prior_pass(h=h)
                alpha_enc = self.encoder_pass(h=h, x=x_t)


                # Sample cluster distribution from alpha_enc, and estimate samples from clusters based on mixture of Gaussian model
                est_pi = model_utils.sample_dir(alpha=alpha_enc)
                est_z = model_utils.gen_samples_from_assign(
                    pi_assign=est_pi, 
                    c_means=self.c_means, 
                    log_c_vars=self.log_c_vars
                )

                # ----- UPDATE CELL STATE ------
                h = self.state_update_pass(h=h, x=x_t, z=est_z)

                # Append objects FOR ALL TIME STEPS
                history["pis"][:, t, :] = est_pi
                history["zs"][:, t, :] = est_z
                history["alpha_encs"][:, t, :] = alpha_enc
                history["mugs"][:, t, :] = mu_g[:, _w_id, :]
                history["log_vargs"][:, t, :] = logvar_g[:, _w_id, :]


                # -------- COMPUTE LOSS FOR TIME t if time is not in the future ---------

                if t < seq_len:
                        
                    # Compute log likelihood loss and KL divergence loss
                    log_lik = LM_utils.torch_log_gaussian_lik(x_t, mu_g[:, _w_id, :], var_g[:, _w_id, :], device=self.device)
                    kl_div = LM_utils.dir_kl_div(a1=alpha_enc, a2=alpha_prior)

                    # Add to loss tracker
                    ELBO += log_lik - kl_div

                    # Append lOSSES TO HISTORY TRACKERS WITHIN THE ALLOWED SEQUENCE OF STEPS
                    history["loss_kl"][t] += torch.mean(kl_div, dim=0)
                    history["loss_loglik"][t] += torch.mean(log_lik, dim=0)


        # Once all times have been computed, make predictions on outcome
        y_pred = self.predictor_pass(z=est_z)
        history["y_preds"] = y_pred

        # Compute log loss of outcome
        pred_loss = LM_utils.cat_cross_entropy(y_true=y, y_pred=y_pred)

        # Add to total loss
        ELBO += pred_loss

        # Compute average per batch
        ELBO = torch.mean(ELBO, dim=0)

        # Append to history tracker
        history["loss_out"] += torch.mean(pred_loss, dim=0)

        return (-1) * ELBO, history      # want to maximize loss, so return negative loss
    
    def fit(self, 
            train_data, val_data,
            K_fold_idx: int = 1,
            lr: float = 0.001, 
            batch_size: int = 32,
            num_epochs: int = 100,
            viz_params: Dict = {}
        ):
        """
        Method to train model given train and validation data, as well as training parameters.

        Params:
            - train_data: Tuple (X, y) of training data, with shape (N, T, D) and (N, O), respectively.
            - val_data: Tuple (X, y) of validation data. If None or (None, None), then no validation is performed.
            - K_fold_idx: index of current fold in K-fold cross-validation. If no Cross validation, then this parameter is set to 1.
            - lr: learning rate for optimizer.
            - batch_size: batch size for training.
            - num_epochs: number of epochs to train for.
            - viz_params: dictionary of parameters for visualization. If None, then some placeholders are passed.
                - save_dir: directory to save visualization results.
                - fold: current fold in K-fold cross-validation.
                - features: list of feature names.
                - outcomes: list of outcome names.

        Outputs:
            - loss: final loss value.
            - history: dictionary with training history, including each loss component.
        """
        # ================== PARAMETER CHECKING AND SAVE HANDLING ==================
        # Parse visualization parameters
        save_dir = viz_params["save_dir"]

        # Create save directory and scores file if it does not exist
        train_fd = f"{save_dir}/train"
        val_fd = f"{save_dir}/val"

        with open(f"{train_fd}/train_losses.csv", "w", newline="") as f:

            # Write header
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "loss_kl", "loss_loglik", "loss_out"])

        with open(f"{val_fd}/val_losses.csv", "w", newline="") as f:
            
            # Write header
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "loss_kl", "loss_loglik", "loss_out"])



        # ================== DATA PREPARATION ==================
        # Unpack data and make data loaders
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Prepare data for training
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer and Logging of weights
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        wandb.watch(
                models=self, 
                log="all", 
                log_freq=1, 
                idx = K_fold_idx
            )
        

        # ================== TRAINING-VALIDATION LOOP ==================
        history_objects = {}
        for epoch in range(1, num_epochs + 1):

            # Set model to train mode and initialize loss tracker
            self.train()
            train_loss = torch.zeros(1, device=self.device)
            train_loglik = torch.zeros(1, device=self.device)
            train_kl = torch.zeros(1, device=self.device)
            train_out = torch.zeros(1, device=self.device)

            # Iterate through batches
            batch_id = 0
            for batch_id, (x, y) in enumerate(train_loader):

                # Zero out gradients for each batch
                optimizer.zero_grad()

                # Compute Loss for single model pass
                loss, history_objects = self.forward(x, y)
                batch_loglik, batch_kl, batch_outl = history_objects["loss_loglik"], history_objects["loss_kl"], history_objects["loss_out"]

                # Back-propagate loss and update weights
                loss.backward()
                optimizer.step()

                # Add to loss tracker
                train_loss += loss.item()
                train_loglik += torch.sum(batch_loglik)         # Sums over time
                train_kl += torch.sum(batch_kl)                 # Sum over time
                train_out += batch_outl                 # Add batch loss total loss over batch

                # Print message of loss per batch, which is re-setted at the end of each epoch
                print("Train epoch: {}   [{:.2f} - {:.0f}%]".format(
                    epoch, loss.item(), 100. * batch_id / len(train_loader)),
                    end="\r")
                
            # Take average over all batches in the train data
            epoch_loss = train_loss / (batch_id + 1)
            epoch_loglik = train_loglik / (batch_id + 1)
            epoch_kl = train_kl / (batch_id + 1)
            epoch_out = train_out / (batch_id + 1)

            # Print Message at the end of each epoch with the main loss and all auxiliary loss functions
            print("Train {} ({:.0f}%):  [loss {:.2f} - loglik {:.2f} - kl {:.2f} - out {:.2f}]".format(
                epoch, 100, 
                epoch_loss.item(), epoch_loglik.item(), epoch_kl.item(), epoch_out.item()), end="     ")
                
            
            # Log objects to Weights and Biases
            wandb.log({"train/epoch": epoch + 1, "train/loss": epoch_loss, "train/loglik": epoch_loglik,
                "train/kldiv": epoch_kl, "train/out_l": epoch_out
                }, 
                step=epoch+1
            )	

            # Save to CSV file
            with open(f"{train_fd}/train_losses.csv", "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, epoch_loss.item(), epoch_kl.item(), epoch_loglik.item(), epoch_out.item()])

    
            # Check performance on validation set if exists
            val_loss, outputs_val = self.validate(X_val, y_val, epoch=epoch, viz_params=viz_params)    

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



    def validate(self, X, y, epoch: int, viz_params: Dict):
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
                val_loss, history_objects = self.forward(X, y)

                # Load individual Losses from tracker
                log_lik = torch.sum(history_objects["loss_loglik"])
                kl_div = torch.sum(history_objects["loss_kl"])
                out_l = history_objects["loss_out"]

                # Log Losses
                wandb.log({
                    "val/epoch": epoch + 1, "val/loss": val_loss, "val/loglik": log_lik,
                    "val/kldiv": kl_div, "val/out_l": out_l,
                    },
                    step=epoch+1
                )


                # Print message
                print("Val epoch {} ({:.0f}%):  [loss {:.2f} - loglik {:.2f} - kl {:.2f} - out {:.2f}]".format(
                    epoch, 100, val_loss.item(), log_lik.item(), 
                    kl_div.item(), out_l.item()
                    )
                )

                # Save to CSV file
                with open(f"{val_fd}/val_losses.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, val_loss.item(), kl_div.item(), log_lik.item(), out_l.item()])


                # Log performance evaluation scores
                logger.logger_sup_scores(y_true=y, y_pred=history_objects["y_pred"],save_dir=val_fd, epoch=epoch, class_names=outcomes)

                # Log clustering evaluation scores
                

                # Log more complex results
                model_params={
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

    def predict(self, X ,y, run_config: Union[Dict, None] = None, class_names: List = [], feat_names: List = [], save_dir: Union[str, None] = None):
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
                _, history_objects = self.forward(X, y)      # Run forward pass

                # Log results
                if save_dir is not None:
                        
                    model_params={
                        "c_means": self.c_means,
                        "log_c_vars": self.log_c_vars,
                        "seed": self.seed,
                    }

                    logger(model_params=model_params, X=X, y=y, log=history_objects, epoch=0, mode="test", class_names=class_names, feat_names=feat_names,
                            save_dir = save_dir)

                # Append Test data
                history_objects["X_test"] = X
                history_objects["y_test"] = y
                history_objects["run_config"] = run_config

                return history_objects
    
    # Useful methods for model
    def x_feat_extr(self, x):
        return self.phi_x_out(self.phi_x(x))
    
    def z_feat_extr(self, z):
        return self.phi_z_out(self.phi_z(z))
    
    def encoder_pass(self, h, x):
        "Single pass of the encoder to obtain alpha param."
        return self.enc_out(
                    self.encoder(
                        torch.cat([
                            self.x_feat_extr(x), # Extract feature from x
                            h
                        ], dim=-1)  # Concatenate x with cell state in last dimension
                    ) 
                ) + eps  # Add small value to avoid numerical instability

    def decoder_pass(self, h, z):
        return self.decoder(
                    torch.cat([
                        self.z_feat_extr(z),  # Extract feature from z
                        h
                    ], dim=-1)  # Concatenate z with cell state in last dimension
                ) 
    
    def state_update_pass(self, h, x, z):
        return self.cell_update_out(            # Final layer of MLP gate
                        self.cell_update(
                            torch.cat([
                                self.z_feat_extr(z), # Extract feature from z
                                self.x_feat_extr(x), # Extract feature from x
                                h            # Previous cell state
                            ], dim=-1) # Concatenate z with cell state in last dimension
                        )
                )
    
    def prior_pass(self, h):
        return self.prior_out(self.prior(h)) + eps

    def predictor_pass(self, z):
        return self.predictor_out(self.predictor(z))
# endregion
        