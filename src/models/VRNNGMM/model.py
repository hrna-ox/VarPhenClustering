"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Implements two Class objects derived from Support Vector Machines to classify multi-dimensional time-series using 2 different approaches.
"""

# Import required packages
import pickle
import numpy as np

import os

from sklearn.mixture import GaussianMixture

"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file defines the main proposed model, and how to train the model.
"""

# ============= IMPORT LIBRARIES ==============
import pickle
import torch
import torch.nn as nn
# import wandb
import os

from torch.utils.data import DataLoader, TensorDataset

import src.metrics.loss_functions as LM_utils
from src.models.deep_learning_base_classes import MLP

import src.models.model_utils as model_utils


class VRNNGMM(nn.Module):
    """
    Implements VRNNGMM model as described in the paper.
    """

    def __init__(
        self,
        input_dims: int,
        output_dim: int,
        K: int,
        latent_dim: int = 10,
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
            - output_dim: dimensionality of output data (number of classes).
            - K: number of clusters to consider.
            - latent_dim: dimensionality of latent space (default = 10).
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
        self.output_dim = output_dim
        self.K = K
        self.latent_dim = latent_dim
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
            output_size=self.latent_dim + self.latent_dim,  # output:
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,
            dropout=dropout,
        )
        self.encoder_output_fn = nn.ReLU()  

        # Initialise Prior Block - estimate alpha parameter of Dirichlet distribution given h.
        self.prior = MLP(
            input_size=self.latent_dim,  # input: h
            output_size=self.latent_dim + self.latent_dim,  # output: K-dimensional vector of cluster probabilities
            hidden_layers=self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes=self.gate_n,  # gate_n nodes per hidden layer
            act_fn=nn.ReLU(),  # default activation function is ReLU
            bias=bias,
            dropout=dropout,
        )
        self.prior_output_fn = nn.ReLU()

        # Initialise decoder block - given z, h, we generate a window_num_obvs sequence of observations, x.
        self.decoder = MLP(
            input_size = self.latent_dim + self.latent_dim,  # input: concat(z, h)
            output_size = 2 * self.input_dims,  # output: input_dims-dimensional vector of generated data
            hidden_layers = self.gate_l,  # gate_l hidden_layers with gate_n nodes each
            hidden_nodes = self.gate_n,  # gate_n nodes per hidden layer
            act_fn = nn.ReLU(),  # default activation function is ReLU
            bias = bias,
            dropout = dropout,
        )

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
        self.cluster_probs = np.ones(shape=(K, output_dim)) / K


    
    "Define methods to simplify passes through network blocks"
    def _apply_encoder(self, x):
        return self.encoder_output_fn(self.encoder(x))

    def _apply_prior(self, h):
        return self.prior_output_fn(self.prior(h))

    def _apply_decoder(self, c):
        return self.decoder(c)
    
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

        Loss is computed by summing (all losses averaged over batch):
            a) Log Lik loss (how good we are at generating data), 
            b) KL loss (how close the updated alpha approximates the posterior, and 
        """

        # ========= Define relevant variables ===========
        x, y = x.to(self.device), y.to(self.device)  # move data to device
        assert mode in ["train", "eval"], "Mode has to be either 'train' or 'eval'."

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()

        # ========== Initialise relevant variables ==========
        h = torch.zeros(batch_size, self.latent_dim, device=self.device)
        zs = torch.zeros(batch_size, self.latent_dim, device=self.device)

        # Initialize ELBO, history tracker and future tracker
        ELBO, Loss_loglik, Loss_kl = torch.zeros(1, device=self.device), torch.zeros(1, device=self.device), torch.zeros(1, device=self.device)
        history = {
            "loss_loglik": torch.zeros(seq_len, device=self.device),
            "loss_kl": torch.zeros(seq_len, device=self.device),
            "loss_out": 0,
            "zs": torch.zeros(batch_size, seq_len, self.latent_dim, device=self.device),
            "gen_means": torch.zeros(batch_size, seq_len, self.input_dims, device=self.device),
            "gen_log_vars": torch.zeros(batch_size, seq_len, self.input_dims, device=self.device)
        }


        # ================== Iterate through time ==================
        """
        Note we iterate over window blocks, and within each block we iterate over individual time steps to compute loss functions.
        There is a past component (data was pre-filled), and a future component (we predict forward self.n_fwd_blocks windows).
        """

        # Iterate through each time step to estimate updated representations and compute loss terms
        for t_idx in range(seq_len):

            # Access input data at time t
            x_t = x[:, t_idx, :]

            # Compute alphas of prior and encoder networks given previous cell estimate and current input data
            z_pri_mean, z_pri_logvar = torch.chunk(
                self._prior_pass(h=h),
                chunks=2,
                dim=-1
            )
            z_inf_mean, z_inf_logvar = torch.chunk(
                self._encoder_pass(h=h, x=x_t),
                chunks=2,
                dim=-1
            )

            # Sample z from distribution
            zs = model_utils.generate_diagonal_multivariate_normal_samples(
                mu = z_inf_mean,
                logvar = z_inf_logvar,
            )

            # Update Cell State
            new_h = self._cell_update_pass(h=h, x=x_t, z=zs)

            # Estimate means 
            _est_mean, _est_logvar = torch.chunk(
                                        self._decoder_pass(h=new_h, z=zs), 
                                        chunks=2, 
                                        dim=-1
                                    )

            # Compute Loss Terms
            log_lik = LM_utils.torch_log_Gauss_likelihood(x=x_t, mu=_est_mean, logvar=_est_logvar, device=self.device)
            kl_div = LM_utils.torch_gaussian_kl_div(
                mu_1=z_inf_mean, logvar_1=z_inf_logvar, 
                mu_2=z_pri_mean, logvar_2=z_pri_logvar
            )

            ELBO += log_lik - kl_div
            Loss_kl += kl_div
            Loss_loglik += log_lik

            # Append objects to history tracker if we are in evaluation mode
            if mode == "eval":
                history["zs"][:, t_idx, :] = zs

                # Append generate mean, generate var
                history["gen_means"][:, t_idx, :] = _est_mean
                history["gen_log_vars"][:, t_idx, :] = _est_logvar

                # Add to loss tracker
                history["loss_kl"][t_idx] += kl_div
                history["loss_loglik"][t_idx] += log_lik

        # Convert ELBO to -1 * ELBO since we want to maximize ELBO
        loss = (-1) * ELBO

        return loss, Loss_loglik, Loss_kl, history  # want to maximize ELBO, so return - ELBO


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
        # wandb.watch(models=self, log="all", log_freq=1, idx=self.K_fold_idx)           

        # Printing Message
        print("Printing Losses loss, Log Lik, KL")

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
                batch_loss, batch_loglik, batch_kl, train_history = self.forward(x, y, mode="train")

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
                    val_loss, val_loglik, val_kl, val_history = self.forward(X, y, mode="eval")

                    # print message
                    print(
                        "Val {} ({}): {:.2f} - {:.2f} - {:.2f}".format(
                            epoch, self.num_epochs, val_loss.item(), val_loglik.item(), val_kl.item()
                        ),
                        end="\n",
                    )
        

        # ===================== GMM =====================

        # Load the whole training set and extract the representations
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=X_train.shape[0], shuffle=False)
        with torch.inference_mode():
            for X, y in train_loader:
                tr_L, tr_loglik, tr_KL, tr_history = self.forward(X, y, mode="eval")
                zs_3D = tr_history["zs"].detach().cpu().numpy()

                # Apply GMM
                self.gmm = GaussianMixture(n_components=self.K, covariance_type="diag", random_state=self.seed,
                                        init_params = "k-means++", verbose=1)
                self.gmm.fit(zs_3D[:, -1, :])

                # Predict cluster memberships by looking at the last time step
                clus_train = self.gmm.predict(zs_3D[:, -1, :])
                print(clus_train)
                
                # Estimate cluster outcome distributions
                for cluster_idx in range(self.K):
                    for outcome_idx in range(self.output_dim):
                        if np.sum(clus_train == cluster_idx) != 0:
                            self.cluster_probs[cluster_idx, outcome_idx] = np.mean(
                                y_train[clus_train == cluster_idx, outcome_idx])


    def predict(self, X, y):
        """Similar to forward method, but focus on inner computations and tracking objects for the model."""

        # ================== INITIALIZATION ==================
        self._test_loss = {"test": []}
        self._test_exp_obj = {"test": {}}

        # ================== DATA PREPARATION ==================

        # Prepare Data
        test_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        test_loader = DataLoader(test_data, batch_size=X.shape[0], shuffle=False)

        # # Initialize logger objects
        # wandb.watch(models=self, log="all", log_freq=1, idx=self.K_fold_idx)           

        # Set model to evaluation mode
        self.eval()

        # Apply forward prediction
        test_history, test_y_pred, test_clus_labels = {}, [], []
        with torch.inference_mode():
            for X, y in test_loader:

                # Pass data through model
                test_loss, test_loglik, test_kl, test_history = self.forward(X, y, mode="eval")

                # Save objects to output
                output_dir = {
                    "test_loss": test_loss,
                    "test_loglik": test_loglik,
                    "test_kl": test_kl,
                    "test_history": test_history
                }

                # Predict Cluster Labels and outcomes
                zs_3D = test_history["zs"].detach().cpu().numpy()

                # Predict cluster memberships by looking at the last time step
                test_clus_labels = self.gmm.predict(zs_3D[:, -1, :])
                test_y_pred = np.eye(self.K)[test_clus_labels] @ self.cluster_probs

        return test_y_pred, test_clus_labels, test_history
        


    
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
            "input_dims": self.input_dims,
            "output_dim": self.output_dim,
            "K": self.K,
            "latent_dim": self.latent_dim,
            "gate_l": self.gate_l,
            "gate_n": self.gate_n,
            "device": self.device,
            "seed": self.seed,
            "model_params": self.state_dict(),
            "gmm_params": self.gmm.get_params(),
            "cluster_probs": self.cluster_probs
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

    def load_run(self, save_dir: str):
        """
        Load model objects. Useful for reproducibility.

        Args:
            save_dir (str): Directory to save the objects.
        
        Returns:
            Tuple [X, y_true, y_pred] of data objects, as well as the trained model.
        """

        # Initialize empty dics
        data_objects, run_info = {}, {}
        
        # First load the model objects
        if os.path.exists(save_dir + "data_objects.pkl"):
            with open(save_dir + "data_objects.pkl", "rb") as f:
                data_objects = pickle.load(f)
        
        # Load run info if exists
        if os.path.exists(save_dir + "run_info.pkl"):
            with open(save_dir + "run_info.pkl", "rb") as f:
                run_info = pickle.load(f)

        # Load model
        with open(save_dir + "models.pkl", "rb") as f:
            models_dic = pickle.load(f)
        
        # Unpack the model objects
        self.input_shape = models_dic["input_shape"]
        self.output_dim = models_dic["output_dim"]
        self.K = models_dic["K"]
        self.latent_dim = models_dic["latent_dim"]
        self.gate_l = models_dic["gate_l"]
        self.gate_n = models_dic["gate_n"]
        self.device = models_dic["device"]
        self.seed = models_dic["seed"]
        self.cluster_probs = models_dic["cluster_probs"]

        # Update GMM params and model weights
        self.gmm.set_params(**models_dic["gmm_params"])
        self.load_state_dict(models_dic["model_params"])

        return run_info, data_objects

