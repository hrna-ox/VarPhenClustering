"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Model file to define GC-DaPh class.
"""

# ============= IMPORT LIBRARIES ==============
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from typing import Tuple, Dict, Union
from torch.utils.data import DataLoader, TensorDataset

import src.models.losses_and_metrics as LM_utils
from src.models.deep_learning_base_classes import MLP, LSTM_Dec_v1, LSTM_Dec_v2

import src.models.Dir_VRNN.auxiliary_functions as model_utils
from src.models.Dir_VRNN.auxiliary_functions import eps

import matplotlib.pyplot as plt


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
        self.K, self.l_size = K, l_size
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
        """

        # ========= Define relevant variables and initialise variables ===========
        x, y = x.to(self.device), y.to(self.device) # move data to device

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()

        # Basic information about the sequence and number of time-steps
        assert seq_len % self.w_size == 0 # Sequence length must be divisible by window size
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
        for window_id in range(num_time_steps + 1):
            "Iterate through each window block"

            # Bottom and high indices
            lower_t, higher_t = window_id * self.w_size, (window_id + 1) * self.w_size

            # First we estimate the observations for the incoming window given current estimates. This is of shape (bs, w_size, 2*input_size)
            _, _, data_gen = self.decoder_pass(h=h, z=est_z)

            # Decompose obvs_pred into mean and log-variance - shape is (bs, T, 2 * input_size)
            mu_g, logvar_g = torch.chunk(data_gen, chunks=2, dim=-1)
            var_g = torch.exp(logvar_g) + eps

            for _w_id, t in enumerate(range(lower_t, higher_t)):
                # Estimate alphas for each time step within the window. 

                # Subset observation to time t
                x_t = x[:, t, :]

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
            train_data, val_data=(None, None),
            K_fold_idx: int = 1,
            lr: float = 0.001, 
            batch_size: int = 32,
            num_epochs: int = 100
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

        Outputs:
            - loss: final loss value.
            - history: dictionary with training history, including each loss component.
        """

        # Unpack data and make data loaders
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Prepare data for training
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer and Logging
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        wandb.watch(
                models=self, 
                log="all", 
                log_freq=1, 
                idx = K_fold_idx
            )

        # ================== TRAINING-VALIDATION LOOP ==================
        for epoch in range(1, num_epochs + 1):

            # Set model to train mode and initialize loss tracker
            self.train()
            train_loss = torch.zeros(1, device=self.device)
            train_loglik = torch.zeros(1, device=self.device)
            train_kl = torch.zeros(1, device=self.device)
            train_out = torch.zeros(1, device=self.device)

            # Iterate through batches
            for batch_id, (x, y) in enumerate(train_loader):

                # Zero out gradients for each batch
                optimizer.zero_grad()

                # Compute Loss for single model pass
                loss, history_objects = self.forward(x, y)
                batch_loglik, batch_kl, batch_out = history_objects["loss_loglik"], history_objects["loss_kl"], history_objects["loss_out"]

                # Back-propagate loss and update weights
                loss.backward()
                optimizer.step()

                # Add to loss tracker
                train_loss += loss.item()
                train_loglik += torch.sum(batch_loglik)         # Sums over batch and over time
                train_kl += torch.sum(batch_kl)                 # Sum over batch and over time
                train_out += torch.sum(batch_out)           # Sum over batch

                # Print message of loss per batch, which is re-setted at the end of each epoch
                print("Train epoch: {}   [{:.5f} - {:.0f}%]".format(
                    epoch, loss.item(), 100. * batch_id / len(train_loader)),
                    end="\r")
                
            # Take average over all samples in the train data
            epoch_loglik = train_loglik / len(train_loader) 
            epoch_kl = train_kl / len(train_loader)
            epoch_out = train_out / len(train_loader)

            # Print Message at the end of each epoch with the main loss and all auxiliary loss functions
            print("Train epoch {} ({:.0f}%):  [L{:.5f} - loglik {:.5f} - kl {:.5f} - out {:.5f}]".format(
                epoch + 1, 100, 
                train_loss, epoch_loglik, epoch_kl, epoch_out))
                
            
            # Log objects to Weights and Biases
            wandb.log({
                "train/epoch": epoch + 1,
                "train/loss": train_loss,
                "train/loglik": epoch_loglik,
                "train/kldiv": epoch_kl,
                "train/out_l": epoch_out
            },
            step=epoch+1
        )	
            
            # Check performance on validation set if exists
            if X_val is not None and y_val is not None:
                self.validate(X_val, y_val, epoch=epoch)
    
    def validate(self, X, y, epoch: int):
        """
        Compute Performance on Val Dataset.

        Params:
            - X: input data of shape (bs, T, input_size)
            - y: outcome data of shape (bs, output_size)
            - epoch: int indicating epoch number
        """

        # Set model to evaluation mode 
        self.eval()
        iter_str = f"val epoch {epoch}"

        # Prepare Data
        val_data = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        val_loader = DataLoader(val_data, batch_size=X.shape[0], shuffle=False)

        # Apply forward prediction
        with torch.inference_mode():
            for X, y in val_loader:
                
                # Run model once through the 
                val_loss, history_objects = self.forward(X, y)

                # Load individual Losses from tracker
                log_lik = torch.sum(history_objects["loss_loglik"], dim=1)
                kl_div = torch.sum(history_objects["loss_kl"], dim=1)
                out_l = history_objects["loss_out"]

                # Log Losses
                wandb.log({
                    "val/epoch": epoch + 1,
                    "val/loss": val_loss,
                    "val/loglik": log_lik,
                    "val/kldiv": kl_div,
                    "val/out_l": out_l,
                    },
                    step=epoch+1
                )


                # Print message
                print("Predict {} ({:.0f}%):  [L{:.5f} - loglik {:.5f} - kl {:.5f} - out {:.5f}]".format(
                    iter_str, 100, 
                    val_loss, 
                    torch.mean(log_lik), 
                    torch.mean(kl_div), 
                    torch.mean(out_l)
                    )
                )

                # Log results
                self.logger(X=X, y=y, log=history_objects, epoch=epoch, mode="val")

                return val_loss, history_objects

    def predict(self, X ,y, run_config: Union[Dict, None] = None):
        # Similar to forward method, but focus on inner computations and tracking objects for the model.
        _, history_objects = self.forward(X, y)      # Run forward pass

        # Log results
        self.logger(X=X, y=y, log=history_objects, epoch=0, mode="test")

        # Append Test data
        history_objects["X_test"] = X
        history_objects["y_test"] = y
        history_objects["run_config"] = run_config

        return history_objects
    
    def logger(self, X: torch.Tensor, y: torch.Tensor, log:Dict, epoch: int = 0, mode: str = "val", *args, **kwargs):
        """
        Logger for the model. 

        Params:
        - X: unseen input data of shape (bs, T, input_size)
        - y: true outcome data of shape (bs, output_size)
        - Log_Dict: dictionary with loss and object information.
        - epoch: current epoch for training. This parameter is disregarded if mode is set to 'test'.
        - mode: str indicating whether the logger is for testing or validation.

        Returns:
        - None, saves into wandb logger
        """

        # Load objects
        temp_pis = log["pis"]
        
        # ================ LOGGING ================

        # Compute useful metrics and plots during training - including:
        # a) cluster assignment distribution, 
        # b) cluster means separability, 
        # c) accuracy, f1 and recall scores,
        # d) confusion matrix, 

        # Compute distribution over cluster memberships at this time step.
        clus_memb_dist = model_utils.torch_get_temp_clus_memb_dist(temp_pis_assign=temp_pis)
        fig, ax = model_utils.plot_clus_memb_evol(temp_clus_memb=clus_memb_dist)
        
        # Log
        wandb.log({f"{mode}/clus-memb-evolution": fig}, step=epoch+1)
        plt.close()

        
        # Get Box Plots of cluster assignments and Log
        fig, ax = model_utils.torch_plot_clus_prob_assign_time(temp_pis_assign=temp_pis)
        wandb.log({f"{mode}/clus_assign_time_boxplot": fig}, step=epoch+1)
        plt.close()


        # Information about cluster separability (means and phenotypes)
        clus_means, clus_vars = self.c_means, torch.exp(self.log_c_vars)
        clus_mean_sep = model_utils.torch_clus_means_separability(clus_means=self.c_means)

        # Log
        wandb.log({
                f"{mode}/cluster_means": 
                    wandb.Table(
                        data=clus_means.detach().numpy(),
                    ),
                f"{mode}/cluster_vars": 
                    wandb.Table(
                        data=clus_vars.detach().numpy(),
                    ),
                f"{mode}/clus_mean_sep": clus_mean_sep.detach().numpy(),
            },
            step=epoch+1
        )
        
        # Log embeddings and other vectors            
        alpha_samples, pi_samples = log["alpha_encs"], log["pis"]
        mug_samps, logvar_samps = log["mugs"], log["log_vargs"]
        z_samples = log["zs"]

        wandb.log({
                f"{mode}/z_samples": 
                    wandb.Table(
                        data=z_samples.detach().numpy()
                    ),
                f"{mode}/alpha_samples": 
                    wandb.Table(
                        data=alpha_samples.detach().numpy()
                    ),
                f"{mode}/pi_samples":
                    wandb.Table(
                        data=pi_samples.detach().numpy()
                    ),
                f"{mode}/mug_samples":
                    wandb.Table(
                        data=mug_samps.detach().numpy()
                    ),
                f"{mode}/logvar_samples":
                    wandb.Table(
                        data=logvar_samps.detach().numpy()
                    )
            },
            step=epoch+1
        )


        # Log Performance Scores
        y_npy = y.detach().numpy()
        y_pred = log["y_preds"].detach().numpy()
        clus_prob = temp_pis.detach().numpy()
        X_npy = X.detach().numpy()
        class_names = [f"Class {i}" for i in range(y_pred.shape[-1])]

        # Compute accuracy, f1 and recall scores
        acc = LM_utils.accuracy_score(y_true=y_npy, y_pred=y_pred)
        
        macro_f1 = LM_utils.macro_f1_score(y_true=y_npy, y_pred=y_pred)
        micro_f1 = LM_utils.micro_f1_score(y_true=y_npy, y_pred=y_pred)
        recall = LM_utils.recall_score(y_true=y_npy, y_pred=y_pred)
        precision = LM_utils.precision_score(y_true=y_npy, y_pred=y_pred)

        # Get roc and prc
        roc_scores = LM_utils.get_roc_auc_score(y_true=y_npy, y_pred=y_pred)
        prc_scores = LM_utils.get_pr_auc_score(y_true=y_npy, y_pred=y_pred)

        # Compute Confusion Matrix
        confusion_matrix = LM_utils.get_confusion_matrix(y_true=y_npy, y_pred=y_pred)

        # Compute confusion matrix, ROC and PR curves
        roc_fig, roc_ax = LM_utils.get_roc_curve(y_true=y_npy, y_pred=y_pred, class_names=[]) 
        pr_curve = LM_utils.get_pr_curve(y_true=y_npy, y_pred=y_pred, class_names=[])
    
        # Compute Data Clustering Metrics
        label_metrics = LM_utils.get_clustering_label_metrics(y_true=y_npy, clus_pred=clus_prob)
        rand, nmi = label_metrics["rand"], label_metrics["nmi"]

        # Compute Unsupervised Clustering Metrics
        unsup_metrics = LM_utils.compute_unsupervised_metrics(X=X_npy, clus_pred=clus_prob, seed=self.seed)
        sil, dbi, vri = unsup_metrics["sil"], unsup_metrics["dbi"], unsup_metrics["vri"]

        # Combine scores to single table
        single_output = pd.Series(
            data = [acc, macro_f1, micro_f1, recall, precision],
            index=["Accuracy", "Macro F1", "Micro F1", "Recall", "Precision"]
        )

        class_outputs = pd.DataFrame(
            data=np.vstack((roc_scores, prc_scores)),
            index=["ROC", "PRC"],
            columns=class_names
        )
        
        temp_outputs = pd.DataFrame(
            data=np.vstack((rand, nmi, sil, dbi, vri)),
            index=["Rand", "NMI", "Silhouette", "DBI", "VRI"],
            columns=list(range(X_npy.shape[1]))
        )

        # Log Objects
        wandb.log({
            f"{mode}/confusion_matrix":
                wandb.Table(data=confusion_matrix, columns=[f"True {_name}" for _name in class_names], 
                            rows=[f"Pred {_name}" for _name in class_names]),
            f"{mode}/single_output": wandb.Table(data=single_output),
            f"{mode}/class_outputs": wandb.Table(data=class_outputs),
            f"{mode}/temp_outputs": wandb.Table(data=temp_outputs),
            f"{mode}/roc_curve": roc_fig,
            f"{mode}/pr_curve": pr_curve
        },
        step=epoch+1
    )


        # Print and Log Phenotype Information
        prob_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="prob")
        onehot_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="one-hot")

        # Log Phenotype Information
        wandb.log({
            f"{mode}/prob_phens": wandb.Table(data=prob_phens.detach().numpy()),
            f"{mode}/onehot_phens": wandb.Table(data=onehot_phens.detach().numpy())
            },
            step=epoch+1
        )




        """

        MISSING IMPLEMENTATION:
        - Get Class Names
        - Get Feature Names
        - Better Sampling Visualization

        """


        # Generate Data Samples and Compare with True Occurence
        gen_samples = model_utils.gen_diagonal_mvn(mug_samps, logvar_samps).detach().numpy()

        # Sample 10 random patients and plot to Wandb
        random_pats_10 = torch.randint(low=0, high=X.shape[0], size=(10,))

        for _pat_id in random_pats_10:

            # Select true data and generated data
            _x_pat = X_npy[_pat_id, :, :]
            _x_gen = gen_samples[_pat_id, :, :]

            # Plot to Wandb
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(_x_pat, label="True")
            ax.plot(_x_gen, label="Generated")
            wandb.log({
                "test/{}-pat".format(_pat_id): wandb.Image(fig)
            })


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
        