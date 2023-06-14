"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Model file to define GC-DaPh class.
"""

# ============= IMPORT LIBRARIES ==============
import torch
import torch.nn as nn
import wandb

from torch.nn import LSTMCell
from torch.utils.data import DataLoader, TensorDataset

import src.models.losses_and_metrics as LM_utils
from src.models.deep_learning_base_classes import MLP

import src.models.Dir_VRNN.auxiliary_functions as model_utils
from src.models.Dir_VRNN.auxiliary_functions import eps



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
        self.i_dim, self.o_dim = output_dim, output_dim        # The output dim must match the input dim for the next time-step
        self.n_layers, self.dropout = num_layers, dropout

        # Define main LSTM layer and output layer
        self.lstm_cell = LSTMCell(input_size=self.i_dim, hidden_size=self.h_dim, bias=True)
        self.fc_out = nn.Linear(self.h_dim, self.o_dim)

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
        cur_input = torch.zeros(bs, self.i_dim).to(c_vector.device)
        h_t, c_t = c_vector, c_vector
        
        # Apply LSTM Cell to generate sequence
        for t in range(T):

            # Pass through LSTM cell and update input-cell-hidden states
            h_t, c_t = self.lstm_cell(cur_input, (h_t, c_t))
            o_t = self.fc_out(c_t)

            # Save states
            c_states[:, t, :] = c_t
            h_states[:, t, :] = h_t
            outputs[:, t, :] = o_t

            # Update input at time step t+1
            cur_input = o_t

        # Return cell and hidden states
        return h_states, c_states, outputs


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
        self.lstm = LSTMCell(input_size=self.i_dim, hidden_dim=self.h_dim, num_layers=self.n_layers, dropout=self.dropout, **kwargs)
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
    def forward(self, x, y):
        """
        Model loss computation given batch objects during training. Note that we implement 
        a window size within our model itself. Time alignment is checked through pre-processing.

        Params:
            - x: pytorch Tensor object of input data with shape (batch_size, max_seq_len, input_size);
            - y: pytorch Tensor object of corresponding outcomes with shape (batch_size, outcome_size).

        Output:
            - loss: pytorch Tensor object of loss value.
            - history: dictionary of relevant training history.
        """

        # ========= Define relevant variables and initialise variables ===========

        x, y = x.to(self.device), y.to(self.device) # move data to device

        # Extract dimensions
        batch_size, seq_len, input_size = x.size()
        num_time_steps = int(seq_len / self.w_size)

        # Initialize the current hidden state as the null vector
        h = torch.zeros(batch_size, self.l_size, device=self.device)

        # Initialise probability of cluster assignment as uniform
        est_pi = torch.ones(batch_size, self.K, device=self.device) / self.K
        est_z = model_utils.gen_samples_from_assign(est_pi, self.c_means, self.log_c_vars)
                

        # Initialise loss value and history dictionary
        ELBO, history = 0, {}
        history["loss_loglik"] = torch.zeros(num_time_steps, device=x.device)
        history["loss_kl"] = torch.zeros(num_time_steps, device=x.device)
        history["loss_out"] = 0

        # Tracker for intermediate objects
        history["pis"] = torch.zeros(batch_size, seq_len, self.K, device=self.device)
        history["zs"] = torch.zeros(batch_size, seq_len, self.l_size, device=self.device)
        history["alpha_encs"] = torch.zeros(batch_size, seq_len, self.K, device=self.device)
        history["mugs"] = torch.zeros(batch_size, seq_len, self.i_size, device=self.device)
        history["log_vargs"] = torch.zeros(batch_size, seq_len, self.i_size, device=self.device)

        # ================== Iteration through time-steps ==============
        # Can also edit this to use a sliding window approach, where t goes from 0 to seq_len - w_size
        for window_id in range(num_time_steps):
            "Iterate through each window block"

            # Bottom and high indices
            lower_t, higher_t = window_id * self.w_size, (window_id + 1) * self.w_size

            # First we estimate the observations for the incoming window given current cell state
            h_dec, c_dec, data_gen = self.decoder_pass(h=h, z=est_z)

            # Decompose obvs_pred into mean and log-variance - shape is (bs, T, 2 * input_size)
            mu_g, logvar_g = torch.chunk(data_gen, chunks=2, dim=-1)
            var_g = torch.exp(logvar_g) + eps

            for _w_id, t in enumerate(range(lower_t, higher_t)):
                # Estimate alphas for each time step within the window. 

                # Subset observation to time t
                x_t = x[:, t, :]

                # Compute alpha based on prior gate
                alpha_prior = self.prior_out(self.prior(h)) + eps 

                # Compute alpha of encoder network
                alpha_enc = self.encoder_pass(h=h, x=x_t)


                # Sample cluster distribution from alpha_enc, and estimate samples from clusters
                est_pi = model_utils.sample_dir(alpha=alpha_enc)
                est_z = model_utils.gen_samples_from_assign(
                    pi_assign=est_pi, 
                    c_means=self.c_means, 
                    log_c_vars=self.log_c_vars
                )
                # pi = torch.divide(alpha_enc, torch.sum(alpha_enc, dim=-1, keepdim=True))

                # -------- COMPUTE LOSS FOR TIME t -----------

                # Data Likelihood component
                log_lik = LM_utils.torch_log_gaussian_lik(x_t, mu_g[:, _w_id, :], var_g[:, _w_id, :], device=self.device)

                # Posterior KL-divergence component
                kl_div = LM_utils.dir_kl_div(a1=alpha_enc, a2=alpha_prior)
                # quot = torch.log(torch.divide(alpha_enc, alpha_prior + 1e-8) + 1e-8)
                # kl_div = torch.mean(torch.sum(alpha_enc * quot, dim=-1))

                # Add to loss tracker
                ELBO += log_lik - kl_div


                # ----- UPDATE CELL STATE ------
                h = self.state_update_pass(h=h, x=x_t, z=est_z)

                # Append losses to history trackers
                history["loss_kl"][window_id] += torch.mean(kl_div, dim=0)
                history["loss_loglik"][window_id] += torch.mean(log_lik, dim=0)

                # Append objects
                history["pis"][:, t, :] = est_pi
                history["zs"][:, t, :] = est_z
                history["alpha_encs"][:, t, :] = alpha_enc
                history["mugs"][:, t, :] = mu_g[:, _w_id, :]
                history["log_vargs"][:, t, :] = logvar_g[:, _w_id, :]

        # Once all times have been computed, make predictions on outcome
        y_pred = self.predictor_out(self.predictor(est_z))
        history["y_preds"] = y_pred

        # Compute log loss of outcome
        pred_loss = LM_utils.cat_cross_entropy(y_true=y, y_pred=y_pred)

        # Add to total loss
        ELBO += pred_loss

        # Compute average per batch
        ELBO = torch.mean(ELBO, dim=0)

        # Append to history tracker
        history["loss_out"] += torch.mean(pred_loss)

        return (-1) * ELBO, history      # want to maximize loss, so return negative loss
    
    def fit(self, data_info, train_info, run_config):
        """
        Train model given data_dic with data information, and training parameters.

        Params:
            - data_info: dictionary with data information. Includes keys:
                a) 'X' - with triple (X_train, X_val, X_test) of observation data;
                b) 'y' - with triple (y_train, y_val, y_test) of outcome data;
                c) 
            - train_info: dictionary with training information. Includes keys:
                a) 'batch_size' - batch size for training;
                b) 'num_epochs' - number of epochs to train for;
                c) 'lr' - learning rate for training;
            - run_config: parameters used for loading data, model and training.

        Outputs:
            - loss: final loss value.
            - history: dictionary with training history, including each loss component.
        """
        # Set Weights and Biases Logger
        run_name = run_config['run_name']
        data_name = data_info['data_load_config']['data_name']
        wandb.init(
                name= "{}-{}".format(run_name, data_name),
                entity="hrna-ox", 
                dir=f"exps/Dir_VRNN/{data_info['data_load_config']['data_name']}",
                project="Dir_VRNN", 
                config=run_config
            )

        # Unpack data and make data loaders
        X_train, X_val, _ = data_info['X']
        y_train, y_val, _ = data_info['y']

        # Unpack train parameters
        batch_size, epochs, lr = train_info['batch_size'], train_info['num_epochs'], train_info['lr']

        # Prepare data for training
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Define optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        wandb.watch(self, log="all", log_freq=1)

        # ================== TRAINING-VALIDATION LOOP ==================
        for epoch in range(1, epochs + 1):

            # Set model to train mode and initialize loss tracker
            self.train()
            train_loss = 0
            train_loglik, train_kl, train_out = 0, 0, 0

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
                train_loglik += batch_loglik
                train_kl += batch_kl
                train_out += batch_out

                # Print message
                print("Train epoch: {}   [{:.5f} - {:.0f}%]".format(
                    epoch, loss.item(), 100. * batch_id / len(train_loader)),
                    end="\r")
                
            # Print message at the end of epoch
            epoch_loglik = torch.sum(train_loglik)
            epoch_kl = torch.sum(train_kl)
            epoch_out = torch.mean(train_out)

            print("Train epoch {} ({:.0f}%):  [L{:.5f} - loglik {:.5f} - kl {:.5f} - out {:.5f}]".format(
                epoch, 100, 
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
            
            # Check performance on validation set
            _, _ = self.validate(X_val, y_val, epoch=epoch)
    
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
                
                # Compute Loss
                val_loss, history_objects = self.forward(X, y)

                # Append to tracker
                log_lik = torch.mean(history_objects["loss_loglik"])
                kl_div = torch.mean(history_objects["loss_kl"])
                out_l = torch.mean(history_objects["loss_out"])

                # Print message
                print("Predict {} ({:.0f}%):  [L{:.5f} - loglik {:.5f} - kl {:.5f} - out {:.5f}]".format(
                    iter_str, 100, 
                    val_loss, log_lik, kl_div, out_l)
                )

                # Compute useful metrics and plots during training - including:
                # a) cluster assignment distribution, 
                # b) cluster means separability, 
                # c) accuracy, f1 and recall scores,
                # d) confusion matrix, 

                # Compute distribution over cluster memberships at this time step.
                "TO DO"

                # Compute cluster means separability
                avg_clus_dist = model_utils.torch_clus_means_separability(clus_means=self.c_means)
                #tsne_projs = model_utils.torch_clus_mean_2D_tsneproj(clus_means=self.c_means, seed=self.seed)
                #tsne_to_fmt = wandb.Table(data=tsne_projs, columns=["tsne dim 1", "tsne dim 2"])

                # Compute accuracy, f1 and recall scores
                y_pred = history_objects["y_preds"]
                acc = LM_utils.accuracy_score(y, y_pred)
                macro_f1 = LM_utils.f1_multiclass(y, y_pred, mode="macro")
                micro_f1 = LM_utils.f1_multiclass(y, y_pred, mode="micro")

                # Compute confusion matrix, ROC and PR curves
                pr_curve = wandb.plot.pr_curve(torch.argmax(y,dim=-1), y_pred, labels=["A", "B", "C", "D"])
                roc_curve = wandb.plot.roc_curve(torch.argmax(y, dim=-1), y_pred, labels=["A", "B", "C", "D"])
            
                # Log objects to Weights and Biases
                wandb.log({
                        "val/epoch": epoch + 1,
                        "val/loss": val_loss,
                        "val/loglik": log_lik,
                        "val/kldiv": kl_div,
                        "val/out_l": out_l,
                        "val/avg_clus_dist": avg_clus_dist,
                        #f"{epoch}/tsne_projs": wandb.plot(tsne_to_fmt, "dim 1", "dim 2", title="Cluster Means TSNE Projection"),
                        "val/acc": acc,
                        "val/macro_f1": macro_f1,
                        "val/micro_f1": micro_f1,
                        "val/Precision_Recall": pr_curve,
                        "val/Receiver_Operating_Char": roc_curve
                    },
                    step=epoch+1
                )	        
    
            return val_loss, history_objects
        
    def predict(self, X, y):
        """Similar to forward, but focus on the computations of the model."""
        _, history_objects = self.forward(X, y)

        # Extract objects from history
        est_pis = history_objects["pis"]
        mugs, log_vargs = history_objects["mugs"], history_objects["log_vargs"]

        # Get phenotypes and append
        history_objects["prob_phens"] = model_utils.torch_get_temp_phens(est_pis, y_true=y, mode="prob")
        history_objects["clus_phens"] = model_utils.torch_get_temp_phens(est_pis, y_true=y, mode="one-hot")

        # Sample from the generated mu_g and var_g
        history_objects["x_samples"] = model_utils.gen_diagonal_mvn(mugs, log_vargs)

        # Log to wandb
        wandb.log({
            "":""
        })

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
        return self.prior_out(self.prior(h))

    def predictor_pass(self, z):
        return self.predictor_out(self.predictor(z))
# endregion
        