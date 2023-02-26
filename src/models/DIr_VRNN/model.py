"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

Main model file for proposed model for Dirichlet Variational Recurrent Neural Network method.
Main model is represented as a class, and then a wrapper class is added at the end to add to the central pipeline.
"""

# ============= IMPORT LIBRARIES ==============
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        1/alphas
    )

    # Output generated samples
    return gamma_samples



# ============= MAIN MODEL DEFINITION =============

class VRNN(nn.Module):
    def _init_(self, input_size, outcome_size, latent_size, gate_layers, gate_nodes, 
               feat_extr_layers, feat_extr_nodes, num_clus):
        super(VRNN, self)._init_()

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
        self.encoder = MLP(input_size = self.latent_dim + self.latent_dim, 
                           output_size = self.K,
                            act_fn = "relu", 
                            hidden_layers = self.gate_layers,
                            hidden_nodes=self.gate_nodes,
                            output_fn = "relu")
        

        # Define the decoder network - computes mean, var of Observation data
        self.decoder = MLP(input_size = self.latent_dim + self.latent_dim,     # input is z and h
                           output_size = self.input_size + self.input_size,    # output is mean/var of observation
                           hidden_layers = gate_layers,
                           hidden_nodes = gate_nodes,
                           act_fn = "relu",
                           output_fn = "tanh")


        # Define the prior network
        self.prior = MLP(input_size = self.latent_dim,
                         output_size = self.K,
                         act_fn = "relu",
                         hidden_layers = self.gate_layers,
                         hidden_nodes = self.gate_nodes,
                         output_fn = "relu")
        

        # Define the output network
        self.predictor = MLP(input_size = self.latent_dim,
                             output_size = self.outcome_size,
                             hidden_layers = gate_layers,
                             hidden_nodes = gate_nodes,
                             act_fn = "relu",
                             output_fn = "tanh")
        
        
        # Define feature extractors
        self.phi_x = MLP(input_size = self.input_size,
                         output_size = self.latent_dim,
                         hidden_layers = feat_extr_layers,
                         hidden_nodes = self.feat_extr_nodes,
                         act_fn = "relu",
                         output_fn = "tanh")

        self.phi_z = MLP(input_size = self.latent_dim,
                         hidden_layers= feat_extr_layers,
                         hidden_nodes= feat_extr_nodes,
                         act_fn = "relu",
                         output_fn = "tanh")


        # Recurrency
        self.rnn = MLP(input_size = self.latent_dim + self.latent_dim + self.latent_dim,
                       output_size = self.latent_dim,
                       hidden_layers = self.gate_layers,
                       hidden_nodes = self.gate_nodes,
                       act_fn = "relu",
                       output_fn = "tanh")
        

    # What to do given input
    def forward(self, x):

        # Extract dimensions
        batch_size, seq_len, _ = x.size()

        # Initialize the current hidden state and latent variable and loss
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        z = torch.zeros(batch_size, self.latent_size).to(x.device)
        loss = 0


        # Iterate over the sequence
        for t in range(seq_len):

            # Observation data
            x_t = x[:, t, :]
            
            # Compute encoder alpha by extracting features from input and concatenating with hidden state
            joint_enc_input = torch.cat([
                        self.phi_x(x_t),        # Extract feature from x
                        h],                            
                        dim = -1                       # Concatenate with cell state in last dimension
                        )
            alpha_enc = self.encoder(joint_enc_input)

            # Sample mechanism
            pi_samples = sample_dirichlet(alpha=alpha_enc)

            # Average over clusters to estimate latent variable
            z_samples = torch.matmul(pi_samples, clus_means)
            

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


            # -------- COMPUTE LOSS FOR TIME t -----------

            # Data Likelihood component
            log_lik = torch.log(
                torch.divide(
                    x_t - mu_g  ,
                )
            )
            

            if t == seq_len - 1:
                pass
            
            else:
                pass
                # Update Cell State


            # Compute the prior distribution
            p_z = self.prior(h)

            # Compute the posterior distribution
            input_z = torch.cat([x[:, t, :], h], dim=-1)
            q_z = self.posterior(input_z)

            # Compute the KL divergence loss
            mu_z, logvar_z = torch.chunk(q_z, 2, dim=-1)
            kl_loss = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

            # Decode the latent variable and hidden state
            input_zh = torch.cat([z, h], dim=-1)
            x_tilde = self.dgecoder(input_zh)

            # Compute the reconstruction loss
            recon_loss = F.binary_cross_entropy(x_tilde, x[:, t, :], reduction='sum')

            # Update the hidden state
            _, h = self.rnn(x[:, t, :].unsqueeze(0), h)

            # Add the losses to the total loss
            loss += kl_loss + recon_loss

        return loss / batch_size