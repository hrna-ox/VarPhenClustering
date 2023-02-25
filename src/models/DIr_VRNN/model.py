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



# ============= MAIN MODEL DEFINITION =============

class VRNN(nn.Module):
    def _init_(self, input_size, latent_size, gate_layers, gate_nodes, 
               feat_extr_layers, feat_extr_nodes, num_clus):
        super(VRNN, self)._init_()

        # Define the dimensions of the input, hidden, and latent variables
        self.input_size = input_size
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
                           output_size = self.latent_dim,
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

        # Initialize the hidden state and latent variable and loss
        h = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        z = torch.zeros(batch_size, self.latent_size).to(x.device)
        loss = 0


        # Iterate over the sequence
        for t in range(seq_len):
            
            # Encode the input and hidden state
            input_hidden = torch.cat([x[:, t, :], h], dim=-1)
            q_z = self.encoder(input_hidden)

            # Sample the latent variable
            mu_z, logvar_z = torch.chunk(q_z, 2, dim=-1)
            eps = torch.randn_like(mu_z)
            z = mu_z + torch.exp(0.5 * logvar_z) * eps

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
            x_tilde = self.decoder(input_zh)

            # Compute the reconstruction loss
            recon_loss = F.binary_cross_entropy(x_tilde, x[:, t, :], reduction='sum')

            # Update the hidden state
            _, h = self.rnn(x[:, t, :].unsqueeze(0), h)

            # Add the losses to the total loss
            loss += kl_loss + recon_loss

        return loss / batch_size