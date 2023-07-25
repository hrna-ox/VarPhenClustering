#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Henrique Aguiar
Date: 05/06/2023 

This file defines base classes for more complex deep learning models, namely:

a) Multi-layer perceptron (MLP)
b) Long Short Term Memory (LSTM)
"""

# region 0 - Imports 
import torch
import torch.nn as nn

from torch.nn import LSTMCell

#endregion


# region 1 - define MLP

class MLP(nn.Module):
    """
    The Multi-Layer Perceptron (MLP) is a feed-forward neural network that maps an input to an output through
    a sequence of hidden layers. The number of hidden layers and corresponding number of nodes is defined 
    as a parameter to the model.
    """
    def __init__(self, 
                input_size: int,
                output_size: int,
                hidden_layers: int = 1, 
                hidden_nodes: int = 20,
                act_fn = nn.ReLU(),
                dropout: float = 0.0,
                bias: bool = True):
        """
        Params:
            - input_size: format of input data
            - output_size: format of output data
            - hidden_layers: number of hidden_layers (default = 1) (e.g. 0 -> one single layer total)
            - hidden_nodes: number of nodes of hidden layers (default = 20)
            - act_fn: activation function for hidden layers (default = nn.ReLU())
            - dropout: dropout rate (default = 0.0)
            - bias: whether to use bias or not (default = True)
        """

        # initialise 
        super().__init__()

        # Base definitions 
        self.nn_layers = nn.ModuleList()
        self.act_fn = self._get_activation(act_fn)
        self.dropout = nn.Dropout(dropout)

        # Iteratively add hidden layers
        for _id in range(hidden_layers):

            # Define hidden layers
            _inter_input_size = input_size if _id == 0 else hidden_nodes
            hidden_layer = nn.Linear(in_features=_inter_input_size, out_features=hidden_nodes, bias=bias)

            # Append to list of nn layers.
            self.nn_layers.append(hidden_layer)

        # Add output layer
        self.nn_layers.append(nn.Linear(hidden_nodes, output_size))


    def forward(self, x):
        """
        Forward pass of the MLP model.

        Params:
        - x: input data of shape (batch_size, input_size)
        
        Returns:
        - output: output data of shape (batch_size, output_size)
        """

        # Intermediate placeholder
        _x = x

        # Iterate through layers
        for layer_i in self.nn_layers[:-1]: # type: ignore

            # Apply layer transformation
            inter_x = layer_i(_x)                # Layer defines map
            inter_x= self.act_fn(inter_x)
            _x = self.dropout(inter_x)

        # Output Layer is the last layer
        output = self.nn_layers[-1](_x)

        return output

    @staticmethod
    def _get_activation(fn_str):
        """
        Get pytorch activation function from string, otherwise assume it is a torch activation function already.
        """

        if isinstance(fn_str, str):
                
            if fn_str.lower() == "relu":
                return nn.ReLU()
            
            elif fn_str.lower() == "softmax":
                return nn.Softmax()

            elif fn_str.lower() == "sigmoid":
                return nn.Sigmoid()

            elif fn_str.lower() == "tanh":
                return nn.Tanh()

            elif fn_str.lower() == "identity" or fn_str is None:
                return nn.Identity()

            else:
                raise ValueError(f"activation function value {fn_str} not allowed.")
        else:
            return fn_str # Assume it is a torch activation function already

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
        self.dropout = dropout

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
        seq_len: int,     # Number of observations to generate
        input_dim: int,  # Dimensionality of input vectors
        output_dim: int, # Dimensionality of output vector
        hidden_dim: int, # Dimensionality of hidden state
        num_layers: int = 1,  # Number of layers (default = 1)
        dropout: float = 0.0, # Dropout rate (default = 0.0, i.e. no dropout)
        **kwargs):

        # Call parent class constructor
        super().__init__()

        # Initialise parameters
        self.seq_len = seq_len
        self.i_dim, self.o_dim, self.h_dim = input_dim, output_dim, hidden_dim
        self.n_layers, self.dropout = num_layers, dropout

        # Define main LSTM layer and output layer
        self.lstm = LSTMCell(input_size=self.i_dim, hidden_size=self.h_dim, **kwargs)
        self.fc_out = nn.Linear(self.h_dim, self.o_dim)  

    # Forward pass
    def forward(self, context_vector: torch.Tensor):
        """
        Forward pass of v2 LSTM Decoder. Given context vector, use it as input sequence for LSTM.
        
        Params:
            - context_vector: context vector of fixed dimensionality, of shape (N, D), where N is batch size and D is latent dimensionality.
        """
        #  Get parameters
        bs, T = context_vector.shape[0], int(self.seq_len)

        # Define input sequence for LSTM
        input_seq = context_vector.unsqueeze(1).expand(-1, T, -1) # (N, T, D)
        batch_size = input_seq.shape[0]

        # Pass through LSTM
        h0 = torch.zeros(bs, T, self.h_dim, device=context_vector.device)
        c0 = torch.zeros(bs, T, self.h_dim, device=context_vector.device)
        output, _ = self.lstm(input_seq, (h0, c0)) # (N, T, D)

        return output
# endregion

