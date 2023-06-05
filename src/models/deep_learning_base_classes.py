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
import torch.nn as nn

# Typing support
from typing import Tuple

#endregion


# region 1 - define MLP

class MLP(nn.Module):
    """
    The Multi-Layer Perceptron (MLP) is a feedforward neural network that maps an input to an output through
    a sequence of hidden layers. The number of hidden layers and corresponding number of nodes is defined 
    as a parameter to the model.
    """
    def __init__(self, 
                input_size: int,
                output_size: int,
                hidden_layers: int = 1, 
                hidden_nodes: int = 20,
                act_fn: str = "relu"):
        """
        input_size: format of input data
        output_size: format of output data
        hidden_layers: number of hidden_layers (default = 1) (e.g. 0 -> one single layer total)
        hidden_nodes: number of nodes of hidden layers (default = 20)
        act_fn: activation function for hidden layers (default = "relu")
        """

        # initialise 
        super().__init__()

        # Load parameters to configuration
        self.config = {
            "input_size": input_size,
            "output_size": output_size,
            "hidden_layers": hidden_layers,
            "hidden_nodes": hidden_nodes,
            "act_fn": act_fn,
        }

        # Base definitions 
        self.act_fn = self._get_activation(act_fn)     # Load activation function
        self.nn_layers = nn.ModuleList()

        # Iteratively add hidden layers
        for _id in range(hidden_layers):

            # Define hidden layers
            _inter_input_size = input_size if _id == 0 else hidden_nodes
            hidden_layer = nn.Linear(in_features=_inter_input_size, on_features=hidden_nodes)

            # Append to list of nn layers.
            self.nn_layers.append(hidden_layer)

        # Add output layer
        self.layers.append(nn.Linear(hidden_nodes, output_size))


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
        for layer_i in self.layers[:-1]:

            # Apply layer transformation
            inter_x = layer_i(_x)                # Layer defines map
            _x= self.act_fn(inter_x)

        # Output Layer is the last layer
        output = self.output_fn(self.layers[-1](_x))

        return output

    @staticmethod
    def _get_activation(fn_str):
        """Get pytorch activation function from string"""

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

return Error
# endregion

# region 2 - define LSTM

class LSTM(nn.Module):
    pass 

# endregion 