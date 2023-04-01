#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

# =========== IMPORT LIBRARIES =============

# Data and Models
import torch.nn as nn

# ============== UTILITY FUNCTIONS =====================


# ============== USEFUL UTILITY CLASSES ================

# Define Multilayer perceptron
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=1, hidden_nodes=20,
                 act_fn="relu", output_fn="sigmoid"):
        """
        input_size: format of input data
        output_size: format of output data
        hidden_layers: number of hidden_layers (default = 1)
        hidden_nodes: number of nodes of hidden layers (default = 20)
        act_fn: activation function for hidden layers (Default = "relu")
        output_fn: activation function for output layer (default = "sigmoid")
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
            "output_fn": output_fn
        }

        # Activation functions
        self.act_fn = self._get_activation(act_fn)
        self.output_fn = self._get_activation(output_fn)

        # Initialise first layer
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_nodes))

        # Add Layers Iteratively
        for _id in range(hidden_layers):
            self.layers.append(
                nn.Linear(hidden_nodes, hidden_nodes)
            )

        # Add output layer
        self.layers.append(nn.Linear(hidden_nodes, output_size))

    def forward(self, x):
        """Model call - given input"""

        # Intermediate placeholder
        x_ = x

        # Iterate through layers
        for layer in self.layers[:-1]:
            x_ = self.act_fn(layer(x_))

        # Output Layer is the last layer
        output = self.output_fn(self.layers[-1](x_))

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
