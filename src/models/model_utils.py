#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:48:57 2021

@author: henrique.aguiar@ds.ccrg.kadooriecentre.org
"""

# =========== IMPORT LIBRARIES =============

# Data and Models
import torch
import torch.nn as nn

# Import Models
import os
from src.models.camelot.model import Model as CamelotModel


# ============== UTILITY FUNCTIONS =====================

# Load model given specified configuration and parameters
def get_model_from_str(data_info: dict, model_config: dict, training_config: dict):
    """
    Function to load correct model from the model name.

    Params:
    - data_info: dictionary with input data information.
    - model_config: model_configuration dictionary
    - training_config: model training configuration dictionary.

    returns: Corresponding model class object.
    """
    model_name = model_config["model_name"]
    gpu = training_config["gpu"] if "gpu" in training_config.keys() else None

    # Load the corresponding model
    if "camelot" in model_name.lower():

        # Check if GPU is accessible
        if gpu is None or gpu == 0:

            # Train only on CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = CamelotModel(data_info=data_info, model_config=model_config, training_config=training_config)

        # If GPU usage
        else:

            # Identify physical devices and limit memory growth
            physical_devices = tf.config.list_physical_devices('GPU')[0]
            print("\nPhysical Devices for Computation: ", physical_devices, sep="\n")
            tf.config.experimental.set_memory_growth(physical_devices, True)

            # If distributed strategy
            if gpu == "strategy":

                # Load strategy
                strategy = tf.distribute.MirroredStrategy(devices=None)

                with strategy.scope():
                    model = CamelotModel(data_info=data_info, model_config=model_config,
                                         training_config=training_config)

            else:
                model = CamelotModel(data_info=data_info, model_config=model_config, training_config=training_config)

    # ACTPC
    elif "actpc" in model_name.lower():

        # Check if GPU is accessible
        if gpu is None or gpu == 0:

            # Train only on CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = ActpcModel(data_info=data_info, model_config=model_config, training_config=training_config)

        # If GPU usage
        else:

            # Identify physical devices and limit memory growth
            physical_devices = tf.config.list_physical_devices('GPU')[0]
            print("\nPhysical Devices for Computation: ", physical_devices, sep="\n")
            tf.config.experimental.set_memory_growth(physical_devices, True)

            # If distributed strategy
            if gpu == "strategy":

                # Load strategy
                strategy = tf.distribute.MirroredStrategy(devices=None)

                with strategy.scope():
                    model = ActpcModel(data_info=data_info, model_config=model_config,
                                       training_config=training_config)

            else:
                model = ActpcModel(data_info=data_info, model_config=model_config, training_config=training_config)

    # Encoder Predictor Model
    elif "enc" in model_name.lower() and "pred" in model_name.lower():

        # Check if GPU is accessible
        if gpu is None or gpu == 0:

            # Train only on CPU
            # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            model = EncPredModel(data_info=data_info, model_config=model_config, training_config=training_config)

        # If GPU usage
        else:

            # Identify physical devices and limit memory growth
            physical_devices = tf.config.list_physical_devices('GPU')[0]
            print("\nPhysical Devices for Computation: ", physical_devices, sep="\n")
            tf.config.experimental.set_memory_growth(physical_devices, True)

            # If distributed strategy
            if gpu == "strategy":

                # Load strategy
                strategy = tf.distribute.MirroredStrategy(devices=None)

                with strategy.scope():
                    model = EncPredModel(data_info=data_info, model_config=model_config,
                                         training_config=training_config)

            else:
                model = EncPredModel(data_info=data_info, model_config=model_config, training_config=training_config)

    elif "svm" in model_name.lower() and "all" in model_name.lower():
        model = SVMAll(data_info=data_info, **model_config)

    elif "svm" in model_name.lower() and "feat" in model_name.lower():
        model = SVMFeat(data_info=data_info, **model_config)

    elif "xgb" in model_name.lower() and "all" in model_name.lower():
        model = XGBAll(data_info=data_info, **model_config)

    elif "xgb" in model_name.lower() and "feat" in model_name.lower():
        model = XGBFeat(data_info=data_info, **model_config)

    elif "tskm" in model_name.lower():
        model = TSKM(data_info=data_info, **model_config)

    elif "news" in model_name.lower():
        model = NEWS(data_info=data_info, **model_config)

    else:
        raise ValueError(f"Correct Model name not specified. Value {model_name} given.")

    return model



# ============== USEFUL UTILITY CLASSES ================

# Define Multilayer perceptron
class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers = 1, hidden_nodes = 20, 
                    act_fn = "relu", output_fn = "sigmoid"):
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
        "Model call - given input"

        # Intermediate placeholder
        x_ = x

        # Iterate through layers
        for layer in self.layers[:-1]:
            x_ = self.act_fn(layer(x_))

        # Output Layer
        output = self.output_fn(layer(x_))

        return output
    

    def _get_activation(self, fn_str):
        "Get pytorch activation function from string"

        if fn_str.lower() == "relu":
            return nn.ReLU()
        
        elif fn_str.lower() == "sigmoid":
            return nn.Sigmoid()
        
        elif fn_str.lower() == "tanh":
            return nn.Tanh()
        
        elif fn_str.lower() == "identity" or fn_str is None:
            return nn.Identity()
        
        else:
            raise ValueError(f"activation function value {fn_str} not allowed.")