"""
Author: Henrique Aguiar
email: henrique.aguiar@eng.ox.ac.uk

Auxiliary Functions to train models
"""

# ============= Import Libraries =============
from src.models.DirVRNN.train import DirVRNN


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
    model_name = model_config["model_name"].lower()
    model = "svm"  # Empty placeholder for now

    if "dir" in model_name and "vrnn" in model_name:
        model = DirVRNN(data_info, model_config, training_config)

    return model
