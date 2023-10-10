"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Implements ESI Classification Model.
"""

# Import required packages
import pickle
import numpy as np

import os

class ESI:
    """
    ESI Classifier. Returns the ESI at the last time point.
    """
    def __init__(self, input_shape, output_dim, ESI_idx: int = 0, *args, **kwargs):
        """
        Initialize the model.

        Args:
            input_shape (tuple): Input shape.
            output_dim (int): Output dimension.
            ESI_idx (int): Index of the ESI feature.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.ESI_idx = ESI_idx

        # Initialize the SVM model
        self.model = None


    def train(self, train_data: tuple[np.ndarray, np.ndarray]):
        """
        Used for legacy. No action needed since ESI has no trainable parameters.
        """
        print("\n\nTraining ESI model...\n\n")

        return None
    
    def predict(self, X, y=None):
        """
        Predict the labels for the input data.
        
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            np.ndarray: Predicted scores of shape (N, num_classes).
        """
        
        # Identify the ESI feature and return data.
        
        # Predict scores
        y_pred = X[:, -1, self.ESI_idx]
        
        return y_pred

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
            "input_shape": self.input_shape,
            "output_dim": self.output_dim,
            "ESI_idx": self.ESI_idx,
            "model": self.model
        }

        with open(save_dir + "models.pkl", "wb") as f:
            pickle.dump(models, f)

        # Save additional objects including any arrays and vectors
        if objects_to_log != {}:
            with open(save_dir + "data_objects.pkl", "wb") as f:
                pickle.dump(objects_to_log, f)

        # Save run information including metrics
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
        self.ESI_idx = models_dic["ESI_idx"]

        return run_info, data_objects
