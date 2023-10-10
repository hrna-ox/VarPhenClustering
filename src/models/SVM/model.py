"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Implements two Class objects derived from Support Vector Machines to classify multi-dimensional time-series using 2 different approaches.
"""

# Import required packages
import pickle
import numpy as np
from sklearn.svm import SVC as sklearn_SVC

import os

class GlobalClassifier:
    """
    SVM-based model for time-series analysis. Assumes that each (time, feature) pair is an independent feature.
    """
    def __init__(self, input_shape, output_dim, model_name: str, *args, **kwargs):
        """
        Initialize the model.

        Args:
            input_shape (tuple): Input shape.
            output_dim (int): Output dimension.
            model_name (str): Name of the model to use. Specifies whether to use XGBClassifier or sklearn_SVC.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_name = model_name

        # Initialize the SVM model
        self.model = sklearn_SVC(*args, **kwargs, verbose=True, probability=True)


    def train(self, train_data: tuple[np.ndarray, np.ndarray]):
        """
        Train the model.

        Args:
            train_data (tuple): Tuple containing the training data and labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Outputs:
            None, trains the model.
        """

        # Unpack the training data and reshape according to the 2D design
        X_train, y_train = train_data
        X_train = X_train.reshape(X_train.shape[0], -1)

        # Train the model
        print(f"\n\nTraining {self.model_name} model...\n\n")
        self.model.fit(X_train, y_train)
    
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
        
        # Reshape the input data according to the 2D design
        X = X.reshape(X.shape[0], -1)
        
        # Predict scores
        y_pred = self.model.predict_proba(X)
        
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
            "model_params": self.model.get_params(),
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
        self.model = models_dic["model"].set_params(**models_dic["model_params"])

        return run_info, data_objects


class ParallelClassifier:
    """
    Implements ParallelClassifier, a model for handling time-series analysis which creates an ensemble model of Classifiers, each dedicated to training individual features, and 
    then combines the predictions of each model into a single score
    """
    def __init__(self, input_shape, output_dim, model_name: str, *args, **kwargs):
        """
        Initialize the model.

        Args:
            input_shape (tuple): Input shape.
            output_dim (int): Output dimension.
            model_name (str): Name of the model. Specifies whether to use XGBoost or SVM
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_name = model_name
        self.num_feats = input_shape[1]

        # Initialize the SVM model
        self.models = {
            feat_idx: sklearn_SVC(*args, **kwargs, verbose=True, probability=True)
            for feat_idx in range(self.num_feats)
        }
        
    def train(self, train_data: tuple[np.ndarray, np.ndarray], *args, **kwargs):
        """
        Train the model.

        Args:
            train_data (tuple): Tuple containing the training data and labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Outputs:
            None, trains the model.
        """

        # Unpack the training data
        X_train, y_train = train_data
        assert self.num_feats == X_train.shape[-1]

        # Train the model
        print(f"\n\nTraining {self.model_name} model...\n\n")
        for feat_idx in range(self.num_feats):
            self.models[feat_idx].fit(X_train[:, :, feat_idx], y_train)
    
    def predict(self, X, y=None, *args, **kwargs):
        """
        Predict the labels for the input data.
        
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        
        # Check that the number of features in the input data matches the number of features in the model
        assert self.num_feats == X.shape[-2]

        # Predict the labels for each feat
        _dic_placeholder = {}
        for feat_idx in range(self.num_feats):
            _dic_placeholder[feat_idx] = self.models[feat_idx].predict_proba(X[:, :, feat_idx])
        
        # Compute average over values
        output_arr = np.zeros_like(_dic_placeholder[0])
        for feat_idx in range(self.num_feats):
            output_arr += _dic_placeholder[feat_idx]
        
        # Average over features
        y_pred = output_arr / self.num_feats

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
            "model_name": self.model_name,
            "num_feats": self.num_feats,
            "model_params": {
                feat_idx: self.models[feat_idx].get_params() 
                for feat_idx in range(self.num_feats)
            },
            "model": {
                feat_idx: self.models[feat_idx]
                for feat_idx in range(self.num_feats)
            }
        }
        with open(save_dir + "models.pkl", "wb") as f:
            pickle.dump(models, f)

        # Save additional objects
        if objects_to_log != {}:
            with open(save_dir + "data_objects.pkl", "wb") as f:
                pickle.dump(objects_to_log, f)
    
        # Save run information
        if run_info != {}:
            with open(save_dir + "run_info.pkl", "wb") as f:
                pickle.dump(run_info, f)

    def load_run(self, save_dir: str) -> tuple:
        """
        Load model objects. Useful for reproducibility.

        Args:
            save_dir (str): Directory to save the objects.
        
        Returns:
            None, loads the objects.
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
        self.model_name = models_dic["model_name"]
        self.num_feats = models_dic["num_feats"]
        self.models = {
            feat_idx: models_dic["model"][feat_idx].set_params(**models_dic["model_params"][feat_idx])
            for feat_idx in range(self.num_feats)
        }

        return run_info, data_objects
