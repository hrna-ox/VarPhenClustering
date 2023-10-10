"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

Implements TSKM Model Class
"""

# Import required packages
import pickle, os
import numpy as np

from tslearn.clustering import TimeSeriesKMeans

class TSKM:
    """
    Time Series K-Means for Clustering Multi-Dimensional Time Series Data.
    """
    def __init__(self, input_shape, output_dim, model_name: str, K: int, random_state: int, metric: str = "dtw",
                 *args, **kwargs):
        """
        Initialize the model.

        Args:
            input_shape (tuple): Input shape.
            output_dim (int): Output dimension.
            model_name (str): Name of the model to use. Specifies whether to use XGBClassifier or sklearn_SVC.
            K (int): Number of clusters.
            random_state (int): Random state.
            metric (str): Metric to use for clustering.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.model_name = model_name
        self.K = K
        self.random_state = random_state
        self.metric = metric
        self.cluster_probs = np.zeros(shape=(K, output_dim))

        # Initialize GMM Model
        self.model = TimeSeriesKMeans(
            n_clusters = K,
            metric = metric,
            n_jobs = -1,
            init = "k-means++",
            random_state = random_state,
            *args, **kwargs
        )


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

        # Unpack the training data and reshape observations to 2D
        X_train, y_train = train_data

        # Train the model
        print(f"\n\nTraining {self.model_name} model...\n\n")
        self.model.fit(X_train)

        # Update the cluster probabilities given the training outcomes
        X_train_clusters = self.model.predict(X_train)

        for cluster_idx in range(self.K):
            for outcome_idx in range(self.output_dim):
                self.cluster_probs[cluster_idx, outcome_idx] = np.mean(y_train[X_train_clusters == cluster_idx, outcome_idx])

    
    def predict(self, X, y=None, *args, **kwargs):
        """
        Predict the labels for the input data.
        
        Args:
            X (np.ndarray): Input data.
            y (np.ndarray): True labels.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        
        Returns:
            np.ndarray: Predicted labels of shape (num_samples, num_outcomes).
        """
        
        # Reshape the input data according to the 2D design
        clus_pred = self.model.predict(X)

        # Compute the predicted labels given the cluster assignments and the cluster probabilities
        y_scores = np.eye(self.K)[clus_pred] @ self.cluster_probs
        
        return y_scores, clus_pred
    
    def get_model_objects(self):

        # Unpack and convert
        means_ts = self.model.cluster_centers_

        return means_ts, self.cluster_probs

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
            "K": self.K,
            "random_state": self.random_state,
            "cluster_probs": self.cluster_probs,
            "cluster_centers": self.model.cluster_centers_,
            "model_params": self.model.get_params(),
            "model": self.model
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
    
    
    def load_run(self, save_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load model objects. Useful for reproducibility.

        Args:
            save_dir (str): Directory to save the objects.
        
        Returns:
            Tuple [X, y_true, y_pred] of data objects, as well as the trained model.
        """
        
        # First load the model objects
        with open(save_dir + "data_objects.pkl", "rb") as f:
            data_objects = pickle.load(f)
        
        # Unpack the model objects
        X = data_objects["X"]
        y_true = data_objects["y_true"]
        y_pred = data_objects["y_pred"]

        # Then load model configuration and parameters
        with open(save_dir + "models.pkl", "rb") as f:
            models = pickle.load(f)
        
        # Unpack the model objects
        self.input_shape = models["input_shape"]
        self.output_dim = models["output_dim"]
        self.model = models["model"].set_params(**models["model_params"])

        return X, y_true, y_pred
