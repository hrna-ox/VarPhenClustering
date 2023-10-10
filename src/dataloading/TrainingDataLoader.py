"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file implements a data Loader with method extensions for aiding model training.
"""

# Import required packages
import numpy as np

from typing import Union, Dict
from sklearn.model_selection import train_test_split, StratifiedKFold

import src.dataloading.data_loading_utils as data_utils
from src.dataloading.BaseDataLoader import BaseDataLoader



class TrainingDataLoader(BaseDataLoader):
    """
    Child object of Base Data Loader that implements specific parameters for preparing data to train models.
    """
    def __init__(self, data_dir: str, time_window: tuple[int, int], feat_subset: Union[list[str], str],
                train_test_ratio: float, train_val_ratio: float, seed: int, num_folds: int = 1, normalize: bool = True, *args, **kwargs):
        """
        Initializes object.

        Args:
            data_dir (str): Directory where data is stored (inherited)
            time_window (tuple[int, int]): Time window to consider for each patient (inherited)
            feat_subset (Union[list[str], str]): List of features to consider (inherited)
            train_test_ratio (float): Ratio of train (ALL of train, including test) to test data
            train_val_ratio (float): Ratio of train to validation data
            num_folds (int): Number of folds to use for cross validation
            normalize (bool): Whether to normalize data or not, defaults to True
            seed (int): Random seed for reproducibility
        """
        
        # Load new attributes related to training
        self.train_test_ratio = train_test_ratio
        self.train_val_ratio = train_val_ratio
        self.num_folds = num_folds
        self.normalize = normalize
        self.seed = seed

        # Initialize parent class
        super().__init__(data_dir=data_dir, time_window=time_window, feat_subset=feat_subset, *args, **kwargs)
        super().transform(time_window=time_window, feat_subset=feat_subset)

        # Compute Cross Validation
        self.train_val_test_data = self._split_data()

    def _split_data(self) -> Dict:
        """
        Split data to obtain a single test-dataset, and such that the training data is split into train-validation folds for parameter optimization.

        Returns:
        - dic with keys:
            "Fold_i": each containing further two keys:
                a1) "train": tuple(X_tr, y_tr, pat_ids_tr, time_ids_tr)
                a2) "val": tuple(X_val, y_val, pat_ids_val, time_ids_val)
                a3) "test": tuple(X_te, y_te, pat_ids_te, time_ids_te)

        Note that the test set is common for all folds, but this helps with ease of coding.
        """

        # Unpack data dictionary
        X_arr, y_arr, pat_ids_arr, time_ids_arr = self.data_dic.values()
        output_dic = {}

        # First split data into training data and a hold-out test set
        X_tr, X_te, y_tr, y_te, pat_ids_tr, pat_ids_te = train_test_split(
            X_arr, y_arr, pat_ids_arr,
            train_size=self.train_test_ratio, random_state=self.seed, 
            stratify=np.argmax(self.data_dic["y_arr"], axis=-1)
        )
        output_dic["global_split"] = {
            "train": (X_tr, y_tr, pat_ids_tr),
            "test": (X_te, y_te, pat_ids_te)
        }

        # Apply Cross Validation Split (or regular train_test_split if number of folds is 1)
        if self.num_folds > 1:
            skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.seed)

            # Iterate over folds
            for idx, (train_idx, val_idx) in enumerate(skf.split(X_tr, np.argmax(y_tr, axis=-1))):

                # Apply indices to train, val, data
                X_train, X_val = X_tr[train_idx, ...], X_tr[val_idx, ...]
                y_train, y_val = y_tr[train_idx, ...], y_tr[val_idx, ...]
                pat_ids_train, pat_ids_val = pat_ids_tr[train_idx, ...], pat_ids_tr[val_idx, ...]

                # Normalize
                if self.normalize:
                    X_train, min_arr, max_arr = data_utils.min_max_normalize(array=X_train, _min=None, _max=None)
                    X_val, _, _ = data_utils.min_max_normalize(array=X_val, _min=min_arr, _max=max_arr)
                    X_te, _, _ = data_utils.min_max_normalize(array=X_te, _min=min_arr, _max=max_arr)

                # Save data
                output_dic[f"Fold_{idx}"] = {
                    "train": (X_train, y_train, pat_ids_train),
                    "val": (X_val, y_val, pat_ids_val),
                    "test": (X_te, y_te, pat_ids_te)
                }
        
        elif self.num_folds == 1:
            
            X_train, X_val, y_train, y_val, pat_ids_train, pat_ids_val = train_test_split(
                X_tr, y_tr, pat_ids_tr,
                train_size=self.train_val_ratio, random_state=self.seed,
                stratify=np.argmax(y_tr, axis=-1)
            )
        
            # Normalize
            if self.normalize:
                X_train, min_arr, max_arr = data_utils.min_max_normalize(array=X_train, _min=None, _max=None)
                X_val, _, _ = data_utils.min_max_normalize(array=X_val, _min=min_arr, _max=max_arr)
                X_te, _, _ = data_utils.min_max_normalize(array=X_te, _min=min_arr, _max=max_arr)

            # Save data
            output_dic["Fold_1"] = {
                "train": (X_train, y_train, pat_ids_train),
                "val": (X_val, y_val, pat_ids_val),
                "test": (X_te, y_te, pat_ids_te)
            }

        else:
            raise ValueError("Number of folds must be greater than 0. Parameter Passed {self.num_folds}}")
            
        return output_dic
    
    def get_train_X_y(self, fold: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Outputs (X, y) training data for the corresponding fold.
        """
        if fold == 0:
            return self.train_val_test_data["global_split"]["train"][:2]
        else:
            return self.train_val_test_data[f"Fold_{fold}"]["train"][:2]

    def get_test_X_y(self, mode: str = "val", fold:  Union[int, str] = 1) -> tuple[np.ndarray, np.ndarray]:
        """
        Outputs (X, y) data, normalized (if specified) according to training normalization.
        """
        assert mode in ["val", "test"], "Mode must be either 'val' or 'test'"
        
        if fold == 0:
            return self.train_val_test_data["global_split"][mode][:2]
        else:
            return self.train_val_test_data[f"Fold_{fold}"][mode][:2]
    
    def _get_data_characteristics(self):
        """
        Retrieve basic data information property from data loader.
        """

        # Unpack
        X, y = self.data_dic["X_arr"], self.data_dic["y_arr"]
        num_samps, num_timestps, num_feats = X.shape
        num_classes = y.shape[-1]

        return {
            "num_samples": num_samps,
            "num_timestamps": num_timestps,
            "num_features": num_feats,
            "num_outcomes": num_classes,
            "features": self.data_config_all["features"],
            "outcomes": self.data_config_all["outcomes"]
        }
    
    def _get_training_data(self):
        """
        Retrieve training data property from data loader.
        """
        return self.train_val_test_data
    
    def _get_num_folds(self):
        return int(self.num_folds)
    