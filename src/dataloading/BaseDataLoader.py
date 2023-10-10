"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This files defines the base data loader class. This is an object that implements a variety of methods to load, process, transform and save data.
"""

# Import required packages
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import Union, Tuple

import json
import src.dataloading.data_loading_utils as data_utils


class BaseDataLoader:
    """Base Class for Data Loading. Implements general processing and loading methods. Particulars are left to individual data loaders."""
    def __init__(self, data_dir: str, *args, **kwargs):
        """
        Object initialization.

        Args:
            data_dir: Directory where data is stored. Must include:
                - "X.csv": file with time-series data in a tabular format.
                - "y.csv": file with target labels.
                - "data_properties.json": file with information about the data.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None, initializes the object.
        """
        # Initialize attributes
        self.data_dir = data_dir

        # Load configuration
        with open(f"{data_dir}/data_properties.json", "r") as f:
            data_properties = json.load(f)
        self.data_properties = data_properties
        
        # Initialize emptry attributes
        self.data_dic = {}
        self.data_config_all = {}

        # Load data
        X_df = pd.read_csv(f"{self.data_dir}/X.csv", header=0)
        self.y_dataframe = pd.read_csv(f"{self.data_dir}/y.csv", header=0)
        
        # Convert time columns and sort
        X_df[data_properties["time_id_col"]] = pd.to_timedelta(X_df.loc[:, data_properties["time_id_col"]].squeeze()).dt.total_seconds() // 3600
        self.X_dataframe = X_df.sort_values(by=[data_properties["patient_id_col"], data_properties["time_id_col"]], ascending=[True, False])

        # print(self.X_dataframe.shape, self.y_dataframe.shape)
        assert self.X_dataframe.loc[:, data_properties["patient_id_col"]].is_monotonic_increasing
        assert self.X_dataframe.groupby(data_properties["patient_id_col"]).apply(lambda x:
                                                                            x.loc[:, data_properties["time_id_col"]].is_monotonic_decreasing).all()
        # assert np.all(np.unique(self.X_dataframe[data_properties["patient_id_col"]].values) == self.y_dataframe.index.values).all()
    

    def transform(self, time_window: tuple[int, int], feat_subset: Union[list[str], str], *args ,**kwargs):
        """
        Transform the data according to input parameters.  

        Args:
            time_window: Time window to use (input data between lower and upper bound). Must be a tuple of integers.
            feat_subset: Feature subset to use. Can be a list of feature names or a string with the name of a feature set, encoded by self._feat_names_map.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None, transforms the data.
        """

        # =================== CHECK IF TRANSFORMED DATA EXISTS (I.E. HAS BEEN PREVIOUSLY COMPUTED)=================
        features = self._get_features_from_input_param(feat_subset)
        outcomes = self.data_properties["outcome_names"]


        # =================== APPLY TRANSFORMATIONS TO DATA ======================
        X, y = self.X_dataframe, self.y_dataframe
        time_lower_bound, time_upper_bound = time_window
        
        # Subset to features and time window and convert to numpy
        x_npy_3d, pat_ids_npy, time_ids_npy = (X
            .query(f'{self.data_properties["time_id_col"]} >= {time_lower_bound} & {self.data_properties["time_id_col"]} <= {time_upper_bound}')
            .loc[:, [self.data_properties['patient_id_col'], self.data_properties['time_id_col']] + features]
            .pipe(self._convert_to_3d_npy_array)
        )

        # Impute Data
        x_npy_3d = data_utils.impute(x_npy_3d)

        # Now convert outcome data
        y_npy = y[self.data_properties["outcome_names"]].to_numpy().astype(int)
        

        # ==================== UPDATE ATTRIBUTES ===============================
        data_utils.update_dictionary(
            self.data_dic,
            [("X_arr", x_npy_3d), ("y_arr", y_npy), ("pat_ids_arr", pat_ids_npy), ("time_ids_arr", time_ids_npy)]
        )
        
        data_utils.update_dictionary(
            self.data_config_all,
            [("features", features), ("outcomes", outcomes), ("time_window", time_window), ("data_properties", self.data_properties)]
        )

    def get_pat_data_dataframe(self, pat_ids: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load Patient Data in Dataframe format for the patients considered.

        Args:
        - pat_ids: np.ndarray of patient ids to consider.

        Returns:
        - tuple of pd.DataFrame with patient and outcome data.
        """
        X_pats = self.X_dataframe.query("@self.data_properties['patient_id_col'] in @pat_ids")
        y_pats = self.y_dataframe.loc[pat_ids, :]

        return X_pats, y_pats

    def _get_features_from_input_param(self, feat_set: Union[list[str], str]) -> list[str]:
        """
        Get the features given an input feat_set name or list.

        Args:
            feat_set (Union[list[str], str]): Feature set name, encoded in self._feat_names_map, or list of feature names.

        Returns:
            list[str]: List of feature names.
        """
        # If the feature set is a string, then it is a single feature
        if isinstance(feat_set, str):
            return self.data_properties["feature_names_map"][feat_set]
        
        # Otherwise, it is a list of features
        else:
            return feat_set
        
        
    def _convert_to_3d_npy_array(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert a pandas dataframe to 3D numpy array of shape (num_samples, num_timestamps, num_variables).
        
        Params:
        - X: pd.DataFrame with observations indexed by id_col, and time_col, attributes of object.
        
        Outputs:
        - Tuple of arrays:
            - data_array with shape (num_samples, num_timestamps, num_variables) with feature information, and np.nan for missing values.
            - pat_idxs_array with shape (num_samples,) with the corresponding patient ids
            - time_idxs_array with shape (num_timestamps,) with the corresponding time indices
        """
        # Compute useful parameters
        _inter_df = X.groupby(self.data_properties["patient_id_col"], as_index=True).count()[self.data_properties["time_id_col"]]
        max_num_timestamps, list_ids, num_ids = _inter_df.max(), _inter_df.index.values, _inter_df.shape[0]
        non_id_feats = [feature for feature in X.columns if feature not in [self.data_properties["patient_id_col"], self.data_properties["time_id_col"]]]

        # INITIALIZE OUTPUT ARRAYS
        data_array = np.empty(shape=(num_ids, max_num_timestamps, X.shape[1] - 2))     # - 2 removes the id and time columns
        data_array[:] = np.nan
        pat_idxs_array = np.array(list_ids, dtype=int)
        time_idxs_array = np.sort(X[self.data_properties["time_id_col"]].unique())[::-1]     # Sort in descending order based on time to final target event

        # Iterate through all patients
        for idx, pat_id in tqdm(enumerate(list_ids)):
            pat_data = X.query(f'{self.data_properties["patient_id_col"]} == {pat_id}').loc[:, non_id_feats]

            # Append to data array from the end
            data_array[idx, -pat_data.shape[0]:, :] = pat_data.values

        return data_array, pat_idxs_array, time_idxs_array

    def _get_data(self):
        return self.data_dic

    def _get_config(self):
        return self.data_config_all
    