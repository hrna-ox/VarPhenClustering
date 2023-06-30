#!/usr/bin/env python3
"""
Utility Functions for Loading Data and Saving/Evaluating Results.
"""

# Import Functions
import os
from typing import Union, List, Tuple
import pickle

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split, StratifiedKFold


# ---------------------------------------------------------------------------------------
"Global variables for specific dataset information loading."

# HAVEN Data
HAVEN_FEATURES = {
    "VITALS": ["HR", "RR", "SBP", "DBP", "SPO2", "FIO2", "TEMP", "AVPU"],
    "STATIC": ["age", "gender", "is_elec", "is_surg"],
    "SERUM": ["HGB", "WBC", "EOS", "BAS", "NEU", "LYM"],
    "BIOCHEM": ["ALB", "CR", "CRP", "POT", "SOD", "UR"],
    "OUTCOMES": ["Healthy", "Death", "ICU", "Card"],
    "TIME_WINDOW": 4
}

# MIMIC Data
MIMIC_FEATURES = {
    "VITALS": ["TEMP", "HR", "RR", "SPO2", "SBP", "DBP"],
    "STATIC": ["age", "gender", "ESI"],
    "OUTCOMES": ["Death", "ICU", "Ward", "Discharge"],
    "TIME_WINDOW": 1
}

# ----------------------------------------------------------------------------------------
"Useful functions to define"


def _get_features(feat_set: Union[str, List[str]], data_name: str) -> List[str]:
    """
    Obtain the list of features to subset.

    Params:
    - feat_set: str, set of features to subset or list (the latter is returned as is)
    - data_name: str, name of the dataset to subset features from
    """

    # If feat_set is list then return as is
    if isinstance(feat_set, list):
        return feat_set

    if data_name == "HAVEN":
        features = HAVEN_FEATURES

    elif data_name == "MIMIC":
        features = MIMIC_FEATURES

    else:
        raise ValueError(
            f"Data Name does not match available datasets. Input provided {data_name}"
        )

    # Add a short-cut in case feat_set is all
    if feat_set == "all":
        feat_set = "vit-sta-ser-bio"

    # Add features based on substrings of feat_set
    feat_subset = []
    if "vit" in feat_set.lower():
        feat_subset.extend(features["VITALS"])

    if "sta" in feat_set.lower():
        feat_subset.extend(features["STATIC"])

    if "ser" in feat_set.lower():
        feat_subset.extend(features["SERUM"])

    if "bio" in feat_set.lower():
        feat_subset.extend(features["BIOCHEM"])

    if "outc" in feat_set.lower():
        feat_subset.extend(features["OUTCOMES"])

    # Ensure uniqueness
    feat_subset = list(set(feat_subset))
    print(
        f"\n{data_name} data has been sub-setted to the following features: \n {feat_subset}."
    )

    return feat_subset


def _numpy_forward_fill(array):
    """
    Forward Fill a numpy array. Time index is axis = 1.
    """
    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Add time indices where not masked, and propagate forward
    inter_array = np.where(
        ~array_mask, np.arange(array_mask.shape[1]).reshape(1, -1, 1), 0
    )
    np.maximum.accumulate(
        inter_array, axis=1, out=inter_array
    )  # For each (n, t, d) missing value, get the previously accessible mask value

    # Index matching for output. For n, d sample as previously, use inter_array for previous time id
    array_out = array_out[
        np.arange(array_out.shape[0])[:, None, None],
        inter_array,
        np.arange(array_out.shape[-1])[None, None, :],
    ]

    return array_out


def _numpy_backward_fill(array):
    """Backward Fill a numpy array. Time index is axis = 1"""
    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Add time indices where not masked, and propagate backward
    inter_array = np.where(
        ~array_mask,
        np.arange(array_mask.shape[1]).reshape(1, -1, 1),
        array_mask.shape[1] - 1,
    )
    inter_array = np.minimum.accumulate(inter_array[:, ::-1], axis=1)[:, ::-1]
    array_out = array_out[
        np.arange(array_out.shape[0])[:, None, None],
        inter_array,
        np.arange(array_out.shape[-1])[None, None, :],
    ]

    return array_out


def _median_fill(array):
    """Median fill a numpy array. Time index is axis = 1"""
    array_mask = np.isnan(array)
    array_out = np.copy(array)

    # Compute median and impute
    array_med = np.nanmedian(
        np.nanmedian(array, axis=0, keepdims=True), 
        axis=1, 
        keepdims=True
    )
    array_out = np.where(array_mask, array_med, array_out)

    return array_out


def impute(X):
    """
    Imputation of 3D array accordingly with time as dimension 1:
    1st - forward value propagation,
    2nd - backwards value propagation,
    3rd - median value imputation.

    Mask returned at the end, corresponding to original missing values.
    """
    impute_step1 = _numpy_forward_fill(X)
    impute_step2 = _numpy_backward_fill(impute_step1)
    impute_step3 = _median_fill(impute_step2)

    # Compute mask
    mask = np.isnan(X)

    return impute_step3, mask


def _check_input_format(X, y):
    """Check conditions to confirm model input."""

    try:
        # Length and shape conditions
        cond1 = X.shape[0] == y.shape[0]
        cond2 = len(X.shape) == 3
        cond3 = len(y.shape) == 2

    except Exception as e:
        print(e)
        raise AssertionError("One of the check conditions has failed.")


# ---------------------------------------------------------------------------------------
"Main Class for Data Processing."

class CSVLoader:
    """
    Loader Class for loading data from CSV files.

    Loads the data from csv file, including correct column parsing and formatting.
    """

    def __init__(
        self,
        data_name: str,
        **kwargs,
    ):
        """
        Class Object Initializer.

        Params:
            - data_name: ('HAVEN', 'MIMIC', ...)
            - **kwargs: additional arguments to be passed to the class (added only for compatibility)
        """

        # Save parameters
        self.data_name = data_name.upper()
        self.id_col = "stay_id"
        self.time_col = "sampled_time_to_end"

    def load_from_csv(self):
        """
        Load data from CSV and get info related to specific data loaded.
        """

        # Make data folder and check existence
        data_fd = f"data/{self.data_name}/processed/"

        # Check correct data loading
        try:
            os.path.exists(data_fd)

        except AssertionError as e:
            print(e)
            raise AssertionError(
                f"Data folder does not exist. Please check data folder name. {data_fd} was provided."
            )
            

        # Load data
        if self.data_name == "HAVEN":
            # Get the right data
            X = pd.read_csv(
                data_fd + "X_process.csv", infer_datetime_format=True, header=0
            )
            y = pd.read_csv(data_fd + "copd_outcomes.csv", index_col=0)

        elif self.data_name == "MIMIC":
            # Load Data
            X = pd.read_csv(
                data_fd + "vitals_process.csv", infer_datetime_format=True, header=0
            )
            y = pd.read_csv(data_fd + f"outcomes_process.csv")

        else:
            raise ValueError(
                f"Data Name does not match available datasets. Input Folder provided {data_fd}"
            )

        # Convert target column to TimeDelta
        X["sampled_time_to_end"] = pd.to_timedelta(X.loc[:, "sampled_time_to_end"])

        # Convert to hours
        X["hours_to_end"] = X["sampled_time_to_end"].dt.total_seconds() / 3600
        self.time_col = "hours_to_end"

        return X, y

class DataTransformer(CSVLoader):
    """
    Transformer Class for transforming data loaded from CSV.
    """

    def __init__(
        self, 
        ts_endpoints: Tuple[float, float], 
        feat_set: Union[str, List], 
        *args, **kwargs
    ):
        """
        Initialise object parameters and attributes.

        Params:
            - ts_endpoints: Tuple of floats, indicating the endpoints of the time series to consider.
            - feat_set: str or list, the set of features to consider.
            - *args, **kwargs: Additional arguments to pass to parent class.
        """
        self.ts_min, self.ts_max = ts_endpoints
        self.features = _get_features(feat_set, self.data_name)
        self.outcomes = _get_features("outcomes", self.data_name)

        # Call parent class
        super(CSVLoader, self).__init__(*args, **kwargs)

    def transform(self, data: Tuple[pd.DataFrame, pd.DataFrame]):
        """
        Transform data according to standard procedure:
            - Truncate time points within allowed window,
            - subset to relevant features.
            - Conversion to 3D array.
            - Normalization.
            - Imputation.

        If data has been previously computed, then we simply load the data as is.
        Params:
            data (Tuple[DF_LIKE, DF_LIKE]): Tuple (X, y) of DF-like data for input variables (X), and outcome variables (y).

        Returns:
            _type_: Transformed data (X, y)
        """

        # Load if previously computed
        data_fd = f"data/{self.data_name}/processed/ts_{self.ts_min}_{self.ts_max}_feats_{''.join(self.features)}/"

        if os.path.exists(data_fd):

            # Load data from pickle file
            with open(data_fd + "data_transformed.pkl", "rb") as f:
                data_dic = pickle.load(f)
        
        # Else compute data
        else:
                
            # Expand data
            x, y = data

            # Apply Processing steps
            x_inter = (
                x.query("hours_to_end >= @self.ts_min")
                .query("hours_to_end < @self.ts_max")
                .loc[:, [self.id_col, self.time_col] + self.features]
            )

            # Processing checks
            self._check_correct_time_conversion(x_inter)
            self._data_matching_check(x_inter, y)

            # --------------- Rest of the steps ----------------- #

            # Convert to 3D array
            x_inter, pat_time_ids = self.convert_to_3darray(x_inter)

            # Impute missing values and get mask
            x_out, mask = impute(x_inter)

            # Do things to y
            outcomes = _get_features(feat_set="outcomes", data_name=self.data_name)
            y_data = y[outcomes]
            y_out = y_data.to_numpy().astype("float32")

            # Check data loaded correctly
            _check_input_format(x_out, y_out)

            # Save data so we do not need to reprocess again for future runs
            if not os.path.exists(data_fd):
                os.makedirs(data_fd)
    
            data_dic = {
                "data_arr": (x_out, y_out),
                "data_og": (x, y),
                "mask": mask,
                "ids": pat_time_ids,
            }

            with open(data_fd + "data_transformed.pkl", "wb") as f:
                pickle.dump(data_dic, f)

        return data_dic

    def load_transform(self):
        """
        Combine Load and transformation methods into a single method.
        """
        data = self.load_from_csv()
        data_dic = self.transform(data)
        return data_dic

    def _check_correct_time_conversion(self, X):
        """Check addition and truncation of time index worked accordingly."""

        # Patients arranged by id col in increasing fashion
        cond1 = X[self.id_col].is_monotonic_increasing
        assert cond1

        # Each patient has observation in decreasing order based on sampled time to end.
        cond2 = (
            X.groupby(self.id_col)
            .apply(lambda x: x.time_col.is_monotonic_decreasing)
            .all()
        )
        assert cond2

        # Endpoint of each patient admission is within the time range
        cond3 = (
            X["hours_to_end"].between(self.ts_min, self.ts_max, inclusive="left").all()
        )
        assert cond3

    def _data_matching_check(self, X, y):
        """Check the patient order is correct before converting to 3D array."""

        # Get ids from both X and y
        x_ids, y_ids = X[self.id_col].unique(), y[self.id_col]

        # Ensure equality
        assert np.all(x_ids == y_ids)

    def convert_to_3darray(self, X):
        """
        Convert a pandas dataframe to 3D numpy array of shape (num_samples, num_timestamps, num_variables).
        
        Params:
        - X: pd.DataFrame with observations indexed by id_col, and time_col, attributes of object.
        
        Outputs:
        - Tuple of arrays:
            - x_array
            - id_array
        """

        # Obtain relevant shape sizes
        max_time_length = X.groupby(self.id_col).count()["hours_to_end"].max()
        list_ids = X[self.id_col].unique()
        num_ids = np.size(list_ids)

        # Other basic definitions
        non_id_feats = [col for col in X.columns if col not in [self.id_col, self.time_col]]


        # INITIALIZE OUTPUT ARRAYS
        out_array = np.empty(shape=(num_ids, max_time_length, len(non_id_feats)))
        out_array[:] = np.nan

        # Make a parallel array indicating id and corresponding time
        id_times_array = np.empty(shape=(num_ids, max_time_length, 2))

        # Set ids in this newly generated array
        id_times_array[:, :, 0] = np.repeat(
            np.expand_dims(list_ids, axis=-1), repeats=max_time_length, axis=-1
        )

        # Iterate through ids
        for id_ in tqdm(list_ids):

            # Subset data to where matches respective id
            index_ = np.where(list_ids == id_)[0]
            x_id = X[X[self.id_col] == id_]

            # Compute negative differences instead of keeping the original times.
            x_id_copy = x_id.copy()
            x_id_copy["time_to_end"] = -x_id["time_to_end"].diff().values

            # Update target output array and time information array
            out_array[index_, : x_id_copy.shape[0], :] = x_id_copy[non_id_feats].values
            id_times_array[index_, : x_id_copy.shape[0], 1] = x_id["time_to_end"].values

        return out_array.astype("float32"), id_times_array.astype("float32")

class DataLoader(DataTransformer):
    """
    Class to prepare data being passed to different inputs. Builds on Data Transformer class by adding methods relevant to:
    a) Train test Split
    b) Data batching
    c) Pytorch Data Loader features (relevant to Deep Learning)_
    """
    def __init__(self, *args, **kwargs):

        # Call parent class
        super(DataLoader, self).__init__(*args, **kwargs)

        # Initialise some empty attributes
        self.train_test_ratio = None
        self.train_val_ratio = None
        self.seed = None
        self._dataloader_fmt = None

        # Other parameters to keep track during transformation
        self.nan_min, self.nan_max = 0, 0

    def prepare_input(
        self,
        seed: int,
        train_val_ratio: float = 0.6,
        train_ratio: float = 0.7,
        shuffle: bool = True,
        K_folds: int = 1,
        **kwargs
    ):
        """
        Main method for Class. Prepare exact data input format as required by the separate models.

        Params:
        - seed: int, random seed for reproducibility
        - train_val_ratio: float, proportion of the whole data that is used as train or val data
        - train_ratio: float, proportion of the training + validation data that is actually fed to the model for training.
        - shuffle: bool, whether to shuffle the data
        - K_folds: int, number of folds to use for (Stratified) cross validation. If 1, then no cross validation is used.
        - kwargs: dict, additional arguments to be passed for compatibility.
        """

        # Load data dictionary with extracted data and unpack
        data_dic = self.load_transform()
        x_og, y_og = data_dic["data_og"]
        x_arr, y_arr = data_dic["data_arr"]
        mask, ids = data_dic["mask"], data_dic["ids"]

        # Prepare data dictionary for output saving
        load_config = {
            "data_name": self.data_name,
            "id_col": self.id_col,
            "time_col": self.time_col,
            "features": self.features,
            "outcomes": self.outcomes,
            "ts_endpoints": (self.ts_min, self.ts_max),
            "train_val_ratio": train_val_ratio,
            "train_ratio": train_ratio,
            "seed": seed,
            "shuffle": shuffle,
        }
        output_dic = {
            "load_config": load_config, 
            "data_og": (x_og, y_og)
        }

        # Print some base information
        print(f"""
            Data {self.data_name} successfully loaded for features {self.features} and outcomes {self.outcomes}.
            (X, y) shape: {x_arr.shape}, {y_arr.shape}
            Outcome Distribution: {y_og.sum(axis=0).astype(int)}
            """
        )  
        
        # Separate into train-test data
        (
            X_train_val, X_test,
            y_train_val, y_test,
            ids_train_val, ids_test,
            mask_train_val, mask_test) = train_test_split(
                x_arr, y_arr, ids, mask,
                train_size=train_val_ratio,
                shuffle=shuffle,
                random_state=seed,
                stratify=np.argmax(y_arr, axis=-1)        # ensure that data split is stratified according to outcome data.
        )

        # Consider 2 scenarios: no K-fold validation, or K-fold validation
        if K_folds == 1:
                
            # Separate into train-val data
            (
                X_train, X_val,
                y_train, y_val,
                ids_train, ids_val,
                mask_train, mask_val) = train_test_split(
                    X_train_val, y_train_val, ids_train_val, mask_train_val,
                    train_size=train_ratio,
                    shuffle=shuffle,
                    random_state=seed,
                    stratify=np.argmax(y_train_val, axis=-1)        # ensure that data split is stratified according to outcome data.
            )

            # Update input vectors according to normalization
            X_train_norm = self.normalize(X_train)
            X_val_norm = self.apply_normalization(X_val)
            X_test_norm = self.apply_normalization(X_test)

            # Save dictionaries for output
            output_dic["fold_1"] = {
                    "X": (X_train_norm, X_val_norm, X_test_norm),
                    "y": (y_train, y_val, y_test),
                    "ids": (ids_train, ids_val, ids_test),
                    "mask": (mask_train, mask_val, mask_test),
                    "nan_bounds": (self.nan_min, self.nan_max)
                }
        
        else:    # Run Stratified Cross Validation, note test data is fixed

            # Initialise K-fold cross validation and iterate
            skf = StratifiedKFold(n_splits=K_folds, shuffle=shuffle, random_state=seed)
            stratified_y = np.argmax(y_train_val, axis=-1)

            # Iterate over folds
            for fold_id, (train_index, val_index) in enumerate(skf.split(X_train_val, stratified_y)):

                # Apply indices to get train, val, data
                X_train, X_val = X_train_val[train_index, :, :], X_train_val[val_index, :, :]
                y_train, y_val = y_train_val[train_index, :], y_train_val[val_index, :]
                ids_train, ids_val = ids_train_val[train_index, :, :], ids_train_val[val_index, :, :]
                mask_train, mask_val = mask_train_val[train_index, :, :], mask_train_val[val_index, :, :]

                # Update input vectors according to normalization
                X_train_norm = self.normalize(X_train)
                X_val_norm = self.apply_normalization(X_val)
                X_test_norm = self.apply_normalization(X_test)

                # Save dictionaries as output
                output_dic[f"fold_{fold_id+1}"] = {
                    "X": (X_train_norm, X_val_norm, X_test_norm),
                    "y": (y_train, y_val, y_test),
                    "ids": (ids_train, ids_val, ids_test),
                    "mask": (mask_train, mask_val, mask_test),
                    "nan_bounds": (self.nan_min, self.nan_max)
                }

        return output_dic

    def normalize(self, X: np.ndarray):
        """
        Compute min and max values of input array, and normalize data according to min-max standardization.
        """

        # Get nan min and nan max (as they take into account missing values)
        self.nan_min = np.nanmin(X, axis=0, keepdims=True)
        self.nan_max = np.nanmax(X, axis=0, keepdims=True)

        # Once min and max are computed, apply normalization
        return self.apply_normalization(X)

        return X_norm
    
    def apply_normalization(self, X: np.ndarray):
        """
        Apply normalization factors to input array. This normalizes data using the current values of min and max.
        """

        # Use np where to ensure that division by zero is avoided (where nan min == nan max, return np.nan, else 
        # return normalized value)
        X_norm = np.where(
            self.nan_max != self.nan_min,
            (X - self.nan_min) / (self.nan_max - self.nan_min),
            np.nan
        )

        return X_norm
    