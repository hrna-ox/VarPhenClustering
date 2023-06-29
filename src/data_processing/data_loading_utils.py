#!/usr/bin/env python3
"""
Utility Functions for Loading Data and Saving/Evaluating Results.
"""

# Import Functions
import os
from typing import Union, List, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm

from src.data_processing.MIMIC.data_utils import convert_to_timedelta

# ---------------------------------------------------------------------------------------
"Global variables for specific dataset information loading."

# HAVEN Data
HAVEN_FEATURES = {
    "VITALS": ["HR", "RR", "SBP", "DBP", "SPO2", "FIO2", "TEMP", "AVPU"],
    "STATIC": ["age", "gender", "is_elec", "is_surg"],
    "SERUM": ["HGB", "WBC", "EOS", "BAS", "NEU", "LYM"],
    "BIOCHEM": ["ALB", "CR", "CRP", "POT", "SOD", "UR"],
    "OUTCOMES": ["Healthy", "Death", "ICU", "Card"],
}

# MIMIC Data
MIMIC_FEATURES = {
    "VITALS": ["TEMP", "HR", "RR", "SPO2", "SBP", "DBP"],
    "STATIC": ["age", "gender", "ESI"],
    "OUTCOMES": ["Death", "ICU", "Ward", "Discharge"],
}

# ----------------------------------------------------------------------------------------
"Useful functions to define"


def _get_features(feat_set, data_name: str):
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
        np.nanmedian(array, axis=0, keepdims=True), axis=1, keepdims=True
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


def _subset_to_balanced(X, y, mask, ids):
    """Subset samples so dataset is more well sampled."""
    class_numbers = np.sum(y, axis=0)
    largest_class, target_num_samples = (
        np.argmax(class_numbers),
        np.sort(class_numbers)[-2],
    )
    print(
        "\nSub-setting class {} from {} to {} samples.".format(
            largest_class, class_numbers[largest_class], target_num_samples
        )
    )

    # Select random
    largest_class_ids = np.arange(y.shape[0])[y[:, largest_class] == 1]
    class_ids_samples = np.random.choice(
        largest_class_ids, size=target_num_samples, replace=False
    )
    ids_to_remove_ = np.setdiff1d(largest_class_ids, class_ids_samples)

    # Remove relevant ids
    X_out = np.delete(X, ids_to_remove_, axis=0)
    y_out = np.delete(y, ids_to_remove_, axis=0)
    mask_out = np.delete(mask, ids_to_remove_, axis=0)
    ids_out = np.delete(ids, ids_to_remove_, axis=0)

    return X_out, y_out, mask_out, ids_out


class CSVLoader:
    """
    Loader Class for loading data from CSV files.

    Loads the data from csv file, including correct column parsing and formatting.
    """

    def __init__(
        self,
        data_name: str,
        feat_set: Union[str, List],
        time_range: Tuple[float, float] = (0, 10000),
    ):
        """
        Class Object Initializer.

        Params:
            - data_name: ('HAVEN', 'MIMIC', ...)
            - feat_set: str or List. Set of features to consider for input data.
            - time_range: tuple of floats, for each admission, subset observations within the two endpoints indicated in time_range.
        """

        # Save parameters
        self.data_name = data_name.upper()
        self.features = feat_set
        self.time_range = time_range
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


def DataTransformer(CSVLoader):
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

        # Other parameters to keep track during transformation
        self.nan_min, self.nan_max = 0, 0

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
        Params:
            data (Tuple[DF_LIKE, DF_LIKE]): Tuple (X, y) of DF-like data for input variables (X), and outcome variables (y).

        Returns:
            _type_: Transformed data (X, y)
        """
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

        # Normalize array
        self.nan_min = np.nanmin(x_inter, axis=0, keepdims=True)
        self.nan_max = np.nanmax(x_inter, axis=0, keepdims=True)
        x_inter = np.divide(x_inter - self.nan_min, self.nan_max - self.nan_min)

        # Impute missing values (including where nan_max - nan_min = 0)
        x_out, mask = impute(x_inter)

        # Do things to y
        outcomes = _get_features(feat_set="outcomes", data_name=self.dataset_name)
        y_data = y[outcomes]
        y_out = y_data.to_numpy().astype("float32")

        # Check data loaded correctly
        _check_input_format(x_out, y_out)

        return {
            "data_arr": (x_out, y_out),
            "data_og": (x, y),
            "mask": mask,
            "ids": pat_time_ids,
        }

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
