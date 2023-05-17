""""
Processing script for extracting vital sign information.

Author: Henrique Aguiar
Last Updated: 15 May 2023

Requirements:
- Run -admissions_processing.py file first.

This files processes MIMIC-IV-ED vital data. The following steps are performed:

    - Subset patients based on the cohort previously processed under admissions_processing.py
    - Subset vital sign measurements taken between intime and outtime of ED admissions.
    - Remove patients based on availability of temporal information.
    - Resample measurements to hourly basis (averaged).
    - Remove patients based on availability of temporal information on the newly sampled data.
"""

# Import Libraries
import json
import os

import pandas as pd

# Important for pandas bar progress
from tqdm import tqdm
tqdm.pandas()

# Test functions to check processing
import src.data_processing.MIMIC.test_functions as tests


# ------------------------------------ Configuration Params --------------------------------------
"""
List of default data variables and cut-off threshold parameters defined in MIMIC_PROCESSING_DEFAULT_VARS.json
file.

DATA_FD: where the raw data is saved.
SAVE_FD: folder path of interim data saving.

ID_COLUMNS: identifiers for admissions, patients and hospital stays.
TIME_COLUMNS: list of datetime object columns.

VITALS_NAMING_DIC: dictionary for renaming columns of dataframe

MIN_NUM_OBVS: minimum number of observations per admission in order to keep it
NA_PROP_THRESH: maximum percentage of missing observations deemed "acceptable"
RESAMPLING_RULE: rule for resampling data into blocks.
LAST_OBVS_TIME_TO_EXIT: The maximum allowed time between the last observation and the outtime of the admission.
"""

with open("src/data_processing/MIMIC/MIMIC_PROCESSING_DEFAULT_VARS.json", "r") as f:
    DEFAULT_CONFIG = json.load(f)
    f.close()

if not os.path.exists(DEFAULT_CONFIG["SAVE_FD"]):
    os.makedirs(DEFAULT_CONFIG["SAVE_FD"])


# ============= AUXILIARY FUNCTIONS ===============

def _resample(x, time_feats, resampling_rule):
    "Resample the dataset given time to end column based on the resampling rule."

    # Check dataframe is sorted
    assert x["time_to_end"].is_monotonic_decreasing

    # Create intermediate output to resample
    _x = x[["time_to_end"] + time_feats]
    resampled_x = _x.resample(on="time_to_end", rule=resampling_rule, closed="left", label="left").mean()

    # Add resampling array
    resampled_x.index.name = "sampled_time_to_end"
    resampled_x.reset_index(drop=False, inplace=True)

    # Now add all other vars
    static_vars = [col for col in x.columns if col not in time_feats + ["time_to_end"]]
    pat_static_info = x[static_vars].iloc[0, :]

    # Add to resampled data
    resampled_x[static_vars] = pat_static_info

    return resampled_x


def main():

    """
    Check admissions cohort has been processed previously. If not, run file.
    """

    # Print Information
    print("/n/n ======= PROCESSING VITALS ======= /n/n")


    try:  # Check admissions intermediate processed exists
        assert os.path.exists(DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv")

    except AssertionError:
        print("/n Admissions need to be processed. We do so now./n")

        # Run admissions_processing.py file otherwise
        os.system("python -m src.data_processing.MIMIC.admissions_processing.py")

    """
    Load observation tables (vitals) and previously processed admissions

    admissions: dataframe indicating the admissions that have been processed.
    vital_signs_ed: dataframe with observation vital sign data in the ED.
    """
    adm_inter = pd.read_csv(
                    DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv",
                    index_col=0,
                    header=0,
                    parse_dates=DEFAULT_CONFIG["VITALS_TIME_VARS"])

    vital_signs_ed = pd.read_csv(
                    DEFAULT_CONFIG["DATA_FD"] + "ed/vitalsign.csv",
                    index_col=None,
                    header=0,
                    low_memory=False,
                    parse_dates=["charttime"])

    # Re-check admissions intermediate, vitals were correctly processed/loaded
    tests.admissions_processed_correctly(adm_inter)
    tests.test_is_complete_ids(vital_signs_ed, "subject_id", "stay_id")    # No NaN on stay id column.

    # ------------------------------------- // -------------------------------------
    """
    Step 1: Subset vital signs data based on the subsetted cohort given by adm_inter. We do so via merging.
    Then rename columns.
    Then keep only observations taken between registered ED intime and ED outtime.
    """

    # Merge data on stay id (there are no NaNs)
    vitals_S1 = (vital_signs_ed
                .merge(right=adm_inter, how="inner", on=["subject_id", "stay_id"]) # merge on stay_id (keep those that are in adm_inter)
                .rename(DEFAULT_CONFIG["VITALS_RENAMING_DIC"], axis=1) # Rename columns
                .query("charttime >= intime")       # Observations after intime
                .query("charttime <= outtime")      # Observations before outtime
    )

    # Test processing and save
    tests.charttime_between_intime_outtime(vitals_S1)
    tests.ids_subset_of_cohort(vitals_S1, adm_inter)
    vitals_S1.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S1.csv", index=True, header=True)
    print("=", end="/r")

    """
    Step 2: Remove admissions with too much missing data. This includes:
    a) less than MIN_NUM_OBSERVS observations
    b) at least NA_PROP_THRESH * # observations NAs on any vital sign
    """

    vitals_S2 = (vitals_S1
          .groupby("stay_id", as_index=True)
          .filter(lambda x: x.shape[0] >= DEFAULT_CONFIG["MIN_NUM_OBSERVS"] and
                    x[DEFAULT_CONFIG["VITALS_RENAMING_DIC"].values()].isna().sum().le(
                        x.shape[0] * DEFAULT_CONFIG["NA_PROP_THRESH"]
                        ).all()
                    )
        .reset_index(drop=True)
    )

    # Test and save
    tests.stays_have_sufficient_data(vitals_S2, DEFAULT_CONFIG)
    vitals_S2.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S2.csv", index=True, header=True)
    print("==", end="/r")

    """
    Step 3: Compute time to end for each charttime measurement.
    Resample based on time to end.
    """

    vitals_S3 = (vitals_S2
            .assign(time_to_end=lambda x: x["outtime"] - x["charttime"])
            .sort_values(by=["stay_id", "time_to_end"], ascending=[True, False])
            .groupby("stay_id", as_index=False)
            .progress_apply(lambda x: _resample(x,
                                    time_feats = list(DEFAULT_CONFIG["VITALS_RENAMING_DIC"].values()),
                                    resampling_rule = DEFAULT_CONFIG["RESAMPLING_RULE"])
                            )
            .reset_index(drop=True)
            )

    # Test and save
    #
    vitals_S3.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S3.csv", index=True, header=True)
    print("===", end="/r")

    """
    Step 4: Apply step 2 with the blocked data
    """

    vitals_S4 = (vitals_S3
        .groupby("stay_id", as_index=True)
        .filter(lambda x: x.shape[0] >= DEFAULT_CONFIG["MIN_NUM_OBSERVS"] and
                x[DEFAULT_CONFIG["VITALS_RENAMING_DIC"].values()].isna().sum().le(
                    x.shape[0] * DEFAULT_CONFIG["NA_PROP_THRESH"]
                    ).all()
                )
    .reset_index(drop=True)
    )

    # Test and save
    vitals_S4.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S4.csv", index=True, header=True)
    print("====", end="/r")

    """
    Step 5 - ensure the closest observation to admission end is not too far away
    """

    # Filter only admissions with time_to_end min below threshold
    vitals_S5 = (vitals_S4
            .groupby("stay_id", as_index=True)
            .filter(lambda x: x["sampled_time_to_end"].dt.total_seconds().min() <=   # Check <= for minimum sampled time
                    DEFAULT_CONFIG["LAST_OBVS_TIME_TO_EXIT"] * 3600   # Convert to horus
            ))


    # Test and save
    vitals_S5.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv", index=True, header=True)
    print("=====", end="/r")

if __name__ == "__main__":
    main()
