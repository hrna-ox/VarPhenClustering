""""
Processing script for computing admission outcome.

Author: Henrique Aguiar
Last Updated: 15 May 2023

Requirements:
- Run -admissions_processing.py file first.
- Run -vitals_processing.py file first (Already checks for admissions_processing).

This files processes MIMIC-IV-ED outcome data for each admission. The following steps are performed:

	- Subset patients based on the cohort previously processed for admissions and vitals.
	- Identify windows of time after the admission to target the outcomes.
	- Compute targets based on a) death, b) ICU, c) Discharge, or d) any medical Ward.
"""

# Import Libraries
import datetime as dt
import json
import os

import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import src.data_processing.MIMIC.data_utils as utils
import src.data_processing.MIMIC.test_functions as tests

# LOAD CONFIGURATION 
with open("src/data_processing/MIMIC/MIMIC_PROCESSING_DEFAULT_VARS.json", "r") as f:
    DEFAULT_CONFIG = json.load(f)
    f.close()

if not os.path.exists(DEFAULT_CONFIG["SAVE_FD"]):
    os.makedirs(DEFAULT_CONFIG["SAVE_FD"])


def main():

    # ------------------------ Checking Data Loaded -------------------------------
    try:
        assert os.path.exists(DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv")
        assert os.path.exists(DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv")
    except AssertionError:
        raise ValueError(f"Running admissions_processing.py and vitals_processing.py prior to running '{__file__}'")
        os.system("python -m src.data_processing.MIMIC.vitals_processing")

    # ------------------------ Configuration params --------------------------------
    """
    Global and Argument variables for Vital sign processing.
    
    TIME_VARS: datetime object variables for patient admission.
    VITALS_TIME_VARS: datetime object variables for observation data.
    """

    # TIME_VARS = ["intime", "outtime", "next_intime", "next_outtime", "dod"]
    # VITALS_TIME_VARS = ["intime", "outtime", "time_to_end_min", "time_to_end_max",
    #                     "time_to_end", f"sampled_time_to_end({resampling_rule})"]


    """
    Data Loading
    """
    vitals = pd.read_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv", 
                         index_col=0, header=0, 
                         parse_dates=DEFAULT_CONFIG["VITALS_TIME_VARS"]
                         )
    transfers_core = pd.read_csv(DEFAULT_CONFIG["DATA_FD"] + "core/transfers.csv", 
                            index_col=None, header=0, parse_dates=["intime", "outtime"])
    vitals["sampled_time_to_end"] = pd.to_timedelta(vitals.loc[:, "sampled_time_to_end"])


    # Check correct computation of admissions
    # tests.vitals_processed_correctly(vitals)

    """
    Step 1: Subset the set of transfers to the already processed cohort.
    """

    transfers_subset = utils.subsetted_by(transfers_core, admissions_subset, ["subject_id", "hadm_id"])
    vitals["chartmax"] = vitals["outtime"] - vitals["time_to_end"]
    vitals["hadm_id"] = admissions.set_index("stay_id").loc[vitals.stay_id.values, "hadm_id"].values

    """
    Potential Outcomes will be regular admission to acute ward, ICU or death. We consider 4 different window sizes:
    - 4 hours, 12 hours, 24 hours and 48 hours.
    """
    time_window_1 = dt.timedelta(hours=4)
    time_window_2 = dt.timedelta(hours=12)
    time_window_3 = dt.timedelta(hours=18)
    time_window_4 = dt.timedelta(hours=24)
    time_window_5 = dt.timedelta(hours=36)
    time_window_6 = dt.timedelta(hours=48)

    # # Need to include Death
    outcomes_4_hours = transfers_subset.groupby("hadm_id", as_index=True).progress_apply(
        lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_1))
    outcomes_12_hours = transfers_subset.groupby("hadm_id", as_index=True).progress_apply(
        lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_2))
    # outcomes_18_hours = transfers_subset.groupby("hadm_id", as_index=True).progress_apply(
    #     lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_3))
    outcomes_24_hours = transfers_subset.groupby("hadm_id", as_index=True).progress_apply(
        lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_4))
    # outcomes_36_hours = transfers_subset.groupby("hadm_id", as_index=True).progress_apply(
    #     lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_5))
    outcomes_48_hours = transfers_subset.groupby("hadm_id", as_index=True).progress_apply(
        lambda x: utils.select_death_icu_acute(x, admissions_subset, time_window_6))

    # # Ensure all patients have only one class
    assert outcomes_4_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
    assert outcomes_12_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
    # assert outcomes_18_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
    assert outcomes_24_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
    # assert outcomes_36_hours.iloc[:, :-1].sum(axis=1).eq(1).all()
    assert outcomes_48_hours.iloc[:, :-1].sum(axis=1).eq(1).all()

    """
    Final processing, ensure admissions and observations match with patient ids
    """
    # Subset vitals and admission data
    admissions_keep = outcomes_4_hours.index.tolist()
    admissions_final = admissions_subset[admissions_subset.hadm_id.isin(admissions_keep)]
    vitals_final = vitals[vitals.hadm_id.isin(admissions_keep)]

    """
    Add static variables to input data.
    """
    static_vars = ["gender", "age", "ESI"]
    vitals_final[static_vars] = admissions_final.set_index("hadm_id").loc[
        vitals_final.hadm_id.values, static_vars].values
    vitals_final["gender"] = vitals_final.loc[:, "gender"].replace(to_replace=["M", "F"], value=[1, 0])
    vitals_final["charttime"] = vitals_final.loc[:, "outtime"].values - vitals_final.loc[:,
                                                                        "sampled_time_to_end(1H)"].values

    """
    Save Data and Print Basic Information.
    """

    # Number of Patients and number of observations.
    print(f"Number of cohort patient: {vitals_final.stay_id.nunique()}")
    print(f"Number of observations: {vitals_final.shape[0]}")

    print(f"Sample outcome distribution: {outcomes_4_hours.sum(axis=0)}")

    # Save to output variables
    process_fd = DATA_FD + "processed/"

    if not os.path.exists(process_fd):
        os.makedirs(process_fd)

    # Save general
    vitals_final.to_csv(SAVE_FD + "vitals_final.csv", index=True, header=True)
    admissions_final.to_csv(SAVE_FD + "admissions_final.csv", index=True, header=True)

    # Save for input
    vitals_final.to_csv(process_fd + "vitals_process.csv", index=True, header=True)
    admissions_final.to_csv(process_fd + "admissions_process.csv", index=True, header=True)
    outcomes_4_hours.to_csv(process_fd + "outcomes_4h_process.csv", index=True, header=True)
    outcomes_12_hours.to_csv(process_fd + "outcomes_12h_process.csv", index=True, header=True)
    # outcomes_18_hours.to_csv(process_fd + "outcomes_18h_process.csv", index=True, header=True)
    outcomes_24_hours.to_csv(process_fd + "outcomes_24h_process.csv", index=True, header=True)
    # outcomes_36_hours.to_csv(process_fd + "outcomes_36h_process.csv", index=True, header=True)
    outcomes_48_hours.to_csv(process_fd + "outcomes_48h_process.csv", index=True, header=True)

if __name__ == "__main__":
    main()
