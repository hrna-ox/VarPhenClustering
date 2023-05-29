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
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import src.data_processing.MIMIC.test_functions as tests

# LOAD CONFIGURATION 
with open("src/data_processing/MIMIC/MIMIC_PROCESSING_DEFAULT_VARS.json", "r") as f:
    DEFAULT_CONFIG = json.load(f)
    f.close()

if not os.path.exists(DEFAULT_CONFIG["SAVE_FD"]):
    os.makedirs(DEFAULT_CONFIG["SAVE_FD"])


# ========== AUXILIARY FUNCTIONS ==========

def _select_outcome(vitals, transfers, window):
    """
    Determine outcome of admission given dataset with vital and static information, 
    and data for transfers.
    output order is [D, I, W, Disc].
    """

    # Load static information from vitals
    try:
        ed_outtime, dod = vitals[["outtime", "deathtime"]].iloc[0, :]
    except IndexError:
        ed_outtime, dod = vitals[["outtime", "deathtime"]].iloc[:]
    

    # Check deathtime first
    if dod != np.nan:
        
        # Get time to death
        time_to_death = dod - ed_outtime
        if time_to_death <= window:
            return [1, 0, 0, 0]       
        
    # If there is no death, or patient died after time window
    lower_bound, upper_bound = ed_outtime, ed_outtime + window
    transfers_within_window = (transfers
                               .query("intime >= @lower_bound")
                               .query("intime <= @upper_bound")
    )
    
    # Identify ICUs
    has_icus = (
        transfers_within_window.careunit.str.contains("(?i)ICU", na=False) |
        transfers_within_window.careunit.str.contains("(?i)Neuro Stepdown", na=False)
    )

    # If ICU admission
    if has_icus.sum() > 0:
        return [0, 1, 0, 0]
    
    # Check to see transfers contain discharge
    has_discharge = transfers_within_window.eventtype.str.contains("discharge", na=False)
    if has_discharge.sum() > 0:
        return [0, 0, 0, 1]
    
    else:
        return [0, 0, 1, 0]




def main():
    # ------------------------ Checking Data Loaded -------------------------------
    try:
        assert os.path.exists(DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv")
        assert os.path.exists(DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv")

    except AssertionError as e:
        print(f"Running admissions_processing.py and vitals_processing.py prior to running '{__file__}'")
        os.system("python -m src.data_processing.MIMIC.vitals_processing")

    # ------------------------ Configuration params --------------------------------

    """
    Data Loading:
    - Load admission and vital data that was previously processed.
    - transfers_core: information about the route patients take throughout the hospital.
    - admissions_core: information about the hospital admissions for different patients.
    """

    # Print Information
    print("\n\n ======== PROCESSING OUTCOMES ======== \n\n")

    # Load previously processed data
    adm_proc = pd.read_csv(
        DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv",
        index_col=0, 
        header=0, 
        parse_dates=["intime", "outtime", "intime_next", "outtime_next", "deathtime"]
    )
    vit_proc = (
        pd.read_csv(
        DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv", 
        index_col=0, 
        header=0, 
        parse_dates=DEFAULT_CONFIG["VITALS_TIME_VARS"]
        )
        .reset_index(drop=False)
        .assign(sampled_time_to_end=lambda x: pd.to_timedelta(x["sampled_time_to_end"]))  # pd does not load timedelta automatically
    )

    # Check correct computation of admissions and vitals
    tests.test_admissions_processed_correctly(adm_proc)
    tests.test_vitals_processed_correctly(vit_proc, config_dic=DEFAULT_CONFIG)


    # Load core info
    transfers_core = pd.read_csv(
        DEFAULT_CONFIG["DATA_FD"] + "core/transfers.csv", 
        index_col=None, 
        header=0, 
        parse_dates=["intime", "outtime"]
    )
    admissions_core = pd.read_csv(
        DEFAULT_CONFIG["DATA_FD"] + "core/admissions.csv",
        index_col=None,
        header=0,
        parse_dates=["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]
    )


    """
    Step 1: Subset the set of transfers/admissions_core to the already processed cohort.

    We do this by merging. 
    """
    merge_ids = ["subject_id", "hadm_id"]

    # Inner merge for transfers core
    transfers_S1 = (
        transfers_core
        .merge(
            vit_proc[merge_ids + ["stay_id"]].drop_duplicates(),   # Drop duplicates as we don't need all the rows
            how="inner",
            on=merge_ids
        )
        .dropna(subset=["hadm_id"])                 # Drop rows with no hadm_id as we can't compare with transfers
    )

    # Inner merge for admissions core
    admissions_S1 = (
        admissions_core
        .merge(
            vit_proc[merge_ids + ["stay_id"]].drop_duplicates(),
            how="inner",
            on=merge_ids
        )
        .dropna(subset=["hadm_id"])            # Drop rows with no hadm_id as we can't compare with transfers
    )

    # Testing and save
    tests.test_ids_subset_of_cohort(transfers_S1, vit_proc, *merge_ids)
    tests.test_ids_subset_of_cohort(admissions_S1, vit_proc, *merge_ids)
    tests.test_is_complete_ids(transfers_S1, *merge_ids, "stay_id")
    tests.test_is_complete_ids(admissions_S1, *merge_ids, "stay_id")

    # Check processing and correctdeness
    transfers_S1.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "transfers_S1.csv", header=True, index=True)

    """
    Step 2: Identify outcome given the transfer information.
    We consider 3 time windows (4, 12 and 24 hours).
    """

    # Define time windows
    _4_hours = dt.timedelta(hours=4)
    _12_hours = dt.timedelta(hours=12)
    _24_hours = dt.timedelta(hours=24)

    # Define output outcome arrays
    list_of_hadms = vitals["hadm_id"].dropna().unique()
    out_4h = pd.DataFrame(np.nan, list_of_hadms, columns=["De", "I", "W", "Di"])
    out_12h = pd.DataFrame(np.nan, list_of_hadms, columns=["De", "I", "W", "Di"])
    out_24h = pd.DataFrame(np.nan, list_of_hadms, columns=["De", "I", "W", "Di"])

    # Apply outcome identification function for each patient
    for cur_hadm in tqdm(list_of_hadms):

        # Get vitals and transfer info for patient
        cur_vital, cur_tr = vitals.query("hadm_id==@cur_hadm"), transfers_S1.query("hadm_id==@cur_hadm")

        # Determine outcome
        out_4h.loc[cur_hadm, :] = _select_outcome(vitals=cur_vital, transfers=cur_tr, window=_4_hours)
        out_12h.loc[cur_hadm, :] = _select_outcome(vitals=cur_vital, transfers=cur_tr, window=_12_hours)
        out_24h.loc[cur_hadm, :] = _select_outcome(vitals=cur_vital, transfers=cur_tr, window=_24_hours)

    # Quick checks (to be moved to test data)
    assert out_4h.sum(axis=1).eq(1).all()
    assert out_12h.sum(axis=1).eq(1).all()
    assert out_24h.sum(axis=1).eq(1).all()

    """
    Final Step:
    - Subset to those admissions that are non-missing.
    - Create charttime time feature.
    """

    # Subset vitals and admission data
    admissions_final = admissions.query("hadm_id.isin(@list_of_hadms)").set_index("hadm_id")
    vitals_subset = vitals.query("hadm_id.isin(@list_of_hadms)")

    # Create charttime variable
    vitals_final = (vitals_subset
                .assign(charttime=lambda x: x["outtime"] - x["sampled_time_to_end"])
                )

    """
    Print Base information, check data processed correctly and save.
    """

    # Number of Patients and number of observations.
    print(f"Number of cohort patient: {vitals_final.stay_id.nunique()}")
    print(f"Number of observations: {vitals_final.shape[0]}")
    print(f"Sample outcome distribution (12 hours window): {out_12h.sum(axis=0)}")


    # Prepare Save Path
    process_fd = DEFAULT_CONFIG["DATA_FD"] + "processed/"

    if not os.path.exists(process_fd):
        os.makedirs(process_fd)

    # Save general
    vitals_final.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_final.csv", index=True, header=True)
    admissions_final.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_final.csv", index=True, header=True)


    # Save for input
    vitals_final.to_csv(process_fd + "vitals_process.csv", index=True, header=True)
    admissions_final.to_csv(process_fd + "admissions_process.csv", index=True, header=True)
    out_4h.to_csv(process_fd + "outcomes_4h_process.csv", index=True, header=True)
    out_12h.to_csv(process_fd + "outcomes_12h_process.csv", index=True, header=True)
    out_24h.to_csv(process_fd + "outcomes_24h_process.csv", index=True, header=True)


if __name__ == "__main__":
    main()
