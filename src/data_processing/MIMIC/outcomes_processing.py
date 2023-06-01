""""
Processing script for computing admission outcome.

Author: Henrique Aguiar
Last Updated: 15 May 2023

Requirements:
- Run -admissions_processing.py file first.
- Run -vitals_processing.py file first (Already checks for admissions_processing).

This files processes MIMIC-IV-ED outcome data for each admission. The following steps are performed:

	- Subset patients based on the cohort previously processed for admissions and vitals.
    - Extra processing to remove nonsensical admissions, e.g., admitted to hospital before ED, etc...
	- Identify windows of time after the admission to target the outcomes.
	- Compute targets based on a) death, b) ICU, c) Discharge, or d) any medical Ward.
"""

# Import Libraries
import src.data_processing.MIMIC.test_functions as tests
import datetime as dt
import json
import os

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()


# LOAD CONFIGURATION
with open("src/data_processing/MIMIC/MIMIC_PROCESSING_DEFAULT_VARS.json", "r") as f:
    DEFAULT_CONFIG = json.load(f)
    f.close()

if not os.path.exists(DEFAULT_CONFIG["SAVE_FD"]):
    os.makedirs(DEFAULT_CONFIG["SAVE_FD"])


# region Auxiliary Functions - used to define outcome

def get_first_death_time(df):
    """
    Given a list of transfers which includes information about the patient hospital admission and other information, get
    the time of death for the patient. This function exists for standardisation.
    """

    # For each stay id (groupby), access the deathtime and compute the minimum if available
    earliest_deathtime = df.groupby("stay_id").deathtime.nth(0)    # Get the first row of deathtime (all rows have the same value)

    return earliest_deathtime


def get_first_icu_time(df):
    """
    Given a list of transfers which includes information about the patient hospital admission and other information, get
    the time of the first ICU entry for the patient if it exists.
    """

    # For each stay id (groupby), identify the transfers to ICU wards, and compute the entry time if available
    earliest_icu_time = (
        df
        .groupby("stay_id")
        .apply(lambda x: (
            x
            # Careunit has ICU in name
            .query("careunit.str.contains('(?i)ICU', na=False, case=False)")
            # Another ICU name
            .query("careunit.str.contains('(?i)Neuro Stepdown', na=False, case=False)")
            # Get transfer entry time
            .intime
            # Get minimum of all ICU entries
            .min()
        )
        )
    )

    return earliest_icu_time

None
def get_first_discharge_time(df):
    """
    Given a list of transfers which includes information about the patient hospital admission and other information, get
    the time of discharge for the patient if it exists.

    Args:
        df (pd.DataFrame): Dataframe with transfers information.
    """

    # For each stay id (groupby), identify the discharge transfer, and compute the time if the location is not 'DIED'
    earliest_discharge_time = (
        df
        .groupby("stay_id")
        .apply(lambda x: (
            x
            # Remove any transfers for death events
            .query("~ eventtype.str.contains('(?i)DIED', na=False, case=False)")
            # Within remaining transfers, get the discharge transfer
            .query("eventtype == 'discharge'")
            .squeeze()                     # Convert to pd.Series, we know there is exactly one discharge eventtype
            .dischtime                     # Get the discharge time
            
            )
        )
    )

    return earliest_discharge_time


def get_first_ward_time(df):
    """
    Given a list of transfers which includes information about the patient hospital admission and other information, get
    the time of the first transfer to a medical ward for the patient if it exists. This function exists for standardisation.

    Args:
        df (pd.DataFrame): Dataframe with transfers information.
    """
    earliest_ward_time = df.groupby("stay_id").intime_next.nth(0) # Get the first row of intime_next (all rows have the same value)

    return earliest_ward_time

# endregion


def main():
    # region checking previous processing
    try:
        assert os.path.exists(
            DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv")
        assert os.path.exists(
            DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv")

    except AssertionError as e:
        print(
            f"Running admissions_processing.py and vitals_processing.py prior to running '{__file__}'")
        os.system("python -m src.data_processing.MIMIC.vitals_processing")

    # endregion

    # region Data Loading
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
        parse_dates=["intime", "outtime",
                     "intime_next", "outtime_next", "deathtime"]
    )
    vit_proc = (
        pd.read_csv(
            DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv",
            index_col=0,
            header=0,
            parse_dates=DEFAULT_CONFIG["VITALS_TIME_VARS"]
        )
        .reset_index(drop=False)
        # pd does not load timedelta automatically
        .assign(sampled_time_to_end=lambda x: pd.to_timedelta(x["sampled_time_to_end"]))
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
        parse_dates=["admittime", "dischtime",
                     "deathtime", "edregtime", "edouttime"]
    )
    # endregion

    # region Step 1
    """
    Step 1: Subset the set of transfers/admissions_core to the already processed cohort.

    We do this by merging. 
    """

    # Define Id for merging. We separate deathtime as one database registers only date, while the other
    # registers everything (i.e. up to second)
    hadm_merge_ids = [
        col for
        col in vit_proc.columns.tolist() if
        col in admissions_core.columns.tolist() and
        "death" not in col
    ]
    # Useful simplification
    merge_ids = ["subject_id", "hadm_id", "stay_id"]

    # Inner merge for admissions core
    admissions_S1 = (
        admissions_core
        .merge(
            # only want one obvs per admission for merging
            vit_proc.drop_duplicates(subset=merge_ids),
            how="inner",
            on=hadm_merge_ids,
            suffixes=("", "_ed")
        )
        # Drop rows with no hadm_id as we can't compare with transfers
        .dropna(subset=["hadm_id"])
        # Sort by merge_ids, in order
        .sort_values(by=merge_ids, ascending=True)
    )

    # Testing
    tests.test_ids_subset_of_cohort(transfers_S1, vit_proc, *merge_ids)
    tests.test_ids_subset_of_cohort(admissions_S1, vit_proc, *merge_ids)
    tests.test_is_complete_ids(transfers_S1, *merge_ids)
    tests.test_is_complete_ids(admissions_S1, *merge_ids)
    # endregion

    # region Step 2
    """
    Step 2: 
        Remove nonsensical admissions, i.e.:
        a) Admissions were admitted to hospital prior to ED entrance.
        b) Next transfer intime is after admission time to hospital (or is missing).
        c) ED outtime is before ED register outtime.
        d) ED intime is prior to ED register intime.
        e) Discharge from hospital is after next transfer outtime (or is missing), allowing for some delay (6 hours).
    """

    admissions_S2 = (
        admissions_S1
        # admissions to hospital after ED admissions
        .query("intime <= admittime")
        # admissions to hospital before next ED transfer
        .query("intime_next >= admittime | intime_next.isna()")
        # transfer outtime before ed exit time
        .query("outtime <= edouttime")
        # transfer intime before ed registration time
        .query("intime <= edregtime")
        .query("dischtime - outtime_next >= @pd.Timedelta('-6h') | outtime_next.isna()")
        # discharge time not earlier than outtime_next (added -6 hours due to some potential delays)
    )

    # Testing
    tests.test_deathtime_match(admissions_S2)
    tests.test_is_unique_ids(admissions_S2, *merge_ids)
    # endregion

    # region Step 3
    """
    Step 3: 
        For each admission, we compute the earliest time of death, ICU entry, discharge or ward transfer. 
        This is an intermediate df that will then be used to actually compute outcomes.
    """

    # First subset Transfers
    tr_merge_ids = ["subject_id", "hadm_id", "stay_id",
                    "outtime", "deathtime", "intime_next", "outtime_next",
                    "dischtime", "discharge_location"]
    transfers_S1 = (
        transfers_core
        .merge(
            admissions_S2[tr_merge_ids],
            how="inner",
            on=["subject_id", "hadm_id"],
            suffixes=("", "_ed")
        )
        .sort_values(by=["subject_id", "stay_id"], ascending=True)
    )

    # Run tests
    tests.test_is_complete_ids(transfers_S1, "subject_id", "hadm_id")
    tests.test_outtimes_match(transfers_S1)
    tests.test_every_patient_has_discharge_transfer(transfers_S1)
    # endregion

    # Now compute the earliest time given the list of transfers.
    earliest_outcome_times = (
        admissions_S1
        .set_index("stay_id")              # Set index to stay_id to match the below
        .assign(first_death=get_first_death_time(transfers_S1))  # Compute first death time
        .assign(first_icu=get_first_icu_time(transfers_S1)) # Compute first icu time
        .assign(first_ward=get_first_ward_time(transfers_S1)) # Compute first ward time
        .assign(first_discharge=get_first_discharge_time(transfers_S1)) # Compute first discharge time
        .loc[:, ["first_death", "first_icu", "first_ward", "first_discharge", 
                "outtime", "discharge_location", "subject_id"]] 
    )

    # Testing for computed outcomes
    tests.test_events_after_outtime(earliest_outcome_times)

    """
    STEP 5:
        Actually computing the outcomes given a different outcome timedelta.
    """

    # Define time windows
    _4_hours = dt.timedelta(hours=4)
    _12_hours = dt.timedelta(hours=12)
    _24_hours = dt.timedelta(hours=24)

    # Define output outcome arrays
    list_of_hadm = vitals["hadm_id"].dropna().unique()
    out_4h = pd.DataFrame(np.nan, list_of_hadm, columns=["De", "I", "W", "Di"])
    out_12h = pd.DataFrame(np.nan, list_of_hadm, columns=[
                           "De", "I", "W", "Di"])
    out_24h = pd.DataFrame(np.nan, list_of_hadm, columns=[
                           "De", "I", "W", "Di"])

    # Apply outcome identification function for each patient
    for cur_hadm in tqdm(list_of_hadm):

        # Get vitals and transfer info for patient
        cur_vital, cur_tr = vitals.query(
            "hadm_id==@cur_hadm"), transfers_S1.query("hadm_id==@cur_hadm")

        # Determine outcome
        out_4h.loc[cur_hadm, :] = _select_outcome(
            vitals=cur_vital, transfers=cur_tr, window=_4_hours)
        out_12h.loc[cur_hadm, :] = _select_outcome(
            vitals=cur_vital, transfers=cur_tr, window=_12_hours)
        out_24h.loc[cur_hadm, :] = _select_outcome(
            vitals=cur_vital, transfers=cur_tr, window=_24_hours)

    # Quick checks (to be moved to test data)
    assert out_4h.sum(axis=1).eq(1).all()
    assert out_12h.sum(axis=1).eq(1).all()
    assert out_24h.sum(axis=1).eq(1).all()

    ##################
    tr_merge_ids = [
        col for
        col in vit_proc.columns.tolist() if
        col in transfers_core.columns.tolist() and
        "death" not in col
    ]
    # Inner merge for transfers core
    transfers_S1 = (
        transfers_core
        .merge(
            # Drop duplicates as we don't need all the rows
            vit_proc.drop_duplicates(subset=merge_ids),
            how="inner",
            on=tr_merge_ids
        )
        # Drop rows with no hadm_id as we can't compare with transfers
        .dropna(subset=["hadm_id"])
        # Sort by subject_id and stay_id
        .sort_values(by=merge_ids, ascending=True)
    )
    ########################

    """
    Final Step:
    - Subset to those admissions that are non-missing.
    - Create charttime time feature.
    """

    # Subset vitals and admission data
    admissions_final = admissions.query(
        "hadm_id.isin(@list_of_hadms)").set_index("hadm_id")
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
    print(
        f"Sample outcome distribution (12 hours window): {out_12h.sum(axis=0)}")

    # Prepare Save Path
    process_fd = DEFAULT_CONFIG["DATA_FD"] + "processed/"

    if not os.path.exists(process_fd):
        os.makedirs(process_fd)

    # Save general
    vitals_final.to_csv(
        DEFAULT_CONFIG["SAVE_FD"] + "vitals_final.csv", index=True, header=True)
    admissions_final.to_csv(
        DEFAULT_CONFIG["SAVE_FD"] + "admissions_final.csv", index=True, header=True)

    # Save for input
    vitals_final.to_csv(process_fd + "vitals_process.csv",
                        index=True, header=True)
    admissions_final.to_csv(
        process_fd + "admissions_process.csv", index=True, header=True)
    out_4h.to_csv(process_fd + "outcomes_4h_process.csv",
                  index=True, header=True)
    out_12h.to_csv(process_fd + "outcomes_12h_process.csv",
                   index=True, header=True)
    out_24h.to_csv(process_fd + "outcomes_24h_process.csv",
                   index=True, header=True)


if __name__ == "__main__":
    main()
