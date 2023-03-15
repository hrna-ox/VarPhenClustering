"""
Processing script for initial ED admission processing.

Author: Henrique Aguiar
Last Updated: 15 March 2023

This files processes MIMIC-IV-ED admissions. The following steps are performed:

    - Computed intime/outtime for each ED admission;
    - Select admissions with ED as first admission;
    - Remove admissions admitted to special wards, such as Partum and Psychiatry;
    - Compute the following transfer location;
    - Add core patient information (i.e. demographic);
    - Remove patients that are too ill or not ill enough (ESI = 1,5).
    - Save processed data.
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

# ------------------------------------ // --------------------------------------
"""
List of variables used for processing. These are pre-defined in the PROCESSING_DEFAULT_CONFIG file.

Data_FD: where the data is saved.
SAVE_FD: folder path of interim data saving.
ID_COLUMNS: identifiers for admissions, patients and hospital stays.
TIME_COLUMNS: list of datetime object columns.
WARDS_TO_REMOVE: list of special wards where patients were transferred to and which represent unique populations. This
list includes Partum and Psychiatry wards, as well as further ED observations, which generally take place when 
the hospital is full.
AGE_LOWER_BOUND: minimum age of patients.
PATIENT_INFO: characteristic information for each patient.
NEXT_TRANSFER_INFO: list of important info to keep related to the subsequent transfer from ED.
"""

with open("src/data_processing/MIMIC/MIMIC_PROCESSING_DEFAULT_CONFIG.json", "r") as f:
    DEFAULT_CONFIG = json.load(f)
    f.close()

if not os.path.exists(DEFAULT_CONFIG["SAVE_FD"]):
    os.makedirs(DEFAULT_CONFIG["SAVE_FD"])


# ============================ AUXILIARY FUNCTIONS ============================


# ============================ MAIN FILE COMPUTATION ============================
def main():
    """
    First, Tables are Loaded. We load 4 tables:
    
    - patients_core: from core/patients filepath. This is a dataframe of patient centralised admission information. 
    We compute and extract base patient information.
    
    - transfer_core: from core/transfers.csv filepath. This is a dataframe with a list of transfers for each patient.
    Includes admissions to ED, but also transfers to wards in the hospital, ICUs, etc...
    
    - admissions_ed: from ed/edstays.csv filepath. This is a dataframe of patient information indicating relevant
    information for any ED admission.
    
    - triage_ed: from ed/triage.csv filepath. This is a dataframe of patient ED admission indicating triage assessments.
    """

    # Hospital Core
    patients_core = pd.read_csv(DEFAULT_CONFIG["DATA_FD"] + "core/patients.csv",
                                index_col=None, header=0, low_memory=False)
    transfers_core = pd.read_csv(DEFAULT_CONFIG["DATA_FD"] + "core/transfers.csv",
                                 index_col=None, header=0, low_memory=False, parse_dates=["intime", "outtime"])

    # ED Admission
    admissions_ed = pd.read_csv(DEFAULT_CONFIG["DATA_FD"] + "ed/edstays.csv",
                                index_col=None, header=0, low_memory=False, parse_dates=["intime", "outtime"])
    triage_ed = pd.read_csv(DEFAULT_CONFIG["DATA_FD"] + "ed/triage.csv", index_col=None, header=0, low_memory=False)

    """
    Step 1: Extract relevant information from admission core data.
        a) For each patient, consider only the first observed ED admission (based on in-time). If there are multiple
        admissions with the same intime, keep the admission with the latest out-time.
    """

    # Assertion checks for duplicates and missing values
    tests.test_unique_stays(admissions_ed)
    tests.test_non_missing_stays_and_times(admissions_ed)

    # Compute the first intime and last outtime per patient
    admissions_ed_S1 = (admissions_ed
                        .groupby("subject_id", as_index=True)
                        .progress_apply(lambda x: x[x.intime == x.intime.min()])  # select first intime
                        # per patient
                        .reset_index(drop=True)
                        .groupby("subject_id", as_index=True)
                        .progress_apply(lambda x: x[x.outtime == x.outtime.max()])  # select last outtime within
                        # the remaining
                        .reset_index(drop=True)
                        )

    # Assertion checks
    tests.test_single_admission_per_patient(admissions_ed_S1)
    tests.test_outtime_after_intime(admissions_ed_S1)

    # Save data
    admissions_ed_S1.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_ed_S1.csv", index=False, header=True)

    """
    Step 2: Subset based on ward data:
        a) Remove admissions where the first ward is not the Emergency Department.
    """

    # Within the list of transfers, subset to list of above patients, and identify the first ward for each patient.
    patients_ed_first_ward = (transfers_core
                              .query("subject_id in @admissions_ed_S1.subject_id.values")   # subset to current pats.
                              .groupby("subject_id", as_index=True)                         # compute per patient
                              .progress_apply(lambda x: x[x.intime == x.intime.min()])  # select first transfer intime
                              .reset_index(drop=True)
                              .query("careunit == 'Emergency Department'")  # remove patients with non-ED first ward
                              .query("eventtype == 'ED'")  # remove patients with non-ED first event type
                              )

    # Merge information
    test = (patients_ed_first_ward
            .merge(admissions_ed_S1, on=["subject_id", "hadm_id", "intime", "outtime"], how="left")
            .dropna(subset=["intime_y"])
            )

    # Check correct processing
    tests.test_is_correctly_merged(test)
    tests.test_ed_is_first_ward(test)

    admissions_ed_S2.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S2.csv", index=True, header=True)

    # Identify First Ward for each patient given the patient id
    transfers_first_ward = utils.endpoint_target_ids(transfers_core, "subject_id", "intime", mode="min")
    ed_first_transfer = transfers_first_ward[(transfers_first_ward["eventtype"] == "ED") &
                                             (transfers_first_ward["careunit"] == "Emergency Department")]

    # Subset to admissions with ED as first transfer
    admissions_ed_S2 = utils.subsetted_by(admissions_ed_S1, ed_first_transfer,
                                          ["subject_id", "hadm_id", "intime", "outtime"])
    transfers_ed_S2 = utils.subsetted_by(transfers_core, admissions_ed_S2, ["subject_id", "hadm_id"])
    admissions_ed_S2.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S2.csv", index=True, header=True)

    """
    Consider only those admissions for which they did not have a subsequent transfer to a Special ward, which includes
    Partum and Psychiatry wards. The full list of wards is identified in WARDS TO REMOVE
    """
    # Remove admissions transferred to irrelevant wards (Partum, Psychiatry). Furthermore, EDObs is also special.
    # Missing check that second intime is after ED outtime
    transfers_second_ward = utils.compute_second_transfer(transfers_ed_S2, "subject_id", "intime",
                                                          transfers_ed_S2.columns)
    transfers_to_relevant_wards = transfers_second_ward[
        ~ transfers_second_ward.careunit.isin(DEFAULT_CONFIG["WARDS_TO_REMOVE"])]
    admissions_ed_S3 = utils.subsetted_by(admissions_ed_S2, transfers_to_relevant_wards, ["subject_id", "hadm_id"])

    # ADD patient core information and next Transfer Information.
    patients_S3 = admissions_ed_S3.subject_id.values
    admissions_ed_S3.loc[:, DEFAULT_CONFIG["PATIENT_INFO"]] = patients_core.set_index("subject_id").loc[
        patients_S3, DEFAULT_CONFIG["PATIENT_INFO"]].values

    for col in DEFAULT_CONFIG["NEXT_TRANSFER_INFO"]:
        admissions_ed_S3.loc[:, "next_" + col] = transfers_to_relevant_wards.set_index("subject_id").loc[
            patients_S3, col].values

    # Compute age and save
    admissions_ed_S3["age"] = admissions_ed_S3.intime.dt.year - admissions_ed_S3["anchor_year"] + admissions_ed_S3[
        "anchor_age"]
    admissions_ed_S3.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S3.csv", index=True, header=True)

    """
    Step 4: Patients must have an age older than AGE LOWER BOUND
    """
    # Compute age and Remove below AGE LOWER BOUND
    admissions_ed_S4 = admissions_ed_S3[admissions_ed_S3["age"] >= DEFAULT_CONFIG["AGE_LOWER_BOUND"]]
    admissions_ed_S4.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S4.csv", index=True, header=True)

    """
    Step 5: Add ESI information, and subset to patients with ESI values and between 2, 3, 4.
    ESI values of 1 and 5 are edge cases (nothing wrong with them, or in extremely critical condition).
    """
    # Compute and remove ESI NAN, ESI 1 and ESI 5 and save
    admissions_ed_S4["ESI"] = triage_ed.set_index("stay_id").loc[admissions_ed_S4.stay_id.values, "acuity"].values
    admissions_ed_S5 = admissions_ed_S4[~ admissions_ed_S4["ESI"].isna()]
    # admissions_ed_S5 = admissions_ed_S5[~ admissions_ed_S5["ESI"].isin([1, 5])]

    # Save data
    admissions_ed_S5.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv", index=True, header=True)
