"""
Processing script for initial ED admission processing.

Author: Henrique Aguiar
Last Updated: 15 May 2023

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

Data_FD: where the raw data is saved.
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

with open("src/data_processing/MIMIC/MIMIC_PROCESSING_DEFAULT_VARS.json", "r") as f:
    DEFAULT_CONFIG = json.load(f)
    f.close()

if not os.path.exists(DEFAULT_CONFIG["SAVE_FD"]):
    os.makedirs(DEFAULT_CONFIG["SAVE_FD"])


# ============================ AUXILIARY FUNCTIONS ============================

def _compute_earliest_transfer_if_exists(df: pd.DataFrame) -> pd.Series:
    """
    Given dataset df DataFrame of patient admission transfers information, compute the second transfer.
    df includes already ONLY transfers that are not the first transfer. Therefore, we want to select the
    earliest transfer.

    Params:
    - df: pd.DataFrame (or Series) with patient transfer information, including careunit, eventtype, intime
    and outime.

    Outputs:
    - pd.Series with second transfer information, including

    """

    # Get first intime for transfer
    _df = df[df["intime"] == df["intime"].min()]

    # If multiple such occurrences, select the one with the earliest outtime
    if _df.shape[0] > 1:
        _df = _df.query("outtime == @_df.outtime.min()")

    # If there are still multiple such transfers with the registered intime and outtime, select the first one
    if _df.shape[0] > 1:
        _df = _df.iloc[0, :]

    # Return the second transfer information
    return _df


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
    admissions_ed_S1 = (
        admissions_ed
        .groupby("subject_id", as_index=True)
        .progress_apply(lambda x: x[x.intime == x.intime.min()])  # select first intime per patient
        .reset_index(drop=True)
        .groupby("subject_id", as_index=True)
        .progress_apply(lambda x: x[x.outtime == x.outtime.max()])  # select last outtime within the remaining
        .reset_index(drop=True)
    )

    # Assertion checks
    tests.test_single_admission_per_patient(admissions_ed_S1)
    tests.test_outtime_after_intime(admissions_ed_S1)

    # Save data
    admissions_ed_S1.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_ed_S1.csv", index=False, header=True)
    print("=", end="\r")    

    """
    Step 2: Subset based on ward data:
        a) Remove admissions where the first ward is not the Emergency Department.
    """

    # Within the list of transfers, subset to list of above patients, and identify the first ward for each patient.
    patients_ed_first_ward = (
        transfers_core
        .query("subject_id in @admissions_ed_S1.subject_id.values")  # subset to current pats.
        .groupby("subject_id", as_index=True)  # compute per patient
        .progress_apply(lambda x: x[x.intime == x.intime.min()])  # select first transfer intime
        .reset_index(drop=True)
        .query("careunit == 'Emergency Department'")  # remove patients with non-ED first ward
        .query("eventtype == 'ED'")  # remove patients with non-ED first event type
    )
    # Note that the code above does not care that some transfers might be recorded multiple times (e.g. same intime)

    # Select only those admissions from S1 that match the admissions we identified previously
    admissions_ed_S2 = (
        patients_ed_first_ward # Merge will consider only those rows from S1 and patients_ed_first_ward
        .merge(admissions_ed_S1, on=["subject_id", "hadm_id", "intime", "outtime"], how="inner")
        # Drop Rows that have missing values in the stay_id column
        .dropna(subset=["stay_id"])
        .reset_index(drop=True)
    )

    # Check correct processing - i.e. did the correct admissions and patients get merged?
    tests.test_is_correctly_merged(admissions_ed_S2)
    print("==", end="\r")

    # Save to CSV
    admissions_ed_S2.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S2.csv", index=True, header=True)

    """
    Step 3: We look at eventual patient hospital transfers.
        a) For each patient, consider all admissions after ED (defined as intime > ED intime)
        b) Identify patients without further admissions;
        c) Identify patients with further admissions into non-admissable wards (e.g. Partum, Psychiatry);
        d) Remove patients satisfying (b) and (c) from the list of patients.
    """

    # Compute list of second or later ward transfers for our patients
    admissible_transfers_post_ED = (
        transfers_core
        .query("subject_id in @admissions_ed_S2.subject_id.values")  # subset to current pats.
        .groupby("subject_id", as_index=True)  # get list of transfers per patient
        .progress_apply(lambda x: (x
                                    .query("intime > @x.intime.min()")
                                    .sort_values("intime")
                                    )
                        )  # Select transfer after first transfer, and sort values
        .reset_index(drop=True)  # Reset index
        .groupby("subject_id", as_index=True)  # Filter per patient
        .filter(lambda x: ~ x.careunit.isin(DEFAULT_CONFIG["WARDS_TO_REMOVE"]).any())
        # Remove patients with transfers to irrelevant wards
        .reset_index(drop=True)  # Reset index again after groupby
    )

    # Compute the first ward after ED (if exists)
    wards_immediately_after_ED = (
        admissible_transfers_post_ED
        .groupby("subject_id", as_index=True)  # Compute second ward per patient
        .progress_apply(_compute_earliest_transfer_if_exists)  # Compute second ward info
        .reset_index(drop=True)
    )

    # Finally, merge the dataframes and remove empty rows for stay id - this is also important for estimating ESI
    ED_admissions_and_next_admissions = (
        wards_immediately_after_ED
        .merge(admissions_ed_S2, on=["subject_id", "hadm_id"], how="right",
            suffixes=("_next", ""))
        .dropna(subset=["stay_id"])
    )

    # Add patient core information
    admissions_ed_S3 = (
        ED_admissions_and_next_admissions
        .merge(patients_core, on="subject_id", how="inner")
        .reset_index(drop=True)
    )

    # Save
    admissions_ed_S3.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S3.csv", index=True, header=True)
    print("===", end="\r")

    """
    Step 4: 
        Feature Extraction - Derive age and ESI. Convert DoD to datetime. Convert gender to 1 (M) or 0 (F).
        Cohort Selection based on age (above admissable threshold) and ESI (2, 3, 4) - missing not allowed.
        Remove mismatches between admission times (intime, outtime, intime_next, outtime_next) <= deattime.
        Removing features that are not needed.
    """

    # Feature Processing
    admissions_ed_S4 = (
        admissions_ed_S3
        .assign(age=lambda x: x.intime.dt.year - x.anchor_year + x.anchor_age)  # derive age
        # derive ESI from triage_ed data
        .assign(ESI=lambda x: triage_ed.set_index("stay_id").loc[x.stay_id.values, "acuity"].values)
        .assign(deathtime=lambda x: pd.to_datetime(x.dod, format="%Y-%m-%d"))
        .assign(gender=lambda x: x.loc[:, "gender"].replace(["M", "F"], [1,0]))
        .query("age >= @DEFAULT_CONFIG['AGE_LOWER_BOUND']", engine="python")  # remove patients below age threshold
        .query("ESI in [2,3,4]")  # remove patients with ESI 1 or 5
        .dropna(subset=["ESI"])  # Remove patients with missing ESI
        .query("intime.dt.date <= deathtime | deathtime.isna()", engine="python")  # Remove if intime > deathtime
        .query("outtime.dt.date <= deathtime | deathtime.isna()", engine="python")  # Remove if outtime > deathtime
        .query("intime_next.dt.date <= deathtime | deathtime.isna()", engine="python")  # Remove if intt_next > deathtime
        .query("outtime_next.dt.date <= deathtime | deathtime.isna()", engine="python")  # Remove if outt_next > deathtime
        .drop(labels=["anchor_year", "anchor_age", "anchor_year_group", "dod"], axis=1)  # remove feats
    )

    # Check Correct Processing
    tests.test_age_ESI_processed_successfully(admissions_ed_S4)

    # Save to CSV
    admissions_ed_S4.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_S4.csv", index=True, header=True)
    print("====", end="\r")

    # Final processing Check
    tests.test_admissions_processed_correctly(admissions_ed_S4)
    admissions_ed_S4.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "admissions_intermediate.csv", index=True, header=True)
    print("=====", end="\r")


if __name__ == "__main__":
    main()
