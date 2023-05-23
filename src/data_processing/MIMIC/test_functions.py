#!./venv/bin/python3
"""

Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk
Last Updated: 15 March 2023

Auxiliary test functions to check data has been processed correctly.

MISSING TYPE HINTING.
"""

# Import libraries and functions
import pandas as pd
from tqdm import tqdm

tqdm.pandas()


# ================= RELEVANT FOR ADMISSION PROCESSING =================
def test_unique_stays(df):
    assert df["stay_id"].nunique() == df.shape[0]


def test_non_missing_stays_and_times(df):
    assert df["stay_id"].isna().sum() == 0
    assert df["intime"].isna().sum() == 0
    assert df["outtime"].isna().sum() == 0


def test_single_admission_per_patient(df):
    assert df["subject_id"].nunique() == df.shape[0]


def test_outtime_after_intime(df):
    assert df["outtime"].ge(df["intime"]).all()


def test_ed_is_first_ward(df):
    assert (df["eventtype"].eq("ED") & df["careunit"].eq("Emergency Department")).all()


def test_is_correctly_merged(df):
    assert df["subject_id"].nunique() == df.shape[0]
    assert df[["subject_id", "intime", "outtime", "stay_id"]].isna().sum().sum() == 0
    assert df["subject_id"].duplicated().sum() == 0
    test_outtime_after_intime(df)  # Check previous tests
    test_ed_is_first_ward(df)  # Check previous tests


def test_next_transfer_is_consistent(df):
    assert (~ df["eventtype_next"].eq("ED") | df["eventtype_next"].isna()).all()  # next event is missing or not ED
    assert (df["intime_next"].ge(df["outtime"]) | df["intime_next"].isna()).all()  # next intime >= outtime (if exists)
    assert (df["outtime_next"].ge(df["intime_next"]) | df["outtime_next"].isna()).all()  # next outtime >= next intime


def test_next_transfer_admissible(df, non_allowed_wards):
    """
    This function determines whether the identification of next transfers makes sense. Checks:
    - missingness is equally spread throughout next transfer information (i.e. either info is present or not).
    - next transfer intime > ED transfer outtime
    - next transfers outtime > next transfer intime
    - next transfer id != current transfer id
    - next eventytpe is not ED and next careunit is not "Emergency Department" or any of the not allowed wards.
    """

    # Check missingness equally spread throughout next transfer info
    next_info_is_missing = df["transfer_id_next"].isna()
    assert df[next_info_is_missing][["transfer_id_next",
                                "eventtype_next",
                                "careunit_next",
                                "intime_next",
                                "outtime_next"]].isna().all().all()

    # Now check time conditions
    assert (df["intime_next"].ge(df["outtime"]) | next_info_is_missing).all()      # intime_next after outtime
    assert (df["outtime_next"].ge(df["intime_next"]) | next_info_is_missing).all()

    # Check ids and ward matches
    assert (df["transfer_id_next"] != df["transfer_id"] | next_info_is_missing).all()    # ids pre-post don't match
    assert not (df["careunit_next"].isin(non_allowed_wards)).any()     # no careunit allowed


def age_ESI_processed_successfully(df):
    assert not df[["age", "ESI"]].isna().any().any()
    assert df["age"].ge(16).all()
    assert 'int' in str(df["age"].dtype)
    assert df["ESI"].isin([2, 3, 4]).all()


def test_admission_times_before_death(df):
    "Note deathtime only contains date values (no hour), therefore we can only compare dates."

    def test_datetime_before_death(col):
        assert (df[col].dt.date.le(df["deathtime"]) | df["deathtime"].isna() | df[col].isna()).all()

    test_datetime_before_death("intime") # intime <= dod (if exists)
    test_datetime_before_death("outtime")  # outtime <= dod (if exists)

    # Also check for next admission times if applicable
    if "intime_next" in df.columns and "outtime_next" in df.columns:
        test_datetime_before_death("intime_next") # intime_next <= dod
        test_datetime_before_death("outtime_next") # outtime_next <= dod


def test_is_unique_ids(df: pd.DataFrame, *args):
    """Check whether there are any duplicate values across all id columns"""
    output = True

    for arg in args:  # Iterate through each column
        has_repeated = df[arg].dropna().duplicated().sum() > 0
        if has_repeated:
            print(f"There are duplicate values for id {arg}")

        output = output and not has_repeated

    assert output


def test_is_complete_ids(df: pd.DataFrame, *args):
    """Check no missing values across id columns"""
    output = True

    for arg in args:  # Iterate through each column
        has_missing = df[arg].isna().sum() > 0
        if has_missing:
            print(f"There are missing values values for id {arg}")

        output = output and not has_missing

    assert output
def admissions_processed_correctly(df: pd.DataFrame):
    """
    Function to check intermediate processing of admissions is correct. The following are done:
    1. Entrance/Exit times are consistent (e.g. outtime>=intime, or next_intime>=outtime, ...) or they are missing.
    2. Death dates are consistent with admission times.
    3. Identifiers are unique.
    4. Subject and Stay id and ED times are complete.
    5. Feature values make sense
    """

    # Within admission time check
    test_outtime_after_intime(df)
    test_next_transfer_is_consistent(df)
    assert df["intime_next"].isna().eq(df["outtime_next"].isna()).all()  # Check both are missing or not

    # Check Death observed any admission times
    test_admission_times_before_death(df)

    # Uniqueness of main.py id columns
    test_is_unique_ids(df, "subject_id", "hadm_id", "stay_id", "transfer_id_next")

    # Completeness of id columns
    test_is_complete_ids(df, "subject_id", "stay_id", "intime", "outtime")

    print("Admissions correctly computed! Safe to go ahead.")


# ========== RELEVANT FOR VITAL PROCESSING ==========

def charttime_between_intime_outtime(df):
    "Check whether charttime observations are within ed emergency endpoints."

    cond1 = df["charttime"].le(df["outtime"]).all()
    cond2 = df["charttime"].ge(df["intime"]).all()

    assert cond1 and cond2

def ids_subset_of_cohort(cur_df, cohort_df):
    "Check whether stay id and subject id are subset of cohort data."

    cond1 = cur_df["subject_id"].isin(cohort_df["subject_id"]).all()
    cond2 = cur_df["stay_id"].isin(cohort_df["stay_id"]).all()

    assert cond1 
    assert cond2

def stays_have_sufficient_data(df, info_dic):
    "Check remaining stays have sufficient data based on info dic parameters"

    # Extract info
    feats = info_dic["VITALS_RENAMING_DIC"].values()
    max_na_prop = info_dic["NA_PROP_THRESH"]
    min_num = info_dic["MIN_NUM_OBSERVS"]

    # COnditions
    cond1 = df.groupby("stay_id").filter(lambda x: x.shape[0] < min_num).empty
    cond2 = df.groupby("stay_id").filter(lambda x: (x[feats].isna().sum() > (
                                    max_na_prop * x.shape[0])).any()).empty
    
    assert cond1
    assert cond2

# def vitals_processed_correctly(df: pd.DataFrame):
#     """
#     Function to check intermediate processing of vitals is correct. The following are done:
#     1. Intime before Outtime
#     2. Time to End Max/Min fall within intime/outtime ED time.
#     3. Sampled Time to End falls within intime/outtime ED time.
#     4. Identifiers are complete.
#     """
#
#     # Intime before Outtime
#     assert test_entrance_before_exit(df["intime"], df["outtime"])
#
#     # Time to End within intime/outtime
#     assert test_entrance_before_exit(df["outtime"] - df["time_to_end_min"], df["outtime"])
#     assert test_time_before_death(df["intime"], df["outtime"] - df["time_to_end_max"])
#
#     # Similar to sampled time to end
#     resampling_rule = "1H"
#     col = f"sampled_time_to_end({resampling_rule})"
#     assert test_entrance_before_exit(df["outtime"] - df[col], df["outtime"])
#     assert test_time_before_death(df["intime"], df["outtime"] - df[col])
#
#     # Completeness of id columns
#     assert test_is_complete_ids(df, "stay_id", col, "intime", "outtime", "time_to_end_min", "time_to_end_max")
#
#     print("Vitals correctly computed! Safe to go ahead.")

# ========== OUTCOME TESTING ===========
