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

    # Print info about test function
    print("\nTesting there are no repeated stay_id.")

    assert df["stay_id"].nunique() == df.shape[0]

    # Output message
    print("Test passed!")

def test_non_missing_stays_and_times(df):

    # Print info about test function
    print("\nTesting there are no missing stay_id, intime or outtime.")

    assert df["stay_id"].isna().sum() == 0
    assert df["intime"].isna().sum() == 0
    assert df["outtime"].isna().sum() == 0

    # Output message
    print("Test passed!")

def test_single_admission_per_patient(df):

    # Print info about test function
    print("\nTesting there is one admission per patient.")

    assert df["subject_id"].nunique() == df.shape[0]

    # Output message
    print("Test passed!")

def test_outtime_after_intime(df):

    # Print info about test function
    print("\nTesting outtime is after intime.")

    assert df["outtime"].ge(df["intime"]).all()

    # Output message
    print("Test passed!")

def test_ed_is_first_ward(df):

    # Print info about test function
    print("\nTesting ED is first ward.")

    assert (df["eventtype"].eq("ED") & df["careunit"].eq("Emergency Department")).all()

    # Output message
    print("Test passed!")

def test_is_correctly_merged(df):

    # Print info about test function
    print("\nTesting admissions are correctly merged.")

    assert df["subject_id"].nunique() == df.shape[0]
    assert df[["subject_id", "intime", "outtime", "stay_id"]].isna().sum().sum() == 0
    assert df["subject_id"].duplicated().sum() == 0
    test_outtime_after_intime(df)  # Check previous tests
    test_ed_is_first_ward(df)  # Check previous tests

    # Output message
    print("Test passed!")

def test_next_transfer_is_consistent(df):

    # Print info about test function
    print("\nTesting next transfer information is consistent.")

    assert (~ df["eventtype_next"].eq("ED") | df["eventtype_next"].isna()).all()  # next event is missing or not ED
    assert (df["intime_next"].ge(df["outtime"]) | df["intime_next"].isna()).all()  # next intime >= outtime (if exists)
    assert (df["outtime_next"].ge(df["intime_next"]) | df["outtime_next"].isna()).all()  # next outtime >= next intime

    # Output message
    print("Test passed!")

def test_age_ESI_processed_successfully(df):

    # Print info about test function
    print("\nTesting age and ESI are correctly processed.")

    assert not df[["age", "ESI"]].isna().any().any()
    assert df["age"].ge(16).all()
    assert 'int' in str(df["age"].dtype)
    assert df["ESI"].isin([2, 3, 4]).all()

    # Output message
    print("Test passed!")

def test_admission_times_before_death(df):
    "Note deathtime only contains date values (no hour), therefore we can only compare dates."

    # Print info about test function
    print("\nTesting admission times are before death (if exists).")

    def test_datetime_before_death(col):
        assert (df[col].dt.date.le(df["deathtime"]) | df["deathtime"].isna() | df[col].isna()).all()

    test_datetime_before_death("intime") # intime <= dod (if exists)
    test_datetime_before_death("outtime")  # outtime <= dod (if exists)

    # Also check for next admission times if applicable
    if "intime_next" in df.columns and "outtime_next" in df.columns:
        test_datetime_before_death("intime_next") # intime_next <= dod
        test_datetime_before_death("outtime_next") # outtime_next <= dod

    # Output message
    print("Test passed!")

def test_is_unique_ids(df: pd.DataFrame, *args):
    """Check whether there are any duplicate values across all id columns"""

    # Print info about test function
    print(f"\nTesting ids are unique for params {args}")

    output = True

    for arg in args:  # Iterate through each column
        has_repeated = df[arg].dropna().duplicated().sum() > 0
        if has_repeated:
            print(f"There are duplicate values for id {arg}")

        output = output and not has_repeated

    assert output

    # Output message
    print("Test passed!")

def test_is_complete_ids(df: pd.DataFrame, *args):
    """Check no missing values across id columns"""

    # Print info about test function
    print(f"\nTesting ids are complete for params {args}")

    output = True

    for arg in args:  # Iterate through each column
        has_missing = df[arg].isna().sum() > 0
        if has_missing:
            print(f"There are missing values values for id {arg}")

        output = output and not has_missing

    assert output

def test_admissions_processed_correctly(df: pd.DataFrame):
    """
    Function to check intermediate processing of admissions is correct. The following are done:
    1. Entrance/Exit times are consistent (e.g. outtime>=intime, or next_intime>=outtime, ...) or they are missing.
    2. Death dates are consistent with admission times.
    3. Identifiers are unique.
    4. Subject and Stay id and ED times are complete.
    5. Feature values make sense
    """

    # Print info about test function
    print("\nTesting admissions processed correctly...")

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

    # Output message
    print("Test passed!")
    print("Admissions correctly computed! Safe to go ahead.")


# ========== RELEVANT FOR VITAL PROCESSING ==========

def test_charttime_between_intime_outtime(df):
    "Check whether charttime observations are within ed emergency endpoints."

    # Print info about test function
    print("\nTesting charttime observations are within ed emergency endpoints.")

    cond1 = df["charttime"].le(df["outtime"]).all()
    cond2 = df["charttime"].ge(df["intime"]).all()

    assert cond1 and cond2

    # Output message
    print("Test passed!")

def test_ids_subset_of_cohort(cur_df, cohort_df):
    "Check whether stay id and subject id are subset of cohort data."

    # Print info about test function
    print("\nTesting stay id and subject id are subset of cohort data.")

    cond1 = cur_df["subject_id"].isin(cohort_df["subject_id"]).all()
    cond2 = cur_df["stay_id"].isin(cohort_df["stay_id"]).all()

    assert cond1 
    assert cond2

    # Output message
    print("Test passed!")

def test_stays_have_sufficient_data(df, info_dic):
    "Check remaining stays have sufficient data based on info dic parameters"

    # Print info about test function
    print("\nTesting stays have sufficient data based on info dic parameters.")

    # Extract info
    feats = info_dic["VITALS_RENAMING_DIC"].values()
    max_na_prop = info_dic["NA_PROP_THRESH"]
    min_num = info_dic["MIN_NUM_OBSERVS"]

    # Conditions
    cond1 = (
        df
        .groupby("stay_id")             # Group by stay id
        .count()                        # Count number of observations per stay
        .iloc[:, 0]                     # Select only first column (count returns all columns with same value)
        .ge(min_num)                    # Check if number of observations is greater than min_num
        .all()                          # Assert all satisfy condition
    )
    cond2 = (
        df
        .groupby("stay_id")            # Group by stay id
        .apply(lambda x:
               x[feats].isna().sum(axis=0) 
               / 
               x.shape[0]
        )                           # Compute proportion of missing values per feature
        .le(max_na_prop)           # Check if proportion of missing values is less than max_na_prop
        .all(axis=1)               # Assert all feats satisfy condition per patient
        .all()                     # Assert all patients satisfy condition
    )
    
    assert cond1
    assert cond2

    # Output message
    print("Test passed!")

def test_sorted_by_resampled_col(df, resampled_col):
    """
    Check dataframe is sorted based on resampled col.
    """

    # Print info about test function
    print("\nTesting dataframe is sorted based on resampled col.")

    assert df.groupby("subject_id")[resampled_col].is_monotonic_increasing.all()

    # Output message
    print("Test passed!")

def test_resampling_starts_at_0(df, resampled_col):
    """
    Tests whether the resampled time col starts at 0 for each patient.

    Params:
    - df: pd.DataFrame with resampled data.
    - resampled_col: string indicating the column of the new data.
    """

    # Print info about test function
    print("\nTesting resampling data starts at 0 for each patient.")

    # Assert per patient
    assert (
        df
        .groupby("subject_id")[resampled_col]            # Compute per patient
        .min()                                          # Get min
        .dt.total_seconds()                             # Convert to seconds
        .eq(0.0)                                        # Check if 0
        .all()                                          # Check all patients are 0
    )

    # Output message
    print("Test passed!")

def test_resampling_from_min_to_max_per_pat(df, resampled_col):
    """
    Tests whether the resampled col is linear from min to max per patient, according to resampling rule.
    """

    # Print info about test function
    print("\nTesting resampling data is linear from min to max per patient.")

    # Useful computation
    zero_td = pd.Timedelta(seconds=0, microseconds=0)

    # Compute resampling rule minimum 
    sampling_td = (
        df
        .query("subject_id==@df.subject_id.iloc[0]")          # get new time col for first patient
        .query(f"{resampled_col}.dt.total_seconds() > 0")     # remove non 0 values (query behaves weird with pure timedelta)
        [resampled_col]                                       # get resampled col
        .min(axis=0)                                          # get min
    )

    # Assert per patient linearity between max and min
    assert (
        df
        .groupby("subject_id")           # Compute per patient
        .progress_apply(lambda x:
            x[resampled_col]            # Get resampled col
            .eq(
                pd.timedelta_range(     # timedelta_range defines series starting at 0 with sampling_td as freq
                    start=zero_td,
                    periods=x.shape[0],
                    freq=sampling_td
                )
                .values
            )
            .all(axis=0)        # Check all time indices match
        )
        .all(axis=0)      # Check condition for all patients
    )

    # Output message
    print("Test passed!")

def test_last_observation_within_window_to_outtime(df, td_window, time_feats):
    """
    Check if last observation is AT MOST td_window before admission outtime.

    Params:
    - df: pd.DataFrame with resampled data.
    - td_window: pd.Timedelta object indicating the max time_window allowed between last observation and admission outtime.
    - time_feats: list of strings indicating the time features in df (from which we can extract the last seen observation).
    """

    # Print info about test function
    print("\nTesting last observation is AT MOST td_window before admission outtime.")

    # Iterate over dataframe
    assert (
        df
        .groupby("subject_id")    
        .progress_apply(lambda x:
            x[~ x[time_feats].isna().all(axis=1)]
            .sampled_time_to_end
            .min()
        )
        .le(td_window)
        .all()
    )

    # Output message
    print("Test passed!")

def test_vitals_processed_correctly(df: pd.DataFrame, config_dic: dict):
    """
    Check vitals were processed correctly and make sense.

    - Check for completeness of identifiers (patient and time).
    - Check for correctedness of next transfer information.
    - Check sufficient temporal data.
    - Check last observation sufficiently close to outtime.
    - Check for correctedness of resampling.
    """

    # Print info about test function
    print("\nTesting vitals were processed correctly and make sense.")

    # First check
    test_is_complete_ids(df, "subject_id", "stay_id", "sampled_time_to_end")

    # second Check
    test_next_transfer_is_consistent(df)

    # Third check
    test_stays_have_sufficient_data(df, config_dic)

    # Fourth Check
    feats = list(config_dic["VITALS_RENAMING_DIC"].values())
    td_window = pd.Timedelta(hours = config_dic["LAST_OBVS_TIME_TO_EXIT"])
    test_last_observation_within_window_to_outtime(df, td_window, feats)

    # Fifth Check
    test_resampling_from_min_to_max_per_pat(df, "sampled_time_to_end")
    
    # Output message
    print("Vitals seem correctly processed!")


# ========== OUTCOME TESTING ===========
