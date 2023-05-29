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

def _resample(x, time_feats, resampling_rule, resampling_col):
	"""
	Resample the dataset given time to end column based on the resampling rule.

	Params:
	- x: dataframe of charttime data for a single patient / stay id. 
	- time_feats: the feature values that are being resampled.
	- resampling_rule: how often to resample the new data.
	- resampling_col: the column under which to do the resampling.

	Outputs;
	- dataframe with resampled data according to resampling_rule. All other information from x, not included in time_feats is also kept (
	assumed to be static
	).
    
	"""

	# Check dataframe is sorted by some variant of 'time to end'
	assert x[resampling_col].is_monotonic_decreasing

	# Create intermediate output to resample
	_x = x[[resampling_col] + time_feats]
	
	# Make empty value at 0.00.00 timedelta to ensure that the resampling starts at 0.
	zero_timedelta_df = pd.DataFrame({resampling_col: [pd.Timedelta(seconds=0)]}, # specify timedelta value
									columns=_x.columns  # all other columns are NaN by default.
								)				
	new_x = pd.concat([_x, zero_timedelta_df], axis=0)

    # Apply resampling method for dataframe
	resampled_x = new_x.resample(on=resampling_col,                     # main column to resample
			   					rule=resampling_rule,                # new sampling rule 
								closed="left",                       # left closed (i.e. [0,1), [1, 2), ...]
								label="left"                         # left label (i.e. "0" means [0,1), "1" means [1, 2)...)
							).mean()								 # Compute mean of all observations within new period.

	# Add index information, which is the resampled time index
	resampled_x.index.name = "sampled_" + resampling_col
	resampled_x.reset_index(drop=False, inplace=True)

	# Now add all other vars
	static_vars = [col for col in x.columns if col not in time_feats + [resampling_col]]
	pat_static_info = x[static_vars].iloc[0, :]
        
    # Add to resampled data
	resampled_x[static_vars] = pat_static_info

	return resampled_x


def _time_series_removal_criteria(df, dic_params):
	"""
	Given dataframe with temporal observations df, as well as input dic_parameters, remove patient admissions based on:
	- Minimum number of observations
	- Maximum percentage of missing observations (below threshold)

	Params:
	- df: dataframe with temporal observations.
	- dic_params: dictionary with parameters for removal criteria.

	Outputs:
	- df: dataframe with temporal observations after removal.
	"""

	# Access parameters
	min_num_obs = dic_params["MIN_NUM_OBSERVS"]
	max_na_prop_thresh = dic_params["NA_PROP_THRESH"]
	feats_to_check = list(dic_params["VITALS_RENAMING_DIC"].values())

	# Transform data
	new_df = (
		df
		.groupby("stay_id", as_index=True)
		.filter(lambda x:
				x.shape[0] >= min_num_obs 
                    and
                x[feats_to_check]
					.isna()
					.sum()
					.le(
						x.shape[0] * max_na_prop_thresh
					)
					.all()
			)
		.reset_index(drop=True)
	)	

	return new_df


def main():

	"""
	Check admissions cohort has been processed previously. If not, run file.	
	"""

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
		parse_dates=DEFAULT_CONFIG["VITALS_TIME_VARS"]
	)
	
	vital_signs_ed = pd.read_csv(
		DEFAULT_CONFIG["DATA_FD"] + "ed/vitalsign.csv", 
		index_col=None, 
		header=0, 
		low_memory=False,
		parse_dates=["charttime"]
	) 

	# Re-check admissions intermediate, vitals were correctly processed/loaded
	tests.test_admissions_processed_correctly(adm_inter)
	tests.test_is_complete_ids(vital_signs_ed, "subject_id", "stay_id")    # No NaN on stay id column.

	# ------------------------------------- // -------------------------------------
	"""
	Step 1: Subset vital signs data based on the subsetted cohort given by adm_inter. We do so via merging.
	Then rename columns.
	Then keep only observations taken between registered ED intime and ED outtime.
	"""

	# Merge data on stay id (there are no NaNs)
	vitals_S1 = (
		vital_signs_ed
		.merge(right=adm_inter, how="inner", on=["subject_id", "stay_id"]) # merge on stay_id (keep those that are in adm_inter)
		.rename(DEFAULT_CONFIG["VITALS_RENAMING_DIC"], axis=1) # Rename columns
		.query("charttime >= intime")       # Observations after intime
		.query("charttime <= outtime")      # Observations before outtime
	)
	
	# Test processing and save
	tests.test_charttime_between_intime_outtime(vitals_S1)
	tests.test_ids_subset_of_cohort(vitals_S1, adm_inter, "subject_id", "stay_id")

	vitals_S1.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S1.csv", index=True, header=True)
	print("=", end="\r")
	
	"""
	Step 2: Remove admissions with too much missing data. This includes:
	a) less than MIN_NUM_OBSERVS observations
	b) at least NA_PROP_THRESH * # observations NAs on any vital sign
	"""

	vitals_S2 = _time_series_removal_criteria(vitals_S1, 
					   						dic_params=DEFAULT_CONFIG)

	# Test and save
	tests.test_stays_have_sufficient_data(vitals_S2, DEFAULT_CONFIG)

	vitals_S2.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S2.csv", index=True, header=True)
	print("==", end="\r")

	"""
	Step 3: Compute time to end for each charttime measurement.
	Resample based on time to end.
	"""

	# Useful objects to compute
	feats_to_check = list(DEFAULT_CONFIG["VITALS_RENAMING_DIC"].values())
	resampling_rule = DEFAULT_CONFIG["RESAMPLING_RULE"]

	vitals_S3 = (
		vitals_S2
		.assign(time_to_end=lambda x: x["outtime"] - x["charttime"])
		.sort_values(by=["stay_id", "time_to_end"], ascending=[True, False])
		.groupby("stay_id", as_index=False)
		.progress_apply(lambda x: 
		  			_resample(x, 
						time_feats = feats_to_check,
						resampling_rule = resampling_rule,
						resampling_col="time_to_end")
					)
		.reset_index(drop=True)
		.assign(charttime_ub=lambda x: x["outtime"] - x["sampled_time_to_end"])     # Define upper bound on charttime by substracting from outtime.
	)
	
	# Test and save
	tests.test_sorted_by_resampled_col(vitals_S3, "sampled_time_to_end")
	tests.test_resampling_starts_at_0(vitals_S3, "sampled_time_to_end")
	tests.test_resampling_from_min_to_max_per_pat(vitals_S3, "sampled_time_to_end")

	vitals_S3.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S3.csv", index=True, header=True)
	print("===", end="\r")


	"""
	Step 4: Apply step 2 with the blocked data
	"""

	vitals_S4 = _time_series_removal_criteria(vitals_S3, 
					   						dic_params=DEFAULT_CONFIG)

	# Test and save
	tests.test_stays_have_sufficient_data(vitals_S4, DEFAULT_CONFIG)

	vitals_S4.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S4.csv", index=True, header=True)
	print("====", end="\r")

	"""
	Step 5 - ensure the closest observation to admission end is not too far away
	"""

	# Useful timedelta defn
	max_allowed_time_before_outtime = pd.Timedelta(hours = DEFAULT_CONFIG["LAST_OBVS_TIME_TO_EXIT"])

	# Ensure the first seen observation is not too far away from end of stay. Resampling starts at 0, however, observations for time varying features
	# will be missing until we see an observation. Therefore, we can "recover" the first observation by taking the minimum of the non-NA observations.
	vitals_S5 = (
		vitals_S4
		.groupby("stay_id", as_index=True)
		.filter(lambda x: 
			x
			.dropna(subset = feats_to_check, how="all")       # Per patient, remove patients where all time feats are missing (i.e. no observations)
			.sampled_time_to_end								# Compute minimum of sampled time to end for all remaining rows
			.min()
				<=
			max_allowed_time_before_outtime						# Impose minimum below threshold
		)
	)	
	
	# Test and save
	tests.test_last_observation_within_window_to_outtime(vitals_S5, max_allowed_time_before_outtime, feats_to_check)

	vitals_S5.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S5.csv", index=True, header=True)
	print("====", end="\r")


	# Testing vital information satisfies above criteria
	tests.test_vitals_processed_correctly(vitals_S5)

	vitals_S5.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_intermediate.csv", index=True, header=True)
	print("=====", end="\r")

if __name__ == "__main__":
    main()
