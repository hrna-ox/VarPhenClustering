""""
Processing script for extracting vital information.

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

import src.data_processing.MIMIC.data_utils as utils


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
					parse_dates=DEFAULT_CONFIG["VITALS_TIME_VARS"])
	
	vital_signs_ed = pd.read_csv(
					DEFAULT_CONFIG["DATA_FD"] + "ed/vitalsign.csv", 
					index_col=0, 
					header=0, 
					low_memory=False,
					parse_dates=["charttime"]) 

	# Re-check admissions intermediate, vitals were correctly processed/loaded
	tests.admissions_processed_correctly(adm_inter)
	tests.test_is_complete_ids(vital_signs_ed, "stay_id")    # No NaN on stay id column.

	# ------------------------------------- // -------------------------------------
	"""
	Step 1: Subset vital signs data based on the subsetted cohort given by adm_inter. We do so via merging.
	Then rename columns.
	Then keep only observations taken between registered ED intime and ED outtime.
	"""

	# Merge data on stay id (there are no NaNs)
	vitals_S1 = (vital_signs_ed
	      		.merge(right=adm_inter, how="inner", on="stay_id") # merge on stay_id (keep those that are in adm_inter)
			    .rename(DEFAULT_CONFIG["VITALS_RENAMING_DIC"], axis=1) # Rename columns
			    .query("chartime >= intime")       # Observations after intime
			    .query("chartime <= outtime")      # Observations before outtime
		  )
	
	# Save
	vitals_S1.to_csv(DEFAULT_CONFIG["SAVE_FD"] + "vitals_S1.csv", index=True, header=True)
	
	"""
	Step 2: Remove admissions with too much missing data. This includes:
	
	"""


	vitals_S1 = utils.subsetted_by(vital_signs_ed, admissions, "stay_id")
	admissions.set_index("stay_id", inplace=True)
	vitals_S1[["intime", "outtime"]] = admissions.loc[vitals_S1.stay_id.values, ["intime", "outtime"]].values
	vitals_S1.to_csv(SAVE_FD + "vitals_S1.csv", index=True, header=True)

	"""
	
	"""
	# Subset Endpoints of vital observations according to ED endpoints
	vitals_S2 = vitals_S1[vitals_S1["charttime"].between(vitals_S1["intime"], vitals_S1["outtime"])]
	vitals_S2.rename(VITALS_NAMING_DIC, axis=1, inplace=True)
	vitals_S2.to_csv(SAVE_FD + "vitals_S2.csv", index=True, header=True)

	"""
	Remove admissions with high amounts of missingness.
	"""
	# Subset to patients with enough data
	vital_feats = list(VITALS_NAMING_DIC.values())
	# vitals_S3 = utils.remove_adms_high_missingness(vitals_S2, vital_feats, "stay_id",
	# 	                                     min_count=admission_min_count, min_frac=vitals_na_threshold)
	vitals_S3 = vitals_S2
	vitals_S3.to_csv(SAVE_FD + "vitals_S3.csv", index=True, header=True)

	"""
	Compute time to end of admission, and group observations into blocks.
	"""
	# Resample admissions according to group length
	vitals_S4 = utils.compute_time_to_end(vitals_S3, id_key="stay_id", time_id="charttime", end_col="outtime")
	vitals_S4 = utils.conversion_to_block(vitals_S4, id_key="stay_id", rule=resampling_rule, time_vars=vital_feats,
		                            static_vars=["stay_id", "intime", "outtime"])
	vitals_S4.to_csv(SAVE_FD + "vitals_S4.csv", index=True, header=True)

	"""
	Apply Step 3 again with the blocked data.
	"""
	# Ensure blocks satisfy conditions - min counts, proportion of missingness AND time to final outcome
	vitals_S5 = utils.remove_adms_high_missingness(vitals_S4, vital_feats, "stay_id",
	                                     min_count=admission_min_count, min_frac=vitals_na_threshold)

	"""
	Consider those admissions with observations with at most an observations 1.5 hours before outtime 
	"""
	vitals_S5 = vitals_S5[vitals_S5["time_to_end_min"].dt.total_seconds() <= admission_min_time_to_outtime * 3600]
	vitals_S5.to_csv(SAVE_FD + "vitals_intermediate.csv", index=True, header=True)

