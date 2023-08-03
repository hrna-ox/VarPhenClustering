"""

Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file implements a logger function to plot and log useful information about the model during training, validation and testing.

"""

# region =============== IMPORT LIBRARIES ===============
import csv
import os
import pickle

import datetime
from typing import Dict, List, Union
import wandb


CLASS_METRICS = ["recall", "precision", "micro_f1", "macro_f1", "roc_auc", "pr_auc"]
CLUS_LABEL_MATCH_METRICS = ["rand", "nmi"]
CLUS_QUALITY_METRICS = ["sil", "dbi", "vri"]
# endregion

# ==================== UTILITY FUNCTION ====================


def _mkdirs_if_not_exist(*args):
    """
    Create directories if they do not exist.

    Params:
    - args: list of directories to create
    """
    for _dir in args:
        if not os.path.exists(_dir):
            try: 
                os.makedirs(_dir)
            except PermissionError as e:
                print(_dir)
                raise e

def _make_csv_if_not_exist(save_path: str, header: List = []):
    """
    Create a csv file if it does not exist, and append the header.

    Args:
    - save_path: path to save the csv file
    - header: list of strings to append to the header
    """
    if not os.path.exists(save_path):
        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
        
def _write_to_csv_if_exists(save_path: str, data: List = []):
    """
    Write data to csv file if it exists. If it does not, then print a warning.

    Args:
    - save_path: path to save the csv file
    - data: list of strings to append to the csv file
    """

    if os.path.exists(save_path):
        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(data)
    else:
        print(f"WARNING: {save_path} does not exist. Data not saved.")


def _get_save_timestamp():
    """
    Given the current timestamp, return a string with the format "YYYY-MM-DD_HH-MM-SS" for adding to save paths.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


# region =============== MAIN ===============

class BaseLogger:
    """
    Python Class to log the training, validation and testing general models. Includes children that specify particular behaviours
    for particular models.
    """
    def __init__(self, save_dir: str, K_fold_idx: int = 1):
        "Object initialization."

        
        self.K_fold_idx = K_fold_idx

        # Get new run to avoid matches
        new_run_dir = self._log_new_run_dir(save_dir)
        self.exp_save_dir = f"{new_run_dir}/fold_{self.K_fold_idx}"
        print("\n\n Logging experiments in {}".format(self.exp_save_dir))


        # Make target folder for experiment
        self._make_save_subfolder()

        # Initialize train/test folders
        self._make_save_subfolder(sub_fd="train")
        self._make_save_subfolder(sub_fd="test")
        self._make_save_subfolder(sub_fd="val")


    def _make_save_subfolder(self, sub_fd: str = ""):
        """
        Make folder based on target sub-folder. The default behavior is None which defaults to creating self.save_dir if it does not
        exist.

        Args:
            sub_fd str (default ""). Sub-folder to create within the main folder structure.
        """

        # Append paths 
        if sub_fd is None:
            target_fd = self.exp_save_dir
        
        else:
            target_fd = f"{self.exp_save_dir}/{sub_fd}"

        _mkdirs_if_not_exist(target_fd)
        


    def _log_new_run_dir(self, save_dir: str = "exps/"):
        """
        Determine new save directory for the current run based on the input save_dir and the existing runs, labelled under "Run_{}"...

        Params:
        - save_dir: directory to save the plots and metrics.

        Outputs:
        - save_dir of the form 'save_dir/Run_XX', where XX is the next available number.
        """

        # Get all directories in save_dir
        _mkdirs_if_not_exist(save_dir)
        dirs = os.listdir(save_dir)

        # Get all Run IDs
        if dirs == []:
            run_ids = []
        else:
            run_ids = [int(_dir.split("_")[1]) for _dir in dirs if "Run_" in _dir]

        # Get next Run ID
        if len(run_ids) == 0:
            next_run_id = 0
        else:
            next_run_id = max(run_ids) + 1

        # Create new directory
        new_dir = f"{save_dir}/Run_{next_run_id}"

        # Append time stamp
        dir_with_time = new_dir + "_" + _get_save_timestamp()

        return dir_with_time


    def log_objects(self, objects: Dict = {}, save_name = "objects"):
        "Log target objects into the experiment folder using pickle"
        
        # Log objects
        with open(f"{self.exp_save_dir}/{save_name}.pkl", "wb") as f:
            pickle.dump(objects, f)

class ClassifierLogger(BaseLogger):
    "Add Logging functionality for classifier related information."
    
    def __init__(self, class_names: List = [], *args, **kwargs):
        "Object Initialization."

        self.class_names = class_names
        self._num_classes = len(self.class_names)

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Initialize supervised score trackers
        self._init_supervised_score_trackers(subdir="train")
        self._init_supervised_score_trackers(subdir="val")
        self._init_supervised_score_trackers(subdir="test")
    
    def _init_supervised_score_trackers(self, subdir: str = "train"):
        "Initialize tracker for supervised scores."

        # Create CSV for accuracy
        accuracy_csv_path = f"{self.exp_save_dir}/{subdir}/accuracy.csv"
        accuracy_csv_header = ["iter", "accuracy"]
        _make_csv_if_not_exist(accuracy_csv_path, accuracy_csv_header)

        # Create CSV for Recall, Precision, Micro F1, Macro F1, ROC AUC, PR AUC
        multiclass_header = ["iter"] + self.class_names

        for metric in CLASS_METRICS:
            csv_path = f"{self.exp_save_dir}/{subdir}/{metric}.csv"
            _make_csv_if_not_exist(csv_path, multiclass_header)

        # Create CSV for confusion matrix
        _cm_header = ["iter"] + [f"Tr_{i}_Pr_{j}" for j in self.class_names for i in self.class_names]
        _cm_path = f"{self.exp_save_dir}/{subdir}/confusion_matrix.csv"
        _make_csv_if_not_exist(_cm_path, _cm_header)


    def log_supervised_performance(self, iter: Union[str, int] = 1, scores_dic: Dict = {}, subdir: str = "train"):
        """
        Log the supervised performance of the model.

        Args:
        - iter: int indicating the current iteration (default = 1)
        - scores_dic: dictionary containing the scores for the different classes.
        - subdir: str indicating whether the subfolder used for the logger in testing or validation or testing.
        """

        # Get Subfolder within experiment based on the mode
        cur_save_fd = f"{self.exp_save_dir}/{subdir}"

        # Make csvs if they do not exists
        self._init_supervised_score_trackers(subdir=subdir)

        # Log accuracy
        if "accuracy" in scores_dic.keys():
            _write_to_csv_if_exists(f"{cur_save_fd}/accuracy.csv", [iter, scores_dic["accuracy"]])

        # Log all multiclass metrics
        for metric in CLASS_METRICS:
            _write_to_csv_if_exists(f"{cur_save_fd}/{metric}.csv", [iter] + scores_dic[metric])

        # Log confusion matrix
        _write_to_csv_if_exists(f"{cur_save_fd}/confusion_matrix.csv", [iter] + scores_dic["confusion_matrix"].flatten().tolist())

class ClusteringLogger(BaseLogger):
    "Expand parent Base Logger class to incorporate clustering metrics for validation and testing."    

    def __init__(self, num_clusters: int = 2, temporal: bool = False, T: int = 1, *args, **kwargs):
        "Object Initialization. Temporal parameter indicates whether the clustering is temporal or not."

        # Unpack object
        self._clus_is_temp = temporal
        self.num_time_steps = T
        self.num_clusters = num_clusters
        self.clus_names = [f"Clus_{i}" for i in range(1, 1 + self.num_clusters)]

        # Initialize parent class
        super().__init__(*args, **kwargs)

        # Initialize unsupervised score trackers
        self._init_unsupervised_score_trackers(subdir="train")
        self._init_unsupervised_score_trackers(subdir="val")
        self._init_unsupervised_score_trackers(subdir="test")


    def _init_unsupervised_score_trackers(self, subdir: str = "train"):
        "Initialize tracker for unsupervised scores."


        # If clustering is not temporal then combine metrics into two CSVs
        if not self._clus_is_temp:

            # Create CSV for any metrics related to cluster labels and ground truth labels (classes)
            _clus_label_match_score_csv_header = ["iter"] + CLUS_LABEL_MATCH_METRICS
            _make_csv_if_not_exist(f"{self.exp_save_dir}/{subdir}/cluster_label_match_scores.csv", _clus_label_match_score_csv_header)

            # Create CSV for Silhouette, Calinski Harabasz, Davies Bouldin
            _clus_quality_score_csv_header = ["iter"] + CLUS_QUALITY_METRICS
            _make_csv_if_not_exist(f"{self.exp_save_dir}/{subdir}/cluster_quality_scores.csv", _clus_quality_score_csv_header)

        
        # If clustering is temporal then create separate CSVs for each metric
        else:

            # Create CSV for all clustering metrics - note they will all have the same header (the time step)
            _temporal_header = ["iter"] + [f"t_{i}" for i in range(1, 1 + self.num_time_steps)]
            for metric in CLUS_LABEL_MATCH_METRICS + CLUS_QUALITY_METRICS:
                
                # Create CSV
                _make_csv_if_not_exist(f"{self.exp_save_dir}/{subdir}/{metric}.csv", _temporal_header)


    def log_clustering_performance(self, iter: Union[str, int] = 1, scores_dic: Dict = {}, subdir: str = "train", temporal: bool = False):
        """
        Log the clustering performance of the model.

        Args:
        - iter: int indicating the current iteration (default = 1)
        - scores_dic: dictionary containing the scores for the different classes.
        - subdir: str indicating whether the subdir the logger is for testing or validation or testing.
        - temporal: bool indicating whether the clustering is assumed to be temporal or not.
        """

        # Get Subfolder within experiment based on the mode
        cur_save_fd = f"{self.exp_save_dir}/{subdir}"

        # Make csvs if they do not exists
        self._init_unsupervised_score_trackers(subdir=subdir)

        # If clustering is not temporal then combine metrics into single CSV
        if not self._clus_is_temp:
                
            # Log METRICS RELATED TO CLUSTER AND LABEL PERFORMANCE
            _clus_label_match_row_append = []
            for metric in CLUS_LABEL_MATCH_METRICS:
                _clus_label_match_row_append.append(scores_dic[metric])
            
            # Log to CSV
            _write_to_csv_if_exists(f"{cur_save_fd}/cluster_label_match_scores.csv", [iter] + _clus_label_match_row_append)

            # Log METRICS RELATED TO CLUSTER QUALITY
            _clus_quality_row_append = []
            for metric in CLUS_QUALITY_METRICS:
                _clus_quality_row_append.append(scores_dic[metric])

            # Log to CSV
            _write_to_csv_if_exists(f"{cur_save_fd}/cluster_quality_scores.csv", [iter] + _clus_quality_row_append)

        
        # If clustering is temporal then create separate CSVs for each metric
        else:
            
            # Iterate over all metrics
            for metric in CLUS_LABEL_MATCH_METRICS + CLUS_QUALITY_METRICS:

                # Log to CSV
                _write_to_csv_if_exists(f"{cur_save_fd}/{metric}.csv", [iter] + scores_dic[metric])


class DLLogger(BaseLogger):
    "Expand parent Base Logger class to incorporate loss tracking for deep learning models and also metrics for validation and testing."    
    def __init__(self, loss_names: List[str] = [], *args, **kwargs):
        
        # Unpack object
        self.loss_names = loss_names

        # Initialize parent class
        super().__init__(*args, **kwargs)


        # Initialize loss trackers
        self._init_loss_trackers(subdir="train")
        self._init_loss_trackers(subdir="val")
        self._init_loss_trackers(subdir="test")

    def _init_loss_trackers(self, subdir: str = "train"):
        "Initialize tracker for losses."

        # Create CSV for losses
        _loss_csv_header = ["epoch"] + self.loss_names
        _make_csv_if_not_exist(f"{self.exp_save_dir}/{subdir}/loss_tracker.csv", _loss_csv_header)


    def log_losses(self, losses: List, epoch: Union[str, int] = 1, subdir: str = "train"):
        """
        Log the losses by deep learning model into save directory. 

        Args:
        - losses: List with loss information
        - epoch: current epoch for training. This parameter is disregarded if mode is set to 'test'.
        - subdir: str indicating the subdir the logger is for testing or validation or training.
        """
        # Make csvs if they do not exists
        self._init_loss_trackers(subdir=subdir)

        # Add epoch information to losses dictionary at the beginning
        losses_as_dict = {f"{subdir}/{self.loss_names[i]}": losses[i] for i in range(len(self.loss_names))}

        # Log to Weights and Biases
        wandb.log(losses_as_dict, step=epoch)


        # Save to CSV file
        _write_to_csv_if_exists(f"{self.exp_save_dir}/{subdir}/loss_tracker.csv", [epoch] + losses)


# ============================= Logger for Particular Models =============================

class DirVRNNLogger(DLLogger, ClusteringLogger, ClassifierLogger):
    "Logger for DirVRNN Model."

    def __init__(self, save_dir: str = "exps/DirVRNN", class_names: List[str] = [], K_fold_idx: int = 1, 
                 loss_names: List[str] = [], *args, **kwargs):
        
        # Initialize parent classes
        super().__init__(save_dir=save_dir, class_names=class_names, K_fold_idx=K_fold_idx,
                         loss_names=loss_names, *args, **kwargs)
    

# endregion
