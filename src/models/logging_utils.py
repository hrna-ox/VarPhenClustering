"""

Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file implements a logger function to plot and log useful information about the model during training, validation and testing.

"""

# region =============== IMPORT LIBRARIES ===============
import csv
import os
import pickle
import numpy as np
import pandas as pd
import torch

import datetime
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import wandb

import src.models.loss_functions as LM_utils

# endregion

# ==================== UTILITY FUNCTION ====================
def _log_new_run_dir(save_dir: str = "exps/"):
    """
    If a save directory is not directly given, estimate a new one based on incremental Run IDs.

    Params:
    - save_dir: directory to save the plots and metrics.

    Outputs:
    - save_dir of the form 'save_dir/Run_XX', where XX is the next available number.
    """

    # Get all directories in save_dir
    _mkdirs_if_not_exist(save_dir)
    dirs = os.listdir(save_dir)

    # Get all Run IDs
    run_ids = [int(_dir.split("_")[-1]) for _dir in dirs if "Run_" in _dir]

    # Get next Run ID
    if len(run_ids) == 0:
        next_run_id = 0
    else:
        next_run_id = max(run_ids) + 1

    # Create new directory
    new_dir = os.path.join(save_dir, f"Run_{next_run_id}")

    # Append time stamp
    dir_with_tmst = new_dir + "_" + _get_save_timestamp()

    return dir_with_tmst


def _mkdirs_if_not_exist(*args):
    """
    Create directories if they do not exist.

    Params:
    - args: list of directories to create
    """
    for _dir in args:
        if not os.path.exists(_dir):
            os.makedirs(_dir)


def _logger_make_csv_if_not_exist(save_path: str, header: List = []):
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
        


def _get_save_timestamp():
    """
    Given the current timestamp, return a string with the format "YYYY-MM-DD_HH-MM-SS" for adding to save paths.
    """
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


# region =============== MAIN ===============

class DLLogger:
    """
    Python Class to log the training, validation and testing of Deep Learning Models. Includes loss tracking and performance metrics.
    """
    def __init__(self, save_dir: str, class_names: List = [], K_fold_idx: int = 1, loss_headers: List = []):
        "Object initialization."

        self.save_dir = save_dir
        self.class_names = class_names
        self.K_fold_idx = K_fold_idx
        self.loss_headers = loss_headers

        # Create Save Folder
        self.save_dir = os.path.join(_log_new_run_dir(self.save_dir), f"fold_{self.K_fold_idx}")
        _mkdirs_if_not_exist(self.save_dir)
        print("Saving Experiment to directory: {}".format(self.save_dir))
        

    def _init_exp_fds(self, mode: str = "train"):
        """
        Initialize the directories to save the outputs of Deep Learning Models (any model with train/val/test split.)

        Args:
        - mode: str indicating whether the logger is for testing or validation or testing.
        """

        # Take Subfolder within experiment based on the mode
        exp_subfd = os.path.join(self.save_dir, mode)
        _mkdirs_if_not_exist(exp_subfd)


        # ====== Create Loss Trackers ======
        _logger_make_csv_if_not_exist(f"{exp_subfd}/loss_tracker.csv", self.loss_headers)

        # ====== Create Metric Trackers if not training (training only sees data 1 batch at a time) =====
        if mode != "train":
                
            # Initialize Metric Trackers and CSV output file
            _multiclass_metric_headers = ["epoch"] + self.class_names
            _logger_make_csv_if_not_exist(f"{exp_subfd}/accuracy_tracker.csv", ["epoch", "accuracy"])
            _logger_make_csv_if_not_exist(f"{exp_subfd}/recall_tracker.csv", _multiclass_metric_headers)
            _logger_make_csv_if_not_exist(f"{exp_subfd}/precision_tracker.csv", _multiclass_metric_headers)
            _logger_make_csv_if_not_exist(f"{exp_subfd}/micro_f1_tracker.csv", _multiclass_metric_headers)
            _logger_make_csv_if_not_exist(f"{exp_subfd}/macro_f1_tracker.csv", _multiclass_metric_headers)
            _logger_make_csv_if_not_exist(f"{exp_subfd}/roc_auc_tracker.csv", _multiclass_metric_headers)
            _logger_make_csv_if_not_exist(f"{exp_subfd}/pr_auc_tracker.csv", _multiclass_metric_headers)

            # Initialize Confusion Matrix Saving
            _cm_header = ["epoch"] + ["Tr_{i}_Pr_{j}".format(i, j) for j in self.class_names for i in self.class_names]
            _logger_make_csv_if_not_exist(f"{exp_subfd}/confusion_matrix.csv", _cm_header)

            # Initialize Metric tracker for Clustering Performance
            _clustering_metric_headers = ["epoch", "ARI", "NMI", "Sil", "CAL", "DBI"]
            _logger_make_csv_if_not_exist(f"{exp_subfd}/clus_metric_tracker.csv", _clustering_metric_headers)


    def log_dirvrnn_losses(self, losses: List, epoch: int = 0, mode: str = "train"):
        """
        Log the losses by DIRVRNN model into save directory. 

        Args:
        - losses: List with loss information
        - epoch: current epoch for training. This parameter is disregarded if mode is set to 'test'.
        - mode: str indicating whether the logger is for testing or validation or training.
        """

        # Add epoch information to losses dictionary at the beginning
        losses_as_dict = {f"{mode}/{self.loss_headers[i]}": losses[i] for i in range(len(self.loss_headers))}

        # Log to Weights and Biases
        wandb.log(losses_as_dict, step=epoch + 1)


        # Save to CSV file
        with open(f"{self.save_dir}/{mode}/loss_tracker.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch] + losses)


def logger_sup_scores(y_true: Union[np.ndarray, torch.Tensor], y_pred: Union[np.ndarray, torch.Tensor], 
                    save_dir: str = "exps/DirVRNN", epoch: int = 1, class_names: List = []):
    """
    Log the Supervised Scores Performance into save directory.

    Args:
    - y_true: true labels of shape (bs, output_size)
    - y_pred: predicted labels of shape (bs, output_size)
    - save_dir: directory to save the scores (individual, and per class)
    - epoch: current epoch for training. For testing, it is set to 1.
    - class_names: list of class names for the outcome visualization and analysis
    """
    

def logger(model_params: Dict, X: torch.Tensor, y: torch.Tensor, log:Dict, epoch: int = 0, mode: str = "val", outcomes: List = [], features: List = [], save_dir: str = "exps/Dir_VRNN/"):
    """
    Logger for the model. 

    Params:
    - model_params: dictionary with model parameters.
    - X: unseen input data of shape (bs, T, input_size)
    - y: true outcome data of shape (bs, output_size)
    - Log_Dict: dictionary with loss and object information.
    - epoch: current epoch for training. This parameter is disregarded if mode is set to 'test'.
    - mode: str indicating whether the logger is for testing or validation.
    - outcomes: list of class names for the outcome visualization and analysis
    - features: list of feature names for the input data visualization and analysis
    - save_dir: directory to save the plots and metrics.

    Returns:
    - None, saves into wandb logger

    Heavier items (Bigger tables and/or images) are saved only when mode == "test". Otherwise, only basic information is kept for validation.
    """

    # Load model parameters
    c_means, log_c_vars = model_params["c_means"], model_params["log_c_vars"]
    seed = model_params["seed"]

    # Load objects
    temp_pis = log["pis"]
    
    # ================ LOGGING ================

    # Compute useful metrics and plots during training - including:
    # a) cluster assignment distribution, 
    # b) cluster means separability, 
    # c) accuracy, f1 and recall scores,
    # d) confusion matrix, 

    # Information about cluster separability (means and phenotypes)
    clus_means, clus_vars = c_means, torch.exp(log_c_vars)
    clus_mean_sep = model_utils.torch_clus_means_separability(clus_means=clus_means)

    # Log Cluster mean separability
    wandb.log({f"{mode}/clus-mean-separability": clus_mean_sep},
        step=epoch+1)


    # Needed for computing Performance Scores
    y_npy = y.detach().numpy()
    y_pred, y_pred_npy = log["y_preds"], log["y_preds"].detach().numpy()
    clus_prob = temp_pis.detach().numpy()
    X_npy = X.detach().numpy()
    class_names = [f"Class {i}" for i in range(y_pred.shape[-1])]

    # Compute Data Clustering Metrics
    label_metrics = LM_utils.get_clustering_label_metrics(y_true=y_npy, clus_pred=clus_prob)
    rand, nmi = label_metrics["rand"], label_metrics["nmi"]

    # Compute Unsupervised Clustering Metrics
    unsup_metrics = LM_utils.compute_unsupervised_metrics(X=X_npy, clus_pred=clus_prob, seed=seed)
    sil, dbi, vri = unsup_metrics["sil"], unsup_metrics["dbi"], unsup_metrics["vri"]

    # Log clustering metrics
    wandb.log({
        f"{mode}/rand-timeavg": np.mean(rand),
        f"{mode}/nmi-timeavg": np.mean(nmi),
        f"{mode}/sil-timeavg": np.mean(sil),
        f"{mode}/dbi-timeavg": np.mean(dbi),
        f"{mode}/vri-timeavg": np.mean(vri),
        f"{mode}/rand-final": rand[-1],
        f"{mode}/nmi-final": nmi[-1],
        f"{mode}/sil-final": sil[-1],
        f"{mode}/dbi-final": dbi[-1],
        f"{mode}/vri-final": vri[-1]
        },
        step=epoch+1
    )


    if mode == "test":       # Log everything else as well

        # Compute distribution over cluster memberships at this time step.
        clus_memb_dist = model_utils.torch_get_temp_clus_memb_dist(temp_pis_assign=temp_pis)
        fig, ax = model_utils.plot_clus_memb_evol(temp_clus_memb=clus_memb_dist)
    
        # Log
        wandb.log({f"{mode}/clus-memb-evolution": wandb.Image(fig)})
        plt.close()

    
        # Get Box Plots of cluster assignments and Log
        fig, ax = model_utils.torch_plot_clus_prob_assign_time(temp_pis_assign=temp_pis)
        wandb.log({f"{mode}/clus_assign_time_boxplot": wandb.Image(fig)})
        plt.close()


        # Log cluster means and variance
        _mean_df = pd.DataFrame(
            data=clus_means.detach().numpy(),
            index=[f"Cluster {i}" for i in range(clus_means.shape[0])],
            columns=[f"F{i}" for i in range(clus_means.shape[1])]
        )
        _vars_df = pd.DataFrame(
            data=clus_vars.detach().numpy().reshape(-1, 1),
            index=[f"Cluster {i}" for i in range(clus_vars.shape[0])],
            columns=["Variance"]
        )
        wandb.log({
                f"{mode}/cluster_means": 
                    wandb.Table(dataframe=_mean_df),
                f"{mode}/cluster_vars": 
                    wandb.Table(dataframe=_vars_df),
            }
        )
        
        # Log embeddings and other vectors            
        mug_samps, logvar_samps = log["mugs"], log["log_vargs"]


        # Get roc and prc
        ovr_roc = LM_utils.get_roc_auc_score(y_true=y_npy, y_pred=y_pred)
        prc_scores = LM_utils.get_torch_pr_auc_score(y_true=y, y_pred=y_pred).detach().numpy()

        # Compute Confusion Matrix
        confusion_matrix = LM_utils.get_confusion_matrix(y_true=y_npy, y_pred=y_pred_npy)

        # Compute confusion matrix, ROC and PR curves
        roc_fig, roc_ax = LM_utils.get_roc_curve(y_true=y_npy, y_pred=y_pred_npy, class_names=class_names) 
        pr_fig, pr_ax = LM_utils.get_torch_pr_curve(y_true=y, y_pred=y_pred, class_names=class_names)



        class_outputs = pd.DataFrame(
            data=np.vstack((ovr_roc, prc_scores)),
            index=["ROC-OVO", "PRC"],
            columns=class_names
        )


        # Log Objects
        wandb.log({
                f"{mode}/confusion_matrix":
                    wandb.Table(data=confusion_matrix, columns=[f"True {_name}" for _name in class_names], 
                                rows=[f"Pred {_name}" for _name in class_names]),
                f"{mode}/class_outputs": wandb.Table(dataframe=class_outputs),
                f"{mode}/roc_curve": wandb.Image(roc_fig),
                f"{mode}/pr_curve": wandb.Image(pr_fig)
            }
        )

        # Print and Log Phenotype Information
        prob_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="prob")
        onehot_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="one-hot")

        # Obtain plots
        prob_fig_1, _ = model_utils.torch_line_plot_phenotypes_per_outcome(phens=prob_phens, class_names=class_names)
        prob_fig_2, _ = model_utils.torch_line_plot_phenotypes_per_cluster(phens=prob_phens, class_names=class_names)
        onehot_fig_1, _ = model_utils.torch_line_plot_phenotypes_per_outcome(phens=onehot_phens, class_names=class_names)
        onehot_fig_2, _ = model_utils.torch_line_plot_phenotypes_per_cluster(phens=onehot_phens, class_names=class_names)

        # Log Phenotype Information over time
        K, T, O = prob_phens.shape
        _prob_phens_npy = prob_phens.detach().numpy()
        _onehot_phens_npy = onehot_phens.detach().numpy()

        for t in range(T):
            wandb.log({
                f"{mode}/Phens_Prob_{t}": wandb.Table(
                    dataframe= pd.DataFrame(
                                _prob_phens_npy[:, t, :],
                                index=[f"C{i}" for i in range(K)],
                                columns=class_names
                            )
                    ),
                f"{mode}/Phens_OH_{t}": wandb.Table(
                    dataframe= pd.DataFrame(
                                _onehot_phens_npy[:, t, :],
                                index=[f"C{i}" for i in range(K)],
                                columns=class_names
                        )
                    )   
                }
            )
        wandb.log({
            f"{mode}/Phens_Prob_per_outc": wandb.Image(prob_fig_1),
            f"{mode}/Phens_Prob_per_clus": wandb.Image(prob_fig_2),
            f"{mode}/Phens_OH_per_outc": wandb.Image(onehot_fig_1),
            f"{mode}/Phens_OH_per_clus": wandb.Image(onehot_fig_2)
            }
        )


        # Generate Data Samples and Compare with True Occurrence
        gen_samples = model_utils.gen_diagonal_mvn(mug_samps, logvar_samps).detach().numpy()
        gen_samples = gen_samples[:, - X_npy.shape[1]: , :]             # Subset generated samples to match X_npy time steps

        # Get time ids for the generated data and plot
        time_ids = np.array(range(1, X_npy.shape[1]+1))[::-1] 
        pat_plots = model_utils.plot_samples(X_npy, gen_samples, num_samples=10, feat_names=feat_names, time_idxs=time_ids)

        # Log to Wandb
        for pat_id, (fig, ax) in pat_plots.items():
            wandb.log({
                f"{mode}/pat_{pat_id}": wandb.Image(fig)
            }
        )


        # Save outputs
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        with open(f"{save_dir}/outputs.pkl", "wb") as f:
            pickle.dump(log, f)

# endregion
