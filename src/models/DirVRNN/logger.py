"""

Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file implements a logger function to plot and log useful information about the model during training, validation and testing.

"""

# region =============== IMPORT LIBRARIES ===============
import os
import pickle
import numpy as np
import pandas as pd
import torch

from typing import Dict, List

import matplotlib.pyplot as plt
import wandb

import src.models.DirVRNN.auxiliary_functions as model_utils
import src.models.losses_and_metrics as LM_utils

# endregion

# region =============== MAIN ===============
def logger(model_params: Dict, X: torch.Tensor, y: torch.Tensor, log:Dict, epoch: int = 0, mode: str = "val", class_names: List = [], feat_names: List = [], save_dir: str = "exps/Dir_VRNN/"):
    """
    Logger for the model. 

    Params:
    - model_params: dictionary with model parameters.
    - X: unseen input data of shape (bs, T, input_size)
    - y: true outcome data of shape (bs, output_size)
    - Log_Dict: dictionary with loss and object information.
    - epoch: current epoch for training. This parameter is disregarded if mode is set to 'test'.
    - mode: str indicating whether the logger is for testing or validation.
    - class_names: list of class names for the outcome visualization and analysis
    - feat_names: list of feature names for the input data visualization and analysis
    - save_dir: directory to save the plots and metrics.

    Returns:
    - None, saves into wandb logger
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

    # Compute distribution over cluster memberships at this time step.
    clus_memb_dist = model_utils.torch_get_temp_clus_memb_dist(temp_pis_assign=temp_pis)
    fig, ax = model_utils.plot_clus_memb_evol(temp_clus_memb=clus_memb_dist)
    
    # Log
    wandb.log({f"{mode}/clus-memb-evolution": wandb.Image(fig)}, step=epoch+1)
    plt.close()

    
    # Get Box Plots of cluster assignments and Log
    fig, ax = model_utils.torch_plot_clus_prob_assign_time(temp_pis_assign=temp_pis)
    wandb.log({f"{mode}/clus_assign_time_boxplot": wandb.Image(fig)}, step=epoch+1)
    plt.close()


    # Information about cluster separability (means and phenotypes)
    clus_means, clus_vars = c_means, torch.exp(log_c_vars)
    clus_mean_sep = model_utils.torch_clus_means_separability(clus_means=clus_means)

    # Log
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
            f"{mode}/clus_mean_sep": clus_mean_sep.detach().numpy(),
        },
        step=epoch+1
    )
    
    # Log embeddings and other vectors            
    mug_samps, logvar_samps = log["mugs"], log["log_vargs"]


    # Log Performance Scores
    y_npy = y.detach().numpy()
    y_pred, y_pred_npy = log["y_preds"], log["y_preds"].detach().numpy()
    clus_prob = temp_pis.detach().numpy()
    X_npy = X.detach().numpy()
    class_names = [f"Class {i}" for i in range(y_pred.shape[-1])]

    # Compute accuracy, f1 and recall scores
    acc = LM_utils.accuracy_score(y_true=y_npy, y_pred=y_pred_npy)
    
    macro_f1 = LM_utils.macro_f1_score(y_true=y_npy, y_pred=y_pred_npy)
    micro_f1 = LM_utils.micro_f1_score(y_true=y_npy, y_pred=y_pred_npy)
    recall = LM_utils.recall_score(y_true=y_npy, y_pred=y_pred_npy)
    precision = LM_utils.precision_score(y_true=y_npy, y_pred=y_pred_npy)

    # Get roc and prc
    ovr_roc = LM_utils.get_roc_auc_score(y_true=y_npy, y_pred=y_pred)
    prc_scores = LM_utils.get_torch_pr_auc_score(y_true=y, y_pred=y_pred).detach().numpy()

    # Compute Confusion Matrix
    confusion_matrix = LM_utils.get_confusion_matrix(y_true=y_npy, y_pred=y_pred_npy)

    # Compute confusion matrix, ROC and PR curves
    roc_fig, roc_ax = LM_utils.get_roc_curve(y_true=y_npy, y_pred=y_pred_npy, class_names=class_names) 
    pr_fig, pr_ax = LM_utils.get_torch_pr_curve(y_true=y, y_pred=y_pred, class_names=class_names)

    # Compute Data Clustering Metrics
    label_metrics = LM_utils.get_clustering_label_metrics(y_true=y_npy, clus_pred=clus_prob)
    rand, nmi = label_metrics["rand"], label_metrics["nmi"]

    # Compute Unsupervised Clustering Metrics
    unsup_metrics = LM_utils.compute_unsupervised_metrics(X=X_npy, clus_pred=clus_prob, seed=seed)
    sil, dbi, vri = unsup_metrics["sil"], unsup_metrics["dbi"], unsup_metrics["vri"]

    # Combine scores to single table
    single_output = pd.DataFrame(
        data = [acc, macro_f1, micro_f1, recall, precision],
        index=["Accuracy", "Macro F1", "Micro F1", "Recall", "Precision"],
        columns=["Scores"]
    )

    class_outputs = pd.DataFrame(
        data=np.vstack((ovr_roc, prc_scores)),
        index=["ROC-OVO", "PRC"],
        columns=class_names
    )

    temp_outputs = pd.DataFrame(
        data=np.vstack((rand, nmi, sil, dbi, vri)),
        index=["Rand", "NMI", "Silhouette", "DBI", "VRI"],
        columns=list(range(np.size(rand))))

    # Log Objects
    wandb.log({
        f"{mode}/confusion_matrix":
            wandb.Table(data=confusion_matrix, columns=[f"True {_name}" for _name in class_names], 
                        rows=[f"Pred {_name}" for _name in class_names]),
        f"{mode}/single_output": wandb.Table(dataframe=single_output),
        f"{mode}/class_outputs": wandb.Table(dataframe=class_outputs),
        f"{mode}/temp_outputs": wandb.Table(dataframe=temp_outputs),
        f"{mode}/roc_curve": wandb.Image(roc_fig),
        f"{mode}/pr_curve": wandb.Image(pr_fig)
    },
    step=epoch+1
)


    # Print and Log Phenotype Information
    prob_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="prob")
    onehot_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="one-hot")

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
            },
            step=epoch+1
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
        },
        step=epoch+1
    )


    """

    MISSING IMPLEMENTATION:
    - Better Phenotype Visualization

    """

    # Save outputs
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    with open(f"{save_dir}/outputs.pkl", "wb") as f:
        pickle.dump(log, f)

# endregion
