"""

Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

This file implements a logger function to plot and log useful information about the model during training, validation and testing.

"""

# region =============== IMPORT LIBRARIES ===============
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
def logger(model_params: Dict, X: torch.Tensor, y: torch.Tensor, log:Dict, epoch: int = 0, mode: str = "val", class_names: List = []):
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
    wandb.log({f"{mode}/clus-memb-evolution": fig}, step=epoch+1)
    plt.close()

    
    # Get Box Plots of cluster assignments and Log
    fig, ax = model_utils.torch_plot_clus_prob_assign_time(temp_pis_assign=temp_pis)
    wandb.log({f"{mode}/clus_assign_time_boxplot": fig}, step=epoch+1)
    plt.close()


    # Information about cluster separability (means and phenotypes)
    clus_means, clus_vars = c_means, torch.exp(log_c_vars)
    clus_mean_sep = model_utils.torch_clus_means_separability(clus_means=clus_means)

    # Log
    wandb.log({
            f"{mode}/cluster_means": 
                wandb.Table(
                    data=clus_means.detach().numpy(),
                ),
            f"{mode}/cluster_vars": 
                wandb.Table(
                    data=clus_vars.detach().numpy(),
                ),
            f"{mode}/clus_mean_sep": clus_mean_sep.detach().numpy(),
        },
        step=epoch+1
    )
    
    # Log embeddings and other vectors            
    alpha_samples, pi_samples = log["alpha_encs"], log["pis"]
    mug_samps, logvar_samps = log["mugs"], log["log_vargs"]
    z_samples = log["zs"]

    wandb.log({
            f"{mode}/z_samples": 
                wandb.Table(
                    data=z_samples.detach().numpy()
                ),
            f"{mode}/alpha_samples": 
                wandb.Table(
                    data=alpha_samples.detach().numpy()
                ),
            f"{mode}/pi_samples":
                wandb.Table(
                    data=pi_samples.detach().numpy()
                ),
            f"{mode}/mug_samples":
                wandb.Table(
                    data=mug_samps.detach().numpy()
                ),
            f"{mode}/logvar_samples":
                wandb.Table(
                    data=logvar_samps.detach().numpy()
                )
        },
        step=epoch+1
    )


    # Log Performance Scores
    y_npy = y.detach().numpy()
    y_pred = log["y_preds"].detach().numpy()
    clus_prob = temp_pis.detach().numpy()
    X_npy = X.detach().numpy()
    class_names = [f"Class {i}" for i in range(y_pred.shape[-1])]

    # Compute accuracy, f1 and recall scores
    acc = LM_utils.accuracy_score(y_true=y_npy, y_pred=y_pred)
    
    macro_f1 = LM_utils.macro_f1_score(y_true=y_npy, y_pred=y_pred)
    micro_f1 = LM_utils.micro_f1_score(y_true=y_npy, y_pred=y_pred)
    recall = LM_utils.recall_score(y_true=y_npy, y_pred=y_pred)
    precision = LM_utils.precision_score(y_true=y_npy, y_pred=y_pred)

    # Get roc and prc
    roc_scores = LM_utils.get_roc_auc_score(y_true=y_npy, y_pred=y_pred)
    prc_scores = LM_utils.get_pr_auc_score(y_true=y_npy, y_pred=y_pred)

    # Compute Confusion Matrix
    confusion_matrix = LM_utils.get_confusion_matrix(y_true=y_npy, y_pred=y_pred)

    # Compute confusion matrix, ROC and PR curves
    roc_fig, roc_ax = LM_utils.get_roc_curve(y_true=y_npy, y_pred=y_pred, class_names=class_names) 
    pr_fig, pr_ax = LM_utils.get_pr_curve(y_true=y_npy, y_pred=y_pred, class_names=class_names)

    # Compute Data Clustering Metrics
    label_metrics = LM_utils.get_clustering_label_metrics(y_true=y_npy, clus_pred=clus_prob)
    rand, nmi = label_metrics["rand"], label_metrics["nmi"]

    # Compute Unsupervised Clustering Metrics
    unsup_metrics = LM_utils.compute_unsupervised_metrics(X=X_npy, clus_pred=clus_prob, seed=seed)
    sil, dbi, vri = unsup_metrics["sil"], unsup_metrics["dbi"], unsup_metrics["vri"]

    # Combine scores to single table
    single_output = pd.Series(
        data = [acc, macro_f1, micro_f1, recall, precision],
        index=["Accuracy", "Macro F1", "Micro F1", "Recall", "Precision"]
    )

    class_outputs = pd.DataFrame(
        data=np.vstack((roc_scores, prc_scores)),
        index=["ROC", "PRC"],
        columns=class_names
    )
    
    temp_outputs = pd.DataFrame(
        data=np.vstack((rand, nmi, sil, dbi, vri)),
        index=["Rand", "NMI", "Silhouette", "DBI", "VRI"],
        columns=list(range(X_npy.shape[1]))
    )

    # Log Objects
    wandb.log({
        f"{mode}/confusion_matrix":
            wandb.Table(data=confusion_matrix, columns=[f"True {_name}" for _name in class_names], 
                        rows=[f"Pred {_name}" for _name in class_names]),
        f"{mode}/single_output": wandb.Table(data=single_output),
        f"{mode}/class_outputs": wandb.Table(data=class_outputs),
        f"{mode}/temp_outputs": wandb.Table(data=temp_outputs),
        f"{mode}/roc_curve": roc_fig,
        f"{mode}/pr_curve": pr_fig
    },
    step=epoch+1
)


    # Print and Log Phenotype Information
    prob_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="prob")
    onehot_phens = model_utils.torch_get_temp_phens(pis_assign=temp_pis, y_true=y, mode="one-hot")

    # Log Phenotype Information
    wandb.log({
        f"{mode}/prob_phens": wandb.Table(data=prob_phens.detach().numpy()),
        f"{mode}/onehot_phens": wandb.Table(data=onehot_phens.detach().numpy())
        },
        step=epoch+1
    )




    """

    MISSING IMPLEMENTATION:
    - Get Class Names
    - Get Feature Names
    - Better Sampling Visualization

    """


    # Generate Data Samples and Compare with True Occurrence
    gen_samples = model_utils.gen_diagonal_mvn(mug_samps, logvar_samps).detach().numpy()

    # Sample 10 random patients and plot to Wandb
    random_pats_10 = torch.randint(low=0, high=X.shape[0], size=(10,))

    for _pat_id in random_pats_10:

        # Select true data and generated data
        _x_pat = X_npy[_pat_id, :, :]
        _x_gen = gen_samples[_pat_id, :, :]

        # Plot to Wandb
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(_x_pat, label="True")
        ax.plot(_x_gen, label="Generated")
        wandb.log({
            "test/{}-pat".format(_pat_id): wandb.Image(fig)
        })

# endregion
