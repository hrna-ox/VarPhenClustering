"""
File to train dirvrnn model.

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""

# ============= Import Libraries =============

import json
from datetime import datetime
import wandb
import numpy as np

from src.models.DIRVRNN.model import DirVRNN
from src.dataloading.TrainingDataLoader import TrainingDataLoader

from src.metrics.summarize import summary_all_metrics
from src.visualization.metrics_and_losses import plot_multiclass_metrics

def main():

    # ---------------------------- Load Configurations --------------------------------------
    with open("src/models/DIRVRNN/run_config.json", "r") as f:
        args = json.load(f)
        f.close()
    
    print("Loaded Configuration Successfully")

    start_time = datetime.now()

    #### LOAD DATA
    data_loader = TrainingDataLoader(
        data_dir=args["data_dir"],
        time_window=args["time_window"],
        feat_subset=args["feat_subset"],
        train_test_ratio=args["train_test_ratio"],
        train_val_ratio=args["train_val_ratio"],
        seed=args["seed"],
        normalize=args["normalize"],
        num_folds=args["num_folds"]
    )
    data_characteristics = data_loader._get_data_characteristics()

    # Unpack
    input_shape = data_characteristics["num_samples"], data_characteristics["num_timestamps"], data_characteristics["num_features"]
    output_dim = data_characteristics["num_outcomes"]


    # Prepare wandb
    wandb.init(project="DirVRNN", config=args)

    # ---------------------------- Load and Process Data --------------------------------------
    print("\nLoading and Processing Data...")

    model = DirVRNN(input_dims = input_shape[-1], 
                    num_classes = output_dim,
                    window_num_obvs=args["model_params"]["window_num_obvs"],
                    K=args["model_params"]["K"],
                    latent_dim=args["model_params"]["latent_dim"],
                    n_fwd_blocks=args["model_params"]["n_fwd_blocks"],
                    seed=args["seed"],
                    dropout=args["model_params"]["dropout"]
                )
    
    ## ACCESS Train Data
    X_train, y_train = data_loader.get_train_X_y(fold=0)
    X_val, y_val = data_loader.get_test_X_y(mode="val", fold=0)

    # Fit
    model.fit(
        train_data=(X_train, y_train),
        val_data=(X_val, y_val),
        batch_size=args["training_params"]["batch_size"],
        lr=args["training_params"]["lr"],
        num_epochs=args["training_params"]["num_epochs"]
    )

    #### EVALUATE MODEL
    # Evaluate on test data
    X_test, y_test = data_loader.get_test_X_y(fold=0, mode="test")
    output = model.predict(X=X_test, y=y_test)
    y_pred = output["outputs_future"]["y_pred"].to_numpy()
    clus_pred = np.argmax(output["outputs_future"]["pis"].to_numpy()[:, -1, :], axis=-1)
    
    # Convert to Labels
    labels_test = np.argmax(y_test, axis=1)

    # Compute Metrics and visualize
    metrics_dict = summary_all_metrics(
        labels_true=labels_test, scores_pred=y_pred,
        X=X_test.reshape(X_test.shape[0], -1), clus_pred=clus_pred
    )
    ax, lachiche_ax = plot_multiclass_metrics(metrics_dict=metrics_dict, class_names=data_characteristics["outcomes"])

    # Log Model, Results and Visualizations
    cur_time_as_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_dir = f"results/{args['model_name']}/{cur_time_as_str}/"

    run_info = {
        "data_characteristics": data_characteristics,
        "args": args,
        "metrics": metrics_dict,
    }
    objects_to_log = {
        "data": {
            "X": (X_train, X_test),
            "y": (y_train, y_test),
        },
        "labels_test": labels_test,
        "y_pred": y_pred
    }
    model.log_model(save_dir=test_dir, objects_to_log=objects_to_log, run_info=run_info)
    
    print("Time taken: ", datetime.now() - start_time)
    print("Analysis Complete.")


if __name__ == "__main__":
    main()