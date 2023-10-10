"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file executes experiments on the data for traditional classifiers.
"""

# Import required packages
import json
import numpy as np

from src.models.VRNNGMM.model import VRNNGMM
from src.dataloading.TrainingDataLoader import TrainingDataLoader

from src.metrics.summarize import summary_multiclass_metrics, summary_clustering_metrics, print_avg_metrics_paper
import src.logging.logger_utils as logger_utils

from datetime import datetime



def main():

    start_time = datetime.now()

    #### LOAD CONFIGURATIONS
    with open("src/models/VRNNGMM/run_config.json", "r") as f:
        args = json.load(f)

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


    #### TRAIN MODEL
    
    # Initialize model
    model = VRNNGMM(
        input_dims=input_shape[-1],
        output_dim=output_dim,
        K=args["model_params"]["K"],
        latent_dim=args["model_params"]["latent_dim"],
        gate_num_hidden_layers=args["model_params"]["gate_num_hidden_layers"],
        gate_num_hidden_nodes=args["model_params"]["gate_num_hidden_nodes"],
        bias=args["model_params"]["bias"],
        dropout=args["model_params"]["dropout"],
        device=args["model_params"]["device"],
        seed=args["model_params"]["seed"],
        K_fold_idx=args["model_params"]["K_fold_idx"],
    )

    # Get whole training data and validation data
    X_train, y_train = data_loader.get_train_X_y(fold=1)
    X_val, y_val = data_loader.get_test_X_y(fold=1, mode="val")
    model.fit(
        train_data=(X_train.astype(np.float32), y_train.astype(np.float32)),
        val_data=(X_val.astype(np.float32), y_val.astype(np.float32)),
        lr=args["train_params"]["lr"],
        batch_size=args["train_params"]["batch_size"],
        num_epochs=args["train_params"]["num_epochs"]
    )


    #### EVALUATE MODEL
    # Evaluate on test data
    X_test, y_test = data_loader.get_test_X_y(fold=1, mode="test")
    y_pred, clus_pred, model_objects = model.predict(X=X_test.astype(np.float32), y=y_test.astype(np.float32))
    
    # Convert to Labels
    labels_test = np.argmax(y_test, axis=1)

    # Compute Metrics and visualize
    multiclass_dic = summary_multiclass_metrics(labels_true=labels_test, scores_pred=y_pred)  # type: ignore
    clustering_dic = summary_clustering_metrics(
        X=X_test.reshape(X_test.shape[0], -1),
        labels_true=labels_test,
        clus_pred=clus_pred                     # type: ignore 
    )
    metrics_dict = {**multiclass_dic, **clustering_dic}

    # Log Model, Results and Visualizations
    cur_time_as_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_dir = f"results/VRNNGMM/{cur_time_as_str}/"

    # Save outputs into data objects and run information
    data_objects = {
        "y_pred": y_pred,
        "labels_test": labels_test,
        "X": (X_train, X_val, X_test),
        "y": (y_train, y_val, y_test),
        "test_output": model_objects
    }

    run_info = {
        "data_characteristics": data_characteristics,
        "args": args,
        "metrics": metrics_dict,
    }
    
    model.log_model(save_dir=test_dir, objects_to_log=data_objects, run_info=run_info)
    print("Time taken: ", datetime.now() - start_time)

    
    # ===================== CSV LOGGING
    csv_path = "results/VRNNGMM/tracker.csv" 
    
    params_header = [key for key in args.keys() if key not in ["model_params", "train_params"]]
    metrics_header = ["F1", "Precision", "Recall", "Auroc", "SIL", "DBI", "VRI"]
    logger_utils.make_csv_if_not_exists(csv_path, params_header + metrics_header)

    # Append Row
    metrics_to_print = print_avg_metrics_paper(metrics_dict)
    row_append = *[args[key] for key in params_header], *metrics_to_print
    logger_utils.write_csv_row(csv_path, row_append)


if __name__ == "__main__":
    main()
