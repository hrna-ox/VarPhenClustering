"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file executes experiments on the data for traditional classifiers.
"""

# Import required packages
import json
import numpy as np

from src.models.TSKM.model import TSKM
from src.dataloading.TrainingDataLoader import TrainingDataLoader

from src.metrics.summarize import summary_all_metrics
from src.visualization.metrics_and_losses import plot_multiclass_metrics

from datetime import datetime



def main():

    start_time = datetime.now()

    #### LOAD CONFIGURATIONS
    with open("src/models/TSKM/run_config.json", "r") as f:
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
    model = TSKM(input_shape=input_shape, 
                output_dim=output_dim, 
                model_name=args["model_name"], 
                K=args["model_params"]["K"], 
                random_state=args["model_params"]["random_state"]
                )

    # Get whole training data and validation data
    X_train, y_train = data_loader.get_train_X_y(fold=0)
    model.train(train_data=(X_train, y_train))


    #### EVALUATE MODEL
    # Evaluate on test data
    X_test, y_test = data_loader.get_test_X_y(fold=0, mode="test")
    y_pred, clus_pred = model.predict(X_test)
    
    # Convert to Labels
    labels_test = np.argmax(y_test, axis=1)

    # Compute Metrics and visualize
    metrics_dict = summary_all_metrics(
        labels_true=labels_test, scores_pred=y_pred,
        X=X_test.reshape(X_test.shape[0], -1), clus_pred=clus_pred)
    ax, lachiche_ax = plot_multiclass_metrics(metrics_dict=metrics_dict, class_names=data_characteristics["outcomes"])

    # Log Model, Results and Visualizations
    cur_time_as_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_dir = f"results/{args['model_name']}/{cur_time_as_str}/"

    means, cluster_probs = model.get_model_objects()
    run_info = {
        "data_characteristics": data_characteristics,
        "args": args,
        "metrics": metrics_dict
    }
    output_dir = {
        "data": {
            "X": (X_train, X_test),
            "y": (y_train, y_test)
        },
        "model": {
            "means": means,
            "cluster_probs": cluster_probs
        },
        "labels_test": labels_test,
        "y_pred": y_pred,
        "clus_pred": clus_pred,
    }
    
    model.log_model(save_dir=test_dir, objects_to_log=output_dir, run_info=run_info)
    print("Time taken: ", datetime.now() - start_time)


if __name__ == "__main__":
    main()
