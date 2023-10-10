"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file executes experiments on the data for traditional classifiers.
"""

# Import required packages
import json
import numpy as np

from src.models.XGB.model import GlobalClassifier, ParallelClassifier
from src.dataloading.TrainingDataLoader import TrainingDataLoader

from src.metrics.summarize import summary_multiclass_metrics, print_avg_metrics_paper
import src.logging.logger_utils as logger_utils


from datetime import datetime

def main():

    start_time = datetime.now()

    #### LOAD CONFIGURATIONS
    with open("src/models/SVM/run_config.json", "r") as f:
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
    if not args["feature_parallelize"]:
        model = GlobalClassifier(input_shape=input_shape, output_dim=output_dim, model_name=args["model_name"], **args["model_params"])
    else:
        model = ParallelClassifier(input_shape=input_shape, output_dim=output_dim, model_name=args["model_name"], **args["model_params"])


    # Get whole training data and validation data
    X_train, y_train = data_loader.get_train_X_y(fold=0)
    model.train(train_data=(X_train, y_train))


    #### EVALUATE MODEL
    # Evaluate on test data
    X_test, y_test = data_loader.get_test_X_y(fold=0, mode="test")
    y_pred = model.predict(X_test)
    
    # Convert to Labels
    labels_test = np.argmax(y_test, axis=1)

    # Compute Metrics and visualize
    metrics_dict = summary_multiclass_metrics(labels_true=labels_test, scores_pred=y_pred)

    # Log Model, Results and Visualizations
    cur_time_as_str = datetime.now().strftime("%Y%m%d-%H%M%S")

    if args["feature_parallelize"]:
        test_dir = f"results/XGB_parallel/{cur_time_as_str}/"
    else:
        test_dir = f"results/XGB_Global/{cur_time_as_str}/"

    # Save outputs into data objects and run information
    data_objects = {
        "y_pred": y_pred,
        "labels_test": labels_test,
        "X": (X_train, X_test),
        "y": (y_train, y_test),
    }

    run_info = {
        "data_characteristics": data_characteristics,
        "args": args,
        "metrics": metrics_dict,
    }
    
    model.log_model(save_dir=test_dir, objects_to_log=data_objects, run_info=run_info)
    print("Time taken: ", datetime.now() - start_time)

    
    # ===================== CSV LOGGING
    csv_path = "results/XGB_Global/tracker.csv" if not args["feature_parallelize"] else "results/XGB_parallel/tracker.csv"
    header = ["data_name"] + [key for key in args.keys() if key not in ["data_name", "model_params"]] + \
        ["F1", "Precision", "Recall", "Auroc", "SIL", "DBI", "VRI"]
    logger_utils.make_csv_if_not_exists(csv_path, header)

    # Add row
    data_name = "MIMIC" if "mimic" in args["data_dir"].lower() else "HAVEN"
    paper_metrics = print_avg_metrics_paper(metrics_dict)
    row_append = data_name, *[item for key, item in args.items() if key not in ["data_name", "model_params"]], *paper_metrics
    logger_utils.write_csv_row(csv_path, row_append)

if __name__ == "__main__":
    main()
