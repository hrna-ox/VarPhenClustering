"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file executes experiments on the data for traditional classifiers.
"""

# Import required packages
import json
import numpy as np

from src.models.ESI.model import ESI
from src.dataloading.TrainingDataLoader import TrainingDataLoader

from src.metrics.summarize import summary_binary_metrics, print_avg_metrics_paper
import src.logging.logger_utils as logger_utils

from datetime import datetime



# ============= MAIN COMPUTATION  ==================
def main():
    start_time = datetime.now()


    #### LOAD CONFIGURATIONS
    with open("src/models/ESI/run_config.json", "r") as f:
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
    ESI_idx = data_characteristics["features"].index("ESI") 


    #### TRAIN MODEL
    model = ESI(input_shape=input_shape, output_dim=output_dim, ESI_idx=ESI_idx)

    # This does nothing for ESI
    X_train, y_train = data_loader.get_train_X_y(fold=0)
    model.train(train_data=(X_train, y_train))


    #### EVALUATE MODEL
    
    # Performance on Test data
    X_test, y_test = data_loader.get_test_X_y(fold=0, mode="test")
    y_pred = model.predict(X_test, y=y_test)
    y_pred = y_pred / np.max(y_pred)
        
    # Prepare labels and output dic
    output_dic = {}
    labels_test = np.argmax(y_test, axis=1)

    # Compute Metrics and visualize
    for out_idx in range(output_dim):

        # Iterate over each outcome
        has_outcome = labels_test == out_idx
        pred_score = y_pred

        # Compute scores
        output_dic[out_idx] = summary_binary_metrics(labels_true=has_outcome, scores_pred=pred_score)

    # Aggreggate all metrics over the keys in output_dic
    metric_keys = output_dic[0].keys()
    metrics_dict = {
        key: [] for key in metric_keys
    }
    for out_idx in range(output_dim):
        for key in metric_keys:
            metrics_dict[key].append(output_dic[out_idx][key])

    # Convert for compatibility
    metrics_dict["macro_f1_score"] = metrics_dict["f1_score"]
    metrics_dict["ovr_auroc"] = metrics_dict["auroc"]


    # Log Model, Results and Visualizations
    cur_time_as_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_dir = f"results/ESI/{cur_time_as_str}/"

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
    csv_path = "results/ESI/tracker.csv"

    params_header = [key for key in args.keys() if key not in ["model_params"]]
    metrics_header = ["F1", "Precision", "Recall", "Auroc", "SIL", "DBI", "VRI"]
    logger_utils.make_csv_if_not_exists(csv_path, params_header + metrics_header)

    # Append Row
    metrics_to_print = print_avg_metrics_paper(metrics_dict)
    row_append = *[args[key] for key in params_header], *metrics_to_print
    logger_utils.write_csv_row(csv_path, row_append)

if __name__ == "__main__":
    main()
