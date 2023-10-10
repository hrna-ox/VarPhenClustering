"""
Author: Henrique Aguiar
Contact: henrique.aguiar at eng.ox.ac.uk

This file executes experiments on the data for traditional classifiers.
"""

# Import required packages
import json
import numpy as np
import hyperopt
from typing import Tuple, Dict

from src.models.TraditionalClassifiers.model import BaseClassifier, ParallelClassifier
from src.dataloading.TrainingDataLoader import TrainingDataLoader

from src.metrics.multiclass_metrics import multiclass_macro_f1_score

def main():

    #### LOAD CONFIGURATIONS
    with open("src/models/TraditionalClassifiers/run_config.json", "r") as f:
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
    data_config = data_loader._get_config()
    data_characteristics = data_loader._get_data_characteristics()

    # Unpack
    input_shape = data_characteristics["num_samples"], data_characteristics["num_timestamps"], data_characteristics["num_features"]
    output_dim = data_characteristics["num_outcomes"]

    #### DEFINE HYPERPARAMETER TUNING OBJECTIVE FUNCTION
    def hyperparameter_loss(params: Dict):
        """
        Function that implements and returns an output for each parameter choice and determines how best to optimize the model.
        """

        # Unpack
        num_folds, training_data = data_loader._get_num_folds(), data_loader._get_training_data()
        total_f1 = 0

        # Iterate over folds
        for fold_idx in range(1, num_folds + 1):
            
            # Get data
            X_tr, y_tr = data_loader.get_train_X_y(fold=fold_idx)
            X_val, y_val = data_loader.get_test_X_y(fold=fold_idx, mode="val")

            # Train model
            if args["feature_parallelize"]:
                model = BaseClassifier(input_shape=input_shape, output_dim=output_dim, **params)
            else:
                model = ParallelClassifier(input_shape=input_shape, output_dim=output_dim, **params)

            model.train(train_data=(X_tr, y_tr))

            # Evaluate on validation data
            y_pred = model.predict(X_val)
            labels_pred = np.argmax(y_pred, axis=1)

            if np.unique(labels_pred) < y_pred.shape[1]:          # There is a class that was not predicted
                f1 = 0

            else:
                # Compute Optimizer Metric
                f1 = np.mean(multiclass_macro_f1_score(labels_true=y_val, labels_pred=labels_pred))

            # Update total f1
            total_f1 += f1
        
        avg_f1 = total_f1 / num_folds

        return {
            "loss": -avg_f1,
            "status": hyperopt.STATUS_OK,
        }
            
    
    ################## OPTIMIZE OVER CONFIGURATION POSSIBILITIES ##################

    # Define search space
    space = {}

    # Initialize Trials
    trials = hyperopt.Trials()

    best = hyperopt.fmin(
        fn=hyperparameter_loss,
        space=space,
        algo=hyperopt.tpe.suggest,
        max_evals=args["num_evals"],
        trials=trials
    )

    ################## TRAIN AND EVALUATE BEST MODEL ##################
    best = {}

    #### Train model
    if args["feature_parallelize"]:
        model = BaseClassifier(input_shape=input_shape, output_dim=output_dim, **best)
    
    else:
        model = ParallelClassifier(input_shape=input_shape, output_dim=output_dim, **best)

    # Get whole training data and validation data
    X_train, y_train = data_loader.get_train_X_y(fold=0)
    model.train(train_data=(X_train, y_train))

    # Evaluate on test data
    X_test, y_test = data_loader.get_test_X_y(fold=0, mode="test")
    y_pred = model.predict(X_test)

    # Log Model and Results

    model.log_model(save_dir=test_dir)
    data_loader.log_data(save_dir=test_dir, X=X_test, y_true=y_test, y_pred=y_pred)