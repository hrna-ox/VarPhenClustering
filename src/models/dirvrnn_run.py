"""
File to train dirvrnn model.

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""

# ============= Import Libraries =============

import json
import os

from src.models.DirVRNN import DirVRNN as Model
from src.data_processing.DataLoader import DataLoader 

def main():

    # ---------------------------- Load Configurations --------------------------------------
    with open("src/models/run_config.json", "r") as f:
        run_config = json.load(f)
        f.close()

    # Extract info
    data_config = run_config["data_config"]
    model_config = run_config["model_config"]
    training_config = run_config["training_config"]

    # ---------------------------- Load and Process Data --------------------------------------
    data_info = DataLoader(**data_config)


    # -------------------------- Loading and Training Model -----------------------------

    model = Model(**run_config)
    model.fit(train_data=, val_data=, lr=, batch_size, num_epochs=, save_params=)

    # Evaluate model
    output_dir = model.predict(X=, y=, save_params=)


    #
    # "Compute results on test data"
    # outputs_dic = model.analyse(data_info)
    # print(outputs_dic.keys())

    # -------------------------------------- Evaluate Scores --------------------------------------

    # "Evaluate scores on the resulting models. Note X_test is converted back to input dimensions."
    # scores = evaluate(**outputs_dic, data_info=data_info, avg=None)


    print("Analysis Complete.")


if __name__ == "__main__":
    main()
