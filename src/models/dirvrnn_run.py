"""
File to train dirvrnn model.

Author: Henrique Aguiar
Please contact via henrique.aguiar@eng.ox.ac.uk
"""

# ============= Import Libraries =============

import json
import wandb

from src.models.DirVRNN import DirVRNN as Model
from src.data_processing.DataLoader import DataLoader 

def main():

    # ---------------------------- Load Configurations --------------------------------------
    with open("src/models/run_config.json", "r") as f:
        run_config = json.load(f)
        f.close()
    
    print("Loaded Configuration Successfully")

    # Extract info
    data_config = run_config["data_config"]
    model_config = run_config["model_config"]
    training_config = run_config["training_config"]
    seed = run_config["seed"]

    # Further unpacking 
    data_name = data_config["data_name"]
    data_range = data_config["time_range"]
    feat_set = data_config["feat_set"]
    train_val_ratio = data_config["train_val_ratio"]
    train_ratio = data_config["train_ratio"]
    K_folds= data_config["K_folds"]

    # Model config
    num_classes = model_config["outcome_size"]
    window_num_obvs = model_config["window_num_obvs"]
    K = model_config["num_clus"]
    latent_dim = model_config["latent_dim"]
    n_fwd_blocks = model_config["n_fwd_blocks"]
    gate_num_hidden_layers = model_config["gate_layers"]
    gate_num_hidden_nodes = model_config["gate_nodes"]
    bias= model_config["bias"]
    dropout = model_config["dropout"]
    
    # Training Config
    batch_size = training_config["batch_size"]
    num_epochs = training_config["num_epochs"]
    lr = training_config["lr"]

    # Prepare wandb
    wandb.init(project="DirVRNN", config=run_config)

    # ---------------------------- Load and Process Data --------------------------------------
    print("\nLoading and Processing Data...")

    loader = DataLoader(data_name=data_name, ts_endpoints=data_range, feat_set=feat_set)
    data_dic = loader.prepare_input(seed=seed, train_val_ratio=train_val_ratio, train_ratio=train_ratio, shuffle=True, K_folds=K_folds)

    for fold in data_dic["CV_folds"].keys():

        # Access Data
        data_fold = data_dic["CV_folds"][fold]

        # Extract train, val, test data
        X_train, X_val, X_test = data_fold["X"]
        y_train, y_val, y_test = data_fold["y"]

        # Compute useful variable
        input_dims = X_train.shape[-1]
        outcome_names = data_dic["load_config"]["outcomes"]
        fold_idx = int(fold.split("_")[-1])

        # Prepare Saving Dictionary
        save_params = {"K_fold_idx": fold_idx, "class_names": outcome_names}

        # -------------------------- Loading and Training Model -----------------------------
        print("\nLoading and Training Model for fold {}...".format(fold_idx))

        # Initialise Model
        model = Model(input_dims=input_dims, 
                    num_classes=num_classes, 
                    window_num_obvs=window_num_obvs, 
                    K=K, 
                    latent_dim=latent_dim, 
                    n_fwd_blocks=n_fwd_blocks, 
                    gate_num_hidden_layers=gate_num_hidden_layers, 
                    gate_num_hidden_nodes=gate_num_hidden_nodes, 
                    bias=bias, 
                    dropout=dropout,  
                    seed=seed
                )

        # Fit Model
        model.fit(train_data=(X_train, y_train), val_data=(X_val, y_val), 
                lr=lr, batch_size=batch_size, num_epochs=num_epochs, 
                save_params=save_params
            )

        # Evaluate model
        output_dir = model.predict(X=X_test, y=y_test, 
                                    save_params=save_params
                                )


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
