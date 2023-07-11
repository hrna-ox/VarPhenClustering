"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

File to train Dirichlet-VRNN Model.
"""

# region =============== IMPORT LIBRARIES ===============s
import json
import os

import torch
import wandb

from src.data_processing.DataLoader import DataLoader
from src.models.DirVRNN.model import DirVRNN

# endregion


# region =============== MAIN ===============
def main():

    # Load Configuration for Run
    with open("src/models/DirVRNN/run_config.json", "r") as f:     
        run_config = json.load(f)
        f.close()

    # Extract some key info
    model_name, data_name, run_name = run_config["model_name"], run_config["data_name"], run_config["run_name"]

    # GPU and model setting
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.backends.cudnn.deterministic = True # type: ignore
    print("\n\nRunning training on device: \n", device, "\n\n")
    
    # Set torch seed for any probability computation
    torch.manual_seed(run_config["seed"])

    # Load Data
    data_dic = DataLoader(
        data_name=run_config["data_name"], 
        feat_set=run_config["feat_set"],
        ts_endpoints=run_config["ts_endpoints"]
    ).prepare_input(
        seed=run_config["seed"],
        train_val_ratio=run_config["train_val_ratio"],
        train_ratio=run_config["train_ratio"],
        shuffle=run_config["shuffle"],
        K_folds=run_config["K_folds"]
    )


    # Iterate over folds
    for idx, (fold_key, data_arrs) in enumerate(data_dic["CV_folds"].items()):

        # Print message
        print("\n\nTraining model on fold {} of {}...\n\n".format(idx+1, run_config["K_folds"]))

        # Extra parameters to pass to fit predict functions for better logging
        viz_params = {
            "class_names": data_dic["load_config"]["outcomes"],
            "feat_names": data_dic["load_config"]["features"],
            "save_dir": f"exps/DirVRNN/{data_name}/{run_name}/{fold_key}/",
            "fold": idx+1
        }
        
        # Initialize WandB Session for logging metrics, results and model weights 
        save_dir = f"exps/DirVRNN/{data_name}/{run_name}/{fold_key}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(f"{save_dir}/train")
            os.makedirs(f"{save_dir}/test")
            os.makedirs(f"{save_dir}/val")
        
        wandb.init(
            name= "{}-{}-{}-{}".format(model_name, data_name, run_name, fold_key),
            entity="hrna-ox", 
            dir=save_dir,
            project="DirVRNN", 
            config=run_config
        )

        # Load model
        model = DirVRNN(
            i_size=run_config["i_size"],
            o_size=run_config["o_size"],
            w_size=run_config["w_size"],
            K=run_config["K"],
            l_size=run_config["l_size"],
            n_fwd_blocks=run_config["n_fwd_blocks"],
            gate_hidden_l=run_config["gate_hidden_l"],
            gate_hidden_n=run_config["gate_hidden_n"],
            bias=run_config["bias"],
            dropout=run_config["dropout"],
            seed=run_config["seed"]
        ).to(device) # type: ignore

        # Access Data
        x_train, x_val, x_test = data_arrs["X"]
        y_train, y_val, y_test = data_arrs["y"]

        # Train Model on Train and Val Data
        model.fit(
            train_data=(x_train, y_train),
            val_data=(x_val, y_val),
            K_fold_idx=idx+1,
            lr=run_config["lr"],
            batch_size=run_config["batch_size"],
            num_epochs=run_config["num_epochs"],
            viz_params=viz_params,
        )

        # Save Model and Model Weights    
        torch.save(model.state_dict(), save_dir + "model.h5")


        # Run on Test data
        print("\n\nRunning model on test data...\n\n")

        # Run model on test data
        log = model.predict(
            x_test, y_test, 
            run_config=run_config, 
            save_dir=save_dir, 
            class_names=data_dic["load_config"]["outcomes"], 
            feat_names=data_dic["load_config"]["features"]
        )

        # Finish recording session and save outputs
        wandb.finish()

# endregion

if __name__ == "__main__":
    # Do something
    main()
