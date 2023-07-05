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
from src.models.Dir_VRNN.model import DirVRNN

# endregion


# region =============== MAIN ===============
def main():

    # Load Configuration for Run
    with open("src/models/Dir_VRNN/run_config.json", "r") as f:     
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

    # Initialize WandB Session for logging metrics, results and model weights 
    wandb.init(
        name= "{}-{}-{}".format(model_name, data_name, run_name),
        entity="hrna-ox", 
        dir=f"exps/Dir_VRNN/{data_name}/{run_name}/",
        project="Dir_VRNN", 
        config=run_config
    )


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

    # Train model on data
    print("\n\nTraining model\n\n")
    print("Training with K={} folds".format(run_config["K_folds"]))

    # Iterate over folds
    for idx, data_arrs in enumerate(data_dic["CV_folds"]):

        # Load model
        model = DirVRNN(
            i_size=run_config["i_size"],
            o_size=run_config["o_size"],
            w_size=run_config["w_size"],
            K=run_config["K"],
            l_size=run_config["l_size"],
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
        )

            
        # Save Model 
        save_fd = "exps/Dir_VRNN/{}/{}".format(run_config["data_name"], run_config["run_name"])
        if not os.path.exists(save_fd):
            os.makedirs(save_fd)
            
        torch.save(model.state_dict(), save_fd + "/model.h5")
        # wandb.save(save_fd + "/model.h5")


        # Run on Test data
        print("\n\nRunning model on test data...\n\n")

        # Prepare Test Data
        X_test = torch.Tensor(x_test, device=device)
        y_test = torch.Tensor(y_test, device=device)

        # Run model on test data
        model.eval()
        log = model.predict(X_test, y_test, run_config=run_config)

        # Finish recording session and save outputs
        wandb.finish()
        torch.save(log, save_fd + "/output_dic.h5")

# endregion

if __name__ == "__main__":
    # Do something
    main()
