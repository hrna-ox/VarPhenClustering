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

    # Get model
    model = DirVRNN(**model_config, device=device).to(device) # type: ignore

    # Train model on data
    print("\n\nTraining model...\n\n")

    model.fit(data_info=data_info, train_info=training_config, run_config=run_config)

    # Save Model 
    save_fd = "exps/Dir_VRNN/{}/{}".format(data_config["data_name"], run_config["run_name"])
    if not os.path.exists(save_fd):
        os.makedirs(save_fd)
        
    torch.save(model.state_dict(), save_fd + "/model.h5")
    # wandb.save(save_fd + "/model.h5")


    # Run on Test data
    print("\n\nRunning model on test data...\n\n")

    # Prepare Test Data
    X_test = torch.Tensor(data_info["X"][-1], device=device)
    y_test = torch.Tensor(data_info["y"][-1], device=device)

    # Run model on test data
    model.eval()
    output_dic = model.predict(X_test, y_test, run_config=run_config)

    # Finish recording session and save outputs
    wandb.finish()
    torch.save(output_dic, save_fd + "/output_dic.h5")

# endregion

if __name__ == "__main__":
    # Do something
    main()
