"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

File to train Dirichlet-VRNN Model.
"""

# region =============== IMPORT LIBRARIES ===============s
import json

import torch
import wandb

from src.data_processing.data_loader import data_loader
from src.models.Dir_VRNN.model import DirVRNN

# endregion


# region =============== MAIN ===============
def main():

    # Load Configuration 
    with open("src/models/DIR_VRNN/run_config.json", "r") as f:
        run_config = json.load(f)
        f.close()

    # Extract info
    data_config = run_config["data_config"]
    model_config = run_config["model_config"]
    training_config = run_config["training_config"]

    # GPU and model setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(run_config["model_config"]["seed"])
    torch.backends.cudnn.deterministic = True

    # Set Weights and Biases Logger
    wandb.init(entity="hrna-ox", dir=f"exps/Dir_VRNN/{data_config['data_name']}",
                project="Dir_VRNN", config=run_config)
    
    # Load and process data
    data_info = data_loader(**data_config)

    # Get model
    model = DirVRNN(**model_config).to(device)
    wandb.watch(model, log="all")

    # Train model on data
    model.train(data_info=data_info, train_info=training_config, run_config=run_config)
    torch.save(model.state_dict(), "DirVRNN.h5")
    wandb.save("model.h5")

    # Prepare Test Data
    X_test, y_test = data_info["X_test"][-1], data_info["y_test"][-1]

    # Run on Test data
    model.eval()
    model.predict(X_test, y_test, epoch="test")

# endregion

if __name__ == "__main__":
    # Do something
    main()
