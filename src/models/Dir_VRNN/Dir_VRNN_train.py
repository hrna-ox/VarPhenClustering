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
    with open("src/models/Dir_VRNN/run_config.json", "r") as f:     
        run_config = json.load(f)
        f.close()

    # Extract info
    run_name = run_config["run_name"]
    data_config = run_config["data_config"]
    model_config = run_config["model_config"]
    training_config = run_config["training_config"]

    # GPU and model setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(run_config["model_config"]["seed"])
    torch.backends.cudnn.deterministic = True # type: ignore
    print("\n\nRunning training on device: \n", device, "\n\n")
    
    # Load and process data
    data_info = data_loader(**data_config)

    # Get model
    model = DirVRNN(**model_config, device=device).to(device) # type: ignore

    # Train model on data
    print("\n\nTraining model...\n\n")

    model.fit(data_info=data_info, train_info=training_config, run_config=run_config)

    # Save Model 
    torch.save(model.state_dict(), "DirVRNN.h5")
    wandb.save("model.h5")


    # Run on Test data
    print("\n\nRunning model on test data...\n\n")

    # Prepare Test Data
    X_test = torch.Tensor(data_info["X"][-1], device=device)
    y_test = torch.Tensor(data_info["y"][-1], device=device)

    # Run model on test data
    model.eval()
    output_dic = model.predict(X_test, y_test)

    # Finish recording session
    wandb.finish()

# endregion

if __name__ == "__main__":
    # Do something
    main()
