"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

File to train Dirichlet-VRNN Model.
"""

# =============== IMPORT LIBRARIES ===============s

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.tensorboard.writer as writer

from src.models.Dir_VRNN.model import BaseModel
import src.models.Dir_VRNN.Dir_VRNN_utils as utils


# =============== Definition of training step ==================

class DirVRNN(BaseModel):
    """
    Wrapper class to Base Model that implements full training and testing algorithms.
    """

    def __init__(self, data_info, model_config, training_config):
        """
        Initialise model object.

        Params:
        - data_info: dictionary containing data information. Includes keys:
            - data_og: a pair of original data (X, y)
            - X: tuple containing training, validation and test data, normalised and imputed accordingly.
            - y: tuple containing training, validation and test labels, normalised and imputed accordingly.
            - ids: tuple containing training, validation and test patient ids.
            - mask: tuple containing training, validation and test mask observation values.
            - data_properties: dictionary containing data relevant properties such as features, normalisers, etc.
            - data_load_config: configuration parameters used to process and load data.

        - model_config: dictionary containing model configuration parameters. Includes keys:
            - "input_size": input size of data.
            - "outcome_size": output size of data.
            - "num_clus": number of clusters.
            - "latent_size": dimensionality of latent space.
            - "gate_layers": number of layers in gating components.
            - "gate_nodes": number of nodes in gating components.
            - "feat_extr_layers": number of layers in feature extraction components.
            - "feat_extr_nodes": number of nodes in feature extraction components.

        - training_config: dictionary containing training configuration parameters. Includes keys:
            - "lr": learning rate for training
            - "epochs_init": number of epochs to train initialisation
            - "epochs": number of epochs for main training
            - "bs": batch size
        """

        # Initialise parameters
        self.data_info = data_info
        self.model_config = model_config
        self.training_config = training_config

        # Unpack relevant data information
        X_train, X_val, X_test = self.data_info["X"]
        y_train, y_val, y_test = self.data_info["y"]
        batch_size = training_config["bs"]

        # Compute dimensionality of input and output
        input_size = X_train.shape[-1]
        outcome_size = y_train.shape[-1]

        # Prepare data for training
        train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
   
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset, batch_size=X_val.shape[0])

        # Prepare data for evaluating
        test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
        self.test_loader = DataLoader(test_dataset, batch_size=X_test.shape[0])

        # Finally initialise parent model
        super().__init__(input_size=input_size, outcome_size=outcome_size,
                         **model_config)

        # Empty attribute which will be updated later.
        self.optimizer = None

        # Useful for saving results and exps
        exp_fd = f"exps/Dir_VRNN/{self.data_info['data_load_config']['data_name']}"
        self.writer = writer.SummaryWriter(log_dir=utils.get_exp_run_path(exp_fd))

    def fit(self):
        """
        Training algorithm method for training DirVRNN model.
        """

        # Unpack training parameters
        train_params = self.training_config
        num_epochs, lr = train_params["epochs"], train_params["lr"]

        # Useful for model training
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # ============ TRAINING LOOP ==============

        # Iterate through epochs
        for epoch in range(1, num_epochs + 1):
            train_loss = self._single_epoch_train(epoch)
            val_loss, val_history = self._single_epoch_val(epoch)

            # Save results to Writer
            self.writer.add_scalars("Training vs Validation Loss",
                                    {"Train": train_loss, "Val": val_loss}, epoch)

    def predict(self, X, y=None):
        """
        Prediction method for DirVRNN model.
        """

        # Compute predictions
        _, history = self._single_epoch_val(test_mode=True, X=X, y=y)

        return history

    def _single_epoch_train(self, epoch):
        """
        Train model over 1 epoch.
        """

        # Set model in training mode
        self.train()

        # Set training loss to 0 at the beginning of each epoch
        train_loss = 0

        # Iterate through batches of train data loader
        for batch_id, (X_batch, y_batch) in enumerate(self.train_loader):

            # Apply optimiser
            self.optimizer.zero_grad()  # Set optimiser gradients to 0 for every batch
            loss, _ = self.forward(X_batch, y_batch)  # Model forward pass on batch data

            # Back_propagate
            loss.backward()
            self.optimizer.step()

            # Add to train loss
            train_loss += loss.item()

            # Print message
            print("Train epoch: {}   [{:.5f} - {:.0f}%]".format(
                epoch, loss.item(), 100. * batch_id / len(self.train_loader)),
                end="\r")

        # Print message at the end of epoch
        print("Train epoch: {}   [{:.5f} - {:.0f}%]".format(
            epoch, train_loss / len(self.train_loader), 100.))

        return train_loss

    def _single_epoch_val(self, epoch=None, test_mode=False, X=None, y=None):
        """
        Validate model over validation epoch for current data.

        If test_mode set to True, then will use test data instead.
        If X and y are not None, then will use these instead of the data loaders.
        """

        # Set the model to evaluation mode
        self.eval()

        # Set validation loss to 0 at the beginning of each epoch
        loss, history = 0, {}

        # Decide which loader to use based on test_mode flag
        if test_mode:
            if X is not None:
                loader = DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(y)), batch_size=X.shape[0])
            else:
                loader = self.test_loader
        else:
            loader = self.val_loader

        # Iterate through (the 1 batch) of validation data loader
        for batch_id, (X_batch, y_batch) in enumerate(loader):

            # Compute Loss and tracker history
            loss, history = self.forward(X_batch, y_batch)

            if test_mode:

                # Add test data to history to allow visualisation of results
                history["X"] = X_batch
                history["y"] = y_batch

                # Print message
                print("Test: [{:.5f} - {:.0f}%]".format(loss, 100. * batch_id / len(loader),
                                                        end="\r"))

            else:
                print("Val epoch: {}   [{:.5f} - {:.0f}%]".format(
                    epoch, loss, 100. * batch_id / len(loader),
                    end="\r"))

            break    # Stop after first batch
            
        return loss, history


if __name__ == "__main__":
    # Do something
    pass
