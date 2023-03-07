"""
Author: Henrique Aguiar
Contact: henrique.aguiar@eng.ox.ac.uk

File to train Dirichlet-VRNN Model.
"""

# =============== IMPORT LIBRARIES ===============
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.Dir_VRNN.model import BaseModel


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
            - data_og: pair of original data (X, y)
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
        train_dataset, val_dataset = TensorDataset(X_train, y_train), TensorDataset(X_val, y_val)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_dataset)

        # Prepare data for evaluating
        test_dataset = TensorDataset(X_test, y_test)
        self.test_loader = DataLoader(test_dataset)

        # Finally initialise parent model
        super().__init__(input_size=input_size, outcome_size=outcome_size,
                         **model_config)

        # Empty attribute which will be updated later.
        self.optimizer = None

    def fit(self):
        """
        Training algorithm method for training DirVRNN model.
        """

        # Unpack training parameters
        train_params = self.training_config
        num_epochs, lr = train_params["epochs"], train_params["lr"]

        # Useful for model training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # ============ TRAINING LOOP ==============

        # Iterate through epochs
        for epoch in range(1, num_epochs + 1):
            self._single_epoch_train(epoch)
            self._single_epoch_val(epoch)

    def test(self):
        """
        Relevant outputs for model performance on test data after training.
        """

        # Set the model to evaluation mode
        self.eval()

        # Iterate through batches of test data loader
        for batch_id, (X_batch, y_batch) in enumerate(self.test_loader):

            # Compute Loss
            loss = self.model(X_batch).item()


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
            self.optimizer.zero_grad()                      # Set optimiser gradients to 0 for every batch
            loss = self.forward(X_batch, y_batch)          # Model forward pass on batch data

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

    def _single_epoch_val(self, epoch):
        """
        Validate model over validation epoch for current data.
        """

        # Set the model to evaluation mode
        self.eval()

        # Set validation loss to 0 at the beginning of each epoch
        loss = 0

        # Iterate through (the 1 batch) of validation data loader
        for batch_id, (X_batch, y_batch) in enumerate(self.val_loader):

            # Compute Loss
            loss = self.model(X_batch).item()

            print("Val epoch: {}   [{:.5f} - {:.0f}%]".format(
                epoch, loss.item(), 100. * batch_id / len(self.val_loader)),
                end="\r")

        return loss


if __name__ == "__main__":

    # Do something
    pass