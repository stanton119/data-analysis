import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import mlflow.pytorch
import argparse

from src.data_generator import generate_dummy_data
from src.dataset import TabularSequenceDataset, create_dataloaders
from src.models import ModelProtocol, get_model


class SequenceDataModule(pl.LightningModule):
    def __init__(self, model: ModelProtocol, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.criterion = torch.nn.MSELoss()
        self.learning_rate = learning_rate

    def forward(self, tabular, sequence):
        return self.model(tabular, sequence)

    def training_step(self, batch, batch_idx):
        tabular, sequence, target = batch["tabular"], batch["sequence"], batch["target"]
        output = self.forward(tabular, sequence)
        loss = self.criterion(output.squeeze(), target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        tabular, sequence, target = batch["tabular"], batch["sequence"], batch["target"]
        output = self.forward(tabular, sequence)
        loss = self.criterion(output.squeeze(), target)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def train_model(
    model_name: str,
    num_epochs: int,
    batch_size: int,
    num_samples: int,
    sequence_length: int,
    num_features: int,
):
    print("Generating dummy data...")
    dummy_data = generate_dummy_data(
        num_samples=num_samples,
        sequence_length=sequence_length,
        num_features=num_features,
    )

    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(dummy_data, batch_size=batch_size)

    print("Initializing model and trainer...")

    model_instance = get_model(model_name, num_features, sequence_length)

    mlflow.set_experiment("Sequence Data Models")

    with mlflow.start_run():
        mlflow.pytorch.autolog()

        # Log hyperparameters
        mlflow.log_params(
            {
                "model_name": model_name,
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "num_samples": num_samples,
                "sequence_length": sequence_length,
                "num_features": num_features,
            }
        )

        model = SequenceDataModule(model=model_instance)
        logger = MLFlowLogger(experiment_name="Sequence Data Models")
        trainer = pl.Trainer(max_epochs=num_epochs, logger=logger)

        print("Starting training...")
        trainer.fit(model, train_loader, val_loader)
        print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sequence data model.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="StatisticalAggregationModel",
        help="Name of the model to train (e.g., StatisticalAggregationModel)",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num_samples", type=int, default=1000, help="Number of dummy data samples"
    )
    parser.add_argument(
        "--sequence_length", type=int, default=50, help="Length of the sequence data"
    )
    parser.add_argument(
        "--num_features", type=int, default=10, help="Number of tabular features"
    )

    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        sequence_length=args.sequence_length,
        num_features=args.num_features,
    )
