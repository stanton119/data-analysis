import argparse
import logging
import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
import torch.nn as nn
import pytorch_lightning as pyl
import yaml
from models import get_model
from dataloaders import get_dataloaders
from datetime import datetime

logger = logging.getLogger(__name__)


class DefaultLightningModule(pyl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 5e-3,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, user_ids, movie_ids):
        return self.model(user_ids, movie_ids)

    def training_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def setup_experiment(experiment_name: str, run_name: str):
    from pytorch_lightning.loggers import MLFlowLogger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri="experiments",
        run_name=run_name + "_" + timestamp,
    )
    return mlf_logger


def get_callbacks():
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", save_top_k=1, monitor="val_loss"
    )

    class EarlyStoppingWithMinEpochs(Callback):
        def __init__(self, min_epochs, **kwargs):
            super().__init__()
            self.min_epochs = min_epochs
            self.early_stopping = EarlyStopping(**kwargs)

        def on_validation_end(self, trainer, pl_module):
            if trainer.current_epoch >= self.min_epochs - 1:
                self.early_stopping.on_validation_end(trainer, pl_module)

        def on_train_end(self, trainer, pl_module):
            self.early_stopping.on_train_end(trainer, pl_module)

    early_stop_callback = EarlyStoppingWithMinEpochs(
        min_epochs=6, monitor="val_loss", patience=2, mode="min"
    )
    callbacks = [checkpoint_callback, early_stop_callback]
    return callbacks


def train(model, train_loader, eval_loader, mlf_logger, callbacks, training_params):
    trainer = pyl.Trainer(
        max_epochs=training_params.get("epochs", 10),
        logger=mlf_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, eval_loader)

    return trainer.test(model, eval_loader)


def main(args):
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuration: {config}")

    mlf_logger = setup_experiment(**config["logging"])

    # Load model dynamically
    logger.info(f"Loading model: {config['model']['architecture']}")
    model = get_model(**config["model"])
    lightning_model = DefaultLightningModule(
        model=model,
        learning_rate=config["training"].get("learning_rate", 5e-3),
    )
    callbacks = get_callbacks()

    # Prepare dataset
    logger.info(f"Loading dataset: {config['dataset']['name']}")
    train_loader, eval_loader = get_dataloaders(**config["dataset"])

    loss = train(
        model=lightning_model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        mlf_logger=mlf_logger,
        callbacks=callbacks,
        training_params=config["training"],
    )

    # mlf_logger.log_metrics({"test_loss": loss[0]["test_loss_epoch"]})
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, callbacks[0].best_model_path)
    mlflow.pytorch.log_model(model, "model")
    mlflow.pytorch.log_model(lightning_model, "lightning_model")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the config.yaml file",
    )

    args = parser.parse_args()
    main(args)
