import pathlib
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
        self.log(
            name="train_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log(
            name="val_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        user_ids, movie_ids, ratings = batch
        predictions = self(user_ids, movie_ids)
        loss = nn.MSELoss()(predictions, ratings)
        self.log(
            name="test_loss", value=loss, on_step=True, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def setup_experiment(experiment_name: str, run_name: str):
    from pytorch_lightning.loggers import MLFlowLogger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_path = pathlib.Path(__file__).absolute().parents[1] / "experiments"
    mlflow.set_tracking_uri(tracking_path)
    mlflow.set_experiment(experiment_name)
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=str(tracking_path),
        run_name=run_name + "_" + timestamp,
        log_model=True,
    )
    return mlf_logger


def get_callbacks():
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints", save_top_k=1, monitor="val_loss"
    )

    class LayerwiseGradientNormLogger(Callback):
        def __init__(self, log_every_n_steps=1):
            super().__init__()
            self.log_every_n_steps = log_every_n_steps

        def on_after_backward(self, trainer, pl_module):
            # Log gradient norm for each layer
            if trainer.global_step % self.log_every_n_steps == 0:
                for name, param in pl_module.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm(2).item()
                        pl_module.log(
                            f"grad_norm_{name}", grad_norm, on_step=True, on_epoch=False
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
    layerwise_gradient_norm_logger = LayerwiseGradientNormLogger(log_every_n_steps=1)

    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        layerwise_gradient_norm_logger,
    ]
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
        model=model, learning_rate=config["training"].get("learning_rate", 5e-3)
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

    with mlflow.start_run(run_id=mlf_logger.run_id):
        model_info = mlflow.pytorch.log_model(model, "model", code_paths=["src"])
        # model_info = mlflow.pytorch.log_model(model, "model2", code_paths=["models"])
        print(model_info.model_uri)
        print(model_info)
        mlflow.pytorch.log_model(lightning_model, "lightning_model")
        mlflow.log_params(config)


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
