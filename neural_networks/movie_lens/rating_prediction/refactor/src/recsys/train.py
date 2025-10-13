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
from torchmetrics import AUROC, AveragePrecision
from recsys.models import get_model
from recsys.movie_lens import get_dataloaders
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DefaultLightningModule(pyl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 5e-3,
        loss_function: str = "mse",
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

        loss_functions = {
            "mse": nn.MSELoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "mae": nn.L1Loss(),
        }
        self.criterion = loss_functions[loss_function]

        # Initialize metrics for binary classification
        # Separate instances needed because torchmetrics are stateful and accumulate across batches
        self.train_auc = AUROC(task="binary")
        self.val_auc = AUROC(task="binary")
        self.test_auc = AUROC(task="binary")

        self.train_pr_auc = AveragePrecision(task="binary")
        self.val_pr_auc = AveragePrecision(task="binary")
        self.test_pr_auc = AveragePrecision(task="binary")

        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model(batch).squeeze()

    def training_step(self, batch, batch_idx):
        ratings = batch.pop("rating")
        predictions = self(batch)
        loss = self.criterion(predictions, ratings)

        # Compute metrics for binary classification
        probs = torch.sigmoid(predictions.squeeze())
        self.train_auc(probs, ratings.int())
        self.train_pr_auc(probs, ratings.int())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train_pr_auc", self.train_pr_auc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ratings = batch.pop("rating")
        predictions = self(batch)
        loss = self.criterion(predictions, ratings)

        # Compute metrics for binary classification
        probs = torch.sigmoid(predictions.squeeze())
        self.val_auc(probs, ratings.int())
        self.val_pr_auc(probs, ratings.int())

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True)
        self.log("val_pr_auc", self.val_pr_auc, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        ratings = batch.pop("rating")
        predictions = self(batch)
        loss = self.criterion(predictions, ratings)

        # Compute metrics for binary classification
        probs = torch.sigmoid(predictions.squeeze())
        self.test_auc(probs, ratings.int())
        self.test_pr_auc(probs, ratings.int())

        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_step=False, on_epoch=True)
        self.log("test_pr_auc", self.test_pr_auc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def setup_experiment(experiment_name: str, run_name: str):
    from pytorch_lightning.loggers import MLFlowLogger

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tracking_path = pathlib.Path(__file__).absolute().parents[2] / "experiments"
    mlflow.set_tracking_uri(tracking_path)
    mlflow.set_experiment(experiment_name)
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=str(tracking_path),
        run_name=run_name + "_" + timestamp,
        # log_model=True,
    )
    return mlf_logger


def get_callbacks():
    from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        monitor="val_loss",
        # filename="best_model",
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


def train(model, train_loader, val_loader, mlf_logger, callbacks, training_params):
    trainer = pyl.Trainer(
        max_epochs=training_params.get("epochs", 10),
        logger=mlf_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
    )
    trainer.fit(model, train_loader, val_loader)

    return trainer


def main(config):
    logger.info(f"Configuration: {config}")

    mlf_logger = setup_experiment(**config["logging"])

    logger.info(f"Loading dataset: {config['dataset']}")
    (
        train_loader,
        val_loader,
        test_loader,
        num_users,
        num_items,
        user_feature_dims,
        item_feature_dims,
    ) = get_dataloaders(**config["dataset"])

    logger.info(f"Loading model: {config['model']['architecture']}")
    model_config = config["model"].copy()
    model_config["num_users"] = num_users
    model_config["num_items"] = num_items
    if user_feature_dims:
        model_config["user_continuous_dim"] = user_feature_dims["continuous_dim"]
        model_config["user_categorical_dims"] = user_feature_dims["categorical_dims"]
    if item_feature_dims:
        model_config["item_continuous_dim"] = item_feature_dims["continuous_dim"]
        model_config["item_categorical_dims"] = item_feature_dims["categorical_dims"]

    model = get_model(**model_config)
    lightning_model = DefaultLightningModule(
        model=model,
        learning_rate=config["training"].get("learning_rate", 5e-3),
        loss_function=config["training"].get("loss_function", "mse"),
    )
    callbacks = get_callbacks()

    trainer = train(
        model=lightning_model,
        train_loader=train_loader,
        val_loader=val_loader,
        mlf_logger=mlf_logger,
        callbacks=callbacks,
        training_params=config["training"],
    )

    with mlflow.start_run(run_id=mlf_logger.run_id):
        mlflow.log_artifact(trainer.checkpoint_callback.best_model_path, "checkpoints")
        mlflow.log_dict(config, "config.yaml")
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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
