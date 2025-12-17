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
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall, RetrievalNormalizedDCG
from recsys.models import get_model
from recsys.movie_lens import get_sequential_dataloaders
from datetime import datetime

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SequentialLightningModule(pyl.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize metrics for ranking
        self.val_recall = RetrievalRecall(top_k=10)
        self.val_ndcg = RetrievalNormalizedDCG(top_k=10)
        self.test_recall = RetrievalRecall(top_k=10)
        self.test_ndcg = RetrievalNormalizedDCG(top_k=10)

        self.save_hyperparameters(ignore=["model"])

    def forward(self, batch):
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        scores = self(batch)
        
        # Create labels: 1 for the positive item, 0 for negative ones
        pos_labels = torch.ones(scores.size(0), 1, device=self.device)
        neg_labels = torch.zeros(scores.size(0), scores.size(1) - 1, device=self.device)
        labels = torch.cat([pos_labels, neg_labels], dim=1)

        loss = self.criterion(scores, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        scores = self(batch)
        
        # Create labels: 1 for the positive item, 0 for negative ones
        pos_labels = torch.ones(scores.size(0), 1, device=self.device)
        neg_labels = torch.zeros(scores.size(0), scores.size(1) - 1, device=self.device)
        labels = torch.cat([pos_labels, neg_labels], dim=1).long()
        
        # For ranking metrics, indexes are needed
        indexes = torch.arange(scores.size(0), device=self.device)

        self.val_recall(scores, labels, indexes=indexes)
        self.val_ndcg(scores, labels, indexes=indexes)

        self.log("val_recall@10", self.val_recall, on_epoch=True, prog_bar=True)
        self.log("val_ndcg@10", self.val_ndcg, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        scores = self(batch)
        
        pos_labels = torch.ones(scores.size(0), 1, device=self.device)
        neg_labels = torch.zeros(scores.size(0), scores.size(1) - 1, device=self.device)
        labels = torch.cat([pos_labels, neg_labels], dim=1).long()
        
        indexes = torch.arange(scores.size(0), device=self.device)

        self.test_recall(scores, labels, indexes=indexes)
        self.test_ndcg(scores, labels, indexes=indexes)

        self.log("test_recall@10", self.test_recall, on_epoch=True, prog_bar=True)
        self.log("test_ndcg@10", self.test_ndcg, on_epoch=True, prog_bar=True)

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
    )
    return mlf_logger


def get_callbacks():
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        save_top_k=1,
        monitor="val_ndcg@10",
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor="val_ndcg@10", patience=3, mode="max"
    )

    callbacks = [
        checkpoint_callback,
        early_stop_callback,
    ]
    return callbacks


def train(model, train_loader, val_loader, mlf_logger, callbacks, training_params):
    trainer = pyl.Trainer(
        max_epochs=training_params.get("epochs", 10),
        logger=mlf_logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer


def main(config):
    logger.info(f"Configuration: {config}")
    pyl.seed_everything(config.get("seed", 42), workers=True)

    mlf_logger = setup_experiment(**config["logging"])

    logger.info(f"Loading dataset: {config['dataset']}")
    (
        train_loader,
        val_loader,
        test_loader,
        num_users,
        num_items,
    ) = get_sequential_dataloaders(**config["dataset"])

    logger.info(f"Loading model: {config['model']['architecture']}")
    model_config = config["model"].copy()
    model_config["num_users"] = num_users
    model_config["num_items"] = num_items
    # Add max_sequence_length to model config from dataset config
    model_config["max_sequence_length"] = config["dataset"]["max_sequence_length"]


    model = get_model(**model_config)
    lightning_model = SequentialLightningModule(
        model=model,
        learning_rate=config["training"].get("learning_rate", 1e-3),
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

    # Test the best model
    trainer.test(dataloaders=test_loader, ckpt_path="best")

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
