"""
Training script for multi-task learning models.

This script provides functionality to train and evaluate different multi-task learning models:
1. Single model per task
2. Shared bottom model (hard parameter sharing)
3. Mixture of Experts (OMoE)
4. Multi-gate Mixture of Experts (MMoE)

It supports both synthetic datasets and the UCI Census Income dataset.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import mlflow
import mlflow.pytorch
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from src.data_sources import (
    load_uci_census_dataset,
    create_synthetic_dataset,
    generate_mmoe_synthetic_data,
    generate_correlation_experiment_datasets,
)
from src.torch_datasets import MultiTaskTabularDataset, create_train_test_dataloaders
from src.models import ModelProtocol, get_model


class MultiTaskLightningModule(pl.LightningModule):
    """PyTorch Lightning module for multi-task learning models."""

    def __init__(
        self,
        model: ModelProtocol,
        task_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize the Lightning module.

        Args:
            model (ModelProtocol): The multi-task learning model
            task_weights (Dict[str, float], optional): Weights for each task's loss
            learning_rate (float): Learning rate for the optimizer
        """
        super().__init__()
        self.model = model
        self.task_weights = task_weights or {task: 1.0 for task in model.task_names}
        self.learning_rate = learning_rate

        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x)

    def _get_loss(self, batch, batch_idx, prefix: str = ""):
        """
        Compute loss for a batch.

        Args:
            batch: The input batch
            batch_idx: Batch index
            prefix: Prefix for logging metrics (e.g., 'train', 'val', 'test')

        Returns:
            Tuple containing total loss and dictionary of task-specific losses
        """
        # Extract features and targets
        features = batch["features"]
        targets = {task: batch[task] for task in self.model.task_names}

        # Forward pass
        outputs = self.forward(features)

        # Initialize task losses dictionary and total loss
        task_losses = {}
        total_loss = 0.0

        # Calculate loss for each task
        for task_name in self.model.task_names:
            # Skip tasks that aren't in the outputs (for SingleTaskModel)
            if task_name not in outputs:
                continue

            # Ensure outputs and targets have the same shape
            output = outputs[task_name]
            target = targets[task_name]

            # Squeeze output if needed to match target shape
            if output.shape != target.shape:
                output = output.squeeze()

            # Calculate appropriate loss based on task type
            if self.model.task_types[task_name] == "binary":
                task_loss = nn.BCELoss()(output, target)
            else:  # regression
                task_loss = nn.MSELoss()(output, target)

            # Apply task weight
            task_weight = self.task_weights.get(task_name, 1.0)
            weighted_task_loss = task_weight * task_loss

            # Store task loss and add to total
            task_losses[task_name] = task_loss
            total_loss += weighted_task_loss

        # Log metrics
        self.log(f"{prefix}loss", total_loss, prog_bar=True)

        # Log task-specific metrics
        for task_name, loss in task_losses.items():
            self.log(f"{prefix}{task_name}_loss", loss, prog_bar=True)

            # For binary classification tasks, compute accuracy
            if self.model.task_types[task_name] == "binary":
                # Ensure shapes match for accuracy calculation
                output = outputs[task_name]
                target = targets[task_name]
                if output.shape != target.shape:
                    output = output.squeeze()

                preds = (output > 0.5).float()
                accuracy = (preds == target).float().mean()
                self.log(f"{prefix}{task_name}_accuracy", accuracy, prog_bar=True)

        return total_loss, task_losses

    def training_step(self, batch, batch_idx):
        """Training step."""
        total_loss, _ = self._get_loss(batch, batch_idx, prefix="train_")
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        total_loss, _ = self._get_loss(batch, batch_idx, prefix="val_")
        return total_loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        total_loss, _ = self._get_loss(batch, batch_idx, prefix="test_")
        return total_loss

    def configure_optimizers(self):
        """Configure optimizer."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def prepare_uci_census_data(batch_size: int = 32, val_ratio: float = 0.15):
    """
    Prepare UCI Census Income dataset for multi-task learning.

    Args:
        batch_size (int): Batch size for dataloaders
        val_ratio (float): Proportion of training data to use for validation

    Returns:
        Dict: Dictionary containing dataloaders, feature dimension, and task information
    """
    # Load UCI Census dataset
    census_data = load_uci_census_dataset(create_multi_task=True)

    # Define categorical columns
    categorical_cols = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    # Define feature columns (exclude target columns)
    feature_cols = [
        col
        for col in census_data["train"].columns
        if col
        not in ["marital_status", "income", "income_binary", "marital_status_binary"]
    ]

    # Define task columns and types
    task_cols = ["income_binary", "marital_status_binary"]
    task_types = {task: "binary" for task in task_cols}

    # Create dataset and dataloaders
    dataset_kwargs = {
        "feature_cols": feature_cols,
        "task_cols": task_cols,
        "categorical_cols": categorical_cols,
        "normalize_features": True,
    }

    dataloaders = create_train_test_dataloaders(
        census_data["train"],
        census_data["test"],
        batch_size=batch_size,
        val_ratio=val_ratio,
        dataset_kwargs=dataset_kwargs,
    )

    # Create a temporary dataset to get feature dimension
    temp_dataset = MultiTaskTabularDataset(census_data["train"], **dataset_kwargs)
    # TODO replace with feature_cols len?

    return {
        "dataloaders": dataloaders,
        "feature_dim": temp_dataset.feature_dim,
        "task_names": task_cols,
        "task_types": task_types,
    }


def prepare_synthetic_data(
    num_samples: int = 10000,
    num_features: int = 20,
    num_tasks: int = 2,
    task_correlation: float = 0.5,
    batch_size: int = 32,
    val_ratio: float = 0.15,
    use_mmoe_synthetic: bool = True,
):
    """
    Prepare synthetic dataset for multi-task learning.

    Args:
        num_samples (int): Number of samples to generate
        num_features (int): Number of features to generate
        num_tasks (int): Number of tasks (target variables)
        task_correlation (float): Correlation between tasks
        batch_size (int): Batch size for dataloaders
        val_ratio (float): Proportion of training data to use for validation
        use_mmoe_synthetic (bool): Whether to use the MMoE synthetic data generation

    Returns:
        Dict: Dictionary containing dataloaders, feature dimension, and task information
    """
    # Generate synthetic data
    if use_mmoe_synthetic:
        synthetic_data = generate_mmoe_synthetic_data(
            num_samples=num_samples,
            num_features=num_features,
            task_correlation=task_correlation,
        )
        # Extract just the train and test dataframes
        data = {"train": synthetic_data["train"], "test": synthetic_data["test"]}
        metadata = synthetic_data["metadata"]
    else:
        # Use simpler synthetic data generation
        data = create_synthetic_dataset(
            num_samples=num_samples,
            num_features=num_features,
            num_tasks=num_tasks,
            task_correlations=[1.0] + [task_correlation] * (num_tasks - 1),
        )
        metadata = {"task_correlation_input": task_correlation, "is_regression": False}

    # Define feature and task columns
    feature_cols = [f"feature_{i}" for i in range(num_features)]
    task_cols = [f"task_{i}" for i in range(num_tasks)]
    task_types = {task: "binary" for task in task_cols}

    # Create dataset and dataloaders
    dataset_kwargs = {
        "feature_cols": feature_cols,
        "task_cols": task_cols,
        "normalize_features": True,
    }

    dataloaders = create_train_test_dataloaders(
        data["train"],
        data["test"],
        batch_size=batch_size,
        val_ratio=val_ratio,
        dataset_kwargs=dataset_kwargs,
    )

    # Create a temporary dataset to get feature dimension
    temp_dataset = MultiTaskTabularDataset(data["train"], **dataset_kwargs)

    return {
        "dataloaders": dataloaders,
        "feature_dim": temp_dataset.feature_dim,
        "task_names": task_cols,
        "task_types": task_types,
        "metadata": metadata,
    }


def train_model(
    model_name: str,
    data_config: Dict[str, Any],
    model_params: Optional[Dict[str, Any]] = None,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    task_weights: Optional[Dict[str, float]] = None,
    experiment_name: str = "Multi-Task Learning Models",
    run_name: Optional[str] = None,
    early_stopping_patience: int = 10,
):
    """
    Train a multi-task learning model.

    Args:
        model_name (str): Name of the model to train
        data_config (Dict): Configuration for the dataset
        model_params (Dict, optional): Model-specific parameters
        num_epochs (int): Maximum number of training epochs
        learning_rate (float): Learning rate for the optimizer
        task_weights (Dict[str, float], optional): Weights for each task's loss
        experiment_name (str): Name of the MLflow experiment
        run_name (str, optional): Name of the MLflow run
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping

    Returns:
        pl.LightningModule: Trained model
    """
    # Extract data configuration
    dataloaders = data_config["dataloaders"]
    feature_dim = data_config["feature_dim"]
    task_names = data_config["task_names"]
    task_types = data_config["task_types"]

    # Initialize model
    model_instance = get_model(
        model_name=model_name,
        num_tabular_features=feature_dim,
        task_names=task_names,
        task_types=task_types,
        model_params=model_params,
    )

    # Set up MLflow tracking
    mlflow.set_experiment(experiment_name)

    # Generate run name if not provided
    if run_name is None:
        run_name = f"{model_name}"
        if model_params:
            # Add key model parameters to run name
            if "target_task" in model_params:
                run_name += f"_{model_params['target_task']}"
            if "num_experts" in model_params:
                run_name += f"_experts{model_params['num_experts']}"

    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        params_to_log = {
            "model_name": model_name,
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "feature_dim": feature_dim,
            "task_names": ",".join(task_names),
            "task_types": str(task_types),
        }

        # Add model-specific parameters
        if model_params:
            for key, value in model_params.items():
                if isinstance(value, (int, float, str, bool)):
                    params_to_log[f"model_{key}"] = value
                elif isinstance(value, list):
                    params_to_log[f"model_{key}"] = str(value)

        # Add task weights
        if task_weights:
            for task, weight in task_weights.items():
                params_to_log[f"weight_{task}"] = weight

        # Add metadata if available
        if "metadata" in data_config:
            for key, value in data_config["metadata"].items():
                if isinstance(value, (int, float, str, bool)):
                    params_to_log[f"data_{key}"] = value

        mlflow.log_params(params_to_log)

        # Create Lightning module
        lightning_model = MultiTaskLightningModule(
            model=model_instance, task_weights=task_weights, learning_rate=learning_rate
        )

        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                filename="{epoch}-{val_loss:.4f}",
            ),
            EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, mode="min"
            ),
        ]

        # Set up trainer with MLflow logger
        # Note: We're using only the PyTorch Lightning MLFlowLogger, not mlflow.pytorch.autolog()
        # to avoid duplicate logging
        logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name)
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            logger=logger,
            callbacks=callbacks,
            log_every_n_steps=10,
        )

        # Train model
        trainer.fit(
            lightning_model,
            train_dataloaders=dataloaders["train"],
            val_dataloaders=dataloaders["val"],
        )

        # Test model
        test_results = trainer.test(lightning_model, dataloaders=dataloaders["test"])

        # Log test metrics
        for key, value in test_results[0].items():
            mlflow.log_metric(key, value)

        return lightning_model


def run_correlation_experiment(
    model_names: List[str],
    correlations: List[float],
    num_samples: int = 10000,
    num_features: int = 20,
    batch_size: int = 64,
    num_epochs: int = 50,
    model_params_dict: Optional[Dict[str, Dict[str, Any]]] = None,
):
    """
    Run an experiment comparing different models across varying task correlations.

    Args:
        model_names (List[str]): List of model names to compare
        correlations (List[float]): List of correlation values to test
        num_samples (int): Number of samples for synthetic datasets
        num_features (int): Number of features for synthetic datasets
        batch_size (int): Batch size for training
        num_epochs (int): Maximum number of training epochs
        model_params_dict (Dict): Dictionary mapping model names to their parameters
    """
    experiment_name = "Correlation Experiment"
    mlflow.set_experiment(experiment_name)

    # Create a parent run for the experiment
    with mlflow.start_run(run_name="Correlation Experiment") as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log experiment parameters
        mlflow.log_params(
            {
                "model_names": ",".join(model_names),
                "correlations": ",".join([str(c) for c in correlations]),
                "num_samples": num_samples,
                "num_features": num_features,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
            }
        )

        # Run experiments for each correlation value
        for correlation in correlations:
            # Prepare data for this correlation
            data_config = prepare_synthetic_data(
                num_samples=num_samples,
                num_features=num_features,
                num_tasks=2,  # Fixed for correlation experiment
                task_correlation=correlation,
                batch_size=batch_size,
                use_mmoe_synthetic=True,
            )

            # Train each model on this data
            for model_name in model_names:
                # Get model-specific parameters
                model_params = None
                if model_params_dict and model_name in model_params_dict:
                    model_params = model_params_dict[model_name]

                # For SingleTaskModel, we need to train one model per task
                if model_name == "SingleTaskModel":
                    for task in data_config["task_names"]:
                        task_model_params = model_params.copy() if model_params else {}
                        task_model_params["target_task"] = task

                        # Create run name
                        run_name = f"corr_{correlation:.2f}_{model_name}_{task}"

                        # Train model with MLflow nested run
                        with mlflow.start_run(run_name=run_name, nested=True):
                            train_model(
                                model_name=model_name,
                                data_config=data_config,
                                model_params=task_model_params,
                                num_epochs=num_epochs,
                                experiment_name=experiment_name,
                                run_name=run_name,
                            )
                else:
                    # Create run name
                    run_name = f"corr_{correlation:.2f}_{model_name}"

                    # Train model with MLflow nested run
                    with mlflow.start_run(run_name=run_name, nested=True):
                        train_model(
                            model_name=model_name,
                            data_config=data_config,
                            model_params=model_params,
                            num_epochs=num_epochs,
                            experiment_name=experiment_name,
                            run_name=run_name,
                        )


def main():
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train multi-task learning models.")

    # Model selection
    parser.add_argument(
        "--model_name",
        type=str,
        default="SharedBottomModel",
        choices=[
            "SingleTaskModel",
            "SharedBottomModel",
            "MixtureOfExperts",
            "MultiGateMixtureOfExperts",
        ],
        help="Name of the model to train",
    )

    # Data options
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "uci_census"],
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10000,
        help="Number of samples for synthetic dataset",
    )
    parser.add_argument(
        "--num_features",
        type=int,
        default=20,
        help="Number of features for synthetic dataset",
    )
    parser.add_argument(
        "--task_correlation",
        type=float,
        default=0.5,
        help="Correlation between tasks for synthetic dataset",
    )

    # Training options
    parser.add_argument(
        "--num_epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate for optimizer"
    )

    # Model-specific options
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="64,32",
        help="Comma-separated list of hidden layer dimensions",
    )
    parser.add_argument(
        "--num_experts", type=int, default=4, help="Number of experts for MoE models"
    )
    parser.add_argument(
        "--target_task", type=str, default=None, help="Target task for SingleTaskModel"
    )

    # Experiment options
    parser.add_argument(
        "--run_correlation_experiment",
        action="store_true",
        help="Run correlation experiment comparing models across different correlation levels",
    )

    args = parser.parse_args()

    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]

    # Set up model parameters
    model_params = {"hidden_dims": hidden_dims}

    # Add model-specific parameters
    if args.model_name in ["MixtureOfExperts", "MultiGateMixtureOfExperts"]:
        model_params["num_experts"] = args.num_experts

    if args.model_name == "SingleTaskModel" and args.target_task:
        model_params["target_task"] = args.target_task

    # Run correlation experiment if requested
    if args.run_correlation_experiment:
        # Define models to compare
        model_names = ["SharedBottomModel", "MixtureOfExperts", "SingleTaskModel"]

        # Define correlations to test
        correlations = [0.9, 0.7, 0.5, 0.3, 0.1, 0.0, -0.1, -0.3, -0.5]

        # Define model parameters
        model_params_dict = {
            "SharedBottomModel": {"hidden_dims": hidden_dims},
            "MixtureOfExperts": {
                "hidden_dims": hidden_dims,
                "num_experts": args.num_experts,
            },
            "SingleTaskModel": {"hidden_dims": hidden_dims},
        }

        # Run experiment
        run_correlation_experiment(
            model_names=model_names,
            correlations=correlations,
            num_samples=args.num_samples,
            num_features=args.num_features,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            model_params_dict=model_params_dict,
        )
    else:
        # Prepare data
        if args.dataset == "uci_census":
            data_config = prepare_uci_census_data(batch_size=args.batch_size)
        else:  # synthetic
            data_config = prepare_synthetic_data(
                num_samples=args.num_samples,
                num_features=args.num_features,
                task_correlation=args.task_correlation,
                batch_size=args.batch_size,
            )

        # For SingleTaskModel, we need a target task
        if args.model_name == "SingleTaskModel":
            if args.target_task is None:
                # Use the first task if not specified
                model_params["target_task"] = data_config["task_names"][0]
                print(
                    f"No target task specified for SingleTaskModel. Using {model_params['target_task']}."
                )
            else:
                # Verify the target task exists
                if args.target_task not in data_config["task_names"]:
                    raise ValueError(
                        f"Target task '{args.target_task}' not found in dataset. "
                        f"Available tasks: {data_config['task_names']}"
                    )
                model_params["target_task"] = args.target_task

        # Train model
        train_model(
            model_name=args.model_name,
            data_config=data_config,
            model_params=model_params,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
        )


if __name__ == "__main__":
    main()
